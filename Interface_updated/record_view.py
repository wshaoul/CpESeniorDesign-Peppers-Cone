# record_view.py
import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


class RecordView(ttk.Frame):
    """
    Record camera video with a live preview. Instead of pre-processing after stop,
    the cone warp is applied ON DEMAND when you open the Cone Screen—mirroring
    the live pipeline. Optionally save a warped copy while it plays.
    """

    def __init__(self, parent, controller=None):
        super().__init__(parent)
        self.controller = controller

        # --- state ---
        self.cap = None
        self.writer = None
        self.record_thread = None
        self.recording = False
        self.last_frame = None
        self.out_path = None
        self._frame_count = 0

        # --------------------------
        # UI LAYOUT
        # --------------------------
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root)
        left.pack(side="left", fill="y", padx=12, pady=12)

        right = ttk.Frame(root)
        right.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=12)

        # Top row: back + title
        title_row = ttk.Frame(left)
        title_row.pack(fill="x")
        ttk.Button(title_row, text="← Back", command=self._go_back).pack(side="left")
        ttk.Label(title_row, text="Record", style="Header.TLabel").pack(side="left", padx=8)

        ttk.Label(
            left,
            text=("Choose a camera, set resolution/fps, and record. "
                  "When you’re ready, click 'Open Cone Screen (process now)' to play the recorded "
                  "video through the same cone-warp pipeline used in Live. "
                  "Optionally save the warped copy while it plays."),
            style="Body.TLabel", wraplength=360
        ).pack(anchor="w", pady=(8, 12))

        # Camera box
        cam_box = ttk.LabelFrame(left, text="Camera")
        cam_box.pack(fill="x", pady=(0, 8))

        self.cam_index = tk.IntVar(value=0)
        row = ttk.Frame(cam_box); row.pack(fill="x", padx=6, pady=6)
        ttk.Label(row, text="Index:").pack(side="left")
        self.cam_entry = ttk.Spinbox(row, from_=0, to=10, width=5, textvariable=self.cam_index)
        self.cam_entry.pack(side="left", padx=6)
        ttk.Button(row, text="Open", command=self._open_camera).pack(side="left", padx=(6, 0))
        ttk.Button(row, text="Close", command=self._close_camera).pack(side="left", padx=6)

        # Video settings
        video_box = ttk.LabelFrame(left, text="Video Settings")
        video_box.pack(fill="x", pady=(0, 8))
        self.width = tk.IntVar(value=1280)
        self.height = tk.IntVar(value=720)
        self.fps = tk.DoubleVar(value=30.0)

        r1 = ttk.Frame(video_box); r1.pack(fill="x", padx=6, pady=(6, 0))
        ttk.Label(r1, text="Resolution:").pack(side="left")
        ttk.Entry(r1, width=6, textvariable=self.width).pack(side="left", padx=4)
        ttk.Label(r1, text="x").pack(side="left")
        ttk.Entry(r1, width=6, textvariable=self.height).pack(side="left", padx=4)

        r2 = ttk.Frame(video_box); r2.pack(fill="x", padx=6, pady=6)
        ttk.Label(r2, text="FPS:").pack(side="left")
        ttk.Entry(r2, width=6, textvariable=self.fps).pack(side="left", padx=4)

        # Output
        out_box = ttk.LabelFrame(left, text="Output")
        out_box.pack(fill="x", pady=(0, 8))
        self.out_var = tk.StringVar(value="")
        r3 = ttk.Frame(out_box); r3.pack(fill="x", padx=6, pady=6)
        ttk.Entry(r3, textvariable=self.out_var).pack(side="left", fill="x", expand=True)
        ttk.Button(r3, text="Browse…", command=self._choose_output).pack(side="left", padx=6)

        # Process mode
        proc_box = ttk.LabelFrame(left, text="Cone Screen Options")
        proc_box.pack(fill="x", pady=(0, 8))
        self.save_while_play_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            proc_box,
            text="Also save warped copy while playing",
            variable=self.save_while_play_var
        ).pack(side="left", padx=6, pady=6)

        # Start/Stop
        actions = ttk.Frame(left)
        actions.pack(fill="x", pady=(4, 0))
        ttk.Button(actions, text="Start Recording", command=self._start_recording).pack(side="left")
        ttk.Button(actions, text="Stop", command=self._stop_recording).pack(side="left", padx=8)

        # Open cone screen (process on demand)
        open_row = ttk.Frame(left)
        open_row.pack(fill="x", pady=(8, 0))
        ttk.Button(
            open_row,
            text="Open Cone Screen (process now)",
            command=self._open_cone_screen_process_now
        ).pack(side="left")

        # Right preview
        ttk.Label(right, text="Live Preview", style="Header.TLabel").pack(anchor="w")
        self.preview_label = ttk.Label(right)
        self.preview_label.pack(fill="both", expand=True, pady=(6, 0))

        # stats
        self.frames = tk.StringVar(value="Frames: 0")
        ttk.Label(right, textvariable=self.frames).pack(anchor="w", pady=6)

        # begin preview loop
        self.after(100, self._preview_tick)

    # ---------- navigation ----------
    def _go_back(self):
        if self.controller and hasattr(self.controller, "show_page"):
            try:
                self.controller.show_page("HomePage")
                return
            except Exception:
                pass
        self.winfo_toplevel().focus_set()

    # ---------- camera & recording ----------
    def _open_camera(self):
        self._close_camera()
        idx = int(self.cam_index.get())
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # Windows-friendly
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Camera", f"Could not open camera index {idx}.")
            self.cap = None
            return

        if self.width.get() > 0 and self.height.get() > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width.get()))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height.get()))
        if self.fps.get() > 0:
            self.cap.set(cv2.CAP_PROP_FPS, float(self.fps.get()))

    def _close_camera(self):
        try:
            if self.cap:
                self.cap.release()
        finally:
            self.cap = None

    def _choose_output(self):
        path = filedialog.asksaveasfilename(
            title="Save recording as",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("All files", "*.*")]
        )
        if path:
            self.out_var.set(path)

    def _start_recording(self):
        if self.cap is None or not self.cap.isOpened():
            self._open_camera()
            if self.cap is None:
                return

        out_path = self.out_var.get().strip()
        if not out_path:
            messagebox.showwarning("Output", "Please choose an output file first.")
            return
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(self.width.get())
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(self.height.get())
        fps = self.cap.get(cv2.CAP_PROP_FPS) or float(self.fps.get() or 30.0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not self.writer or not self.writer.isOpened():
            messagebox.showerror("Record", "Could not open VideoWriter. Try a different path or filename.")
            self.writer = None
            return

        self.out_path = out_path
        self.recording = True
        self._frame_count = 0

        self.record_thread = threading.Thread(target=self._record_loop, args=(w, h), daemon=True)
        self.record_thread.start()
        messagebox.showinfo("Recording", f"Recording started.\nSaving to: {out_path}")

    def _record_loop(self, w, h):
        while self.recording and self.cap and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            self.last_frame = frame
            self.writer.write(frame)
            self._frame_count += 1
            time.sleep(0.001)

    def _stop_recording(self):
        if self.recording:
            self.recording = False
            try:
                if self.record_thread:
                    self.record_thread.join(timeout=1.0)
            except Exception:
                pass
            try:
                if self.writer:
                    self.writer.release()
            finally:
                self.writer = None

        self._close_camera()  # turn off LED / free device
        messagebox.showinfo("Recording", "Recording stopped.")

    # ---------- preview ----------
    def _preview_tick(self):
        if self.cap and self.cap.isOpened():
            frame = self.last_frame
            if frame is None:
                ok, frm = self.cap.read()
                if ok:
                    frame = frm
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb.shape
                lbl_w = max(320, self.preview_label.winfo_width() or w)
                lbl_h = max(180, self.preview_label.winfo_height() or h)
                scale = min(lbl_w / w, lbl_h / h)
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                im = Image.fromarray(rgb).resize(new_size, Image.BILINEAR)
                self._tk_img = ImageTk.PhotoImage(im)
                self.preview_label.config(image=self._tk_img)
        self.frames.set(f"Frames: {self._frame_count}")
        self.after(33, self._preview_tick)

    # ---------- background removal helpers ----------
    def _build_segmentor(self):
        try:
            import mediapipe as mp
            return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        except Exception:
            return None

    def _segment_person(self, bgr_square, segmentor):
        if segmentor is None:
            return bgr_square
        rgb = cv2.cvtColor(bgr_square, cv2.COLOR_BGR2RGB)
        seg = segmentor.process(rgb)
        raw = seg.segmentation_mask.astype("float32")
        raw = cv2.GaussianBlur(raw, (7, 7), 0)
        mask = (raw > 0.35).astype("uint8") * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        fg = cv2.bitwise_and(bgr_square, bgr_square, mask=mask)
        return fg

    # ---------- cone screen: process RECORDED video ON DEMAND ----------
    def _open_cone_screen_process_now(self):
        """
        Open a fullscreen OpenCV window that plays the recorded file through the
        cone warp pipeline in real time (like Live). Optionally save a warped copy.
        """
        # Choose the most recent output path
        in_path = self.out_path or self.out_var.get().strip()
        if not in_path or not os.path.exists(in_path):
            messagebox.showwarning("Cone Screen", "No recorded video found. Please record and stop first.")
            return

        save_copy = bool(self.save_while_play_var.get())
        threading.Thread(
            target=self._cone_player_worker,
            args=(in_path, save_copy),
            daemon=True
        ).start()

    def _cone_player_worker(self, in_path: str, save_copy: bool):
        # Import helpers from live_view
        try:
            from live_view import build_cone_maps, enhance_saturation_contrast, FRAME_SIZE, CANVAS_SIZE
        except Exception as e:
            messagebox.showerror("Cone Screen", f"Could not import live_view helpers:\n{e}")
            return

        # Open video (FFMPEG if possible)
        try:
            cap = cv2.VideoCapture(in_path, cv2.CAP_FFMPEG)
            if not cap or not cap.isOpened():
                raise RuntimeError("FFMPEG failed")
        except Exception:
            cap = cv2.VideoCapture(in_path)
            if not cap or not cap.isOpened():
                messagebox.showerror("Cone Screen", f"Could not open video:\n{in_path}")
                return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps < 1.0:
            fps = 30.0
        delay = max(1, int(1000.0 / float(min(60.0, max(15.0, fps)))))

        # Build warp maps once
        params = dict(
            span_deg=200,
            rotate_deg=270.0,
            r_inner_frac=0.08,
            r_outer_frac=0.995,
            center_frac=(0.50, 0.50),
            radius_frac=1.00,
        )
        try:
            map_x, map_y = build_cone_maps(
                frame_size=int(FRAME_SIZE),
                canvas_size=int(CANVAS_SIZE),
                **params
            )
        except Exception as e:
            cap.release()
            messagebox.showerror("Cone Screen", f"Failed to build warp maps:\n{e}")
            return

        # Optional writer (save warped copy while playing)
        writer = None
        out_path = None
        if save_copy:
            base, _ = os.path.splitext(in_path)
            out_path = base + "_cone.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (int(CANVAS_SIZE), int(CANVAS_SIZE)))
            if not writer.isOpened():
                writer = None
                out_path = None

        # Background remover
        segmentor = self._build_segmentor()

        # Fullscreen OpenCV window
        win = "Cone Screen (Processed Live)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Play (loop)
        while True:
            ok, frame = cap.read()
            if not ok:
                # loop
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # 1) square to FRAME_SIZE
            sq = cv2.resize(frame, (int(FRAME_SIZE), int(FRAME_SIZE)), interpolation=cv2.INTER_AREA)

            # 2) background removal
            fg = self._segment_person(sq, segmentor)

            # 3) center + scale like Live
            scale = 0.6
            scaled = cv2.resize(fg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            padded = np.zeros_like(fg)
            y_off = (int(FRAME_SIZE) - scaled.shape[0]) // 2
            x_off = (int(FRAME_SIZE) - scaled.shape[1]) // 2
            padded[y_off:y_off + scaled.shape[0], x_off:x_off + scaled.shape[1]] = scaled

            # 4) cone warp
            warped = cv2.remap(
                padded, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

            # 5) color pop
            enhanced = enhance_saturation_contrast(
                warped, saturation_scale=1.4, contrast_alpha=1.8, brightness_beta=-25
            )

            # show
            cv2.imshow(win, enhanced)

            # optionally save
            if writer is not None:
                writer.write(enhanced)

            key = cv2.waitKey(delay) & 0xFF
            if key in (27, ord('q')):  # Esc or q to quit
                break

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyWindow(win)

        # Notify if we saved
        if save_copy and out_path:
            self.after(0, lambda: messagebox.showinfo("Cone Screen", f"Warped copy saved:\n{out_path}"))

    # ---------- progress helpers (unused in this on-demand flow; kept for parity) ----------
    def _open_progress(self, out_path: str, total: int):
        try:
            self._progress = tk.Toplevel(self)
            self._progress.title("Warping video…")
            self._progress.resizable(False, False)
            ttk.Label(self._progress, text=f"Warping to {os.path.basename(out_path)}").pack(padx=12, pady=(12, 6))
            self._pbar = ttk.Progressbar(self._progress, mode="determinate", length=320, maximum=max(1, total))
            self._pbar.pack(padx=12, pady=(0, 12))
            self._progress.transient(self.winfo_toplevel())
            self._progress.grab_set()
            self._progress.update()
        except Exception:
            self._progress = None
            self._pbar = None

    def _tick_progress(self, value: int):
        if getattr(self, "_pbar", None) is not None and getattr(self, "_progress", None) is not None:
            try:
                self._pbar["value"] = value
                self._progress.update()
            except Exception:
                pass

    def _close_progress(self):
        if getattr(self, "_progress", None) is not None:
            try:
                self._progress.grab_release()
                self._progress.destroy()
            except Exception:
                pass
        self._progress = None
        self._pbar = None
