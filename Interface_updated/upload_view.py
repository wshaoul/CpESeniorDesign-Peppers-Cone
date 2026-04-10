# upload_view.py
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np


class UploadView(ttk.Frame):
    """
    Upload a video and display it through the cone-warp pipeline (like Live/Record).
    Optionally save a warped copy while it plays.
    """

    def __init__(self, parent, controller=None):
        super().__init__(parent)
        self.controller = controller

        # ---------- state ----------
        self.in_path = tk.StringVar(value="")
        self.save_while_play_var = tk.BooleanVar(value=False)

        # ---------- layout ----------
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        left = ttk.Frame(root)
        left.pack(side="left", fill="y", padx=12, pady=12)

        right = ttk.Frame(root)
        right.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=12)

        # Header row
        title_row = ttk.Frame(left)
        title_row.pack(fill="x")
        ttk.Button(title_row, text="← Back", command=self._go_back).pack(side="left")

        # Use plain tk.Label so it doesn't look like a button
        hdr = tk.Label(title_row, text="Upload", font=("Segoe UI", 12, "bold"), bd=0, highlightthickness=0)
        hdr.pack(side="left", padx=8)

        tk.Label(
            left,
            text=("Choose a video file, then click 'Open Cone Screen (process now)' "
                  "to play it through the same cone-warp pipeline used in Live/Record. "
                  "Optionally save a warped copy while it plays."),
            justify="left", wraplength=360
        ).pack(anchor="w", pady=(8, 12))

        # File chooser
        file_box = ttk.LabelFrame(left, text="Video File")
        file_box.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(file_box); row.pack(fill="x", padx=6, pady=6)
        ttk.Entry(row, textvariable=self.in_path).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse…", command=self._choose_file).pack(side="left", padx=6)

        # Options
        opt_box = ttk.LabelFrame(left, text="Cone Screen Options")
        opt_box.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(
            opt_box,
            text="Also save warped copy while playing",
            variable=self.save_while_play_var
        ).pack(side="left", padx=6, pady=6)

        # Action
        actions = ttk.Frame(left)
        actions.pack(fill="x")
        ttk.Button(
            actions,
            text="Open Cone Screen (process now)",
            command=self._open_cone_screen_process_now
        ).pack(side="left")

        # Right side “placeholder” (keeps page balanced with other views)
        preview_hdr = tk.Label(right, text="Preview", font=("Segoe UI", 12, "bold"))
        preview_hdr.pack(anchor="w")
        right_hint = tk.Label(
            right,
            text="(Choose a file on the left, then open the cone screen.)",
            justify="left"
        )
        right_hint.pack(anchor="w", pady=6)

    # ---------- navigation ----------
    def _go_back(self):
        if self.controller and hasattr(self.controller, "show_page"):
            try:
                self.controller.show_page("HomePage")
                return
            except Exception:
                pass
        self.winfo_toplevel().focus_set()

    # ---------- file choose ----------
    def _choose_file(self):
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[
                ("Video files", "*.mp4;*.mov;*.m4v;*.avi;*.mkv;*.webm"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self.in_path.set(path)

    # ---------- open cone screen (process ON DEMAND like Live) ----------
    def _open_cone_screen_process_now(self):
        in_path = self.in_path.get().strip()
        if not in_path or not os.path.exists(in_path):
            messagebox.showwarning("Cone Screen", "Please choose an existing video file first.")
            return

        save_copy = bool(self.save_while_play_var.get())
        threading.Thread(
            target=self._cone_player_worker,
            args=(in_path, save_copy),
            daemon=True
        ).start()

    # ---------- worker: process + play uploaded video ----------
    def _cone_player_worker(self, in_path: str, save_copy: bool):
        # Import helpers from live_view (same ones your Live/Record use)
        try:
            from live_view import build_cone_maps, enhance_saturation_contrast, FRAME_SIZE, CANVAS_SIZE
        except Exception as e:
            messagebox.showerror("Cone Screen", f"Could not import live_view helpers:\n{e}")
            return

        # Open input video (prefer FFMPEG)
        try:
            cap = cv2.VideoCapture(in_path, cv2.CAP_FFMPEG)
            if not cap or not cap.isOpened():
                raise RuntimeError("FFMPEG backend failed")
        except Exception:
            cap = cv2.VideoCapture(in_path)
            if not cap or not cap.isOpened():
                messagebox.showerror("Cone Screen", f"Could not open video:\n{in_path}")
                return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps < 1.0:
            fps = 30.0
        delay = max(1, int(1000.0 / float(min(60.0, max(15.0, fps)))))

        # Build maps once (same params as Live/Record)
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

        # Optional writer: save warped copy while playing
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
                # loop the file
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

    # ---------- helpers ----------
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
