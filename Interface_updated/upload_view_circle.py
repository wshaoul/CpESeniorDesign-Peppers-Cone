# upload_view_circle.py
#
# Upload a video and display it through the four-view circle hologram pipeline.
# Mirrors upload_view.py but uses live_view_circle instead of live_view.

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from platform_support import create_selfie_segmenter


class UploadView(ttk.Frame):
    """
    Upload a video and display it through the four-view circle hologram pipeline.
    Optionally saves a processed copy while playing.
    """

    def __init__(self, parent, controller=None):
        super().__init__(parent)
        self.controller = controller

        # ---------- state ----------
        self.in_path           = tk.StringVar(value="")
        self.save_while_play_var = tk.BooleanVar(value=False)

        # ---------- layout ----------
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        left  = ttk.Frame(root)
        left.pack(side="left", fill="y", padx=12, pady=12)

        right = ttk.Frame(root)
        right.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=12)

        # Header row
        title_row = ttk.Frame(left)
        title_row.pack(fill="x")
        ttk.Button(title_row, text="← Back", command=self._go_back).pack(side="left")

        hdr = tk.Label(title_row, text="Upload",
                       font=("Segoe UI", 12, "bold"), bd=0, highlightthickness=0)
        hdr.pack(side="left", padx=8)

        tk.Label(
            left,
            text=("Choose a video file, then click 'Open Circle Screen (process now)' "
                  "to play it through the four-view circle hologram pipeline. "
                  "Optionally save a processed copy while it plays."),
            justify="left", wraplength=360
        ).pack(anchor="w", pady=(8, 12))

        # File chooser
        file_box = ttk.LabelFrame(left, text="Video File")
        file_box.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(file_box); row.pack(fill="x", padx=6, pady=6)
        ttk.Entry(row, textvariable=self.in_path).pack(
            side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse…",
                   command=self._choose_file).pack(side="left", padx=6)

        # Options
        opt_box = ttk.LabelFrame(left, text="Circle Screen Options")
        opt_box.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(
            opt_box,
            text="Also save circle-hologram copy while playing",
            variable=self.save_while_play_var
        ).pack(side="left", padx=6, pady=6)

        # Action
        actions = ttk.Frame(left)
        actions.pack(fill="x")
        ttk.Button(
            actions,
            text="Open Circle Screen (process now)",
            command=self._open_circle_screen_process_now
        ).pack(side="left")

        # Right-side hint
        preview_hdr = tk.Label(right, text="Preview",
                                font=("Segoe UI", 12, "bold"))
        preview_hdr.pack(anchor="w")
        tk.Label(
            right,
            text="(Choose a file on the left, then open the circle screen.)",
            justify="left"
        ).pack(anchor="w", pady=6)

    # ---------- navigation ----------
    def _go_back(self):
        if self.controller and hasattr(self.controller, "show_page"):
            try:
                self.controller.show_page("HomePage")
                return
            except Exception:
                pass
        self.winfo_toplevel().focus_set()

    def shutdown(self):
        player = getattr(self, "_player", None)
        if player is not None and player.winfo_exists():
            player.close()

    # ---------- file choose ----------
    def _choose_file(self):
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.m4v *.avi *.mkv *.webm"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self.in_path.set(path)

    # ---------- open circle screen (process ON DEMAND) ----------
    def _open_circle_screen_process_now(self):
        in_path = self.in_path.get().strip()
        if not in_path or not os.path.exists(in_path):
            messagebox.showwarning(
                "Circle Screen", "Please choose an existing video file first.")
            return

        from cone_player import open_circle_cone_player
        try:
            self._player = open_circle_cone_player(
                self, in_path, bool(self.save_while_play_var.get())
            )
        except Exception as exc:
            messagebox.showerror("Circle Screen", str(exc))

    # ---------- worker: process + play uploaded video ----------
    def _circle_player_worker(self, in_path: str, save_copy: bool):
        # Import helpers from live_view_circle
        try:
            from live_view_circle import (build_four_cone_maps, apply_four_cone_views,
                                          FRAME_SIZE, CANVAS_SIZE)
        except Exception as e:
            messagebox.showerror(
                "Circle Screen",
                f"Could not import live_view_circle helpers:\n{e}")
            return

        # Open input video (prefer FFMPEG)
        try:
            cap = cv2.VideoCapture(in_path, cv2.CAP_FFMPEG)
            if not cap or not cap.isOpened():
                raise RuntimeError("FFMPEG backend failed")
        except Exception:
            cap = cv2.VideoCapture(in_path)
            if not cap or not cap.isOpened():
                messagebox.showerror("Circle Screen",
                                     f"Could not open video:\n{in_path}")
                return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps < 1.0:
            fps = 30.0
        delay = max(1, int(1000.0 / float(min(60.0, max(15.0, fps)))))

        # Build four cone-warp maps once (same defaults as LiveView)
        warp_maps = build_four_cone_maps(
            frame_size=int(FRAME_SIZE),
            canvas_size=int(CANVAS_SIZE),
            span_deg=90,
            r_inner_frac=0.08,
            r_outer_frac=0.995,
            center_frac=(0.50, 0.50),
            radius_frac=1.00,
            base_rotate=270.0,
        )

        # Optional writer: save processed copy while playing
        writer   = None
        out_path = None
        if save_copy:
            base, _ = os.path.splitext(in_path)
            out_path = base + "_circle.mp4"
            fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
            writer  = cv2.VideoWriter(out_path, fourcc, fps,
                                      (int(CANVAS_SIZE), int(CANVAS_SIZE)))
            if not writer.isOpened():
                writer   = None
                out_path = None

        # Background remover
        segmentor = self._build_segmentor()

        # Fullscreen OpenCV window
        win = "Circle Hologram Screen"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Play (loop)
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # 1) Resize to FRAME_SIZE square
            sq = cv2.resize(frame, (int(FRAME_SIZE), int(FRAME_SIZE)),
                            interpolation=cv2.INTER_AREA)

            # 2) Background removal
            fg = self._segment_person(sq, segmentor)

            # 3) Four-face cone-warp hologram
            canvas = apply_four_cone_views(fg, warp_maps, frame_size=int(FRAME_SIZE))

            # Show
            cv2.imshow(win, canvas)

            # Optionally save
            if writer is not None:
                writer.write(canvas)

            key = cv2.waitKey(delay) & 0xFF
            if key in (27, ord('q')):
                break

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyWindow(win)

        if save_copy and out_path:
            self.after(0, lambda: messagebox.showinfo(
                "Circle Screen", f"Circle hologram copy saved:\n{out_path}"))

    # ---------- helpers ----------
    def _build_segmentor(self):
        return create_selfie_segmenter(model_selection=1)

    def _segment_person(self, bgr_square, segmentor):
        if segmentor is None:
            return bgr_square
        rgb = cv2.cvtColor(bgr_square, cv2.COLOR_BGR2RGB)
        seg = segmentor.process(rgb)
        raw = seg.segmentation_mask.astype("float32")
        raw = cv2.GaussianBlur(raw, (7, 7), 0)
        mask = (raw > 0.35).astype("uint8") * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                 np.ones((5, 5), np.uint8), iterations=1)
        fg = cv2.bitwise_and(bgr_square, bgr_square, mask=mask)
        return fg
