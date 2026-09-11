# live_view_circle.py
#
# Live display — four-view cone-warp circle hologram.
# Four 90°-rotated copies of the camera feed are each warped through the
# Pepper's Cone arc mapping (from live_view.py) and composited symmetrically
# around the centre, producing a full 360° cone hologram display.

import time
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import math
import cv2
from PIL import Image, ImageTk
from platform_support import (
    IS_WINDOWS,
    camera_backends,
    create_selfie_segmenter,
    open_camera,
)

# Optional RealSense support
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

PREVIEW_W = 800
PREVIEW_H = 450

FRAME_SIZE  = 400
CANVAS_SIZE = 800   # same as the original live_view.py


# ---------- Warp helpers (copied from live_view.py so this file is self-contained) ----------

def enhance_saturation_contrast(image_bgr, saturation_scale=1.3,
                                 contrast_alpha=1.2, brightness_beta=10):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast_alpha, beta=brightness_beta)
    return enhanced


def build_cone_maps(frame_size, canvas_size, span_deg=90, rotate_deg=270.0,
                    r_inner_frac=0.08, r_outer_frac=0.995,
                    center_frac=(0.50, 0.50), radius_frac=1.00):
    """
    Warp for a single Pepper's Cone arc face.
    Returns map_x, map_y (float32) — same function as in live_view.py.
    """
    map_x = np.full((canvas_size, canvas_size), -1, dtype=np.float32)
    map_y = np.full((canvas_size, canvas_size), -1, dtype=np.float32)

    cx = int(center_frac[0] * canvas_size)
    cy = int(center_frac[1] * canvas_size)
    R  = int((canvas_size * 0.5) * max(0.10, min(2.0, radius_frac)))

    r_in_frac  = max(0.0, min(0.99, r_inner_frac))
    r_out_frac = max(r_in_frac + 1.0 / max(1, R), min(1.0, r_outer_frac))
    r_in  = r_in_frac  * R
    r_out = r_out_frac * R

    half = math.radians(max(1, min(359, span_deg))) * 0.5
    rot  = math.radians(rotate_deg)

    y_coords, x_coords = np.ogrid[0:canvas_size, 0:canvas_size]
    dx = x_coords - cx
    dy = y_coords - cy

    r   = np.sqrt(dx**2 + dy**2)
    ang = np.arctan2(dy, dx) - rot
    ang = np.where(ang < -np.pi, ang + 2 * np.pi, ang)
    ang = np.where(ang >  np.pi, ang - 2 * np.pi, ang)

    valid = (r >= r_in) & (r <= r_out) & (ang >= -half) & (ang <= half)

    if np.any(valid):
        u = np.clip((ang[valid] + half) / (2 * half), 0.0, 1.0)
        v = np.clip(1.0 - (r[valid] - r_in) / max(1.0, r_out - r_in), 0.0, 1.0)
        map_x[valid] = (u * (frame_size - 1)).astype(np.float32)
        map_y[valid] = (v * (frame_size - 1)).astype(np.float32)

    return map_x, map_y


def build_four_cone_maps(frame_size, canvas_size, span_deg=90,
                          r_inner_frac=0.08, r_outer_frac=0.995,
                          center_frac=(0.50, 0.50), radius_frac=1.00,
                          base_rotate=270.0):
    """
    Build four cone-warp map pairs, each rotated 90° from the last, covering
    the full 360° around the centre.

    base_rotate=270 places the first face pointing upward (matching the
    default orientation of the original single-face live_view.py).

    Returns a list of four (map_x, map_y) tuples.
    """
    maps = []
    for i in range(4):
        rotate_deg = (base_rotate + i * 90.0) % 360.0
        mx, my = build_cone_maps(
            frame_size=frame_size,
            canvas_size=canvas_size,
            span_deg=span_deg,
            rotate_deg=rotate_deg,
            r_inner_frac=r_inner_frac,
            r_outer_frac=r_outer_frac,
            center_frac=center_frac,
            radius_frac=radius_frac,
        )
        maps.append((mx, my))
    return maps


def apply_four_cone_views(frame_bgr, warp_maps, frame_size=FRAME_SIZE,
                           subject_scale=0.6,
                           saturation_scale=1.4, contrast_alpha=1.8,
                           brightness_beta=-25):
    """
    Apply four cone-warp faces and composite them onto one canvas.

    Parameters
    ----------
    frame_bgr     : BGR image (already background-removed if desired).
    warp_maps     : list of four (map_x, map_y) pairs from build_four_cone_maps.
    frame_size    : side length of the square input panel.
    subject_scale : shrink factor so the subject doesn't bleed to the panel edge.

    Returns
    -------
    canvas : (CANVAS_SIZE, CANVAS_SIZE, 3) uint8 BGR composite image.
    """
    # 1) Resize to square
    sq = cv2.resize(frame_bgr, (frame_size, frame_size), interpolation=cv2.INTER_AREA)

    # 2) Optional subject centring/scaling
    if 0.0 < subject_scale < 1.0:
        scaled = cv2.resize(sq, (0, 0), fx=subject_scale, fy=subject_scale,
                            interpolation=cv2.INTER_AREA)
        padded = np.zeros_like(sq)
        y_off  = (frame_size - scaled.shape[0]) // 2
        x_off  = (frame_size - scaled.shape[1]) // 2
        padded[y_off:y_off + scaled.shape[0],
               x_off:x_off + scaled.shape[1]] = scaled
        sq = padded

    # 3) Source frames for each arc face.
    #
    # The four arc positions are related by 90° rotations of the canvas.
    # Tracing the warp-map u/v coordinates through each rotation shows:
    #
    #   Face 0 – top    (base_rotate=270°): reference frame = flip(sq, 1)
    #   Face 1 – right  (+90°, pointing right):  u_right = u_top  → same frame as top
    #   Face 2 – bottom (+180°, pointing down):  u_bottom = u_top → same frame as top
    #   Face 3 – left   (+270°, pointing left):  u_left = 1−u_top → flip u → use sq
    #
    # Result on canvas: right = top_CW, bottom = top_180°, left = top_CCW.
    top_view = cv2.flip(sq, 1)    # horizontal flip — the reference "top" orientation
    orientations = [
        top_view,   # face 0 (top)                      → top view as-is
        top_view,   # face 1 (right, arc at 0°)         → appears as top rotated 90° CW
        top_view,   # face 2 (bottom, arc at 90°)       → appears as top rotated 180°
        sq,         # face 3 (left, arc at 180°)        → appears as top rotated 90° CCW
    ]

    # 4) Remap each face and composite with np.maximum (handles any slight overlap)
    canvas_size = warp_maps[0][0].shape[0]
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    for (mx, my), rotated in zip(warp_maps, orientations):
        warped = cv2.remap(rotated, mx, my,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0))
        np.maximum(canvas, warped, out=canvas)

    # 5) Colour enhancement
    canvas = enhance_saturation_contrast(
        canvas,
        saturation_scale=saturation_scale,
        contrast_alpha=contrast_alpha,
        brightness_beta=brightness_beta,
    )

    return canvas


# ---------- RealSense helpers ----------
RS_W, RS_H, RS_FPS = 640, 360, 60
RS_TIMEOUT_MS      = 3000
RS_WARMUP_FRAMES   = 10


def _rs_start():
    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
    cfg.enable_stream(rs.stream.depth, RS_W, RS_H, rs.format.z16,  RS_FPS)
    profile  = pipeline.start(cfg)
    align    = rs.align(rs.stream.color)
    dev  = profile.get_device()
    name = dev.get_info(rs.camera_info.name)
    sn   = dev.get_info(rs.camera_info.serial_number)
    print(f"[RealSense] Connected: {name}  s/n {sn}  ({RS_W}x{RS_H} @ {RS_FPS}fps)")
    return pipeline, align


def _rs_get_frames(pipeline, align):
    frames  = pipeline.wait_for_frames(timeout_ms=RS_TIMEOUT_MS)
    aligned = align.process(frames)
    cf = aligned.get_color_frame()
    df = aligned.get_depth_frame()
    color_bgr = np.asanyarray(cf.get_data()) if cf else None
    depth_vis = None
    if df:
        depth_mm   = np.asanyarray(df.get_data())
        depth_norm = (np.clip(depth_mm, 0, 4000).astype(np.float32) / 4000.0 * 255).astype(np.uint8)
        depth_vis  = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
    return color_bgr, depth_vis


# ---------- Camera backends ----------
BACKENDS = camera_backends()


def _open_by_index(index, api_pref):
    cap = open_camera(index, api_pref)
    if cap is not None:
        return cap, "index", api_pref
    return None, None, None


def _open_by_name_dshow(name):
    cap = cv2.VideoCapture(f"video={name}", cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap, "name", cv2.CAP_DSHOW
    cap.release()
    return None, None, None


def _list_dshow_devices_via_ffmpeg():
    try:
        proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-list_devices", "true",
             "-f", "dshow", "-i", "dummy"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out, _ = proc.communicate(timeout=6)
    except Exception:
        return [], []

    import re
    video, audio = [], []
    m = re.compile(r'^\[dshow .*?\]\s+"([^"]+)"\s+\((video|audio)\)\s*$',
                   re.IGNORECASE)
    for line in (out or "").splitlines():
        mm = m.match(line.strip())
        if not mm:
            continue
        name, kind = mm.group(1), mm.group(2).lower()
        (video if kind == "video" else audio).append(name)

    def dedup(xs):
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return dedup(video), dedup(audio)


# ============================================================
# LiveView  —  four-face cone-warp circle hologram
# ============================================================
class LiveView(ttk.Frame):
    """
    Live display using four Pepper's Cone arc faces arranged symmetrically
    around the centre, each one being a 90°-rotated copy of the camera feed
    passed through the same cone-warp pipeline as the original live_view.py.
    """

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self._cam_src = tk.StringVar(value="webcam")

        self._segmentor = create_selfie_segmenter(model_selection=1)

        # --- Title ---
        ttk.Label(self, text="Live Display  –  Circle Hologram (Cone Warp)",
                  style="Header.TLabel").pack(pady=(10, 4))
        ttk.Label(self,
                  text="Four Pepper's Cone arc faces arranged 360° around the centre. "
                       "Tune span, radius, and rotation, then open fullscreen.",
                  style="Body.TLabel").pack(pady=(0, 10))

        top   = ttk.Frame(self); top.pack(fill="both", expand=True)
        left  = ttk.Frame(top);  left.pack(side="left",  fill="both", expand=True, padx=(0, 8))
        right = ttk.Frame(top);  right.pack(side="right", fill="both", expand=True, padx=(8, 0))

        # --- Camera Source ---
        src_box = ttk.LabelFrame(left, text="Camera Source")
        src_box.pack(fill="x", padx=4, pady=(0, 10))
        ttk.Radiobutton(src_box, text="Webcam", variable=self._cam_src,
                        value="webcam", command=self._on_src_change).pack(
                            side="left", padx=14, pady=6)
        ttk.Radiobutton(src_box, text="RealSense D455/D555",
                        variable=self._cam_src, value="realsense",
                        command=self._on_src_change,
                        state="normal" if REALSENSE_AVAILABLE else "disabled").pack(
                            side="left", padx=14, pady=6)
        if not REALSENSE_AVAILABLE:
            ttk.Label(src_box, text="(install pyrealsense2 to enable)",
                      foreground="#888").pack(side="left", padx=6)

        # --- Camera Selection ---
        cam_box = ttk.LabelFrame(left, text="Camera Selection")
        cam_box.pack(fill="x", padx=4, pady=(0, 10))
        cam_box.grid_columnconfigure(0, weight=1)

        self.sel_mode = tk.StringVar(value="index")
        r1 = ttk.Radiobutton(cam_box, text="By Index",
                              variable=self.sel_mode, value="index",
                              command=self._update_controls)
        r2 = ttk.Radiobutton(cam_box, text="By Name (DirectShow)",
                              variable=self.sel_mode, value="name",
                              command=self._update_controls,
                              state="normal" if IS_WINDOWS else "disabled")

        idx_row = ttk.Frame(cam_box)
        ttk.Label(idx_row, text="Index:").grid(row=0, column=0, padx=(0, 6))
        self.idx_combo = ttk.Combobox(idx_row, state="readonly", width=36,
                                       values=["0 – Default Camera"])
        self.idx_combo.set(self.idx_combo["values"][0])
        self.idx_combo.grid(row=0, column=1, sticky="w")
        ttk.Button(idx_row, text="Rescan",
                   command=self._rescan_indices).grid(row=0, column=2, padx=6)
        ttk.Label(idx_row, text="Backend:").grid(row=0, column=3, padx=(12, 6))
        self.backend_combo = ttk.Combobox(
            idx_row, state="readonly",
            values=[label for (label, _) in BACKENDS], width=22)
        self.backend_combo.set(BACKENDS[0][0])
        self.backend_combo.grid(row=0, column=4, padx=(0, 6))

        name_row = ttk.Frame(cam_box)
        ttk.Label(name_row, text="Device Name:").grid(row=0, column=0, padx=(0, 6))
        self.name_entry = ttk.Entry(name_row, width=36)
        self.name_entry.grid(row=0, column=1)
        ttk.Button(name_row, text="List Cameras (ffmpeg)",
                   command=self._list_names_ffmpeg).grid(row=0, column=2, padx=6)
        self.names_combo = ttk.Combobox(name_row, state="readonly", width=36, values=[])
        self.names_combo.grid(row=1, column=1, pady=(6, 0), sticky="w")
        ttk.Button(name_row, text="Use Selected",
                   command=self._use_selected_name).grid(row=1, column=2, padx=6, pady=(6, 0))

        r1.grid(row=0, column=0, sticky="w", padx=8, pady=(6, 2))
        idx_row.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 6))
        r2.grid(row=2, column=0, sticky="w", padx=8, pady=(8, 2))
        name_row.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 6))

        # --- Video Settings ---
        settings = ttk.LabelFrame(left, text="Video Settings")
        settings.pack(fill="x", padx=4, pady=(0, 10))
        ttk.Label(settings, text="Resolution:").grid(row=0, column=0,
                                                      padx=(8, 6), pady=8, sticky="w")
        self.res_combo = ttk.Combobox(settings, state="readonly", width=12,
                                       values=["1280x720", "1920x1080", "640x480"])
        self.res_combo.set("1280x720")
        self.res_combo.grid(row=0, column=1, sticky="w")
        ttk.Label(settings, text="FPS:").grid(row=0, column=2,
                                               padx=(16, 6), pady=8, sticky="w")
        self.fps_entry = ttk.Entry(settings, width=6)
        self.fps_entry.insert(0, "30")
        self.fps_entry.grid(row=0, column=3, sticky="w")

        # --- Cone Warp Tuning ---
        tuning = ttk.LabelFrame(left, text="Cone Warp Tuning  (applied to all 4 faces)")
        tuning.pack(fill="x", padx=4, pady=(0, 10))

        def _row(label, var, from_, to_, default, row, fmt="{:.0f}"):
            ttk.Label(tuning, text=label).grid(
                row=row, column=0, padx=(8, 4), pady=4, sticky="w")
            lbl = ttk.Label(tuning, text=fmt.format(default))
            slider = ttk.Scale(tuning, from_=from_, to=to_, orient="horizontal",
                               variable=var, command=self._on_warp_change)
            slider.grid(row=row, column=1, sticky="ew", padx=4)
            lbl.grid(row=row, column=2, padx=4)
            return slider, lbl

        self.span_var        = tk.DoubleVar(value=90)
        self.r_inner_var     = tk.DoubleVar(value=0.08)
        self.r_outer_var     = tk.DoubleVar(value=0.995)
        self.center_x_var    = tk.DoubleVar(value=0.50)
        self.center_y_var    = tk.DoubleVar(value=0.50)
        self.base_rotate_var = tk.DoubleVar(value=270.0)
        self.scale_var       = tk.DoubleVar(value=0.6)

        _, self.span_lbl        = _row("Span (deg):",    self.span_var,
                                        60,  180,  90,    0)
        _, self.r_inner_lbl     = _row("Inner Radius:",  self.r_inner_var,
                                        0.0, 0.95, 0.08,  1, "{:.3f}")
        _, self.r_outer_lbl     = _row("Outer Radius:",  self.r_outer_var,
                                        0.5, 1.0,  0.995, 2, "{:.3f}")
        _, self.center_x_lbl    = _row("Center X:",      self.center_x_var,
                                        0.0, 1.0,  0.50,  3, "{:.2f}")
        _, self.center_y_lbl    = _row("Center Y:",      self.center_y_var,
                                        0.0, 1.0,  0.50,  4, "{:.2f}")
        _, self.base_rotate_lbl = _row("Base Rotation:", self.base_rotate_var,
                                        0,   360,  270,   5)
        _, self.scale_lbl       = _row("Subject Scale:", self.scale_var,
                                        0.2, 1.0,  0.6,   6, "{:.2f}")

        tuning.grid_columnconfigure(1, weight=1)
        ttk.Button(tuning, text="Reset to Defaults",
                   command=self._reset_warp_params).grid(
                       row=7, column=0, columnspan=3, pady=8)

        # --- Actions ---
        actions = ttk.Frame(left)
        actions.pack(pady=(6, 2))
        self.btn_preview = ttk.Button(actions, text="Start Preview",
                                       command=self._start_preview)
        self.btn_stop_preview = ttk.Button(actions, text="Stop Preview",
                                            command=self._stop_preview, state="disabled")
        self.btn_fullscreen = ttk.Button(actions, text="Open Fullscreen",
                                          command=self._start_fullscreen)
        self.btn_close_fullscreen = ttk.Button(actions, text="Close Fullscreen",
                                                command=self._stop_fullscreen,
                                                state="disabled")
        back_btn = ttk.Button(actions, text="Back",
                               command=lambda: controller.show_page("HomePage"))
        self.btn_preview.grid(row=0, column=0, padx=6)
        self.btn_stop_preview.grid(row=0, column=1, padx=6)
        self.btn_fullscreen.grid(row=0, column=2, padx=6)
        self.btn_close_fullscreen.grid(row=0, column=3, padx=6)
        back_btn.grid(row=0, column=4, padx=6)

        # --- Status ---
        status_box = ttk.Frame(left)
        status_box.pack(fill="x", padx=4, pady=(8, 0))
        self.status = tk.StringVar(value="Status: idle")
        ttk.Label(status_box, textvariable=self.status).pack(anchor="w")

        # --- Right: fixed-size preview pane ---
        ttk.Label(right, text="Preview", style="Body.TLabel").pack()
        preview_container = tk.Frame(right, width=PREVIEW_W, height=PREVIEW_H,
                                     bg="black", highlightthickness=0)
        preview_container.pack(padx=4, pady=4)
        preview_container.pack_propagate(False)
        self._preview_label = tk.Label(preview_container, bg="black",
                                        bd=0, highlightthickness=0)
        self._preview_label.place(relx=0.5, rely=0.5, anchor="center")

        # --- Internal state ---
        self._preview_img      = None
        self._last_bgr         = None
        self._frame_lock       = threading.Lock()
        self._cap              = None
        self._preview_thread   = None
        self._stop_preview_evt = threading.Event()
        self._rs_pipeline      = None
        self._rs_align         = None
        self._last_depth_vis   = None

        self._webcam_only_widgets = [
            self.idx_combo, self.backend_combo,
            self.name_entry, self.names_combo,
            self.res_combo,  self.fps_entry,
        ]

        self.fs_win         = None
        self._fs_label      = None
        self._fs_img        = None
        self._fs_running    = False
        self._fs_mode       = "normal"
        self._fs_mode_label = None

        # Build initial warp maps
        self._warp_maps = self._build_maps()

        self._update_controls()
        self.after(33, self._preview_tick)

    # ---------- Warp map management ----------
    def _build_maps(self):
        return build_four_cone_maps(
            frame_size   = FRAME_SIZE,
            canvas_size  = CANVAS_SIZE,
            span_deg     = int(self.span_var.get()),
            r_inner_frac = self.r_inner_var.get(),
            r_outer_frac = self.r_outer_var.get(),
            center_frac  = (self.center_x_var.get(), self.center_y_var.get()),
            radius_frac  = 1.00,
            base_rotate  = self.base_rotate_var.get(),
        )

    def _on_warp_change(self, _=None):
        self.span_lbl.config(       text=f"{int(self.span_var.get())}")
        self.r_inner_lbl.config(    text=f"{self.r_inner_var.get():.3f}")
        self.r_outer_lbl.config(    text=f"{self.r_outer_var.get():.3f}")
        self.center_x_lbl.config(   text=f"{self.center_x_var.get():.2f}")
        self.center_y_lbl.config(   text=f"{self.center_y_var.get():.2f}")
        self.base_rotate_lbl.config(text=f"{int(self.base_rotate_var.get())}")
        self.scale_lbl.config(      text=f"{self.scale_var.get():.2f}")
        self._warp_maps = self._build_maps()

    def _reset_warp_params(self):
        self.span_var.set(90)
        self.r_inner_var.set(0.08)
        self.r_outer_var.set(0.995)
        self.center_x_var.set(0.50)
        self.center_y_var.set(0.50)
        self.base_rotate_var.set(270.0)
        self.scale_var.set(0.6)
        self._on_warp_change()

    # ---------- UI helpers ----------
    def _update_controls(self):
        mode = self.sel_mode.get()
        for w in (self.idx_combo, self.backend_combo):
            w.config(state=("normal" if mode == "index" else "disabled"))
        for w in (self.name_entry, self.names_combo):
            w.config(state=("normal" if mode == "name" else "disabled"))
        if self._cam_src.get() == "realsense":
            for w in self._webcam_only_widgets:
                try:
                    w.config(state="disabled")
                except Exception:
                    pass

    def _on_src_change(self):
        is_webcam = self._cam_src.get() == "webcam"
        if is_webcam:
            self._update_controls()
            self.res_combo.config(state="readonly")
            self.fps_entry.config(state="normal")
        else:
            for w in self._webcam_only_widgets:
                try:
                    w.config(state="disabled")
                except Exception:
                    pass
        if self._preview_thread and self._preview_thread.is_alive():
            self._stop_preview()
            self.after(250, self._start_preview)

    def _list_names_ffmpeg(self):
        if not IS_WINDOWS:
            messagebox.showinfo(
                "Camera names",
                "Camera-name selection is only available on Windows. Use a camera index on this Mac.",
            )
            return
        vids, _ = _list_dshow_devices_via_ffmpeg()
        if vids:
            self.names_combo["values"] = vids
            self.names_combo.set(vids[0])
        else:
            messagebox.showwarning("ffmpeg",
                "No DirectShow video devices found.\n"
                "Close Zoom/Teams/OBS and retry.")

    def _use_selected_name(self):
        name = self.names_combo.get().strip()
        if name:
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, name)

    def _scan_indices(self, max_probe=6):
        friendly = []
        vids, _ = _list_dshow_devices_via_ffmpeg() if IS_WINDOWS else ([], [])
        for i in range(max_probe):
            for label, api in BACKENDS:
                cap = cv2.VideoCapture(i, api)
                if cap.isOpened():
                    cap.release()
                    pretty = vids[i] if i < len(vids) else "Camera"
                    friendly.append(f"{i} – {pretty} ({label})")
                    break
        return friendly or ["(no cameras found)"]

    def _rescan_indices(self):
        vals = self._scan_indices()
        self.idx_combo["values"] = vals
        self.idx_combo.set(vals[0])

    # ---------- Preview ----------
    def _start_preview(self):
        if self._preview_thread and self._preview_thread.is_alive():
            return

        if self._cam_src.get() == "realsense":
            if not REALSENSE_AVAILABLE:
                messagebox.showerror("RealSense",
                    "pyrealsense2 is not installed.\n\nRun:\n"
                    "  winvenv310\\Scripts\\pip install pyrealsense2")
                return
            try:
                self._rs_pipeline, self._rs_align = _rs_start()
            except Exception as e:
                messagebox.showerror("RealSense",
                    f"Could not open RealSense camera:\n\n{e}")
                return
            self._cap      = None
            status_msg     = f"Status: previewing (RealSense {RS_W}x{RS_H} @ {RS_FPS}fps)"
        else:
            if self.sel_mode.get() == "index":
                sel = self.idx_combo.get()
                try:
                    raw = sel.split("–")[0] if "–" in sel else sel.split("-")[0]
                    idx = int(raw.strip())
                except Exception:
                    idx = 0
                api_label = self.backend_combo.get()
                api_pref  = dict(BACKENDS)[api_label]
                cap, _, _ = _open_by_index(idx, api_pref)
                if not cap:
                    messagebox.showerror("Camera",
                        f"Could not open index {idx} with {api_label}.")
                    return
            else:
                name = self.name_entry.get().strip()
                if not name:
                    messagebox.showwarning("Camera", "Enter/select a device name first.")
                    return
                cap, _, _ = _open_by_name_dshow(name)
                if not cap:
                    messagebox.showerror("Camera",
                        f"Could not open device by name:\n{name}")
                    return

            try:
                w_str, h_str = self.res_combo.get().split("x")
                width, height = int(w_str), int(h_str)
            except Exception:
                width, height = 1280, 720
            try:
                fps = max(1, int(self.fps_entry.get()))
            except Exception:
                fps = 30

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS,          fps)

            self._cap            = cap
            self._rs_pipeline    = None
            self._rs_align       = None
            self._last_depth_vis = None
            status_msg           = "Status: previewing"

        self._stop_preview_evt.clear()
        self._preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._preview_thread.start()
        self.btn_preview.config(state="disabled")
        self.btn_stop_preview.config(state="normal")
        self.status.set(status_msg)

    def _stop_preview(self):
        self._stop_preview_evt.set()
        if self._preview_thread:
            self._preview_thread.join(timeout=1.5)
        self._preview_thread = None
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
        if self._rs_pipeline:
            try:
                self._rs_pipeline.stop()
            except Exception:
                pass
        self._rs_pipeline = None
        self._rs_align    = None
        self.btn_preview.config(state="normal")
        self.btn_stop_preview.config(state="disabled")
        self.status.set("Status: idle")

    def shutdown(self):
        self._stop_fullscreen()
        self._stop_preview()
        close = getattr(self._segmentor, "close", None)
        if callable(close):
            close()

    def _preview_loop(self):
        if self._rs_pipeline is not None:
            self._preview_loop_realsense()
        else:
            self._preview_loop_webcam()

    def _preview_loop_webcam(self):
        try:
            while not self._stop_preview_evt.is_set():
                ok, frame = self._cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                with self._frame_lock:
                    self._last_bgr = frame
        except Exception as e:
            self.after(0, lambda e=e: messagebox.showerror("Preview error", str(e)))

    def _preview_loop_realsense(self):
        warmup = RS_WARMUP_FRAMES
        fail_count = 0
        while not self._stop_preview_evt.is_set():
            try:
                color_bgr, depth_vis = _rs_get_frames(self._rs_pipeline, self._rs_align)
                fail_count = 0
            except RuntimeError as e:
                fail_count += 1
                self.after(0, lambda n=fail_count: self.status.set(
                    f"Status: frame timeout ({n})"))
                if fail_count >= 10:
                    self.after(0, lambda: self.status.set(
                        "Status: ERROR — unplug & replug RealSense"))
                    break
                continue
            except Exception as e:
                self.after(0, lambda m=str(e): self.status.set(f"Status: Error: {m}"))
                break

            if color_bgr is None:
                continue
            if warmup > 0:
                warmup -= 1
                continue

            with self._frame_lock:
                self._last_bgr       = color_bgr
                self._last_depth_vis = depth_vis

    def _preview_tick(self):
        """Show raw (unwarped) camera frame in the side preview pane."""
        frame = None
        with self._frame_lock:
            if self._last_bgr is not None:
                frame = self._last_bgr

        if frame is not None:
            fh, fw = frame.shape[:2]
            scale  = min(PREVIEW_W / fw, PREVIEW_H / fh)
            new_w  = max(1, int(fw * scale))
            new_h  = max(1, int(fh * scale))
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._preview_label.config(image=img)
            self._preview_label.image = img

        self.after(33, self._preview_tick)

    # ---------- Fullscreen ----------
    def _start_fullscreen(self):
        if self.fs_win and self._fs_running:
            return
        if self._preview_thread is None or not self._preview_thread.is_alive():
            messagebox.showwarning("Fullscreen", "Start the camera preview first.")
            return

        self.fs_win = tk.Toplevel(self)
        self.fs_win.title("Circle Hologram Display")
        self.fs_win.attributes("-fullscreen", True)
        self.fs_win.configure(bg="black")
        self.fs_win.bind("<Escape>", lambda e: self._stop_fullscreen())
        self.fs_win.bind("q",        lambda e: self._stop_fullscreen())
        self.fs_win.bind("Q",        lambda e: self._stop_fullscreen())
        self.fs_win.bind("m",        lambda e: self._fs_cycle_mode())
        self.fs_win.bind("M",        lambda e: self._fs_cycle_mode())
        self.fs_win.bind("s",        lambda e: self._fs_switch_source())
        self.fs_win.bind("S",        lambda e: self._fs_switch_source())
        self.fs_win.protocol("WM_DELETE_WINDOW", self._stop_fullscreen)

        self._fs_label = tk.Label(self.fs_win, bg="black")
        self._fs_label.pack(fill="both", expand=True)

        self._fs_mode       = "normal"
        self._fs_mode_label = tk.Label(
            self.fs_win, text=self._fs_mode_text(),
            font=("Segoe UI", 11), fg="#cccccc", bg="#1a1a1a",
            padx=12, pady=5)
        self._fs_mode_label.place(x=14, y=14, anchor="nw")

        self._fs_hint = tk.Label(
            self.fs_win,
            text="Q / Esc: exit   ·   M: cycle mode   ·   S: switch source",
            font=("Segoe UI", 10), fg="#aaaaaa", bg="#1a1a1a",
            padx=14, pady=5)
        self._fs_hint.place(relx=0.5, rely=1.0, anchor="s", y=-14)

        self._fs_running = True
        self.btn_fullscreen.config(state="disabled")
        self.btn_close_fullscreen.config(state="normal")
        self.status.set("Status: fullscreen output")
        self._fullscreen_tick()

    def _stop_fullscreen(self):
        self._fs_running = False
        if self.fs_win:
            try:
                self.fs_win.destroy()
            except Exception:
                pass
        self.fs_win         = None
        self._fs_label      = None
        self._fs_img        = None
        self._fs_mode_label = None
        self.btn_fullscreen.config(state="normal")
        self.btn_close_fullscreen.config(state="disabled")
        self.status.set("Status: idle")

    _MODE_LABELS = {
        "normal": "Normal  (background removed)",
        "raw":    "Raw  (no segmentation)",
        "depth":  "Depth map  (RealSense)",
    }

    def _fs_mode_text(self):
        return f"Mode:  {self._MODE_LABELS.get(self._fs_mode, self._fs_mode)}"

    def _fs_cycle_mode(self):
        modes = ["normal", "raw"]
        with self._frame_lock:
            has_depth = self._last_depth_vis is not None
        if has_depth:
            modes.append("depth")
        if self._fs_mode not in modes:
            self._fs_mode = "normal"
        else:
            self._fs_mode = modes[(modes.index(self._fs_mode) + 1) % len(modes)]
        if self._fs_mode_label:
            self._fs_mode_label.config(text=self._fs_mode_text())

    def _fs_switch_source(self):
        if not REALSENSE_AVAILABLE:
            return
        new_src = "realsense" if self._cam_src.get() == "webcam" else "webcam"
        self._cam_src.set(new_src)
        if new_src == "webcam" and self._fs_mode == "depth":
            self._fs_mode = "normal"
            if self._fs_mode_label:
                self._fs_mode_label.config(text=self._fs_mode_text())
        self._on_src_change()

    def _fullscreen_tick(self):
        if not self._fs_running:
            return

        frame = None
        with self._frame_lock:
            if self._fs_mode == "depth" and self._last_depth_vis is not None:
                frame = self._last_depth_vis.copy()
            elif self._last_bgr is not None:
                frame = self._last_bgr.copy()

        if frame is not None:
            use_seg = (self._fs_mode == "normal")
            canvas  = self._apply_circle_hologram(frame, use_segmentation=use_seg)

            try:
                sw = self.fs_win.winfo_width()
                sh = self.fs_win.winfo_height()
                fh, fw = canvas.shape[:2]
                scale  = min(sw / fw, sh / fh)
                new_w  = max(1, int(fw * scale))
                new_h  = max(1, int(fh * scale))
                interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
                disp   = cv2.resize(canvas, (new_w, new_h), interpolation=interp)
            except Exception:
                disp = canvas

            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            if self._fs_label is not None:
                self._fs_label.config(image=img)
                self._fs_label.image = img
                self._fs_img = img

        self.after(16, self._fullscreen_tick)

    # ---------- Rendering ----------
    def _apply_circle_hologram(self, frame_bgr, use_segmentation=True):
        """
        Background-remove the frame, then pass it through four cone-warp arcs
        composited symmetrically around the centre.
        """
        # 1) Square input
        sq = cv2.resize(frame_bgr, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)

        # 2) Optional background removal
        if use_segmentation and self._segmentor is not None:
            rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
            try:
                seg  = self._segmentor.process(rgb)
                mask = seg.segmentation_mask > 0.5
                fg   = np.zeros_like(sq)
                fg[mask] = sq[mask]
            except Exception:
                fg = sq
        else:
            fg = sq

        # 3) Apply four cone-warp faces via the module-level helper
        return apply_four_cone_views(
            fg,
            warp_maps      = self._warp_maps,
            frame_size     = FRAME_SIZE,
            subject_scale  = float(self.scale_var.get()),
            saturation_scale = 1.4,
            contrast_alpha   = 1.8,
            brightness_beta  = -25,
        )
