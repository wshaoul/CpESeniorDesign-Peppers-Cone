# live_view.py
#
# Live display view.
# - Camera selection by index (MSMF/DSHOW/ANY) or by name (DSHOW via ffmpeg listing)
# - Stable, LARGE in-app preview on the right (fixed-size container; UI doesn't jump)
# - Fullscreen output window to show your warped result (placeholder provided)

import time
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import mediapipe as mp
import math

import cv2
from PIL import Image, ImageTk

# Optional RealSense support
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

# Big (but reasonable) 16:9 preview area; tweak if you want
PREVIEW_W = 800
PREVIEW_H = 450

FRAME_SIZE = 400
CANVAS_SIZE = 800


# ---------- Warp Helpers ----------
def enhance_saturation_contrast(image_bgr, saturation_scale=1.3, contrast_alpha=1.2, brightness_beta=10):
    # BGR → HSV (float), boost S, back to BGR, then contrast/brightness
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast_alpha, beta=brightness_beta)
    return enhanced


def build_cone_maps(
    frame_size: int,
    canvas_size: int,
    span_deg: int = 170,
    rotate_deg: float = 0.0,
    r_inner_frac: float = 0.10,
    r_outer_frac: float = 0.99,
    center_frac=(0.50, 0.50),  # <<< NEW: move center of cone
    radius_frac: float = 1.00,
):  # <<< NEW: scale radius a bit
    """
    Warp for Pepper's Cone showing ONE image on the front arc, mapped only to a radial band.
    Returns map_x, map_y (float32) with -1 for out-of-bounds.
    """

    # Initialize output arrays
    map_x = np.full((canvas_size, canvas_size), -1, dtype=np.float32)
    map_y = np.full((canvas_size, canvas_size), -1, dtype=np.float32)

    # Center and radius
    cx = int(center_frac[0] * canvas_size)
    cy = int(center_frac[1] * canvas_size)
    R = int((canvas_size * 0.5) * max(0.10, min(2.0, radius_frac)))

    # Clamp radii
    r_in_frac = max(0.0, min(0.99, r_inner_frac))
    r_out_frac = max(r_in_frac + 1.0 / max(1, R), min(1.0, r_outer_frac))
    r_in = r_in_frac * R
    r_out = r_out_frac * R

    # Angle setup
    half = math.radians(max(1, min(359, span_deg))) * 0.5
    rot = math.radians(rotate_deg)

    # Create coordinate grids
    y_coords, x_coords = np.ogrid[0:canvas_size, 0:canvas_size]
    dx = x_coords - cx
    dy = y_coords - cy

    # Calculate radius and angle for all pixels at once
    r = np.sqrt(dx**2 + dy**2)
    ang = np.arctan2(dy, dx) - rot

    # Wrap angles to [-pi, pi]
    ang = np.where(ang < -np.pi, ang + 2 * np.pi, ang)
    ang = np.where(ang > np.pi, ang - 2 * np.pi, ang)

    # Create mask for valid pixels (inside the band and angle range)
    valid_mask = (r >= r_in) & (r <= r_out) & (ang >= -half) & (ang <= half)

    # Calculate UV coordinates only for valid pixels
    if np.any(valid_mask):
        angle_to_u = 1.0 / (2 * half)
        r_to_v = 1.0 / max(1.0, (r_out - r_in))

        u = (ang[valid_mask] + half) * angle_to_u
        v = 1.0 - ((r[valid_mask] - r_in) * r_to_v)

        # Clamp to [0, 1] and convert to pixel coordinates
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)

        sx = (u * (frame_size - 1)).astype(np.float32)
        sy = (v * (frame_size - 1)).astype(np.float32)

        # Assign to output maps
        map_x[valid_mask] = sx
        map_y[valid_mask] = sy

    return map_x, map_y

# ---------- RealSense helpers ----------
# for smoother camera output, change to 640, 360, 60 (lower resolution but higher fps)
# RS_W, RS_H, RS_FPS  = 1280, 720, 30 
RS_W, RS_H, RS_FPS  = 640, 360, 60
RS_TIMEOUT_MS       = 3000
RS_WARMUP_FRAMES    = 10


def _rs_start():
    """Start RealSense color+depth pipeline. Raises on failure."""
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
    """
    Grab one aligned frameset.
    Returns (color_bgr, depth_vis) where depth_vis is a COLORMAP_TURBO BGR image.
    Either value may be None if the corresponding frame is unavailable.
    """
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


# ---------- Backends ----------
BACKENDS = [
    ("MSMF (Windows 10/11)", cv2.CAP_MSMF),
    ("DSHOW (DirectShow)", cv2.CAP_DSHOW),
    ("ANY (let OpenCV pick)", cv2.CAP_ANY),
]


def _open_by_index(index: int, api_pref: int):
    cap = cv2.VideoCapture(index, api_pref)
    if cap.isOpened():
        return cap, "index", api_pref
    cap.release()
    return None, None, None


def _open_by_name_dshow(name: str):
    # Mirrors record_view for consistency.
    src = f"video={name}"
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap, "name", cv2.CAP_DSHOW
    cap.release()
    return None, None, None


def _list_dshow_devices_via_ffmpeg():
    """Return (video_names, audio_names) using ffmpeg's DirectShow listing."""
    try:
        proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out, _ = proc.communicate(timeout=6)
    except Exception:
        return [], []

    import re

    video, audio = [], []
    m = re.compile(r'^\[dshow .*?\]\s+"([^"]+)"\s+\((video|audio)\)\s*$', re.IGNORECASE)
    for line in (out or "").splitlines():
        line = line.strip()
        mm = m.match(line)
        if not mm:
            continue
        name, kind = mm.group(1), mm.group(2).lower()
        (video if kind == "video" else audio).append(name)

    # dedupe while preserving order
    def dedup(xs):
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return dedup(video), dedup(audio)


class LiveView(ttk.Frame):
    """Live display (preview + fullscreen) using the same open logic as record_view."""

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Source toggle — initialized here so UI widgets can bind to it immediately
        self._cam_src = tk.StringVar(value="webcam")

        # --- Title ---
        ttk.Label(self, text="Live Display", style="Header.TLabel").pack(pady=(10, 4))
        ttk.Label(self, text="Pick a camera (index or name), preview it, then open fullscreen for the warped output.", style="Body.TLabel").pack(pady=(0, 10))

        # Two panes: left controls, right preview
        top = ttk.Frame(self)
        top.pack(fill="both", expand=True)
        left = ttk.Frame(top)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right = ttk.Frame(top)
        right.pack(side="right", fill="both", expand=True, padx=(8, 0))

        # --- Camera Source toggle ---
        src_box = ttk.LabelFrame(left, text="Camera Source")
        src_box.pack(fill="x", padx=4, pady=(0, 10))
        ttk.Radiobutton(
            src_box, text="Webcam", variable=self._cam_src,
            value="webcam", command=self._on_src_change,
        ).pack(side="left", padx=14, pady=6)
        ttk.Radiobutton(
            src_box, text="RealSense D455/D555", variable=self._cam_src,
            value="realsense", command=self._on_src_change,
            state="normal" if REALSENSE_AVAILABLE else "disabled",
        ).pack(side="left", padx=14, pady=6)
        if not REALSENSE_AVAILABLE:
            ttk.Label(src_box, text="(install pyrealsense2 to enable)",
                      foreground="#888").pack(side="left", padx=6)

        # --- Camera selection ---
        cam_box = ttk.LabelFrame(left, text="Camera Selection")
        cam_box.pack(fill="x", padx=4, pady=(0, 10))
        cam_box.grid_columnconfigure(0, weight=1)

        self.sel_mode = tk.StringVar(value="index")
        r1 = ttk.Radiobutton(cam_box, text="By Index", variable=self.sel_mode, value="index", command=self._update_controls)
        r2 = ttk.Radiobutton(cam_box, text="By Name (DSHOW)", variable=self.sel_mode, value="name", command=self._update_controls)

        # index row
        idx_row = ttk.Frame(cam_box)
        ttk.Label(idx_row, text="Index:").grid(row=0, column=0, padx=(0, 6))
        self.idx_combo = ttk.Combobox(idx_row, state="readonly", width=36, values=self._scan_indices())
        self.idx_combo.set(self.idx_combo["values"][0])
        self.idx_combo.grid(row=0, column=1, sticky="w")
        ttk.Button(idx_row, text="Rescan", command=self._rescan_indices).grid(row=0, column=2, padx=6)
        ttk.Label(idx_row, text="Backend:").grid(row=0, column=3, padx=(12, 6))
        self.backend_combo = ttk.Combobox(idx_row, state="readonly", values=[label for (label, _) in BACKENDS], width=22)
        self.backend_combo.set(BACKENDS[0][0])
        self.backend_combo.grid(row=0, column=4, padx=(0, 6))

        # name row
        name_row = ttk.Frame(cam_box)
        ttk.Label(name_row, text="Device Name:").grid(row=0, column=0, padx=(0, 6))
        self.name_entry = ttk.Entry(name_row, width=36)
        self.name_entry.grid(row=0, column=1)
        ttk.Button(name_row, text="List Cameras (ffmpeg)", command=self._list_names_ffmpeg).grid(row=0, column=2, padx=6)
        self.names_combo = ttk.Combobox(name_row, state="readonly", width=36, values=[])
        self.names_combo.grid(row=1, column=1, pady=(6, 0), sticky="w")
        ttk.Button(name_row, text="Use Selected", command=self._use_selected_name).grid(row=1, column=2, padx=6, pady=(6, 0))

        # Arrange like: radio, its row; radio, its row
        r1.grid(row=0, column=0, sticky="w", padx=8, pady=(6, 2))
        idx_row.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 6))
        r2.grid(row=2, column=0, sticky="w", padx=8, pady=(8, 2))
        name_row.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 6))

        # --- Video settings ---
        settings = ttk.LabelFrame(left, text="Video Settings")
        settings.pack(fill="x", padx=4, pady=(0, 10))
        ttk.Label(settings, text="Resolution:").grid(row=0, column=0, padx=(8, 6), pady=8, sticky="w")
        self.res_combo = ttk.Combobox(settings, state="readonly", width=12, values=["1280x720", "1920x1080", "640x480"])
        self.res_combo.set("1280x720")
        self.res_combo.grid(row=0, column=1, sticky="w")
        ttk.Label(settings, text="FPS:").grid(row=0, column=2, padx=(16, 6), pady=8, sticky="w")
        self.fps_entry = ttk.Entry(settings, width=6)
        self.fps_entry.insert(0, "30")
        self.fps_entry.grid(row=0, column=3, sticky="w")
        
        # --- Cone Warp Tuning (add after Video Settings section) ---
        tuning = ttk.LabelFrame(left, text="Cone Warp Tuning")
        tuning.pack(fill="x", padx=4, pady=(0, 10))

        # Center X/Y
        ttk.Label(tuning, text="Center X:").grid(row=0, column=0, padx=(8,4), pady=4, sticky="w")
        self.center_x_var = tk.DoubleVar(value=0.50)
        self.center_x_slider = ttk.Scale(tuning, from_=0.0, to=1.0, orient="horizontal", 
                                        variable=self.center_x_var, command=self._on_warp_change)
        self.center_x_slider.grid(row=0, column=1, sticky="ew", padx=4)
        self.center_x_label = ttk.Label(tuning, text="0.50")
        self.center_x_label.grid(row=0, column=2, padx=4)

        ttk.Label(tuning, text="Center Y:").grid(row=1, column=0, padx=(8,4), pady=4, sticky="w")
        self.center_y_var = tk.DoubleVar(value=0.50)
        self.center_y_slider = ttk.Scale(tuning, from_=0.0, to=1.0, orient="horizontal",
                                        variable=self.center_y_var, command=self._on_warp_change)
        self.center_y_slider.grid(row=1, column=1, sticky="ew", padx=4)
        self.center_y_label = ttk.Label(tuning, text="0.50")
        self.center_y_label.grid(row=1, column=2, padx=4)

        # Inner/Outer radius
        ttk.Label(tuning, text="Inner Radius:").grid(row=2, column=0, padx=(8,4), pady=4, sticky="w")
        self.r_inner_var = tk.DoubleVar(value=0.08)
        self.r_inner_slider = ttk.Scale(tuning, from_=0.0, to=0.95, orient="horizontal",
                                        variable=self.r_inner_var, command=self._on_warp_change)
        self.r_inner_slider.grid(row=2, column=1, sticky="ew", padx=4)
        self.r_inner_label = ttk.Label(tuning, text="0.08")
        self.r_inner_label.grid(row=2, column=2, padx=4)

        ttk.Label(tuning, text="Outer Radius:").grid(row=3, column=0, padx=(8,4), pady=4, sticky="w")
        self.r_outer_var = tk.DoubleVar(value=0.995)
        self.r_outer_slider = ttk.Scale(tuning, from_=0.5, to=1.0, orient="horizontal",
                                        variable=self.r_outer_var, command=self._on_warp_change)
        self.r_outer_slider.grid(row=3, column=1, sticky="ew", padx=4)
        self.r_outer_label = ttk.Label(tuning, text="0.995")
        self.r_outer_label.grid(row=3, column=2, padx=4)

        # Span and Rotation
        ttk.Label(tuning, text="Span (deg):").grid(row=4, column=0, padx=(8,4), pady=4, sticky="w")
        self.span_var = tk.DoubleVar(value=200)
        self.span_slider = ttk.Scale(tuning, from_=90, to=360, orient="horizontal",
                                    variable=self.span_var, command=self._on_warp_change)
        self.span_slider.grid(row=4, column=1, sticky="ew", padx=4)
        self.span_label = ttk.Label(tuning, text="200")
        self.span_label.grid(row=4, column=2, padx=4)

        ttk.Label(tuning, text="Rotation (deg):").grid(row=5, column=0, padx=(8,4), pady=4, sticky="w")
        self.rotate_var = tk.DoubleVar(value=270)
        self.rotate_slider = ttk.Scale(tuning, from_=0, to=360, orient="horizontal",
                                        variable=self.rotate_var, command=self._on_warp_change)
        self.rotate_slider.grid(row=5, column=1, sticky="ew", padx=4)
        self.rotate_label = ttk.Label(tuning, text="270")
        self.rotate_label.grid(row=5, column=2, padx=4)

        tuning.grid_columnconfigure(1, weight=1)

        # Reset button
        ttk.Button(tuning, text="Reset to Defaults", command=self._reset_warp_params).grid(
            row=6, column=0, columnspan=3, pady=8)
        
        # --- Actions ---
        actions = ttk.Frame(left)
        actions.pack(pady=(6, 2))
        self.btn_preview = ttk.Button(actions, text="Start Preview", command=self._start_preview)
        self.btn_stop_preview = ttk.Button(actions, text="Stop Preview", command=self._stop_preview, state="disabled")
        self.btn_fullscreen = ttk.Button(actions, text="Open Fullscreen", command=self._start_fullscreen)
        self.btn_close_fullscreen = ttk.Button(actions, text="Close Fullscreen", command=self._stop_fullscreen, state="disabled")
        back_btn = ttk.Button(actions, text="Back", command=lambda: controller.show_page("HomePage"))
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

        # --- Right: LARGE fixed-size preview (no jumping) ---
        ttk.Label(right, text="Preview", style="Body.TLabel").pack()
        preview_container = tk.Frame(right, width=PREVIEW_W, height=PREVIEW_H, bg="black", highlightthickness=0)
        preview_container.pack(padx=4, pady=4)
        preview_container.pack_propagate(False)  # keep container size fixed

        self._preview_label = tk.Label(preview_container, bg="black", bd=0, highlightthickness=0)
        self._preview_label.place(relx=0.5, rely=0.5, anchor="center")

        # Preview state
        self._preview_img = None
        self._last_bgr = None
        self._frame_lock = threading.Lock()
        self._cap = None
        self._preview_thread = None
        self._stop_preview_evt = threading.Event()

        # RealSense pipeline state (None when webcam is active)
        self._rs_pipeline  = None
        self._rs_align     = None
        self._last_depth_vis = None   # colorized depth frame; only set when RS is active

        # Collect webcam-only widgets so _on_src_change can batch-disable them
        self._webcam_only_widgets = [
            self.idx_combo, self.backend_combo,
            self.name_entry, self.names_combo,
            self.res_combo, self.fps_entry,
        ]

        # Fullscreen window state
        self.fs_win        = None
        self._fs_label     = None
        self._fs_img       = None
        self._fs_running   = False
        self._fs_mode      = "normal"   # "normal" | "raw" | "depth"
        self._fs_mode_label = None      # tk.Label overlay showing current mode

        # Precompute warp maps (once)
        self._map_x, self._map_y = build_cone_maps(
            frame_size=FRAME_SIZE,
            canvas_size=CANVAS_SIZE,
            span_deg=270,  # adjust width
            rotate_deg=270,  # adjust rotation
            r_inner_frac=0.06,  # adjust band
            r_outer_frac=0.98,
            center_frac=(0.5, 0.55),  # move center (right/left, up/down)
            radius_frac=1.00,
        )

        self._segmentor = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

        # init
        self._update_controls()
        self.after(33, self._preview_tick)  # UI repaint timer

    # ---------- UI helpers ----------
    def _update_controls(self):
        """Apply index-vs-name state, then honour the camera source toggle."""
        mode = self.sel_mode.get()
        for w in (self.idx_combo, self.backend_combo):
            w.config(state=("normal" if mode == "index" else "disabled"))
        for w in (self.name_entry, self.names_combo):
            w.config(state=("normal" if mode == "name" else "disabled"))
        # If RealSense is selected, override all webcam widgets to disabled
        if self._cam_src.get() == "realsense":
            for w in self._webcam_only_widgets:
                try: w.config(state="disabled")
                except Exception: pass

    def _on_src_change(self):
        """Toggle between Webcam and RealSense. Restarts preview if already running."""
        is_webcam = self._cam_src.get() == "webcam"
        if is_webcam:
            self._update_controls()           # restore index/name sub-state
            self.res_combo.config(state="readonly")
            self.fps_entry.config(state="normal")
        else:
            for w in self._webcam_only_widgets:
                try: w.config(state="disabled")
                except Exception: pass

        # Auto-restart preview when toggled mid-session
        if self._preview_thread and self._preview_thread.is_alive():
            self._stop_preview()
            self.after(250, self._start_preview)

    def _list_names_ffmpeg(self):
        vids, _ = _list_dshow_devices_via_ffmpeg()
        if vids:
            self.names_combo["values"] = vids
            self.names_combo.set(vids[0])
        else:
            messagebox.showwarning("ffmpeg", "No DirectShow video devices found.\nClose Zoom/Teams/OBS and retry.")

    def _use_selected_name(self):
        name = self.names_combo.get().strip()
        if name:
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, name)

    def _scan_indices(self, max_probe=6):
        """Build friendly labels like '0 – Integrated Webcam (MSMF)'."""
        friendly = []
        vids, _ = _list_dshow_devices_via_ffmpeg()  # may be empty; that's fine
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
            return  # already running

        if self._cam_src.get() == "realsense":
            # ── RealSense path ────────────────────────────────────────────────
            if not REALSENSE_AVAILABLE:
                messagebox.showerror("RealSense",
                    "pyrealsense2 is not installed.\n\nRun:\n"
                    "  winvenv310\\Scripts\\pip install pyrealsense2")
                return
            try:
                self._rs_pipeline, self._rs_align = _rs_start()
            except Exception as e:
                messagebox.showerror("RealSense",
                    f"Could not open RealSense camera:\n\n{e}\n\n"
                    "Check:\n"
                    "  • Camera plugged into USB 3.x (blue/SS) port\n"
                    "  • Intel RealSense Viewer is closed\n"
                    "  • No other app is using the camera")
                return
            self._cap = None
            status_msg = f"Status: previewing (RealSense {RS_W}x{RS_H} @ {RS_FPS}fps)"
        else:
            # ── Webcam path ───────────────────────────────────────────────────
            if self.sel_mode.get() == "index":
                sel = self.idx_combo.get()
                try:
                    raw = sel.split("–")[0] if "–" in sel else sel.split("-")[0]
                    idx = int(raw.strip())
                except Exception:
                    try:
                        idx = int(sel.strip())
                    except Exception:
                        idx = 0
                api_label = self.backend_combo.get()
                api_pref = dict(BACKENDS)[api_label]
                cap, _, _ = _open_by_index(idx, api_pref)
                if not cap:
                    messagebox.showerror("Camera", f"Could not open index {idx} with {api_label}. Try name mode.")
                    return
            else:
                name = self.name_entry.get().strip()
                if not name:
                    messagebox.showwarning("Camera", "Enter/select a device name first.")
                    return
                cap, _, _ = _open_by_name_dshow(name)
                if not cap:
                    messagebox.showerror("Camera", f"Could not open device by name:\n{name}\n(Use the list button and match exactly.)")
                    return

            # size/fps
            try:
                w_str, h_str = self.res_combo.get().split("x")
                width, height = int(w_str), int(h_str)
            except Exception:
                width, height = 1280, 720
            try:
                fps = max(1, int(self.fps_entry.get()))
            except Exception:
                fps = 30

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)

            self._cap            = cap
            self._rs_pipeline    = None
            self._rs_align       = None
            self._last_depth_vis = None   # no depth when using webcam
            status_msg = "Status: previewing"

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
            try: self._cap.release()
            except Exception: pass
        self._cap = None
        if self._rs_pipeline:
            try: self._rs_pipeline.stop()
            except Exception: pass
        self._rs_pipeline = None
        self._rs_align    = None
        self.btn_preview.config(state="normal")
        self.btn_stop_preview.config(state="disabled")
        self.status.set("Status: idle")

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
        warmup     = RS_WARMUP_FRAMES
        fail_count = 0
        print(f"[RS capture] thread started, warming up {warmup} frames...")
        while not self._stop_preview_evt.is_set():
            try:
                color_bgr, depth_vis = _rs_get_frames(self._rs_pipeline, self._rs_align)
                fail_count = 0
            except RuntimeError as e:
                fail_count += 1
                print(f"[RS capture] timeout ({fail_count}): {e}")
                self.after(0, lambda n=fail_count: self.status.set(f"Status: frame timeout ({n})"))
                if fail_count >= 10:
                    self.after(0, lambda: self.status.set(
                        "Status: ERROR — unplug & replug RealSense camera"))
                    break
                continue
            except Exception as e:
                print(f"[RS capture] fatal: {e}")
                self.after(0, lambda m=str(e): self.status.set(f"Status: Error: {m}"))
                break

            if color_bgr is None:
                continue
            if warmup > 0:
                warmup -= 1
                if warmup == 0:
                    print("[RS capture] warmup done, displaying frames")
                continue

            with self._frame_lock:
                self._last_bgr       = color_bgr
                self._last_depth_vis = depth_vis
        print("[RS capture] thread exiting")

    def _preview_tick(self):
        """Show RAW camera frame (unwarped) in the small preview pane."""
        frame = None
        with self._frame_lock:
            if self._last_bgr is not None:
                frame = self._last_bgr

        if frame is not None:
            fh, fw = frame.shape[:2]
            scale = min(PREVIEW_W / fw, PREVIEW_H / fh)
            new_w = max(1, int(fw * scale))
            new_h = max(1, int(fh * scale))
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # IMPORTANT: no warp here — preview shows the unwarped camera feed
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._preview_label.config(image=img)
            self._preview_label.image = img  # keep a reference

        self.after(33, self._preview_tick)  # ~30 fps UI update

    # ---------- Fullscreen display ----------
    def _start_fullscreen(self):
        if self.fs_win and self._fs_running:
            return
        # Guard: check the preview thread, not _cap (which is None when RealSense is active)
        if self._preview_thread is None or not self._preview_thread.is_alive():
            messagebox.showwarning("Fullscreen", "Start the camera preview first.")
            return

        self.fs_win = tk.Toplevel(self)
        self.fs_win.title("Pepper's Cone Display")
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

        # Mode label — top-left, updated by _fs_cycle_mode
        self._fs_mode      = "normal"
        self._fs_mode_label = tk.Label(
            self.fs_win,
            text=self._fs_mode_text(),
            font=("Segoe UI", 11),
            fg="#cccccc", bg="#1a1a1a",
            padx=12, pady=5,
        )
        self._fs_mode_label.place(x=14, y=14, anchor="nw")

        # Hint overlay — bottom-centre
        self._fs_hint = tk.Label(
            self.fs_win,
            text="Q / Esc: exit   ·   M: cycle mode   ·   S: switch source",
            font=("Segoe UI", 10),
            fg="#aaaaaa", bg="#1a1a1a",
            padx=14, pady=5,
        )
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

    # ---------- Fullscreen mode / source helpers ----------

    _MODE_LABELS = {
        "normal": "Normal  (background removed)",
        "raw":    "Raw  (no segmentation)",
        "depth":  "Depth map  (RealSense)",
    }

    def _fs_mode_text(self):
        return f"Mode:  {self._MODE_LABELS.get(self._fs_mode, self._fs_mode)}"

    def _fs_cycle_mode(self):
        """Cycle display mode: normal → raw → depth (depth skipped if unavailable)."""
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
        """Toggle between Webcam and RealSense while fullscreen stays open."""
        if not REALSENSE_AVAILABLE:
            return
        new_src = "realsense" if self._cam_src.get() == "webcam" else "webcam"
        self._cam_src.set(new_src)
        # If we were showing depth but switching to webcam, fall back to normal
        if new_src == "webcam" and self._fs_mode == "depth":
            self._fs_mode = "normal"
            if self._fs_mode_label:
                self._fs_mode_label.config(text=self._fs_mode_text())
        self._on_src_change()   # stops current capture, restarts with new source

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
            warped  = self._apply_warp(frame, use_segmentation=use_seg)

            # Fit to current screen size while preserving aspect
            try:
                sw = self.fs_win.winfo_width()
                sh = self.fs_win.winfo_height()
                fh, fw = warped.shape[:2]
                scale = min(sw / fw, sh / fh)
                new_w, new_h = max(1, int(fw * scale)), max(1, int(fh * scale))
                interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
                disp = cv2.resize(warped, (new_w, new_h), interpolation=interp)
            except Exception:
                disp = warped

            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            if self._fs_label is not None:
                self._fs_label.config(image=img)
                self._fs_label.image = img
                self._fs_img = img  # keep a reference

        # Aim ~60 Hz for smoother motion; adjust as needed
        self.after(16, self._fullscreen_tick)

    # ---------- Warp Display ----------
    def _apply_warp(self, frame_bgr, use_segmentation=True):
        """
        Takes a BGR frame from the camera, returns a warped BGR image.
        Steps mirror your CircularConeLive.py:
        - resize to FRAME_SIZE x FRAME_SIZE
        - optional background removal (MediaPipe)
        - center/scale subject
        - remap with precomputed circular-cone maps
        - saturation/contrast enhancement
        """
        # 1) square input
        sq = cv2.resize(frame_bgr, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)

        # 2) optional background removal
        if use_segmentation and self._segmentor is not None:
            rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
            try:
                seg = self._segmentor.process(rgb)
                mask = seg.segmentation_mask > 0.5
                fg = np.zeros_like(sq)
                fg[mask] = sq[mask]
            except Exception:
                fg = sq
        else:
            fg = sq

        # 3) center + scale (same defaults as prototype: 0.6)
        scale = 0.6
        scaled = cv2.resize(fg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        padded = np.zeros_like(fg)
        y_off = (FRAME_SIZE - scaled.shape[0]) // 2
        x_off = (FRAME_SIZE - scaled.shape[1]) // 2
        padded[y_off : y_off + scaled.shape[0], x_off : x_off + scaled.shape[1]] = scaled

        # 4) cone warp
        warped = cv2.remap(padded, self._map_x, self._map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # 5) color pop
        enhanced = enhance_saturation_contrast(warped, saturation_scale=1.4, contrast_alpha=1.8, brightness_beta=-25)

        return enhanced
    
    def _on_warp_change(self, _=None):
        """Rebuild warp maps when any slider changes."""
        # Update labels
        self.center_x_label.config(text=f"{self.center_x_var.get():.2f}")
        self.center_y_label.config(text=f"{self.center_y_var.get():.2f}")
        self.r_inner_label.config(text=f"{self.r_inner_var.get():.3f}")
        self.r_outer_label.config(text=f"{self.r_outer_var.get():.3f}")
        self.span_label.config(text=f"{int(self.span_var.get())}")
        self.rotate_label.config(text=f"{int(self.rotate_var.get())}")
        
        # Rebuild maps
        self._rebuild_warp_maps()

    def _rebuild_warp_maps(self):
        """Regenerate warp maps with current slider values."""
        self._map_x, self._map_y = build_cone_maps(
            frame_size=FRAME_SIZE,
            canvas_size=CANVAS_SIZE,
            span_deg=int(self.span_var.get()),
            rotate_deg=self.rotate_var.get(),
            r_inner_frac=self.r_inner_var.get(),
            r_outer_frac=self.r_outer_var.get(),
            center_frac=(self.center_x_var.get(), self.center_y_var.get()),
            radius_frac=1.00,
        )

    def _reset_warp_params(self):
        """Reset all sliders to default values."""
        self.center_x_var.set(0.50)
        self.center_y_var.set(0.50)
        self.r_inner_var.set(0.08)
        self.r_outer_var.set(0.995)
        self.span_var.set(200)
        self.rotate_var.set(270)
        self._on_warp_change()

