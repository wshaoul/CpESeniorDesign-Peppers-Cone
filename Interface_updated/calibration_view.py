# calibration_view.py
#
# Front-audience cone calibration (Windows port of the Mac calibration studio).
# - Shows green/magenta Gray-code patterns fullscreen on the chosen display (the TV)
# - A camera at the audience position records how the cone bends each pattern
# - Decodes the reflection into a TV -> viewer distortion map (cone_calibration.py)
# - Compare grids with and without correction, then save the profile

import os
import time
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk

from cone_calibration import CalibrationSequence, build_profile, decode_sequence, verification_grid

PREVIEW_W = 800
PREVIEW_H = 450

# Median of a few frames per pattern hides sensor noise and rolling-shutter tearing.
SAMPLES_PER_PATTERN = 3

BACKENDS = [
    ("MSMF (Windows 10/11)", cv2.CAP_MSMF),
    ("DSHOW (DirectShow)", cv2.CAP_DSHOW),
    ("ANY (let OpenCV pick)", cv2.CAP_ANY),
]


def default_profile_path():
    root = Path(os.environ.get("APPDATA", Path.home())) / "PeppersCone"
    return root / "front_audience_calibration.npz"


# ---------- Display enumeration ----------
def list_monitors(root):
    """Return [(label, x, y, w, h)] for every display, primary first."""
    found = []
    try:
        import ctypes
        from ctypes import wintypes

        class MONITORINFOEXW(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
                        ("szDevice", wintypes.WCHAR * 32)]

        # Private handle so argtypes don't leak into other ctypes users of user32.
        user32 = ctypes.WinDLL("user32")
        user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.c_void_p]
        enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                                       ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)

        def on_monitor(hmon, _hdc, _rect, _lparam):
            info = MONITORINFOEXW()
            info.cbSize = ctypes.sizeof(info)
            if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
                r = info.rcMonitor
                found.append((bool(info.dwFlags & 1), r.left, r.top, r.right - r.left, r.bottom - r.top))
            return True

        user32.EnumDisplayMonitors(None, None, enum_proc(on_monitor), 0)
    except Exception:
        found = []

    if not found:
        found = [(True, 0, 0, root.winfo_screenwidth(), root.winfo_screenheight())]

    found.sort(key=lambda m: not m[0])
    monitors = []
    for i, (primary, x, y, w, h) in enumerate(found):
        label = f"Display {i + 1}: {w}x{h}" + (" (primary)" if primary else "")
        monitors.append((label, x, y, w, h))
    return monitors


# ---------- Camera ----------
class CalibrationCamera:
    """Continuously drain the camera so pattern captures don't use queued frames."""

    def __init__(self, index, backend):
        self.index = index
        self.backend = backend
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = ""
        self.thread = threading.Thread(target=self.run, daemon=True)

    def _open(self):
        backends = [self.backend] + [api for (_, api) in BACKENDS if api != self.backend]
        for api in backends:
            cap = cv2.VideoCapture(self.index, api)
            if cap.isOpened():
                return cap
            cap.release()
        return None

    def run(self):
        capture = None
        try:
            capture = self._open()
            if capture is None:
                raise RuntimeError("Camera unavailable. Stop, choose another camera index, and try again.")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            capture.set(cv2.CAP_PROP_FPS, 30)
            while not self.done.is_set():
                started = time.monotonic()
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera disconnected or stopped returning frames.")
                with self.lock:
                    self.latest = (started, frame)
        except Exception as exc:
            self.error = str(exc)
        finally:
            if capture is not None:
                capture.release()

    def snapshot(self):
        with self.lock:
            return self.latest

    def stop(self):
        # Only the worker releases the capture, so the UI never blocks on a stuck read.
        self.done.set()


# ---------- Fullscreen pattern output ----------
class PatternWindow:
    """Borderless window that covers one display and shows a single BGR frame."""

    def __init__(self, master, on_dismiss):
        self.master = master
        self.on_dismiss = on_dismiss
        self.win = None
        self.label = None
        self.image = None
        self.size = (0, 0)

    def open_on(self, monitor):
        _, x, y, w, h = monitor
        if self.win is None or not self.win.winfo_exists():
            self.win = tk.Toplevel(self.master)
            self.win.title("Cone calibration pattern")
            self.win.overrideredirect(True)
            self.win.configure(bg="black", cursor="none")
            self.win.attributes("-topmost", True)
            self.label = tk.Label(self.win, bg="black", bd=0, highlightthickness=0)
            self.label.pack(fill="both", expand=True)
            for key in ("<Escape>", "q", "Q"):
                self.win.bind(key, lambda e: self.on_dismiss())
        self.size = (w, h)
        self.win.geometry(f"{w}x{h}+{x}+{y}")
        self.win.deiconify()
        self.win.lift()
        self.win.focus_force()

    def set_frame(self, frame):
        if self.win is None:
            return
        w, h = self.size
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.label.config(image=self.image)
        self.win.update_idletasks()

    def is_visible(self):
        return self.win is not None and self.win.winfo_exists() and self.win.winfo_viewable()

    def hide(self):
        if self.win is not None and self.win.winfo_exists():
            self.win.withdraw()


# ---------- Page ----------
class CalibrationView(ttk.Frame):
    """Capture a structured-light distortion map of the cone for one audience viewpoint."""

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        # Capture state
        self._camera = None
        self._pending_cameras = []
        self._latest_frame = None
        self._last_frame_time = 0
        self._tick_id = None
        self._sequence = None
        self._pattern_index = -1
        self._pattern_started = 0
        self._pattern_samples = []
        self._captured_frames = []
        self._capture_roi = None
        self._capture_shape = None
        self._capture_monitor = None
        self._profile = None

        # Preview state: ROI is normalized (x1, y1, x2, y2) in camera-frame coordinates
        self._roi = (0.2, 0.05, 0.8, 0.78)
        self._drag_start = None
        self._view = (0, 0, 1, 1)  # left, top, draw_w, draw_h of the frame on the canvas
        self._preview_img = None
        self._map_img = None

        self._pattern_win = PatternWindow(self, self._cancel_output)

        # --- Title ---
        ttk.Label(self, text="Cone Calibration", style="Header.TLabel").pack(pady=(10, 4))
        ttk.Label(self, text="Project green/magenta patterns onto the TV and measure how the cone bends them for the front audience.",
                  style="Body.TLabel").pack(pady=(0, 10))

        top = ttk.Frame(self)
        top.pack(fill="both", expand=True)
        left = ttk.Frame(top)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right = ttk.Frame(top)
        right.pack(side="right", fill="both", expand=True, padx=(8, 0))

        # --- Setup steps ---
        steps = ttk.LabelFrame(left, text="Setup")
        steps.pack(fill="x", padx=4, pady=(0, 10))
        ttk.Label(steps, wraplength=440, justify="left", text=(
            "1. Connect the TV as an extended display (Win+P > Extend), not duplicated.\n"
            "2. Mount the camera at the audience's eye height and distance, facing the cone. "
            "Turn off auto-framing / background effects.\n"
            "3. Start the camera, then drag a box around the cone reflection only; "
            "exclude the visible TV below it.\n"
            "4. Dim the room lights and keep the camera and cone still during capture."
        )).pack(anchor="w", padx=8, pady=6)

        # --- Camera ---
        cam_box = ttk.LabelFrame(left, text="Camera")
        cam_box.pack(fill="x", padx=4, pady=(0, 10))
        ttk.Label(cam_box, text="Index:").grid(row=0, column=0, padx=(8, 6), pady=8, sticky="w")
        self.cam_index = ttk.Spinbox(cam_box, from_=0, to=10, width=5, state="readonly")
        self.cam_index.set(0)
        self.cam_index.grid(row=0, column=1, sticky="w")
        ttk.Label(cam_box, text="Backend:").grid(row=0, column=2, padx=(16, 6), sticky="w")
        self.backend_combo = ttk.Combobox(cam_box, state="readonly", values=[label for (label, _) in BACKENDS], width=22)
        self.backend_combo.set(BACKENDS[0][0])
        self.backend_combo.grid(row=0, column=3, sticky="w", padx=(0, 8))

        # --- Pattern display ---
        disp_box = ttk.LabelFrame(left, text="Pattern Display")
        disp_box.pack(fill="x", padx=4, pady=(0, 10))
        ttk.Label(disp_box, text="Display:").grid(row=0, column=0, padx=(8, 6), pady=8, sticky="w")
        self.display_combo = ttk.Combobox(disp_box, state="readonly", width=30)
        self.display_combo.grid(row=0, column=1, sticky="w")
        self.display_combo.bind("<<ComboboxSelected>>", lambda e: self._update_sequence_label())
        self.btn_refresh = ttk.Button(disp_box, text="Refresh", command=self._refresh_monitors)
        self.btn_refresh.grid(row=0, column=2, padx=6)
        ttk.Label(disp_box, text="Seconds per pattern:").grid(row=1, column=0, padx=(8, 6), pady=(0, 8), sticky="w")
        self.settle_var = tk.DoubleVar(value=1.5)
        self.settle_spin = ttk.Spinbox(disp_box, from_=0.5, to=5.0, increment=0.5, width=5,
                                       textvariable=self.settle_var, state="readonly")
        self.settle_spin.grid(row=1, column=1, sticky="w", pady=(0, 8))
        self.sequence_label = ttk.Label(disp_box, text="")
        self.sequence_label.grid(row=2, column=0, columnspan=3, padx=8, pady=(0, 8), sticky="w")

        # --- Actions ---
        actions = ttk.Frame(left)
        actions.pack(pady=(6, 2))
        self.btn_camera = ttk.Button(actions, text="Start Camera", command=self._toggle_camera)
        self.btn_capture = ttk.Button(actions, text="Capture Patterns", command=self._start_calibration, state="disabled")
        self.btn_cancel = ttk.Button(actions, text="Cancel / Hide TV", command=self._cancel_output)
        back_btn = ttk.Button(actions, text="Back", command=self._go_back)
        self.btn_camera.grid(row=0, column=0, padx=6)
        self.btn_capture.grid(row=0, column=1, padx=6)
        self.btn_cancel.grid(row=0, column=2, padx=6)
        back_btn.grid(row=0, column=3, padx=6)

        # --- Verification ---
        verify = ttk.LabelFrame(left, text="Verify")
        verify.pack(fill="x", padx=4, pady=(10, 0))
        self.btn_plain = ttk.Button(verify, text="Grid without correction", command=lambda: self._show_grid(False))
        self.btn_corrected = ttk.Button(verify, text="Grid with correction", command=lambda: self._show_grid(True))
        self.btn_save = ttk.Button(verify, text="Save profile", command=self._save_profile)
        for i, btn in enumerate((self.btn_plain, self.btn_corrected, self.btn_save)):
            btn.config(state="disabled")
            btn.grid(row=0, column=i, padx=6, pady=8)

        # --- Progress / Status ---
        self.progress = ttk.Progressbar(left, mode="determinate")
        self.progress.pack(fill="x", padx=4, pady=(12, 4))
        self.status = tk.StringVar(value="Status: start the camera to set the cone area.")
        ttk.Label(left, textvariable=self.status, wraplength=440, justify="left").pack(anchor="w", padx=4)

        # --- Right: camera preview with ROI box, and the measured map ---
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(padx=4, pady=4)

        self.canvas = tk.Canvas(self.tabs, width=PREVIEW_W, height=PREVIEW_H, bg="black", highlightthickness=0)
        self.canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2, fill="#9aa7bc",
                                text="Start the camera, then drag a box around only the cone reflection.")
        self.canvas.bind("<ButtonPress-1>", self._on_roi_press)
        self.canvas.bind("<B1-Motion>", self._on_roi_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_roi_release)
        self.tabs.add(self.canvas, text="Camera / reflection")

        map_frame = tk.Frame(self.tabs, width=PREVIEW_W, height=PREVIEW_H, bg="black")
        map_frame.pack_propagate(False)
        self.map_label = tk.Label(map_frame, bg="black", fg="#9aa7bc",
                                  text="Capture patterns to see the measured distortion map.")
        self.map_label.pack(fill="both", expand=True)
        self.tabs.add(map_frame, text="Distortion map")

        ttk.Label(right, wraplength=PREVIEW_W, justify="left", text=(
            "Map: red = horizontal source position, green = vertical; black = unsupported. "
            "Coverage is not an accuracy score."
        )).pack(anchor="w", padx=4)

        self._refresh_monitors()

    # ---------- Displays ----------
    def _refresh_monitors(self):
        self._monitors = list_monitors(self)
        self.display_combo["values"] = [m[0] for m in self._monitors]
        # Default to the first non-primary display, which is normally the TV.
        self.display_combo.current(1 if len(self._monitors) > 1 else 0)
        self._update_sequence_label()

    def _selected_monitor(self):
        index = self.display_combo.current()
        return self._monitors[max(0, index)]

    def _update_sequence_label(self):
        _, _, _, w, h = self._selected_monitor()
        count = len(CalibrationSequence(w, h))
        self.sequence_label.config(text=f"Calibration sequence: {w} x {h}  ·  {count} patterns")
        self.btn_capture.config(text=f"Capture {count} Patterns")

    def _settle_seconds(self):
        try:
            return min(5.0, max(0.5, float(self.settle_var.get())))
        except (tk.TclError, ValueError):
            return 1.5

    # ---------- Camera ----------
    def _toggle_camera(self):
        if self._camera is not None:
            self._stop_camera()
            return
        self._pending_cameras = [c for c in self._pending_cameras if c.thread.is_alive()]
        if self._pending_cameras:
            self.status.set("Status: waiting for the previous camera to close. Try again in a moment.")
            return
        backend = dict(BACKENDS).get(self.backend_combo.get(), cv2.CAP_ANY)
        self._camera = CalibrationCamera(int(self.cam_index.get()), backend)
        self._camera.thread.start()
        self._profile = None
        self.map_label.config(image="", text="Capture patterns to see the measured distortion map.")
        self._set_verify_enabled(False)
        self.btn_camera.config(text="Stop Camera")
        self.cam_index.config(state="disabled")
        self.backend_combo.config(state="disabled")
        self.btn_capture.config(state="disabled")
        self._latest_frame = None
        self._last_frame_time = 0
        self.status.set("Status: opening camera… If it fails, try another index or backend.")
        self._tick_id = self.after(40, self._tick)

    def _stop_camera(self):
        self._cancel_output()
        if self._tick_id is not None:
            self.after_cancel(self._tick_id)
            self._tick_id = None
        if self._camera is not None:
            self._camera.stop()
            self._pending_cameras.append(self._camera)
            self._camera = None
        self._latest_frame = None
        self.btn_camera.config(text="Start Camera", state="normal")
        self.cam_index.config(state="readonly")
        self.backend_combo.config(state="readonly")
        self.btn_capture.config(state="disabled")

    def _tick(self):
        self._tick_id = None
        if self._camera is None:
            return
        if self._camera.error:
            message = self._camera.error
            self._stop_camera()
            self.status.set(f"Status: {message}")
            return

        now = time.monotonic()
        if self._pattern_index >= 0 and now - self._pattern_started > self._settle_seconds() + 8:
            self._abort_calibration("No fresh camera frames arrived for this pattern. Check the camera and try again.")

        sample = self._camera.snapshot()
        if sample is not None and sample[0] > self._last_frame_time:
            timestamp, frame = sample
            if self._last_frame_time == 0:
                self.status.set(f"Status: camera ready ({frame.shape[1]} x {frame.shape[0]}). "
                                "Drag a box around the cone reflection, then capture.")
            self._last_frame_time = timestamp
            self._latest_frame = frame
            self._draw_preview(frame)

            if self._pattern_index < 0:
                self.btn_capture.config(state="normal")
            elif frame.shape != self._capture_shape:
                self._abort_calibration("Camera resolution changed. Keep the camera fixed, then capture again.")
            elif timestamp >= self._pattern_started + self._settle_seconds():
                x1, y1, x2, y2 = self._capture_roi
                self._pattern_samples.append(frame[y1:y2, x1:x2].copy())
                if len(self._pattern_samples) == SAMPLES_PER_PATTERN:
                    self._capture_pattern()

        if self._camera is not None:
            self._tick_id = self.after(40, self._tick)

    # ---------- Preview + ROI ----------
    def _roi_pixels(self, shape):
        height, width = shape[:2]
        x1, y1, x2, y2 = self._roi
        return (max(0, min(width - 1, round(x1 * width))),
                max(0, min(height - 1, round(y1 * height))),
                max(1, min(width, round(x2 * width))),
                max(1, min(height, round(y2 * height))))

    def _draw_preview(self, frame):
        fh, fw = frame.shape[:2]
        scale = min(PREVIEW_W / fw, PREVIEW_H / fh)
        draw_w, draw_h = max(1, int(fw * scale)), max(1, int(fh * scale))
        left, top = (PREVIEW_W - draw_w) // 2, (PREVIEW_H - draw_h) // 2
        self._view = (left, top, draw_w, draw_h)

        resized = cv2.resize(frame, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._preview_img = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.canvas.delete("all")
        self.canvas.create_image(left, top, anchor="nw", image=self._preview_img)
        x1, y1, x2, y2 = self._roi
        rx1, ry1 = left + x1 * draw_w, top + y1 * draw_h
        rx2, ry2 = left + x2 * draw_w, top + y2 * draw_h
        self.canvas.create_rectangle(rx1, ry1, rx2, ry2, outline="#ffdc28", width=2)
        self.canvas.create_text(rx1 + 6, ry1 + 6, anchor="nw", fill="#ffdc28",
                                font=("Segoe UI", 10, "bold"), text="CONE REFLECTION ONLY")

    def _canvas_to_norm(self, event):
        left, top, draw_w, draw_h = self._view
        x = float(np.clip((event.x - left) / max(1, draw_w), 0, 1))
        y = float(np.clip((event.y - top) / max(1, draw_h), 0, 1))
        return x, y

    def _roi_editable(self):
        return self._latest_frame is not None and self._pattern_index < 0

    def _on_roi_press(self, event):
        if self._roi_editable():
            self._drag_start = self._canvas_to_norm(event)

    def _on_roi_drag(self, event):
        if self._drag_start is None:
            return
        left, top, draw_w, draw_h = self._view
        sx, sy = self._drag_start
        self.canvas.delete("drag")
        self.canvas.create_rectangle(left + sx * draw_w, top + sy * draw_h, event.x, event.y,
                                     outline="#ffffff", dash=(4, 3), tags="drag")

    def _on_roi_release(self, event):
        if self._drag_start is None:
            return
        end = self._canvas_to_norm(event)
        x1, x2 = sorted((self._drag_start[0], end[0]))
        y1, y2 = sorted((self._drag_start[1], end[1]))
        if x2 - x1 > 0.04 and y2 - y1 > 0.04:
            self._roi = (x1, y1, x2, y2)
        self._drag_start = None
        if self._latest_frame is not None:
            self._draw_preview(self._latest_frame)

    # ---------- Calibration sequence ----------
    def _start_calibration(self):
        if self._camera is None or self._latest_frame is None:
            self.status.set("Status: start the camera and wait for a preview first.")
            return
        if time.monotonic() - self._last_frame_time > 1:
            self.status.set("Status: wait for a fresh camera preview before starting.")
            return
        if len(self._monitors) < 2 and not messagebox.askokcancel(
                "Calibration",
                "Only one display is connected, so the patterns will cover this screen.\n\n"
                "Press Esc or Q to stop them early. Continue?"):
            return
        x1, y1, x2, y2 = self._roi_pixels(self._latest_frame.shape)
        if x2 - x1 < 40 or y2 - y1 < 40:
            self.status.set("Status: draw a larger box around the cone reflection.")
            return

        monitor = self._selected_monitor()
        _, _, _, w, h = monitor
        self._sequence = CalibrationSequence(w, h)
        self._capture_roi = (x1, y1, x2, y2)
        self._capture_shape = self._latest_frame.shape
        self._capture_monitor = monitor
        self._profile = None
        self.map_label.config(image="", text="Capturing a new map…")
        self._set_verify_enabled(False)
        self._set_setup_enabled(False)

        self._pattern_index = 0
        self._captured_frames = []
        self.progress.config(maximum=len(self._sequence), value=0)
        self._pattern_win.open_on(monitor)
        self._show_pattern()

    def _show_pattern(self):
        if self._pattern_index < 0:
            return
        if self._pattern_index >= len(self._sequence):
            self._finish_calibration()
            return
        self._pattern_win.set_frame(self._sequence.frame(self._pattern_index))
        self._pattern_started = time.monotonic()
        self._pattern_samples = []
        self.status.set(f"Status: showing {self._pattern_index + 1}/{len(self._sequence)}: "
                        f"{self._sequence.label(self._pattern_index)}. Keep the camera and cone still.")

    def _capture_pattern(self):
        if self._pattern_index < 0 or len(self._pattern_samples) != SAMPLES_PER_PATTERN:
            return
        self._captured_frames.append(np.median(self._pattern_samples, axis=0).astype(np.uint8))
        self._pattern_index += 1
        self.progress.config(value=self._pattern_index)
        self._show_pattern()

    def _finish_calibration(self):
        self._pattern_win.hide()
        self.status.set("Status: building and smoothing the distortion map…")
        self.update_idletasks()
        size = (self._sequence.width, self._sequence.height)
        try:
            screen_x, screen_y, valid = decode_sequence(self._captured_frames, size)
            profile = build_profile(screen_x, screen_y, valid, size)
            if profile.coverage < 0.01:
                raise ValueError("Too little of the TV reflection was decoded. Tighten the box around "
                                 "the cone and reduce room light.")
        except Exception as exc:
            self._abort_calibration(str(exc))
            return

        self._profile = profile
        self._abort_calibration(
            f"Map built from {profile.sample_count:,} samples ({profile.coverage:.1%} TV coverage). "
            "Compare both grids through the camera: look for straight lines, square cells and a round "
            "circle. Not saved yet.")

        vis = profile.visualization()
        vh, vw = vis.shape[:2]
        scale = min(PREVIEW_W / vw, PREVIEW_H / vh)
        vis = cv2.resize(vis, (max(1, int(vw * scale)), max(1, int(vh * scale))), interpolation=cv2.INTER_AREA)
        self._map_img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)))
        self.map_label.config(image=self._map_img, text="")
        self.tabs.select(1)
        self._set_verify_enabled(True)

    def _abort_calibration(self, message):
        self._pattern_win.hide()
        self._pattern_index = -1
        self._captured_frames = []
        self._pattern_samples = []
        self._set_setup_enabled(True)
        self.btn_capture.config(state="normal" if self._camera is not None else "disabled")
        self.status.set(f"Status: {message}")

    def _cancel_output(self):
        if self._pattern_index >= 0 or self._pattern_win.is_visible():
            self._abort_calibration("TV output hidden. Any unfinished capture was cancelled; "
                                    "the saved profile was not changed.")

    def _set_setup_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.btn_camera.config(state=state)
        self.btn_refresh.config(state=state)
        self.display_combo.config(state="readonly" if enabled else "disabled")
        self.settle_spin.config(state="readonly" if enabled else "disabled")
        if not enabled:
            self.btn_capture.config(state="disabled")

    def _set_verify_enabled(self, enabled):
        for btn in (self.btn_plain, self.btn_corrected, self.btn_save):
            btn.config(state="normal" if enabled else "disabled")

    # ---------- Verify / save ----------
    def _show_grid(self, corrected):
        if self._profile is None:
            return
        width, height = self._profile.screen_size
        # Use the capture's aspect ratio so square cells stay square in the
        # calibrated camera plane, even when the selected reflection is tall.
        x1, y1, x2, y2 = self._capture_roi
        target = verification_grid(x2 - x1, y2 - y1)
        output = self._profile.apply(target) if corrected else cv2.resize(target, (width, height))
        self._pattern_win.open_on(self._capture_monitor)
        self._pattern_win.set_frame(output)
        self.tabs.select(0)
        self.status.set(("Status: corrected" if corrected else "Status: uncorrected") +
                        " grid on TV. Compare the reflection in the camera view. Esc/Q or Cancel hides it.")

    def _save_profile(self):
        if self._profile is None:
            return
        path = default_profile_path()
        try:
            self._profile.save(path)
        except (OSError, ValueError) as exc:
            self.status.set(f"Status: could not save calibration: {exc}")
            return
        self._pattern_win.hide()
        self.status.set(f"Status: profile saved to {path}. Moving the cone, TV or viewpoint requires recalibration.")

    def _go_back(self):
        self._stop_camera()
        self.controller.show_page("HomePage")
