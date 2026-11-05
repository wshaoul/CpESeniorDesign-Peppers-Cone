# calibration_view.py
#
# Front-audience cone calibration (Windows port of the Mac calibration studio).
# - Shows green/magenta Gray-code patterns fullscreen on the chosen display (the TV)
# - A camera at the audience position records how the cone bends each pattern
# - Decodes the reflection into a TV -> viewer distortion map (cone_calibration.py)
# - Compare grids with and without correction, then save the profile
# - Or: play the patterns with no camera attached, record them on a phone,
#   and decode the video afterwards ("record now, calibrate later")

import os
import time
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox, filedialog

import numpy as np
import cv2
from PIL import Image, ImageTk

from cone_calibration import (CalibrationSequence, build_profile, decode_sequence,
                              default_profile_path, verification_grid)
from studio_theme import Card, ScrollPanel

# Sized to fit the studio's preview card at the default 1280 px window width.
PREVIEW_W = 720
PREVIEW_H = 405

# Median of a few frames per pattern hides sensor noise and rolling-shutter tearing.
SAMPLES_PER_PATTERN = 3

BACKENDS = [
    ("MSMF (Windows 10/11)", cv2.CAP_MSMF),
    ("DSHOW (DirectShow)", cv2.CAP_DSHOW),
    ("ANY (let OpenCV pick)", cv2.CAP_ANY),
]


# ---------- Recorded calibration ----------
TAIL_SECONDS = 3           # black after the last pattern; marks the end of the sequence
RECORDING_MAX_WIDTH = 1920  # 4K phone video is downscaled to keep 24 stored frames small
RECORDING_SAMPLES = 5      # frames median-combined per pattern


def frame_signals(frame):
    """Whole-frame brightness and green-vs-magenta balance, cheap enough for every frame."""
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
    b, g, r = (small[:, :, c].mean() for c in range(3))
    return (b + g + r) / 3, g - (b + r) / 2


def change_peaks(times, changes, min_gap=0.3):
    """Moments where the picture changes sharply, as [(time, strength)].

    Each pattern switch is a large frame-to-frame change; stripes, solid colours
    and the final black all produce one, whatever the room lighting.
    """
    times = np.asarray(times, float)
    changes = np.asarray(changes, float)
    if changes.size < 5:
        return []
    threshold = max(0.35 * np.percentile(changes, 97), 4 * np.median(changes), 0.5)
    peaks = []
    for i in range(1, len(changes) - 1):
        if changes[i] > threshold and changes[i] >= changes[i - 1] and changes[i] >= changes[i + 1]:
            if peaks and times[i] - peaks[-1][0] < min_gap:
                if changes[i] > peaks[-1][1]:
                    peaks[-1] = (times[i], changes[i])
            else:
                peaks.append((times[i], changes[i]))
    return peaks


def follow_cadence(times, changes, start_hint):
    """Follow the evenly spaced pattern switches that begin at the green onset.

    Returns (start, end, switches, period): the black-to-green switch, the last
    switch in the chain, how many switches came after the first, and the median
    gap between them. None when no regular cadence can be found.
    """
    peaks = change_peaks(times, changes)
    if len(peaks) < 4:
        return None
    pt = np.array([p[0] for p in peaks])
    first = int(np.argmin(np.abs(pt - start_hint)))
    if abs(pt[first] - start_hint) > 0.4:
        return None
    gaps = np.diff(pt[first:first + 6])
    gaps = gaps[(gaps >= 0.3) & (gaps <= 6)]
    if len(gaps) < 3:
        return None
    period = float(np.median(gaps))

    chain, switches = [pt[first]], 0
    while True:
        for skipped in (1, 2):  # tolerate one weak switch the detector missed
            expected = chain[-1] + skipped * period
            near = np.nonzero(np.abs(pt - expected) <= 0.35 * period)[0]
            if near.size:
                chain.append(pt[max(near, key=lambda j: peaks[j][1])])
                switches += skipped
                break
        else:
            break
    return chain[0], chain[-1], switches, period


def find_pattern_run(times, levels, greens, count, changes=None):
    """Locate the played sequence in a recording.

    The green reference is the greenest moment in the video. When per-frame
    ``changes`` are supplied the pattern switches are followed from there, which
    works in a lit room and also counts the patterns actually played. Otherwise
    the sequence is taken to end when the frame drops back to black.
    Returns (start, period) in seconds.
    """
    times = np.asarray(times, float)
    levels = np.asarray(levels, float)
    greens = np.asarray(greens, float)
    peak = int(np.argmax(greens))
    if greens[peak] < 3:
        raise ValueError("No green reference frame found. Make sure the cone reflection is in the shot.")
    onset = peak
    while onset > 0 and greens[onset - 1] >= greens[peak] / 2:
        onset -= 1
    start = times[onset]

    before = levels[(times >= start - 1.0) & (times < start - 0.15)]
    if before.size == 0:
        raise ValueError("Start recording a few seconds before the TV turns green.")

    cadence = follow_cadence(times, changes, start) if changes is not None else None
    if cadence is not None:
        start, end, switches, period = cadence
        if switches == count - 1 and times[-1] - end < 1.5 * period:
            raise ValueError("The recording stops before the patterns finished. "
                             "Keep recording until the TV goes black again.")
        if switches != count:
            raise ValueError(
                f"The recording shows {switches} patterns but the selected display needs {count}. "
                "Select the display the patterns were played on. Windows display scaling "
                "changes the size the app sees, and with it the number of patterns.")
        period = (end - start) / count
        if not 0.3 <= period <= 6:
            raise ValueError(f"Found {period:.2f} s per pattern, which doesn't look like the played sequence. "
                             "Check that the selected display matches the TV the patterns were played on.")
        magenta = (times >= start + 1.25 * period) & (times <= start + 1.75 * period)
        if not magenta.any() or np.median(greens[magenta]) >= 0:
            raise ValueError("The pattern timing didn't line up (no magenta after green). Record again.")
        return start, period

    during = levels[(times >= start + 0.1) & (times < start + 0.3)]
    if during.size == 0:
        during = levels[onset:onset + 1]
    dark, bright = np.median(before), np.median(during)
    if bright - dark < 2:
        raise ValueError("The TV turning on and off is barely visible in the video. "
                         "Dim the room and lock the phone's exposure.")
    threshold = (dark + bright) / 2

    # Patterns never go black, so the first sustained dark stretch is the tail.
    end = None
    for k in range(onset + 1, len(times)):
        if levels[k] < threshold:
            window = (times >= times[k]) & (times < times[k] + 0.4)
            if np.all(levels[window] < threshold):
                end = times[k]
                break
    if end is None:
        raise ValueError("The recording stops before the patterns finished. "
                         "Keep recording until the TV goes black again.")
    period = (end - start) / count
    if not 0.3 <= period <= 6:
        raise ValueError(f"Found {period:.2f} s per pattern, which doesn't look like the played sequence. "
                         "Check that the selected display matches the TV the patterns were played on.")

    magenta = (times >= start + 1.25 * period) & (times <= start + 1.75 * period)
    if not magenta.any() or np.median(greens[magenta]) >= 0:
        raise ValueError("The pattern timing didn't line up (no magenta after green). Record again.")
    return start, period


def read_recording(path, count, report=lambda message, fraction: None):
    """Two passes over the video: find the pattern timing, then median-sample each pattern."""
    def open_video():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError("Could not open the video. On the iPhone, try Settings > Camera > "
                               "Formats > Most Compatible, then record again.")
        return cap

    cap = open_video()
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    times, levels, greens, changes = [], [], [], []
    previous = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            level, green = frame_signals(frame)
            levels.append(level)
            greens.append(green)
            small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
            changes.append(0.0 if previous is None else float(np.abs(small - previous).mean()))
            previous = small
            if len(times) % 30 == 0:
                report(f"Scanning video… {len(times)}/{total} frames", 0.7 * min(1, len(times) / total))
    finally:
        cap.release()
    if len(times) < 2:
        raise RuntimeError("The video has no readable frames.")
    times = np.asarray(times)
    if np.any(np.diff(times) <= 0):
        times = np.arange(len(times)) / fps  # container without usable timestamps

    start, period = find_pattern_run(times, levels, greens, count, changes)

    # Sample the middle half of each pattern, away from the switch-over.
    wanted = {}
    for i in range(count):
        a, b = start + (i + 0.25) * period, start + (i + 0.75) * period
        idx = np.nonzero((times >= a) & (times <= b))[0]
        if idx.size == 0:
            raise ValueError(f"No video frames landed on pattern {i + 1}. Record at 30 fps or "
                             "raise the seconds per pattern.")
        picks = idx[np.linspace(0, idx.size - 1, min(RECORDING_SAMPLES, idx.size)).round().astype(int)]
        for k in picks:
            wanted[int(k)] = i

    samples = [[] for _ in range(count)]
    last = max(wanted)
    cap = open_video()
    try:
        for k in range(last + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if k in wanted:
                h, w = frame.shape[:2]
                if w > RECORDING_MAX_WIDTH:
                    frame = cv2.resize(frame, (RECORDING_MAX_WIDTH, round(h * RECORDING_MAX_WIDTH / w)),
                                       interpolation=cv2.INTER_AREA)
                samples[wanted[k]].append(frame)
            if k % 30 == 0:
                report("Collecting pattern frames…", 0.7 + 0.3 * k / (last + 1))
    finally:
        cap.release()
    if any(not s for s in samples):
        raise RuntimeError("The video ended early while collecting pattern frames.")
    return [np.median(np.stack(s), axis=0).astype(np.uint8) for s in samples], period


class RecordingAnalysis:
    """Runs read_recording off the UI thread; the page polls message/fraction/result."""

    def __init__(self, path, count, screen_size):
        self.path = path
        self.count = count
        self.screen_size = screen_size
        self.message = "Opening video…"
        self.fraction = 0.0
        self.result = None
        self.error = ""
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            self.result = read_recording(self.path, self.count, self._report)
        except Exception as exc:
            self.error = str(exc) or exc.__class__.__name__

    def _report(self, message, fraction):
        self.message, self.fraction = message, fraction


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
        super().__init__(parent, style="Shell.TFrame")
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

        # Record-now-calibrate-later state
        self._playback = None       # (started, steps, sequence) while playing for a phone recording
        self._playback_id = None
        self._analysis = None
        self._recording = None      # {"frames": [...], "size": (w, h)} once a video is decoded

        # Preview state: ROI is normalized (x1, y1, x2, y2) in camera-frame coordinates
        self._roi = (0.2, 0.05, 0.8, 0.78)
        self._drag_start = None
        self._view = (0, 0, 1, 1)  # left, top, draw_w, draw_h of the frame on the canvas
        self._preview_img = None
        self._map_img = None

        self._pattern_win = PatternWindow(self, self._cancel_output)

        root = ttk.Frame(self, style="Shell.TFrame")
        root.pack(fill="both", expand=True)

        controls = ScrollPanel(root, width=420)
        controls.pack(side="left", fill="y", padx=(0, 18))
        left = controls.content

        preview_card = Card(root)
        preview_card.pack(side="left", fill="both", expand=True)
        right = preview_card.content

        # --- Title ---
        title_row = ttk.Frame(left)
        title_row.pack(fill="x")
        ttk.Button(title_row, text="← Back", command=self._go_back).pack(side="left")
        ttk.Label(title_row, text="Calibrate", style="Header.TLabel").pack(side="left", padx=8)
        ttk.Label(left, text="Project green/magenta patterns onto the TV and measure how the cone bends them for the front audience.",
                  style="Body.TLabel", wraplength=360).pack(anchor="w", pady=(8, 12))

        # --- Setup steps ---
        steps = ttk.LabelFrame(left, text="Setup")
        steps.pack(fill="x", pady=(0, 8))
        ttk.Label(steps, wraplength=340, justify="left", text=(
            "1. Connect the TV as an extended display (Win+P > Extend), not duplicated.\n"
            "2. Mount the camera at the audience's eye height and distance, facing the cone. "
            "Turn off auto-framing / background effects.\n"
            "3. Start the camera (or load a recording), then drag a box around the cone "
            "reflection only; exclude the visible TV below it.\n"
            "4. Dim the room lights and keep the camera and cone still during capture."
        )).pack(anchor="w")

        # --- Camera ---
        cam_box = ttk.LabelFrame(left, text="Camera")
        cam_box.pack(fill="x", pady=(0, 8))
        ttk.Label(cam_box, text="Index:").grid(row=0, column=0, padx=(8, 6), pady=8, sticky="w")
        self.cam_index = ttk.Spinbox(cam_box, from_=0, to=10, width=5, state="readonly")
        self.cam_index.set(0)
        self.cam_index.grid(row=0, column=1, sticky="w")
        ttk.Label(cam_box, text="Backend:").grid(row=0, column=2, padx=(16, 6), sticky="w")
        self.backend_combo = ttk.Combobox(cam_box, state="readonly", values=[label for (label, _) in BACKENDS], width=18)
        self.backend_combo.set(BACKENDS[0][0])
        self.backend_combo.grid(row=0, column=3, sticky="w", padx=(0, 8))

        # --- Pattern display ---
        disp_box = ttk.LabelFrame(left, text="Pattern Display")
        disp_box.pack(fill="x", pady=(0, 8))
        ttk.Label(disp_box, text="Display:").grid(row=0, column=0, padx=(8, 6), pady=8, sticky="w")
        self.display_combo = ttk.Combobox(disp_box, state="readonly", width=22)
        self.display_combo.grid(row=0, column=1, columnspan=2, sticky="w")
        self.display_combo.bind("<<ComboboxSelected>>", lambda e: self._update_sequence_label())
        self.btn_refresh = ttk.Button(disp_box, text="Refresh", command=self._refresh_monitors)
        self.btn_refresh.grid(row=1, column=2, padx=6, pady=(0, 8), sticky="e")
        ttk.Label(disp_box, text="Seconds per pattern:").grid(row=1, column=0, padx=(8, 6), pady=(0, 8), sticky="w")
        self.settle_var = tk.DoubleVar(value=1.5)
        self.settle_spin = ttk.Spinbox(disp_box, from_=0.5, to=5.0, increment=0.5, width=5,
                                       textvariable=self.settle_var, state="readonly")
        self.settle_spin.grid(row=1, column=1, sticky="w", pady=(0, 8))
        self.sequence_label = ttk.Label(disp_box, text="")
        self.sequence_label.grid(row=2, column=0, columnspan=3, padx=8, pady=(0, 8), sticky="w")

        # --- Actions ---
        actions = ttk.Frame(left)
        actions.pack(fill="x", pady=(4, 0))
        self.btn_camera = ttk.Button(actions, text="Start Camera", command=self._toggle_camera)
        self.btn_capture = ttk.Button(actions, text="Capture Patterns", style="Primary.TButton",
                                      command=self._start_calibration, state="disabled")
        self.btn_camera.pack(side="left")
        self.btn_capture.pack(side="left", padx=8)
        self.btn_cancel = ttk.Button(left, text="Cancel / Hide TV", style="TV.TButton", command=self._cancel_output)
        self.btn_cancel.pack(anchor="w", pady=(8, 0))

        # --- Record now, calibrate later ---
        rec_box = ttk.LabelFrame(left, text="Record now, calibrate later")
        rec_box.pack(fill="x", pady=(12, 0))
        ttk.Label(rec_box, wraplength=340, justify="left", text=(
            "No cable to the phone? Start recording with the phone's Camera app, click Play, "
            "and keep recording until the TV goes black. Then copy the video here and load it."
        )).grid(row=0, column=0, columnspan=3, padx=8, pady=(8, 6), sticky="w")
        ttk.Label(rec_box, text="Black lead-in (s):").grid(row=1, column=0, padx=(8, 6), pady=(0, 8), sticky="w")
        self.lead_var = tk.IntVar(value=10)
        self.lead_spin = ttk.Spinbox(rec_box, from_=3, to=30, increment=1, width=5,
                                     textvariable=self.lead_var, state="readonly")
        self.lead_spin.grid(row=1, column=1, sticky="w", pady=(0, 8))
        rec_actions = ttk.Frame(rec_box)
        rec_actions.grid(row=2, column=0, columnspan=3, padx=8, pady=(0, 8), sticky="w")
        self.btn_play = ttk.Button(rec_actions, text="Play for Recording", style="TV.TButton",
                                   command=self._play_for_recording)
        self.btn_load = ttk.Button(rec_actions, text="Load Recording…", command=self._load_recording)
        self.btn_build = ttk.Button(rec_actions, text="Build Map", style="Primary.TButton",
                                    command=self._build_from_recording, state="disabled")
        self.btn_play.pack(side="left")
        self.btn_load.pack(side="left", padx=6)
        self.btn_build.pack(side="left")

        # --- Verification ---
        verify = ttk.LabelFrame(left, text="Verify")
        verify.pack(fill="x", pady=(12, 0))
        self.btn_plain = ttk.Button(verify, text="Grid without correction", command=lambda: self._show_grid(False))
        self.btn_corrected = ttk.Button(verify, text="Grid with correction", command=lambda: self._show_grid(True))
        for btn in (self.btn_plain, self.btn_corrected):
            btn.config(state="disabled")
            btn.pack(fill="x", pady=(0, 6))

        # Where the calibration camera stood. Optional; saved with the profile as a note,
        # because the map is only exact from roughly this viewpoint.
        viewpoint = ttk.Frame(verify)
        viewpoint.pack(fill="x", pady=(2, 6))
        ttk.Label(viewpoint, text="Camera height (in):").grid(row=0, column=0, padx=(0, 6), pady=(0, 4), sticky="w")
        self.cam_height_var = tk.StringVar(value="")
        ttk.Entry(viewpoint, textvariable=self.cam_height_var, width=7).grid(row=0, column=1, pady=(0, 4), sticky="w")
        ttk.Label(viewpoint, text="Distance to cone (ft):").grid(row=1, column=0, padx=(0, 6), sticky="w")
        self.cam_distance_var = tk.StringVar(value="")
        ttk.Entry(viewpoint, textvariable=self.cam_distance_var, width=7).grid(row=1, column=1, sticky="w")
        ttk.Label(viewpoint, style="Muted.TLabel", wraplength=340, justify="left", text=(
            "Optional. Measure lens height from the floor and floor distance to the cone's centre. "
            "Saved with the profile so you know which seats it suits.")).grid(
            row=2, column=0, columnspan=2, pady=(4, 0), sticky="w")

        self.btn_save = ttk.Button(verify, text="Save profile", style="Primary.TButton", command=self._save_profile)
        self.btn_save_as = ttk.Button(verify, text="Save profile as…", command=self._save_profile_as)
        for btn in (self.btn_save, self.btn_save_as):
            btn.config(state="disabled")
            btn.pack(fill="x", pady=(0, 6))

        # --- Progress / Status ---
        self.progress = ttk.Progressbar(left, mode="determinate")
        self.progress.pack(fill="x", pady=(12, 4))
        self.status = tk.StringVar(value="Status: start the camera to set the cone area.")
        ttk.Label(left, textvariable=self.status, wraplength=360, justify="left",
                  style="Muted.TLabel").pack(anchor="w")

        # --- Right: camera preview with ROI box, and the measured map ---
        ttk.Label(right, text="Camera Preview", style="Header.TLabel").pack(anchor="w")
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(anchor="w", pady=(6, 4))

        self.canvas = tk.Canvas(self.tabs, width=PREVIEW_W, height=PREVIEW_H, bg="#0f172a", highlightthickness=0)
        self.canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2, fill="#9aa7bc",
                                text="Start the camera or load a recording, then drag a box around only the cone reflection.")
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

        ttk.Label(right, wraplength=PREVIEW_W, justify="left", style="Muted.TLabel", text=(
            "Map: red = horizontal source position, green = vertical; black = unsupported. "
            "Coverage is not an accuracy score."
        )).pack(anchor="w")

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
        if self._recording is not None:
            # Fall back to the loaded recording's still so its box can still be edited.
            self._latest_frame = self._recording["frames"][0]
            self._draw_preview(self._latest_frame)

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
        if not self._confirm_output():
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
            self._finish_calibration(self._captured_frames, (self._sequence.width, self._sequence.height))
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

    def _finish_calibration(self, frames, size):
        """Decode ROI-cropped pattern frames (live or recorded) into a profile."""
        self._pattern_win.hide()
        self.status.set("Status: building and smoothing the distortion map…")
        self.update_idletasks()
        try:
            screen_x, screen_y, valid = decode_sequence(frames, size)
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
        self._stop_playback()
        self._pattern_win.hide()
        self._pattern_index = -1
        self._captured_frames = []
        self._pattern_samples = []
        self._set_setup_enabled(True)
        self.btn_capture.config(state="normal" if self._camera is not None else "disabled")
        self.status.set(f"Status: {message}")

    def _cancel_output(self):
        if self._pattern_index >= 0 or self._playback is not None or self._pattern_win.is_visible():
            self._abort_calibration("TV output hidden. Any unfinished capture was cancelled; "
                                    "the saved profile was not changed.")

    def _confirm_output(self):
        return len(self._monitors) > 1 or messagebox.askokcancel(
            "Calibration",
            "Only one display is connected, so the patterns will cover this screen.\n\n"
            "Press Esc or Q to stop them early. Continue?")

    def _set_setup_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        self.btn_camera.config(state=state)
        self.btn_refresh.config(state=state)
        self.btn_play.config(state=state)
        self.btn_load.config(state=state)
        self.btn_build.config(state="normal" if enabled and self._recording is not None else "disabled")
        self.display_combo.config(state="readonly" if enabled else "disabled")
        self.settle_spin.config(state="readonly" if enabled else "disabled")
        self.lead_spin.config(state="readonly" if enabled else "disabled")
        if not enabled:
            self.btn_capture.config(state="disabled")

    def _set_verify_enabled(self, enabled):
        for btn in (self.btn_plain, self.btn_corrected, self.btn_save, self.btn_save_as):
            btn.config(state="normal" if enabled else "disabled")

    # ---------- Record now, calibrate later ----------
    def _lead_in_seconds(self):
        try:
            return min(30, max(3, int(self.lead_var.get())))
        except (tk.TclError, ValueError):
            return 10

    def _play_for_recording(self):
        """Play black, the full sequence on a fixed clock, then black, for a phone to record."""
        if self._pattern_index >= 0 or self._playback is not None or not self._confirm_output():
            return
        monitor = self._selected_monitor()
        _, _, _, w, h = monitor
        sequence = CalibrationSequence(w, h)
        lead_in, hold = self._lead_in_seconds(), self._settle_seconds()
        # (seconds after Play, pattern index); -1 = black, None = done
        steps = [(0, -1)] + [(lead_in + i * hold, i) for i in range(len(sequence))]
        steps += [(lead_in + len(sequence) * hold, -1), (lead_in + len(sequence) * hold + TAIL_SECONDS, None)]

        self._capture_monitor = monitor
        self._set_setup_enabled(False)
        self.progress.config(maximum=len(sequence), value=0)
        self._pattern_win.open_on(monitor)
        self._playback = (time.monotonic(), steps, sequence)
        self._play_step(0)

    def _play_step(self, index):
        self._playback_id = None
        if self._playback is None:
            return
        started, steps, sequence = self._playback
        pattern = steps[index][1]
        if pattern is None:
            self._abort_calibration(
                "Done. Stop the phone recording, copy the video to this PC, then click Load Recording… "
                "with this same display selected.")
            self.progress.config(value=0)
            return
        if pattern < 0:
            _, _, _, w, h = self._capture_monitor
            self._pattern_win.set_frame(np.zeros((h, w, 3), np.uint8))
            if index == 0:
                self.status.set(f"Status: TV is black for {self._lead_in_seconds()} s. Make sure the phone "
                                "is recording; patterns start after that.")
            else:
                self.status.set("Status: last pattern shown. Keep recording until this finishes.")
        else:
            self._pattern_win.set_frame(sequence.frame(pattern))
            self.progress.config(value=pattern + 1)
            self.status.set(f"Status: playing {pattern + 1}/{len(sequence)}: {sequence.label(pattern)}. "
                            "Keep the phone still and recording.")
        # Schedule against the start time so per-step delays don't accumulate.
        delay = started + steps[index + 1][0] - time.monotonic()
        self._playback_id = self.after(max(1, int(delay * 1000)), lambda: self._play_step(index + 1))

    def _stop_playback(self):
        if self._playback_id is not None:
            self.after_cancel(self._playback_id)
            self._playback_id = None
        self._playback = None

    def _load_recording(self):
        path = filedialog.askopenfilename(
            title="Choose the calibration recording",
            filetypes=[("Videos", "*.mov *.mp4 *.m4v *.avi"), ("All files", "*.*")])
        if not path:
            return
        self._stop_camera()
        _, _, _, w, h = self._selected_monitor()
        self._analysis = RecordingAnalysis(path, len(CalibrationSequence(w, h)), (w, h))
        self._analysis.thread.start()
        self._set_setup_enabled(False)
        self.progress.config(maximum=100, value=0)
        self._poll_analysis()

    def _poll_analysis(self):
        analysis = self._analysis
        if analysis is None:
            return
        if analysis.thread.is_alive():
            self.progress.config(value=analysis.fraction * 100)
            self.status.set(f"Status: {analysis.message}")
            self.after(100, self._poll_analysis)
            return
        self._analysis = None
        self.progress.config(value=0)
        if analysis.error:
            self._set_setup_enabled(True)
            self.status.set(f"Status: {analysis.error}")
            return

        frames, period = analysis.result
        w, h = analysis.screen_size
        self._recording = {"frames": frames, "size": (w, h)}
        self._profile = None
        self.map_label.config(image="", text="Build the map to see the measured distortion map.")
        self._set_verify_enabled(False)
        self._set_setup_enabled(True)
        self._latest_frame = frames[0]  # green reference: the reflection shows up clearly
        self._draw_preview(self._latest_frame)
        self.tabs.select(0)
        # The recording's length divided by the expected pattern count gives the period. If the
        # wrong display is selected the count is off, the period comes out wrong, and every
        # later pattern is sampled from the neighbouring one without any other error.
        settle = self._settle_seconds()
        mismatch = ""
        if abs(period - settle) > 0.025 * settle:
            mismatch = (f" WARNING: that is {period:.2f} s per pattern but 'Seconds per pattern' is "
                        f"{settle:.1f} s. Either that setting changed since you played the patterns, "
                        "or the selected display isn't the one they were played on (it sets how many "
                        "patterns to expect). Fix it and load the recording again.")
        self.status.set(f"Status: found all {len(frames)} patterns ({period:.2f} s each, decoding for "
                        f"{w} x {h}). The green frame shows the reflection; drag the box around the "
                        f"cone reflection only, then click Build Map.{mismatch}")

    def _build_from_recording(self):
        if self._recording is None:
            return
        frames = self._recording["frames"]
        x1, y1, x2, y2 = self._roi_pixels(frames[0].shape)
        if x2 - x1 < 40 or y2 - y1 < 40:
            self.status.set("Status: draw a larger box around the cone reflection.")
            return
        self._capture_roi = (x1, y1, x2, y2)
        self._capture_monitor = self._selected_monitor()
        self._finish_calibration([f[y1:y2, x1:x2] for f in frames], self._recording["size"])

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

    def _read_viewpoint(self):
        """(height_in, distance_in) from the optional fields; None for a blank field.

        Raises ValueError with a message for the status line when a field isn't a positive number.
        """
        def parse(var, name, scale):
            text = var.get().strip()
            if not text:
                return None
            try:
                value = float(text)
            except ValueError:
                raise ValueError(f"{name} must be a number (or left blank).") from None
            if not (value > 0 and np.isfinite(value)):
                raise ValueError(f"{name} must be greater than zero (or left blank).")
            return value * scale
        return (parse(self.cam_height_var, "Camera height", 1.0),
                parse(self.cam_distance_var, "Distance to cone", 12.0))

    def _save_profile_as(self):
        if self._profile is None:
            return
        default = default_profile_path()
        default.parent.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save distortion map as",
            initialdir=str(default.parent),
            initialfile=time.strftime("cone_calibration_%Y-%m-%d_%H%M.npz"),
            defaultextension=".npz",
            filetypes=[("Calibration profile", "*.npz")])
        if path:
            self._save_profile(Path(path))

    def _save_profile(self, path=None):
        if self._profile is None:
            return
        try:
            self._profile.camera_height_in, self._profile.camera_distance_in = self._read_viewpoint()
        except ValueError as exc:
            self.status.set(f"Status: not saved. {exc}")
            return
        is_default = path is None
        path = path or default_profile_path()
        try:
            self._profile.save(path)
        except (OSError, ValueError) as exc:
            self.status.set(f"Status: could not save calibration: {exc}")
            return
        self._pattern_win.hide()
        where = self._profile.viewpoint_text()
        where = f" Measured from {where}." if where else ""
        note = ("" if is_default else
                " This is a copy; the Live tab uses the profile saved with 'Save profile'.")
        self.status.set(f"Status: profile saved to {path}.{where}{note} "
                        "Moving the cone, TV or viewpoint requires recalibration.")

    def on_hide(self):
        """Called by the studio when another tab is selected; frees the camera for Live/Record."""
        self._stop_camera()

    def _go_back(self):
        self.controller.show_page("HomePage")
