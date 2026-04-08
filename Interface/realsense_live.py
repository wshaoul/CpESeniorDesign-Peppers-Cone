# realsense_live.py
#
# Standalone RealSense viewer for Pepper's Cone — pre-alpha milestone.
# Reuses build_cone_maps and enhance_saturation_contrast from live_view.py.
# Does NOT modify live_view.py in any way.
#
# Run:  python Interface/realsense_live.py
# Quit: press 'q' in any OpenCV window, or close the tkinter window.

import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk

# ── Reuse warp helpers from live_view — no code duplication ─────────────────
from live_view import build_cone_maps, enhance_saturation_contrast

# ── RealSense (optional — clear error if missing) ────────────────────────────
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

# ── Layout constants (match live_view.py defaults) ───────────────────────────
FRAME_SIZE  = 400
CANVAS_SIZE = 800
PREVIEW_W   = 640
PREVIEW_H   = 480

# RealSense stream resolution
RS_W, RS_H, RS_FPS = 640, 480, 30

# Depth colormap (change to cv2.COLORMAP_JET if preferred)
DEPTH_COLORMAP = cv2.COLORMAP_TURBO


# ── RealSense helpers ─────────────────────────────────────────────────────────

def rs_start():
    """Start the RealSense pipeline. Returns (pipeline, align) or raises."""
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
    cfg.enable_stream(rs.stream.depth, RS_W, RS_H, rs.format.z16,  RS_FPS)
    profile = pipeline.start(cfg)
    align   = rs.align(rs.stream.color)

    dev = profile.get_device()
    print(f"[RealSense] {dev.get_info(rs.camera_info.name)}  "
          f"s/n {dev.get_info(rs.camera_info.serial_number)}")
    return pipeline, align


def rs_get_frame(pipeline, align):
    """
    Grab one aligned frameset.
    Returns (color_bgr, depth_raw_mm, depth_colorized) or (None, None, None).
    """
    frames  = pipeline.wait_for_frames()
    aligned = align.process(frames)

    cf = aligned.get_color_frame()
    df = aligned.get_depth_frame()
    if not cf or not df:
        return None, None, None

    color_bgr = np.asanyarray(cf.get_data())          # (H,W,3) uint8
    depth_mm  = np.asanyarray(df.get_data())           # (H,W)   uint16

    # Normalise 0–4 m to 0–255 for display
    depth_vis = cv2.applyColorMap(
        (np.clip(depth_mm, 0, 4000).astype(np.float32) / 4000.0 * 255).astype(np.uint8),
        DEPTH_COLORMAP,
    )
    return color_bgr, depth_mm, depth_vis


# ── Main View ─────────────────────────────────────────────────────────────────

class RealSenseLiveView(ttk.Frame):
    """
    Self-contained tkinter page with:
      - RealSense connect / disconnect controls
      - Frame-source toggle: Color | Depth | Side-by-side
      - Warp tuning sliders (same as live_view.py)
      - In-app preview (raw, unwarped)
      - Fullscreen cone output (warped + enhanced)
    """

    def __init__(self, parent, controller=None):
        super().__init__(parent)
        self.controller = controller

        # ── shared state ─────────────────────────────────────────────────────
        self._pipeline   = None
        self._align      = None
        self._last_bgr   = None          # latest frame for display (any source)
        self._last_depth = None          # raw depth uint16 for debug
        self._frame_lock = threading.Lock()
        self._cap_thread  = None
        self._stop_evt    = threading.Event()

        self._fs_win     = None
        self._fs_label   = None
        self._fs_running = False

        self._fps        = 0.0

        # ── warp maps (rebuilt by sliders) ───────────────────────────────────
        self._map_x, self._map_y = build_cone_maps(
            frame_size=FRAME_SIZE, canvas_size=CANVAS_SIZE,
            span_deg=200, rotate_deg=270,
            r_inner_frac=0.06, r_outer_frac=0.98,
            center_frac=(0.5, 0.55), radius_frac=1.00,
        )

        self._build_ui()
        self.after(33, self._preview_tick)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        ttk.Label(self, text="RealSense Live View",
                  font=("Segoe UI", 14, "bold")).pack(pady=(10, 2))
        ttk.Label(self, text="Intel D455/D555 → Pepper's Cone pipeline",
                  font=("Segoe UI", 9)).pack(pady=(0, 8))

        top = ttk.Frame(self)
        top.pack(fill="both", expand=True)
        left  = ttk.Frame(top); left.pack(side="left",  fill="both", expand=True, padx=(0, 8))
        right = ttk.Frame(top); right.pack(side="right", fill="both", expand=True)

        # ── Camera controls ───────────────────────────────────────────────────
        cam_box = ttk.LabelFrame(left, text="RealSense Camera")
        cam_box.pack(fill="x", padx=4, pady=(0, 8))

        self._status_var = tk.StringVar(value="Not connected")
        ttk.Label(cam_box, textvariable=self._status_var).pack(anchor="w", padx=8, pady=(4, 2))

        btn_row = ttk.Frame(cam_box)
        btn_row.pack(padx=8, pady=(0, 6))
        self._btn_connect    = ttk.Button(btn_row, text="Connect",    command=self._connect)
        self._btn_disconnect = ttk.Button(btn_row, text="Disconnect", command=self._disconnect, state="disabled")
        self._btn_connect.grid(row=0, column=0, padx=4)
        self._btn_disconnect.grid(row=0, column=1, padx=4)

        # ── Frame source selector ─────────────────────────────────────────────
        src_box = ttk.LabelFrame(left, text="Frame Source (preview + cone)")
        src_box.pack(fill="x", padx=4, pady=(0, 8))

        self._src_var = tk.StringVar(value="color")
        for text, val in [("Color only", "color"),
                           ("Depth only", "depth"),
                           ("Side-by-side (Color | Depth)", "both")]:
            ttk.Radiobutton(src_box, text=text, variable=self._src_var,
                            value=val).pack(anchor="w", padx=8, pady=1)

        # ── Warp tuning (mirrors live_view.py) ───────────────────────────────
        tuning = ttk.LabelFrame(left, text="Cone Warp Tuning")
        tuning.pack(fill="x", padx=4, pady=(0, 8))
        tuning.grid_columnconfigure(1, weight=1)

        sliders = [
            ("Center X",      "cx",      0.50,  0.0,  1.0),
            ("Center Y",      "cy",      0.55,  0.0,  1.0),
            ("Inner Radius",  "r_inner", 0.06,  0.0,  0.95),
            ("Outer Radius",  "r_outer", 0.98,  0.5,  1.0),
            ("Span (deg)",    "span",    200,    90,   360),
            ("Rotation (deg)","rot",     270,    0,    360),
        ]
        self._sv = {}   # DoubleVar keyed by slider id
        self._sl = {}   # Label keyed by slider id
        for row_i, (label, sid, default, lo, hi) in enumerate(sliders):
            ttk.Label(tuning, text=label).grid(row=row_i, column=0, sticky="w", padx=(8,4), pady=2)
            var = tk.DoubleVar(value=default)
            lbl = ttk.Label(tuning, text=f"{default}")
            ttk.Scale(tuning, from_=lo, to=hi, orient="horizontal", variable=var,
                      command=lambda _, s=sid: self._on_slider(s)
                      ).grid(row=row_i, column=1, sticky="ew", padx=4)
            lbl.grid(row=row_i, column=2, padx=4)
            self._sv[sid] = var
            self._sl[sid] = lbl

        ttk.Button(tuning, text="Reset to Defaults",
                   command=self._reset_sliders).grid(row=len(sliders), column=0,
                                                     columnspan=3, pady=6)

        # ── Fullscreen actions ────────────────────────────────────────────────
        act = ttk.Frame(left)
        act.pack(pady=(4, 2))
        self._btn_fs_open  = ttk.Button(act, text="Open Fullscreen",  command=self._fs_open)
        self._btn_fs_close = ttk.Button(act, text="Close Fullscreen", command=self._fs_close, state="disabled")
        self._btn_fs_open.grid(row=0, column=0, padx=6)
        self._btn_fs_close.grid(row=0, column=1, padx=6)
        if self.controller:
            ttk.Button(act, text="Back",
                       command=lambda: self.controller.show_page("HomePage")
                       ).grid(row=0, column=2, padx=6)

        # ── Debug status bar ─────────────────────────────────────────────────
        self._debug_var = tk.StringVar(value="—")
        ttk.Label(left, textvariable=self._debug_var, font=("Courier", 8)).pack(
            fill="x", padx=4, pady=(2, 0))

        # ── Right: fixed-size preview panel ──────────────────────────────────
        ttk.Label(right, text="Preview (unwarped)").pack()
        container = tk.Frame(right, width=PREVIEW_W, height=PREVIEW_H,
                             bg="black", highlightthickness=0)
        container.pack(padx=4, pady=4)
        container.pack_propagate(False)
        self._preview_label = tk.Label(container, bg="black", bd=0, highlightthickness=0)
        self._preview_label.place(relx=0.5, rely=0.5, anchor="center")

    # ── Slider callbacks ──────────────────────────────────────────────────────

    def _on_slider(self, sid):
        val = self._sv[sid].get()
        fmt = ".3f" if sid in ("r_inner", "r_outer") else ".2f" if sid in ("cx","cy") else ".0f"
        self._sl[sid].config(text=f"{val:{fmt}}")
        self._rebuild_maps()

    def _rebuild_maps(self):
        self._map_x, self._map_y = build_cone_maps(
            frame_size=FRAME_SIZE, canvas_size=CANVAS_SIZE,
            span_deg=int(self._sv["span"].get()),
            rotate_deg=self._sv["rot"].get(),
            r_inner_frac=self._sv["r_inner"].get(),
            r_outer_frac=self._sv["r_outer"].get(),
            center_frac=(self._sv["cx"].get(), self._sv["cy"].get()),
            radius_frac=1.00,
        )

    def _reset_sliders(self):
        defaults = dict(cx=0.50, cy=0.55, r_inner=0.06, r_outer=0.98, span=200, rot=270)
        for sid, val in defaults.items():
            self._sv[sid].set(val)
            self._on_slider(sid)

    # ── RealSense connect / capture ───────────────────────────────────────────

    def _connect(self):
        if not REALSENSE_AVAILABLE:
            messagebox.showerror(
                "pyrealsense2 missing",
                "pyrealsense2 is not installed.\n\n"
                "Run:  pip install pyrealsense2\n\n"
                "Also install the Intel RealSense SDK 2.0:\n"
                "https://github.com/IntelRealSense/librealsense/releases",
            )
            return

        try:
            self._pipeline, self._align = rs_start()
        except Exception as e:
            messagebox.showerror("RealSense Error",
                                 f"Could not open camera:\n{e}\n\n"
                                 "Make sure:\n"
                                 "  • Camera is plugged into a USB 3.x (blue) port\n"
                                 "  • RealSense Viewer is closed\n"
                                 "  • No other app is using the camera")
            return

        self._stop_evt.clear()
        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._cap_thread.start()

        self._btn_connect.config(state="disabled")
        self._btn_disconnect.config(state="normal")
        self._status_var.set("Connected")

    def _disconnect(self):
        self._stop_evt.set()
        if self._cap_thread:
            self._cap_thread.join(timeout=2.0)
        self._cap_thread = None
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None
        self._btn_connect.config(state="normal")
        self._btn_disconnect.config(state="disabled")
        self._status_var.set("Not connected")

    def _capture_loop(self):
        """Background thread: pulls frames from RealSense, stores in self._last_bgr."""
        frame_count = 0
        t0 = time.perf_counter()

        while not self._stop_evt.is_set():
            try:
                color_bgr, depth_mm, depth_vis = rs_get_frame(self._pipeline, self._align)
            except Exception:
                break   # pipeline stopped

            if color_bgr is None:
                continue

            # Pick the frame to send downstream based on current source toggle
            src = self._src_var.get()
            if src == "color":
                out_frame = color_bgr
            elif src == "depth":
                out_frame = depth_vis
            else:  # "both"
                out_frame = np.hstack([color_bgr, depth_vis])

            with self._frame_lock:
                self._last_bgr   = out_frame
                self._last_depth = depth_mm

            # FPS
            frame_count += 1
            elapsed = time.perf_counter() - t0
            if elapsed >= 1.0:
                self._fps = frame_count / elapsed
                frame_count = 0
                t0 = time.perf_counter()

    # ── UI tick: update preview label ─────────────────────────────────────────

    def _preview_tick(self):
        frame = None
        depth = None
        with self._frame_lock:
            if self._last_bgr is not None:
                frame = self._last_bgr.copy()
                depth = self._last_depth

        if frame is not None:
            fh, fw = frame.shape[:2]
            scale  = min(PREVIEW_W / fw, PREVIEW_H / fh)
            new_w  = max(1, int(fw * scale))
            new_h  = max(1, int(fh * scale))
            thumb  = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb    = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            img    = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._preview_label.config(image=img)
            self._preview_label.image = img

            # Debug bar
            if depth is not None:
                valid = depth[depth > 0]
                if valid.size:
                    self._debug_var.set(
                        f"FPS:{self._fps:5.1f}  "
                        f"frame:{frame.shape}  "
                        f"depth mm — min:{valid.min():4d}  max:{valid.max():5d}  mean:{int(valid.mean()):5d}"
                    )
            else:
                self._debug_var.set(f"FPS:{self._fps:5.1f}  frame:{frame.shape}")

        self.after(33, self._preview_tick)

    # ── Warp + fullscreen ─────────────────────────────────────────────────────

    def _apply_warp(self, frame_bgr):
        """Same pipeline as live_view.py but without MediaPipe (not needed for depth cam test)."""
        sq = cv2.resize(frame_bgr, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)

        # Center + scale (matching live_view.py default 0.6)
        scale  = 0.6
        scaled = cv2.resize(sq, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        padded = np.zeros_like(sq)
        y_off  = (FRAME_SIZE - scaled.shape[0]) // 2
        x_off  = (FRAME_SIZE - scaled.shape[1]) // 2
        padded[y_off:y_off+scaled.shape[0], x_off:x_off+scaled.shape[1]] = scaled

        warped   = cv2.remap(padded, self._map_x, self._map_y,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        enhanced = enhance_saturation_contrast(warped, saturation_scale=1.4,
                                               contrast_alpha=1.8, brightness_beta=-25)
        return enhanced

    def _fs_open(self):
        frame = None
        with self._frame_lock:
            frame = self._last_bgr
        if frame is None:
            messagebox.showwarning("Fullscreen", "Connect to the camera first.")
            return

        self._fs_win = tk.Toplevel(self)
        self._fs_win.title("Pepper's Cone — RealSense")
        self._fs_win.attributes("-fullscreen", True)
        self._fs_win.configure(bg="black")
        self._fs_win.bind("<Escape>", lambda _: self._fs_close())
        self._fs_win.protocol("WM_DELETE_WINDOW", self._fs_close)

        self._fs_label = tk.Label(self._fs_win, bg="black")
        self._fs_label.pack(fill="both", expand=True)

        self._fs_running = True
        self._btn_fs_open.config(state="disabled")
        self._btn_fs_close.config(state="normal")
        self._fs_tick()

    def _fs_close(self):
        self._fs_running = False
        if self._fs_win:
            try:
                self._fs_win.destroy()
            except Exception:
                pass
        self._fs_win = None
        self._fs_label = None
        self._btn_fs_open.config(state="normal")
        self._btn_fs_close.config(state="disabled")

    def _fs_tick(self):
        if not self._fs_running:
            return
        frame = None
        with self._frame_lock:
            if self._last_bgr is not None:
                frame = self._last_bgr.copy()

        if frame is not None:
            warped = self._apply_warp(frame)
            try:
                sw = self._fs_win.winfo_width()
                sh = self._fs_win.winfo_height()
                fh, fw = warped.shape[:2]
                s = min(sw / fw, sh / fh)
                disp = cv2.resize(warped, (max(1, int(fw*s)), max(1, int(fh*s))),
                                  interpolation=cv2.INTER_AREA)
            except Exception:
                disp = warped

            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            if self._fs_label:
                self._fs_label.config(image=img)
                self._fs_label.image = img

        self.after(16, self._fs_tick)

    # ── Clean shutdown ────────────────────────────────────────────────────────

    def destroy(self):
        self._disconnect()
        super().destroy()


# ── Standalone entry point ────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.title("RealSense Live — Pepper's Cone")
    root.geometry("1280x780")
    root.minsize(900, 600)

    view = RealSenseLiveView(root, controller=None)
    view.pack(fill="both", expand=True)

    root.protocol("WM_DELETE_WINDOW", lambda: (view.destroy(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
