# realsense_live.py
#
# Standalone RealSense viewer for Pepper's Cone — pre-alpha milestone.
# Reuses build_cone_maps and enhance_saturation_contrast from live_view.py.
# Does NOT modify live_view.py in any way.
#
# Run:  winvenv310\Scripts\python.exe realsense_live.py
# Quit: close the window

import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk

# Reuse warp helpers from live_view — no code duplication
from live_view import build_cone_maps, enhance_saturation_contrast

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
FRAME_SIZE  = 400
CANVAS_SIZE = 800
PREVIEW_W   = 640
PREVIEW_H   = 480

# D555 supported resolutions: 1280x800, 1280x720, 896x504, 640x360, 448x252
RS_W, RS_H, RS_FPS = 640, 360, 60

DEPTH_COLORMAP  = cv2.COLORMAP_TURBO
FRAME_TIMEOUT   = 3000   # ms — timeout per wait_for_frames call
WARMUP_FRAMES   = 10     # skip first N frames while camera stabilises


# ── RealSense helpers ─────────────────────────────────────────────────────────

def rs_start():
    """Start pipeline + aligner. Raises on failure."""
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, RS_W, RS_H, rs.format.bgr8, RS_FPS)
    cfg.enable_stream(rs.stream.depth, RS_W, RS_H, rs.format.z16,  RS_FPS)
    profile = pipeline.start(cfg)
    align   = rs.align(rs.stream.color)
    dev = profile.get_device()
    name = dev.get_info(rs.camera_info.name)
    sn   = dev.get_info(rs.camera_info.serial_number)
    print(f"[RealSense] Connected: {name}  s/n {sn}  ({RS_W}x{RS_H} @ {RS_FPS}fps)")
    return pipeline, align


def rs_get_frame(pipeline, align):
    """
    Grab one aligned frameset with a timeout.
    Returns (color_bgr, depth_mm, depth_vis) or raises RuntimeError on timeout.
    """
    frames  = pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT)
    aligned = align.process(frames)

    cf = aligned.get_color_frame()
    df = aligned.get_depth_frame()
    if not cf or not df:
        return None, None, None

    color_bgr = np.asanyarray(cf.get_data())   # (H,W,3) uint8 BGR
    depth_mm  = np.asanyarray(df.get_data())   # (H,W)   uint16 mm

    depth_norm = (np.clip(depth_mm, 0, 4000).astype(np.float32) / 4000.0 * 255).astype(np.uint8)
    depth_vis  = cv2.applyColorMap(depth_norm, DEPTH_COLORMAP)

    return color_bgr, depth_mm, depth_vis


# ── View ──────────────────────────────────────────────────────────────────────

class RealSenseLiveView(ttk.Frame):

    def __init__(self, parent, controller=None):
        super().__init__(parent)
        self.controller = controller

        self._pipeline   = None
        self._align      = None
        self._last_bgr   = None
        self._last_depth = None
        self._frame_lock = threading.Lock()
        self._cap_thread = None
        self._stop_evt   = threading.Event()
        self._fps        = 0.0
        self._fs_win     = None
        self._fs_label   = None
        self._fs_running = False

        self._map_x, self._map_y = build_cone_maps(
            frame_size=FRAME_SIZE, canvas_size=CANVAS_SIZE,
            span_deg=200, rotate_deg=270,
            r_inner_frac=0.06, r_outer_frac=0.98,
            center_frac=(0.5, 0.55), radius_frac=1.00,
        )

        # Incremented by the capture thread each time a new frame lands.
        # _fs_tick compares against this to skip warp work on duplicate frames.
        self._frame_id      = 0
        self._fs_last_id    = -1

        self._build_ui()
        self.after(33, self._preview_tick)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        ttk.Label(self, text="RealSense Live View",
                  font=("Segoe UI", 14, "bold")).pack(pady=(10, 2))
        ttk.Label(self, text="Intel D555 → Pepper's Cone pipeline",
                  font=("Segoe UI", 9)).pack(pady=(0, 8))

        top   = ttk.Frame(self);  top.pack(fill="both", expand=True)
        left  = ttk.Frame(top);  left.pack(side="left",  fill="both", expand=True, padx=(0, 8))
        right = ttk.Frame(top);  right.pack(side="right", fill="both", expand=True)

        # Camera controls
        cam_box = ttk.LabelFrame(left, text="RealSense Camera")
        cam_box.pack(fill="x", padx=4, pady=(0, 8))
        self._status_var = tk.StringVar(value="Not connected")
        ttk.Label(cam_box, textvariable=self._status_var,
                  font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8, pady=(6, 2))
        btn_row = ttk.Frame(cam_box); btn_row.pack(padx=8, pady=(0, 8))
        self._btn_connect    = ttk.Button(btn_row, text="Connect",    command=self._connect)
        self._btn_disconnect = ttk.Button(btn_row, text="Disconnect", command=self._disconnect, state="disabled")
        self._btn_connect.grid(row=0, column=0, padx=4)
        self._btn_disconnect.grid(row=0, column=1, padx=4)

        # Frame source
        src_box = ttk.LabelFrame(left, text="Frame Source")
        src_box.pack(fill="x", padx=4, pady=(0, 8))
        self._src_var = tk.StringVar(value="color")
        for text, val in [("Color only", "color"),
                           ("Depth only", "depth"),
                           ("Side-by-side", "both")]:
            ttk.Radiobutton(src_box, text=text, variable=self._src_var,
                            value=val).pack(anchor="w", padx=8, pady=1)

        # Warp tuning
        tuning = ttk.LabelFrame(left, text="Cone Warp Tuning")
        tuning.pack(fill="x", padx=4, pady=(0, 8))
        tuning.grid_columnconfigure(1, weight=1)
        sliders = [
            ("Center X",       "cx",      0.50, 0.0,  1.0),
            ("Center Y",       "cy",      0.55, 0.0,  1.0),
            ("Inner Radius",   "r_inner", 0.06, 0.0,  0.95),
            ("Outer Radius",   "r_outer", 0.98, 0.5,  1.0),
            ("Span (deg)",     "span",    200,  90,   360),
            ("Rotation (deg)", "rot",     270,  0,    360),
        ]
        self._sv = {}; self._sl = {}
        for i, (label, sid, default, lo, hi) in enumerate(sliders):
            ttk.Label(tuning, text=label).grid(row=i, column=0, sticky="w", padx=(8, 4), pady=2)
            var = tk.DoubleVar(value=default)
            lbl = ttk.Label(tuning, text=str(default))
            ttk.Scale(tuning, from_=lo, to=hi, orient="horizontal", variable=var,
                      command=lambda _, s=sid: self._on_slider(s)
                      ).grid(row=i, column=1, sticky="ew", padx=4)
            lbl.grid(row=i, column=2, padx=4)
            self._sv[sid] = var; self._sl[sid] = lbl
        ttk.Button(tuning, text="Reset", command=self._reset_sliders
                   ).grid(row=len(sliders), column=0, columnspan=3, pady=6)

        # Fullscreen buttons
        act = ttk.Frame(left); act.pack(pady=(4, 2))
        self._btn_fs_open  = ttk.Button(act, text="Open Fullscreen",  command=self._fs_open)
        self._btn_fs_close = ttk.Button(act, text="Close Fullscreen", command=self._fs_close, state="disabled")
        self._btn_fs_open.grid(row=0, column=0, padx=6)
        self._btn_fs_close.grid(row=0, column=1, padx=6)

        # Debug bar
        self._debug_var = tk.StringVar(value="—")
        ttk.Label(left, textvariable=self._debug_var,
                  font=("Courier", 8)).pack(fill="x", padx=4, pady=(4, 0))

        # Preview panel
        ttk.Label(right, text="Preview (unwarped)").pack()
        container = tk.Frame(right, width=PREVIEW_W, height=PREVIEW_H,
                             bg="black", highlightthickness=0)
        container.pack(padx=4, pady=4)
        container.pack_propagate(False)
        self._preview_label = tk.Label(container, bg="black", bd=0, highlightthickness=0)
        self._preview_label.place(relx=0.5, rely=0.5, anchor="center")

    # ── Sliders ───────────────────────────────────────────────────────────────

    def _on_slider(self, sid):
        val = self._sv[sid].get()
        fmt = ".3f" if sid in ("r_inner","r_outer") else ".2f" if sid in ("cx","cy") else ".0f"
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
        for sid, val in dict(cx=0.50, cy=0.55, r_inner=0.06,
                              r_outer=0.98, span=200, rot=270).items():
            self._sv[sid].set(val); self._on_slider(sid)

    # ── Camera ────────────────────────────────────────────────────────────────

    def _connect(self):
        if not REALSENSE_AVAILABLE:
            messagebox.showerror("Missing package",
                "pyrealsense2 is not installed.\n\nRun:\n"
                "  winvenv310\\Scripts\\pip install pyrealsense2")
            return

        self._status_var.set("Connecting...")
        self.update_idletasks()

        try:
            self._pipeline, self._align = rs_start()
        except Exception as e:
            self._status_var.set("Connection failed")
            messagebox.showerror("RealSense Error",
                f"Could not open camera:\n\n{e}\n\n"
                "Check:\n"
                "  • Camera plugged into USB 3.x (blue/SS port)\n"
                "  • RealSense Viewer is closed\n"
                "  • No other app is using the camera")
            return

        self._stop_evt.clear()
        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._cap_thread.start()
        self._btn_connect.config(state="disabled")
        self._btn_disconnect.config(state="normal")
        self._status_var.set(f"Connected  ({RS_W}x{RS_H} @ {RS_FPS}fps)")

    def _disconnect(self):
        self._stop_evt.set()
        if self._cap_thread:
            self._cap_thread.join(timeout=3.0)
        self._cap_thread = None
        if self._pipeline:
            try: self._pipeline.stop()
            except Exception: pass
        self._pipeline = None
        self._btn_connect.config(state="normal")
        self._btn_disconnect.config(state="disabled")
        self._status_var.set("Not connected")

    def _capture_loop(self):
        """
        Background thread: grabs frames from RealSense, stores latest in _last_bgr.
        - Skips WARMUP_FRAMES to let the camera stabilise.
        - On timeout, retries instead of dying.
        - Reports persistent errors to the status label.
        """
        warmup     = WARMUP_FRAMES
        fail_count = 0
        frame_count = 0
        t0 = time.perf_counter()

        print(f"[capture] thread started, warming up {warmup} frames...")

        while not self._stop_evt.is_set():
            try:
                color_bgr, depth_mm, depth_vis = rs_get_frame(self._pipeline, self._align)
                fail_count = 0   # reset on success
            except RuntimeError as e:
                fail_count += 1
                msg = f"Frame timeout ({fail_count})"
                print(f"[capture] {msg}: {e}")
                self.after(0, lambda m=msg: self._status_var.set(m))
                if fail_count >= 10:
                    self.after(0, lambda: self._status_var.set(
                        "ERROR: no frames — unplug & replug camera"))
                    break
                continue
            except Exception as e:
                print(f"[capture] fatal: {e}")
                self.after(0, lambda m=str(e): self._status_var.set(f"Error: {m}"))
                break

            if color_bgr is None:
                continue

            # skip warmup frames
            if warmup > 0:
                warmup -= 1
                if warmup == 0:
                    print("[capture] warmup done, displaying frames")
                continue

            src = self._src_var.get()
            if src == "color":
                out_frame = color_bgr
            elif src == "depth":
                out_frame = depth_vis
            else:
                out_frame = np.hstack([color_bgr, depth_vis])

            with self._frame_lock:
                self._last_bgr   = out_frame
                self._last_depth = depth_mm
                self._frame_id  += 1

            frame_count += 1
            elapsed = time.perf_counter() - t0
            if elapsed >= 1.0:
                self._fps   = frame_count / elapsed
                frame_count = 0
                t0          = time.perf_counter()

        print("[capture] thread exiting")

    # ── Preview tick ──────────────────────────────────────────────────────────

    def _preview_tick(self):
        frame = depth = None
        with self._frame_lock:
            if self._last_bgr is not None:
                frame = self._last_bgr.copy()
                depth = self._last_depth

        if frame is not None:
            fh, fw = frame.shape[:2]
            scale  = min(PREVIEW_W / fw, PREVIEW_H / fh)
            thumb  = cv2.resize(frame, (max(1, int(fw*scale)), max(1, int(fh*scale))),
                                interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._preview_label.config(image=img)
            self._preview_label.image = img

            if depth is not None:
                valid = depth[depth > 0]
                if valid.size:
                    self._debug_var.set(
                        f"FPS:{self._fps:5.1f}  shape:{frame.shape}  "
                        f"depth mm  min:{valid.min()}  max:{valid.max()}  mean:{int(valid.mean())}"
                    )
            else:
                self._debug_var.set(f"FPS:{self._fps:5.1f}  shape:{frame.shape}")

        self.after(33, self._preview_tick)

    # ── Warp / fullscreen ─────────────────────────────────────────────────────

    def _apply_warp(self, frame_bgr):
        sq     = cv2.resize(frame_bgr, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        scale  = 0.6
        scaled = cv2.resize(sq, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        padded = np.zeros_like(sq)
        yo = (FRAME_SIZE - scaled.shape[0]) // 2
        xo = (FRAME_SIZE - scaled.shape[1]) // 2
        padded[yo:yo+scaled.shape[0], xo:xo+scaled.shape[1]] = scaled
        warped   = cv2.remap(padded, self._map_x, self._map_y,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return enhance_saturation_contrast(warped, saturation_scale=1.4,
                                           contrast_alpha=1.8, brightness_beta=-25)

    def _fs_open(self):
        with self._frame_lock:
            has_frame = self._last_bgr is not None
        if not has_frame:
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
        self._fs_running  = True
        self._fs_last_id  = -1   # force first frame to render immediately
        self._btn_fs_open.config(state="disabled")
        self._btn_fs_close.config(state="normal")
        self._fs_tick()

    def _fs_close(self):
        self._fs_running = False
        if self._fs_win:
            try: self._fs_win.destroy()
            except Exception: pass
        self._fs_win = None; self._fs_label = None
        self._btn_fs_open.config(state="normal")
        self._btn_fs_close.config(state="disabled")

    def _fs_tick(self):
        if not self._fs_running:
            return

        frame    = None
        frame_id = -1
        with self._frame_lock:
            if self._last_bgr is not None:
                frame_id = self._frame_id
                # Only copy if this is genuinely a new frame
                if frame_id != self._fs_last_id:
                    frame = self._last_bgr.copy()

        if frame is not None:
            self._fs_last_id = frame_id
            warped = self._apply_warp(frame)
            try:
                sw, sh = self._fs_win.winfo_width(), self._fs_win.winfo_height()
                fh, fw = warped.shape[:2]
                s = min(sw / fw, sh / fh)
                # INTER_AREA is only good for downscaling; use INTER_LINEAR for upscaling
                interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
                disp = cv2.resize(warped, (max(1, int(fw * s)), max(1, int(fh * s))),
                                  interpolation=interp)
            except Exception:
                disp = warped
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            if self._fs_label:
                self._fs_label.config(image=img)
                self._fs_label.image = img

        # Poll at ~33 ms to match the 30 fps camera rate — no benefit running faster
        self.after(33, self._fs_tick)

    def destroy(self):
        self._disconnect()
        super().destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.title("RealSense Live — Pepper's Cone")
    root.geometry("1280x780")
    root.minsize(900, 600)
    view = RealSenseLiveView(root)
    view.pack(fill="both", expand=True)
    root.protocol("WM_DELETE_WINDOW", lambda: (view.destroy(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
