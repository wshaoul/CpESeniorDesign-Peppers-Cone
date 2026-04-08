# realsense_test.py
#
# Milestone test: Intel RealSense D455/D555 color + depth capture
# - Displays live color frame in one window
# - Displays colorized depth frame in a second window (or side-by-side)
# - Prints debug info: shape, dtype, depth min/max, FPS
#
# Run: python Interface/realsense_test.py
# Quit: press 'q' in either window

import sys
import time
import numpy as np
import cv2

# ── RealSense import with a clear error on failure ──────────────────────────
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


# ── Configuration ────────────────────────────────────────────────────────────
COLOR_W, COLOR_H = 640, 480
DEPTH_W, DEPTH_H = 640, 480
FPS_TARGET = 30

# Depth colormap: COLORMAP_TURBO gives a perceptually clear hot→cold gradient.
# Options: cv2.COLORMAP_JET, cv2.COLORMAP_TURBO, cv2.COLORMAP_MAGMA
DEPTH_COLORMAP = cv2.COLORMAP_TURBO

# How often (in frames) to print debug info to console
DEBUG_INTERVAL = 30


# ── RealSense pipeline setup ─────────────────────────────────────────────────
def make_pipeline():
    """Configure and start the RealSense pipeline. Returns (pipeline, align)."""
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, FPS_TARGET)
    config.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H, rs.format.z16, FPS_TARGET)

    profile = pipeline.start(config)

    # Align depth to color so both frames share the same viewport
    align = rs.align(rs.stream.color)

    # Print device info
    dev = profile.get_device()
    print(f"[RealSense] Connected: {dev.get_info(rs.camera_info.name)}")
    print(f"[RealSense] Serial:    {dev.get_info(rs.camera_info.serial_number)}")
    print(f"[RealSense] Firmware:  {dev.get_info(rs.camera_info.firmware_version)}")
    print(f"[RealSense] Color stream: {COLOR_W}x{COLOR_H} @ {FPS_TARGET} fps (BGR8)")
    print(f"[RealSense] Depth stream: {DEPTH_W}x{DEPTH_H} @ {FPS_TARGET} fps (Z16)")
    print()

    return pipeline, align


# ── Single-frame capture + processing ────────────────────────────────────────
def get_frames(pipeline, align):
    """
    Block until a coherent frameset arrives, return:
      color_bgr  – (H, W, 3) uint8 BGR image
      depth_raw  – (H, W) uint16 distance in millimetres
      depth_vis  – (H, W, 3) uint8 colorized depth for display
    """
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()

    if not color_frame or not depth_frame:
        return None, None, None

    color_bgr = np.asanyarray(color_frame.get_data())   # (H, W, 3) uint8
    depth_raw = np.asanyarray(depth_frame.get_data())   # (H, W)    uint16 mm

    # Clamp to a visible range (0–4 m) then normalise to 0–255 for colormap
    depth_clipped = np.clip(depth_raw, 0, 4000).astype(np.float32)
    depth_norm = (depth_clipped / 4000.0 * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_norm, DEPTH_COLORMAP)

    return color_bgr, depth_raw, depth_vis


# ── Debug print ───────────────────────────────────────────────────────────────
def print_debug(color_bgr, depth_raw, fps):
    # Only consider non-zero depth pixels (0 = no reading)
    valid = depth_raw[depth_raw > 0]
    d_min = int(valid.min()) if valid.size else 0
    d_max = int(valid.max()) if valid.size else 0
    d_mean = int(valid.mean()) if valid.size else 0

    print(
        f"FPS: {fps:5.1f} | "
        f"Color {color_bgr.shape} {color_bgr.dtype} | "
        f"Depth {depth_raw.shape} {depth_raw.dtype} | "
        f"Depth mm — min:{d_min:4d}  max:{d_max:5d}  mean:{d_mean:5d}"
    )


# ── Main loop ─────────────────────────────────────────────────────────────────
def run():
    if not REALSENSE_AVAILABLE:
        print("ERROR: pyrealsense2 is not installed.")
        print("Install it with:  pip install pyrealsense2")
        print()
        print("Also ensure the Intel RealSense SDK 2.0 is installed:")
        print("  https://github.com/IntelRealSense/librealsense/releases")
        sys.exit(1)

    print("[RealSense Test] Starting — press 'q' to quit.\n")

    try:
        pipeline, align = make_pipeline()
    except Exception as e:
        print(f"ERROR: Could not start RealSense pipeline.\n  {e}")
        print()
        print("Check that:")
        print("  1. The D455/D555 is plugged in via USB 3.x (blue port)")
        print("  2. No other app is holding the camera (RealSense Viewer, etc.)")
        sys.exit(1)

    frame_count = 0
    t_start = time.perf_counter()
    fps = 0.0

    try:
        while True:
            color_bgr, depth_raw, depth_vis = get_frames(pipeline, align)

            if color_bgr is None:
                continue

            frame_count += 1
            elapsed = time.perf_counter() - t_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t_start = time.perf_counter()

            # ── Console debug (every N frames) ──────────────────────────────
            if frame_count % DEBUG_INTERVAL == 1:
                print_debug(color_bgr, depth_raw, fps)

            # ── Overlay FPS on color frame ───────────────────────────────────
            color_display = color_bgr.copy()
            cv2.putText(
                color_display,
                f"FPS: {fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                color_display,
                "Color (BGR)",
                (10, COLOR_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # ── Overlay depth range on depth frame ──────────────────────────
            valid = depth_raw[depth_raw > 0]
            if valid.size:
                d_min = int(valid.min())
                d_max = int(valid.max())
                cv2.putText(
                    depth_vis,
                    f"min:{d_min}mm  max:{d_max}mm",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                depth_vis,
                "Depth (colorized, 0-4m)",
                (10, DEPTH_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # ── Display: side-by-side in one window ──────────────────────────
            # If you prefer two separate windows, swap to the two imshow() calls below.
            side_by_side = np.hstack([color_display, depth_vis])
            cv2.imshow("RealSense Test — Color | Depth  (q to quit)", side_by_side)

            # Separate windows (optional — uncomment and comment out side_by_side above):
            # cv2.imshow("Color", color_display)
            # cv2.imshow("Depth", depth_vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n[RealSense Test] Stopped.")


if __name__ == "__main__":
    run()
