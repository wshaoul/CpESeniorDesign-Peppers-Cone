# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Pepper's Cone holographic display system — a Python/OpenCV application that warps a camera feed into a cone-shaped reflector pattern to create a 3D hologram illusion. The goal is to scale to a 72-inch display for classroom use, where a remote instructor appears as a life-sized holographic presence.

**Active workspace:** `Interface_updated/` — do not modify `Interface/` (legacy).

## Running the Application

All commands assume `Interface_updated/` as the working directory, using the bundled Python 3.10 virtual environment:

```bash
# Main app (single-face Pepper's Cone)
winvenv310/Scripts/python.exe studio_main.py

# Circle hologram variant (four 90°-rotated faces for 360° viewing)
winvenv310/Scripts/python.exe studio_circle.py

# RealSense-only standalone viewer
winvenv310/Scripts/python.exe realsense_live.py

# RealSense hardware diagnostic (verify camera before running main app)
winvenv310/Scripts/python.exe realsense_test.py
```

## Dependencies

**Required:** `opencv-python`, `numpy`, `Pillow`, `mediapipe>=0.10.21`, `tkinter` (built-in)  
**Optional:** `pyrealsense2` (Intel RealSense D455/D555 support), `ttkbootstrap` (modern UI themes), `ffmpeg` system binary (DirectShow device listing on Windows)

The app degrades gracefully when optional deps are missing — RealSense option is hidden, ttkbootstrap falls back to standard ttk.

## Architecture

### Multi-page Tkinter router
`studio_main.py` hosts a stack of `ttk.Frame` subclasses. `controller.show_page(name)` raises the requested page. Pages declare `start_async()` for background init. The same pattern is used in `studio_circle.py`.

### Core warp pipeline (`live_view.py`)
`build_cone_maps(frame_size, canvas_size, span_deg, rotate_deg, r_inner_frac, r_outer_frac, center_frac)` → `(map_x, map_y)` float32 arrays for `cv2.remap()`. It maps each canvas pixel to a source-frame pixel by computing polar coordinates and fitting them into the arc band. Key defaults:

| Constant | Value | Meaning |
|----------|-------|---------|
| `FRAME_SIZE` | 400 | Input frame side length |
| `CANVAS_SIZE` | 800 | Warp output side length |
| `span_deg` | 200 | Arc width of the visible cone face |
| `rotate_deg` | 270 | Arc rotation (270 = facing up) |
| `r_inner_frac` | 0.08 | Inner radius of the annular band |
| `r_outer_frac` | 0.995 | Outer radius of the annular band |

### Four-face circle hologram (`live_view_circle.py`)
`build_four_cone_maps()` calls `build_cone_maps()` four times with 90° rotation steps. `apply_four_cone_views()` composites all four warped frames using `np.maximum` (additive bright blend). Each of the four faces flips/rotates the source frame independently before warping.

### Threading model
Camera capture runs in a daemon thread; the latest frame is stored in `_last_bgr` behind a `threading.Lock`. UI preview ticks via `after(33, ...)` at ~30 FPS; fullscreen OpenCV window updates at `after(16, ...)` for ~60 FPS.

### Background removal
MediaPipe `selfie_segmentation` (model 1, high-accuracy) produces a float mask; pixels above 0.5 are kept, then smoothed with Gaussian blur and morphological closing before masking the BGR frame.

### Camera selection
The app enumerates DirectShow devices via `ffmpeg -list_devices` on Windows and lets the user pick by name or index. It tries `cv2.CAP_MSMF`, then `cv2.CAP_DSHOW`, then `cv2.CAP_ANY` in sequence. The RealSense uses `pyrealsense2` directly (not via OpenCV), at 640×360 @ 60 FPS for D455; D555 needs 1280×720 minimum.

## Key Files

| File | Role |
|------|------|
| `Interface_updated/studio_main.py` | Entry point, page router, HomePage dashboard |
| `Interface_updated/live_view.py` | Core warp algorithm, live camera, segmentation, sliders |
| `Interface_updated/live_view_circle.py` | Four-face 360° variant of the warp pipeline |
| `Interface_updated/record_view.py` | Record raw feed → play back through warp |
| `Interface_updated/upload_view.py` | Load existing video → play through warp |
| `Interface_updated/studio_circle.py` | Router for circle hologram app |
| `Interface_updated/realsense_live.py` | Standalone RealSense viewer with warp |
| `Interface_updated/realsense_test.py` | Hardware diagnostic for RealSense camera |
| `Interface_updated/REALSENSE_SETUP.md` | RealSense setup guide and troubleshooting |

## Warp Orientation Convention

Arc `rotate_deg` maps to visual direction as follows: 270° = top of canvas, 90° = bottom, 0° = right, 180° = left. When adding new warp faces, account for source-frame flip/rotation *before* warp — see `apply_four_cone_views()` for the pattern.
