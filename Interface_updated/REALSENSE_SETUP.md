# RealSense Depth Camera — Setup & Usage Guide

**Pepper's Cone Holographic Display — University of Florida CEN3907C**  
Applies to: `Interface/realsense_live.py` and `Interface/realsense_test.py`

---

## What This Is

Two new Python scripts were added to integrate the **Intel RealSense D455/D555** depth camera into the Pepper's Cone pipeline:

| File | Purpose |
|---|---|
| `realsense_live.py` | Full GUI viewer — live color + depth feed with cone warp output. **Use this for demos.** |
| `realsense_test.py` | Minimal diagnostic — confirms the camera is working and prints frame/depth stats. Use this to verify setup. |

Neither file modifies any existing code. The existing `studio_main.py` / `live_view.py` app is untouched.

---

## Hardware Requirements

- **Intel RealSense D455 or D555** depth camera
- **USB 3.x port** — must be the blue port or labeled `SS` (SuperSpeed). USB 2.0 will not deliver frames.
- Windows 10 or 11

> The D555 supports these resolutions: `1280x800`, `1280x720`, `896x504`, `640x360`, `448x252`.  
> It does **not** support `640x480`. The app is configured for `1280x720 @ 30fps`.

---

## Software Prerequisites

### 1. Intel RealSense SDK 2.0 (required — Windows drivers)

Download and install from:  
**https://github.com/IntelRealSense/librealsense/releases/latest**

Look for `Intel.RealSense.SDK-WIN10-<version>.exe` and run the installer.  
This installs the USB drivers that Windows needs to talk to the camera.

To verify the SDK is working, open **Intel RealSense Viewer** (installed with the SDK) and confirm you see a live color and depth feed.

### 2. Python Environment

This project uses the existing `winvenv310` virtual environment (Python 3.10) already in the repo.

If you have not set it up yet:
```powershell
cd CpESeniorDesign-Peppers-Cone/Interface
py -3.10 -m venv winvenv310
.\winvenv310\Scripts\Activate.ps1
pip install opencv-python mediapipe numpy pillow
pip install mediapipe==0.10.21
```

### 3. Install pyrealsense2

With the venv active, run:
```powershell
pip install pyrealsense2
```

Or without activating:
```powershell
winvenv310\Scripts\pip install pyrealsense2
```

---

## How to Run

**Always use the `winvenv310` Python — not the system Python.**

### Full GUI Viewer (recommended)
```powershell
cd CpESeniorDesign-Peppers-Cone/Interface
winvenv310\Scripts\python.exe realsense_live.py
```

### Diagnostic Test (verify camera only)
```powershell
cd CpESeniorDesign-Peppers-Cone/Interface
winvenv310\Scripts\python.exe realsense_test.py
```

---

## Using the GUI Viewer (`realsense_live.py`)

### Step-by-step

1. **Run the script** — a window opens with controls on the left and a black preview on the right.
2. **Click Connect** — the status bar will show `Connected (1280x720 @ 30fps)` and the terminal will print the camera name and serial number.
3. **Choose a frame source** using the radio buttons:
   - `Color only` — RGB camera feed
   - `Depth only` — colorized depth map (blue = far, red = near)
   - `Side-by-side` — color and depth next to each other
4. **Preview panel** (right side) — shows the raw unwarped feed at 30fps.
5. **Click Open Fullscreen** — opens a fullscreen window with the cone-warped output, same as the main app.
6. **Tune the warp** using the sliders (Center X/Y, Inner/Outer Radius, Span, Rotation). These mirror the controls in `live_view.py`.
7. **Press Escape** to close fullscreen. Click **Disconnect** before closing the window.

### Debug bar

The bottom of the left panel shows a live readout:
```
FPS: 29.8  shape:(720, 1280, 3)  depth mm  min:312  max:3847  mean:1204
```
This confirms frames are flowing and shows the depth range in millimetres.

---

## Important Rules

**1. Close Intel RealSense Viewer before running the Python app.**  
Only one application can hold the camera at a time. If the Viewer is open, Python will connect but receive no frames. Always exit the Viewer using its own close button — do not force-kill it, as this leaves the camera in a locked state requiring an unplug/replug to recover.

**2. Use a USB 3.x port.**  
If the camera is on a USB 2.0 port, the pipeline will start but frames will never arrive. Look for the blue port or `SS` label on your laptop/hub.

**3. If frames stop arriving or you see "ERROR: no frames":**
   - Unplug the camera
   - Wait 10 seconds
   - Replug into a USB 3.x port
   - Rerun the script

---

## Verifying the Camera is Detected

Run this one-liner to confirm Python can see the camera before launching the app:

```powershell
winvenv310\Scripts\python.exe -c "import pyrealsense2 as rs; d=rs.context().query_devices(); print('Found:', len(d), 'device(s)'); [print(' ', x.get_info(rs.camera_info.name), '|', x.get_info(rs.camera_info.serial_number)) for x in d]"
```

Expected output:
```
Found: 1 device(s)
  Intel RealSense D555 | 419222300404
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No module named 'pyrealsense2'` | Package not installed | `winvenv310\Scripts\pip install pyrealsense2` |
| `No module named 'mediapipe'` | Wrong Python being used | Use `winvenv310\Scripts\python.exe`, not `python` |
| `Found: 0 device(s)` | SDK drivers not installed | Install Intel RealSense SDK 2.0 |
| Connected but no frames display | RealSense Viewer is open | Close the Viewer, unplug/replug camera |
| `Frame didn't arrive within 3000` | USB 2.0 port or bad state | Use USB 3.x port; unplug/replug if needed |
| `Couldn't resolve requests` | Invalid stream resolution | Already fixed in code — D555 uses 1280x720 |
| `Conflict profiles with other stream` | Camera in locked state | Unplug and replug the camera |

---

## How It Connects to the Cone Pipeline

`realsense_live.py` imports two functions directly from `live_view.py`:

```python
from live_view import build_cone_maps, enhance_saturation_contrast
```

The RealSense color frame is a standard `(H, W, 3) uint8 BGR` numpy array — identical format to what `cv2.VideoCapture` produces. When you click **Open Fullscreen**, the same warp + enhancement pipeline from the main app runs on the RealSense feed.

To integrate RealSense into the main Studio app later, add one line to `studio_main.py`:
```python
from realsense_live import RealSenseLiveView
```
And register it as a page — no other changes needed.

---

## File Reference

```
Interface/
├── realsense_live.py       # Full GUI viewer (use for demos)
├── realsense_test.py       # Minimal frame diagnostic
├── live_view.py            # Existing app — NOT modified
├── studio_main.py          # Existing app — NOT modified
└── winvenv310/             # Python 3.10 venv with all dependencies
```
