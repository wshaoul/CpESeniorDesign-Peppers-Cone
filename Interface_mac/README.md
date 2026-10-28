# Pepper's Cone for Mac

We added Mac support in this folder. The Windows app stays in `Interface_updated`
and runs separately.

## Run the app

From the repository folder, open Terminal and run:

```bash
cd Interface_mac
./setup_and_run.command
```

The launcher sets up the Python environment and installs the required packages.
Allow camera access when macOS asks. If needed, check **System Settings →
Privacy & Security → Camera**.

## Use the live display

1. Connect the TV and set it as an extended display, not a mirrored display.
2. Choose the TV under **Show on**. Click **Find my TV** if it isn't listed.
3. Click **Start camera**, then **Show on TV**.
4. Press **Escape** or **Q** to close the TV image. Click **Stop** to stop capture.

The live tab shows your camera and the cone output side by side. It includes
background removal and output-resolution options. **720p output** is the
default; the higher-resolution options may run more slowly. These options change
the cone output resolution, not the camera's captured detail.

## Advanced settings

Use these to select a camera, request a camera resolution, adjust the cone fit,
change reflection direction, or choose four, six, or eight repeated views.
**Alignment rings** and **Orientation test card** help with setup. Click
**Save cone fit** to keep your alignment settings on this Mac.

Camera `0` is usually the built-in camera. Try another number for a USB camera.
This Mac app does not support RealSense depth capture.

## Test a real 3D model

Open **3D Model** to render a cube or sphere with perspective. Rotate the viewing
angle or enable automatic rotation. **Four different angles** renders front,
right, back, and left views from the geometry instead of repeating one image.
You can also load an image as the model texture and send the experimental cone
layout fullscreen to the TV.

Choose **Tiger demo** to test an articulated model with walking and breathing
motion. A saved front-audience profile is applied to each animation frame.

### Calibrate the front audience view

1. Connect the TV as an extended display and place the iPhone at the audience
   position using Continuity Camera.
2. In **3D Model**, press **Calibrate cone for front audience**, select the TV,
   and crop the camera view to the reflection inside the cone.
3. Capture the patterns and compare the corrected and uncorrected grids.
4. Save the map only if the corrected grid looks better, then test the cube and
   **Tiger demo**.

Use [LAB_CALIBRATION_TEST.md](LAB_CALIBRATION_TEST.md) for the full lab setup,
pass criteria, and results checklist.

With the environment already installed, open calibration directly:

```bash
cd Interface_mac
macenv/bin/python app.py --calibrate
```

## What still needs testing

The layouts repeat the same camera view; they aren't true 360° video. More
repeated views don't guarantee a better image. The external TV and physical cone
still need testing for sharpness, alignment, and visibility from different angles.

Start with the [lab calibration guide](LAB_CALIBRATION_TEST.md), then use the
[physical test plan](research/physical_test_plan.md) for multi-angle testing.

## Run the tests

```bash
QT_QPA_PLATFORM=offscreen macenv/bin/python -m unittest test_processing test_projection test_live_worker test_live_ui -v
```
