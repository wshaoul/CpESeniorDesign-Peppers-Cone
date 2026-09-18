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
background removal and output-resolution options. **Smoother motion** is the
default; the higher-resolution options may run more slowly. These options change
the cone output resolution, not the camera's captured detail.

## Advanced settings

Use these to select a camera, request a camera resolution, adjust the cone fit,
change reflection direction, or choose four, six, or eight repeated views.
**Alignment rings** and **Orientation test card** help with setup. Click
**Save cone fit** to keep your alignment settings on this Mac.

Camera `0` is usually the built-in camera. Try another number for a USB camera.
This Mac app does not support RealSense depth capture.

## What still needs testing

The layouts repeat the same camera view; they aren't true 360° video. More
repeated views don't guarantee a better image. The external TV and physical cone
still need testing for sharpness, alignment, and visibility from different angles.

See the [test plan](research/physical_test_plan.md) and the
[short improvement paper](research/cone_display_improvement_paper.md).

## Run the tests

```bash
QT_QPA_PLATFORM=offscreen macenv/bin/python -m unittest test_processing test_projection test_live_worker test_live_ui -v
```
