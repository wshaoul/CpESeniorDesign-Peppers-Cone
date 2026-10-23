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

Choose **Tiger demo** for a built-in articulated 3D calibration subject. It has
procedural stripes, different geometry from every viewing angle, and a simple
walking/breathing motion. Once a front-audience profile is saved, every tiger
animation frame is sent through that same measured correction before TV output.

The model is real 3D geometry. The quick cone layout is only a starting
approximation; the front-audience workflow below measures the physical setup and
builds a correction profile for the deformation caused by the reflector.

### Calibrate the front audience view

In **3D Model**, press **Calibrate cone for front audience**.

For an in-person team test with pass/fail criteria and the evidence to post on
the draft PR, follow [LAB_CALIBRATION_TEST.md](LAB_CALIBRATION_TEST.md).

1. Connect the TV as an extended display. Mount the iPhone at the audience's
   eye height and distance with its rear camera facing the cone.
2. Connect the iPhone using Continuity Camera (USB recommended). Use the same
   Apple Account on both devices, trust the Mac when asked, and turn off camera
   effects such as Center Stage. This camera connection still needs testing on
   your hardware. [Apple setup guide](https://support.apple.com/en-us/102546).
3. Start the camera. If it shows the Mac camera, stop it, try another camera
   number, and restart. Confirm the preview is the iPhone's view.
4. Select the TV and drag a tight box around the reflection inside the cone.
   Exclude the directly visible TV surface. Keep everything still.
5. Capture the patterns (23 at 720p, 24 at 1080p). Allow roughly 40 seconds at
   the default timing. The app waits for each pattern and takes three fresh
   samples. Cancel, Close, and Escape/Q on the TV stop the sequence.
6. Inspect **Distortion map**, then compare **Grid without correction** and
   **Grid with correction** in the camera preview. Look for straighter lines,
   square cells and a round circle. Coverage measures mapped area, not accuracy.
7. Press **Save and use this map** only after checking. It replaces the saved
   profile and selects calibrated output in the model tab. Close calibration,
   test the cube, then select **Tiger demo**.

New maps keep the camera crop's proportions to avoid stretching the model.
Use a dark background and dim lights. Moving the cone, TV or audience viewpoint
requires another calibration. Check nearby viewing positions separately.

With the environment already installed, open calibration directly:

```bash
cd Interface_mac
macenv/bin/python app.py --calibrate
```

This follows the capture, inverse mapping, and weighted Gaussian smoothing
approach summarized in Will's research notes and described in the original
[Pepper's Cone paper, pages 4–5](https://grail.cs.washington.edu/wp-content/uploads/2017/10/luo2017pca.pdf).
It is a prototype; software tests do not establish optical accuracy on the cone.

## What still needs testing

The layouts repeat the same camera view; they aren't true 360° video. More
repeated views don't guarantee a better image. The external TV and physical cone
still need testing for sharpness, alignment, and visibility from different angles.

See the [test plan](research/physical_test_plan.md) and the
[short improvement paper](research/cone_display_improvement_paper.md). Lab
testers should start with the shorter
[front-audience calibration guide](LAB_CALIBRATION_TEST.md).

## Run the tests

```bash
QT_QPA_PLATFORM=offscreen macenv/bin/python -m unittest test_processing test_projection test_live_worker test_live_ui -v
```
