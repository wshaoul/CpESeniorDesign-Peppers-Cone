# Pepper's Cone Studio for macOS

This folder is a standalone Mac application. It does not import or modify the
working Windows application in `Interface_updated`.

## First run

In Terminal:

```bash
cd Interface_mac
chmod +x setup_and_run.command
./setup_and_run.command
```

The first run creates `Interface_mac/macenv` and installs the tested dependency
versions. Later launches can use the same command without reinstalling changed
packages.

Older `.venv` installations are migrated automatically. The launcher clears
macOS hidden-file flags from the environment because Qt otherwise skips its
`cocoa` platform plugin.

When macOS asks, allow camera access. If permission was denied earlier, enable
Terminal or Python under **System Settings → Privacy & Security → Camera**.

## Camera selection

Camera index `0` is normally the built-in camera. A RealSense color sensor may
appear as another AVFoundation camera index, so try `1`, `2`, and so on. Intel
does not currently publish a `pyrealsense2` wheel for macOS, so depth frames are
not enabled in this Mac build; the Windows application remains unchanged and
retains its RealSense color/depth implementation.

Press **Q** or **Escape** to close a fullscreen cone display.
