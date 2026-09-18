# Mac prototype verification

Date: September 18 2026. Platform: Apple M4 Mac, arm64, macOS 26.3.
Branch: Sebastian-Garcia. Changes remain uncommitted and unpushed pending the
physical TV/cone test.

Verified on the Mac:

- New studio window and camera/test-pattern previews open.
- Orientation card and alignment rings render without camera input.
- Fullscreen window opens on the selected built-in display and closes cleanly.
- Actual camera frames are 1920×1080 with a 1920×1080 request.
- Source changes restart the worker and release the previous camera.
- Camera reader and processor stop without leaving the camera active in the
  completed GUI tests.
- Unit tests check sector population, fourfold repeat placement, mapping bounds,
  black preservation, aspect-ratio fitting, mirroring, radial inversion, cached
  maps, segmentation masking, and worker startup/error/shutdown.
- `Interface_updated` has no file differences from the original UI merge
  commit `6ac5dde`. No Windows files were changed.

Latest short GUI performance sample, background removal enabled:

| Render size | Processed rate | Latest frame processing time |
| --- | --- | --- |
| 1280×720 | 29.9 fps | 16 ms |
| 1920×1080 | 22.3 fps | 61 ms |
| 3024×1964 built-in display match | 9.1 fps | 91 ms |

Another short run gave 24.3 fps at HD and 10.1 fps at display-matched size.
Rates use a rolling window; times are single-frame samples. These brief runs
are not sustained benchmarks. Rates vary with other work on the Mac. They do
not measure TV presentation rate, camera-to-reflection delay, or optical detail.
The interface warns when processing is below 24 fps. Larger output is a quality
versus motion tradeoff, not an automatic recommendation to use 4K.

Not verified: external TV fullscreen routing, physical cone geometry, reflected
orientation, all-angle readability, duplicates, full-path latency, sustained
ten-minute performance, LED fan integration, or projector integration. Follow
`physical_test_plan.md` before claiming optical improvement or pushing.

The launcher must continue clearing hidden flags in `macenv`; otherwise Qt can
fail to discover Cocoa even when its plugin exists. One direct-interpreter test
reproduced that startup error before applying the launcher's existing fix.

## Simplified interface check

The default screen now shows display choice, picture quality, background removal,
and Start camera, Show on TV, and Stop. Smoother motion is the default. Advanced
settings starts collapsed and contains fitting, layouts, camera resolution,
test patterns, and technical diagnostics. Added interface tests confirm that
Show on TV stays disabled until the first frame and that advanced controls still
affect projection settings. A native Mac GUI check also verified camera capture,
the advanced panel's scrolling layout, fullscreen show/hide, and clean shutdown.
No Windows files were changed. Physical cone acceptance is still pending.
