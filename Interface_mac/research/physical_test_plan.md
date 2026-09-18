# Physical cone test

Status: not yet performed. Do not infer optical success from software tests.

Keep TV brightness, ambient lighting, viewer distance, cone placement, camera,
and subject framing fixed. Mark twelve viewer positions every 30°. Check sector
transitions as well as marked positions, and include the physical cone seam.
Test at relevant eye heights and with two simultaneous opposite viewers.

Compare the existing app, the new single arc, four repeats, six repeats, and
eight repeats. Compare HD and screen-matched rendering. Use Balanced to check
whether a lower processing load improves motion. Report segmentation failures
separately from optical duplication or distortion.

Record setup: cone diameter/height/material, TV input resolution if available,
Mac display report, actual camera resolution, render resolution, source framing,
layout, geometry/orientation settings, eye height and distance, and lighting.

| Angle | Layout | Full upright subject | Face detail 1–5 | Duplicates or gaps | Left/right correct | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 0° | | | | | | |
| 30° | | | | | | |
| 60° | | | | | | |
| 90° | | | | | | |
| 120° | | | | | | |
| 150° | | | | | | |
| 180° | | | | | | |
| 210° | | | | | | |
| 240° | | | | | | |
| 270° | | | | | | |
| 300° | | | | | | |
| 330° | | | | | | |

Run motion with hands, hair, and dark clothing. Record processed FPS and dropped
or stalled output. Measure total delay with a high-frame-rate camera seeing the
subject and reflected screen at once; processing milliseconds exclude capture
and display delay. Run a ten-minute stability check. Keep bad angles visible in
the report rather than averaging them away. Confirm the new reflected live view
is better before committing and pushing the Mac-only changes.
