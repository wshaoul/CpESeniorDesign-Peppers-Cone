# Review & Amendments — Wireless Component Plan (pi_wireless_hologram_plan.md)

This is a review of your `pi_wireless_hologram_plan.md`, checked against what's actually in `Interface_updated/` (the active codebase per `CLAUDE.md`) and against the team's existing `PepperCone_Research_and_Wireless_Test_Plan.pdf`. The overall shape of your plan (Phase 0 → 1 → 2 → 3, get SSH working, prove it with a manual image, then a real push pipeline, then streaming) is sound and I wouldn't restructure it. The changes below are things the codebase and your team's own prior research tell you that the generic plan couldn't have known.

**Nothing needed here:** Phase 0 (SSH, fixed IP) is fine as written — no codebase dependency, do it as-is.

---

## 1. What you're actually sending is not "an image" — it's the app's warp output

Your plan (Phases 1–3) treats the payload as generic images/video. In this codebase, the real content is the output of the cone-warp pipeline:

- Single-face app: `live_view.py`'s `_apply_warp()` → a `CANVAS_SIZE`×`CANVAS_SIZE` (800×800) BGR frame, already background-removed (MediaPipe `selfie_segmentation`) and already cone-warped via `build_cone_maps()`.
- Four-face circle app: `live_view_circle.py`'s `apply_four_cone_views()` → same idea, four 90°-rotated warps composited with `np.maximum`.

Two consequences:

- **Phase 1's test image should be square**, ideally an actual exported frame from `_apply_warp()`, not an arbitrary photo — you want to be testing the real payload shape/size from day one, not swapping it out later and rediscovering aspect issues.
- **The natural place to hook a "sender"** isn't a new capture path — it's inside `_fullscreen_tick()` (and the four-face equivalent), which already holds "the current frame, freshly warped, right now" right before it converts the array to a Tkinter `PhotoImage`. That's where you'd add a POST-to-Pi / WebSocket-push call in Phase 2, rather than building a second thing that re-derives the frame.
- The fullscreen tick already runs at `after(16, ...)` (~60 FPS). That's your real frame-availability rate on the laptop side — use it (or a deliberate fraction of it) as your Phase 3 target instead of picking a number from scratch.

---

## 2. The output canvas is square — your display pipeline must not stretch it

`build_cone_maps()` produces a fixed square canvas by mapping polar coordinates (angle/radius) from a center point — the geometry is aspect-sensitive. Stretching it to fill a 16:9 screen will misalign the reflection with the physical cone.

- **Phase 1:** `feh --fullscreen` on a non-square image may stretch by default — verify it isn't distorting the test image before calling this a success. If it is, look at `feh`'s fill;/geometry flags (or letterbox manually).
- **Phase 2:** the kiosk HTML page must render the incoming image/stream with `object-fit: contain` (letterboxed on black), never `cover` or an unconstrained `width:100%; height:100%`. Get this wrong and every later test will look "off" for a reason that has nothing to do with your networking work, which will cost you debugging time in the wrong place.

---

## 3. Confirm the Pi is actually driving the cone rig's display, not a separate wall TV

The plan's wording ("the Pi can display it on the TV") reads like a vertical wall-mounted screen. Pepper's Cone's physical rig (per the repo's README and `PepperCone_Physical_Improvement.pdf`) is a **horizontal display sitting beneath the cone reflector**, not a TV on a wall. Before wiring HDMI for real:

- Confirm which physical panel the Pi's HDMI output is actually going to.
- If it's a different panel/orientation than whatever screen `span_deg`/`rotate_deg` were tuned against (recall the convention: 270° = top of canvas, 90° = bottom, 0° = right, 180° = left), the existing warp tuning likely won't transfer as-is — budget time to retune those two sliders for the new screen's physical mounting, don't assume it'll just work.

---

## 4. Reconcile this plan with the team's own prior wireless decision

`PepperCone_Research_and_Wireless_Test_Plan.pdf` (already in the repo) documents the team's own earlier conclusions:

- *"Do not buy a Raspberry Pi until the laptop test works"* — a two-laptop (no Pi) wireless test was supposed to come first.
- Comparison order already agreed on: **MJPEG → H.264/WebRTC → NDI → direct HDMI**, targeting a stable 30 fps and under 150 ms latency (ideally under 100 ms).

Two things worth resolving, not just noting:

1. **Has the two-laptop baseline test happened?** If not, this Pi plan is running ahead of the team's documented decision. Either do that quick laptop-to-laptop comparison first (cheap, you already have the render engine), or explicitly tell the team you're superseding that decision by moving straight to Pi hardware — don't let it be an accidental divergence someone discovers later.
2. **Phase 3, option 1 in your plan ("VLC/ffmpeg over the network") is a different mechanism than what the team already picked to try first (MJPEG).** VLC/UDP means a second, unrelated player process outside the browser. Since Phase 2 already stands up a local web server + kiosk browser page, the natural first thing to test in Phase 3 is **an MJPEG stream over that same HTTP server** — e.g. a `/stream.mjpg` endpoint your existing kiosk `<img>` tag points at. It reuses the Phase 2 architecture instead of introducing a second playback mechanism, and it matches what your team already agreed to try first. Keep VLC/UDP as a fallback comparison, not the first thing you build.

---

## 5. The NDI section's framing doesn't match this codebase

Your plan suggests asking "whoever owns the render engine" if it "already speaks NDI," name-checking Unity/TouchDesigner/Resolume. Per this repo, the render engine **is** the Python/Tkinter/OpenCV app in `Interface_updated/` — it isn't one of those tools, and none of your current dependencies (`opencv-python`, `mediapipe`, `numpy`, `Pillow`) include NDI support. Adding NDI here means adding the NDI SDK and its Python bindings as a new native dependency, on top of the app's existing "degrades gracefully when optional deps are missing" design (RealSense, ttkbootstrap).

Given that Phase 3 already flags NDI-on-Pi as unverified, I'd weight it down further specifically for this codebase: MJPEG and WebRTC are both achievable with pure-Python/browser code and no new native SDK. Keep NDI as a "revisit if the team explicitly wants it" item, not a near-term path.

---

## 6. Small platform gotcha: your dev environment is Windows

This repo's tooling (bundled `winvenv310`, PowerShell-based setup docs) confirms your laptop-side work happens in PowerShell, not a Unix shell. One command in your plan needs adjusting:

- `curl -F "file=@image.png" http://raspberrypi.local:3000/upload` — in PowerShell, `curl` is aliased to `Invoke-WebRequest`, which doesn't understand `-F`. Either call `curl.exe` explicitly (Windows 10/11 ship a real one), or use `Invoke-WebRequest -Uri http://raspberrypi.local:3000/upload -Form @{file=Get-Item image.png}`.
- `scp` / `ssh` are fine as written — Windows 10/11 ship OpenSSH client by default.

---

## Summary of concrete edits to make in your plan

| Phase | Change |
|---|---|
| Phase 1 | Use a real square warped-frame export as the test image; verify `feh` isn't stretching it |
| Phase 2 | Hook the sender into `_fullscreen_tick()` in `live_view.py` (or the four-face equivalent), not a new capture path; kiosk page must use `object-fit: contain` |
| Phase 2 | Fix the `curl -F` command for PowerShell, or use `Invoke-WebRequest -Form` |
| Phase 3 | Swap option 1 from "VLC/ffmpeg over UDP" to "MJPEG over the existing Phase 2 HTTP server" as the first thing to try — matches the team's own prior decision and reuses your kiosk page |
| Phase 3 | Reframe the NDI discussion: the render engine is this repo's Python app, not Unity/TouchDesigner/Resolume — NDI means adding a new native SDK dependency, not just checking an existing one |
| General | Before going further, confirm with the team whether the two-laptop wireless test (from `PepperCone_Research_and_Wireless_Test_Plan.pdf`) already happened, since that doc explicitly said to hold off on Pi hardware until it did |
| General | Confirm the Pi's HDMI target is the horizontal cone-rig display (not a wall TV), and budget time to retune `span_deg`/`rotate_deg` if it's a different screen than what those were last tuned against |
