# Improving the Live Circular Cone Display

Engineering study for the Pepper's Cone project

September 18 2026

## Abstract

We recommend testing a higher quality Mac television output before adding a rotating LED display or a projector. The immediate objective is a recognizable live person around a smooth circular cone. A separate objective is correct perspective and depth for several viewers at once. The first can be explored with repeated camera views and careful fitting. The second requires directional optical delivery and suitable three dimensional content; repeating a webcam image cannot supply it.

This study evaluates the existing horizontal television and transparent cone, a rotating LED fan at the center, and projection alternatives. Our proposed Mac prototype preserves camera aspect ratio, uses a larger output canvas, maintains digital black, and exposes repeat count and fitting controls. It is an experimental display layout, not a calibrated optical reconstruction. Hardware purchases should follow measurements of angle coverage, distortion, latency, and contrast.

## What the existing system produces

The television emits a flat image. The cone reflects part of that light toward an observer while transmitting the scene behind it. The observer perceives a virtual image rather than light emitted by an object suspended in the center. Curvature changes the reflected rays across the cone. A polar image on the television is therefore a useful starting layout, but it is not automatically the inverse of the real optical distortion.

Luo, Lawrence, and Seitz demonstrated Pepper's Cone using a tablet, cone, rendered scene, and distortion calibration. Their project uses orientation to update the view. Their paper discusses calibrated viewing position and treats broader viewer movement and multiple viewers as further work [1, 2]. This supports calibration as a next step, not a claim that any cone with four pictures becomes a universal 360 degree display.

## Requirements that must remain separate

All around visibility means that someone can walk around and find a readable image. Simultaneous visibility means that several people can see it at once. Correct motion parallax means that moving sideways reveals the corresponding side of an object. Binocular depth means that the two eyes receive appropriate distinct views. A display can satisfy visibility without satisfying either depth requirement. We should report each separately.

[PAGE]

## Limits of the television and cone

One camera measures one visible surface view. It does not capture an accurate back view, hidden clothing, or surfaces blocked by the subject. A generated side view can be an approximation, but it must not be described as measured live geometry. Multiple synchronized cameras or a reconstructed model can improve content, but the cone still needs to deliver the correct view to each observer.

A smooth transparent reflector does not inherently isolate angular view channels. More repeated sectors may place useful light in more azimuths, but observers may see duplicate or overlapping images, seams, narrowing, and orientation changes. The result depends on cone geometry and eye position. Six or eight sectors can be worse than four. A larger repeat count is a test variable, not an improvement guarantee.

## Quality limits independent of the webcam

A high resolution webcam cannot recover detail discarded before display. The previous Mac live path reduced the source to 400 by 400 pixels, then warped it into an 800 pixel square. Its square resize also changed the shape of landscape camera images. The new live path samples from the original source and lets us choose HD, a lower load setting, or the selected display's reported pixel size, capped at 3840 by 2160.

Only the active cone footprint uses those pixels. On a wide television, the circular footprint is limited by the shorter dimension. Repetition divides the angular sampling among views. As a nominal geometric example, at a radius of 400 pixels the circumference is about 2513 pixels. Four equal sectors have about 628 pixels of arc each; eight have about 314. These are source-plane arc lengths, not measured reflected resolution. Curvature, viewing distance, source cropping, and optical blur can reduce visible detail further.

Digital black also matters. The old enhancement used an absolute-value operation with a negative offset, so zero-valued background became gray. The new live path uses a gain lookup table that leaves zero at zero. This removes a software glow source but cannot make a television panel emit no light. Background segmentation can lose hair, hands, translucent objects, or dark clothing. Use a controlled dark background as a comparison.

## What the Mac prototype changes

The live studio provides four repeated views by default, six and eight experimental repetitions, and a single 200 degree arc for comparison. It offers a centered portrait crop or whole-frame fitting without aspect distortion, rotation, diameter, inner radius, center position, subject size, reflection mirroring, radial inversion, and a small sector gap. Orientation and ring patterns run without a camera. The television window contains only the image on black, while the Mac shows controls and previews.

Capture and processing run outside the interface thread. Only the newest completed result is retained for the interface; there is no growing application display queue. Camera-driver buffering can still add latency. The status reports the actual source dimensions, render dimensions, output rate, and processing time. Neither requested camera resolution nor rendered resolution proves the television's physical resolution or the quality seen through the cone.

[PAGE]

## A rotating LED fan in the center

We interpret the proposed fan as a commercial rotating LED persistence of vision display. If an ordinary air fan was intended, it supplies no image-forming mechanism by itself. Adding fog or another scattering medium would be a different system with containment, air quality, and optical requirements.

A rotating LED display lights moving emitters at controlled positions. Their swept paths create a visible image surface. HYPERVSN describes a four-ray LED display and documents HDMI and streaming options on some SmartV products [3, 4]. This establishes that live input exists on particular hardware. It does not establish compatibility with every low-cost fan or support for our Mac output without format negotiation.

## What a single fan could improve

An emissive display can provide a strong floating-looking image without the television reflection path. It may be attractive for a front-facing live demonstration. Its circular area can suit a centered portrait. We should compare a protected standalone fan against the television and cone before combining them. A standalone test tells us whether the fan improves the image independently of the cone.

## Why the center position is uncertain

A conventional fan produces an image in a swept plane. Rotation in that plane does not distribute correct side and back views around a person. Seen nearly edge-on, its image area is strongly foreshortened. HYPERVSN advertises a wide 178 degree viewing angle for SmartV Solo; that is a vendor viewing specification, not a promise of 360 degree perspective [4].

Placing the fan inside the cone adds another optical path. Observers might see the emitting disc directly, reflections of it, the motor, and the protective housing. The fan and housing can obstruct television light and change the cone's reflected image. A horizontal disc may be a replacement source plane; a vertical disc may favor a front-facing direct view. Neither arrangement is validated for this cone. Reflections, brightness, and viewing geometry require measurement or ray tracing before a combined design can be recommended.

## Safety and integration conditions

The transparent cone is not a certified blade guard. The rotor's apparent invisibility while spinning does not remove its swept volume. Use the manufacturer's approved enclosure, mount, clearance, ventilation, and shutdown procedure. HYPERVSN's public instructions require a protective barrier or enclosure for accessible installations, and instruct users to unplug and wait for a complete stop before maintenance [4]. These are product-specific instructions; the actual selected device's manual controls its installation.

Do not place an unguarded rotor on the television or inside a thin cone for a public demonstration. Do not assume a housing is safe merely because it fits. Obtain manufacturer approval for the proposed orientation and environment. Before purchase, confirm live Mac input, accepted resolution, full-path latency, image diameter, noise, mounting load, enclosure size, replacement parts, and operation without unnecessary cloud dependence. No exact device or budget has been selected.

[PAGE]

## Projector configurations

A conventional projector sends spatially modulated light toward an image-forming surface. A clean beam in clear air is not a bright floating image. A smooth transparent cone primarily transmits and specularly reflects light; it is not a conventional diffuse projection screen. Projecting onto it directly can produce localized reflections and stray light rather than a uniformly visible image. A coating could improve scattering, but would change transparency and the original reflection illusion.

The most conservative projection experiment replaces the television with a stationary rear-projection source plane beneath the cone. The projector forms the required image on an appropriate screen; the cone then reflects that image. This may offer a larger source footprint or different packaging, but retains the cone's view-selection limits. It adds throw distance, focus, image orientation, cooling, and screen hotspot considerations. Its benefit must be compared with the existing television using the same image scale and room lighting.

Projector brightness alone is insufficient. Stray illumination and source black level can make the virtual background visible. Epson's guidance discusses contrast and ambient illumination [5]. For our installation, measure the whole reflected system rather than relying on a projector's headline contrast ratio. Keep projector light away from viewers' eyes and follow the chosen product's mounting, ventilation, and optical safety instructions.

## Systems that address angular views

USC's interactive 360 degree light field display used a high speed projector, a spinning mirror, an anisotropic diffuser, and synchronized view rendering [6]. It delivered different horizontal views to multiple observers. This is evidence that a rotating optical system can address angular delivery. It is not evidence that a normal projector plus a normal LED fan can do the same. Its timing, optics, containment, and control requirements make it a substantially different project.

An MIT thesis combined a cone with a radial parallax barrier and reported a light field with approximately 40 degrees of field of view [7]. This supports investigation of directional optics, while also showing that adding them need not yield full-circle coverage. Barriers sacrifice light and spatial sampling to separate views; alignment and crosstalk become important design variables.

## Comparison of candidate paths

| Configuration | Likely use | Remaining limitation |
| --- | --- | --- |
| TV and repeated sectors | First angle-coverage trial | Duplicate views and optical distortion |
| TV and calibrated warp | Improve one viewing region | Several viewers may need incompatible warps |
| Protected standalone LED fan | Bright front-facing live image | Planar source and device-specific live input |
| LED fan inside cone | Exploratory optical bench test | Occlusion and reflections not validated |
| Rear projection source plane | Source size or packaging change | Cone view limits remain |
| Directional or synchronized optics | Correct angular-view research | Major optical and control redesign |

[PAGE]

## Validation before any push or hardware purchase

We should test the Mac prototype using the existing television and cone first. Physical acceptance remains open until that hardware is tested. Software tests can establish correct arrays, black backgrounds, and repeat placement, but cannot establish visibility through an unmeasured cone.

1. Connect the television as an extended display, not a mirrored desktop. Select it in the studio. Disable overscan if the television provides that option. Start with HD rendering, then compare the display-matched setting and Balanced mode. Record the Mac's display report and the television's own input information if available.
2. Display alignment rings with the cone stationary. Fit the center and outer footprint to the real installation. Use the orientation card to choose mirroring and head direction. Save the fit. Keep the room lighting, television brightness, source size, and viewer distance fixed during comparisons.
3. Mark twelve positions at 30 degree intervals around the cone. At each position, check the center of the sector and the transition to the next sector. Record whether one complete upright subject is visible, whether left and right are correct, and whether duplicates or gaps appear. Include the cone's physical seam. Repeat at several relevant eye heights.
4. Compare the single arc, four, six, and eight repetitions. Test a person standing and moving, hands near the edge, dark clothing, and hair. Compare segmentation on and off with a dark physical background. Do not choose eight just because it places more copies on the screen.
5. Test two viewers at opposing positions at the same time. If one setting improves one viewer but distorts the other, report that conflict. A single-observer walkaround does not prove simultaneous quality.
6. Measure full-path delay using an external high frame rate camera that sees the subject and reflected display together during a sudden movement. The studio's processing time excludes exposure, camera buffering, display scanout, and optical observation. Run at least ten minutes to check stability and thermal performance.

## Proposed acceptance criteria

These are project targets, not measured results. Require one recognizable upright subject at all twelve sampled azimuths and selected eye heights, with no transition that loses the complete subject. Compare facial detail and duplicate severity against the single arc and current app. Target at least 24 output frames per second and no more than 150 milliseconds of full-path delay for conversation, then revise those thresholds if the intended interaction requires more. Record disagreements instead of averaging away bad angles. Passing sampled positions still does not prove every possible angle.

## Recommended development order

First, choose the best repeated layout from the physical test and retain the Windows code unchanged. Push the Mac changes only after the user confirms the reflected display is better. Second, measure cone dimensions, source footprint, viewer position, and distortion, then prototype a calibrated inverse warp for one viewing region. Head tracking is a possible later step for one moving observer, not a solution for everyone at once.

If brightness or packaging remains inadequate, borrow or demonstrate a protected live-input fan separately, or test a rear-projection source plane. A fan inside the cone remains an experimental option. If simultaneous correct side views are mandatory, define a new directional-display architecture and content capture plan before buying equipment. The distinction between a clearer illusion and a multi-view display should drive the decision.

[PAGE]

## References

Primary research and manufacturer documentation consulted September 18 2026. Product claims below are manufacturer specifications, not independent measurements of our system.

[1] Xuan Luo, Jason Lawrence, and Steven M Seitz. Pepper's Cone An Inexpensive Do It Yourself 3D Display. UIST 2017, pages 623 to 633. Author project and paper.
https://roxanneluo.github.io/PeppersCone.html

[2] Xuan Luo. Pepper's Cone Unity implementation and calibration guide.
https://github.com/roxanneluo/Pepper-s-Cone-Unity

[3] HYPERVSN. How HYPERVSN Works. Description of the rotating four-ray LED display.
https://hypervsn.com/how-it-works

[4] HYPERVSN. SmartV Solo Knowledge Base. Viewing specification, input modes, and installation instructions. Confirm these against the exact model and current manual before installation.
https://hypervsn.com/blog/knowledge_categories/smartv-solo

[5] Epson. Projector Guide Contrast Ratio. Contrast and ambient lighting considerations.
https://epson.com/projector-guide-how-to-buy-a-projector-contrast-ratio

[6] Andrew Jones, Ian McDowall, Hideshi Yamada, Mark Bolas, and Paul Debevec. Rendering for an Interactive 360 Degree Light Field Display. SIGGRAPH 2007. USC research project.
https://vgl.ict.usc.edu/Research/3DDisplay/

[7] Emily M Van Belleghem. 3 dimensional autostereoscopic displays with 4K televisions. MIT Master of Engineering thesis, 2018. Repository abstract describing a cone and radial parallax barrier with approximately 40 degree field of view.
https://dspace.mit.edu/entities/publication/c173db21-9c49-488b-94a5-e303c18b51c1

## Evidence status

Published prototypes establish that calibrated cone displays and specialized angular displays are possible. Manufacturer documentation establishes particular product features and safety conditions. The center-fan combination, projector substitution, and repeated-view angle coverage are engineering proposals for this installation. Their predicted benefits are not verified physical outcomes. The television model, cone dimensions, reflection properties, and final viewing geometry must be measured before a hardware design is finalized.
