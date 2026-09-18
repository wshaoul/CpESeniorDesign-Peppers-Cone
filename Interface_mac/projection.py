"""Experimental repeated-view cone mapping, not a calibrated 360 light field."""
from dataclasses import dataclass

import cv2
import numpy as np

from processing import make_segmenter


@dataclass(frozen=True)
class ProjectionSettings:
    width: int = 1920
    height: int = 1080
    views: int = 4
    diameter: float = .94
    inner: float = .12
    rotation: float = 270
    center_x: float = .5
    center_y: float = .5
    zoom: float = .85
    gap: float = 2
    mirror: bool = True
    invert: bool = False
    gain: float = 1.15
    remove_background: bool = True
    portrait_crop: bool = True


def build_maps(source_shape, settings):
    """Sample the original camera aspect ratio directly into a TV-sized annulus.

    Each sector repeats the same frontal image. The polar mapping is a fitting
    aid, not an inverse optical calibration of an arbitrary cone.
    """
    s = settings
    if not (320 <= s.width <= 3840 and 240 <= s.height <= 2160):
        raise ValueError("Output must be between 320x240 and 3840x2160")
    if s.views not in (1, 4, 6, 8):
        raise ValueError("Supported view counts are 1, 4, 6 and 8")
    if not (0 <= s.inner < .9 and .1 <= s.diameter <= 1 and .2 <= s.zoom <= 1.5):
        raise ValueError("Invalid cone geometry")
    h, w = source_shape[:2]
    yy, xx = np.ogrid[:s.height, :s.width]
    dx = xx.astype(np.float32) - s.center_x * s.width
    dy = yy.astype(np.float32) - s.center_y * s.height
    radius = np.hypot(dx, dy)
    outer = min(s.width, s.height) * .5 * s.diameter
    inner = outer * s.inner
    angle = (np.arctan2(dy, dx) - np.deg2rad(s.rotation) + np.pi) % (2*np.pi) - np.pi
    span = np.deg2rad(200 if s.views == 1 else 360 / s.views)
    local = angle if s.views == 1 else (angle + span/2) % span - span/2
    active_span = span - np.deg2rad(min(max(s.gap, 0), np.rad2deg(span)*.2))
    valid = (radius >= inner) & (radius <= outer) & (np.abs(local) <= active_span/2)
    u = local / active_span + .5
    v = 1 - (radius - inner) / (outer - inner)
    if s.invert:
        v = 1-v
    # Crop the central square for a larger portrait, or fit the whole source.
    # Both choices use equal x and y scale, so people are never squashed.
    side = (min(h,w) if s.portrait_crop else max(h,w)) / s.zoom
    mx = (u-.5) * side + (w-1)/2
    my = (v-.5) * side + (h-1)/2
    if s.mirror:
        mx = (w-1)-mx
    if s.portrait_crop:
        crop = min(h,w)
        valid &= (mx >= (w-crop)/2) & (mx <= (w+crop)/2-1) & (my >= (h-crop)/2) & (my <= (h+crop)/2-1)
    return np.where(valid, mx, -1).astype(np.float32), np.where(valid, my, -1).astype(np.float32)


def test_card():
    image = np.zeros((720, 720, 3), np.uint8)
    cv2.circle(image, (360, 145), 65, (230, 230, 230), -1)
    cv2.rectangle(image, (290, 220), (430, 510), (230, 230, 230), -1)
    cv2.line(image, (305, 495), (270, 650), (230, 230, 230), 35)
    cv2.line(image, (415, 495), (450, 650), (230, 230, 230), 35)
    cv2.arrowedLine(image, (110, 340), (250, 340), (0, 180, 255), 12)
    cv2.putText(image, "LEFT", (40, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,180,255), 3)
    cv2.putText(image, "RIGHT", (470, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,180,0), 3)
    cv2.putText(image, "HEAD", (300, 45), cv2.FONT_HERSHEY_SIMPLEX, .9, (255,255,255), 2)
    return image


def alignment_pattern(s):
    image = np.zeros((s.height, s.width, 3), np.uint8)
    center = (round(s.center_x*s.width), round(s.center_y*s.height))
    radius = round(min(s.width,s.height)*.5*s.diameter)
    for fraction in (s.inner, (1+s.inner)/2, 1):
        cv2.circle(image, center, round(radius*fraction), (200,200,200), 2)
    for i in range(s.views):
        a = np.deg2rad(s.rotation + i*360/s.views)
        end = (round(center[0]+radius*np.cos(a)), round(center[1]+radius*np.sin(a)))
        cv2.line(image, center, end, (0,180,255), 2)
        label = (round(center[0]+radius*.7*np.cos(a)), round(center[1]+radius*.7*np.sin(a)))
        cv2.putText(image, str(i+1), label, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return image


class LiveProjector:
    def __init__(self):
        self.segmenter = make_segmenter()
        self.key = None
        self.maps = None
        self.alpha_maps = None

    def process(self, frame, settings, segment=True):
        alpha = None
        if segment and settings.remove_background and self.segmenter is not None:
            h,w = frame.shape[:2]
            small = cv2.resize(frame, (round(w*min(1,384/w)), round(h*min(1,384/w))))
            result = self.segmenter.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            alpha = np.clip((result.segmentation_mask-.25)/.35, 0, 1)
        key = (frame.shape[:2], settings)
        if key != self.key:
            self.maps = build_maps(frame.shape, settings)
            self.alpha_maps = None
            self.key = key
        warped = cv2.remap(frame, *self.maps, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        # Apply source-sized segmentation directly in output coordinates rather
        # than expanding a float mask over every full-resolution camera pixel.
        if alpha is not None:
            ah,aw = alpha.shape
            if self.alpha_maps is None:
                self.alpha_maps = (self.maps[0]*(aw-1)/max(1,w-1),self.maps[1]*(ah-1)/max(1,h-1))
            output_alpha = cv2.remap(alpha,*self.alpha_maps,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            # OpenCV performs the mixed-type multiply without allocating two
            # full RGB float temporaries, important for HD and 4K output.
            warped = cv2.multiply(warped,cv2.merge((output_alpha,output_alpha,output_alpha)),dtype=cv2.CV_8U)
        # Unlike convertScaleAbs with a negative bias, zero stays truly black.
        lut = np.clip(np.arange(256)*settings.gain, 0, 255).astype(np.uint8)
        return cv2.LUT(warped,lut)

    def close(self):
        if self.segmenter is not None:
            self.segmenter.close()
