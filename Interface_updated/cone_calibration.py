"""Structured-light calibration for a front-facing Pepper's Cone viewer."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


GREEN = np.array((0, 255, 0), np.uint8)
MAGENTA = np.array((255, 0, 255), np.uint8)


def default_profile_path():
    """Where the Calibrate tab saves the map and the Live tab looks for it."""
    root = Path(os.environ.get("APPDATA", Path.home())) / "PeppersCone"
    return root / "front_audience_calibration.npz"


def place_subject(image, zoom=1.0, mirror=False):
    """Scale a frame about its centre without changing its size.

    zoom < 1 shrinks the picture onto black; zoom > 1 enlarges it and crops the
    edges. mirror flips it left to right.
    """
    zoom = float(np.clip(zoom, 0.05, 4.0))
    if mirror:
        image = cv2.flip(image, 1)
    if abs(zoom - 1) < 1e-3:
        return image
    height, width = image.shape[:2]
    new_width, new_height = max(1, round(width * zoom)), max(1, round(height * zoom))
    scaled = cv2.resize(image, (new_width, new_height),
                        interpolation=cv2.INTER_AREA if zoom < 1 else cv2.INTER_LINEAR)
    placed = np.zeros_like(image)
    src_x, src_y = max(0, (new_width - width) // 2), max(0, (new_height - height) // 2)
    dst_x, dst_y = max(0, (width - new_width) // 2), max(0, (height - new_height) // 2)
    w, h = min(width, new_width), min(height, new_height)
    placed[dst_y:dst_y + h, dst_x:dst_x + w] = scaled[src_y:src_y + h, src_x:src_x + w]
    return placed


def verification_grid(width, height):
    """Square cells and a circle reveal curvature and unequal axis scaling."""
    image = np.zeros((height, width, 3), np.uint8)
    spacing = max(12, min(width, height) // 8)
    for x in range(width // 2 % spacing, width, spacing):
        cv2.line(image, (x, 0), (x, height-1), (180, 180, 180), 1)
    for y in range(height // 2 % spacing, height, spacing):
        cv2.line(image, (0, y), (width-1, y), (180, 180, 180), 1)
    cv2.circle(image, (width//2, height//2), min(width, height)//4,
               (0, 220, 255), 2, cv2.LINE_AA)
    return image


def _bit_count(size):
    return max(1, int(np.ceil(np.log2(max(2, size)))))


@dataclass(frozen=True)
class CalibrationSequence:
    """The two reference frames and Gray-code frames used for calibration."""

    width: int
    height: int

    @property
    def x_bits(self):
        return _bit_count(self.width)

    @property
    def y_bits(self):
        return _bit_count(self.height)

    def __len__(self):
        return 2 + self.x_bits + self.y_bits

    def label(self, index):
        if index == 0:
            return "green reference"
        if index == 1:
            return "magenta reference"
        if index < 2 + self.x_bits:
            return f"vertical code {index - 1}/{self.x_bits}"
        return f"horizontal code {index - 1 - self.x_bits}/{self.y_bits}"

    def frame(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index == 0:
            return np.full((self.height, self.width, 3), GREEN, np.uint8)
        if index == 1:
            return np.full((self.height, self.width, 3), MAGENTA, np.uint8)

        if index < 2 + self.x_bits:
            bit = self.x_bits - 1 - (index - 2)
            values = np.arange(self.width, dtype=np.uint32)
            code = values ^ (values >> 1)
            on = ((code >> bit) & 1).astype(bool)[None, :]
            on = np.broadcast_to(on, (self.height, self.width))
        else:
            bit = self.y_bits - 1 - (index - 2 - self.x_bits)
            values = np.arange(self.height, dtype=np.uint32)
            code = values ^ (values >> 1)
            on = ((code >> bit) & 1).astype(bool)[:, None]
            on = np.broadcast_to(on, (self.height, self.width))

        image = np.empty((self.height, self.width, 3), np.uint8)
        image[on] = GREEN
        image[~on] = MAGENTA
        return image


def _color_difference(frame):
    data = frame.astype(np.int16)
    return data[:, :, 1] - ((data[:, :, 0] + data[:, :, 2]) // 2)


def _gray_to_binary(gray):
    value = gray.copy()
    shift = 1
    while shift < 32:
        value ^= value >> shift
        shift <<= 1
    return value


def decode_sequence(frames, screen_size, roi_mask=None, min_contrast=12,
                    confidence=0.25, max_cell=32):
    """Decode camera observations into camera-to-screen correspondences.

    Frames must already be cropped to the reflected cone region. Returned x/y
    arrays contain TV pixel coordinates for every valid camera pixel.

    Every pixel is judged against its own green and magenta reference levels,
    so a dim reflection or an ambient colour cast shifts the decision point
    instead of flipping bits. A bit only counts when it lands at least
    ``confidence`` of the way from that midpoint toward green or magenta.

    The finest stripes are often blurred past what the camera can resolve
    (small reflection, compressed video, a screen scaled by Windows). Rather
    than discard a pixel whose last few bits are unreadable, the leading
    readable bits are decoded and the pixel is placed at the centre of the TV
    cell they identify. Cells wider than ``max_cell`` TV pixels are dropped,
    so the position error of every kept pixel is at most ``max_cell / 2``.
    """
    width, height = screen_size
    sequence = CalibrationSequence(width, height)
    if len(frames) != len(sequence):
        raise ValueError(f"Expected {len(sequence)} frames, received {len(frames)}")
    shape = frames[0].shape[:2]
    if any(frame.shape[:2] != shape for frame in frames):
        raise ValueError("Calibration frames must have matching dimensions")

    green_reference = _color_difference(frames[0]).astype(np.float32)
    magenta_reference = _color_difference(frames[1]).astype(np.float32)
    swing = green_reference - magenta_reference
    midpoint = (green_reference + magenta_reference) / 2
    valid = ((green_reference > midpoint) & (magenta_reference < midpoint)
             & (swing >= 2 * min_contrast))
    if roi_mask is not None:
        if roi_mask.shape != shape:
            raise ValueError("ROI mask must match the camera frame")
        valid &= roi_mask.astype(bool)
    half_swing = np.maximum(swing / 2, 1e-3)

    def decode_axis(first, bits, size):
        gray = np.zeros(shape, np.uint32)
        depth = np.zeros(shape, np.int32)
        readable = valid.copy()
        for index in range(bits):
            level = (_color_difference(frames[first + index]).astype(np.float32)
                     - midpoint) / half_swing
            readable &= np.abs(level) >= confidence
            gray = np.where(readable, (gray << 1) | (level > 0), gray)
            depth += readable
        # Gray-to-binary only looks at more significant bits, so the leading
        # bits alone give the exact top bits of the position.
        binary = _gray_to_binary(gray).astype(np.int64)
        unknown = (bits - depth).astype(np.int64)
        cell = np.left_shift(1, unknown)
        centre = np.left_shift(binary, unknown) + (cell >> 1)
        usable = (depth > 0) & (cell <= max_cell) & (centre >= 0) & (centre < size)
        return np.clip(centre, 0, size - 1).astype(np.int32), usable

    screen_x, usable_x = decode_axis(2, sequence.x_bits, width)
    screen_y, usable_y = decode_axis(2 + sequence.x_bits, sequence.y_bits, height)
    return screen_x, screen_y, valid & usable_x & usable_y


@dataclass
class CalibrationProfile:
    """Dense TV-to-virtual-view map stored in normalized source coordinates."""

    map_x: np.ndarray
    map_y: np.ndarray
    valid: np.ndarray
    sample_count: int = 0
    view_size: tuple | None = None
    # Where the calibration camera stood. The map is only exact from that viewpoint
    # (eye height matters most), so it is kept as a note; it does not change the warp.
    camera_height_in: float | None = None     # lens height above the floor, inches
    camera_distance_in: float | None = None   # horizontal distance to the cone's centre, inches
    _lookup: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def screen_size(self):
        return self.map_x.shape[1], self.map_x.shape[0]

    def viewpoint_text(self):
        """e.g. 'camera 48 in high, 13 ft from the cone'; '' when neither was recorded."""
        parts = []
        if self.camera_height_in is not None:
            parts.append(f"{self.camera_height_in:g} in high")
        if self.camera_distance_in is not None:
            parts.append(f"{self.camera_distance_in / 12:g} ft from the cone")
        return ("camera " + ", ".join(parts)) if parts else ""

    @property
    def coverage(self):
        return float(np.count_nonzero(self.valid) / self.valid.size)

    def _lookup_for(self, width, height):
        """Pixel lookup tables for a source image size, kept for the next frame."""
        key = (width, height)
        if key not in self._lookup:
            self._lookup.clear()  # live video keeps one size; don't accumulate old ones
            self._lookup[key] = (
                np.where(self.valid, self.map_x * max(1, width - 1), -1).astype(np.float32),
                np.where(self.valid, self.map_y * max(1, height - 1), -1).astype(np.float32))
        return self._lookup[key]

    def apply(self, image):
        if self.view_size is not None:
            view_width,view_height = self.view_size
            height,width = image.shape[:2]
            factor = min(view_width/width,view_height/height)
            fitted_width = max(1,round(width*factor))
            fitted_height = max(1,round(height*factor))
            fitted = np.zeros((view_height,view_width,3),np.uint8)
            left,top = (view_width-fitted_width)//2,(view_height-fitted_height)//2
            fitted[top:top+fitted_height,left:left+fitted_width] = cv2.resize(
                image,(fitted_width,fitted_height),interpolation=cv2.INTER_LINEAR)
            image = fitted
        height, width = image.shape[:2]
        map_x, map_y = self._lookup_for(width, height)
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def visualization(self):
        """Red = source x, green = source y; black = no supported mapping."""
        image = np.zeros((*self.valid.shape, 3), np.uint8)
        image[:, :, 2] = np.clip(self.map_x * 255, 0, 255).astype(np.uint8)
        image[:, :, 1] = np.clip(self.map_y * 255, 0, 255).astype(np.uint8)
        image[~self.valid] = 0
        return image

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, map_x=self.map_x.astype(np.float32),
                            map_y=self.map_y.astype(np.float32),
                            valid=self.valid.astype(np.uint8),
                            sample_count=np.int64(self.sample_count), version=np.int64(1),
                            view_size=np.asarray(self.view_size or (0,0),dtype=np.int32),
                            # NaN = not recorded. Older readers ignore these extra keys.
                            camera_height_in=np.float64(np.nan if self.camera_height_in is None
                                                        else self.camera_height_in),
                            camera_distance_in=np.float64(np.nan if self.camera_distance_in is None
                                                          else self.camera_distance_in))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != 1:
                raise ValueError("Unsupported calibration profile version")
            view_size = tuple(int(value) for value in data["view_size"]) if "view_size" in data else None
            if view_size == (0,0):
                view_size = None

            def optional(key):
                if key not in data:
                    return None  # profile saved before viewpoints were recorded
                value = float(data[key])
                return None if np.isnan(value) else value

            return cls(data["map_x"].astype(np.float32),
                       data["map_y"].astype(np.float32),
                       data["valid"].astype(bool), int(data["sample_count"]),view_size,
                       optional("camera_height_in"), optional("camera_distance_in"))


def refine_samples(screen_x, screen_y, valid, sigma=3.0, tolerance=48.0, max_shift=16.0):
    """Recover sub-cell positions from coarse decodes.

    A pixel whose finest bits were unreadable is only known to lie somewhere in a
    cell of TV pixels, but the real mapping changes smoothly from one camera pixel
    to the next. Averaging a pixel's decoded position with its neighbours turns the
    staircase of cell centres back into a ramp (the move is limited to ``max_shift``
    so it never leaves the cell it was decoded in). Pixels that disagree with their
    neighbourhood by more than ``tolerance`` TV pixels are wrong decodes or the
    cone seam and are dropped. Returns (x, y, kept).
    """
    size = int(sigma * 6) | 1
    sx = screen_x.astype(np.float32)
    sy = screen_y.astype(np.float32)
    keep = valid.copy()
    for _ in range(3):
        weight = keep.astype(np.float32)
        total = np.maximum(cv2.GaussianBlur(weight, (size, size), sigma, borderType=cv2.BORDER_CONSTANT), 1e-6)
        mean_x = cv2.GaussianBlur(sx * weight, (size, size), sigma, borderType=cv2.BORDER_CONSTANT) / total
        mean_y = cv2.GaussianBlur(sy * weight, (size, size), sigma, borderType=cv2.BORDER_CONSTANT) / total
        keep = (valid & (total > 0.05)
                & (np.abs(sx - mean_x) <= tolerance) & (np.abs(sy - mean_y) <= tolerance))
    weight = keep.astype(np.float32)
    total = cv2.GaussianBlur(weight, (size, size), sigma, borderType=cv2.BORDER_CONSTANT)
    safe = np.maximum(total, 1e-6)
    mean_x = cv2.GaussianBlur(sx * weight, (size, size), sigma, borderType=cv2.BORDER_CONSTANT) / safe
    mean_y = cv2.GaussianBlur(sy * weight, (size, size), sigma, borderType=cv2.BORDER_CONSTANT) / safe
    # Next to the edge of the data the average leans inward (a ramp has more of it on one
    # side); only pixels with nearly their whole neighbourhood present are refined.
    trusted = total >= 0.9
    out_x = np.where(trusted, sx + np.clip(mean_x - sx, -max_shift, max_shift), sx)
    out_y = np.where(trusted, sy + np.clip(mean_y - sy, -max_shift, max_shift), sy)
    return out_x, out_y, keep


def _tidy_map(dense, map_x, map_y, sigma, min_piece=0.10):
    """Drop stray fragments, fill pinholes, round the edge, and smooth the values."""
    pieces, labels, stats, _ = cv2.connectedComponentsWithStats(dense.astype(np.uint8), connectivity=8)
    if pieces > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        for index, area in enumerate(areas, start=1):
            if area < min_piece * areas.max():
                dense = dense & (labels != index)
    data = dense.astype(np.float32)

    mask = cv2.morphologyEx(dense.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    mask = cv2.GaussianBlur(mask.astype(np.float32), (13, 13), 4) > 0.5   # rounds the stair-stepped edge

    if sigma >= 0.5:
        size = int(sigma * 6) | 1
        edge = cv2.BORDER_CONSTANT  # don't invent mirrored data past the screen edge
        total = cv2.GaussianBlur(data, (size, size), sigma, borderType=edge)
        safe = np.maximum(total, 1e-6)
        map_x = cv2.GaussianBlur(map_x * data, (size, size), sigma, borderType=edge) / safe
        map_y = cv2.GaussianBlur(map_y * data, (size, size), sigma, borderType=edge) / safe
        mask &= total > 1e-3
    else:
        # Pinholes take their value from the data around them.
        total = np.maximum(cv2.GaussianBlur(data, (9, 9), 2), 1e-6)
        map_x = np.where(dense, map_x, cv2.GaussianBlur(map_x * data, (9, 9), 2) / total)
        map_y = np.where(dense, map_y, cv2.GaussianBlur(map_y * data, (9, 9), 2) / total)
    map_x = np.where(mask, map_x, 0).astype(np.float32)
    map_y = np.where(mask, map_y, 0).astype(np.float32)
    return mask, map_x, map_y


def build_profile(screen_x, screen_y, valid, screen_size, kernel=43, sigma=7, smooth=8.0):
    """Invert sparse camera-to-TV samples and smooth them into a dense map.

    ``smooth`` (TV pixels on a 1382-wide screen, scaled to the real width) sets how
    much the finished map is smoothed; 8 was the best fit to the decoded samples,
    and the result gets sub-cell positions, no stray fragments and a rounded edge.
    ``smooth=0`` skips all of that and gives the plain map from the paper.
    """
    width, height = screen_size
    if screen_x.shape != screen_y.shape or screen_x.shape != valid.shape:
        raise ValueError("Decoded maps and validity mask must have matching shapes")
    camera_height, camera_width = valid.shape
    if smooth > 0:
        screen_x, screen_y, valid = refine_samples(screen_x, screen_y, valid)
    ys, xs = np.nonzero(valid)
    if len(xs) < 16:
        raise ValueError("Not enough reflected calibration pixels were detected")
    qx = screen_x[ys, xs].astype(np.float32)
    qy = screen_y[ys, xs].astype(np.float32)
    inside = (qx >= 0) & (qx <= width - 1) & (qy >= 0) & (qy <= height - 1)
    xs, ys, qx, qy = xs[inside], ys[inside], qx[inside], qy[inside]

    weight = np.zeros((height, width), np.float32)
    sum_x = np.zeros_like(weight)
    sum_y = np.zeros_like(weight)
    normalized_x = xs.astype(np.float32) / max(1, camera_width - 1)
    normalized_y = ys.astype(np.float32) / max(1, camera_height - 1)
    # Share each sample between the four nearest TV pixels, so a position between
    # pixels isn't rounded away (whole-pixel positions land entirely on one pixel).
    x0, y0 = np.floor(qx).astype(np.int64), np.floor(qy).astype(np.int64)
    frac_x, frac_y = qx - x0, qy - y0
    for step_x, step_y, share in ((0, 0, (1 - frac_x) * (1 - frac_y)), (1, 0, frac_x * (1 - frac_y)),
                                  (0, 1, (1 - frac_x) * frac_y), (1, 1, frac_x * frac_y)):
        column = np.clip(x0 + step_x, 0, width - 1)
        row = np.clip(y0 + step_y, 0, height - 1)
        np.add.at(weight, (row, column), share)
        np.add.at(sum_x, (row, column), share * normalized_x)
        np.add.at(sum_y, (row, column), share * normalized_y)

    kernel = max(3, int(kernel) | 1)
    blurred_weight = cv2.GaussianBlur(weight, (kernel, kernel), sigma)
    blurred_x = cv2.GaussianBlur(sum_x, (kernel, kernel), sigma)
    blurred_y = cv2.GaussianBlur(sum_y, (kernel, kernel), sigma)
    # A tenth or so of one sample's peak keeps round footprints; the old 1e-6 left
    # every isolated sample as a hard 43-pixel square.
    dense = blurred_weight > (1e-6 if smooth <= 0 else 0.15 / (2 * np.pi * sigma ** 2))
    map_x = np.zeros_like(weight)
    map_y = np.zeros_like(weight)
    map_x[dense] = blurred_x[dense] / blurred_weight[dense]
    map_y[dense] = blurred_y[dense] / blurred_weight[dense]
    if smooth > 0:
        dense, map_x, map_y = _tidy_map(dense, map_x, map_y, smooth * width / 1382.0)
    return CalibrationProfile(map_x, map_y, dense, int(len(xs)),
                              (camera_width, camera_height))
