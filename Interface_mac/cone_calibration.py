"""Structured-light calibration for a front-facing Pepper's Cone viewer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


GREEN = np.array((0, 255, 0), np.uint8)
MAGENTA = np.array((255, 0, 255), np.uint8)


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


def decode_sequence(frames, screen_size, roi_mask=None, min_contrast=12):
    """Decode camera observations into camera-to-screen correspondences.

    Frames must already be cropped to the reflected cone region. Returned x/y
    arrays contain TV pixel coordinates for every valid camera pixel.
    """
    width, height = screen_size
    sequence = CalibrationSequence(width, height)
    if len(frames) != len(sequence):
        raise ValueError(f"Expected {len(sequence)} frames, received {len(frames)}")
    shape = frames[0].shape[:2]
    if any(frame.shape[:2] != shape for frame in frames):
        raise ValueError("Calibration frames must have matching dimensions")

    green_reference = _color_difference(frames[0])
    magenta_reference = -_color_difference(frames[1])
    valid = (green_reference >= min_contrast) & (magenta_reference >= min_contrast)
    if roi_mask is not None:
        if roi_mask.shape != shape:
            raise ValueError("ROI mask must match the camera frame")
        valid &= roi_mask.astype(bool)

    gray_x = np.zeros(shape, np.uint32)
    gray_y = np.zeros(shape, np.uint32)
    confidence = np.full(shape, 32767, np.int16)
    offset = 2
    for index in range(sequence.x_bits):
        difference = _color_difference(frames[offset + index])
        gray_x = (gray_x << 1) | (difference > 0)
        confidence = np.minimum(confidence, np.abs(difference))
    offset += sequence.x_bits
    for index in range(sequence.y_bits):
        difference = _color_difference(frames[offset + index])
        gray_y = (gray_y << 1) | (difference > 0)
        confidence = np.minimum(confidence, np.abs(difference))

    screen_x = _gray_to_binary(gray_x).astype(np.int32)
    screen_y = _gray_to_binary(gray_y).astype(np.int32)
    valid &= confidence >= max(3, min_contrast // 2)
    valid &= (screen_x >= 0) & (screen_x < width)
    valid &= (screen_y >= 0) & (screen_y < height)
    return screen_x, screen_y, valid


@dataclass
class CalibrationProfile:
    """Dense TV-to-virtual-view map stored in normalized source coordinates."""

    map_x: np.ndarray
    map_y: np.ndarray
    valid: np.ndarray
    sample_count: int = 0
    view_size: tuple | None = None

    @property
    def screen_size(self):
        return self.map_x.shape[1], self.map_x.shape[0]

    @property
    def coverage(self):
        return float(np.count_nonzero(self.valid) / self.valid.size)

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
        map_x = np.where(self.valid, self.map_x * max(1, width - 1), -1).astype(np.float32)
        map_y = np.where(self.valid, self.map_y * max(1, height - 1), -1).astype(np.float32)
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
                            view_size=np.asarray(self.view_size or (0,0),dtype=np.int32))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != 1:
                raise ValueError("Unsupported calibration profile version")
            view_size = tuple(int(value) for value in data["view_size"]) if "view_size" in data else None
            if view_size == (0,0):
                view_size = None
            return cls(data["map_x"].astype(np.float32),
                       data["map_y"].astype(np.float32),
                       data["valid"].astype(bool), int(data["sample_count"]),view_size)


def build_profile(screen_x, screen_y, valid, screen_size, kernel=43, sigma=7):
    """Invert sparse camera-to-TV samples and smooth them into a dense map."""
    width, height = screen_size
    if screen_x.shape != screen_y.shape or screen_x.shape != valid.shape:
        raise ValueError("Decoded maps and validity mask must have matching shapes")
    camera_height, camera_width = valid.shape
    ys, xs = np.nonzero(valid)
    if len(xs) < 16:
        raise ValueError("Not enough reflected calibration pixels were detected")
    qx = screen_x[ys, xs]
    qy = screen_y[ys, xs]
    inside = (qx >= 0) & (qx < width) & (qy >= 0) & (qy < height)
    xs, ys, qx, qy = xs[inside], ys[inside], qx[inside], qy[inside]

    weight = np.zeros((height, width), np.float32)
    sum_x = np.zeros_like(weight)
    sum_y = np.zeros_like(weight)
    normalized_x = xs.astype(np.float32) / max(1, camera_width - 1)
    normalized_y = ys.astype(np.float32) / max(1, camera_height - 1)
    np.add.at(weight, (qy, qx), 1)
    np.add.at(sum_x, (qy, qx), normalized_x)
    np.add.at(sum_y, (qy, qx), normalized_y)

    kernel = max(3, int(kernel) | 1)
    blurred_weight = cv2.GaussianBlur(weight, (kernel, kernel), sigma)
    blurred_x = cv2.GaussianBlur(sum_x, (kernel, kernel), sigma)
    blurred_y = cv2.GaussianBlur(sum_y, (kernel, kernel), sigma)
    dense = blurred_weight > 1e-6
    map_x = np.zeros_like(weight)
    map_y = np.zeros_like(weight)
    map_x[dense] = blurred_x[dense] / blurred_weight[dense]
    map_y[dense] = blurred_y[dense] / blurred_weight[dense]
    return CalibrationProfile(map_x, map_y, dense, int(len(xs)),
                              (camera_width,camera_height))
