"""Frame processing for the standalone macOS Pepper's Cone app."""

from __future__ import annotations

import math

import cv2
import numpy as np


FRAME_SIZE = 400
CANVAS_SIZE = 800


def build_cone_maps(
    frame_size=FRAME_SIZE,
    canvas_size=CANVAS_SIZE,
    span_deg=200,
    rotate_deg=270.0,
    inner_radius=0.08,
    outer_radius=0.995,
    center=(0.5, 0.5),
):
    map_x = np.full((canvas_size, canvas_size), -1, dtype=np.float32)
    map_y = np.full((canvas_size, canvas_size), -1, dtype=np.float32)
    cx, cy = int(center[0] * canvas_size), int(center[1] * canvas_size)
    radius_max = canvas_size * 0.5
    radius_in = max(0.0, min(0.99, inner_radius)) * radius_max
    radius_out = max(
        radius_in + 1.0,
        min(1.0, outer_radius) * radius_max,
    )
    half_angle = math.radians(max(1, min(359, span_deg))) * 0.5
    rotation = math.radians(rotate_deg)

    yy, xx = np.ogrid[:canvas_size, :canvas_size]
    dx, dy = xx - cx, yy - cy
    radius = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx) - rotation
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    valid = (
        (radius >= radius_in)
        & (radius <= radius_out)
        & (angle >= -half_angle)
        & (angle <= half_angle)
    )
    u = (angle[valid] + half_angle) / (2 * half_angle)
    v = 1.0 - ((radius[valid] - radius_in) / (radius_out - radius_in))
    map_x[valid] = np.clip(u, 0.0, 1.0) * (frame_size - 1)
    map_y[valid] = np.clip(v, 0.0, 1.0) * (frame_size - 1)
    return map_x, map_y


def enhance(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
    saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv2.convertScaleAbs(saturated, alpha=1.8, beta=-25)


def make_segmenter():
    try:
        import mediapipe as mp

        return mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
    except Exception:
        return None


class ConeProcessor:
    def __init__(self):
        self.segmenter = make_segmenter()
        self.map_x, self.map_y = build_cone_maps()

    @property
    def segmentation_available(self):
        return self.segmenter is not None

    def process(self, frame, remove_background=True):
        square = cv2.resize(
            frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA
        )
        foreground = square
        if remove_background and self.segmenter is not None:
            rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            result = self.segmenter.process(rgb)
            mask = cv2.GaussianBlur(
                result.segmentation_mask.astype(np.float32), (7, 7), 0
            )
            binary = (mask > 0.35).astype(np.uint8) * 255
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                np.ones((5, 5), np.uint8),
                iterations=1,
            )
            foreground = cv2.bitwise_and(square, square, mask=binary)

        scaled = cv2.resize(
            foreground, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA
        )
        padded = np.zeros_like(foreground)
        y = (FRAME_SIZE - scaled.shape[0]) // 2
        x = (FRAME_SIZE - scaled.shape[1]) // 2
        padded[y:y + scaled.shape[0], x:x + scaled.shape[1]] = scaled
        warped = cv2.remap(
            padded,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return enhance(warped)

    def close(self):
        close = getattr(self.segmenter, "close", None)
        if callable(close):
            close()


def open_mac_camera(index):
    """Open a Mac camera with AVFoundation and a safe automatic fallback."""
    for backend in (cv2.CAP_AVFOUNDATION, cv2.CAP_ANY):
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            return capture
        capture.release()
    return None
