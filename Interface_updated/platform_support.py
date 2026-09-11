"""Cross-platform helpers shared by the Pepper's Cone Tkinter views."""

from __future__ import annotations

import platform
import sys

import cv2


IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = sys.platform.startswith("win")
SYSTEM_NAME = platform.system() or sys.platform


def camera_backends():
    """Return OpenCV capture backends appropriate for the current OS."""
    if IS_MACOS:
        return [
            ("AVFoundation (macOS)", cv2.CAP_AVFOUNDATION),
            ("Automatic", cv2.CAP_ANY),
        ]
    if IS_WINDOWS:
        return [
            ("MSMF (Windows)", cv2.CAP_MSMF),
            ("DirectShow (Windows)", cv2.CAP_DSHOW),
            ("Automatic", cv2.CAP_ANY),
        ]
    return [
        ("V4L2 (Linux)", cv2.CAP_V4L2),
        ("Automatic", cv2.CAP_ANY),
    ]


def default_camera_backend():
    return camera_backends()[0][1]


def open_camera(index: int, api_preference: int | None = None):
    """Open a camera using the native backend, then OpenCV's auto backend."""
    preferred = default_camera_backend() if api_preference is None else api_preference
    attempted = []
    for backend in (preferred, cv2.CAP_ANY):
        if backend in attempted:
            continue
        attempted.append(backend)
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return None


def create_selfie_segmenter(model_selection: int = 1):
    """Create MediaPipe's legacy segmenter, or gracefully disable segmentation."""
    try:
        import mediapipe as mp

        solutions = getattr(mp, "solutions", None)
        if solutions is None:
            return None
        return solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )
    except Exception:
        return None


def maximize_window(window):
    """Maximize where supported without raising Tk's macOS 'zoomed' error."""
    if IS_MACOS:
        return
    try:
        window.state("zoomed")
    except Exception:
        pass
