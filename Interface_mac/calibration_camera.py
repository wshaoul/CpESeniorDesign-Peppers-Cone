"""Continuously drain the camera so pattern captures don't use queued frames."""
import threading
import time

import cv2

from processing import open_mac_camera


class CalibrationCamera:
    def __init__(self, index):
        self.index = index
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = ""
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        capture = None
        try:
            capture = open_mac_camera(self.index)
            if capture is None:
                raise RuntimeError("Camera unavailable. Stop, choose another camera number, and try again.")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            capture.set(cv2.CAP_PROP_FPS, 30)
            while not self.done.is_set():
                started = time.monotonic()
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera disconnected or stopped returning frames.")
                with self.lock:
                    self.latest = (started, frame)
        except Exception as exc:
            self.error = str(exc)
        finally:
            if capture is not None:
                capture.release()

    def snapshot(self):
        with self.lock:
            return self.latest

    def stop(self):
        # Only this worker releases the capture. The UI remains usable even if
        # the device is temporarily stuck in an OS-level read or permission prompt.
        self.done.set()
