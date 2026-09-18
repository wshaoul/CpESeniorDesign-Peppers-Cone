import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from live_studio import LiveWorker
from projection import ProjectionSettings


class FakeCapture:
    def __init__(self):
        self.released = threading.Event()
    def set(self,*args):
        return True
    def read(self):
        time.sleep(.005)
        return True,np.zeros((240,320,3),np.uint8)
    def release(self):
        self.released.set()


class FakeProjector:
    segmenter = None
    def process(self,frame,settings,segment=True):
        time.sleep(.01)
        return np.zeros((settings.height,settings.width,3),np.uint8)
    def close(self):
        pass


class WorkerTests(unittest.TestCase):
    def test_camera_result_and_clean_two_thread_shutdown(self):
        capture = FakeCapture()
        with patch("live_studio.LiveProjector",FakeProjector),patch("live_studio.open_mac_camera",return_value=capture):
            worker = LiveWorker(0,(320,240),30,ProjectionSettings(width=640,height=480),"Camera")
            worker.thread.start()
            deadline = time.monotonic()+2
            try:
                while worker.latest is None and time.monotonic()<deadline:
                    time.sleep(.01)
                self.assertIsNotNone(worker.latest)
                self.assertEqual(worker.latest[2].shape,(480,640,3))
            finally:
                stopped = worker.stop()
            self.assertTrue(stopped)
            self.assertTrue(capture.released.is_set())
            self.assertFalse(worker.camera_thread.is_alive())

    def test_camera_error_is_reported(self):
        with patch("live_studio.LiveProjector",FakeProjector),patch("live_studio.open_mac_camera",return_value=None):
            worker = LiveWorker(0,(320,240),30,ProjectionSettings(),"Camera")
            worker.thread.start()
            worker.thread.join(2)
            self.assertIn("Camera unavailable",worker.error)
            self.assertTrue(worker.stop())

    def test_pattern_does_not_open_camera(self):
        with patch("live_studio.LiveProjector",FakeProjector),patch("live_studio.open_mac_camera") as camera:
            worker = LiveWorker(0,(320,240),30,ProjectionSettings(width=640,height=480),"Alignment rings")
            worker.thread.start()
            deadline = time.monotonic()+2
            try:
                while worker.latest is None and time.monotonic()<deadline:
                    time.sleep(.01)
                self.assertIsNotNone(worker.latest)
            finally:
                self.assertTrue(worker.stop())
            camera.assert_not_called()


if __name__ == "__main__":
    unittest.main()
