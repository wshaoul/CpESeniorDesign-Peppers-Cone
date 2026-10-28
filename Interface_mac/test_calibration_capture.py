import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from calibration_camera import CalibrationCamera
from calibration_studio import CalibrationDialog
from cone_calibration import CalibrationSequence


class FakeCamera:
    def __init__(self):
        self.latest = None
        self.error = ""
        self.stopped = False
        self.thread = threading.Thread()

    def snapshot(self):
        return self.latest

    def stop(self):
        self.stopped = True


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.dialog = CalibrationDialog(output_size=(64,48))
        self.camera = FakeCamera()
        self.dialog.capture = self.camera
        self.frame = np.full((80,100,3),80,np.uint8)
        self.dialog.latest_frame = self.frame
        self.dialog.preview.set_frame(self.frame)
        self.dialog.preview.roi = (0,0,1,1)
        self.dialog.last_frame_time = 1
        self.open = patch.object(self.dialog.pattern_window,"open_on")
        self.open.start()

    def tearDown(self):
        self.dialog.close()
        self.open.stop()

    def begin(self):
        self.dialog.last_frame_time = time.monotonic()
        with patch("calibration_studio.QApplication.screens",return_value=[1,2]):
            self.dialog.start_calibration()

    def test_capture_waits_for_fresh_frames_and_uses_fixed_roi(self):
        self.begin()
        started = self.dialog.pattern_started
        self.camera.latest = (started+.1,self.frame)
        self.dialog.update_camera()
        self.assertEqual(len(self.dialog.pattern_samples),0)
        # A repeated frame cannot advance the sequence.
        self.dialog.update_camera()
        self.assertEqual(self.dialog.pattern_index,0)
        self.dialog.preview.roi = (.2,.2,.8,.8)
        for offset,value in ((2,80),(2.1,250),(2.2,82)):
            self.camera.latest = (started+offset,np.full_like(self.frame,value))
            self.dialog.update_camera()
        self.assertEqual(self.dialog.pattern_index,1)
        self.assertEqual(self.dialog.captured_frames[0].shape,self.frame.shape)
        self.assertTrue((self.dialog.captured_frames[0]==82).all())

    def test_cancel_and_dialog_reject_stop_capture(self):
        self.begin()
        self.dialog.cancel_button.click()
        self.assertEqual(self.dialog.pattern_index,-1)
        self.assertTrue(self.dialog.preview.isEnabled())
        self.dialog.done(QDialog.DialogCode.Rejected)
        self.assertTrue(self.camera.stopped)
        self.assertIsNone(self.dialog.capture)
        self.assertFalse(self.dialog.pattern_window.isVisible())

    def test_q_on_output_cancels_the_sequence(self):
        self.begin()
        self.dialog.pattern_window.show()
        QTest.keyClick(self.dialog.pattern_window,Qt.Key.Key_Q)
        self.assertEqual(self.dialog.pattern_index,-1)
        self.assertFalse(self.dialog.pattern_window.isVisible())

    def test_changed_camera_resolution_aborts(self):
        self.begin()
        self.camera.latest = (self.dialog.pattern_started+2,np.zeros((60,100,3),np.uint8))
        self.dialog.update_camera()
        self.assertEqual(self.dialog.pattern_index,-1)
        self.assertIn("resolution changed",self.dialog.status.text())

    def test_map_is_not_saved_until_user_accepts(self):
        sequence = CalibrationSequence(64,48)
        self.dialog.captured_frames = [sequence.frame(i) for i in range(len(sequence))]
        self.dialog.capture_roi = (0,0,64,48)
        self.dialog.capture_screen = QApplication.primaryScreen()
        with patch("cone_calibration.CalibrationProfile.save") as save:
            self.dialog.finish_calibration()
            self.assertIsNotNone(self.dialog.profile)
            save.assert_not_called()
            self.dialog.show_grid(True)
            self.assertIsNotNone(self.dialog.pattern_window.image.pixmap())
            self.dialog.save_button.click()
            save.assert_called_once()

    def test_single_display_has_actionable_message(self):
        with patch("calibration_studio.QApplication.screens",return_value=[1]):
            self.dialog.start_calibration()
        self.assertEqual(self.dialog.pattern_index,-1)
        self.assertIn("extended display",self.dialog.status.text())

    def test_camera_worker_releases_device_on_stop(self):
        released = threading.Event()
        class Capture:
            def set(self,*args):
                pass
            def read(self):
                time.sleep(.005)
                return True,np.zeros((48,64,3),np.uint8)
            def release(self):
                released.set()
        with patch("calibration_camera.open_mac_camera",return_value=Capture()):
            worker = CalibrationCamera(0)
            worker.thread.start()
            try:
                deadline = time.monotonic()+2
                while worker.snapshot() is None and time.monotonic()<deadline:
                    time.sleep(.01)
                self.assertIsNotNone(worker.snapshot())
            finally:
                worker.stop()
                worker.thread.join(2)
        self.assertTrue(released.is_set())
        self.assertFalse(worker.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
