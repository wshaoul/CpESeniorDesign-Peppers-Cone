"""Recorded calibration -> Mac map -> saved, compatible profile."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication

from calibration_recording import read_recording
from calibration_studio import CalibrationDialog
from cone_calibration import CalibrationSequence, CalibrationProfile

TV = (64, 48)
CAMERA = (320, 240)
ROI = (80, 40, 240, 160)


def write_recording(path):
    sequence = CalibrationSequence(*TV)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, CAMERA)
    if not writer.isOpened():
        raise RuntimeError("Test video encoder unavailable")
    def emit(image, count):
        frame = np.full((240, 320, 3), 12, np.uint8)
        frame[40:160, 80:240] = cv2.resize(image, (160, 120), interpolation=cv2.INTER_NEAREST)
        for _ in range(count):
            writer.write(frame)
    black = np.zeros((48, 64, 3), np.uint8)
    emit(black, 45)
    for i in range(len(sequence)):
        emit(sequence.frame(i), 15)
    emit(black, 30)
    writer.release()
    return sequence


class RecordedCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "clip.mp4"
        self.sequence = write_recording(self.path)
        self.dialog = CalibrationDialog(output_size=TV)

    def tearDown(self):
        analysis = self.dialog.analysis
        self.dialog.close()
        if analysis is not None:
            analysis.thread.join(5)
        self.directory.cleanup()

    def load(self):
        self.dialog.load_recording(str(self.path))
        analysis = self.dialog.analysis
        analysis.thread.join(5)
        self.assertFalse(analysis.thread.is_alive())
        self.dialog.poll_recording()
        self.assertIsNotNone(self.dialog.recording, self.dialog.status.text())

    def test_import_build_and_save_keeps_full_workflow_on_mac(self):
        self.load()
        self.assertFalse(self.dialog.save_button.isEnabled())
        self.assertTrue(self.dialog.build_button.isEnabled())
        self.assertIn("14 patterns", self.dialog.status.text())
        self.dialog.preview.roi = (.25, 1 / 6, .75, 2 / 3)
        self.dialog.build_from_recording()
        self.assertIsNotNone(self.dialog.profile)
        self.assertEqual(self.dialog.profile.screen_size, TV)
        target = Path(self.directory.name) / "saved.npz"
        self.assertFalse(target.exists())
        with patch("calibration_studio.default_profile_path", return_value=target):
            self.dialog.save_profile()
        loaded = CalibrationProfile.load(target)
        self.assertEqual(loaded.screen_size, TV)
        self.assertGreater(loaded.sample_count, 1000)

    def test_changed_source_size_is_rejected_before_building(self):
        self.load()
        self.dialog.output_width.setValue(65)
        self.dialog.build_from_recording()
        self.assertIsNone(self.dialog.profile)
        self.assertIn("source pattern size", self.dialog.status.text())

    def test_wrong_pattern_count_is_rejected_by_importer(self):
        with self.assertRaisesRegex(ValueError, "selected display needs"):
            read_recording(self.path, 24)

    def test_mac_file_permission_error_is_distinct_from_a_codec_failure(self):
        with patch("builtins.open", side_effect=PermissionError), self.assertRaisesRegex(RuntimeError, "macOS denied access"):
            read_recording(self.path, len(self.sequence))

    def test_playback_uses_fixed_clock_and_black_tail_and_cancel_stops_it(self):
        clock = [100.0]
        self.dialog.lead_seconds.setValue(3)
        self.dialog.settle_seconds.setValue(.5)
        screen = self.dialog.screen.currentData()
        with patch("calibration_studio.time.monotonic", side_effect=lambda: clock[0]), \
             patch("calibration_studio.QApplication.screens", return_value=[screen, object()]), \
             patch.object(self.dialog.pattern_window, "open_on"), \
             patch.object(self.dialog.pattern_window, "set_frame") as show:
            self.dialog.play_for_recording()
            self.assertFalse(show.call_args.args[0].any())
            clock[0] = 103.1
            self.dialog.playback_tick()
            np.testing.assert_array_equal(show.call_args.args[0], self.sequence.frame(0))
            clock[0] = 110.1
            self.dialog.playback_tick()
            self.assertFalse(show.call_args.args[0].any())
            clock[0] = 113.1
            self.dialog.playback_tick()
            self.assertIsNone(self.dialog.playback)
            self.assertFalse(self.dialog.playback_timer.isActive())
            self.assertTrue(self.dialog.load_button.isEnabled())


if __name__ == "__main__":
    unittest.main()
