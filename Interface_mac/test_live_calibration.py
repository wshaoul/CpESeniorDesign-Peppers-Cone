"""Mac saved-map controls and calibrated live rendering without a camera."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication

from cone_calibration import CalibrationProfile
from live_studio import LivePage
from projection import LiveProjector, ProjectionSettings


def profile():
    yy, xx = np.indices((54, 96), dtype=np.float32)
    return CalibrationProfile(1 - xx / 95, yy / 53, np.ones((54, 96), bool), 5000, (60, 30))


class CalibratedRenderingTests(unittest.TestCase):
    def setUp(self):
        with patch("projection.make_segmenter", return_value=None):
            self.projector = LiveProjector()
        self.frame = np.zeros((72, 128, 3), np.uint8)
        self.frame[20:50, 10:40] = (0, 0, 255)
        self.settings = ProjectionSettings(width=640, height=480, calibration=profile(),
                                           mirror=False, zoom=.85, gain=1, remove_background=False)

    def tearDown(self):
        self.projector.close()

    def test_saved_map_uses_native_size_and_preserves_black(self):
        image = self.projector.process(self.frame, self.settings)
        self.assertEqual(image.shape, (54, 96, 3))
        self.assertEqual(int(image[0, 0].max()), 0)
        self.assertGreater(np.nonzero(image[:, :, 2] > 200)[1].mean(), 48)

    def test_zoom_mirror_and_gain_apply_before_or_after_the_map(self):
        normal = self.projector.process(self.frame, self.settings)
        small = self.projector.process(self.frame, replace(self.settings, zoom=.425))
        self.assertLess(np.count_nonzero(small[:, :, 2] > 200), np.count_nonzero(normal[:, :, 2] > 200))
        mirrored = self.projector.process(self.frame, replace(self.settings, mirror=True))
        self.assertLess(np.nonzero(mirrored[:, :, 2] > 200)[1].mean(), 48)
        dark = self.projector.process(self.frame, replace(self.settings, gain=.5))
        self.assertLessEqual(int(dark.max()), 128)

    def test_switching_off_returns_to_existing_polar_layout(self):
        calibrated = self.projector.process(self.frame, self.settings)
        polar = self.projector.process(self.frame, replace(self.settings, calibration=None))
        self.assertEqual(calibrated.shape, (54, 96, 3))
        self.assertEqual(polar.shape, (480, 640, 3))

    def test_segmentation_applies_to_calibrated_output(self):
        class Segmenter:
            def process(self, image):
                return SimpleNamespace(segmentation_mask=np.zeros(image.shape[:2], np.float32))
            def close(self):
                pass
        self.projector.segmenter = Segmenter()
        output = self.projector.process(self.frame, replace(self.settings, remove_background=True))
        self.assertFalse(output.any())


class LiveCalibrationUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        storage = SimpleNamespace(value=lambda key, default=None, **kwargs: default)
        with patch("live_studio.QSettings", return_value=storage):
            self.page = LivePage()
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "map.npz"
        profile().save(self.path)

    def tearDown(self):
        self.page.shutdown()
        self.page.close()
        self.directory.cleanup()

    def test_load_a_windows_profile_and_toggle_it_without_recalibrating(self):
        self.page.load_calibration(str(self.path))
        self.assertTrue(self.page.use_calibration.isChecked())
        self.assertEqual(self.page.settings().calibration.screen_size, (96, 54))
        self.assertEqual((self.page.settings().width, self.page.settings().height), (96, 54))
        self.assertFalse(self.page.quality.isEnabled())
        self.assertFalse(self.page.spins["diameter"].isEnabled())
        self.assertTrue(self.page.spins["zoom"].isEnabled())
        self.page.use_calibration.setChecked(False)
        self.assertIsNone(self.page.settings().calibration)
        self.assertTrue(self.page.quality.isEnabled())
        self.page.use_calibration.setChecked(True)
        self.assertEqual(self.page.settings().calibration.screen_size, (96, 54))

    def test_missing_saved_profile_leaves_the_polar_view_available(self):
        self.page.calibration_path = Path(self.directory.name) / "missing.npz"
        self.page.use_calibration.setChecked(True)
        self.assertFalse(self.page.use_calibration.isChecked())
        self.assertIsNone(self.page.settings().calibration)
        self.assertIn("Could not load", self.page.calibration_label.text())

    def test_bad_import_does_not_replace_an_existing_working_profile(self):
        self.page.load_calibration(str(self.path))
        previous = self.page.calibration_profile
        bad = Path(self.directory.name) / "bad.npz"
        bad.write_bytes(b'not a map')
        self.page.load_calibration(str(bad))
        self.assertIs(self.page.calibration_profile, previous)
        self.assertIn("Could not load", self.page.calibration_label.text())

    def test_opening_calibration_releases_live_camera_and_accepts_saved_map(self):
        with patch.object(self.page, "stop", return_value=True) as stop, patch("live_studio.CalibrationDialog") as dialog:
            self.page.open_calibration()
            stop.assert_called_once()
            dialog.return_value.profile_ready.connect.assert_called_once_with(self.page.use_calibration_profile)
            dialog.return_value.show.assert_called_once()
        self.page.calibration_dialog = None
        self.page.use_calibration_profile(profile())
        self.assertTrue(self.page.use_calibration.isChecked())

    def test_map_smoothing_can_be_compared_to_the_original_without_reloading(self):
        self.page.load_calibration(str(self.path))
        original = self.page.calibration_profile
        self.assertTrue(self.page.map_smoothing.isEnabled())
        first = self.page.settings().calibration
        self.assertIsNot(first, original)
        self.assertIs(self.page.settings().calibration, first)
        np.testing.assert_array_equal(first.valid, original.valid)
        self.page.map_smoothing.setCurrentIndex(0)
        self.assertIs(self.page.settings().calibration, original)
        self.page.map_smoothing.setCurrentIndex(1)
        light = self.page.settings().calibration
        self.assertIsNot(light, first)
        self.assertIs(self.page.settings().calibration, light)
        self.page.use_calibration.setChecked(False)
        self.assertFalse(self.page.map_smoothing.isEnabled())
        self.assertIsNone(self.page.settings().calibration)
        # The disk profile remains the original map for future comparisons.
        loaded = CalibrationProfile.load(self.path)
        np.testing.assert_array_equal(loaded.map_x, original.map_x)

    def test_smoothing_updates_a_running_worker_without_restarting_capture(self):
        self.page.load_calibration(str(self.path))
        updates = []
        self.page.worker = SimpleNamespace(update=lambda settings, mode: updates.append(settings),
                                            stop=lambda: True)
        self.page.map_smoothing.setCurrentIndex(0)
        self.assertIs(updates[-1].calibration, self.page.calibration_profile)
        self.page.map_smoothing.setCurrentIndex(3)
        self.assertIsNot(updates[-1].calibration, self.page.calibration_profile)
        self.page.worker = None


if __name__ == "__main__":
    unittest.main()
