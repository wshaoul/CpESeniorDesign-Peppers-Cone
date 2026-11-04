"""Live tab: the "Use my saved calibration" box, without a camera or a TV.

Run with: python -m unittest test_live_calibration
"""
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import live_view
from cone_calibration import CalibrationProfile
from studio_theme import configure_appearance

TV = (96, 54)
VIEW = (60, 30)


def make_profile(path):
    """A map that shows the camera view mirrored left to right on a 96x54 'TV'."""
    yy, xx = np.indices((TV[1], TV[0]), dtype=np.float32)
    profile = CalibrationProfile(1 - xx / (TV[0] - 1), yy / (TV[1] - 1),
                                 np.ones((TV[1], TV[0]), bool), 5000, VIEW)
    profile.save(path)
    return profile


class LiveCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        configure_appearance(self.root)
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "front_audience_calibration.npz"
        # Never read or write the developer's real saved map from a test.
        patcher = patch.object(live_view, "default_profile_path", return_value=self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        controller = type("Controller", (), {"show_page": lambda self, name: None})()
        with patch.object(live_view.LiveView, "_scan_indices", return_value=["(no cameras found)"]):
            self.view = live_view.LiveView(self.root, controller)
        self.view._displays = [(0, 0, *TV), (1920, 0, 1280, 720)]
        self.view.display_choice.configure(values=["Display 1", "Display 2"])
        self.view.display_choice.current(0)
        self.frame = np.zeros((72, 128, 3), np.uint8)
        self.frame[20:50, 10:40] = (0, 0, 255)          # red block, left of centre

    def tearDown(self):
        self.root.destroy()
        self.directory.cleanup()

    def toggle(self, on):
        self.view.use_calibration.set(on)
        self.view._toggle_calibration()

    def label(self):
        return self.view._calibration_label.cget("text")

    def test_the_option_sits_in_the_main_setup_panel(self):
        self.assertFalse(self.view.use_calibration.get())
        self.assertIsNone(self.view._calibration)
        self.assertIn("Calibrate tab", self.label())

    def test_label_says_when_a_saved_map_is_ready(self):
        make_profile(self.path)
        self.view._refresh_calibration_label()
        self.assertIn("ready to use", self.label())
        self.assertIsNone(self.view._calibration)       # saved, but not switched on

    def test_circle_view_does_not_offer_the_single_viewpoint_map(self):
        import live_view_circle
        controller = type("Controller", (), {"show_page": lambda self, name: None})()
        with patch.object(live_view_circle.LiveView, "_scan_indices", return_value=["(no cameras found)"]):
            circle = live_view_circle.LiveView(self.root, controller)
        self.assertFalse(hasattr(circle, "use_calibration"))

    def test_without_a_saved_map_the_box_clears_and_the_old_warp_still_runs(self):
        self.toggle(True)
        self.assertFalse(self.view.use_calibration.get())
        self.assertIsNone(self.view._calibration)
        self.assertIn("Calibrate tab", self.label())
        self.view.remove_background.set(False)
        output = self.view._apply_warp(self.frame, use_segmentation=False)
        self.assertEqual(output.shape[:2], (live_view.CANVAS_SIZE, live_view.CANVAS_SIZE))

    def test_saved_map_replaces_the_polar_warp_at_tv_resolution(self):
        make_profile(self.path)
        self.toggle(True)
        self.assertTrue(self.view.use_calibration.get())
        self.view.remove_background.set(False)
        self.view.zoom_var.set(0.6)                      # default size: fit the view
        output = self.view._apply_warp(self.frame, use_segmentation=False)
        self.assertEqual(output.shape[:2], (TV[1], TV[0]))
        # Black stays black (the polar path's contrast step would lift it to grey).
        self.assertEqual(int(output[0, 0].max()), 0)
        # The map mirrors the view, so the red block lands right of centre.
        ys, xs = np.nonzero(output[:, :, 2] > 200)
        self.assertGreater(xs.mean(), TV[0] / 2)
        self.assertIn(f"{TV[0]} × {TV[1]}", self.label())
        self.assertNotIn("Pick the TV", self.label())

    def test_unticking_goes_back_to_the_polar_warp(self):
        make_profile(self.path)
        self.toggle(True)
        self.toggle(False)
        self.assertIsNone(self.view._calibration)
        self.view.remove_background.set(False)
        output = self.view._apply_warp(self.frame, use_segmentation=False)
        self.assertEqual(output.shape[:2], (live_view.CANVAS_SIZE, live_view.CANVAS_SIZE))

    def test_warns_when_the_chosen_display_is_not_the_one_calibrated(self):
        make_profile(self.path)
        self.toggle(True)
        self.view.display_choice.current(1)
        self.view._refresh_calibration_label()
        self.assertIn("Pick the TV you calibrated on", self.label())
        self.assertIn("1280 × 720", self.label())

    def test_zoom_and_mirror_sliders_still_apply_to_the_calibrated_picture(self):
        make_profile(self.path)
        self.toggle(True)
        self.view.remove_background.set(False)
        self.view.zoom_var.set(0.3)                      # half of the default size
        small = self.view._apply_warp(self.frame, use_segmentation=False)
        self.view.zoom_var.set(0.6)
        normal = self.view._apply_warp(self.frame, use_segmentation=False)
        self.assertLess(np.count_nonzero(small[:, :, 2] > 200), np.count_nonzero(normal[:, :, 2] > 200))
        self.view.mirror_var.set(True)
        mirrored = self.view._apply_warp(self.frame, use_segmentation=False)
        self.assertLess(np.nonzero(mirrored[:, :, 2] > 200)[1].mean(), TV[0] / 2)

    def test_unreadable_file_is_reported_not_raised(self):
        self.path.write_bytes(b"not a calibration")
        self.toggle(True)
        self.assertIsNone(self.view._calibration)
        self.assertFalse(self.view.use_calibration.get())
        self.assertIn("Couldn't read", self.label())


if __name__ == "__main__":
    unittest.main()
