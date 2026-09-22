"""Run with QT_QPA_PLATFORM=offscreen for camera-free interface checks."""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication
from live_studio import LivePage


class LiveUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        storage = SimpleNamespace(value=lambda key,default=None,**kwargs: default)
        with patch("live_studio.QSettings",return_value=storage):
            self.page = LivePage()
        self.page.show()
        self.app.processEvents()

    def tearDown(self):
        self.page.shutdown()
        self.page.close()
        self.page.deleteLater()
        self.app.processEvents()

    def test_simple_defaults_and_advanced_toggle(self):
        self.assertTrue(self.page.advanced_panel.isHidden())
        self.assertFalse(self.page.output_button.isEnabled())
        self.assertEqual(self.page.start_button.text(),"Start camera")
        self.assertEqual(self.page.settings().width,1280)
        self.assertEqual(self.page.settings().views,4)
        self.page.advanced_button.click()
        self.app.processEvents()
        self.assertFalse(self.page.advanced_panel.isHidden())
        self.page.spins["rotation"].setValue(180)
        self.assertEqual(self.page.settings().rotation,180)
        self.page.advanced_button.click()
        self.assertTrue(self.page.advanced_panel.isHidden())

    def test_tv_button_waits_for_first_frame(self):
        with patch("live_studio.LiveWorker") as worker_class:
            worker = worker_class.return_value
            worker.stop.return_value = True
            worker.error = ""
            worker.lock = threading.Lock()
            worker.latest = None
            worker.segmentation_available = True
            self.page.start()
            self.assertFalse(self.page.output_button.isEnabled())
            frame = np.zeros((240,320,3),np.uint8)
            worker.latest = (1,frame,frame,30,20)
            self.page.next_frame()
            self.assertTrue(self.page.output_button.isEnabled())
            self.assertIn("Preview is ready",self.page.status.text())
            self.assertIn("processed fps",self.page.technical_status.text())
            self.page.stop()
            self.assertFalse(self.page.output_button.isEnabled())
            self.assertTrue(self.page.start_button.isEnabled())

    def test_capture_settings_restart_running_camera(self):
        self.page.worker = SimpleNamespace(stop=lambda: True)
        with patch.object(self.page,"start") as restart:
            self.page.resolution.setCurrentIndex(1)
            restart.assert_called_once()
            restart.reset_mock()
            self.page.camera_index.setValue(1)
            restart.assert_called_once()
        self.page.worker = None

    def test_retina_preview_keeps_physical_pixel_detail(self):
        preview = self.page.preview
        preview.resize(320,180)
        frame = np.zeros((1080,1920,3),np.uint8)
        with patch.object(type(preview),"devicePixelRatioF",return_value=2), patch.object(preview,"setPixmap") as display:
            preview.set_frame(frame)
            self.assertEqual(preview.source.width(),640)
            self.assertEqual(display.call_args.args[0].devicePixelRatio(),2)

    def test_detail_view_uses_original_frame_and_updates_live(self):
        preview = self.page.preview
        preview.set_frame(np.zeros((1080,1920,3),np.uint8))
        preview.show_detail()
        self.app.processEvents()
        self.assertTrue(preview.detail_window.isVisible())
        self.assertIn("1920 × 1080",preview.detail_info.text())
        self.assertEqual(preview.detail_image.pixmap().width(),1920)
        preview.set_frame(np.zeros((720,1280,3),np.uint8))
        self.assertIn("1280 × 720",preview.detail_info.text())
        self.assertEqual(preview.detail_image.pixmap().width(),1280)
        preview.detail_window.close()


if __name__ == "__main__":
    unittest.main()
