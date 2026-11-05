import unittest
from unittest.mock import patch

import numpy as np

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from model_studio import ModelPage
from cone_calibration import CalibrationProfile
from calibration_studio import CalibrationDialog


class ModelUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.page = ModelPage()
        self.page.auto_rotate.setChecked(False)
        self.page.show()
        self.app.processEvents()

    def tearDown(self):
        self.page.shutdown()
        self.page.close()
        self.page.deleteLater()
        self.app.processEvents()

    def test_model_controls_update_render(self):
        original = self.page.current_raw.copy()
        self.page.yaw.setValue(self.page.yaw.value()+30)
        self.assertFalse((original==self.page.current_raw).all())
        self.page.layout_mode.setCurrentIndex(1)
        self.assertTrue(self.page.current_output.any())

    def test_tiger_demo_has_animation_control(self):
        self.page.model.setCurrentText("Tiger demo")
        self.assertTrue(self.page.animate_model.isEnabled())
        before = self.page.current_raw.copy()
        self.page.animation_phase += 1
        self.page.render()
        self.assertFalse(np.array_equal(before,self.page.current_raw))

    def test_one_display_opens_windowed_output(self):
        with patch("model_studio.QApplication.screens",return_value=[QApplication.primaryScreen()]):
            self.page.toggle_tv()
        self.assertTrue(self.page.tv.isVisible())
        self.assertFalse(self.page.tv.isFullScreen())
        self.assertIn("movable window",self.page.status.text())
        self.page.tv.close()

    def test_tv_close_restores_button(self):
        self.page.show_button.setText("Hide TV output")
        self.page.tv_closed()
        self.assertEqual(self.page.show_button.text(),"Show model on TV")

    def test_real_hide_button_click_closes_output(self):
        QTest.mouseClick(self.page.show_button,Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertTrue(self.page.tv.isVisible())
        self.assertEqual(self.page.show_button.text(),"Hide TV output")
        QTest.mouseClick(self.page.show_button,Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertFalse(self.page.tv.isVisible())
        self.assertEqual(self.page.show_button.text(),"Show model on TV")

    def test_real_q_key_closes_output(self):
        QTest.mouseClick(self.page.show_button,Qt.MouseButton.LeftButton)
        self.app.processEvents()
        self.assertTrue(self.page.tv.isVisible())
        QTest.keyClick(self.page.tv,Qt.Key.Key_Q)
        self.app.processEvents()
        self.assertFalse(self.page.tv.isVisible())
        self.assertEqual(self.page.show_button.text(),"Show model on TV")

    def test_model_has_live_style_reflection_controls(self):
        original = self.page.current_output.copy()
        self.page.mirror.setChecked(not self.page.mirror.isChecked())
        self.assertFalse((original==self.page.current_output).all())
        self.page.advanced_button.click()
        self.assertFalse(self.page.advanced_panel.isHidden())
        original_pixels = (self.page.current_raw!=0).any(axis=2).sum()
        self.page.model_scale.setValue(140)
        larger_pixels = (self.page.current_raw!=0).any(axis=2).sum()
        self.assertGreater(larger_pixels,original_pixels)

    def test_model_turn_and_output_side_controls(self):
        start = self.page.yaw.value()
        QTest.mouseClick(self.page.flip_button,Qt.MouseButton.LeftButton)
        self.assertEqual(self.page.yaw.value(),(start+180)%360)
        self.page.output_position.setCurrentText("Bottom")
        self.assertEqual(self.page.spins["rotation"].value(),90)
        bottom = self.page.current_output.copy()
        self.page.output_position.setCurrentText("Top")
        self.assertFalse((bottom==self.page.current_output).all())

    def test_calibrated_front_mode_uses_saved_profile(self):
        width,height = 320,180
        map_x = np.tile(np.linspace(0,1,width,dtype=np.float32),(height,1))
        map_y = np.tile(np.linspace(0,1,height,dtype=np.float32)[:,None],(1,width))
        profile = CalibrationProfile(map_x,map_y,np.ones((height,width),bool),5000)
        self.page.use_calibration_profile(profile)
        self.assertEqual(self.page.output_mode.currentIndex(),1)
        self.assertFalse(self.page.layout_mode.isEnabled())
        self.assertEqual(self.page.current_output.shape,(height,width,3))
        self.assertEqual(self.page.warp_tag.text(),"CALIBRATED FRONT VIEW")

    def test_front_mode_requires_a_profile(self):
        self.page.calibration_profile = None
        self.page.output_mode.setCurrentIndex(1)
        self.assertEqual(self.page.output_mode.currentIndex(),0)
        self.assertIn("No front-audience calibration",self.page.status.text())

    def test_calibration_dialog_prepares_paper_sized_sequence(self):
        dialog = CalibrationDialog(output_size=(1920,1080),parent=self.page)
        self.assertEqual(len(dialog.sequence),24)
        self.assertIn("24 patterns",dialog.pattern_size.text())
        dialog.close()


if __name__ == "__main__":
    unittest.main()
