"""Tests for preview dispatch and aspect-ratio fitting, without a camera."""
import types
import unittest
from unittest.mock import MagicMock, patch

from live_layout import update_cone_preview


class ConePreviewTests(unittest.TestCase):
    def test_both_renderers_receive_background_setting_and_fit_preview(self):
        for circle in (False, True):
            with self.subTest(circle=circle):
                cv = MagicMock()
                pil = MagicMock()
                renderer = MagicMock(return_value=types.SimpleNamespace(shape=(800, 800, 3)))
                view = types.SimpleNamespace(
                    _last_cone_preview=0, _cone_preview_interval=.1,
                    remove_background=MagicMock(), _output_container=MagicMock(),
                    _output_label=MagicMock())
                setattr(view, "_apply_circle_hologram" if circle else "_apply_warp", renderer)
                view.remove_background.get.return_value = False
                view._output_container.winfo_width.return_value = 700
                view._output_container.winfo_height.return_value = 240
                frame = object()
                with patch.dict("sys.modules", {"cv2": cv, "PIL": pil}):
                    with patch("live_layout.time.monotonic", return_value=1):
                        update_cone_preview(view, frame)
                        update_cone_preview(view, frame)
                renderer.assert_called_once_with(frame, use_segmentation=False)
                self.assertEqual(cv.resize.call_args.args[1], (240, 240))
                view._output_label.configure.assert_called_once()


if __name__ == "__main__":
    unittest.main()
