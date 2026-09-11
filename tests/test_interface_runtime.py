import sys
import unittest
from pathlib import Path

import numpy as np


INTERFACE_DIR = Path(__file__).resolve().parents[1] / "Interface_updated"
sys.path.insert(0, str(INTERFACE_DIR))

import platform_support
from cone_player import create_circle_processor, create_single_processor
from live_view import build_cone_maps


class InterfaceRuntimeTests(unittest.TestCase):
    def test_native_camera_backend_is_first(self):
        label, backend = platform_support.camera_backends()[0]
        if sys.platform == "darwin":
            self.assertIn("AVFoundation", label)
            self.assertEqual(backend, platform_support.cv2.CAP_AVFOUNDATION)
        elif sys.platform.startswith("win"):
            self.assertIn("Windows", label)

    def test_cone_map_dimensions_and_content(self):
        map_x, map_y = build_cone_maps(64, 128)
        self.assertEqual(map_x.shape, (128, 128))
        self.assertEqual(map_y.shape, (128, 128))
        self.assertTrue(np.any(map_x >= 0))
        self.assertTrue(np.any(map_y >= 0))

    def test_single_cone_processor(self):
        processor, resources = create_single_processor()
        frame = np.full((240, 320, 3), (20, 120, 220), dtype=np.uint8)
        output = processor(frame)
        self.assertEqual(output.shape, (800, 800, 3))
        self.assertGreater(np.count_nonzero(output), 0)
        for resource in resources:
            resource.close()

    def test_circle_processor(self):
        processor, resources = create_circle_processor()
        frame = np.full((240, 320, 3), (20, 120, 220), dtype=np.uint8)
        output = processor(frame)
        self.assertEqual(output.shape, (800, 800, 3))
        self.assertGreater(np.count_nonzero(output), 0)
        for resource in resources:
            resource.close()


if __name__ == "__main__":
    unittest.main()
