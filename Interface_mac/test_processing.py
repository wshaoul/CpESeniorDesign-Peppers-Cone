import unittest

import numpy as np

from processing import ConeProcessor, build_cone_maps


class ProcessingTests(unittest.TestCase):
    def test_maps_have_expected_shape(self):
        map_x, map_y = build_cone_maps(frame_size=64, canvas_size=128)
        self.assertEqual(map_x.shape, (128, 128))
        self.assertEqual(map_y.shape, (128, 128))
        self.assertTrue(np.any(map_x >= 0))

    def test_processor_outputs_cone_frame(self):
        processor = ConeProcessor()
        frame = np.full((240, 320, 3), (20, 120, 220), dtype=np.uint8)
        output = processor.process(frame, remove_background=False)
        processor.close()
        self.assertEqual(output.shape, (800, 800, 3))
        self.assertGreater(np.count_nonzero(output), 0)


if __name__ == "__main__":
    unittest.main()
