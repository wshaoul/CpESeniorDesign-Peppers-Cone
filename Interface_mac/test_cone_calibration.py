import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2

from cone_calibration import (CalibrationProfile, CalibrationSequence,
                              build_profile, decode_sequence)


class ConeCalibrationTests(unittest.TestCase):
    def test_1080p_sequence_uses_24_patterns(self):
        sequence = CalibrationSequence(1920, 1080)
        self.assertEqual(len(sequence), 24)
        self.assertEqual(sequence.frame(0).shape, (1080, 1920, 3))
        self.assertIn("vertical", sequence.label(2))

    def test_identity_sequence_decodes_and_reconstructs_image(self):
        width, height = 64, 48
        sequence = CalibrationSequence(width, height)
        frames = [sequence.frame(index) for index in range(len(sequence))]
        screen_x, screen_y, valid = decode_sequence(frames, (width, height))
        yy, xx = np.indices((height, width))
        np.testing.assert_array_equal(screen_x, xx)
        np.testing.assert_array_equal(screen_y, yy)
        self.assertTrue(valid.all())

        profile = build_profile(screen_x, screen_y, valid, (width, height),
                                kernel=3, sigma=.5)
        target = np.dstack((xx * 4, yy * 5, (xx + yy) * 2)).astype(np.uint8)
        reconstructed = profile.apply(target)
        self.assertLess(float(np.mean(np.abs(reconstructed.astype(np.int16) -
                                             target.astype(np.int16)))), 2)
        self.assertGreater(profile.coverage, .99)

    def test_profile_round_trip(self):
        map_x = np.tile(np.linspace(0, 1, 10, dtype=np.float32), (8, 1))
        map_y = np.tile(np.linspace(0, 1, 8, dtype=np.float32)[:, None], (1, 10))
        profile = CalibrationProfile(map_x, map_y, np.ones((8, 10), bool), 80,(10,8))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.npz"
            profile.save(path)
            loaded = CalibrationProfile.load(path)
        np.testing.assert_allclose(loaded.map_x, profile.map_x)
        np.testing.assert_allclose(loaded.map_y, profile.map_y)
        self.assertEqual(loaded.sample_count, 80)
        self.assertEqual(loaded.view_size,(10,8))

    def test_portrait_calibration_does_not_stretch_model(self):
        yy,xx = np.indices((80,40),dtype=np.float32)
        profile = CalibrationProfile(xx/39,yy/79,np.ones((80,40),bool),3200,(40,80))
        image = np.zeros((40,80,3),np.uint8)
        image[10:30,30:50] = 255
        result = profile.apply(image)
        ys,xs = np.nonzero(result[:,:,0]>200)
        self.assertEqual(xs.max()-xs.min(),ys.max()-ys.min())

    def test_map_preview_is_black_where_unsupported(self):
        profile = CalibrationProfile(np.ones((8,10),np.float32),
                                     np.ones((8,10),np.float32),np.zeros((8,10),bool))
        self.assertFalse(profile.visualization().any())

    def test_profile_corrects_a_warped_camera_view(self):
        width,height = 80,60
        sequence = CalibrationSequence(width,height)
        yy,xx = np.indices((height,width),dtype=np.float32)
        camera_to_screen_x = xx*.88+4
        camera_to_screen_y = yy*.82+3
        observations = [cv2.remap(sequence.frame(index),camera_to_screen_x,
                                  camera_to_screen_y,cv2.INTER_NEAREST)
                        for index in range(len(sequence))]
        screen_x,screen_y,valid = decode_sequence(observations,(width,height))
        profile = build_profile(screen_x,screen_y,valid,(width,height),kernel=3,sigma=.5)

        target = np.dstack((xx*3,yy*4,(xx+yy)*1.5)).astype(np.uint8)
        tv_output = profile.apply(target)
        reflected = cv2.remap(tv_output,camera_to_screen_x,camera_to_screen_y,
                              cv2.INTER_LINEAR)
        safe = valid & (xx<width-6) & (yy<height-6)
        error = np.abs(reflected.astype(np.int16)-target.astype(np.int16))[safe]
        self.assertLess(float(error.mean()),5)

    def test_sequence_rejects_weak_or_outside_pixels(self):
        sequence = CalibrationSequence(32, 24)
        frames = [sequence.frame(index) for index in range(len(sequence))]
        frames = [frame.copy() for frame in frames]
        for frame in frames:
            frame[:4, :4] = 80
        _, _, valid = decode_sequence(frames, (32, 24))
        self.assertFalse(valid[:4, :4].any())
        self.assertTrue(valid[5:, 5:].all())


if __name__ == "__main__":
    unittest.main()
