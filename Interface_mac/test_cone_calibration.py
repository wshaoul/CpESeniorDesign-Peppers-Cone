import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import cv2

from cone_calibration import (CalibrationProfile, CalibrationSequence,
                              build_profile, decode_sequence,
                              default_profile_path, place_subject)


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

    def test_blurred_fine_stripes_still_decode_to_a_coarse_cell(self):
        """A camera that can't resolve the last bits must still yield positions."""
        width, height = 128, 96
        sequence = CalibrationSequence(width, height)
        frames = []
        for index in range(len(sequence)):
            frame = sequence.frame(index).astype(np.float32)
            if index >= len(sequence) - 3 or 2 + sequence.x_bits - 3 <= index < 2 + sequence.x_bits:
                # The last three bits of each axis average to grey, like an
                # unresolved stripe pattern.
                frame[:] = np.array((127, 127, 127), np.float32)
            frames.append(frame.astype(np.uint8))
        screen_x, screen_y, valid = decode_sequence(frames, (width, height), max_cell=8)
        yy, xx = np.indices((height, width))
        self.assertGreater(valid.mean(), .95)
        # Position is the centre of an 8-pixel cell: off by at most 4 pixels.
        self.assertLessEqual(int(np.abs(screen_x - xx)[valid].max()), 4)
        self.assertLessEqual(int(np.abs(screen_y - yy)[valid].max()), 4)

        # The old all-bits rule would have rejected every pixel; a tighter
        # max_cell must reject them too rather than guess.
        _, _, strict = decode_sequence(frames, (width, height), max_cell=4)
        self.assertFalse(strict.any())

    def test_dim_reflection_with_colour_cast_decodes_correctly(self):
        """Per-pixel midpoint: a +30 green cast must not flip any bits."""
        width, height = 64, 48
        sequence = CalibrationSequence(width, height)
        frames = []
        for index in range(len(sequence)):
            frame = sequence.frame(index).astype(np.float32) * .15   # dim reflection
            frame[:, :, 1] += 30                                     # green ambient cast
            frames.append(np.clip(frame, 0, 255).astype(np.uint8))
        screen_x, screen_y, valid = decode_sequence(frames, (width, height))
        yy, xx = np.indices((height, width))
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(screen_x, xx)
        np.testing.assert_array_equal(screen_y, yy)

    def test_noise_level_contrast_is_not_trusted(self):
        """Bits that sit near the midpoint must be dropped, not guessed."""
        width, height = 64, 48
        sequence = CalibrationSequence(width, height)
        rng = np.random.default_rng(1)
        frames = []
        for index in range(len(sequence)):
            frame = sequence.frame(index).astype(np.float32)
            if index >= 2:
                frame = 127 + (frame - 127) * .02 + rng.normal(0, 3, frame.shape)
            frames.append(np.clip(frame, 0, 255).astype(np.uint8))
        _, _, valid = decode_sequence(frames, (width, height), max_cell=2)
        self.assertLess(valid.mean(), .05)

    def test_place_subject_keeps_the_frame_size_and_scales_about_the_centre(self):
        image = np.zeros((60, 100, 3), np.uint8)
        image[20:40, 40:60] = 255                       # 20x20 block in the middle
        self.assertIs(place_subject(image), image)
        for zoom in (.5, 2.0):
            result = place_subject(image, zoom)
            self.assertEqual(result.shape, image.shape)
            ys, xs = np.nonzero(result[:, :, 0] > 200)
            self.assertAlmostEqual(xs.max() - xs.min() + 1, 20 * zoom, delta=2)
            self.assertAlmostEqual((xs.max() + xs.min()) / 2, 49.5, delta=1.5)
            self.assertAlmostEqual((ys.max() + ys.min()) / 2, 29.5, delta=1.5)

    def test_place_subject_zooming_in_crops_and_mirror_flips(self):
        image = np.zeros((40, 40, 3), np.uint8)
        image[:, :4] = 255                              # bright strip on the left edge
        self.assertFalse(place_subject(image, 3.0)[:, :, 0].any())   # pushed out of frame
        flipped = place_subject(image, mirror=True)
        self.assertTrue(flipped[:, -4:].all())
        self.assertFalse(flipped[:, :4].any())

    def test_profile_reuses_its_lookup_for_the_same_frame_size(self):
        yy, xx = np.indices((30, 50), dtype=np.float32)
        profile = CalibrationProfile(xx / 49, yy / 29, np.ones((30, 50), bool), 1500, (50, 30))
        image = np.dstack((xx * 5, yy * 8, xx + yy)).astype(np.uint8)
        first = profile.apply(image)
        lookup = dict(profile._lookup)
        np.testing.assert_array_equal(profile.apply(image), first)
        self.assertEqual(list(profile._lookup), list(lookup))
        self.assertIs(next(iter(profile._lookup.values()))[0], next(iter(lookup.values()))[0])
        profile.apply(cv2.resize(image, (25, 15)))      # a different size replaces it
        self.assertEqual(len(profile._lookup), 1)

    def test_saved_map_lives_in_one_place_for_both_tabs(self):
        with unittest.mock.patch.dict("os.environ", {"APPDATA": r"C:\Users\someone\AppData\Roaming"}):
            path = default_profile_path()
        self.assertEqual(path.name, "front_audience_calibration.npz")
        self.assertEqual(path.parent.name, "PeppersCone")

    def test_sequence_rejects_weak_or_outside_pixels(self):
        sequence = CalibrationSequence(32, 24)
        frames = [sequence.frame(index) for index in range(len(sequence))]
        frames = [frame.copy() for frame in frames]
        for frame in frames:
            frame[:4, :4] = 80
        _, _, valid = decode_sequence(frames, (32, 24))
        self.assertFalse(valid[:4, :4].any())
        self.assertTrue(valid[5:, 5:].all())

    def test_smoothing_reduces_quantized_coordinate_steps(self):
        height, width, cell = 320, 512, 32
        yy, xx = np.indices((height, width), dtype=np.float32)
        quantized_x = (xx // cell * cell + (cell - 1) / 2) / (width - 1)
        quantized_y = (yy // cell * cell + (cell - 1) / 2) / (height - 1)
        original = CalibrationProfile(quantized_x, quantized_y,
                                       np.ones((height, width), bool), height * width, (width, height))
        smooth = original.smoothed(16)
        interior = np.s_[64:-64, 64:-64]
        for before, after, truth in ((original.map_x, smooth.map_x, xx / (width - 1)),
                                      (original.map_y, smooth.map_y, yy / (height - 1))):
            old_error = np.abs(before - truth)[interior].mean()
            new_error = np.abs(after - truth)[interior].mean()
            self.assertLess(float(new_error), float(old_error) * .1)
        self.assertEqual(smooth.sample_count, original.sample_count)
        self.assertEqual(smooth.view_size, original.view_size)

    def test_smoothing_does_not_expand_support_or_mix_in_black_coordinates(self):
        valid = np.zeros((80, 100), bool)
        valid[10:70, 15:85] = True
        valid[30:45, 35:55] = False
        mx = np.where(valid, .72, np.nan).astype(np.float32)
        my = np.where(valid, .68, np.nan).astype(np.float32)
        original = CalibrationProfile(mx, my, valid, 100, (100, 80))
        smooth = original.smoothed(16)
        np.testing.assert_array_equal(smooth.valid, valid)
        np.testing.assert_allclose(smooth.map_x[valid], .72, atol=1e-6)
        np.testing.assert_allclose(smooth.map_y[valid], .68, atol=1e-6)
        self.assertTrue(np.isfinite(smooth.map_x).all())
        self.assertFalse(smooth.visualization()[~valid].any())
        self.assertTrue(np.isnan(original.map_x[~valid]).all())
        self.assertEqual(smooth.coverage, original.coverage)

    def test_smoothing_reduces_stair_steps_in_a_rendered_slanted_edge(self):
        height, width, cell = 320, 512, 32
        yy, xx = np.indices((height, width), dtype=np.float32)
        quantized_x = xx // cell * cell + (cell - 1) / 2
        quantized_y = yy // cell * cell + (cell - 1) / 2
        profile = CalibrationProfile(quantized_x / (width - 1),
            (quantized_y + .35 * (quantized_x - width / 2)) / (height - 1),
            np.ones((height, width), bool))
        source = np.zeros((height, width, 3), np.uint8)
        source[height // 2:] = 255
        def edge_deviation(result):
            columns = np.arange(64, width - 64)
            edge = np.argmax(result[:, 64:-64, 0] > 127, axis=0)
            straight_line = np.polyval(np.polyfit(columns, edge, 1), columns)
            return float(np.sqrt(np.mean((edge - straight_line) ** 2)))
        before = edge_deviation(profile.apply(source))
        after = edge_deviation(profile.smoothed(16).apply(source))
        self.assertGreater(before, 5)
        self.assertLess(after, before * .1)

    def test_original_lookup_and_map_survive_smoothing_comparison(self):
        yy, xx = np.indices((60, 80), dtype=np.float32)
        profile = CalibrationProfile(xx / 79, yy / 59, np.ones((60, 80), bool))
        image = np.dstack((xx * 3, yy * 4, xx + yy)).astype(np.uint8)
        before = profile.apply(image)
        old_lookup = profile._lookup_for(80, 60)
        smoothed = profile.smoothed(8)
        self.assertIs(profile.smoothed(0), profile)
        self.assertFalse(smoothed._lookup)
        smoothed.apply(image)
        self.assertIs(profile._lookup_for(80, 60), old_lookup)
        np.testing.assert_array_equal(profile.apply(image), before)

    def test_smoothing_empty_support_and_invalid_strength(self):
        empty = CalibrationProfile(np.zeros((8, 10), np.float32),
                                    np.zeros((8, 10), np.float32), np.zeros((8, 10), bool))
        self.assertFalse(empty.smoothed(16).visualization().any())
        for sigma in (-1, float('nan'), 100):
            with self.assertRaises(ValueError):
                empty.smoothed(sigma)


if __name__ == "__main__":
    unittest.main()
