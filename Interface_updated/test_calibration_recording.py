import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from calibration_view import find_pattern_run, read_recording
from cone_calibration import CalibrationSequence, decode_sequence


TV = (64, 36)
CAMERA = (640, 360)
REGION = (160, 60, 480, 240)  # where the mirrored "reflection" lands in the camera frame
FPS = 30


def write_fake_recording(path, hold=0.5):
    """Desktop, black lead-in, the sequence mirrored into REGION, black tail, desktop."""
    sequence = CalibrationSequence(*TV)
    x1, y1, x2, y2 = REGION
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, CAMERA)

    def emit(tv_image, seconds):
        frame = np.full((CAMERA[1], CAMERA[0], 3), 12, np.uint8)
        if tv_image is not None:
            reflected = cv2.resize(tv_image, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
            frame[y1:y2, x1:x2] = reflected[:, ::-1]
        for _ in range(round(seconds * FPS)):
            writer.write(frame)

    desktop = np.full((TV[1], TV[0], 3), 200, np.uint8)
    emit(desktop, 1.0)
    emit(np.zeros_like(desktop), 2.0)
    for i in range(len(sequence)):
        emit(sequence.frame(i), hold)
    emit(np.zeros_like(desktop), 1.5)
    emit(desktop, 0.5)
    writer.release()
    return sequence


def lit_room_signals(patterns=23, period=1.5, start=9.5, drop_switch=None, cut_before_end=False):
    """Per-frame signals from a recording made with the room lights on.

    The room is bright even when the TV is black (level 62), and the solid green
    frame settles *below* the level it flashed to when it first appeared, so a
    brightness threshold between 'black' and 'green' lands under the green frame.
    Taken from a real recording that the old logic read as 0.04 s per pattern.
    """
    fps = 30
    end = start + patterns * period
    times = np.arange(int((end + 3) * fps)) / fps
    levels = np.full(times.size, 62.0)
    greens = np.zeros(times.size)
    changes = np.full(times.size, 0.3)
    running = (times >= start) & (times < end)
    levels[running] = 80.0
    levels[(times >= start) & (times < start + 0.3)] = 77.0            # exposure overshoot
    levels[(times >= start + 0.3) & (times < start + period)] = 69.0   # settled green frame
    levels[times >= end] = 55.0                                        # black tail, exposure adapted
    greens[(times >= start) & (times < start + period)] = 40.0
    greens[(times >= start + period) & (times < start + 2 * period)] = -60.0
    for k in range(patterns + 1):
        if k == drop_switch:
            continue
        changes[int(round((start + k * period) * fps))] = 10.0
    if cut_before_end:
        keep = times < end - 0.3
        times, levels, greens, changes = times[keep], levels[keep], greens[keep], changes[keep]
    return times, levels, greens, changes


class RecordingTests(unittest.TestCase):
    def test_lit_room_recording_is_read_by_following_the_switches(self):
        times, levels, greens, changes = lit_room_signals()
        start, period = find_pattern_run(times, levels, greens, 23, changes)
        self.assertAlmostEqual(start, 9.5, delta=0.1)
        self.assertAlmostEqual(period, 1.5, delta=0.03)

    def test_lit_room_recording_defeats_the_brightness_only_fallback(self):
        times, levels, greens, _ = lit_room_signals()
        with self.assertRaisesRegex(ValueError, "per pattern"):
            find_pattern_run(times, levels, greens, 23)

    def test_one_missed_switch_does_not_break_the_count(self):
        times, levels, greens, changes = lit_room_signals(drop_switch=11)
        _, period = find_pattern_run(times, levels, greens, 23, changes)
        self.assertAlmostEqual(period, 1.5, delta=0.03)

    def test_wrong_display_is_reported_with_the_pattern_count(self):
        times, levels, greens, changes = lit_room_signals()
        with self.assertRaisesRegex(ValueError, "23 patterns but the selected display needs 24"):
            find_pattern_run(times, levels, greens, 24, changes)
        with self.assertRaisesRegex(ValueError, "23 patterns but the selected display needs 14"):
            find_pattern_run(times, levels, greens, 14, changes)

    def test_lit_room_recording_cut_off_before_the_end_is_rejected(self):
        times, levels, greens, changes = lit_room_signals(cut_before_end=True)
        with self.assertRaisesRegex(ValueError, "stops before"):
            find_pattern_run(times, levels, greens, 23, changes)

    def test_recorded_sequence_decodes_to_tv_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording.mp4"
            sequence = write_fake_recording(path)
            frames, period = read_recording(path, len(sequence))

        self.assertEqual(len(frames), len(sequence))
        self.assertAlmostEqual(period, 0.5, delta=0.05)

        x1, y1, x2, y2 = REGION
        screen_x, screen_y, valid = decode_sequence([f[y1:y2, x1:x2] for f in frames], TV)
        self.assertGreater(valid.mean(), 0.5)
        ys, xs = np.nonzero(valid)
        scale = (x2 - x1) / TV[0]
        expected_x = TV[0] - 1 - (xs / scale).astype(int)
        expected_y = (ys / scale).astype(int)
        self.assertLessEqual(np.median(np.abs(screen_x[ys, xs] - expected_x)), 1)
        self.assertLessEqual(np.median(np.abs(screen_y[ys, xs] - expected_y)), 1)

    def test_recording_cut_off_before_the_tail_is_rejected(self):
        times = np.arange(120) / FPS
        levels = np.r_[np.full(30, 5.0), np.full(90, 60.0)]
        greens = np.r_[np.zeros(30), np.full(15, 40.0), np.full(75, -20.0)]
        with self.assertRaisesRegex(ValueError, "stops before"):
            find_pattern_run(times, levels, greens, 14)


if __name__ == "__main__":
    unittest.main()
