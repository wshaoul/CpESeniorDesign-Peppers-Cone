"""Offline calibration timing and sampling, ported from Will's Windows workflow."""
import threading
import cv2
import numpy as np

TAIL_SECONDS = 3
RECORDING_MAX_WIDTH = 1920
RECORDING_SAMPLES = 5

def frame_signals(frame):
    """Whole-frame brightness and green-vs-magenta balance, cheap enough for every frame."""
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
    b, g, r = (small[:, :, c].mean() for c in range(3))
    return (b + g + r) / 3, g - (b + r) / 2


def change_peaks(times, changes, min_gap=0.3):
    """Moments where the picture changes sharply, as [(time, strength)].

    Each pattern switch is a large frame-to-frame change; stripes, solid colours
    and the final black all produce one, whatever the room lighting.
    """
    times = np.asarray(times, float)
    changes = np.asarray(changes, float)
    if changes.size < 5:
        return []
    threshold = max(0.35 * np.percentile(changes, 97), 4 * np.median(changes), 0.5)
    peaks = []
    for i in range(1, len(changes) - 1):
        if changes[i] > threshold and changes[i] >= changes[i - 1] and changes[i] >= changes[i + 1]:
            if peaks and times[i] - peaks[-1][0] < min_gap:
                if changes[i] > peaks[-1][1]:
                    peaks[-1] = (times[i], changes[i])
            else:
                peaks.append((times[i], changes[i]))
    return peaks


def follow_cadence(times, changes, start_hint):
    """Follow the evenly spaced pattern switches that begin at the green onset.

    Returns (start, end, switches, period): the black-to-green switch, the last
    switch in the chain, how many switches came after the first, and the median
    gap between them. None when no regular cadence can be found.
    """
    peaks = change_peaks(times, changes)
    if len(peaks) < 4:
        return None
    pt = np.array([p[0] for p in peaks])
    first = int(np.argmin(np.abs(pt - start_hint)))
    if abs(pt[first] - start_hint) > 0.4:
        return None
    gaps = np.diff(pt[first:first + 6])
    gaps = gaps[(gaps >= 0.3) & (gaps <= 6)]
    if len(gaps) < 3:
        return None
    period = float(np.median(gaps))

    chain, switches = [pt[first]], 0
    while True:
        for skipped in (1, 2):  # tolerate one weak switch the detector missed
            expected = chain[-1] + skipped * period
            near = np.nonzero(np.abs(pt - expected) <= 0.35 * period)[0]
            if near.size:
                chain.append(pt[max(near, key=lambda j: peaks[j][1])])
                switches += skipped
                break
        else:
            break
    return chain[0], chain[-1], switches, period


def find_pattern_run(times, levels, greens, count, changes=None):
    """Locate the played sequence in a recording.

    The green reference is the greenest moment in the video. When per-frame
    ``changes`` are supplied the pattern switches are followed from there, which
    works in a lit room and also counts the patterns actually played. Otherwise
    the sequence is taken to end when the frame drops back to black.
    Returns (start, period) in seconds.
    """
    times = np.asarray(times, float)
    levels = np.asarray(levels, float)
    greens = np.asarray(greens, float)
    peak = int(np.argmax(greens))
    if greens[peak] < 3:
        raise ValueError("No green reference frame found. Make sure the cone reflection is in the shot.")
    onset = peak
    while onset > 0 and greens[onset - 1] >= greens[peak] / 2:
        onset -= 1
    start = times[onset]

    before = levels[(times >= start - 1.0) & (times < start - 0.15)]
    if before.size == 0:
        raise ValueError("Start recording a few seconds before the TV turns green.")

    cadence = follow_cadence(times, changes, start) if changes is not None else None
    if cadence is not None:
        start, end, switches, period = cadence
        if switches == count - 1 and times[-1] - end < 1.5 * period:
            raise ValueError("The recording stops before the patterns finished. "
                             "Keep recording until the TV goes black again.")
        if switches != count:
            raise ValueError(
                f"The recording shows {switches} patterns but the selected display needs {count}. "
                "Select the display the patterns were played on. Windows display scaling "
                "changes the size the app sees, and with it the number of patterns.")
        period = (end - start) / count
        if not 0.3 <= period <= 6:
            raise ValueError(f"Found {period:.2f} s per pattern, which doesn't look like the played sequence. "
                             "Check that the selected display matches the TV the patterns were played on.")
        magenta = (times >= start + 1.25 * period) & (times <= start + 1.75 * period)
        if not magenta.any() or np.median(greens[magenta]) >= 0:
            raise ValueError("The pattern timing didn't line up (no magenta after green). Record again.")
        return start, period

    during = levels[(times >= start + 0.1) & (times < start + 0.3)]
    if during.size == 0:
        during = levels[onset:onset + 1]
    dark, bright = np.median(before), np.median(during)
    if bright - dark < 2:
        raise ValueError("The TV turning on and off is barely visible in the video. "
                         "Dim the room and lock the phone's exposure.")
    threshold = (dark + bright) / 2

    # Patterns never go black, so the first sustained dark stretch is the tail.
    end = None
    for k in range(onset + 1, len(times)):
        if levels[k] < threshold:
            window = (times >= times[k]) & (times < times[k] + 0.4)
            if np.all(levels[window] < threshold):
                end = times[k]
                break
    if end is None:
        raise ValueError("The recording stops before the patterns finished. "
                         "Keep recording until the TV goes black again.")
    period = (end - start) / count
    if not 0.3 <= period <= 6:
        raise ValueError(f"Found {period:.2f} s per pattern, which doesn't look like the played sequence. "
                         "Check that the selected display matches the TV the patterns were played on.")

    magenta = (times >= start + 1.25 * period) & (times <= start + 1.75 * period)
    if not magenta.any() or np.median(greens[magenta]) >= 0:
        raise ValueError("The pattern timing didn't line up (no magenta after green). Record again.")
    return start, period


def read_recording(path, count, report=lambda message, fraction: None):
    """Two passes over the video: find the pattern timing, then median-sample each pattern."""
    def open_video():
        try:
            with open(path, "rb"):
                pass
        except PermissionError as exc:
            raise RuntimeError("macOS denied access to this video. Choose it again with Load recording, "
                               "or copy it into the project folder with Finder and load that copy.") from exc
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError("Could not open the video. On the iPhone, try Settings > Camera > "
                               "Formats > Most Compatible, then record again.")
        return cap

    cap = open_video()
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    times, levels, greens, changes = [], [], [], []
    previous = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            level, green = frame_signals(frame)
            levels.append(level)
            greens.append(green)
            small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
            changes.append(0.0 if previous is None else float(np.abs(small - previous).mean()))
            previous = small
            if len(times) % 30 == 0:
                report(f"Scanning video… {len(times)}/{total} frames", 0.7 * min(1, len(times) / total))
    finally:
        cap.release()
    if len(times) < 2:
        raise RuntimeError("The video has no readable frames.")
    times = np.asarray(times)
    if np.any(np.diff(times) <= 0):
        times = np.arange(len(times)) / fps  # container without usable timestamps

    start, period = find_pattern_run(times, levels, greens, count, changes)

    # Sample the middle half of each pattern, away from the switch-over.
    wanted = {}
    for i in range(count):
        a, b = start + (i + 0.25) * period, start + (i + 0.75) * period
        idx = np.nonzero((times >= a) & (times <= b))[0]
        if idx.size == 0:
            raise ValueError(f"No video frames landed on pattern {i + 1}. Record at 30 fps or "
                             "raise the seconds per pattern.")
        picks = idx[np.linspace(0, idx.size - 1, min(RECORDING_SAMPLES, idx.size)).round().astype(int)]
        for k in picks:
            wanted[int(k)] = i

    samples = [[] for _ in range(count)]
    last = max(wanted)
    cap = open_video()
    try:
        for k in range(last + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if k in wanted:
                h, w = frame.shape[:2]
                if w > RECORDING_MAX_WIDTH:
                    frame = cv2.resize(frame, (RECORDING_MAX_WIDTH, round(h * RECORDING_MAX_WIDTH / w)),
                                       interpolation=cv2.INTER_AREA)
                samples[wanted[k]].append(frame)
            if k % 30 == 0:
                report("Collecting pattern frames…", 0.7 + 0.3 * k / (last + 1))
    finally:
        cap.release()
    if any(not s for s in samples):
        raise RuntimeError("The video ended early while collecting pattern frames.")
    return [np.median(np.stack(s), axis=0).astype(np.uint8) for s in samples], period


class RecordingAnalysis:
    """Runs read_recording off the UI thread; the page polls message/fraction/result."""

    def __init__(self, path, count, screen_size):
        self.path = path
        self.count = count
        self.screen_size = screen_size
        self.message = "Opening video…"
        self.fraction = 0.0
        self.result = None
        self.error = ""
        self.cancelled = False
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            self.result = read_recording(self.path, self.count, self._report)
        except Exception as exc:
            self.error = str(exc) or exc.__class__.__name__

    def _report(self, message, fraction):
        if self.cancelled:
            raise RuntimeError("Recording analysis cancelled")
        self.message, self.fraction = message, fraction

    def cancel(self):
        self.cancelled = True


