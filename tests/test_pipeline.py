import threading

import numpy as np
import pytest

from neko_box.detector import DetectionError
from neko_box.overlay import BOX_COLOR
from neko_box.pipeline import DetectionWorker, SnapshotSchedule, play, run_live, run_offline, safe_detect
from tests.fakes import CAT, FakeDetector, FakeDisplay, FakeReader, FakeWriter


def _has_box(frame) -> bool:
    return bool((frame == np.array(BOX_COLOR, dtype=np.uint8)).all(axis=2).any())


# --- SnapshotSchedule -------------------------------------------------------


def test_schedule_one_second_at_24fps():
    s = SnapshotSchedule.from_seconds(24.0, 1.0)
    assert s.every_n_frames == 24
    assert [i for i in range(72) if s.is_snapshot(i)] == [0, 24, 48]


def test_schedule_never_below_one_frame():
    assert SnapshotSchedule.from_seconds(10.0, 0.01).every_n_frames == 1


def test_schedule_rounds_fractional_fps():
    assert SnapshotSchedule.from_seconds(29.97, 1.0).every_n_frames == 30


@pytest.mark.parametrize(("fps", "interval"), [(0, 1), (24, 0), (-1, 1)])
def test_schedule_rejects_non_positive(fps, interval):
    with pytest.raises(ValueError, match="positive"):
        SnapshotSchedule.from_seconds(fps, interval)


# --- safe_detect -------------------------------------------------------------


def test_safe_detect_swallows_detection_error(caplog):
    det = FakeDetector([DetectionError("429")])
    assert safe_detect(det, np.zeros((1, 1, 3)), 7) == []
    assert "frame 7: detection failed" in caplog.text


def test_safe_detect_logs_no_cat(caplog):
    caplog.set_level("INFO")
    assert safe_detect(FakeDetector([None]), np.zeros((1, 1, 3)), 0) == []
    assert "no cat" in caplog.text


# --- run_offline -------------------------------------------------------------


def test_offline_detects_once_per_second_and_holds_box():
    reader = FakeReader(n_frames=72)  # 3 s at 24 fps
    detector = FakeDetector([CAT, None, CAT])
    writer = FakeWriter()

    count = run_offline(reader, detector, SnapshotSchedule(24), writer)

    assert count == 72
    assert len(detector.frames) == 3
    flags = [_has_box(f) for f in writer.frames]
    assert all(flags[0:24])
    assert not any(flags[24:48])
    assert all(flags[48:72])


def test_offline_continues_after_api_error():
    writer = FakeWriter()
    run_offline(FakeReader(n_frames=48), FakeDetector([DetectionError("x"), CAT]), SnapshotSchedule(24), writer)
    flags = [_has_box(f) for f in writer.frames]
    assert not any(flags[:24])
    assert all(flags[24:])


# --- DetectionWorker ---------------------------------------------------------


class BlockingDetector:
    """Blocks inside detect() until released, to exercise the pending-slot logic."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.seen: list[int] = []

    def detect(self, frame):
        self.seen.append(int(frame[0, 0, 0]))
        self.started.set()
        self.release.wait(5)
        return [CAT]


def test_worker_publishes_latest_result():
    worker = DetectionWorker(FakeDetector([CAT]))
    try:
        assert worker.latest == []
        worker.submit(np.zeros((2, 2, 3), np.uint8), 0)
        worker.wait_idle()
        assert worker.latest == [CAT]
    finally:
        worker.close()


def test_worker_replaces_stale_pending_snapshot():
    detector = BlockingDetector()
    worker = DetectionWorker(detector)
    try:
        worker.submit(np.full((2, 2, 3), 1, np.uint8), 0)
        assert detector.started.wait(5)
        worker.submit(np.full((2, 2, 3), 2, np.uint8), 24)  # pending
        worker.submit(np.full((2, 2, 3), 3, np.uint8), 48)  # replaces pending
        detector.release.set()
        worker.wait_idle()
        assert detector.seen == [1, 3]
    finally:
        detector.release.set()
        worker.close()


def test_worker_copies_submitted_frame():
    detector = FakeDetector([CAT])
    worker = DetectionWorker(detector)
    frame = np.zeros((2, 2, 3), np.uint8)
    try:
        worker.submit(frame, 0)
        frame[:] = 9
        worker.wait_idle()
        assert not detector.frames[0].any()
    finally:
        worker.close()


# --- run_live ----------------------------------------------------------------


def test_live_shows_and_writes_every_frame():
    reader = FakeReader(n_frames=48)
    detector = FakeDetector([CAT, CAT])
    display = FakeDisplay()
    writer = FakeWriter()

    shown = run_live(reader, detector, SnapshotSchedule(24), display, writer)

    assert shown == 48
    assert display.shown == 48
    assert len(writer.frames) == 48
    assert 1 <= len(detector.frames) <= 2


def test_live_stops_when_user_quits():
    display = FakeDisplay(quit_after=5)
    shown = run_live(FakeReader(n_frames=48), FakeDetector([]), SnapshotSchedule(24), display)
    assert shown == 5


def test_live_eventually_draws_box():
    # 1 fps so each frame is a snapshot; the worker result appears on a later frame.
    reader = FakeReader(n_frames=200)
    display = FakeDisplay()
    writer = FakeWriter()
    run_live(reader, FakeDetector([CAT] * 200), SnapshotSchedule(1), display, writer)
    assert any(_has_box(f) for f in writer.frames)


# --- play ----------------------------------------------------------------


def test_play_shows_all_frames():
    display = FakeDisplay()
    play(FakeReader(n_frames=10), display)
    assert display.shown == 10


def test_play_stops_on_quit():
    display = FakeDisplay(quit_after=3)
    play(FakeReader(n_frames=10), display)
    assert display.shown == 3


def test_safe_detect_logs_every_cat(caplog):
    caplog.set_level("INFO")
    two = [CAT, CAT]
    assert safe_detect(FakeDetector([two]), np.zeros((1, 1, 3)), 3) == two
    assert "frame 3: cat 95% | box 90%, cat 95% | box 90%" in caplog.text
