"""Snapshot scheduling, background detection and the live/offline loops."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

import numpy as np

from neko_box.detector import CatDetector, DetectionError
from neko_box.geometry import Detection
from neko_box.overlay import draw_detections
from neko_box.video_io import Display, VideoReader, VideoWriter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SnapshotSchedule:
    """Selects one frame every ``interval_s`` seconds of video."""

    every_n_frames: int

    @classmethod
    def from_seconds(cls, fps: float, interval_s: float) -> SnapshotSchedule:
        if fps <= 0 or interval_s <= 0:
            msg = f"fps and interval must be positive, got fps={fps} interval={interval_s}"
            raise ValueError(msg)
        return cls(max(1, round(fps * interval_s)))

    def is_snapshot(self, frame_index: int) -> bool:
        return frame_index % self.every_n_frames == 0


def safe_detect(detector: CatDetector, frame: np.ndarray, frame_index: int) -> list[Detection]:
    """Run detection, logging (not raising) API failures so playback keeps going."""
    try:
        detections = detector.detect(frame)
    except DetectionError as exc:
        logger.warning("frame %d: detection failed: %s", frame_index, exc)
        return []
    summary = ", ".join(d.label() for d in detections) or "no cat"
    logger.info("frame %d: %s", frame_index, summary)
    return detections


class DetectionWorker:
    """Runs detection on a background thread so playback never blocks on the API.

    Holds at most one pending snapshot: a newer snapshot replaces an unstarted one.
    """

    _STOP = object()

    def __init__(self, detector: CatDetector) -> None:
        self._detector = detector
        self._queue: queue.Queue[object] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest: list[Detection] = []
        self._thread = threading.Thread(target=self._run, name="detection-worker", daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray, frame_index: int) -> None:
        item = (frame.copy(), frame_index)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                logger.debug("dropping stale snapshot in favour of frame %d", frame_index)
            except queue.Empty:  # pragma: no cover - worker took it meanwhile
                pass
            self._queue.put(item)

    @property
    def latest(self) -> list[Detection]:
        with self._lock:
            return self._latest

    def wait_idle(self) -> None:
        """Block until every submitted snapshot has been processed."""
        self._queue.join()

    def close(self) -> None:
        self._queue.put(self._STOP)
        self._thread.join()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                frame, index = item  # type: ignore[misc]
                result = safe_detect(self._detector, frame, index)
                with self._lock:
                    self._latest = result
            finally:
                self._queue.task_done()


def run_live(
    reader: VideoReader,
    detector: CatDetector,
    schedule: SnapshotSchedule,
    display: Display,
    writer: VideoWriter | None = None,
) -> int:
    """Play the video, detecting asynchronously; the latest boxes stay until replaced.

    Returns the number of frames shown.
    """
    delay_ms = max(1, round(1000 / reader.fps))
    worker = DetectionWorker(detector)
    shown = 0
    try:
        for index, frame in enumerate(reader.frames()):
            if schedule.is_snapshot(index):
                worker.submit(frame, index)
            annotated = draw_detections(frame, worker.latest)
            if writer is not None:
                writer.write(annotated)
            shown += 1
            if display.show(annotated, delay_ms):
                logger.info("quit requested")
                break
    finally:
        worker.close()
    return shown


def run_offline(
    reader: VideoReader,
    detector: CatDetector,
    schedule: SnapshotSchedule,
    writer: VideoWriter,
) -> int:
    """Detect synchronously at each snapshot and write a perfectly aligned annotated video.

    Returns the number of frames written.
    """
    latest: list[Detection] = []
    written = 0
    for index, frame in enumerate(reader.frames()):
        if schedule.is_snapshot(index):
            latest = safe_detect(detector, frame, index)
        writer.write(draw_detections(frame, latest))
        written += 1
    return written


def play(reader: VideoReader, display: Display) -> None:
    """Play a video as-is (used to show the offline result)."""
    delay_ms = max(1, round(1000 / reader.fps))
    for frame in reader.frames():
        if display.show(frame, delay_ms):
            return
