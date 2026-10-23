"""Test doubles shared across unit tests."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np

from neko_box.geometry import Detection, NormalizedBox, Probability


def make_response(*, noul=0.9, left=0.1, top=0.2, right=0.5, bottom=0.6, confidence=0.8):
    """Mimic the attributes of ``typesafe_sdk.SystemOneResponse`` the detector reads."""
    return SimpleNamespace(
        nouls={"present": SimpleNamespace(noul=noul)},
        boxes={
            "cat": SimpleNamespace(
                box=SimpleNamespace(left=left, top=top, right=right, bottom=bottom),
                confidence=confidence,
            )
        },
    )


def _box(edges, confidence):
    left, top, right, bottom = edges
    return SimpleNamespace(box=SimpleNamespace(left=left, top=top, right=right, bottom=bottom), confidence=confidence)


def make_count_response(nouls, edges, confidence=0.8):
    """First multi-cat response: ``present_1..n`` Nouls plus the first cat's box."""
    return SimpleNamespace(
        nouls={f"present_{k}": SimpleNamespace(noul=v) for k, v in enumerate(nouls, start=1)},
        boxes={"cat": _box(edges, confidence)},
    )


def make_box_response(edges, confidence=0.6):
    """Follow-up multi-cat response on the masked image: one box only."""
    return SimpleNamespace(nouls={}, boxes={"cat": _box(edges, confidence)})


class FakeClient:
    """Records calls and returns a canned response (or raises)."""

    def __init__(self, response=None, error: Exception | None = None):
        # A list is served one response per call; anything else is reused for every call.
        self.response = response if response is not None else make_response()
        self.error = error
        self.calls: list[dict] = []

    def system_one(self, state, questions, *, model=None, extra_headers=None):
        self.calls.append({"state": state, "questions": questions, "model": model, "extra_headers": extra_headers})
        if self.error:
            raise self.error
        if isinstance(self.response, list):
            return self.response[len(self.calls) - 1]
        return self.response


CAT = Detection(NormalizedBox(0.25, 0.25, 0.75, 0.75), Probability(0.9), Probability(0.95))


class FakeDetector:
    """Returns scripted results in order; an Exception entry is raised.

    Entries may be a list of detections, a single ``Detection`` or ``None`` (no cat).
    """

    def __init__(self, results):
        self._results = list(results)
        self.frames: list[np.ndarray] = []

    def detect(self, frame):
        self.frames.append(frame)
        item = self._results.pop(0) if self._results else None
        if isinstance(item, Exception):
            raise item
        if item is None:
            return []
        return item if isinstance(item, list) else [item]


class FakeReader:
    def __init__(self, n_frames=48, fps=24.0, size=(320, 300)):
        self.fps = fps
        self.size = size
        self._n = n_frames
        self.released = False

    def frames(self) -> Iterator[np.ndarray]:
        w, h = self.size
        for i in range(self._n):
            yield np.full((h, w, 3), i % 256, dtype=np.uint8)

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self):
        self.frames: list[np.ndarray] = []
        self.released = False

    def write(self, frame):
        self.frames.append(frame.copy())

    def release(self):
        self.released = True


class FakeDisplay:
    def __init__(self, quit_after: int | None = None):
        self.shown = 0
        self.quit_after = quit_after
        self.closed = False

    def show(self, frame, delay_ms) -> bool:
        self.shown += 1
        return self.quit_after is not None and self.shown >= self.quit_after

    def close(self):
        self.closed = True
