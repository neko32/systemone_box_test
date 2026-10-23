"""OpenCV-backed video input, output and display behind small protocols."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

QUIT_KEYS = {ord("q"), 27}  # q or ESC


class VideoReader(Protocol):
    fps: float
    size: tuple[int, int]

    def frames(self) -> Iterator[np.ndarray]: ...
    def release(self) -> None: ...


class VideoWriter(Protocol):
    def write(self, frame: np.ndarray) -> None: ...
    def release(self) -> None: ...


class Display(Protocol):
    def show(self, frame: np.ndarray, delay_ms: int) -> bool:
        """Show a frame; return True when the user asked to quit."""
        ...

    def close(self) -> None: ...


class CvVideoReader:
    def __init__(self, path: Path) -> None:
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            msg = f"cannot open video: {path}"
            raise FileNotFoundError(msg)
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            yield frame

    def release(self) -> None:
        self._cap.release()


class CvVideoWriter:
    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
        if not self._writer.isOpened():
            msg = f"cannot open video writer: {path}"
            raise OSError(msg)

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()


class CvDisplay:  # pragma: no cover - needs a GUI session
    def __init__(self, title: str = "neko-box") -> None:
        self._title = title
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    def show(self, frame: np.ndarray, delay_ms: int) -> bool:
        cv2.imshow(self._title, frame)
        return (cv2.waitKey(max(1, delay_ms)) & 0xFF) in QUIT_KEYS

    def close(self) -> None:
        cv2.destroyWindow(self._title)
