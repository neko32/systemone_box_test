"""Validated value objects for box coordinates and detections."""

from __future__ import annotations

from dataclasses import dataclass


def _check_fraction(name: str, value: float) -> float:
    if not 0.0 <= value <= 1.0:
        msg = f"{name} must be within [0, 1], got {value}"
        raise ValueError(msg)
    return float(value)


@dataclass(frozen=True, slots=True)
class NormalizedBox:
    """Box edges as fractions of image width (left/right) and height (top/bottom)."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        for name in ("left", "top", "right", "bottom"):
            _check_fraction(name, getattr(self, name))
        if self.left >= self.right:
            msg = f"left ({self.left}) must be < right ({self.right})"
            raise ValueError(msg)
        if self.top >= self.bottom:
            msg = f"top ({self.top}) must be < bottom ({self.bottom})"
            raise ValueError(msg)

    @property
    def area(self) -> float:
        return (self.right - self.left) * (self.bottom - self.top)

    def iou(self, other: NormalizedBox) -> float:
        """Intersection over union with another box (0 = disjoint, 1 = identical)."""
        w = min(self.right, other.right) - max(self.left, other.left)
        h = min(self.bottom, other.bottom) - max(self.top, other.top)
        if w <= 0 or h <= 0:
            return 0.0
        inter = w * h
        return inter / (self.area + other.area - inter)

    def to_pixels(self, width: int, height: int) -> PixelBox:
        """Convert to pixel coordinates for an image of the given size."""
        if width <= 0 or height <= 0:
            msg = f"image size must be positive, got {width}x{height}"
            raise ValueError(msg)
        return PixelBox(
            x1=round(self.left * width),
            y1=round(self.top * height),
            x2=min(width - 1, round(self.right * width)),
            y2=min(height - 1, round(self.bottom * height)),
        )


@dataclass(frozen=True, slots=True)
class PixelBox:
    """Box corners in pixels: (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True, slots=True)
class Probability:
    """A probability or confidence value in [0, 1]."""

    value: float

    def __post_init__(self) -> None:
        _check_fraction("probability", self.value)


@dataclass(frozen=True, slots=True)
class Detection:
    """A cat located in a snapshot."""

    box: NormalizedBox
    confidence: Probability | None
    presence: Probability
    rank: int | None = None
    """1-based discovery order in multi-cat mode; ``None`` in single-cat mode."""

    def label(self) -> str:
        """Overlay label, e.g. ``cat 96% | box 22%`` or ``cat#2 95% | box 30%``.

        ``cat`` is the Noul presence probability; ``box`` is the Box confidence,
        which reflects how certain the edges are rather than whether a cat is there.
        ASCII only: OpenCV's Hershey fonts cannot draw other characters.
        """
        name = "cat" if self.rank is None else f"cat#{self.rank}"
        text = f"{name} {self.presence.value:.0%}"
        if self.confidence is not None:
            text += f" | box {self.confidence.value:.0%}"
        return text
