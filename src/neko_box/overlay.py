"""Draw detections onto frames."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from neko_box.geometry import Detection

# BGR colours; single-cat mode and cat#1 use the first one (warm orange).
PALETTE = (
    (80, 180, 255),  # orange
    (255, 160, 60),  # blue
    (120, 220, 90),  # green
    (200, 110, 255),  # pink
    (60, 230, 230),  # yellow
)
BOX_COLOR = PALETTE[0]
TEXT_COLOR = (255, 255, 255)
THICKNESS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8


def color_for(detection: Detection) -> tuple[int, int, int]:
    rank = detection.rank or 1
    return PALETTE[(rank - 1) % len(PALETTE)]


def draw_detections(frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Return a copy of ``frame`` with each detection's box and label drawn on it."""
    out = frame.copy()
    for detection in detections:
        _draw(out, detection)
    return out


def _draw(out: np.ndarray, detection: Detection) -> None:
    height, width = out.shape[:2]
    px = detection.box.to_pixels(width, height)
    color = color_for(detection)
    cv2.rectangle(out, (px.x1, px.y1), (px.x2, px.y2), color, THICKNESS)

    label = detection.label()
    (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, 2)
    # Put the label above the box, or inside it when the box touches the top edge.
    y_top = px.y1 - th - baseline - 6 if px.y1 >= th + baseline + 6 else px.y1
    cv2.rectangle(out, (px.x1, y_top), (px.x1 + tw + 10, y_top + th + baseline + 6), color, cv2.FILLED)
    cv2.putText(out, label, (px.x1 + 5, y_top + th + 3), FONT, FONT_SCALE, TEXT_COLOR, 2, cv2.LINE_AA)
