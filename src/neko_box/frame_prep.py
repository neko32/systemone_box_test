"""Fit video frames to the TypeSafe image limits and encode them as JPEG.

Limits (https://docs.typesafe.ai/images): each side 280..1568 px. Images outside
the limits are rejected by the API rather than resized, so we resize here.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from neko_box.geometry import NormalizedBox

MIN_SIDE = 280
MAX_SIDE = 1568
JPEG_QUALITY = 90


@dataclass(frozen=True)
class PreparedFrame:
    """A frame ready to send, plus the share of it covered by the original content.

    Padding is added only on the right and bottom, so the original content
    occupies ``[0, content_w] x [0, content_h]`` in normalized coordinates.
    """

    image: np.ndarray
    content_w: float = 1.0
    content_h: float = 1.0

    def to_original(self, box: NormalizedBox) -> NormalizedBox:
        """Map a box normalized to the padded image back to the original frame."""
        if self.content_w == 1.0 and self.content_h == 1.0:
            return box
        right = min(1.0, box.right / self.content_w)
        bottom = min(1.0, box.bottom / self.content_h)
        left = min(box.left / self.content_w, right - 1e-6)
        top = min(box.top / self.content_h, bottom - 1e-6)
        return NormalizedBox(left=max(0.0, left), top=max(0.0, top), right=right, bottom=bottom)


def fit_to_limits(frame: np.ndarray) -> PreparedFrame:
    """Resize (and if needed pad) a frame so each side lies within [MIN_SIDE, MAX_SIDE].

    Aspect ratio is kept. When both limits cannot be met (extreme aspect
    ratios) downscaling wins and the short side is padded up to MIN_SIDE.
    """
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        msg = "frame must not be empty"
        raise ValueError(msg)

    scale = 1.0
    if max(width, height) > MAX_SIDE:
        scale = MAX_SIDE / max(width, height)
    elif min(width, height) < MIN_SIDE:
        scale = min(MIN_SIDE / min(width, height), MAX_SIDE / max(width, height))

    if scale != 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        frame = cv2.resize(frame, size, interpolation=interp)

    height, width = frame.shape[:2]
    pad_bottom = max(0, MIN_SIDE - height)
    pad_right = max(0, MIN_SIDE - width)
    if not (pad_bottom or pad_right):
        return PreparedFrame(frame)
    padded = cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return PreparedFrame(padded, content_w=width / (width + pad_right), content_h=height / (height + pad_bottom))


def encode_jpeg(frame: np.ndarray) -> bytes:
    """Encode a BGR frame as a baseline JPEG."""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        msg = "failed to encode frame as JPEG"
        raise RuntimeError(msg)
    return buf.tobytes()
