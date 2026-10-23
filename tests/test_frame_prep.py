import cv2
import numpy as np
import pytest

from neko_box.frame_prep import MAX_SIDE, MIN_SIDE, PreparedFrame, encode_jpeg, fit_to_limits
from neko_box.geometry import NormalizedBox


def _frame(w, h):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_frame_within_limits_is_unchanged():
    frame = _frame(1280, 720)
    prepared = fit_to_limits(frame)
    assert prepared.image is frame
    assert (prepared.content_w, prepared.content_h) == (1.0, 1.0)


def test_large_frame_is_downscaled_keeping_aspect():
    h, w = fit_to_limits(_frame(3840, 2160)).image.shape[:2]
    assert w == MAX_SIDE
    assert h == round(2160 * MAX_SIDE / 3840)


def test_small_frame_is_upscaled():
    h, w = fit_to_limits(_frame(200, 100)).image.shape[:2]
    assert min(w, h) >= MIN_SIDE
    assert max(w, h) <= MAX_SIDE


def test_extreme_aspect_is_padded_and_records_content_share():
    prepared = fit_to_limits(_frame(4000, 100))
    h, w = prepared.image.shape[:2]
    assert (w, h) == (MAX_SIDE, MIN_SIDE)
    assert prepared.content_w == 1.0
    assert prepared.content_h == pytest.approx(round(100 * MAX_SIDE / 4000) / MIN_SIDE)


def test_empty_frame_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        fit_to_limits(np.zeros((0, 0, 3), dtype=np.uint8))


@pytest.mark.parametrize("size", [(MIN_SIDE, MIN_SIDE), (MAX_SIDE, MAX_SIDE)])
def test_boundary_sizes_are_accepted(size):
    frame = _frame(*size)
    assert fit_to_limits(frame).image is frame


def test_to_original_is_identity_without_padding():
    box = NormalizedBox(0.1, 0.2, 0.3, 0.4)
    assert PreparedFrame(_frame(300, 300)).to_original(box) is box


def test_to_original_rescales_padded_coordinates():
    prepared = PreparedFrame(_frame(300, 300), content_w=1.0, content_h=0.5)
    out = prepared.to_original(NormalizedBox(0.1, 0.1, 0.5, 0.4))
    assert (out.left, out.top, out.right, out.bottom) == pytest.approx((0.1, 0.2, 0.5, 0.8))


def test_to_original_clamps_box_inside_padding():
    prepared = PreparedFrame(_frame(300, 300), content_w=1.0, content_h=0.5)
    out = prepared.to_original(NormalizedBox(0.1, 0.7, 0.5, 0.9))
    assert out.bottom == 1.0
    assert out.top < out.bottom


def test_encode_jpeg_produces_decodable_jpeg():
    data = encode_jpeg(_frame(320, 300))
    assert data[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (300, 320, 3)


def test_encode_jpeg_failure_raises(monkeypatch):
    monkeypatch.setattr(cv2, "imencode", lambda *_a, **_k: (False, None))
    with pytest.raises(RuntimeError, match="encode"):
        encode_jpeg(_frame(320, 300))
