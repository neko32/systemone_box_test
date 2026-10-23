import numpy as np

from neko_box.geometry import Detection, NormalizedBox, Probability
from neko_box.overlay import BOX_COLOR, PALETTE, color_for, draw_detections


def _frame():
    return np.zeros((300, 400, 3), dtype=np.uint8)


def _det(box, rank=None):
    return Detection(NormalizedBox(*box), Probability(0.9), Probability(0.9), rank=rank)


def test_empty_list_returns_unmodified_copy():
    frame = _frame()
    out = draw_detections(frame, [])
    assert out is not frame
    assert not out.any()


def test_box_edges_are_drawn_and_input_untouched():
    frame = _frame()
    out = draw_detections(frame, [_det((0.25, 0.5, 0.75, 0.9))])
    assert not frame.any()
    # left edge at x=100, mid-height of the box
    assert tuple(out[220, 100]) == BOX_COLOR
    # centre of the box stays empty
    assert not out[220, 200].any()


def test_label_fits_inside_when_box_touches_top():
    out = draw_detections(_frame(), [Detection(NormalizedBox(0.0, 0.0, 0.5, 0.5), None, Probability(0.7))])
    # filled label background starts at the top-left corner
    assert tuple(out[2, 2]) == BOX_COLOR


def test_each_cat_gets_its_own_colour():
    out = draw_detections(_frame(), [_det((0.05, 0.5, 0.3, 0.9), rank=1), _det((0.6, 0.5, 0.9, 0.9), rank=2)])
    assert tuple(out[220, 20]) == PALETTE[0]  # left edge of cat#1 (x=20)
    assert tuple(out[220, 240]) == PALETTE[1]  # left edge of cat#2 (x=240)


def test_colours_cycle_and_single_mode_uses_first():
    assert color_for(_det((0, 0, 1, 1))) == PALETTE[0]
    assert color_for(_det((0, 0, 1, 1), rank=len(PALETTE) + 1)) == PALETTE[0]
