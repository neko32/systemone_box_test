import pytest

from neko_box.geometry import Detection, NormalizedBox, PixelBox, Probability


def test_to_pixels_scales_edges():
    box = NormalizedBox(left=0.25, top=0.5, right=0.75, bottom=1.0)
    assert box.to_pixels(1280, 720) == PixelBox(x1=320, y1=360, x2=960, y2=719)


def test_to_pixels_clamps_right_and_bottom_inside_image():
    box = NormalizedBox(left=0.0, top=0.0, right=1.0, bottom=1.0)
    assert box.to_pixels(100, 50) == PixelBox(x1=0, y1=0, x2=99, y2=49)


@pytest.mark.parametrize(("w", "h"), [(0, 10), (10, 0), (-1, 10)])
def test_to_pixels_rejects_non_positive_size(w, h):
    with pytest.raises(ValueError, match="positive"):
        NormalizedBox(0.1, 0.1, 0.2, 0.2).to_pixels(w, h)


@pytest.mark.parametrize(
    "edges",
    [
        (-0.1, 0.0, 0.5, 0.5),
        (0.0, 0.0, 1.1, 0.5),
        (0.0, -0.01, 0.5, 0.5),
        (0.0, 0.0, 0.5, 1.5),
    ],
)
def test_box_rejects_out_of_range(edges):
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        NormalizedBox(*edges)


def test_box_rejects_inverted_horizontal():
    with pytest.raises(ValueError, match="left"):
        NormalizedBox(left=0.5, top=0.0, right=0.5, bottom=1.0)


def test_box_rejects_inverted_vertical():
    with pytest.raises(ValueError, match="top"):
        NormalizedBox(left=0.0, top=0.6, right=1.0, bottom=0.2)


@pytest.mark.parametrize("value", [-0.001, 1.001])
def test_probability_rejects_out_of_range(value):
    with pytest.raises(ValueError, match="probability"):
        Probability(value)


def test_label_shows_presence_and_box_confidence():
    det = Detection(NormalizedBox(0, 0, 1, 1), Probability(0.22), Probability(0.96))
    assert det.label() == "cat 96% | box 22%"


def test_label_is_ascii_for_opencv_fonts():
    det = Detection(NormalizedBox(0, 0, 1, 1), Probability(0.22), Probability(0.96))
    assert det.label().isascii()


def test_label_omits_box_when_confidence_missing():
    det = Detection(NormalizedBox(0, 0, 1, 1), None, Probability(0.81))
    assert det.label() == "cat 81%"


def test_label_includes_rank_in_multi_cat_mode():
    det = Detection(NormalizedBox(0, 0, 1, 1), Probability(0.30), Probability(0.95), rank=2)
    assert det.label() == "cat#2 95% | box 30%"


def test_iou_identical_boxes_is_one():
    box = NormalizedBox(0.1, 0.1, 0.5, 0.5)
    assert box.iou(box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert NormalizedBox(0.0, 0.0, 0.2, 0.2).iou(NormalizedBox(0.5, 0.5, 0.9, 0.9)) == 0.0


def test_iou_touching_edges_is_zero():
    assert NormalizedBox(0.0, 0.0, 0.5, 0.5).iou(NormalizedBox(0.5, 0.0, 1.0, 0.5)) == 0.0


def test_iou_half_overlap():
    a = NormalizedBox(0.0, 0.0, 0.4, 0.4)
    b = NormalizedBox(0.2, 0.0, 0.6, 0.4)
    # intersection 0.08, union 0.16 + 0.16 - 0.08 = 0.24
    assert a.iou(b) == pytest.approx(1 / 3)
