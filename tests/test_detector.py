import cv2
import numpy as np
import pytest
from typesafe_sdk import Box, Image, Noul, TypeSafeError

from neko_box.detector import DEFAULT_PROMPT, MASK_COLOR, DetectionError, TypeSafeBoxDetector
from neko_box.geometry import NormalizedBox, Probability
from tests.fakes import FakeClient, make_box_response, make_count_response, make_response


def _frame(w=1280, h=720):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_detect_sends_box_and_noul_questions_with_image_header():
    client = FakeClient()
    TypeSafeBoxDetector(client).detect(_frame())

    (call,) = client.calls
    assert isinstance(call["state"], Image)
    assert call["model"] == "jev-preview"
    assert call["extra_headers"] == {"TypeSafe-Danger-Zone-Images": "true"}
    assert isinstance(call["questions"]["present"], Noul)
    box_q = call["questions"]["cat"]
    assert isinstance(box_q, Box)
    assert box_q.instructions == DEFAULT_PROMPT == "猫はどこですか"


def test_detect_returns_detection_when_cat_present():
    (det,) = TypeSafeBoxDetector(FakeClient(make_response(noul=0.9, confidence=0.8))).detect(_frame())
    assert det.rank is None
    assert det.box == NormalizedBox(0.1, 0.2, 0.5, 0.6)
    assert det.confidence == Probability(0.8)
    assert det.presence == Probability(0.9)


def test_detect_returns_none_when_cat_absent():
    assert TypeSafeBoxDetector(FakeClient(make_response(noul=0.1))).detect(_frame()) == []


def test_threshold_boundary_counts_as_present():
    assert len(TypeSafeBoxDetector(FakeClient(make_response(noul=0.5)), presence_threshold=0.5).detect(_frame())) == 1


def test_missing_confidence_is_allowed():
    (det,) = TypeSafeBoxDetector(FakeClient(make_response(confidence=None))).detect(_frame())
    assert det.confidence is None


def test_custom_prompt_is_forwarded():
    client = FakeClient()
    TypeSafeBoxDetector(client, prompt="the orange cat").detect(_frame())
    assert client.calls[0]["questions"]["cat"].instructions == "the orange cat"


def test_padded_frame_box_is_mapped_back_to_original():
    # 4000x100 is padded vertically; content occupies the top part only.
    client = FakeClient(make_response(top=0.0, bottom=0.1, left=0.0, right=1.0))
    (det,) = TypeSafeBoxDetector(client).detect(_frame(4000, 100))
    assert det.box.bottom > 0.1


@pytest.mark.parametrize(
    "error",
    [TypeSafeError("boom"), KeyError("cat")],
)
def test_client_errors_are_wrapped(error):
    with pytest.raises(DetectionError):
        TypeSafeBoxDetector(FakeClient(error=error)).detect(_frame())


def test_invalid_box_from_api_is_wrapped():
    client = FakeClient(make_response(left=0.6, right=0.4))
    with pytest.raises(DetectionError, match="left"):
        TypeSafeBoxDetector(client).detect(_frame())


def test_empty_prompt_is_rejected():
    with pytest.raises(ValueError, match="prompt"):
        TypeSafeBoxDetector(FakeClient(), prompt="  ")


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError, match="probability"):
        TypeSafeBoxDetector(FakeClient(), presence_threshold=1.5)


# --- multi-cat mode ----------------------------------------------------------

LEFT = (0.05, 0.3, 0.25, 0.8)
MIDDLE = (0.40, 0.3, 0.60, 0.8)
RIGHT = (0.70, 0.3, 0.95, 0.8)


def _decode(image: Image) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(bytes(image.data), np.uint8), cv2.IMREAD_COLOR)


def test_multi_cat_first_request_counts_and_locates_first_cat():
    client = FakeClient(make_count_response([0.9, 0.1, 0.1], LEFT))
    TypeSafeBoxDetector(client, max_cats=3).detect(_frame())

    (call,) = client.calls
    questions = call["questions"]
    assert list(questions) == ["present_1", "present_2", "present_3", "cat"]
    assert questions["present_2"].instructions == "この画像に猫が少なくとも2匹写っていますか？"
    assert questions["cat"].instructions == "1匹の猫"


def test_multi_cat_asks_again_on_masked_image_until_count_reached():
    client = FakeClient(
        [
            make_count_response([0.95, 0.9, 0.85, 0.1], LEFT, 0.8),
            make_box_response(MIDDLE, 0.6),
            make_box_response(RIGHT, 0.5),
        ]
    )
    dets = TypeSafeBoxDetector(client, max_cats=4).detect(_frame())

    assert len(client.calls) == 3
    assert list(client.calls[1]["questions"]) == ["cat"]
    assert [d.rank for d in dets] == [1, 2, 3]
    assert [d.box for d in dets] == [NormalizedBox(*LEFT), NormalizedBox(*MIDDLE), NormalizedBox(*RIGHT)]
    assert dets[1].label() == "cat#2 90% | box 60%"


def test_multi_cat_masks_found_cats_grey():
    client = FakeClient([make_count_response([0.9, 0.9], LEFT), make_box_response(RIGHT)])
    TypeSafeBoxDetector(client, max_cats=2).detect(_frame())

    first, second = (_decode(c["state"]) for c in client.calls)
    centre_of_left = (int(0.55 * 720), int(0.15 * 1280))
    assert not first[centre_of_left].any()  # original frame is black
    assert abs(int(second[centre_of_left][0]) - MASK_COLOR[0]) < 8  # masked grey (JPEG tolerance)


def test_multi_cat_stops_at_first_no():
    # "at least 3" yes but "at least 2" no is inconsistent: trust the first "no".
    client = FakeClient(make_count_response([0.9, 0.2, 0.9], LEFT))
    dets = TypeSafeBoxDetector(client, max_cats=3).detect(_frame())
    assert [d.rank for d in dets] == [1]
    assert len(client.calls) == 1


def test_multi_cat_no_cats_makes_one_request():
    client = FakeClient(make_count_response([0.05, 0.03], LEFT, 0.02))
    assert TypeSafeBoxDetector(client, max_cats=2).detect(_frame()) == []
    assert len(client.calls) == 1


def test_multi_cat_stops_when_masked_image_repeats_a_box():
    almost_left = (0.06, 0.31, 0.25, 0.8)
    client = FakeClient(
        [
            make_count_response([0.9, 0.9, 0.9], LEFT),
            make_box_response(almost_left),
            make_box_response(RIGHT),
        ]
    )
    dets = TypeSafeBoxDetector(client, max_cats=3).detect(_frame())
    assert [d.rank for d in dets] == [1]
    assert len(client.calls) == 2


def test_multi_cat_missing_answer_is_wrapped():
    client = FakeClient(make_count_response([0.9], LEFT))  # present_2 missing
    with pytest.raises(DetectionError):
        TypeSafeBoxDetector(client, max_cats=2).detect(_frame())


@pytest.mark.parametrize("max_cats", [0, 11])
def test_max_cats_out_of_range_is_rejected(max_cats):
    with pytest.raises(ValueError, match="max_cats"):
        TypeSafeBoxDetector(FakeClient(), max_cats=max_cats)
