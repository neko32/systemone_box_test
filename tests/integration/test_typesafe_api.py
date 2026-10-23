"""Integration tests against the real TypeSafe API (needs API_KEY_JEV; costs tokens).

Run with: uv run pytest -m integration
"""

import os
from pathlib import Path

import pytest
from typesafe_sdk import TypeSafeClient

from neko_box.detector import TypeSafeBoxDetector
from neko_box.video_io import CvVideoReader

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("API_KEY_JEV"), reason="API_KEY_JEV not set"),
]

MOVIES = Path(__file__).resolve().parents[2] / "movie"
NEKO = MOVIES / "neko.mp4"
MANY_CATS = MOVIES / "many_cats.mp4"


@pytest.fixture(scope="module")
def client():
    with TypeSafeClient(api_key=os.environ["API_KEY_JEV"], timeout=60.0) as c:
        yield c


def _frames_every_second(video: Path, limit: int):
    reader = CvVideoReader(video)
    try:
        step = round(reader.fps)
        for i, frame in enumerate(reader.frames()):
            if i % step == 0:
                yield i, frame
                limit -= 1
                if limit == 0:
                    return
    finally:
        reader.release()


def _assert_valid(detections):
    for det in detections:
        assert 0.0 <= det.box.left < det.box.right <= 1.0
        assert 0.0 <= det.box.top < det.box.bottom <= 1.0


@pytest.mark.skipif(not NEKO.exists(), reason="movie/neko.mp4 missing")
def test_real_api_finds_a_cat_somewhere_in_the_video(client):
    detector = TypeSafeBoxDetector(client)
    results = {i: detector.detect(frame) for i, frame in _frames_every_second(NEKO, limit=10)}
    found = {i: dets for i, dets in results.items() if dets}
    assert found, f"no cat found in any snapshot: {results}"
    for dets in found.values():
        assert len(dets) == 1
        _assert_valid(dets)


@pytest.mark.skipif(not MANY_CATS.exists(), reason="movie/many_cats.mp4 missing")
def test_real_api_finds_several_cats_at_once(client):
    detector = TypeSafeBoxDetector(client, max_cats=5)
    results = {i: detector.detect(frame) for i, frame in _frames_every_second(MANY_CATS, limit=10)}
    assert max(len(d) for d in results.values()) >= 2, f"never more than one cat: {results}"
    for dets in results.values():
        _assert_valid(dets)
        assert [d.rank for d in dets] == sorted(d.rank for d in dets)
