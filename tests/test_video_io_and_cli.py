"""Real OpenCV file I/O against a generated clip (no network)."""

import numpy as np
import pytest

from neko_box import cli, video_io
from neko_box.video_io import CvVideoReader, CvVideoWriter
from tests.fakes import CAT, FakeDetector, FakeDisplay


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = CvVideoWriter(path, 10.0, (320, 288))
    for i in range(25):
        writer.write(np.full((288, 320, 3), i * 10, dtype=np.uint8))
    writer.release()
    return path


def test_reader_reads_metadata_and_frames(clip):
    reader = CvVideoReader(clip)
    try:
        assert reader.size == (320, 288)
        assert reader.fps == pytest.approx(10.0)
        assert sum(1 for _ in reader.frames()) == 25
    finally:
        reader.release()


def test_reader_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        CvVideoReader(tmp_path / "nope.mp4")


def test_writer_unopenable_path(tmp_path, monkeypatch):
    class ClosedWriter:
        fourcc = staticmethod(lambda *_a: 0)

        def __init__(self, *_a):
            pass

        def isOpened(self):  # noqa: N802 - mirrors cv2 API
            return False

    monkeypatch.setattr(video_io.cv2, "VideoWriter", ClosedWriter)
    with pytest.raises(OSError, match="writer"):
        CvVideoWriter(tmp_path / "x.mp4", 10.0, (10, 10))


def test_parse_args_defaults():
    args = cli.parse_args([])
    assert args.mode == "live"
    assert args.interval == 1.0
    assert args.prompt == "猫はどこですか"


def test_read_api_key_from_api_key_jev():
    assert cli.read_api_key({"API_KEY_JEV": "  secret  "}) == "secret"


def test_read_api_key_missing():
    with pytest.raises(SystemExit, match="API_KEY_JEV"):
        cli.read_api_key({})


def test_run_offline_no_display_writes_output(clip, tmp_path):
    out = tmp_path / "out" / "boxed.mp4"
    args = cli.parse_args(["--input", str(clip), "--output", str(out), "--mode", "offline", "--no-display"])
    detector = FakeDetector([CAT, None, CAT])

    count = cli.run(args, detector)

    assert count == 25
    assert len(detector.frames) == 3  # frames 0, 10, 20
    result = CvVideoReader(out)
    try:
        assert sum(1 for _ in result.frames()) == 25
    finally:
        result.release()


def test_run_offline_with_display_plays_result(clip, tmp_path):
    display = FakeDisplay()
    args = cli.parse_args(["--input", str(clip), "--output", str(tmp_path / "o.mp4"), "--mode", "offline"])
    cli.run(args, FakeDetector([]), display_factory=lambda: display)
    assert display.shown == 25
    assert display.closed


def test_run_live_with_display(clip, tmp_path):
    display = FakeDisplay()
    args = cli.parse_args(["--input", str(clip), "--output", str(tmp_path / "o.mp4")])
    assert cli.run(args, FakeDetector([CAT] * 3), display_factory=lambda: display) == 25
    assert display.closed


def test_run_live_without_display_is_rejected(clip, tmp_path):
    args = cli.parse_args(["--input", str(clip), "--output", str(tmp_path / "o.mp4"), "--no-display"])
    with pytest.raises(SystemExit, match="live mode needs a display"):
        cli.run(args, FakeDetector([]))


def test_parse_args_max_cats():
    assert cli.parse_args([]).max_cats == 1
    assert cli.parse_args(["--max-cats", "5"]).max_cats == 5


@pytest.mark.parametrize("value", ["0", "11"])
def test_parse_args_rejects_bad_max_cats(value):
    with pytest.raises(SystemExit):
        cli.parse_args(["--max-cats", value])
