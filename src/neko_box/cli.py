"""Command line entry point."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

from dotenv import load_dotenv

from neko_box.detector import (
    DEFAULT_MAX_CATS,
    DEFAULT_PRESENCE_THRESHOLD,
    DEFAULT_PROMPT,
    MAX_CATS_LIMIT,
    CatDetector,
    TypeSafeBoxDetector,
)
from neko_box.pipeline import SnapshotSchedule, play, run_live, run_offline
from neko_box.video_io import CvDisplay, CvVideoReader, CvVideoWriter, Display

API_KEY_ENV = "API_KEY_JEV"

logger = logging.getLogger("neko_box")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="neko-box", description="Find cats in a video with TypeSafe System One box.")
    p.add_argument("--input", type=Path, default=Path("movie/neko.mp4"), help="input video")
    p.add_argument("--output", type=Path, default=Path("out/neko_boxed.mp4"), help="annotated output video")
    p.add_argument("--mode", choices=("live", "offline"), default="live", help="live: play while detecting; offline: analyse then write")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between snapshots")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="box instructions sent to System One (single-cat mode)")
    p.add_argument(
        "--max-cats",
        type=int,
        default=DEFAULT_MAX_CATS,
        choices=range(1, MAX_CATS_LIMIT + 1),
        metavar=f"1..{MAX_CATS_LIMIT}",
        help="cats to look for per snapshot; >1 asks one Box per cat in the same request",
    )
    p.add_argument("--threshold", type=float, default=DEFAULT_PRESENCE_THRESHOLD, help="min presence probability")
    p.add_argument("--no-display", action="store_true", help="do not open a window")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def read_api_key(env: dict[str, str] | os._Environ[str] = os.environ) -> str:
    key = env.get(API_KEY_ENV, "").strip()
    if not key:
        msg = f"environment variable {API_KEY_ENV} is not set"
        raise SystemExit(msg)
    return key


def build_detector(args: argparse.Namespace) -> CatDetector:  # pragma: no cover - wires the real SDK
    from typesafe_sdk import TypeSafeClient  # noqa: PLC0415

    client = TypeSafeClient(api_key=read_api_key(), timeout=60.0)
    return TypeSafeBoxDetector(client, prompt=args.prompt, presence_threshold=args.threshold, max_cats=args.max_cats)


def run(
    args: argparse.Namespace,
    detector: CatDetector,
    display_factory: Callable[[], Display] = CvDisplay,
) -> int:
    """Run the selected mode; returns the number of processed frames."""
    reader = CvVideoReader(args.input)
    schedule = SnapshotSchedule.from_seconds(reader.fps, args.interval)
    writer = CvVideoWriter(args.output, reader.fps, reader.size)
    display = None if args.no_display else display_factory()
    logger.info("%s: %dx%d @ %.1f fps, snapshot every %d frames", args.input, *reader.size, reader.fps, schedule.every_n_frames)
    try:
        if args.mode == "live":
            if display is None:
                msg = "live mode needs a display; use --mode offline with --no-display"
                raise SystemExit(msg)
            count = run_live(reader, detector, schedule, display, writer)
        else:
            count = run_offline(reader, detector, schedule, writer)
    finally:
        reader.release()
        writer.release()
    logger.info("wrote %d frames to %s", count, args.output)
    if display is not None:
        try:
            if args.mode == "offline":
                result = CvVideoReader(args.output)
                try:
                    play(result, display)
                finally:
                    result.release()
        finally:
            display.close()
    return count


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - thin wrapper
    load_dotenv()
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("typesafe_sdk", "httpx2", "httpcore2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run(args, build_detector(args))
