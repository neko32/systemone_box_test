"""Cat detection via the TypeSafe System One ``box`` primitive.

A Box question always returns four edges, even when nothing matches, so each
request also carries a Noul (yes/no) question to check that a cat is present.

Box returns one box per question and ignores ordinal instructions such as
"the 2nd cat from the left" (it keeps answering the most salient cat). Multi-cat
mode therefore:

1. asks Noul "at least k cats?" for k = 1..max_cats together with one Box, which
   fixes the count and locates the first cat;
2. paints every cat found so far grey and asks Box again, until the count is reached.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol

import cv2
import numpy as np
from typesafe_sdk import Box, Image, Noul, TypeSafeError

from neko_box.frame_prep import PreparedFrame, encode_jpeg, fit_to_limits
from neko_box.geometry import Detection, NormalizedBox, Probability

logger = logging.getLogger(__name__)

MODEL = "jev-preview"
IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}
DEFAULT_PROMPT = "猫はどこですか"
PRESENCE_PROMPT = "この画像に猫は写っていますか？"
COUNT_PROMPT = "この画像に猫が少なくとも{k}匹写っていますか？"
# Japanese "猫" has no singular/plural: "猫はどこですか" makes Box enclose a whole group
# of cats, so multi-cat mode asks explicitly for one cat.
SINGLE_CAT_PROMPT = "1匹の猫"
DEFAULT_PRESENCE_THRESHOLD = 0.5
DEFAULT_MAX_CATS = 1
MAX_CATS_LIMIT = 10
DUPLICATE_IOU = 0.7
MASK_COLOR = (128, 128, 128)

BOX_KEY = "cat"
PRESENCE_KEY = "present"


class DetectionError(RuntimeError):
    """A snapshot could not be analysed."""


class CatDetector(Protocol):
    """Locates cats in a BGR frame; returns an empty list when there are none."""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class SystemOneClient(Protocol):
    """The subset of ``typesafe_sdk.TypeSafeClient`` we depend on."""

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> Any: ...


def _to_box(answer: Any) -> tuple[NormalizedBox, Probability | None]:
    edges = answer.box
    box = NormalizedBox(left=edges.left, top=edges.top, right=edges.right, bottom=edges.bottom)
    confidence = None if answer.confidence is None else Probability(answer.confidence)
    return box, confidence


def _mask(image: np.ndarray, box: NormalizedBox) -> None:
    height, width = image.shape[:2]
    px = box.to_pixels(width, height)
    cv2.rectangle(image, (px.x1, px.y1), (px.x2, px.y2), MASK_COLOR, cv2.FILLED)


class TypeSafeBoxDetector:
    """``CatDetector`` backed by System One Box + Noul questions."""

    def __init__(
        self,
        client: SystemOneClient,
        *,
        prompt: str = DEFAULT_PROMPT,
        presence_threshold: float = DEFAULT_PRESENCE_THRESHOLD,
        max_cats: int = DEFAULT_MAX_CATS,
        model: str = MODEL,
    ) -> None:
        if not prompt.strip():
            msg = "prompt must not be empty"
            raise ValueError(msg)
        if not 1 <= max_cats <= MAX_CATS_LIMIT:
            msg = f"max_cats must be within [1, {MAX_CATS_LIMIT}], got {max_cats}"
            raise ValueError(msg)
        self._client = client
        self._prompt = prompt
        self._threshold = Probability(presence_threshold)
        self._max_cats = max_cats
        self._model = model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        prepared = fit_to_limits(frame)
        try:
            if self._max_cats == 1:
                return self._detect_one(prepared)
            return self._detect_many(prepared)
        except (TypeSafeError, KeyError, ValueError) as exc:
            raise DetectionError(str(exc) or type(exc).__name__) from exc

    def _ask(self, image: np.ndarray, questions: dict[str, Box | Noul]) -> Any:
        return self._client.system_one(
            state=Image(encode_jpeg(image)),
            questions=questions,
            model=self._model,
            extra_headers=IMAGES_HEADER,
        )

    def _detect_one(self, prepared: PreparedFrame) -> list[Detection]:
        result = self._ask(
            prepared.image,
            {PRESENCE_KEY: Noul(instructions=PRESENCE_PROMPT), BOX_KEY: Box(instructions=self._prompt)},
        )
        presence = Probability(result.nouls[PRESENCE_KEY].noul)
        box, confidence = _to_box(result.boxes[BOX_KEY])
        logger.debug("presence=%.3f confidence=%s box=%s", presence.value, confidence, box)
        if presence.value < self._threshold.value:
            return []
        return [Detection(prepared.to_original(box), confidence, presence)]

    def _count(self, result: Any) -> list[Probability]:
        """Presence of the 1st..k-th cat; stops at the first "no" so the count stays monotonic."""
        presences: list[Probability] = []
        for k in range(1, self._max_cats + 1):
            presence = Probability(result.nouls[f"{PRESENCE_KEY}_{k}"].noul)
            if presence.value < self._threshold.value:
                break
            presences.append(presence)
        return presences

    def _detect_many(self, prepared: PreparedFrame) -> list[Detection]:
        questions: dict[str, Box | Noul] = {
            f"{PRESENCE_KEY}_{k}": Noul(instructions=COUNT_PROMPT.format(k=k)) for k in range(1, self._max_cats + 1)
        }
        questions[BOX_KEY] = Box(instructions=SINGLE_CAT_PROMPT)
        result = self._ask(prepared.image, questions)
        presences = self._count(result)
        logger.debug("count=%d presences=%s", len(presences), [p.value for p in presences])
        if not presences:
            return []

        found: list[tuple[NormalizedBox, Probability | None, Probability]] = []
        masked = prepared.image.copy()
        for k, presence in enumerate(presences, start=1):
            if k > 1:
                result = self._ask(masked, {BOX_KEY: Box(instructions=SINGLE_CAT_PROMPT)})
            box, confidence = _to_box(result.boxes[BOX_KEY])
            logger.debug("cat %d: confidence=%s box=%s", k, confidence, box)
            if any(box.iou(prev) >= DUPLICATE_IOU for prev, _, _ in found):
                logger.debug("cat %d repeats an earlier box; stopping", k)
                break
            found.append((box, confidence, presence))
            _mask(masked, box)

        # cat#k is the k-th cat found; its presence is P(at least k cats).
        return [
            Detection(prepared.to_original(box), confidence, presence, rank=rank)
            for rank, (box, confidence, presence) in enumerate(found, start=1)
        ]
