"""Locating model weights, and making sure there are always some.

A demo nobody can start is worth nothing, so the rule here is simple: the
app must run from a fresh clone with no manual setup. Custom weights are
preferred when present, stock COCO weights are fetched when they are not,
and the UI is told which of the two it got so it can say so honestly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Where a custom, conveyor-trained model is expected to live once one exists.
CUSTOM_WEIGHTS = PROJECT_ROOT / "models" / "best_model.pt"

#: Ultralytics resolves a bare name like this by downloading it on first use
#: and caching it. `s` rather than `n`: on the conveyor sample images, `n`
#: finds nothing at all at a sane threshold while `s` finds six items in
#: ~70ms on CPU. `m` adds ~1 item for 2.3x the download and slower inference.
STOCK_WEIGHTS = "yolov8s.pt"


@dataclass(frozen=True)
class WeightsChoice:
    """Which weights were selected, and whether they are the real thing."""

    path: str
    is_custom: bool

    @property
    def display_name(self) -> str:
        return Path(self.path).stem if self.is_custom else "YOLOv8s (COCO)"

    @property
    def caveat(self) -> str:
        """A user-facing warning, or empty when the custom model is loaded.

        Stock weights were trained on COCO, whose 80 classes describe
        everyday objects rather than waste. The gap is not subtle: COCO has
        no class for a drink can -- the single most common item in a real
        recycling stream -- so cans get reported as "cup" or "bowl" and
        routed accordingly. That is exactly the case for training a
        purpose-built model, and exactly why this caveat is shown rather
        than buried.
        """
        if self.is_custom:
            return ""
        return (
            "Running on stock COCO weights, not a waste-trained model. COCO "
            "has no class for a drink can, so cans are misread as cups or "
            "bowls and routed wrongly. Low-certainty routes are flagged for "
            "review below. Treat results as indicative, not accurate."
        )


def resolve_weights(custom_path: str | Path | None = None) -> WeightsChoice:
    """Pick the best available weights.

    Prefers a custom model, falls back to stock. Never raises: the fallback
    is always available because ultralytics downloads it on demand.
    """
    candidate = Path(custom_path) if custom_path is not None else CUSTOM_WEIGHTS

    if candidate.is_file():
        logger.info("using custom weights at %s", candidate)
        return WeightsChoice(path=str(candidate), is_custom=True)

    logger.info("no custom weights at %s; falling back to %s", candidate, STOCK_WEIGHTS)
    return WeightsChoice(path=STOCK_WEIGHTS, is_custom=False)
