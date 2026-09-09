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
#: and caching it. Small and CPU-friendly, which matters for a hosted demo.
STOCK_WEIGHTS = "yolov8n.pt"


@dataclass(frozen=True)
class WeightsChoice:
    """Which weights were selected, and whether they are the real thing."""

    path: str
    is_custom: bool

    @property
    def display_name(self) -> str:
        return Path(self.path).stem if self.is_custom else "YOLOv8n (COCO)"

    @property
    def caveat(self) -> str:
        """A user-facing warning, or empty when the custom model is loaded.

        Stock weights were trained on COCO, which has no notion of waste
        material. They detect "bottle" and "wine glass" perfectly well, and
        the routing policy translates those into bins -- but the vocabulary
        is a general-purpose one, so accuracy on a real waste stream is not
        what a purpose-trained model would give.
        """
        if self.is_custom:
            return ""
        return (
            "Running on stock COCO weights, not a waste-trained model. "
            "Detection is limited to COCO's 80 everyday-object classes, so "
            "results are indicative rather than production-accurate."
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
