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
    #: Overrides the derived name. Set by detectors that know better -- an
    #: open-vocabulary model is identified by its vocabulary, not its file.
    display_name_override: str = ""
    #: Overrides the derived caveat. A fine-tuned model is not stock, but it
    #: is not unconditionally trustworthy either -- see `caveat`.
    caveat_override: str = ""

    @property
    def display_name(self) -> str:
        if self.display_name_override:
            return self.display_name_override
        return Path(self.path).stem if self.is_custom else "YOLOv8s (COCO)"

    @property
    def caveat(self) -> str:
        """A user-facing warning, or empty when the custom model is loaded.

        Stock weights were trained on COCO, whose 80 classes describe
        everyday objects rather than waste. The gap is not subtle: COCO has
        no class for a drink can -- the single most common item in a real
        recycling stream -- so cans arrive labelled "cup" or "bowl". A
        context-aware policy can absorb some of that (see the MRF policy),
        but no rule file recovers information the model never had. That is
        the case for training a purpose-built model, and why this caveat is
        shown rather than buried.
        """
        if self.caveat_override:
            return self.caveat_override
        if self.is_custom:
            return ""
        return (
            "Running on stock COCO weights, not a waste-trained model. COCO's "
            "vocabulary is everyday objects and has no class for a drink can, "
            "so containers arrive labelled 'cup' or 'bowl'. A routing policy "
            "can compensate for that where it knows the context, but it cannot "
            "recover what the model never saw. Treat results as indicative."
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

    if not candidate.parent.is_dir():
        # Created rather than reported: the directory is gitignored, so a fresh
        # clone has no models/, and `cp ... models/best_model.pt` then fails
        # with a message naming the weights file rather than the folder.
        candidate.parent.mkdir(parents=True, exist_ok=True)

    logger.info("no custom weights at %s; falling back to %s", candidate, STOCK_WEIGHTS)
    return WeightsChoice(path=STOCK_WEIGHTS, is_custom=False)


#: Where ultralytics writes runs. Searched newest-first when no weights are named.
RUNS_ROOT = PROJECT_ROOT / "runs"


def latest_trained_weights() -> Path | None:
    """The most recently written best.pt, wherever it lives.

    Typing a timestamped run path by hand is a constant small source of
    error, and nothing about "score the model I just trained" requires
    knowing it.

    Strictly newest-first, models/best_model.pt included. An earlier version
    preferred models/ unconditionally, reasoning that putting weights there
    is the deliberate act of naming a current model. That is true the day you
    do it and false a week later: a model copied there in one session
    silently shadowed every run afterwards, and an evaluation meant to score
    a fresh run re-scored the old one instead — returning plausible numbers
    identical to the previous week's, which is the worst way for it to fail.
    """
    candidates = []
    if CUSTOM_WEIGHTS.is_file():
        candidates.append(CUSTOM_WEIGHTS)
    if RUNS_ROOT.is_dir():
        candidates += [p for p in RUNS_ROOT.glob("*/*/weights/best.pt") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def shadowed_by(chosen: Path) -> list[Path]:
    """Other trained weights older than the one chosen, for reporting.

    Silence here is what let a stale model be scored as a fresh one.
    """
    others = []
    if CUSTOM_WEIGHTS.is_file() and chosen != CUSTOM_WEIGHTS:
        others.append(CUSTOM_WEIGHTS)
    if RUNS_ROOT.is_dir():
        others += [p for p in RUNS_ROOT.glob("*/*/weights/best.pt") if p.is_file() and p != chosen]
    return sorted(others, key=lambda path: path.stat().st_mtime, reverse=True)
