"""Scoring a trained model the way this project actually uses it.

Detection mAP answers "did it name the object right". This project needs
"did it put the object in the right bin", and those come apart: confusing an
aluminium can for a steel one is a class error worth nothing, because both go
in the same bin, while confusing a drinking glass for a jar is a class error
that contaminates a batch.

So a model is scored twice -- once on classes, once on the destinations those
classes map to through a policy. A drop in class accuracy that does not move
routing accuracy is not worth chasing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .models import BoundingBox
from .policy import RoutingPolicy

#: Overlap at which a prediction is considered to be *about* a ground-truth
#: object. Deliberately loose: this measures whether the right label reached
#: the right object, not how tightly the box is drawn.
DEFAULT_IOU = 0.45


@dataclass
class RoutingScore:
    """How often predictions reach the right bin, and where they do not."""

    matched: int = 0
    class_correct: int = 0
    bin_correct: int = 0
    #: (predicted class, true class) -> count, for pairs that changed the bin.
    harmful_confusions: Counter = field(default_factory=Counter)
    #: (predicted class, true class) -> count, for pairs that did not.
    harmless_confusions: Counter = field(default_factory=Counter)

    @property
    def class_accuracy(self) -> float:
        return self.class_correct / self.matched if self.matched else 0.0

    @property
    def routing_accuracy(self) -> float:
        return self.bin_correct / self.matched if self.matched else 0.0

    @property
    def forgiven(self) -> int:
        """Class errors that did not change the destination."""
        return self.bin_correct - self.class_correct

    def report(self) -> str:
        lines = [
            f"matched instances   {self.matched}",
            f"class accuracy      {self.class_accuracy:.1%}",
            f"routing accuracy    {self.routing_accuracy:.1%}",
        ]
        if self.forgiven:
            lines.append(f"  ({self.forgiven} class error(s) landed in the right bin anyway)")
        if self.harmful_confusions:
            lines.append("")
            lines.append("confusions that changed the bin:")
            for (predicted, actual), count in self.harmful_confusions.most_common(10):
                lines.append(f"  {predicted!r} predicted for {actual!r}  x{count}")
        if self.harmless_confusions:
            lines.append("")
            lines.append("confusions that did not matter:")
            for (predicted, actual), count in self.harmless_confusions.most_common(5):
                lines.append(f"  {predicted!r} predicted for {actual!r}  x{count}")
        return "\n".join(lines)


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """Intersection over union of two boxes, 0.0 when they do not overlap."""
    left = max(a.x1, b.x1)
    top = max(a.y1, b.y1)
    right = min(a.x2, b.x2)
    bottom = min(a.y2, b.y2)
    if right <= left or bottom <= top:
        return 0.0

    overlap = (right - left) * (bottom - top)
    union = a.area + b.area - overlap
    return overlap / union if union > 0 else 0.0


@dataclass
class MatchResult:
    """Predictions paired with ground truth, plus what went unpaired."""

    pairs: list[tuple[str, str]] = field(default_factory=list)
    #: Predictions that matched no ground-truth object.
    false_positives: list[str] = field(default_factory=list)
    #: Ground-truth objects no prediction reached.
    missed: list[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        found = len(self.pairs) + len(self.false_positives)
        return len(self.pairs) / found if found else 0.0

    @property
    def recall(self) -> float:
        findable = len(self.pairs) + len(self.missed)
        return len(self.pairs) / findable if findable else 0.0


def match_detections(
    predictions: list[tuple[str, BoundingBox, float]],
    truth: list[tuple[str, BoundingBox]],
    threshold: float = DEFAULT_IOU,
) -> MatchResult:
    """Greedily pair predictions to ground truth by overlap.

    Highest-confidence predictions claim their object first, and each
    ground-truth object is claimed once -- so two boxes on the same can count
    as one match and one false positive, rather than both scoring.
    """
    result = MatchResult()
    claimed: set[int] = set()

    for label, box, _confidence in sorted(predictions, key=lambda p: -p[2]):
        best_index, best_iou = None, threshold
        for index, (_truth_label, truth_box) in enumerate(truth):
            if index in claimed:
                continue
            overlap = iou(box, truth_box)
            if overlap >= best_iou:
                best_index, best_iou = index, overlap

        if best_index is None:
            result.false_positives.append(label)
        else:
            claimed.add(best_index)
            result.pairs.append((label, truth[best_index][0]))

    result.missed = [label for i, (label, _) in enumerate(truth) if i not in claimed]
    return result


#: Buckets for the best-overlap histogram. The boundaries matter: a
#: prediction landing at 0.0 was nowhere near an object, while one at 0.3 found
#: the object and drew it loosely. Those need completely different fixes, and a
#: single recall number cannot tell them apart.
IOU_BUCKETS = [
    (0.0, 0.01, "no overlap at all"),
    (0.01, 0.10, "grazing"),
    (0.10, 0.25, "found it, badly placed"),
    (0.25, 0.45, "near miss"),
    (0.45, 1.01, "matched"),
]


def best_overlaps(
    predictions: list[tuple[str, BoundingBox, float]],
    truth: list[tuple[str, BoundingBox]],
) -> list[float]:
    """For each prediction, its best overlap with any ground-truth box.

    Unlike matching, nothing is claimed here -- this asks only "did the model
    put a box anywhere near a real object", which is the question that
    separates a detector that cannot see the objects from a harness that is
    comparing the wrong coordinates.
    """
    if not truth:
        return [0.0] * len(predictions)
    return [max(iou(box, t_box) for _t_label, t_box in truth) for _label, box, _c in predictions]


def overlap_histogram(overlaps: list[float]) -> list[tuple[str, int, float]]:
    """Bucket best-overlap values into (label, count, share)."""
    total = len(overlaps)
    rows = []
    for low, high, label in IOU_BUCKETS:
        count = sum(1 for value in overlaps if low <= value < high)
        rows.append((label, count, count / total if total else 0.0))
    return rows


def _bin_of(policy: RoutingPolicy, label: str) -> str | None:
    """The bin a class routes to, or None when the policy ignores it."""
    from .models import BoundingBox, Detection

    routed = policy.route(Detection(label=label, confidence=1.0, box=BoundingBox(0, 0, 1, 1)))
    return routed.bin.key if routed else None


def score_routing(pairs: list[tuple[str, str]], policy: RoutingPolicy) -> RoutingScore:
    """Score (predicted class, true class) pairs on class *and* on destination.

    Pairs come from whatever matcher produced them -- ultralytics' own
    IoU matching, or a hand-graded file. This function only cares about the
    labels, which keeps it testable without a model.
    """
    score = RoutingScore(matched=len(pairs))

    for predicted, actual in pairs:
        if predicted == actual:
            score.class_correct += 1
            score.bin_correct += 1
            continue

        predicted_bin = _bin_of(policy, predicted)
        actual_bin = _bin_of(policy, actual)
        if predicted_bin is not None and predicted_bin == actual_bin:
            score.bin_correct += 1
            score.harmless_confusions[(predicted, actual)] += 1
        else:
            score.harmful_confusions[(predicted, actual)] += 1

    return score
