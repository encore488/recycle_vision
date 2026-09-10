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

from .policy import RoutingPolicy


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
