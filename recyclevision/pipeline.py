"""Wiring the detector and the policy together.

This is the whole application in one small class: see what is there, decide
where each thing goes, hand back a `SortResult`. Everything downstream --
Streamlit today, an HTTP API or a robot controller later -- consumes that
same result.
"""

from __future__ import annotations

from pathlib import Path

from .detector import Detector
from .models import SortResult
from .policy import RoutingPolicy

DEFAULT_POLICY = Path(__file__).resolve().parent.parent / "policies" / "household.yaml"


class SortingPipeline:
    """Detect, then route."""

    def __init__(self, detector: Detector, policy: RoutingPolicy) -> None:
        self.detector = detector
        self.policy = policy

    @classmethod
    def build(
        cls,
        policy_path: str | Path = DEFAULT_POLICY,
        weights_path: str | Path | None = None,
    ) -> SortingPipeline:
        """Construct the real pipeline, loading weights and policy from disk."""
        from .detector import YoloDetector  # deferred: pulls in torch
        from .weights import resolve_weights

        return cls(
            detector=YoloDetector(resolve_weights(weights_path)),
            policy=RoutingPolicy.load(policy_path),
        )

    def sort(self, image, confidence: float = 0.25) -> SortResult:
        """Run the full pipeline over one image."""
        result = SortResult(
            model_name=self.detector.name,
            policy_name=self.policy.name,
        )

        for detection in self.detector.detect(image, confidence=confidence):
            routed = self.policy.route(detection)
            if routed is None:
                # Not waste -- a person, a car, a sofa. Recorded so the UI can
                # say what it saw, but kept out of every metric.
                result.ignored.append(detection)
            else:
                result.items.append(routed)

        # Highest-confidence first: the detail list should lead with what the
        # model is surest about.
        result.items.sort(key=lambda item: item.confidence, reverse=True)
        return result

    def unroutable_classes(self) -> list[str]:
        """Detector classes the policy has no rule for.

        A coverage check, mainly useful when a new model or a new policy is
        introduced and the two may have drifted apart.
        """
        names = getattr(self.detector, "class_names", [])
        return sorted(n for n in names if not self.policy.covers(n))
