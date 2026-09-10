"""The vision layer.

`Detector` is the seam between "what the model saw" and everything else. The
pipeline, the policy and the UI all speak `Detection`, so the day a custom
conveyor-trained model replaces stock YOLO, this is the only file that
notices -- and the tests, which run against a stub, need no weights at all.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import BoundingBox, Detection
from .weights import WeightsChoice, resolve_weights


@runtime_checkable
class Detector(Protocol):
    """Anything that can find objects in an image."""

    @property
    def name(self) -> str:
        """Human-readable model name, for display."""
        ...

    def detect(self, image, confidence: float = 0.25) -> list[Detection]:
        """Locate objects in a PIL image."""
        ...


class YoloDetector:
    """Ultralytics YOLO behind the `Detector` interface.

    Importing ultralytics drags in torch, which is slow and heavy, so the
    import is deferred to construction time. That keeps `recyclevision`
    importable -- and testable -- in environments that have neither.
    """

    def __init__(self, weights: WeightsChoice | None = None) -> None:
        from ultralytics import YOLO  # deferred: heavy import

        self.weights = weights or resolve_weights()
        self._model = YOLO(self.weights.path)

    @property
    def name(self) -> str:
        return self.weights.display_name

    @property
    def class_names(self) -> list[str]:
        """Every class this model can emit.

        Read from the model rather than hardcoded, so a policy can be checked
        against reality instead of against an assumption.
        """
        return list(self._model.names.values())

    def detect(self, image, confidence: float = 0.25) -> list[Detection]:
        result = self._model.predict(image, conf=confidence, verbose=False)[0]

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    label=self._model.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    box=BoundingBox(x1, y1, x2, y2),
                )
            )
        return detections


class StubDetector:
    """A `Detector` that returns whatever it was given.

    Lets the whole pipeline -- routing, metrics, rendering -- be tested
    deterministically without weights, torch, or a network.
    """

    def __init__(self, detections: list[Detection] | None = None, name: str = "Stub") -> None:
        self._detections = detections or []
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def detect(self, image, confidence: float = 0.25) -> list[Detection]:
        return [d for d in self._detections if d.confidence >= confidence]
