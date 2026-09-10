"""The vision layer.

`Detector` is the seam between "what the model saw" and everything else. The
pipeline, the policy and the UI all speak `Detection`, so the day a custom
conveyor-trained model replaces stock YOLO, this is the only file that
notices -- and the tests, which run against a stub, need no weights at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .models import BoundingBox, Detection
from .vocabulary import DEFAULT_VOCAB, Vocabulary
from .weights import WeightsChoice, resolve_weights


def _to_detections(result, names) -> list[Detection]:
    """Convert an ultralytics result into the project's own vocabulary."""
    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        detections.append(
            Detection(
                label=names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                box=BoundingBox(x1, y1, x2, y2),
            )
        )
    return detections


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
        return _to_detections(result, self._model.names)


class OpenVocabularyDetector:
    """YOLOE told what to look for in words.

    The point of this class is that its class list is configuration. COCO has
    no label for a drink can, so a COCO detector reports one as "cup" or
    "bowl" and no downstream rule can undo that. Here the vocabulary simply
    asks for "aluminum drink can", and the detector answers in those terms.

    Text embeddings are loaded from the cache built by
    `scripts/build_vocab_embeddings.py`, never computed here: the text encoder
    is an order of magnitude larger than the detector, and needing it at
    request time would make the app undeployable on a small host.
    """

    def __init__(self, vocabulary: Vocabulary | None = None) -> None:
        from ultralytics import YOLOE  # deferred: heavy import

        self.vocabulary = vocabulary or Vocabulary.load(DEFAULT_VOCAB)
        embeddings = self.vocabulary.load_embeddings()

        self._model = YOLOE(self.vocabulary.model)
        self._model.set_classes(self.vocabulary.classes, embeddings)

    @property
    def name(self) -> str:
        return f"{Path(self.vocabulary.model).stem} · {self.vocabulary.name}"

    @property
    def class_names(self) -> list[str]:
        return list(self.vocabulary.classes)

    @property
    def weights(self) -> WeightsChoice:
        """Presented as a custom model: its vocabulary is purpose-built.

        Still not a *trained* waste model -- the detector is zero-shot and
        has never seen a labelled conveyor belt -- but the everyday-objects
        caveat that applies to COCO does not apply here.
        """
        return WeightsChoice(
            path=self.vocabulary.model,
            is_custom=True,
            display_name_override=self.name,
        )

    def detect(self, image, confidence: float = 0.25) -> list[Detection]:
        result = self._model.predict(image, conf=confidence, verbose=False)[0]
        return _to_detections(result, self._model.names)


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
