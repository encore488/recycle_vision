"""Domain models for the sorting pipeline.

The vocabulary here is deliberately small:

    Detection   something the vision model saw
    Bin         somewhere an item can be put
    RoutedItem  a Detection paired with the Bin it belongs in, and why
    SortResult  everything learned about one image

Nothing in this module imports a vision library, so it stays cheap to import
and trivial to test.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum


class Certainty(str, Enum):
    """How much the routing policy trusts a decision.

    Distinct from detector confidence: the model can be certain it sees a
    bottle while the policy remains unsure whether that bottle is glass or
    plastic, and therefore which bin it belongs in.
    """

    HIGH = "high"
    LOW = "low"

    @property
    def needs_review(self) -> bool:
        return self is Certainty.LOW


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in pixel coordinates, origin top-left.

    Corners are normalised on construction so that x1 <= x2 and y1 <= y2.
    Detectors are not required to agree on corner order, and an inverted box
    would otherwise reach the renderer -- where PIL raises -- or produce a
    negative area. Normalising once here means nothing downstream has to
    think about it.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        # Frozen dataclass, so normalisation goes through object.__setattr__.
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "x2", x2)
        object.__setattr__(self, "y1", y1)
        object.__setattr__(self, "y2", y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass(frozen=True)
class Detection:
    """One object located by the detector, in the detector's own vocabulary.

    `label` is whatever the model calls it -- "bottle" under COCO weights, or
    whatever a future conveyor-trained model emits. Translating that into a
    destination is the routing policy's job, not the detector's.
    """

    label: str
    confidence: float
    box: BoundingBox


@dataclass(frozen=True)
class Bin:
    """A destination an item can be routed to.

    Bins are facility-specific and come from a policy file. A materials
    recovery facility separates PET from HDPE from aluminium; a household
    has three containers under the sink. Same code, different config.
    """

    key: str
    name: str
    color: str
    description: str = ""
    #: False for bins whose contents leave the recycling stream (landfill,
    #: residue). Drives the diversion rate.
    diverted: bool = True

    @property
    def swatch(self) -> str:
        """The bin colour as an (r, g, b)-friendly hex string."""
        return self.color


@dataclass(frozen=True)
class RoutedItem:
    """A detection paired with its destination and the reasoning behind it."""

    detection: Detection
    bin: Bin
    certainty: Certainty = Certainty.HIGH
    #: Plain-language name for the item, e.g. "Drinking glass".
    item_name: str = ""
    #: What the material actually is. An attribute, never the destination.
    material: str = "unknown"
    #: What the user should do before binning it, e.g. "Rinse; leave cap on".
    handling: str = ""
    #: Why this bin and not the obvious one. Only set where it is surprising.
    rationale: str = ""

    @property
    def label(self) -> str:
        return self.item_name or self.detection.label.title()

    @property
    def confidence(self) -> float:
        return self.detection.confidence

    @property
    def needs_review(self) -> bool:
        return self.certainty.needs_review


@dataclass
class SortResult:
    """Everything the pipeline learned about a single image."""

    items: list[RoutedItem] = field(default_factory=list)
    #: Detections the policy classified as not-waste (people, furniture, pets).
    #: Kept rather than dropped so the UI can be honest about what it saw.
    ignored: list[Detection] = field(default_factory=list)
    #: Name of the model that produced the detections.
    model_name: str = ""
    #: Name of the policy that produced the routes.
    policy_name: str = ""

    def __len__(self) -> int:
        return len(self.items)

    @property
    def total_items(self) -> int:
        return len(self.items)

    @property
    def counts_by_bin(self) -> Counter[str]:
        """Item counts keyed by bin key."""
        return Counter(item.bin.key for item in self.items)

    @property
    def counts_by_material(self) -> Counter[str]:
        return Counter(item.material for item in self.items)

    @property
    def bins_used(self) -> list[Bin]:
        """Distinct bins in play, in first-seen order."""
        seen: dict[str, Bin] = {}
        for item in self.items:
            seen.setdefault(item.bin.key, item.bin)
        return list(seen.values())

    @property
    def items_for_review(self) -> list[RoutedItem]:
        return [item for item in self.items if item.needs_review]

    @property
    def diverted_count(self) -> int:
        """Items headed somewhere other than landfill/residue."""
        return sum(1 for item in self.items if item.bin.diverted)

    @property
    def diversion_rate(self) -> float:
        """Share of items kept out of landfill, in 0.0-1.0.

        The headline number: what fraction of this stream is actually being
        recovered. Zero items means zero, not a division error.
        """
        if not self.items:
            return 0.0
        return self.diverted_count / self.total_items

    @property
    def contamination_rate(self) -> float:
        """Share of items that are NOT recoverable, in 0.0-1.0.

        The metric materials recovery facilities actually track.
        """
        if not self.items:
            return 0.0
        return 1.0 - self.diversion_rate

    @property
    def average_confidence(self) -> float:
        if not self.items:
            return 0.0
        return sum(item.confidence for item in self.items) / self.total_items

    def items_in(self, bin_key: str) -> list[RoutedItem]:
        return [item for item in self.items if item.bin.key == bin_key]
