"""Accumulating results across several images.

One photo answers "where does this go". A run of photos answers "how is this
stream doing", which is the question a facility actually has. The aggregate is
kept as its own type rather than by mutating a `SortResult`, so a single image
and a batch of two hundred are described by different things and cannot be
confused for one another.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .models import Bin, RoutedItem, SortResult


@dataclass(frozen=True)
class Entry:
    """One image and what was found in it."""

    source: str
    result: SortResult


@dataclass
class Session:
    """Every image processed so far, and the totals across them."""

    entries: list[Entry] = field(default_factory=list)

    def add(self, source: str, result: SortResult) -> None:
        self.entries.append(Entry(source=source, result=result))

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def image_count(self) -> int:
        return len(self.entries)

    @property
    def items(self) -> list[RoutedItem]:
        return [item for entry in self.entries for item in entry.result.items]

    @property
    def total_items(self) -> int:
        return len(self.items)

    @property
    def counts_by_bin(self) -> Counter[str]:
        return Counter(item.bin.key for item in self.items)

    @property
    def counts_by_item(self) -> Counter[str]:
        """What was seen, by plain-language name — the stream's composition."""
        return Counter(item.label for item in self.items)

    @property
    def bins_used(self) -> list[Bin]:
        seen: dict[str, Bin] = {}
        for item in self.items:
            seen.setdefault(item.bin.key, item.bin)
        return list(seen.values())

    @property
    def diverted_count(self) -> int:
        return sum(1 for item in self.items if item.bin.diverted)

    @property
    def diversion_rate(self) -> float:
        if not self.items:
            return 0.0
        return self.diverted_count / self.total_items

    @property
    def contamination_rate(self) -> float:
        if not self.items:
            return 0.0
        return 1.0 - self.diversion_rate

    @property
    def items_for_review(self) -> list[RoutedItem]:
        return [item for item in self.items if item.needs_review]

    @property
    def average_confidence(self) -> float:
        if not self.items:
            return 0.0
        return sum(item.confidence for item in self.items) / self.total_items

    def items_in(self, bin_key: str) -> list[RoutedItem]:
        return [item for item in self.items if item.bin.key == bin_key]

    def as_result(self) -> SortResult:
        """The whole session flattened into one `SortResult`.

        Lets anything that already consumes a single result -- the impact
        model, the renderer's legend, the exporters -- work on a batch with no
        special case. Provenance collapses to the newest entry, which is why
        this is a view rather than the session's own representation.
        """
        newest = self.entries[-1].result if self.entries else SortResult()
        return SortResult(
            items=list(self.items),
            ignored=[d for entry in self.entries for d in entry.result.ignored],
            model_name=newest.model_name,
            policy_name=newest.policy_name,
        )
