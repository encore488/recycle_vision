"""Estimating the mass and avoided emissions of a sorted stream.

Deliberately built so the arithmetic and the constants are separate things.
The arithmetic here is sound; the constants shipped in `impact/factors.yaml`
are order-of-magnitude placeholders, and the `verified` flag carries that fact
all the way to the screen rather than letting it get lost between the file and
the user.

Only diverted items count toward avoided emissions. Recycling something is
what avoids the emission; sending it to landfill avoids nothing, so a stream
that is 100% contamination correctly scores zero rather than scoring for the
material it contains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .models import SortResult

DEFAULT_FACTORS = Path(__file__).resolve().parent.parent / "impact" / "factors.yaml"

GRAMS_PER_KG = 1000.0


class ImpactError(ValueError):
    """Raised when a factors file is unusable."""


@dataclass(frozen=True)
class Estimate:
    """What a sorted stream is worth, and how much to trust the figure."""

    total_mass_kg: float = 0.0
    diverted_mass_kg: float = 0.0
    co2e_avoided_kg: float = 0.0
    #: Items whose class has no mass factor, so they contribute nothing.
    unpriced_items: list[str] = field(default_factory=list)
    #: False when the factors are placeholders. Never suppress this.
    verified: bool = False
    factors_name: str = ""

    @property
    def coverage_note(self) -> str:
        if not self.unpriced_items:
            return ""
        names = ", ".join(sorted(set(self.unpriced_items)))
        return f"No mass factor for: {names}. These are excluded from the totals."

    @property
    def caveat(self) -> str:
        if self.verified:
            return ""
        return (
            "Estimated from unverified placeholder factors — indicative only, "
            "not suitable for reporting. See impact/factors.yaml."
        )


class ImpactModel:
    """Mass and carbon factors, applied to a `SortResult`."""

    def __init__(self, name: str, materials: dict, items: dict, verified: bool = False) -> None:
        self.name = name
        self.verified = verified
        self._materials = materials
        self._items = {self._normalize(k): v for k, v in items.items()}

    @staticmethod
    def _normalize(label: str) -> str:
        return label.strip().lower().replace("_", " ").replace("-", " ")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_FACTORS) -> ImpactModel:
        path = Path(path)
        if not path.is_file():
            raise ImpactError(f"no factors file at {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ImpactError(f"{path} is not valid YAML: {exc}") from exc
        if not isinstance(raw, dict) or "items" not in raw:
            raise ImpactError(f"{path} needs at least an 'items' section")

        return cls(
            name=raw.get("name", path.stem),
            materials=raw.get("materials", {}),
            items=raw["items"],
            # Absent means unverified. Trust has to be asserted, not assumed.
            verified=bool(raw.get("verified", False)),
        )

    def estimate(self, result: SortResult) -> Estimate:
        total_g = diverted_g = co2e_kg = 0.0
        unpriced: list[str] = []

        for item in result.items:
            entry = self._items.get(self._normalize(item.detection.label))
            if entry is None:
                unpriced.append(item.detection.label)
                continue

            mass_g = float(entry.get("mass_g", 0.0))
            total_g += mass_g
            if not item.bin.diverted:
                continue

            diverted_g += mass_g
            material = self._materials.get(entry.get("material", ""), {})
            co2e_kg += (mass_g / GRAMS_PER_KG) * float(material.get("co2e_kg_per_kg", 0.0))

        return Estimate(
            total_mass_kg=total_g / GRAMS_PER_KG,
            diverted_mass_kg=diverted_g / GRAMS_PER_KG,
            co2e_avoided_kg=co2e_kg,
            unpriced_items=unpriced,
            verified=self.verified,
            factors_name=self.name,
        )
