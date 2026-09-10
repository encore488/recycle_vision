"""Loading and applying routing policies.

A policy answers one question: given a thing the detector saw, which bin does
it go in and what should the user know about it? It is data, not code -- see
`policies/*.yaml` -- so that adapting to a new municipality or a new sorting
facility never requires a deploy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .models import Bin, Certainty, Detection, RoutedItem

#: Sentinel `default:` value meaning "this class is not waste; don't count it".
IGNORE = "ignore"


class PolicyError(ValueError):
    """Raised when a policy file is malformed.

    Policies are hand-edited YAML, so failing loudly with a specific message
    matters more here than in code paths users never touch.
    """


@dataclass(frozen=True)
class Rule:
    """What a policy says about one detector class."""

    bin_key: str
    item_name: str = ""
    material: str = "unknown"
    handling: str = ""
    certainty: Certainty = Certainty.HIGH
    rationale: str = ""


class RoutingPolicy:
    """Maps detector class names onto bins.

    Lookup is case- and whitespace-insensitive so that a policy written for
    COCO's "wine glass" still matches a model that emits "Wine_Glass".
    """

    def __init__(
        self,
        name: str,
        bins: list[Bin],
        rules: dict[str, Rule],
        description: str = "",
        default: str = IGNORE,
    ) -> None:
        self.name = name
        self.description = description
        self.default = default
        self._bins = {b.key: b for b in bins}
        self._rules = {self._normalize(k): v for k, v in rules.items()}

        unknown = {r.bin_key for r in rules.values()} - set(self._bins)
        if self.default != IGNORE:
            unknown |= {self.default} - set(self._bins)
        if unknown:
            raise PolicyError(
                f"policy {name!r} routes to undefined bins: {sorted(unknown)}. "
                f"Defined bins are: {sorted(self._bins)}"
            )

    @staticmethod
    def _normalize(label: str) -> str:
        return label.strip().lower().replace("_", " ").replace("-", " ")

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, path: str | Path) -> RoutingPolicy:
        """Read a policy from a YAML file."""
        path = Path(path)
        if not path.is_file():
            raise PolicyError(f"no policy file at {path}")

        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise PolicyError(f"{path} is not valid YAML: {exc}") from exc

        if not isinstance(raw, dict):
            raise PolicyError(f"{path} should contain a mapping at the top level")

        return cls.from_dict(raw, source=str(path))

    @classmethod
    def from_dict(cls, raw: dict, source: str = "<dict>") -> RoutingPolicy:
        """Build a policy from an already-parsed mapping."""
        bins = [cls._parse_bin(b, source) for b in cls._require(raw, "bins", source)]
        rules = {
            label: cls._parse_rule(label, body, source)
            for label, body in cls._require(raw, "rules", source).items()
        }
        return cls(
            name=raw.get("name", "Unnamed policy"),
            description=raw.get("description", ""),
            default=raw.get("default", IGNORE),
            bins=bins,
            rules=rules,
        )

    @staticmethod
    def _require(raw: dict, key: str, source: str):
        value = raw.get(key)
        if not value:
            raise PolicyError(f"{source} is missing a non-empty {key!r} section")
        return value

    @staticmethod
    def _parse_bin(raw: dict, source: str) -> Bin:
        try:
            return Bin(
                key=raw["key"],
                name=raw.get("name", raw["key"].title()),
                color=raw.get("color", "#6B7280"),
                description=raw.get("description", ""),
                diverted=bool(raw.get("diverted", True)),
            )
        except (KeyError, TypeError) as exc:
            raise PolicyError(f"{source} has a malformed bin entry: {raw!r}") from exc

    @staticmethod
    def _parse_rule(label: str, raw: dict, source: str) -> Rule:
        if not isinstance(raw, dict):
            raise PolicyError(f"{source}: rule for {label!r} should be a mapping, got {raw!r}")
        if "bin" not in raw:
            raise PolicyError(f"{source}: rule for {label!r} is missing a 'bin'")

        certainty = str(raw.get("certainty", "high")).lower()
        try:
            certainty = Certainty(certainty)
        except ValueError as exc:
            valid = [c.value for c in Certainty]
            raise PolicyError(
                f"{source}: rule for {label!r} has certainty {certainty!r}; expected one of {valid}"
            ) from exc

        return Rule(
            bin_key=raw["bin"],
            item_name=raw.get("item", ""),
            material=raw.get("material", "unknown"),
            handling=raw.get("handling", ""),
            certainty=certainty,
            rationale=raw.get("rationale", ""),
        )

    # ------------------------------------------------------------------ routing

    @property
    def bins(self) -> list[Bin]:
        """Every bin this policy defines, in declaration order."""
        return list(self._bins.values())

    def bin(self, key: str) -> Bin:
        try:
            return self._bins[key]
        except KeyError as exc:
            raise PolicyError(f"policy {self.name!r} has no bin {key!r}") from exc

    def covers(self, label: str) -> bool:
        """Whether this policy has an explicit rule for a detector class."""
        return self._normalize(label) in self._rules

    def route(self, detection: Detection) -> RoutedItem | None:
        """Decide where a detection goes.

        Returns None when the policy considers the class not-waste, which is
        the common case under COCO weights: people, cars and furniture are
        detected constantly and must not be counted as items in the stream.
        """
        rule = self._rules.get(self._normalize(detection.label))

        if rule is None:
            if self.default == IGNORE:
                return None
            rule = Rule(
                bin_key=self.default,
                item_name=detection.label.title(),
                certainty=Certainty.LOW,
                rationale=(
                    "No rule for this item, so the policy default applies. Worth a human check."
                ),
            )

        return RoutedItem(
            detection=detection,
            bin=self.bin(rule.bin_key),
            certainty=rule.certainty,
            item_name=rule.item_name or detection.label.title(),
            material=rule.material,
            handling=rule.handling,
            rationale=rule.rationale,
        )
