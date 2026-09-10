"""Bringing an outside dataset into this project's vocabulary.

Public waste datasets label **materials** -- PET, HDPE, metal, cardboard.
This project routes **bins** from **items**, which is a finer distinction: a
jar and a drinking glass are both glass and go to different places. So an
outside dataset cannot simply be trained on; its class names have to be
translated, and the translation loses information in a way worth stating
rather than discovering later in a confusion matrix.

That translation lives in a mapping file rather than in code, for the same
reason routing rules do: it is knowledge about one facility's labelling
scheme, and it differs for every dataset.

An unmapped class is an error, never a silent drop. A class quietly discarded
becomes a region of the image the model is taught contains nothing, which is
worse than not training on it at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


class MappingError(ValueError):
    """Raised when a dataset mapping is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class ClassMapping:
    """How one outside dataset's classes translate into ours."""

    name: str
    source: str = ""
    description: str = ""
    #: Outside class name -> our vocabulary class, or None to drop it on
    #: purpose (background, "other", classes we cannot route).
    mapping: dict[str, str | None] = field(default_factory=dict)
    #: Distinctions this dataset cannot make, recorded so the loss is on the
    #: record. A model trained only on it will never learn these.
    cannot_express: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> ClassMapping:
        path = Path(path)
        if not path.is_file():
            raise MappingError(f"no mapping file at {path}")
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise MappingError(f"{path} is not valid YAML: {exc}") from exc
        if not isinstance(raw, dict) or "mapping" not in raw:
            raise MappingError(f"{path} needs a 'mapping' section")

        return cls(
            name=raw.get("name", path.stem),
            source=raw.get("source", ""),
            description=raw.get("description", ""),
            mapping=dict(raw["mapping"]),
            cannot_express=list(raw.get("cannot_express", [])),
        )

    def translate(self, source_class: str) -> str | None:
        """Our class name for one of theirs, or None if deliberately dropped."""
        if source_class not in self.mapping:
            raise MappingError(
                f"{self.name} has no rule for dataset class {source_class!r}. "
                "Add one (use `null` to drop it deliberately) — discarding it "
                "silently would teach the model those objects are background."
            )
        return self.mapping[source_class]

    def unmapped(self, source_classes: list[str]) -> list[str]:
        """Dataset classes with no rule. Empty means the mapping is complete."""
        return [c for c in source_classes if c not in self.mapping]

    def unknown_targets(self, vocabulary_classes: list[str]) -> list[str]:
        """Mapping targets that our vocabulary does not contain."""
        known = set(vocabulary_classes)
        return sorted({t for t in self.mapping.values() if t is not None and t not in known})

    def dropped(self) -> list[str]:
        return sorted(k for k, v in self.mapping.items() if v is None)


def read_yolo_data_yaml(path: str | Path) -> tuple[Path, list[str]]:
    """Read an external dataset descriptor, returning its root and class names.

    Handles both shapes ultralytics accepts for `names`: a plain list, and a
    mapping of index to name.
    """
    path = Path(path)
    if not path.is_file():
        raise MappingError(f"no dataset descriptor at {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "names" not in raw:
        raise MappingError(f"{path} has no 'names' section")

    names = raw["names"]
    if isinstance(names, dict):
        try:
            ordered = [names[key] for key in sorted(names, key=int)]
        except (ValueError, TypeError) as exc:
            raise MappingError(f"{path}: 'names' keys should be integers") from exc
    elif isinstance(names, list):
        ordered = list(names)
    else:
        raise MappingError(f"{path}: 'names' should be a list or an index mapping")

    root = Path(raw.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    return root, ordered


def build_index_map(
    source_classes: list[str], mapping: ClassMapping, vocabulary_classes: list[str]
) -> dict[int, int | None]:
    """Source class index -> our class index, or None for a deliberate drop."""
    target_index = {name: i for i, name in enumerate(vocabulary_classes)}
    return {
        source_id: (None if (t := mapping.translate(name)) is None else target_index[t])
        for source_id, name in enumerate(source_classes)
    }


def remap_label_line(line: str, index_map: dict[int, int | None]) -> str | None:
    """Rewrite one YOLO label line into our class indices.

    Returns None for a line whose class is dropped or unreadable. Geometry is
    untouched: it is already normalised, so a remap never has to reason about
    coordinates.
    """
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        source_id = int(parts[0])
    except ValueError:
        return None

    target = index_map.get(source_id)
    if target is None:
        return None
    return " ".join([str(target)] + parts[1:])
