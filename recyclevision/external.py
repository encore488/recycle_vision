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


#: A YOLO label file holds one of these. Ultralytics will not train a
#: segmentation model on `BOXES`, and says so only after scanning the whole
#: dataset -- several minutes into a run that was never going to start.
BOXES = "boxes"
POLYGONS = "polygons"
MIXED = "mixed"
EMPTY = "empty"


def read_split_images_dir(path: str | Path, split: str = "train") -> Path | None:
    """Where one split's images live, per a dataset descriptor.

    Returns None when the descriptor does not name that split, rather than
    guessing a conventional location that may not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise MappingError(f"no dataset descriptor at {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise MappingError(f"{path} is not a dataset descriptor")

    entry = raw.get(split)
    if isinstance(entry, list):
        entry = entry[0] if entry else None
    if not isinstance(entry, str):
        return None

    root = Path(raw.get("path", path.parent))
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    candidate = Path(entry)
    return candidate if candidate.is_absolute() else (root / candidate).resolve()


def label_dir_for(images_dir: Path) -> Path:
    """The labels directory ultralytics pairs with an images directory.

    Mirrors ultralytics' own rule -- swap the last `images` path segment for
    `labels` -- because inventing a different one here would report a healthy
    dataset as empty, or an empty one as healthy.
    """
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_dir


def inspect_label_geometry(labels_dir: Path, sample: int = 300) -> str:
    """Whether a label directory holds boxes, polygons, both, or nothing.

    A detection line is `cls cx cy w h`: five fields. A segmentation line is
    `cls x1 y1 x2 y2 ...`: an odd count of at least seven, being a class plus
    three or more points.

    Sampled rather than read whole, and strided rather than taken from the
    front -- a sorted listing's first few hundred files are often one scene,
    which would miss a dataset that is only partly polygons.
    """
    files = sorted(labels_dir.rglob("*.txt")) if labels_dir.is_dir() else []
    if not files:
        return EMPTY

    step = max(1, len(files) // sample)
    has_boxes = has_polygons = False
    for path in files[::step][:sample]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            count = len(line.split())
            if count == 5:
                has_boxes = True
            elif count >= 7 and count % 2 == 1:
                has_polygons = True
        if has_boxes and has_polygons:
            return MIXED

    if has_polygons:
        return POLYGONS
    if has_boxes:
        return BOXES
    return EMPTY


def trees_overlap(a: Path, b: Path) -> bool:
    """Whether two directory paths are the same, or one contains the other.

    Importing a dataset into its own source rewrites the labels being read and
    then fails partway through copying files onto themselves, leaving the
    source half-converted. Cheap to rule out, expensive to discover.
    """
    a, b = a.resolve(), b.resolve()
    return a == b or a in b.parents or b in a.parents


def looks_already_imported(source_classes: list[str], vocabulary_classes: list[str]) -> bool:
    """Whether a "source" dataset is really this project's own import output.

    Pointing at `datasets/warp/data.yaml` instead of the original download is
    an easy mistake -- both are called data.yaml and one sits in the working
    tree. Every class already being one of ours is the giveaway, and saying so
    beats listing sixteen class names the user never wrote.
    """
    return bool(source_classes) and set(source_classes) <= set(vocabulary_classes)
