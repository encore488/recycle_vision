"""Reading COCO annotations into this project's YOLO-shaped datasets.

ZeroWaste ships COCO JSON with polygons; WaRP and SortWaste ship YOLO text.
Both describe the same kind of thing, so the difference is confined here
rather than spread through the importer.

Two conversions matter and both are easy to get silently wrong:

- **Geometry.** COCO boxes are `[x_min, y_min, width, height]` in absolute
  pixels; YOLO wants centre-x, centre-y, width, height normalised to the
  image. A transposed pair or a missing divide produces labels that look
  plausible in a file and sit nowhere near the objects.
- **Where the images are.** `file_name` is relative to something the JSON
  does not state. ZeroWaste keeps frames in `data/` beside a `sem_seg/`
  directory of masks, so resolving against the JSON's own directory finds
  nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

#: Directories holding annotation rasters rather than photographs. ZeroWaste
#: puts segmentation masks in `sem_seg/` under the SAME filenames as the
#: frames in `data/`, so a basename lookup can resolve to a mask -- a
#: greyscale label map imported as a training image, which trains fine and
#: teaches nothing.
MASK_DIRS = {"sem_seg", "semantic", "seg", "masks", "mask", "gt", "annotations", "labels"}


def looks_like_mask(path: Path) -> bool:
    """Whether a path sits inside an annotation-raster directory."""
    return any(part.lower() in MASK_DIRS for part in path.parts)


class CocoError(ValueError):
    """Raised when a COCO file is missing, malformed, or unusable."""


@dataclass(frozen=True)
class CocoImage:
    """One image and the annotations that landed on it."""

    file_name: str
    width: int
    height: int
    #: (category name, x_min, y_min, box width, box height) in pixels.
    boxes: list[tuple[str, float, float, float, float]] = field(default_factory=list)


def to_yolo_line(
    class_id: int, x: float, y: float, w: float, h: float, width: int, height: int
) -> str | None:
    """One COCO box as a YOLO label line, or None if it cannot be expressed.

    Clamped to the image: COCO boxes occasionally run a pixel or two past the
    edge, and ultralytics rejects a whole label file over a coordinate above
    1.0 rather than the single line that caused it.
    """
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        return None

    centre_x = (x + w / 2) / width
    centre_y = (y + h / 2) / height
    norm_w = w / width
    norm_h = h / height

    centre_x = min(max(centre_x, 0.0), 1.0)
    centre_y = min(max(centre_y, 0.0), 1.0)
    norm_w = min(max(norm_w, 0.0), 1.0)
    norm_h = min(max(norm_h, 0.0), 1.0)
    if norm_w <= 0 or norm_h <= 0:
        return None

    return f"{class_id} {centre_x:.6f} {centre_y:.6f} {norm_w:.6f} {norm_h:.6f}"


def read_coco(path: Path) -> tuple[list[str], list[CocoImage]]:
    """Category names and per-image annotations from a COCO file.

    Images with no annotations are kept. In YOLO an empty label file means
    "nothing here", which is a real training signal and exactly what an
    unannotated frame is.
    """
    path = Path(path)
    if not path.is_file():
        raise CocoError(f"no COCO file at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CocoError(f"{path} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict) or "annotations" not in payload:
        raise CocoError(f"{path} has no 'annotations' — is it a COCO file?")

    categories = {c["id"]: c.get("name", str(c["id"])) for c in payload.get("categories", [])}
    records: dict[int, CocoImage] = {}
    for entry in payload.get("images", []):
        records[entry["id"]] = CocoImage(
            file_name=entry.get("file_name", ""),
            width=int(entry.get("width", 0)),
            height=int(entry.get("height", 0)),
            boxes=[],
        )

    for annotation in payload.get("annotations", []):
        record = records.get(annotation.get("image_id"))
        if record is None:
            continue
        box = annotation.get("bbox")
        if not box or len(box) != 4:
            continue
        name = categories.get(annotation.get("category_id"))
        if name is None:
            continue
        record.boxes.append((name, float(box[0]), float(box[1]), float(box[2]), float(box[3])))

    return sorted(categories.values()), [records[k] for k in sorted(records)]


def index_images(root: Path) -> dict[str, Path]:
    """Every image under `root`, keyed by basename and by relative path.

    COCO's `file_name` is relative to a directory the file never names, and
    different releases mean different things by it. Indexing both forms lets
    a lookup succeed either way, and basenames are unique in practice within
    one split.
    """
    found: dict[str, Path] = {}
    # Sorted, so the same tree always indexes the same way. Relying on
    # filesystem order here decides whether frames or masks get imported.
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            continue
        # A relative path is unambiguous; first one wins.
        found.setdefault(str(path.relative_to(root)), path)

        # A basename is not. Where two files share one, a photograph beats an
        # annotation raster -- always, regardless of which was walked first.
        existing = found.get(path.name)
        if existing is None or (looks_like_mask(existing) and not looks_like_mask(path)):
            found[path.name] = path
    return found


def locate(file_name: str, index: dict[str, Path]) -> Path | None:
    """Resolve one COCO `file_name` against an image index."""
    if not file_name:
        return None
    for key in (file_name, Path(file_name).name, file_name.lstrip("./")):
        hit = index.get(key)
        if hit is not None:
            return hit
    return None
