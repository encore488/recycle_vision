"""Building a training dataset from conveyor footage.

The expensive resource in this project is not compute, it is a person's
attention. Everything here is shaped to spend as little of it as possible:

- Video frames are near-duplicates of each other. Labelling two frames 1/30th
  of a second apart costs twice as much and teaches the model nothing, while
  quietly inflating validation scores because the val set contains frames the
  train set has already seen. `sample_frames` drops them.
- Nobody should draw a polygon. The shipped detector already emits masks, so
  pre-labels carry segmentation for free, and a human corrects boxes.
- The train/val split is by *frame index block*, not at random, because random
  splitting of video frames leaks: adjacent frames land on both sides and the
  model is scored on data it trained on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

#: Frames closer than this (mean absolute pixel difference on a downscaled
#: greyscale thumbnail, 0-255) are treated as the same frame.
DEFAULT_MIN_DIFF = 6.0

#: Thumbnail size used for the similarity comparison. Small on purpose: this
#: is asking "is anything different here", not comparing detail.
THUMBNAIL = (64, 64)


@dataclass
class DatasetStats:
    """What came out of a pre-labelling run, so it can be judged before use."""

    images: int = 0
    instances: int = 0
    per_class: dict[str, int] = field(default_factory=dict)
    empty_images: list[str] = field(default_factory=list)

    @property
    def instances_per_image(self) -> float:
        return self.instances / self.images if self.images else 0.0

    def report(self) -> str:
        lines = [
            f"{self.images} image(s), {self.instances} instance(s) "
            f"({self.instances_per_image:.1f} per image)"
        ]
        if self.per_class:
            lines.append("")
            width = max(len(name) for name in self.per_class)
            for name, count in sorted(self.per_class.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {name:<{width}}  {count:5}")

        # A class with very few instances will not train, and saying so here is
        # cheaper than discovering it after an hour of labelling.
        thin = [n for n, c in self.per_class.items() if c < 50]
        if thin:
            lines.append("")
            lines.append(
                f"  ⚠️ under 50 instances, likely too few to learn: {', '.join(sorted(thin))}"
            )
        if self.empty_images:
            lines.append(f"  {len(self.empty_images)} image(s) had no detections at all")
        return "\n".join(lines)


def thumbnail_of(image) -> list[float]:
    """A tiny greyscale signature of a frame, as a flat list of floats."""
    small = image.convert("L").resize(THUMBNAIL)
    # `tobytes` rather than `getdata`: the latter is deprecated in Pillow 14,
    # and for a single-band image the raw bytes are exactly the pixel values.
    return list(small.tobytes())


def frame_difference(a: list[float], b: list[float]) -> float:
    """Mean absolute difference between two thumbnails, 0-255."""
    if not a or not b or len(a) != len(b):
        return 255.0
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) / len(a)


def is_novel(signature: list[float], kept: list[float] | None, min_diff: float) -> bool:
    """Whether a frame differs enough from the last kept one to be worth labelling."""
    if kept is None:
        return True
    return frame_difference(signature, kept) >= min_diff


def block_split(count: int, val_fraction: float = 0.2) -> tuple[list[int], list[int]]:
    """Split frame indices into train and val by contiguous block.

    Video frames are temporally correlated, so a random split puts adjacent --
    nearly identical -- frames in both sets. The model then scores well on
    validation by memorising, and the number means nothing. Taking validation
    from one end keeps the two sets genuinely separate.
    """
    if count <= 0:
        return [], []
    val_size = max(1, round(count * val_fraction)) if val_fraction > 0 else 0
    val_size = min(val_size, count - 1) if count > 1 else 0
    split_at = count - val_size
    return list(range(split_at)), list(range(split_at, count))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def box_line(class_id: int, box, width: int, height: int) -> str:
    """One YOLO detection label line: `class cx cy w h`, all normalised."""
    cx = ((box.x1 + box.x2) / 2) / width
    cy = ((box.y1 + box.y2) / 2) / height
    w = box.width / width
    h = box.height / height
    return " ".join([str(class_id)] + [f"{_clamp01(v):.6f}" for v in (cx, cy, w, h)])


def polygon_line(class_id: int, polygon, width: int, height: int) -> str:
    """One YOLO segmentation label line: `class x1 y1 x2 y2 ...`, normalised.

    Returns an empty string for a polygon too small to be a shape; YOLO
    rejects a label with fewer than three points and the whole file with it.
    """
    points = [(float(x) / width, float(y) / height) for x, y in polygon]
    if len(points) < 3:
        return ""
    coords = [f"{_clamp01(v):.6f}" for point in points for v in point]
    return " ".join([str(class_id)] + coords)


def write_data_yaml(root: Path, classes: list[str]) -> Path:
    """The dataset descriptor ultralytics trains from."""
    path = root / "data.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": dict(enumerate(classes)),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def prepare_tree(root: Path) -> None:
    """Create the images/ and labels/ layout ultralytics expects."""
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (root / kind / split).mkdir(parents=True, exist_ok=True)
