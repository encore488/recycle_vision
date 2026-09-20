"""Report what an unfamiliar dataset actually contains, before importing it.

    python scripts/inspect_dataset.py ~/Documents/recycle_data/zerowaste_data

Public waste datasets arrive in whatever format their authors used: YOLO text
files beside a data.yaml, COCO JSON, Pascal VOC XML, or masks. Guessing wrong
wastes a download and, worse, can half-succeed -- an importer that finds some
images and no labels produces a dataset of empty annotations, which trains a
model to believe every image is background.

So this looks first and reports: which format, where the annotations live,
how many images and instances, and the exact class names. Those class names
are what a `mappings/*.yaml` file needs, and they are the one thing that
cannot be guessed from outside.

Read-only. It writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

#: Deep enough to find a nested release directory, shallow enough that a
#: dataset of 100k files does not take a minute to walk.
MAX_DEPTH = 6


def find(root: Path, predicate, limit: int = 40) -> list[Path]:
    """Paths under `root` matching `predicate`, bounded in depth and count."""
    found: list[Path] = []
    stack = [(root, 0)]
    while stack and len(found) < limit:
        current, depth = stack.pop()
        if depth > MAX_DEPTH:
            continue
        try:
            entries = sorted(current.iterdir())
        except (OSError, PermissionError):
            continue
        for entry in entries:
            if entry.is_dir():
                stack.append((entry, depth + 1))
            elif predicate(entry):
                found.append(entry)
                if len(found) >= limit:
                    break
    return found


def looks_like_coco(path: Path) -> dict | None:
    """Parse a JSON file if it is a COCO annotation set, else None."""
    try:
        if path.stat().st_size > 600_000_000:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if isinstance(payload, dict) and "annotations" in payload and "categories" in payload:
        return payload
    return None


def report_coco(path: Path, payload: dict) -> list[str]:
    categories = {c["id"]: c.get("name", str(c["id"])) for c in payload.get("categories", [])}
    counts: Counter[str] = Counter()
    has_polygons = False
    for annotation in payload.get("annotations", []):
        counts[categories.get(annotation.get("category_id"), "?")] += 1
        segmentation = annotation.get("segmentation")
        if segmentation and not isinstance(segmentation, dict):
            has_polygons = True

    lines = [
        f"  images      {len(payload.get('images', []))}",
        f"  instances   {sum(counts.values())}",
        f"  geometry    {'polygons + boxes' if has_polygons else 'boxes only'}",
        "  classes:",
    ]
    width = max((len(n) for n in categories.values()), default=1)
    for name in sorted(categories.values()):
        lines.append(f"    {name:<{width}}  {counts.get(name, 0)}")
    return lines


def report_yolo(data_yaml: Path) -> list[str]:
    import yaml

    try:
        raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [f"  unreadable: {exc}"]
    if not isinstance(raw, dict):
        return ["  not a mapping — probably not a YOLO descriptor"]

    names = raw.get("names")
    if isinstance(names, dict):
        ordered = [names[k] for k in sorted(names, key=lambda k: int(k))]
    elif isinstance(names, list):
        ordered = list(names)
    else:
        ordered = []

    lines = [f"  splits      {[k for k in ('train', 'val', 'test') if k in raw]}"]
    lines.append(f"  classes     {len(ordered)}")
    for index, name in enumerate(ordered):
        lines.append(f"    {index:>3}  {name}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        parser.error(f"not a directory: {args.root}")

    print(f"inspecting {args.root}\n")

    images = find(args.root, lambda p: p.suffix.lower() in IMAGE_SUFFIXES, limit=5000)
    print(f"images found (capped at 5000): {len(images)}")
    if images:
        print(f"  e.g. {images[0].relative_to(args.root)}")

    yolo = find(args.root, lambda p: p.name in {"data.yaml", "dataset.yaml", "data.yml"})
    jsons = find(args.root, lambda p: p.suffix.lower() == ".json", limit=30)
    txt_labels = find(args.root, lambda p: p.suffix.lower() == ".txt" and p.parent.name != "", 5)

    found_any = False
    for descriptor in yolo:
        found_any = True
        print(f"\nYOLO descriptor: {descriptor.relative_to(args.root)}")
        for line in report_yolo(descriptor):
            print(line)

    for candidate in jsons:
        payload = looks_like_coco(candidate)
        if payload is None:
            continue
        found_any = True
        print(f"\nCOCO annotations: {candidate.relative_to(args.root)}")
        for line in report_coco(candidate, payload):
            print(line)

    if not found_any:
        print("\nNo YOLO descriptor and no COCO JSON found.")
        if txt_labels:
            print("There are .txt files, so this may be raw YOLO labels with no data.yaml:")
            for path in txt_labels[:3]:
                print(f"  {path.relative_to(args.root)}: {path.read_text()[:60]!r}")
        print("\nSend this output and the directory listing:")
        print(f"  find {args.root} -maxdepth 3 | head -50")
        return 1

    print("\nSend this output — the class names are what a mappings/*.yaml needs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
