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


#: Archive extensions, including the numbered parts of a split zip.
ARCHIVE_SUFFIXES = {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".tgz"}


def is_archive(path: Path) -> bool:
    """Whether a file is an archive, split-zip parts included.

    A split zip is `name.z01`, `name.z02`, ... alongside a terminal
    `name.zip`, and no part extracts on its own -- unzip needs the whole set
    and reads them through the final `.zip`. Downloading one part and finding
    "no dataset here" is an easy hour to lose.
    """
    suffix = path.suffix.lower()
    if suffix in ARCHIVE_SUFFIXES:
        return True
    # .z01 .. .z99, and the .r00 form some tools still emit.
    return len(suffix) == 4 and suffix[1] in "zr" and suffix[2:].isdigit()


def describe_archives(archives: list[Path], root: Path) -> list[str]:
    """Explain an unextracted download, including an incomplete split set."""
    lines = ["Found archives rather than a dataset. Extract them first:"]
    for path in sorted(archives)[:12]:
        size = path.stat().st_size / 1e9 if path.exists() else 0
        lines.append(f"  {path.relative_to(root)}  ({size:.2f} GB)")

    parts = [p for p in archives if p.suffix.lower() != ".zip" and is_archive(p)]
    if parts:
        stems = {p.stem for p in parts}
        terminal = {p.stem for p in archives if p.suffix.lower() == ".zip"}
        missing = stems - terminal
        lines.append("")
        lines.append("These are parts of a SPLIT zip. No part extracts on its own —")
        lines.append("unzip needs every part plus the terminal .zip, in one directory.")
        if missing:
            lines.append("")
            for stem in sorted(missing):
                lines.append(f"  {stem}.zip is MISSING — the split cannot be opened without it.")
        lines.append("")
        lines.append("Once every part is present:")
        lines.append("  cd <that directory>")
        lines.append("  zip -s0 <name>.zip --out whole.zip   # join the parts")
        lines.append("  unzip whole.zip")
    return lines


def clean_name(name: str) -> str:
    """Strip an index prefix some converters bake into the class name.

    SortWaste's YOLO descriptor lists "0 pet", "1 pead", ... "4 ecal" -- the
    ORIGINAL category id, kept inside the name, while the YOLO index beside it
    is a fresh 0-based renumbering. The originals skipped 3, so the two
    disagree from that point on. A converter careless enough to do that is
    careless enough to have renumbered the names without renumbering the label
    files, which would silently mislabel a third of the dataset.
    """
    head, _, tail = name.strip().partition(" ")
    return tail.strip() if head.isdigit() and tail else name.strip()


def count_yolo_labels(root: Path, names: list[str]) -> Counter:
    """Instances per class name across every YOLO label file under `root`."""
    counts: Counter[str] = Counter()
    for label in root.rglob("*.txt"):
        if label.parent.name in {"images"} or "label" not in str(label.parent).lower():
            continue
        try:
            text = label.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                index = int(fields[0])
            except ValueError:
                continue
            if 0 <= index < len(names):
                counts[clean_name(names[index])] += 1
    return counts


def cross_check(yolo_counts: Counter, coco_counts: Counter) -> list[str]:
    """Compare a YOLO conversion against the COCO annotations it came from.

    Both describe the same images, so the per-class totals must agree. If they
    do not, the conversion reindexed something it should not have, and
    training on it would be training on wrong labels with nothing to show for
    it -- every metric would look plausible.
    """
    lines = []
    every = sorted(set(yolo_counts) | set(coco_counts))
    width = max((len(n) for n in every), default=1)
    lines.append(f"  {'class':<{width}}  {'YOLO':>8}  {'COCO':>8}")
    disagreements = 0
    for name in every:
        mine, theirs = yolo_counts.get(name, 0), coco_counts.get(name, 0)
        flag = "" if mine == theirs else "   <-- MISMATCH"
        if mine != theirs:
            disagreements += 1
        lines.append(f"  {name:<{width}}  {mine:>8}  {theirs:>8}{flag}")

    if disagreements:
        lines.append("")
        lines.append(f"  {disagreements} class(es) disagree. The YOLO conversion does not match")
        lines.append("  the COCO annotations it was derived from — do not train on it.")
        lines.append("  Import from the COCO files instead.")
    else:
        lines.append("")
        lines.append("  YOLO conversion agrees with the COCO annotations. Safe to import.")
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
    coco_totals: Counter[str] = Counter()
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
        categories = {c["id"]: c.get("name", str(c["id"])) for c in payload.get("categories", [])}
        for annotation in payload.get("annotations", []):
            coco_totals[clean_name(categories.get(annotation.get("category_id"), "?"))] += 1

    # When a dataset ships both, they describe the same images and must agree.
    if yolo and coco_totals:
        import yaml as _yaml

        descriptor = yolo[0]
        raw = _yaml.safe_load(descriptor.read_text(encoding="utf-8"))
        names = raw.get("names") if isinstance(raw, dict) else None
        if isinstance(names, dict):
            ordered = [names[k] for k in sorted(names, key=lambda k: int(k))]
        elif isinstance(names, list):
            ordered = list(names)
        else:
            ordered = []
        if ordered:
            print("\ncross-checking the YOLO conversion against the COCO annotations")
            yolo_counts = count_yolo_labels(descriptor.parent, ordered)
            if not yolo_counts:
                print("  no YOLO label files found beside the descriptor — cannot check")
            else:
                for line in cross_check(yolo_counts, coco_totals):
                    print(line)

    if not found_any:
        archives = find(args.root, is_archive, limit=30)
        if archives:
            print()
            for line in describe_archives(archives, args.root):
                print(line)
            return 1

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
