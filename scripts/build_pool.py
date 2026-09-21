"""Assemble a balanced training pool from several imported datasets.

    python scripts/build_pool.py --from datasets/sortwaste --from datasets/zerowaste \
        --out datasets/pool --images 3000

Two problems this solves, both measured rather than assumed.

**Imbalance.** Across the imported sources, plastic bottle is 42.5% of all
instances and glass bottle is 0.4% -- 99:1. A random subset preserves that
ratio, so the rare classes arrive in numbers too small to learn while most of
the images taken teach the majority class something it already knows.
`balanced_selection` fills classes rarest first instead.

**Validation cost.** Detection accuracy scales with the log of dataset size,
so a subset costs little. Validation cost scales linearly and is paid every
epoch: 1,499 ZeroWaste images took 17m43s, so a full 3,057-image val split
would spend about 24 hours validating across a 40-epoch run -- more than the
training. The val split is therefore capped and spread-sampled, which changes
the metric's precision but not what it measures.

Images are hardlinked where the filesystem allows, so a pool costs no disk
beyond its label files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recyclevision.dataset import (  # noqa: E402
    balanced_selection,
    prepare_tree,
    spread_sample,
    write_data_yaml,
)
from recyclevision.external import MappingError, read_yolo_data_yaml  # noqa: E402
from recyclevision.vocabulary import DEFAULT_VOCAB, Vocabulary  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def label_counts(label: Path, names: list[str]) -> Counter:
    """Instances per class name in one YOLO label file."""
    counts: Counter = Counter()
    try:
        text = label.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return counts
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        if 0 <= index < len(names):
            counts[names[index]] += 1
    return counts


def collect(root: Path, split: str, names: list[str]) -> list[tuple[Path, Counter]]:
    """Every labelled image in one split, with what it contains."""
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    if not images_dir.is_dir():
        return []
    found = []
    for image in sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        label = labels_dir / f"{image.stem}.txt"
        if label.is_file():
            found.append((image, label_counts(label, names)))
    return found


def place(image: Path, label: Path, out: Path, split: str) -> None:
    """Hardlink an image and copy its label into the pool."""
    destination = out / "images" / split / image.name
    if destination.exists():
        destination.unlink()
    try:
        destination.hardlink_to(image)
    except (OSError, FileExistsError):
        shutil.copy2(image, destination)
    if label.is_file():
        shutil.copy2(label, out / "labels" / split / f"{image.stem}.txt")


def report(title: str, counts: Counter) -> None:
    total = sum(counts.values())
    print(f"\n  {title}")
    if not total:
        print("    (nothing)")
        return
    width = max(len(name) for name in counts)
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<{width}}  {count:>7}  {count / total:5.1%}")
    ratio = max(counts.values()) / max(1, min(counts.values()))
    print(f"    {'imbalance':<{width}}  {ratio:>6.0f}:1")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="sources", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", type=int, default=3000, help="training images to keep")
    parser.add_argument(
        "--val-images",
        type=int,
        default=500,
        help="validation images to keep. Validation is paid every epoch, so a full "
        "split can cost more hours than the training it is measuring.",
    )
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--exclude",
        type=Path,
        default=None,
        help="a gold_eval manifest.json; reserved images are kept out",
    )
    args = parser.parse_args(argv)

    vocabulary = Vocabulary.load(args.vocabulary)
    names = list(vocabulary.classes)

    reserved: set[str] = set()
    if args.exclude is not None:
        if not args.exclude.is_file():
            parser.error(f"no manifest at {args.exclude}")
        held = json.loads(args.exclude.read_text(encoding="utf-8")).get("images", {})
        reserved = {Path(e.get("source_image", "")).name for e in held.values()}
        reserved.discard("")
        print(f"excluding {len(reserved)} image(s) reserved for evaluation")

    train_pool: list[tuple[Path, Counter]] = []
    val_pool: list[tuple[Path, Counter]] = []
    per_source: dict[str, dict[str, int]] = {}

    for source in args.sources:
        if not source.is_dir():
            parser.error(f"no imported dataset at {source}")
        try:
            _root, source_names = read_yolo_data_yaml(source / "data.yaml")
        except MappingError as exc:
            parser.error(f"{source}: {exc}")
        if source_names != names:
            parser.error(
                f"\n{source} uses a different class list from {args.vocabulary}.\n"
                "Pooling datasets whose indices disagree would silently relabel "
                "every box.\nRe-import it against this vocabulary."
            )

        train = [(i, c) for i, c in collect(source, "train", names) if i.name not in reserved]
        val = [(i, c) for i, c in collect(source, "val", names) if i.name not in reserved]
        train_pool += train
        val_pool += val
        per_source[source.name] = {"train": len(train), "val": len(val)}
        print(f"{source.name}: {len(train)} train, {len(val)} val")

    if not train_pool:
        parser.error("no training images found in any source")

    before = Counter()
    for _image, counts in train_pool:
        before.update(counts)

    by_path = {image: (image, counts) for image, counts in train_pool}
    chosen = balanced_selection([(str(i), c) for i, c in train_pool], args.images, seed=args.seed)
    chosen_pairs = [by_path[Path(key)] for key in chosen]

    val_chosen = spread_sample(val_pool, min(args.val_images, len(val_pool)), seed=args.seed)

    prepare_tree(args.out)
    after = Counter()
    for image, counts in chosen_pairs:
        place(
            image,
            image.parent.parent.parent / "labels" / "train" / f"{image.stem}.txt",
            args.out,
            "train",
        )
        after.update(counts)
    for image, _counts in val_chosen:
        place(
            image,
            image.parent.parent.parent / "labels" / "val" / f"{image.stem}.txt",
            args.out,
            "val",
        )

    write_data_yaml(args.out, names)
    (args.out / "pool.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seed": args.seed,
                "sources": per_source,
                "train_images": len(chosen_pairs),
                "val_images": len(val_chosen),
                "train_instances": dict(after),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\npool -> {args.out}")
    print(f"  train {len(chosen_pairs)} image(s) of {len(train_pool)} available")
    print(f"  val   {len(val_chosen)} image(s) of {len(val_pool)} available")
    report("instance balance BEFORE (all available training data)", before)
    report("instance balance AFTER (what was selected)", after)
    print(f"\nNext:\n  python train.py --data {args.out / 'data.yaml'} --epochs 40 --cache disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
