"""Reserve images from several datasets for a hand-labelled evaluation set.

    python scripts/build_eval_set.py --from datasets/warp --from datasets/zerowaste \
        --out datasets/gold_eval --per-source 60

Why this exists, when all three source datasets ship annotations of their own:

**They are sparse, and they are in someone else's schema.** WaRP labels 1.74
objects per image in frames holding dozens -- measured, not guessed. Scoring
against that gives a precision figure whose true value lies somewhere between
4% and 56%, and no amount of care narrows it, because the gap is the
annotation policy rather than the model. Every dataset here has its own
version of that problem, and each translates into this project's vocabulary
with its own losses.

An evaluation set labelled by hand, exhaustively, in this project's classes is
the only thing that fixes it. It is small and expensive, which is exactly why
it should be built once, early, and then frozen.

**This script writes no labels.** Not empty ones either: in YOLO an empty
label file means "this image contains nothing", which is a real training
signal and would be a lie here. Images arrive bare and stay bare until a human
has been through them.

The source's own labels are translated and kept in `reference/`, out of the
way. Do not look at them while labelling -- the whole point is an independent
opinion. Afterwards, diffing the two says how complete the source annotations
were, which is worth knowing before training on them.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recyclevision.dataset import spread_sample, write_data_yaml  # noqa: E402
from recyclevision.external import MappingError, read_yolo_data_yaml  # noqa: E402
from recyclevision.vocabulary import DEFAULT_VOCAB, Vocabulary  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

#: The evaluation set is one split. Calling it `val` lets evaluate.py and
#: ultralytics read it with no special handling.
SPLIT = "val"


def collect(root: Path) -> list[tuple[Path, str]]:
    """Every labelled image in an imported dataset, with the split it came from.

    Sorted, so the same dataset always yields the same order and therefore the
    same sample.
    """
    found = []
    for split in ("train", "val"):
        images = root / "images" / split
        labels = root / "labels" / split
        if not images.is_dir():
            continue
        for image in sorted(p for p in images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
            if (labels / f"{image.stem}.txt").is_file():
                found.append((image, split))
    return found


def eval_name(image: Path, split: str, taken: set[str]) -> str:
    """A name unique within the flattened evaluation set.

    `import_dataset` prefixes by source, which keeps sources apart but not
    splits: a dataset may hold `frame_001.jpg` in both train and val, and
    those live in separate directories until this script flattens them into
    one. Left alone that silently drops an image and quietly shrinks the
    evaluation set, which is the kind of loss nothing downstream reports.
    """
    if image.name not in taken:
        return image.name
    return f"{image.stem}--{split}{image.suffix}"


def report_progress(root: Path) -> int:
    """How much of the evaluation set has actually been labelled."""
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        print(f"no manifest at {manifest_path}")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    labels = root / "labels" / SPLIT
    images = root / "images" / SPLIT
    done, empty, todo = [], [], []
    for name in sorted(manifest.get("images", {})):
        label = labels / f"{Path(name).stem}.txt"
        if not label.is_file():
            todo.append(name)
        elif label.read_text(encoding="utf-8").strip():
            done.append(name)
        else:
            # A deliberate "nothing here" is legitimate, but it is also what an
            # annotation tool writes when a frame was opened and skipped, so
            # it is counted apart rather than folded into either side.
            empty.append(name)

    total = len(manifest.get("images", {}))
    print(f"{root}: {total} image(s) reserved")
    print(f"  labelled          {len(done)}")
    print(f"  marked empty      {len(empty)}")
    print(f"  not yet labelled  {len(todo)}")
    if todo:
        print(f"\n  next few: {', '.join(todo[:5])}")
        print(f"  images are in {images}")
    by_source: dict[str, list[int]] = {}
    for name, entry in manifest.get("images", {}).items():
        stat = by_source.setdefault(entry.get("source_dataset", "?"), [0, 0])
        stat[1] += 1
        if (labels / f"{Path(name).stem}.txt").is_file():
            stat[0] += 1
    print("\n  per source:")
    for source, (finished, count) in sorted(by_source.items()):
        print(f"    {Path(source).name:<20} {finished:>4}/{count}")
    return 0 if not todo else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        type=Path,
        default=None,
        help="report labelling progress for an existing eval set and exit",
    )
    parser.add_argument(
        "--from",
        dest="sources",
        type=Path,
        action="append",
        help="an imported dataset root (repeat for each source)",
    )
    parser.add_argument("--out", type=Path, default=Path("datasets/gold_eval"))
    parser.add_argument("--per-source", type=int, default=60)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="same seed, same selection — an eval set has to be reproducible",
    )
    args = parser.parse_args(argv)

    if args.check is not None:
        return report_progress(args.check)
    if not args.sources:
        parser.error("--from is required unless --check is given")

    vocabulary = Vocabulary.load(args.vocabulary)
    images_out = args.out / "images" / SPLIT
    reference_out = args.out / "reference"
    images_out.mkdir(parents=True, exist_ok=True)
    reference_out.mkdir(parents=True, exist_ok=True)
    (args.out / "labels" / SPLIT).mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    for source in args.sources:
        if not source.is_dir():
            parser.error(f"not a directory: {source}")
        try:
            read_yolo_data_yaml(source / "data.yaml")
        except MappingError as exc:
            parser.error(f"{source} does not look like an imported dataset: {exc}")

        candidates = collect(source)
        if not candidates:
            parser.error(f"no labelled images under {source}")

        picked = spread_sample(candidates, args.per_source, seed=args.seed)
        print(f"{source.name}: {len(picked)} of {len(candidates)} image(s)")

        for image, split in picked:
            name = eval_name(image, split, set(manifest))
            shutil.copy2(image, images_out / name)
            native = source / "labels" / split / f"{image.stem}.txt"
            if native.is_file():
                shutil.copy2(native, reference_out / f"{Path(name).stem}.txt")
            manifest[name] = {
                "source_dataset": str(source),
                "source_image": str(image),
                "source_split": split,
                "labelled": False,
            }

    write_data_yaml(args.out, vocabulary.classes)
    # train points at the same place only so ultralytics does not complain;
    # nothing should ever be trained on this set.
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "purpose": "hand-labelled evaluation set — never train on this",
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seed": args.seed,
                "vocabulary": str(args.vocabulary),
                "images": manifest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n{len(manifest)} image(s) reserved -> {images_out}")
    print(f"reference labels (do not look until done) -> {reference_out}")
    print("\nNext:")
    print(f"  1. Label {images_out} from scratch in Label Studio or CVAT.")
    print("     Exhaustively: every object you can identify, not just the easy ones.")
    print("  2. Keep these images OUT of training:")
    print(f"       python scripts/import_dataset.py ... --exclude {args.out / 'manifest.json'}")
    print("  3. Check progress:")
    print(f"       python scripts/build_eval_set.py --check {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
