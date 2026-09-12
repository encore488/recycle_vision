"""Import an external YOLO dataset into this project's vocabulary.

    python scripts/import_dataset.py path/to/external/data.yaml \
        --mapping mappings/sortwaste.yaml --out datasets/sortwaste

Public waste datasets label materials; this project routes bins from items.
The mapping file translates one to the other, and every source class must have
a rule -- an unmapped class would be silently dropped, teaching the model that
those objects are background, which is worse than not training on them.

Nothing is written until the mapping is verified complete, so a run either
produces a correct dataset or produces an error and no files.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recyclevision.dataset import prepare_tree, write_data_yaml  # noqa: E402
from recyclevision.external import (  # noqa: E402
    ClassMapping,
    MappingError,
    build_index_map,
    looks_already_imported,
    read_yolo_data_yaml,
    remap_label_line,
    trees_overlap,
)
from recyclevision.vocabulary import DEFAULT_VOCAB, Vocabulary  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_yaml", type=Path, help="the external dataset's data.yaml")
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument(
        "--link",
        action="store_true",
        help="hardlink images instead of copying — worth it for large datasets",
    )
    args = parser.parse_args(argv)

    try:
        source_root, source_classes = read_yolo_data_yaml(args.data_yaml)
        mapping = ClassMapping.load(args.mapping)
    except MappingError as exc:
        parser.error(str(exc))

    print(f"{mapping.name}: {len(source_classes)} source class(es)")
    for index, name in enumerate(source_classes):
        print(f"  {index:3}  {name}")

    vocabulary = Vocabulary.load(args.vocabulary)

    # Everything below refuses before touching the filesystem, so a wrong
    # invocation costs nothing.

    if trees_overlap(source_root, args.out):
        parser.error(
            f"\n--out {args.out} is the same tree as the source at {source_root}.\n"
            "The import would rewrite the labels it is reading and then fail "
            "copying files onto themselves, leaving the source half-converted.\n"
            "Point --out at a new directory."
        )

    if looks_already_imported(source_classes, vocabulary.classes):
        parser.error(
            f"\n{args.data_yaml} is already in the {vocabulary.name} vocabulary — "
            "it looks like\nthis script's own output, not an outside dataset.\n\n"
            "Point at the original download instead. Both files are called "
            "data.yaml, so:\n"
            "    find ~ -name data.yaml -not -path '*/datasets/*'\n\n"
            "To pick up a vocabulary change, re-import from that original — "
            "re-importing an\nimport cannot recover what the first one discarded."
        )

    missing = mapping.unmapped(source_classes)
    if missing:
        parser.error(
            f"\n{args.mapping} has no rule for: {missing}\n"
            "Add each one (use `null` to drop it deliberately). Leaving a class "
            "unmapped would teach the model those objects are background."
        )

    unknown = mapping.unknown_targets(vocabulary.classes)
    if unknown:
        parser.error(
            f"\n{args.mapping} maps to classes {vocabulary.name} does not have: {unknown}\n"
            "Either add them to the vocabulary or map to an existing class."
        )

    index_map = build_index_map(source_classes, mapping, vocabulary.classes)

    dropped = [n for n in mapping.dropped() if n in source_classes]
    if dropped:
        print(f"\ndeliberately dropped: {', '.join(dropped)}")
    if mapping.cannot_express:
        print("\nthis dataset cannot express:")
        for item in mapping.cannot_express:
            print(f"  - {item}")

    stale = prepare_tree(args.out)
    for cache in stale:
        # Ultralytics would otherwise reuse this and ignore what we write.
        print(f"  removed stale label cache: {cache}")
    counts: Counter[str] = Counter()
    copied = unlabelled = 0

    for split in ("train", "val"):
        images_root = source_root / "images" / split
        labels_root = source_root / "labels" / split
        if not images_root.is_dir():
            print(f"\nno {split} split at {images_root} — skipping")
            continue

        images = sorted(p for p in images_root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        for image in images:
            label = labels_root / image.relative_to(images_root).with_suffix(".txt")
            if not label.is_file():
                unlabelled += 1
                continue

            lines = []
            for line in label.read_text(encoding="utf-8").splitlines():
                remapped = remap_label_line(line, index_map)
                if remapped:
                    lines.append(remapped)
                    counts[vocabulary.classes[int(remapped.split()[0])]] += 1

            destination = args.out / "images" / split / image.name
            if args.link:
                try:
                    destination.hardlink_to(image)
                except (OSError, FileExistsError):
                    shutil.copy2(image, destination)
            else:
                shutil.copy2(image, destination)

            (args.out / "labels" / split / f"{image.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            copied += 1

        print(f"  {split}: {len(images)} image(s)")

    data_yaml = write_data_yaml(args.out, vocabulary.classes)

    print(f"\nimported {copied} image(s)")
    if unlabelled:
        print(f"skipped {unlabelled} image(s) with no label file")
    if counts:
        width = max(len(name) for name in counts)
        print("\ninstances per class, after translation:")
        for name, count in counts.most_common():
            print(f"  {name:<{width}}  {count:7}")
        thin = [n for n, c in counts.items() if c < 100]
        if thin:
            print(f"\n  ⚠️ under 100 instances, may not learn: {', '.join(sorted(thin))}")

    print(f"\ndataset -> {args.out}\ndescriptor -> {data_yaml}")
    print(f"\nNext:\n  python scripts/zeroshot_eval.py --data {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
