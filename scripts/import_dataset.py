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
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recyclevision.coco import (  # noqa: E402
    CocoError,
    index_images,
    locate,
    read_coco,
    to_yolo_line,
)
from recyclevision.dataset import prepare_tree, write_data_yaml  # noqa: E402
from recyclevision.external import (  # noqa: E402
    ClassMapping,
    MappingError,
    build_index_map,
    label_dir_for,
    looks_already_imported,
    read_split_images_dir,
    read_yolo_data_yaml,
    remap_label_line,
    trees_overlap,
)
from recyclevision.vocabulary import DEFAULT_VOCAB, Vocabulary  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


SPLIT_NAMES = {"train", "val", "valid", "test"}


def source_split_tag(path: Path, fallback: str) -> str:
    """Which source split a file belongs to, from where it sits on disk.

    Two splits of one dataset can hold the same filename -- ZeroWaste's val
    and test share two frame names -- and both land in this project's single
    `val`. Without the source split in the output name the second overwrites
    the first, and the only trace is an image count two lower than the sum of
    the parts.

    SortWaste keeps its COCO at <split>/annotations/x.json, ZeroWaste at
    <split>/labels.json, so walking up for a split-shaped directory covers
    both.
    """
    for parent in path.parents:
        if parent.name.lower() in SPLIT_NAMES:
            return parent.name.lower()
    return fallback


def read_sources(root: Path) -> dict:
    """Which source contributed which filename prefix to a merged dataset."""
    path = root / "sources.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_sources(root: Path, record: dict) -> Path:
    """Record provenance beside the data, so a merged set can be accounted for.

    A training set assembled from several public datasets is otherwise
    unattributable a month later, and "which facility did this come from" is
    the question a cross-domain split depends on.
    """
    path = root / "sources.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def slugify(name: str) -> str:
    """A short filesystem-safe tag for one source dataset."""
    kept = [c.lower() if c.isalnum() else "-" for c in name]
    return "-".join(part for part in "".join(kept).split("-") if part)[:32]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_yaml",
        type=Path,
        help="the external dataset's data.yaml, or a COCO annotations .json",
    )
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default=None,
        help="COCO only: which split this file becomes. A COCO file describes "
        "one split and does not say which, so it has to be stated.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument(
        "--link",
        action="store_true",
        help="hardlink images instead of copying — worth it for large datasets",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="import even when a declared split resolved to an empty directory",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        default=None,
        help="a gold_eval manifest.json; images reserved for hand-labelled "
        "evaluation are skipped. Training on an image that is also in the "
        "evaluation set makes every number after it meaningless, and nothing "
        "in the output would look wrong.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="prepended to every output filename; defaults to a slug of the mapping "
        "name. Public datasets number their frames from 1, so importing two into one "
        "directory without this silently overwrites the overlap. Pass '' to disable.",
    )
    args = parser.parse_args(argv)

    is_coco = args.data_yaml.suffix.lower() == ".json"
    if is_coco and args.split is None:
        parser.error(
            "--split is required for a COCO file: it describes one split and does not say which."
        )

    try:
        mapping = ClassMapping.load(args.mapping)
        if is_coco:
            source_classes, coco_images = read_coco(args.data_yaml)
            source_root = args.data_yaml.parent
        else:
            source_root, source_classes = read_yolo_data_yaml(args.data_yaml)
            coco_images = []
    except (MappingError, CocoError) as exc:
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
    prefix = slugify(mapping.name) + "-" if args.prefix is None else args.prefix
    sources = read_sources(args.out)

    # Refuse before writing: a prefix already claimed by a different source
    # means this import would overwrite that source's images, and the label
    # count afterwards would look perfectly healthy.
    owner = sources.get(prefix, {}).get("mapping")
    if owner is not None and owner != mapping.name:
        parser.error(
            f"\nprefix {prefix!r} in {args.out} already belongs to {owner!r}.\n"
            f"Importing {mapping.name!r} over it would overwrite that source's images.\n"
            "Pass a distinct --prefix."
        )
    if prefix:
        print(f"\nfilenames prefixed {prefix!r}, so this can be merged with other sources")
    empty_splits: list[str] = []
    reserved: set[str] = set()
    if args.exclude is not None:
        if not args.exclude.is_file():
            parser.error(f"no manifest at {args.exclude}")
        held = json.loads(args.exclude.read_text(encoding="utf-8")).get("images", {})
        # Matched on the OUTPUT filename, which is the only identity the two
        # sides share. The evaluation set was built from an imported dataset,
        # so its recorded paths point into that import; this run is walking
        # the raw source, whose paths never coincide with them. What does
        # coincide is the name this import will write -- prefix plus original
        # stem -- which is exactly the basename the eval set copied.
        reserved = {Path(entry.get("source_image", "")).name for entry in held.values()}
        reserved.discard("")
        print(f"excluding {len(reserved)} image(s) reserved for evaluation")

    counts: Counter[str] = Counter()
    copied = unlabelled = excluded = 0

    if is_coco:
        # A COCO file is as often in an `annotations/` directory beside the
        # images as it is among them. SortWaste puts it at
        # <split>/annotations/train_coco.json with frames at <split>/images/;
        # ZeroWaste puts it at <split>/labels.json with frames at
        # <split>/data/. Search where it sits, then step up until images
        # appear, rather than demanding one convention.
        index = index_images(source_root)
        for _ in range(2):
            if index:
                break
            source_root = source_root.parent
            index = index_images(source_root)
        if not index:
            parser.error(
                f"\nfound no image files at or above {args.data_yaml.parent}.\n"
                "The annotations are there but the frames are not — check the "
                "archive extracted fully."
            )
        source_tag = source_split_tag(args.data_yaml, args.split)
        print(f"\nimages resolved under {source_root} (source split {source_tag!r})")
        missing = 0
        for record in coco_images:
            image = locate(record.file_name, index)
            if image is None:
                missing += 1
                continue

            width, height = record.width, record.height
            if width <= 0 or height <= 0:
                # Some releases omit the dimensions; the file itself has them.
                from PIL import Image as _Image

                with _Image.open(image) as opened:
                    width, height = opened.size

            lines = []
            for name, x, y, w, h in record.boxes:
                target = mapping.translate(name)
                if target is None:
                    continue
                line = to_yolo_line(vocabulary.classes.index(target), x, y, w, h, width, height)
                if line:
                    lines.append(line)
                    counts[target] += 1

            base = f"{source_tag}-{Path(record.file_name).stem}"
            stem = f"{prefix}{base}" if prefix else base
            destination = args.out / "images" / args.split / f"{stem}{image.suffix}"
            if reserved and destination.name in reserved:
                excluded += 1
                continue
            if args.link:
                try:
                    if destination.exists():
                        destination.unlink()
                    os.link(image, destination)
                except (OSError, FileExistsError):
                    shutil.copy2(image, destination)
            else:
                shutil.copy2(image, destination)
            (args.out / "labels" / args.split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            copied += 1

        print(f"\n{args.split}: {copied} image(s) written")
        if missing:
            # Never silent: a COCO file listing images that are not on disk
            # means a partial extract, and importing the remainder quietly
            # would be the same failure as the empty-split case below.
            parser.error(
                f"\n{missing} image(s) named in {args.data_yaml.name} were not found "
                f"under {source_root}.\n"
                "The archive probably did not extract fully. Nothing has been kept."
            )
    else:
        # Read the layout from the descriptor rather than assuming one. SortWaste
        # is <split>/images/, WaRP is images/<split>/, and both are valid YOLO --
        # ultralytics only ever follows the paths the descriptor declares. Guessing
        # produced a silent success that imported nothing.
        for split in ("train", "val", "test"):
            images_root = read_split_images_dir(args.data_yaml, split)
            if images_root is None:
                continue
            if not images_root.is_dir():
                print(f"\n{split}: declared as {images_root}, which does not exist — skipping")
                continue
            labels_root = label_dir_for(images_root)

            # This project's datasets have two splits. A source `test` split is
            # real held-out data and throwing it away would discard thousands of
            # instances, so it joins val, and the line below says so.
            out_split = "train" if split == "train" else "val"

            images = sorted(p for p in images_root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
            # Name the directory actually read. A descriptor written on another
            # machine declares paths that cannot exist here, and the resolver
            # falls back to matching the tail -- worth seeing rather than
            # trusting silently.
            destination_note = f" -> {out_split}" if split != out_split else ""
            print(f"\n{split}{destination_note}: {len(images)} image(s) in {images_root}")
            if not images:
                empty_splits.append(split)
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

                base = f"{split}-{image.stem}"
                stem = f"{prefix}{base}" if prefix else base
                destination = args.out / "images" / out_split / f"{stem}{image.suffix}"
                if reserved and destination.name in reserved:
                    excluded += 1
                    continue
                if args.link:
                    try:
                        if destination.exists():
                            destination.unlink()
                        os.link(image, destination)
                    except (OSError, FileExistsError):
                        shutil.copy2(image, destination)
                else:
                    shutil.copy2(image, destination)

                (args.out / "labels" / out_split / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
                copied += 1

    # A split that resolved to a real directory holding no images is not a
    # missing split -- it is a directory that should have had images in it.
    # Importing the remainder and calling it a dataset is how a fraction of
    # the intended data gets trained on with nothing looking wrong.
    if empty_splits and copied and not args.allow_partial:
        parser.error(
            f"\n{', '.join(empty_splits)} resolved to real directories containing no "
            f"images, while other splits had some.\n"
            "That is a partial dataset, not a complete one, so nothing has been kept.\n\n"
            "Usually the archive did not extract fully, or the images live somewhere "
            "other than\nwhere the descriptor says. Check what is actually there:\n"
            + "\n".join(
                f"  ls {read_split_images_dir(args.data_yaml, s)} | head" for s in empty_splits
            )
            + "\n\nRe-run once they are populated, or pass --allow-partial to import "
            "what exists."
        )

    # An import that wrote nothing is a failure, however calmly it ran. The
    # previous version printed "imported 0 image(s)" beside a dataset path and
    # a suggested next command, which is how an empty dataset gets trained on.
    if not copied:
        parser.error(
            f"\nimported no images from {args.data_yaml}.\n"
            + (
                f"{unlabelled} image(s) were found but had no matching label file — "
                "check the\nlabels directory sits beside images/ as YOLO expects.\n"
                if unlabelled
                else "No split in the descriptor pointed at a directory containing "
                "images.\nRun scripts/inspect_dataset.py on the source to see its "
                "real layout.\n"
            )
            + "No images or labels were written; the output directory is empty."
        )

    data_yaml = write_data_yaml(args.out, vocabulary.classes)
    sources[prefix] = {
        "mapping": mapping.name,
        "source": str(args.data_yaml),
        "images": copied,
        "instances": dict(counts),
    }
    write_sources(args.out, sources)

    print(f"\nimported {copied} image(s)")
    if excluded:
        print(f"held back {excluded} image(s) reserved for evaluation")
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
