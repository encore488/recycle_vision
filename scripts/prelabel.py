"""Pre-label frames with the current detector, ready for human correction.

    python scripts/prelabel.py datasets/raw --out datasets/conveyor

Writes a complete YOLO segmentation dataset -- images, labels, data.yaml --
using the shipped open-vocabulary model. The labels are wrong often enough
that they must be corrected, but correcting a box is several times faster than
drawing one, and the model already emits masks, so nobody has to trace a
polygon by hand.

Open the result in any YOLO-format annotation tool (Label Studio and CVAT both
import it directly), fix what is wrong, and train on the corrected version.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from recyclevision.dataset import (  # noqa: E402
    DatasetStats,
    block_split,
    box_line,
    polygon_line,
    prepare_tree,
    write_data_yaml,
)
from recyclevision.vocabulary import DEFAULT_VOCAB, Vocabulary  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, help="directory of extracted frames")
    parser.add_argument("--out", type=Path, default=Path("datasets/conveyor"))
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="pre-label with your own trained model instead of the stock open-vocabulary "
        "one. This is the second and every later round: label a batch, train on it, "
        "then let that model pre-label the next batch. Each round gets faster.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.10,
        help="deliberately low: a spurious box is faster to delete than a missed object is to draw",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--boxes-only", action="store_true", help="write detection labels instead of segmentation"
    )
    args = parser.parse_args(argv)

    if not args.images.is_dir():
        parser.error(f"not a directory: {args.images}")
    frames = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not frames:
        parser.error(f"no images found in {args.images}")

    vocabulary = Vocabulary.load(args.vocabulary)
    class_index = {name: i for i, name in enumerate(vocabulary.classes)}

    if args.weights:
        from recyclevision.detector import TrainedSegmentationDetector

        detector = TrainedSegmentationDetector(args.weights)
        unknown = [n for n in detector.class_names if n not in class_index]
        if unknown:
            parser.error(
                f"{args.weights} emits classes the vocabulary does not list: {unknown}. "
                f"Pre-label with the vocabulary the model was trained on."
            )
        print(f"pre-labelling with {args.weights}")
    else:
        from recyclevision.detector import OpenVocabularyDetector

        detector = OpenVocabularyDetector(vocabulary)
        print(f"pre-labelling with the stock open vocabulary ({vocabulary.name})")

    prepare_tree(args.out)
    train_idx, val_idx = block_split(len(frames), args.val_fraction)
    split_of = dict.fromkeys(train_idx, "train") | dict.fromkeys(val_idx, "val")

    stats = DatasetStats(images=len(frames))
    per_class: Counter[str] = Counter()

    for index, frame in enumerate(frames):
        split = split_of[index]
        image = Image.open(frame).convert("RGB")
        shutil.copy2(frame, args.out / "images" / split / frame.name)

        lines = []
        for detection, outline in detector.detect_with_masks(image, confidence=args.confidence):
            class_id = class_index.get(detection.label)
            if class_id is None:
                continue
            if args.boxes_only or outline is None:
                line = box_line(class_id, detection.box, image.width, image.height)
            else:
                line = polygon_line(class_id, outline, image.width, image.height)
            if line:
                lines.append(line)
                per_class[detection.label] += 1

        if not lines:
            stats.empty_images.append(frame.name)
        # An empty .txt is how YOLO says "this image contains nothing", which
        # is a valid and useful training signal -- so write it either way.
        (args.out / "labels" / split / f"{frame.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

        print(f"  [{index + 1}/{len(frames)}] {frame.name} -> {len(lines)} instance(s) ({split})")

    stats.instances = sum(per_class.values())
    stats.per_class = dict(per_class)

    data_yaml = write_data_yaml(args.out, vocabulary.classes)

    print(f"\n{stats.report()}")
    print(f"\ntrain {len(train_idx)} / val {len(val_idx)} (split by frame block, not at random)")
    print(f"dataset -> {args.out}")
    print(f"descriptor -> {data_yaml}")
    print("\nThese labels are a starting point, not ground truth. Correct them before training:")
    print(f"  label-studio start   # or: cvat, or any YOLO-format editor, on {args.out}")
    print(f"  python train.py --data {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
