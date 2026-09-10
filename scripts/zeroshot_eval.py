"""Measure a vocabulary against a labelled dataset, without training anything.

    python scripts/zeroshot_eval.py --data datasets/sortwaste/data.yaml
    python scripts/zeroshot_eval.py --data datasets/sortwaste/data.yaml \
        --vocabulary vocab/waste_v3.yaml

This is the harness prompt tuning needs. Rewriting prompts is the cheapest
lever this project has -- it took class accuracy from 62% to 85% once already
-- but tuning against a handful of images is just overfitting with extra
steps. Pointed at a real labelled val split, the same edit becomes measurable.

Reports precision and recall (did it find the objects), class accuracy (did it
name them), and routing accuracy (did they reach the right bin). The last is
the one that matters, and it is always the highest of the three: a confusion
between two classes that share a bin costs nothing.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from recyclevision.evaluate import DEFAULT_IOU, match_detections, score_routing  # noqa: E402
from recyclevision.external import MappingError, read_yolo_data_yaml  # noqa: E402
from recyclevision.models import BoundingBox  # noqa: E402
from recyclevision.pipeline import DEFAULT_POLICY  # noqa: E402
from recyclevision.policy import RoutingPolicy  # noqa: E402
from recyclevision.vocabulary import DEFAULT_VOCAB, Vocabulary  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def read_truth(
    label_path: Path, names: list[str], width: int, height: int
) -> list[tuple[str, BoundingBox]]:
    """Ground truth from a YOLO label file, denormalised to pixels.

    Accepts both label shapes: `class cx cy w h` and a polygon, since a
    segmentation dataset stores the latter. A polygon collapses to its
    bounding box, which is all the matcher needs.
    """
    if not label_path.is_file():
        return []

    truth = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            values = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if class_id >= len(names):
            continue

        if len(values) == 4:
            cx, cy, w, h = values
            box = BoundingBox(
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            )
        else:
            xs = [v * width for v in values[0::2]]
            ys = [v * height for v in values[1::2]]
            if not xs or not ys:
                continue
            box = BoundingBox(min(xs), min(ys), max(xs), max(ys))

        truth.append((names[class_id], box))
    return truth


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="dataset data.yaml")
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--limit", type=int, default=200, help="images to evaluate")
    args = parser.parse_args(argv)

    try:
        root, names = read_yolo_data_yaml(args.data)
    except MappingError as exc:
        parser.error(str(exc))

    images_root = root / "images" / args.split
    labels_root = root / "labels" / args.split
    if not images_root.is_dir():
        parser.error(f"no {args.split} images at {images_root}")

    images = sorted(p for p in images_root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        parser.error(f"no images found in {images_root}")
    images = images[: args.limit]

    from recyclevision.detector import OpenVocabularyDetector

    vocabulary = Vocabulary.load(args.vocabulary)
    detector = OpenVocabularyDetector(vocabulary)
    policy = RoutingPolicy.load(args.policy)

    print(f"vocabulary  {vocabulary.name} ({len(vocabulary.classes)} classes)")
    print(f"policy      {policy.name}")
    print(f"evaluating  {len(images)} {args.split} image(s) at conf {args.confidence}\n")

    all_pairs: list[tuple[str, str]] = []
    false_positives: Counter[str] = Counter()
    missed: Counter[str] = Counter()

    for index, image_path in enumerate(images, 1):
        image = Image.open(image_path).convert("RGB")
        truth = read_truth(labels_root / f"{image_path.stem}.txt", names, image.width, image.height)
        predictions = [
            (d.label, d.box, d.confidence)
            for d in detector.detect(image, confidence=args.confidence)
        ]

        match = match_detections(predictions, truth, threshold=args.iou)
        all_pairs.extend(match.pairs)
        false_positives.update(match.false_positives)
        missed.update(match.missed)

        if index % 25 == 0 or index == len(images):
            print(f"  {index}/{len(images)}")

    matched = len(all_pairs)
    found = matched + sum(false_positives.values())
    findable = matched + sum(missed.values())

    print("\ndetection")
    print(f"  precision  {matched / found:.1%}" if found else "  precision  n/a")
    print(f"  recall     {matched / findable:.1%}" if findable else "  recall     n/a")
    print(f"  matched    {matched} of {findable} labelled object(s)")

    if not all_pairs:
        print("\nnothing matched — no classes or routing to score")
        return 1

    print("\nclassification and routing")
    print(score_routing(all_pairs, policy).report())

    if missed:
        print("\nmost missed ground-truth classes:")
        for name, count in missed.most_common(8):
            print(f"  {name:<24} {count:6}")
    if false_positives:
        print("\nmost common false-positive labels:")
        for name, count in false_positives.most_common(8):
            print(f"  {name:<24} {count:6}")

    print(
        "\nTo test a prompt change: copy the vocabulary, edit it, rebuild its "
        "embeddings, and re-run with --vocabulary. Same data, same policy, so "
        "the difference is the prompts."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
