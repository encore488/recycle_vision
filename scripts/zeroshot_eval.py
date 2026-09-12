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

from recyclevision.detector import DEFAULT_IMGSZ, DEFAULT_MAX_AREA_FRACTION  # noqa: E402
from recyclevision.evaluate import (  # noqa: E402
    DEFAULT_IOU,
    best_overlaps,
    match_detections,
    overlap_histogram,
    per_class_counts,
    score_routing,
)
from recyclevision.external import MappingError, read_yolo_data_yaml  # noqa: E402
from recyclevision.models import BoundingBox  # noqa: E402
from recyclevision.pipeline import DEFAULT_POLICY  # noqa: E402
from recyclevision.policy import RoutingPolicy  # noqa: E402
from recyclevision.vocabulary import DEFAULT_VOCAB, Vocabulary  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

#: Confidences reported by --sweep.
#:
#: Extends well below 0.01 because on WaRP the best F1 landed at the lowest
#: threshold tested, which means the optimum was off the bottom of the range
#: and the sweep could not see it. A sweep whose best value is at its own edge
#: has not finished answering the question.
SWEEP_THRESHOLDS = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.10, 0.15, 0.25, 0.40]


#: Below this many instances a per-class warning is noise, not signal.
MIN_CLASS_INSTANCES = 10


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
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help="inference resolution. A 1920x1080 frame at the old 640 default "
        "shrinks every object threefold; raise it if recall looks impossible.",
    )
    parser.add_argument("--limit", type=int, default=200, help="images to evaluate")
    parser.add_argument(
        "--max-area",
        type=float,
        default=DEFAULT_MAX_AREA_FRACTION,
        help="drop predictions covering more than this share of the frame",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="detect once at a very low threshold and report metrics at several "
        "confidences. Answers whether the objects are being found at all or "
        "merely scored below the cutoff — one inference pass, not six.",
    )
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
    detector = OpenVocabularyDetector(vocabulary, imgsz=args.imgsz, max_area_fraction=args.max_area)
    policy = RoutingPolicy.load(args.policy)

    print(f"vocabulary  {vocabulary.name} ({len(vocabulary.classes)} classes)")
    print(f"policy      {policy.name}")
    print(
        f"evaluating  {len(images)} {args.split} image(s) at conf {args.confidence}, "
        f"imgsz {args.imgsz}\n"
    )

    thresholds = SWEEP_THRESHOLDS if args.sweep else [args.confidence]
    floor = min(thresholds)

    # One inference pass at the lowest threshold; every higher threshold is a
    # filter over the same predictions. Re-running detection per threshold
    # would cost six passes to learn the same thing.
    per_image: list[tuple[list, list]] = []
    area_fractions: list[float] = []
    for index, image_path in enumerate(images, 1):
        image = Image.open(image_path).convert("RGB")
        truth = read_truth(labels_root / f"{image_path.stem}.txt", names, image.width, image.height)
        predictions = [
            (d.label, d.box, d.confidence) for d in detector.detect(image, confidence=floor)
        ]
        per_image.append((predictions, truth))
        frame_area = float(image.width * image.height)
        if frame_area:
            area_fractions.extend(box.area / frame_area for _label, box, _c in predictions)

        if index % 25 == 0 or index == len(images):
            print(f"  {index}/{len(images)}")

    # Which of our classes this dataset actually annotates.
    #
    # WaRP labels 1.74 objects per image in frames containing dozens: it
    # annotates its own 28 target classes and ignores everything else present.
    # A prediction of "plastic wrapper" on a real plastic wrapper therefore
    # scores as a false positive, because the dataset never labelled wrappers
    # -- not because the model was wrong.
    #
    # Raw precision on such a dataset measures the annotation policy as much as
    # the model. Restricting false positives to classes the dataset does label
    # is fairer. It is still only an upper bound on error: even for those
    # classes the annotation may not be exhaustive.
    labelled_classes = {label for _p, truth in per_image for label, _b in truth}

    # Where the model put its boxes, before any threshold is applied.
    overlaps = [v for predictions, truth in per_image for v in best_overlaps(predictions, truth)]
    print(f"\nwhere {len(overlaps)} prediction(s) landed, at conf >= {floor}")
    histogram = overlap_histogram(overlaps)
    for label, count, share in histogram:
        print(f"  {label:24} {count:6}  {share:5.1%}")

    # How much of the frame each box covers. A prediction spanning most of the
    # image is the model describing the scene rather than finding an object in
    # it, which overlaps nothing and wrecks precision while adding no
    # detections. Reported here because no other number reveals it.
    if area_fractions:
        scene_sized = sum(1 for v in area_fractions if v > 0.5)
        large = sum(1 for v in area_fractions if v > 0.25)
        print("\nbox sizes, as a share of the frame")
        print(
            f"  over 50% of the frame    {scene_sized:6}  {scene_sized / len(area_fractions):5.1%}"
        )
        print(f"  over 25% of the frame    {large:6}  {large / len(area_fractions):5.1%}")
        if scene_sized:
            print("  -> scene-sized boxes present. Lower --max-area to drop them;")
            print("     they overlap nothing and cost precision for no recall.")

    if args.sweep:
        print("\nby confidence threshold")
        print("  ('fair' counts a false positive only for classes this dataset labels;")
        print("   F1 combines fair precision with recall, to pick an operating point)")
        header = (
            f"  {'conf':>7}{'preds':>8}{'matched':>9}{'prec':>7}{'fair':>7}{'recall':>8}{'F1':>7}"
        )
        print(header)
        recalls: list[float] = []
        fairs: list[float] = []
        f1s: list[float] = []
        raws: list[float] = []
        for threshold in thresholds:
            kept = [
                ([p for p in predictions if p[2] >= threshold], truth)
                for predictions, truth in per_image
            ]
            total_matched = total_found = scoreable = 0
            for predictions, truth in kept:
                result = match_detections(predictions, truth, threshold=args.iou)
                total_matched += len(result.pairs)
                total_found += len(result.pairs) + len(result.false_positives)
                scoreable += len(result.pairs) + sum(
                    1 for label in result.false_positives if label in labelled_classes
                )
            findable = sum(len(truth) for _p, truth in kept)
            precision = total_matched / total_found if total_found else 0.0
            fair = total_matched / scoreable if scoreable else 0.0
            recall = total_matched / findable if findable else 0.0
            f1 = 2 * fair * recall / (fair + recall) if (fair + recall) else 0.0
            recalls.append(recall)
            fairs.append(fair)
            f1s.append(f1)
            raws.append(precision)
            print(
                f"  {threshold:>7.3f}{total_found:>8}{total_matched:>9}"
                f"{precision:>7.1%}{fair:>7.1%}{recall:>8.1%}{f1:>7.3f}"
            )
        # Judge from the curve, not from a canned sentence. The histogram above
        # can look damning while this table shows the detections plainly exist,
        # and an earlier version of this script printed exactly that
        # contradiction.
        print(f"\n  this dataset annotates only: {', '.join(sorted(labelled_classes))}")

        if f1s:
            best_index = f1s.index(max(f1s))
            best_conf = thresholds[best_index]
            print(
                f"  best operating point: conf {best_conf:.3f} — "
                f"fair precision {fairs[best_index]:.1%}, recall {recalls[best_index]:.1%}, "
                f"F1 {f1s[best_index]:.3f}"
            )
            if best_index == 0:
                print("  ⚠️ that is the lowest threshold tested, so the true optimum may be")
                print("     lower still. Widen SWEEP_THRESHOLDS before trusting it.")

            # Precision here is a range, not a measurement, and saying so is the
            # difference between an honest report and a flattering one.
            print(
                f"\n  at that point true precision lies between {raws[best_index]:.1%} and "
                f"{fairs[best_index]:.1%}:\n"
                f"    raw  counts correct detections of unlabelled material as errors\n"
                f"    fair assumes every unlabelled-class prediction is correct\n"
                f"  Neither bound is the answer, and the gap is this dataset's doing."
            )

        # Per class, at the operating point the table just chose. The aggregate
        # above cannot distinguish "this class is never found" from "this class
        # is found and called something else", and those need opposite fixes.
        #
        # Two label spaces are in play and they must not share a column: a
        # ground-truth object is *found* by whatever prediction covers it,
        # whatever that prediction is called, while a prompt *fires* under its
        # own name. An earlier version of this table printed one count of each
        # per row and produced rows reading "4 predictions matched 140
        # objects", which is impossible and was believed anyway.
        if f1s:
            best_conf = thresholds[f1s.index(max(f1s))]
            counts = per_class_counts(
                [
                    ([p for p in predictions if p[2] >= best_conf], truth)
                    for predictions, truth in per_image
                ],
                threshold=args.iou,
            )

            print(f"\n  per class, at conf {best_conf:.3f}")
            print("    found  = truth objects covered by a prediction of ANY name")
            print("    named  = ...of those, the ones the model also called correctly")
            print("    fires  = predictions carrying this label (its own label space,")
            print("             so it is not comparable to the columns on its left)")
            print(
                f"    {'class':<18}{'truth':>7}{'found':>7}{'recall':>8}"
                f"{'named':>7}{'correct':>9}{'fires':>7}"
            )
            silent = []
            misnamed = []
            for name in sorted(counts, key=lambda n: -counts[n].truth):
                c = counts[name]
                if not c.truth and not c.fires:
                    continue
                print(
                    f"    {name:<18}{c.truth:>7}{c.found:>7}{c.recall:>8.1%}"
                    f"{c.named:>7}{c.naming_accuracy:>9.1%}{c.fires:>7}"
                )
                # Thresholds, not zero: two ground-truth objects decide nothing,
                # and a warning drawn from them costs more than it is worth.
                if c.truth >= MIN_CLASS_INSTANCES and not c.fires:
                    silent.append(name)
                elif c.found >= MIN_CLASS_INSTANCES and c.naming_accuracy < 0.25:
                    misnamed.append(name)

            if silent:
                print(
                    f"\n  ⚠️ prompt never fires: {', '.join(silent)}\n"
                    "     Not a training problem — the text encoder is reading those\n"
                    "     words as something else. Reword in the vocabulary and rebuild\n"
                    "     embeddings."
                )
            if misnamed:
                print(
                    f"\n  ⚠️ found but misnamed: {', '.join(misnamed)}\n"
                    "     Located reliably and called something else, so the boxes are\n"
                    "     right and the labels are not. This is what fine-tuning fixes;\n"
                    "     check first whether the confusion even changes a bin."
                )

        best = max(recalls) if recalls else 0.0
        at_default = recalls[thresholds.index(0.15)] if 0.15 in thresholds else 0.0
        print()
        if best > 3 * max(at_default, 1e-9) and best > 0.2:
            print(f"  recall rises from {at_default:.1%} to {best:.1%} as the threshold drops:")
            print("  the detections EXIST and are scored too low. This is a calibration")
            print("  problem, not blindness — thresholds and filtering will help, and")
            print("  fine-tuning would mostly be fixing the scores rather than the eyes.")
        elif best < 0.1:
            print(f"  recall never exceeds {best:.1%} at any threshold: the detections do")
            print("  not exist. No threshold or prompt change fixes that; training does.")
        else:
            print(f"  recall peaks at {best:.1%}. Partial detection — worth looking at the")
            print("  diagnostic images before choosing between tuning and training.")
        return 0

    all_pairs: list[tuple[str, str]] = []
    false_positives: Counter[str] = Counter()
    missed: Counter[str] = Counter()

    for predictions, truth in per_image:
        match = match_detections(predictions, truth, threshold=args.iou)
        all_pairs.extend(match.pairs)
        false_positives.update(match.false_positives)
        missed.update(match.missed)

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
