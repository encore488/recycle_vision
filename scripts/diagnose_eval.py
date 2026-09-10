"""Draw ground truth against predictions, so a bad score can be looked at.

    python scripts/diagnose_eval.py --data datasets/warp/data.yaml --out qa/diagnose

A precision of 4% has two very different explanations, and no metric can tell
them apart:

- the model is genuinely failing on this domain, or
- the harness is comparing the wrong things -- wrong scale, wrong file pairing,
  boxes in the wrong coordinate space.

Both produce the same number. Only a picture separates them. Green is ground
truth, red is prediction; if the greens sit on objects and the reds sit
somewhere else, the model is at fault, and if the greens are in the wrong place
entirely, the harness is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from recyclevision.detector import DEFAULT_IMGSZ  # noqa: E402
from recyclevision.evaluate import DEFAULT_IOU, match_detections  # noqa: E402
from recyclevision.external import MappingError, read_yolo_data_yaml  # noqa: E402
from recyclevision.render import _load_font  # noqa: E402
from recyclevision.vocabulary import DEFAULT_VOCAB, Vocabulary  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zeroshot_eval import IMAGE_SUFFIXES, read_truth  # noqa: E402

TRUTH_COLOUR = (40, 220, 90)
PREDICTION_COLOUR = (240, 60, 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("qa/diagnose"))
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--count", type=int, default=6, help="images to draw")
    args = parser.parse_args(argv)

    try:
        root, names = read_yolo_data_yaml(args.data)
    except MappingError as exc:
        parser.error(str(exc))

    images_root = root / "images" / args.split
    labels_root = root / "labels" / args.split
    images = sorted(p for p in images_root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        parser.error(f"no images in {images_root}")

    # Spread the sample across the split rather than taking the first N, which
    # in a sorted listing often means one scene shot from one angle.
    step = max(1, len(images) // args.count)
    sample = images[::step][: args.count]

    from recyclevision.detector import OpenVocabularyDetector

    detector = OpenVocabularyDetector(Vocabulary.load(args.vocabulary), imgsz=args.imgsz)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"drawing {len(sample)} image(s) at imgsz {args.imgsz}, conf {args.confidence}")
    print("  green = ground truth, red = prediction\n")

    for image_path in sample:
        image = Image.open(image_path).convert("RGB")
        truth = read_truth(labels_root / f"{image_path.stem}.txt", names, image.width, image.height)
        predictions = [
            (d.label, d.box, d.confidence)
            for d in detector.detect(image, confidence=args.confidence)
        ]
        match = match_detections(predictions, truth, threshold=args.iou)

        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        font = _load_font(canvas.width)
        width = max(2, canvas.width // 500)

        for label, box in truth:
            draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=TRUTH_COLOUR, width=width)
            draw.text((box.x1 + 3, box.y1 + 3), label, fill=TRUTH_COLOUR, font=font)

        for label, box, confidence in predictions:
            draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=PREDICTION_COLOUR, width=width)
            draw.text(
                (box.x1 + 3, max(0, box.y1 - 18)),
                f"{label} {confidence:.0%}",
                fill=PREDICTION_COLOUR,
                font=font,
            )

        out = args.out / f"{image_path.stem}_diagnose.jpg"
        canvas.save(out, quality=88)
        print(
            f"  {image_path.name}: {len(truth)} truth, {len(predictions)} predicted, "
            f"{len(match.pairs)} matched -> {out.name}"
        )

    print(f"\nwritten to {args.out}")
    print("\nWhat to look for:")
    print("  greens on objects, reds elsewhere  -> the model is failing on this domain")
    print("  greens NOT on objects              -> the harness is reading labels wrongly")
    print("  reds roughly right but not matched -> raise --imgsz or lower --iou")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
