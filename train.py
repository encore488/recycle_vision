"""Fine-tune a waste detector on corrected conveyor labels.

    python train.py --data datasets/conveyor/data.yaml

Fine-tuning, not training from scratch: the pretrained backbone already knows
what edges, materials and specular highlights look like, and a few hundred
conveyor frames is nowhere near enough to learn that again. What it *is*
enough for is teaching the head this project's classes, on this belt, under
this lighting.

Every run is written to runs/ with its arguments, so a number can always be
traced back to what produced it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from recyclevision.external import BOXES, EMPTY, MIXED, POLYGONS  # noqa: E402

#: Starting weights, per dataset geometry. A segmentation model cannot train
#: on a boxes-only dataset, so the dataset picks the model rather than the
#: other way round -- see `choose_model`.
DETECT_MODEL = "yolo11s.pt"
SEGMENT_MODEL = "yolo11s-seg.pt"


def resolve_device(requested: str | None) -> str:
    """Pick a device, and say why.

    Ultralytics fails with a raw CUDA traceback when asked for `device=0` on a
    machine that has no NVIDIA GPU, which is the common case on a Mac. Apple
    Silicon has a perfectly good accelerator that answers to `mps` and is worth
    an order of magnitude over CPU, so guessing well here saves a wasted run.
    """
    import torch

    if requested and requested != "auto":
        if requested.isdigit() and not torch.cuda.is_available():
            available = "mps" if torch.backends.mps.is_available() else "cpu"
            raise SystemExit(
                f"--device {requested} asks for a CUDA GPU, and this machine has none.\n"
                f"Use --device {available}"
                + (
                    " (Apple Silicon GPU — far faster than CPU)."
                    if available == "mps"
                    else " , or train on a hosted GPU: see docs/TRAINING_ON_GPU.md"
                )
            )
        return requested

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_geometry(data: Path) -> tuple[str, Path | None]:
    """What the dataset's training labels actually contain."""
    from recyclevision.external import inspect_label_geometry, label_dir_for, read_split_images_dir

    images = read_split_images_dir(data, "train")
    if images is None:
        return EMPTY, None
    labels = label_dir_for(images)
    return inspect_label_geometry(labels), labels


def choose_model(requested: str | None, geometry: str, data: Path) -> str:
    """Match the starting weights to the dataset, or explain why they cannot be.

    Ultralytics discovers this mismatch itself, but only after caching every
    label in the dataset -- minutes into a run that could never have started.
    Worse, it phrases the fix as "supply a segment dataset", which points at
    the data when the model is the thing that should change.
    """
    if geometry == MIXED:
        raise SystemExit(
            f"{data} mixes polygon and box labels, which ultralytics will not train on.\n"
            "Every label file has to be one or the other. Re-run the import or the\n"
            "pre-labelling step with --boxes-only to make them all boxes."
        )

    if requested is None:
        return SEGMENT_MODEL if geometry == POLYGONS else DETECT_MODEL

    wants_masks = "-seg" in Path(requested).stem
    if wants_masks and geometry == BOXES:
        raise SystemExit(
            f"{data} is labelled with bounding boxes, and {requested} is a segmentation\n"
            f"model — it has nothing to learn masks from.\n\n"
            f"Drop --model and train.py picks {DETECT_MODEL}, which fits this dataset.\n\n"
            "Boxes lose nothing for routing: a bin decision needs to know what an item\n"
            "is and where it is, and a box says both. Masks earn their keep on\n"
            "pre-labelling and on items that overlap on a belt — which is what your own\n"
            "footage is for."
        )
    return requested


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/conveyor/data.yaml"))
    parser.add_argument(
        "--model",
        default=None,
        help=f"starting weights; by default {SEGMENT_MODEL} for a dataset with "
        f"polygons and {DETECT_MODEL} for one with boxes",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="larger than the 640 default: conveyor items are small in frame",
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto' (default) picks CUDA, then Apple Silicon 'mps', then 'cpu'. "
        "Override with '0', 'mps' or 'cpu'.",
    )
    parser.add_argument("--name", default=None, help="run name under runs/")
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="stop when validation has not improved for this many epochs",
    )
    args = parser.parse_args(argv)

    if not args.data.is_file():
        parser.error(
            f"no dataset descriptor at {args.data}.\n"
            "Build one first:\n"
            "  python scripts/extract_frames.py belt.mp4 --out datasets/raw\n"
            "  python scripts/prelabel.py datasets/raw --out datasets/conveyor\n"
            "...then correct the labels before training on them."
        )

    geometry, labels_dir = describe_geometry(args.data)
    if geometry == EMPTY:
        parser.error(
            f"found no label files for the train split of {args.data}"
            + (f" (looked in {labels_dir})" if labels_dir else "")
            + ".\nA dataset with no labels teaches the model that every image is empty."
        )
    model_name = choose_model(args.model, geometry, args.data)

    from ultralytics import YOLO

    device = resolve_device(args.device)
    name = args.name or f"conveyor_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"

    print(f"fine-tuning {model_name} on {args.data}")
    task = "segmentation" if geometry == POLYGONS else "detection"
    chosen = "--model given" if args.model else f"a {task} model, to match"
    print(f"  labels are {geometry} — {chosen}")
    print(f"  epochs {args.epochs} · imgsz {args.imgsz} · batch {args.batch} · device {device}")
    if device == "cpu":
        print("  CPU only — this will take many hours. See docs/TRAINING_ON_GPU.md")
    elif device == "mps":
        print("  Apple Silicon GPU. Much faster than CPU, still slower than a hosted T4.")

    model = YOLO(model_name)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        # No `project=`: ultralytics already nests runs under runs/<task>/,
        # and passing one produced runs/segment/runs/<name>.
        name=name,
        exist_ok=False,
    )

    run_dir = Path(results.save_dir)
    (run_dir / "run_args.json").write_text(
        json.dumps(
            {
                "data": str(args.data),
                "model": model_name,
                "label_geometry": geometry,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "patience": args.patience,
                "started_utc": name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    weights = run_dir / "weights" / "best.pt"
    print(f"\nbest weights -> {weights}")
    print("\nNext:")
    print(f"  python scripts/evaluate.py --weights {weights} --data {args.data}")
    print(f"  cp {weights} models/best_model.pt    # the app prefers this over stock weights")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
