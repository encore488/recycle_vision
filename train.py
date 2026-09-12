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

DEFAULT_MODEL = "yolo11s-seg.pt"


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/conveyor/data.yaml"))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="starting weights")
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

    from ultralytics import YOLO

    device = resolve_device(args.device)
    name = args.name or f"conveyor_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"

    print(f"fine-tuning {args.model} on {args.data}")
    print(f"  epochs {args.epochs} · imgsz {args.imgsz} · batch {args.batch} · device {device}")
    if device == "cpu":
        print("  CPU only — this will take many hours. See docs/TRAINING_ON_GPU.md")
    elif device == "mps":
        print("  Apple Silicon GPU. Much faster than CPU, still slower than a hosted T4.")

    model = YOLO(args.model)
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
                "model": args.model,
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
