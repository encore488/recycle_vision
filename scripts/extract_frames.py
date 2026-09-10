"""Sample diverse frames from conveyor footage.

    python scripts/extract_frames.py belt.mp4 --out datasets/raw --max 400

Consecutive video frames are near-identical. Labelling both costs twice as
much and teaches the model nothing, so frames too similar to the last one kept
are dropped. Getting this wrong is not merely wasteful: near-duplicates spread
across a train/val split make validation scores meaningless.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from recyclevision.dataset import DEFAULT_MIN_DIFF, is_novel, thumbnail_of  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--every", type=int, default=10, help="consider every Nth frame")
    parser.add_argument("--max", type=int, default=400, help="stop after this many kept frames")
    parser.add_argument(
        "--min-diff",
        type=float,
        default=DEFAULT_MIN_DIFF,
        help="drop frames closer than this to the last kept one (0-255, 0 keeps everything)",
    )
    args = parser.parse_args(argv)

    if not args.video.is_file():
        parser.error(f"no such video: {args.video}")

    import cv2

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        parser.error(f"could not open {args.video}")

    args.out.mkdir(parents=True, exist_ok=True)

    considered = kept = index = 0
    last_signature = None

    while kept < args.max:
        ok, frame = capture.read()
        if not ok:
            break
        index += 1
        if index % args.every:
            continue

        considered += 1
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        signature = thumbnail_of(image)
        if not is_novel(signature, last_signature, args.min_diff):
            continue

        last_signature = signature
        kept += 1
        image.save(args.out / f"{args.video.stem}_{index:06d}.jpg", quality=92)

    capture.release()

    dropped = considered - kept
    print(f"read {index} frame(s), considered {considered}, kept {kept} -> {args.out}")
    if dropped:
        print(f"dropped {dropped} near-duplicate(s) — that is {dropped} fewer to label")
    if kept == args.max:
        print("hit --max; raise it if you want more of the footage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
