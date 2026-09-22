"""Find labels that will make a training run produce NaN instead of a model.

    python scripts/check_labels.py datasets/pool/data.yaml

Reads the label files only -- no torch, no images, no GPU -- so it answers in
seconds a question that otherwise costs a night of training to ask.

The defect it exists for is a box with zero width or height. The loss divides
by box area, so one such label NaNs the batch it lands in, and ultralytics
responds by restoring `last.pt` and re-running the epoch, which hits the same
label again. Its retry counter resets on every successful recovery, so the run
never exhausts it: it reports byte-identical metrics for hours and looks
exactly like a model that has converged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recyclevision.external import MappingError, read_yolo_data_yaml  # noqa: E402
from recyclevision.labels import scan_labels  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="a dataset's data.yaml")
    parser.add_argument(
        "--splits",
        default="train,val",
        help="which splits to scan (default: train,val)",
    )
    args = parser.parse_args(argv)

    if not args.data.is_file():
        parser.error(f"no dataset descriptor at {args.data}")

    try:
        root, names = read_yolo_data_yaml(args.data)
    except MappingError as error:
        parser.error(str(error))

    fatal = 0
    for split in (s.strip() for s in args.splits.split(",") if s.strip()):
        labels = root / "labels" / split
        print(f"\n{split}  ({labels})")
        if not labels.is_dir():
            print("  no such directory")
            continue
        scan = scan_labels(labels, classes=len(names))
        print("  " + scan.report().replace("\n", "\n  "))
        fatal += len(scan.fatal)

    if fatal:
        print(
            f"\n{fatal} label(s) will produce NaN. Re-import the affected source with a\n"
            "current checkout — box_line now drops a box too small to have an area,\n"
            "the same way polygon_line already dropped a polygon too small to be a shape."
        )
        return 1

    print("\nNo NaN-producing labels. If a run still NaNs, the cause is not the data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
