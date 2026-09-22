"""Check a dataset before spending a night training on it.

    python scripts/preflight.py datasets/pool/data.yaml

Reads label files and directory listings only -- no torch, no GPU, seconds.
Every check here corresponds to a failure this project actually had, or to one
that is cheap to cause and expensive to notice.

Exit status is 1 when something blocking is found, so it can gate a run:

    python scripts/preflight.py datasets/pool/data.yaml \
        && python train.py --data datasets/pool/data.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recyclevision.external import MappingError, read_yolo_data_yaml  # noqa: E402
from recyclevision.preflight import check_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="a dataset's data.yaml")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="the resolution you will train at; box sizes are judged in pixels at it",
    )
    args = parser.parse_args(argv)

    if not args.data.is_file():
        parser.error(f"no dataset descriptor at {args.data}")

    try:
        root, names = read_yolo_data_yaml(args.data)
    except MappingError as error:
        parser.error(str(error))

    print(f"{args.data}\n  root {root}\n  {len(names)} declared class(es), imgsz {args.imgsz}\n")
    result = check_dataset(root, names, imgsz=args.imgsz)
    print(result.report())
    return 1 if result.blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
