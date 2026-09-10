"""Score a trained model on classes and on destinations.

    python scripts/evaluate.py --weights runs/.../best.pt --data datasets/conveyor/data.yaml

Reports ultralytics' own detection metrics, and then the number this project
actually cares about: how often a prediction reaches the right bin. Those come
apart. Confusing an aluminium can for a steel one is a class error worth
nothing, because both route the same way; confusing a drinking glass for a jar
contaminates a batch. A model that trades the first for the second looks better
on mAP and is worse in practice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recyclevision.evaluate import score_routing  # noqa: E402
from recyclevision.pipeline import DEFAULT_POLICY  # noqa: E402
from recyclevision.policy import RoutingPolicy  # noqa: E402


def _pairs_from_confusion(matrix, names: list[str]) -> list[tuple[str, str]]:
    """Expand a confusion matrix into (predicted, actual) pairs.

    Ultralytics' matrix carries a trailing background row/column for missed
    and spurious detections. Those are recall and precision problems, not
    routing ones, so they are excluded here rather than silently counted as
    class errors.
    """
    pairs: list[tuple[str, str]] = []
    size = len(names)
    for predicted in range(size):
        for actual in range(size):
            count = int(matrix[predicted][actual])
            pairs.extend([(names[predicted], names[actual])] * count)
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    for path in (args.weights, args.data):
        if not path.is_file():
            parser.error(f"no such file: {path}")

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    metrics = model.val(data=str(args.data), imgsz=args.imgsz, device=args.device, verbose=False)

    print("\ndetection metrics")
    print(f"  mAP50      {metrics.box.map50:.3f}")
    print(f"  mAP50-95   {metrics.box.map:.3f}")
    print(f"  precision  {metrics.box.mp:.3f}")
    print(f"  recall     {metrics.box.mr:.3f}")

    names = list(model.names.values())
    pairs = _pairs_from_confusion(metrics.confusion_matrix.matrix, names)
    if not pairs:
        print("\nno matched instances — nothing to score for routing")
        return 0

    print("\nrouting metrics")
    print(score_routing(pairs, RoutingPolicy.load(args.policy)).report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
