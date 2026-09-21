"""Score a trained model on classes and on destinations.

    python scripts/evaluate.py --data datasets/conveyor/data.yaml

Weights default to models/best_model.pt, or the newest run's best.pt, so a
timestamped run directory never has to be typed.

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

from recyclevision.evaluate import destinations_reachable, score_routing  # noqa: E402
from recyclevision.pipeline import DEFAULT_POLICY  # noqa: E402
from recyclevision.policy import RoutingPolicy  # noqa: E402
from recyclevision.weights import latest_trained_weights  # noqa: E402


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
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="defaults to models/best_model.pt, or the most recent run's best.pt",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    if args.weights is None:
        args.weights = latest_trained_weights()
        if args.weights is None:
            parser.error(
                "no weights given and none found.\n"
                "Train one first, or pass --weights explicitly:\n"
                "  python train.py --data <a dataset>/data.yaml"
            )
        print(f"using {args.weights}")

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

    policy = RoutingPolicy.load(args.policy)
    print(f"\nrouting metrics ({policy.name})")
    print(score_routing(pairs, policy).report())

    # Routing accuracy is only a measurement when the classes under test can
    # reach more than one bin. They frequently cannot: an external dataset is
    # usually one material family, and a household policy sends the whole
    # family to one place. The metric then reads 100% without the model having
    # been tested at all, which is worse than not reporting it.
    involved = {label for pair in pairs for label in pair}
    reachable = destinations_reachable(involved, policy)
    if len(set(reachable.values())) <= 1:
        only = next(iter(reachable.values()), "one bin")
        print(
            f"\n  ⚠️ every class here routes to {only!r} under {policy.name}, so routing\n"
            "     accuracy cannot fail and this 100% measures nothing. Score against a\n"
            "     policy that separates them before quoting it:\n"
            "       --policy policies/mrf_conveyor.yaml"
        )
    else:
        print(
            f"\n  {len(set(reachable.values()))} destinations in play across "
            f"{len(reachable)} class(es), so this number can fail."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
