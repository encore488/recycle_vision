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
from recyclevision.external import MappingError, read_yolo_data_yaml  # noqa: E402
from recyclevision.pipeline import DEFAULT_POLICY  # noqa: E402
from recyclevision.policy import RoutingPolicy  # noqa: E402
from recyclevision.runs import inspect_run, run_dir_of  # noqa: E402
from recyclevision.weights import latest_trained_weights, shadowed_by  # noqa: E402

#: Below this many labelled objects per image, a dataset's own annotation
#: policy dominates precision. SortWaste sits at 16.6, ZeroWaste 5.9, WaRP 3.5.
SPARSE_BELOW = 5.0


def annotation_density(data_yaml: Path) -> float | None:
    """Labelled objects per image in a dataset's val split, or None."""
    try:
        root, _names = read_yolo_data_yaml(data_yaml)
    except MappingError:
        return None
    labels = root / "labels" / "val"
    if not labels.is_dir():
        return None
    files = list(labels.rglob("*.txt"))
    if not files:
        return None
    instances = 0
    for path in files:
        try:
            instances += sum(
                1
                for line in path.read_text(encoding="utf-8").splitlines()
                if len(line.split()) >= 5
            )
        except (OSError, UnicodeDecodeError):
            continue
    return instances / len(files)


def labelled_instances(matrix, names: list[str]) -> int:
    """Ground-truth instances in the holdout, including every one missed.

    Ultralytics' matrix is [predicted][actual] with a trailing background row
    and column. Summing a real class's *column* over every row -- background
    included -- counts every labelled object of that class, whether a
    prediction landed on it or not. That denominator is what turns routing
    accuracy into a figure comparable across models.
    """
    size = len(names)
    return sum(
        int(matrix[predicted][actual]) for actual in range(size) for predicted in range(size + 1)
    )


def classes_present(data_yaml: Path, names: list[str]) -> list[str]:
    """Class names the holdout's val labels actually contain.

    A model can only be wrong about a class the stream can hold. WaRP has no
    film at all, yet a pool-trained model predicted `plastic bag` for
    `plastic bottle` 200 times -- a third of every instance it matched. The
    same mechanism was measured on prompts earlier in this project
    (docs/history/), where scoping the vocabulary to what the line can contain
    took precision from 5.1% to 50%.

    Deriving the list from the labels rather than asking for it typed keeps
    the experiment one flag wide.
    """
    try:
        root, _names = read_yolo_data_yaml(data_yaml)
    except MappingError:
        return []
    labels = root / "labels" / "val"
    if not labels.is_dir():
        return []
    seen: set[int] = set()
    for path in labels.rglob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for line in text.splitlines():
            fields = line.split()
            if len(fields) < 5:
                continue
            try:
                seen.add(int(fields[0]))
            except ValueError:
                continue
    return [names[i] for i in sorted(seen) if 0 <= i < len(names)]


def resolve_classes(requested: str | None, data: Path, names: list[str]) -> list[int] | None:
    """Turn --classes into indices ultralytics can filter on, or explain why not."""
    if requested is None:
        return None
    if requested == "present":
        wanted = classes_present(data, names)
        if not wanted:
            raise SystemExit(
                f"--classes present could not read any labels under {data}.\n"
                "Name the classes instead: --classes 'plastic bottle,metal can'"
            )
    else:
        wanted = [part.strip() for part in requested.split(",") if part.strip()]

    unknown = [name for name in wanted if name not in names]
    if unknown:
        raise SystemExit(
            "--classes names classes this model does not have: "
            + ", ".join(repr(name) for name in unknown)
            + "\nIt knows: "
            + ", ".join(names)
        )
    return [names.index(name) for name in wanted]


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
    parser.add_argument(
        "--classes",
        default=None,
        help="restrict predictions to these class names, comma separated, or "
        "'present' to use whatever the holdout's own labels contain. A class "
        "the stream cannot hold is a pure false-positive source: scoping is "
        "free accuracy and needs no retraining.",
    )
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
        print(f"using {args.weights}  (newest of {len(shadowed_by(args.weights)) + 1})")
        for other in shadowed_by(args.weights)[:3]:
            print(f"  not: {other}")

    for path in (args.weights, args.data):
        if not path.is_file():
            parser.error(f"no such file: {path}")

    # A run that died leaves exactly what a run that converged leaves. Say so
    # before the numbers appear, not after they have been quoted.
    run_dir = run_dir_of(args.weights)
    health = inspect_run(run_dir) if run_dir else None
    if health is not None and not health.healthy:
        print()
        print(health.report())

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    names = list(model.names.values())
    keep = resolve_classes(args.classes, args.data, names)
    if keep is not None:
        print(
            f"\nscoped to {len(keep)} of {len(names)} class(es): "
            + ", ".join(names[i] for i in keep)
            + "\n  the rest cannot appear in this holdout, so any prediction of one"
            "\n  could only ever be wrong."
        )

    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        device=args.device,
        classes=keep,
        verbose=False,
    )

    # How completely the holdout is annotated decides whether its precision
    # means anything. Measured on WaRP earlier in this project: 3.5 labelled
    # objects per image in frames holding dozens, and true precision provably
    # somewhere between a raw 4.1% and a fair 56.0%. A model scored there can
    # be penalised for finding objects the dataset simply never labelled.
    density = annotation_density(args.data)
    if density is not None and density < SPARSE_BELOW:
        print(
            f"\n⚠️ this holdout labels {density:.1f} object(s) per image.\n"
            "   Correct detections of unlabelled objects score as false positives,\n"
            "   so precision and mAP below are LOWER BOUNDS, not measurements.\n"
            "   Read recall and routing accuracy; treat precision as uninterpretable."
        )

    print("\ndetection metrics")
    print(f"  mAP50      {metrics.box.map50:.3f}")
    print(f"  mAP50-95   {metrics.box.map:.3f}")
    print(f"  precision  {metrics.box.mp:.3f}")
    print(f"  recall     {metrics.box.mr:.3f}")

    pairs = _pairs_from_confusion(metrics.confusion_matrix.matrix, names)
    if not pairs:
        print("\nno matched instances — nothing to score for routing")
        return 0

    policy = RoutingPolicy.load(args.policy)
    labelled = labelled_instances(metrics.confusion_matrix.matrix, names)
    print(f"\nrouting metrics ({policy.name})")
    print(score_routing(pairs, policy, labelled=labelled).report())

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
