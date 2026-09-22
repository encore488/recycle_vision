"""Whether a training run finished, or merely stopped.

A run that dies mid-way leaves exactly what a run that converged leaves: a
`best.pt`, a `results.csv`, and a directory named after its start time.
Nothing in the filenames says which happened, so a dead run gets scored, its
numbers get quoted, and the plateau everybody then reasons about is an
artefact of the corpse.

That is not hypothetical. A pool run hit fp16 overflow on MPS at epoch 10,
produced NaN losses, and ultralytics restored `last.pt` and re-ran -- four
times, each recovery returning byte-identical validation metrics because the
epoch had produced nothing. By epoch 22 `last.pt` was itself NaN. The run was
dead 18 epochs short of its budget, and its `best.pt` was evaluated a day
later as though it were the finished article.

So the signals below are the ones that failure actually emits:

* **NaN in any row.** The run is over at that epoch whatever follows.
* **Repeated byte-identical validation rows.** Ultralytics' NaN recovery
  re-runs an epoch and reports the previous metrics again; three or more in a
  row are epochs that did no work.
* **Fewer epochs than were asked for**, without early stopping accounting for
  it.

None of these needs torch, a model, or the dataset -- just the two files
every run already writes.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

#: Consecutive identical validation rows before it stops being coincidence.
#: Two epochs can legitimately score alike; a third repeat to every decimal
#: place is the recovery loop, not the model.
STALLED_RUN_OF = 3


def _is_nan(value: str) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


@dataclass
class RunHealth:
    """What a run's own logs say about how it ended."""

    #: Epochs with a row in results.csv, NaN ones included.
    epochs_completed: int = 0
    #: Epochs asked for in run_args.json, when it exists.
    epochs_requested: int | None = None
    #: First epoch whose metrics went NaN, if any.
    nan_at: int | None = None
    #: Runs of epochs that re-reported the previous epoch's validation metrics.
    stalled_epochs: list[int] = field(default_factory=list)
    #: Early-stopping budget, which legitimately explains a short run.
    patience: int | None = None

    @property
    def died(self) -> bool:
        return self.nan_at is not None

    @property
    def truncated(self) -> bool:
        """Stopped short of its budget by more than early stopping explains."""
        if self.epochs_requested is None:
            return False
        if self.epochs_completed >= self.epochs_requested:
            return False
        if self.died:
            return True
        # Early stopping is a legitimate short finish, and it needs `patience`
        # epochs of no improvement before it fires.
        return self.patience is None or self.epochs_completed < self.patience

    @property
    def healthy(self) -> bool:
        return not (self.died or self.truncated or self.stalled_epochs)

    def report(self) -> str:
        """A warning when something is wrong, and nothing when it is not."""
        if self.healthy:
            return ""

        budget = f" of {self.epochs_requested}" if self.epochs_requested is not None else ""
        lines = [
            f"⚠️ this run ran {self.epochs_completed} epoch(s){budget}, and did not finish clean:"
        ]

        if self.died:
            lines.append(
                f"   · losses went NaN at epoch {self.nan_at}. Everything after it is "
                "garbage,\n     and best.pt was chosen from a run that was already failing."
            )
        if self.stalled_epochs:
            shown = ", ".join(str(e) for e in self.stalled_epochs[:8])
            more = "…" if len(self.stalled_epochs) > 8 else ""
            lines.append(
                f"   · epoch(s) {shown}{more} re-reported the previous epoch's validation\n"
                "     metrics exactly — ultralytics restoring last.pt and re-running. "
                "Those\n     epochs did no work."
            )
        if self.truncated and not self.died:
            lines.append(
                "   · it stopped short of its epoch budget, and early stopping does not\n"
                "     account for the gap."
            )

        lines.append(
            "   On Apple Silicon this is usually fp16 overflow: train.py's --amp\n"
            "   defaults to auto, which turns mixed precision off on mps. A run\n"
            "   started before that default cannot benefit from it — re-run it.\n"
            "   Scores below describe THIS run, not the approach."
        )
        return "\n".join(lines)


def _read_args(run_dir: Path) -> tuple[int | None, int | None]:
    """Epoch budget and patience from run_args.json, when train.py wrote one."""
    path = run_dir / "run_args.json"
    if not path.is_file():
        return None, None
    try:
        args = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    epochs = args.get("epochs")
    patience = args.get("patience")
    return (
        epochs if isinstance(epochs, int) else None,
        patience if isinstance(patience, int) else None,
    )


def inspect_run(run_dir: Path) -> RunHealth | None:
    """Read a run's own logs. None when there are none to read."""
    results = run_dir / "results.csv"
    if not results.is_file():
        return None

    try:
        with results.open(encoding="utf-8", newline="") as handle:
            rows = [
                {(key or "").strip(): (value or "").strip() for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    if not rows:
        return None

    epochs, patience = _read_args(run_dir)
    health = RunHealth(
        epochs_completed=len(rows),
        epochs_requested=epochs,
        patience=patience,
    )

    # Validation columns only. Training losses differ every epoch even when a
    # recovery discards the work, because the data order and augmentation
    # still change -- it is the *validation* metrics that repeat.
    val_columns = [key for key in rows[0] if key.startswith("val/") or key.startswith("metrics/")]

    previous: tuple[str, ...] | None = None
    streak = 0
    for index, row in enumerate(rows, start=1):
        if health.nan_at is None and any(_is_nan(value) for value in row.values()):
            health.nan_at = index

        if not val_columns:
            continue
        current = tuple(row.get(key, "") for key in val_columns)
        if not any(current):
            previous = None
            streak = 0
            continue
        if current == previous:
            streak += 1
            if streak >= STALLED_RUN_OF - 1:
                health.stalled_epochs.append(index)
        else:
            streak = 0
        previous = current

    return health


def run_dir_of(weights: Path) -> Path | None:
    """The run directory a best.pt came from: runs/<task>/<name>/weights/best.pt."""
    parent = weights.parent
    if parent.name != "weights":
        return None
    return parent.parent


class StallDetector:
    """Halt a run that is re-running the same epoch instead of progressing.

    Ultralytics recovers from a NaN epoch by restoring `last.pt` and re-running
    it, and allows three attempts. But the counter resets on every *successful*
    recovery, so a run that NaNs on alternate epochs never exhausts it: it
    alternates failure and restore indefinitely, reporting identical validation
    metrics each time.

    A run of this project spent ten epochs and roughly four hours that way,
    reporting `0.555 0.514 0.53 0.403` six times over. Nothing in the output
    says "this is not working" -- the progress bars complete, the metrics
    print, and the numbers are plausible.

    So: identical fitness for `limit` consecutive epochs means the weights have
    not moved. Stop and say so, rather than spending the night proving it.
    """

    def __init__(self, limit: int = 3) -> None:
        self.limit = limit
        self.previous: float | None = None
        self.repeats = 0

    def observe(self, fitness: float | None) -> bool:
        """Record one epoch's fitness. True when the run should stop.

        A NaN epoch is not itself a stall -- ultralytics is entitled to recover
        from one -- so it breaks the streak rather than counting toward it, and
        needs no special case to do so: NaN compares unequal to everything
        including itself, so the equality test below resets on it. An explicit
        guard here was removed because no test could make it fail, and a branch
        that cannot fail is not a guard.

        What counts is the *recovered* epoch landing on the same weights again.
        """
        if self.previous is not None and fitness is not None and fitness == self.previous:
            self.repeats += 1
        else:
            self.repeats = 0
        self.previous = fitness
        return self.repeats >= self.limit - 1

    def message(self) -> str:
        return (
            f"\nSTOPPING: {self.repeats + 1} consecutive epochs scored identically "
            f"({self.previous}).\nThe weights are not moving — ultralytics is restoring "
            "last.pt after a NaN and\nre-running the same epoch. Its retry counter resets "
            "on each recovery, so it\nwill not stop on its own.\n\n"
            "  python scripts/check_labels.py <this run's data.yaml>\n\n"
            "A box with zero width or height divides by zero in the loss and NaNs every\n"
            "epoch that touches it."
        )
