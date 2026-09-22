"""Telling a run that finished from a run that merely stopped.

Both leave a best.pt, a results.csv and a timestamped directory. A pool run
died of fp16 overflow on MPS at epoch 22 of 40 — after four epochs that
re-reported the previous epoch's metrics because ultralytics kept restoring
last.pt — and was evaluated a day later as though it had converged. The
plateau everybody then reasoned about was an artefact of the corpse.
"""

from __future__ import annotations

import json
from pathlib import Path

from recyclevision.runs import StallDetector, inspect_run, run_dir_of

HEADER = "epoch,train/box_loss,train/cls_loss,val/box_loss,val/cls_loss\n"


def make_run(root: Path, rows: list[str], args: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "results.csv").write_text(HEADER + "".join(r + "\n" for r in rows), encoding="utf-8")
    if args is not None:
        (root / "run_args.json").write_text(json.dumps(args), encoding="utf-8")
    return root


def clean_rows(count: int) -> list[str]:
    """Losses that fall every epoch, as a healthy run's do."""
    return [
        f"{i},{1.0 - i * 0.01},{2.0 - i * 0.02},{0.9 - i * 0.005},{1.5 - i * 0.01}"
        for i in range(1, count + 1)
    ]


class TestHealthyRuns:
    def test_a_run_that_used_its_whole_budget_says_nothing(self, tmp_path):
        health = inspect_run(make_run(tmp_path / "r", clean_rows(40), {"epochs": 40}))

        assert health.healthy
        assert health.report() == ""

    def test_early_stopping_is_not_a_failure(self, tmp_path):
        """Patience firing is the run working, not the run dying."""
        health = inspect_run(
            make_run(tmp_path / "r", clean_rows(30), {"epochs": 100, "patience": 25})
        )

        assert not health.truncated
        assert health.healthy

    def test_two_alike_epochs_are_coincidence_not_a_stall(self, tmp_path):
        rows = clean_rows(5) + ["6,0.5,0.5,0.777,1.111", "7,0.4,0.4,0.777,1.111"]
        health = inspect_run(make_run(tmp_path / "r", rows, {"epochs": 7}))

        assert health.stalled_epochs == []
        assert health.healthy

    def test_no_results_file_is_not_a_verdict(self, tmp_path):
        (tmp_path / "empty").mkdir()

        assert inspect_run(tmp_path / "empty") is None


class TestDeadRuns:
    def test_nan_losses_name_the_epoch_they_started(self, tmp_path):
        rows = clean_rows(21) + ["22,0.70545,0.53765,nan,nan"]
        health = inspect_run(make_run(tmp_path / "r", rows, {"epochs": 40, "patience": 25}))

        assert health.died
        assert health.nan_at == 22
        assert health.truncated
        assert "NaN at epoch 22" in health.report()

    def test_repeated_validation_metrics_are_epochs_that_did_no_work(self, tmp_path):
        """The recovery signature: ultralytics re-runs and reports the same.

        Training losses still move, because the data order does. Only the
        validation metrics repeat, which is why they are what gets compared.
        """
        rows = clean_rows(9) + [
            "10,0.78179,0.70035,0.86872,1.36035",
            "11,0.78556,0.70443,0.86872,1.36035",
            "12,0.77755,0.69404,0.86872,1.36035",
            "13,0.78277,0.70264,0.86872,1.36035",
        ]
        health = inspect_run(make_run(tmp_path / "r", rows, {"epochs": 40}))

        assert health.stalled_epochs == [12, 13]
        assert not health.healthy
        assert "did no work" in health.report()

    def test_moving_training_loss_does_not_hide_a_stall(self, tmp_path):
        """Reading train columns too would mask every recovery."""
        rows = clean_rows(3) + [
            "4,0.11,0.22,0.5,1.0",
            "5,0.12,0.23,0.5,1.0",
            "6,0.13,0.24,0.5,1.0",
        ]
        health = inspect_run(make_run(tmp_path / "r", rows, {"epochs": 40}))

        assert health.stalled_epochs == [6]

    def test_a_short_run_with_no_patience_to_explain_it_is_truncated(self, tmp_path):
        health = inspect_run(make_run(tmp_path / "r", clean_rows(5), {"epochs": 40}))

        assert health.truncated
        assert "stopped short" in health.report()

    def test_without_run_args_no_budget_can_be_claimed(self, tmp_path):
        """train.py writes run_args.json; a run from elsewhere may not."""
        health = inspect_run(make_run(tmp_path / "r", clean_rows(5)))

        assert health.epochs_requested is None
        assert not health.truncated

    def test_dying_on_the_last_epoch_is_still_dying(self, tmp_path):
        """A NaN run that used its whole budget is not truncated, and is dead.

        Without this the NaN check is redundant: every dead run so far was
        also short, so `truncated` alone would have carried both tests.
        """
        rows = clean_rows(9) + ["10,0.7,0.7,nan,nan"]
        health = inspect_run(make_run(tmp_path / "r", rows, {"epochs": 10}))

        assert not health.truncated
        assert health.died
        assert not health.healthy
        assert "NaN at epoch 10" in health.report()

    def test_the_report_says_the_scores_describe_this_run(self, tmp_path):
        rows = clean_rows(9) + ["10,0.7,0.7,nan,nan"]
        report = inspect_run(make_run(tmp_path / "r", rows, {"epochs": 40})).report()

        assert "not the approach" in report
        assert "--amp" in report


class TestRunDirOf:
    def test_finds_the_run_a_best_pt_came_from(self):
        weights = Path("runs/detect/conveyor_20260921_023058/weights/best.pt")

        assert run_dir_of(weights) == Path("runs/detect/conveyor_20260921_023058")

    def test_weights_kept_anywhere_else_have_no_run(self):
        assert run_dir_of(Path("models/best_model.pt")) is None


class TestStallDetector:
    def test_improving_fitness_never_stops(self):
        stall = StallDetector(limit=3)

        assert [stall.observe(f) for f in (0.30, 0.35, 0.40, 0.42)] == [False] * 4

    def test_three_identical_epochs_stop_the_run(self):
        """The observed failure: 0.403 reported six times in a row."""
        stall = StallDetector(limit=3)

        assert stall.observe(0.403) is False
        assert stall.observe(0.403) is False
        assert stall.observe(0.403) is True

    def test_two_identical_epochs_are_tolerated(self):
        """Fitness can legitimately repeat once; three times it has not moved."""
        stall = StallDetector(limit=3)
        stall.observe(0.403)

        assert stall.observe(0.403) is False

    def test_a_nan_epoch_resets_rather_than_counting(self):
        """Ultralytics is entitled to recover from one NaN.

        Counting the NaN itself would stop a run that then recovers properly.
        What matters is the recovered epoch landing on the same weights.
        """
        stall = StallDetector(limit=3)
        stall.observe(0.403)
        stall.observe(float("nan"))

        assert stall.observe(0.403) is False

    def test_a_missing_fitness_does_not_count(self):
        stall = StallDetector(limit=3)
        stall.observe(0.403)
        stall.observe(None)
        stall.observe(0.403)

        assert stall.observe(0.403) is False

    def test_the_message_points_at_the_label_checker(self):
        stall = StallDetector(limit=2)
        stall.observe(0.403)
        stall.observe(0.403)

        assert "check_labels.py" in stall.message()
        assert "will not stop on its own" in stall.message()
