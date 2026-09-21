"""End-to-end routing, and scoping predictions to what a stream can hold.

Both guard the same near-miss. A pool-trained model scored 42.0% routing
accuracy on WaRP against the previous model's 56.5% and looked like a
regression. It had found 38.5% of the labelled objects where the old one
found 6.3%, so end to end it routed 16.2% of the stream correctly against
3.6% — four and a half times better, reported as worse.

The cause of the remaining errors was a class WaRP cannot contain: `plastic
bag` predicted for `plastic bottle` 200 times, a third of every matched
instance, on a holdout with no film in it at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import evaluate  # noqa: E402

from recyclevision.evaluate import score_routing  # noqa: E402

NAMES = ["metal can", "plastic bottle", "plastic bag"]


class TestEndToEndRouting:
    def test_missed_objects_count_against_the_score(self, policy):
        """The whole point: unmatched objects were not routed anywhere."""
        pairs = [("metal can", "metal can")] * 5
        score = score_routing(pairs, policy, labelled=20)

        assert score.routing_accuracy == 1.0  # of what it found
        assert score.end_to_end_accuracy == 0.25  # of what was there
        assert score.match_recall == 0.25

    def test_higher_recall_can_lower_routing_accuracy_while_improving(self, policy):
        """The comparison that nearly reversed a correct decision."""
        cautious = score_routing([("metal can", "metal can")] * 6, policy, labelled=100)
        thorough = score_routing(
            [("metal can", "metal can")] * 16 + [("metal can", "toothbrush")] * 22,
            policy,
            labelled=100,
        )

        assert thorough.routing_accuracy < cautious.routing_accuracy
        assert thorough.end_to_end_accuracy > cautious.end_to_end_accuracy

    def test_without_a_denominator_the_end_to_end_figures_stay_silent(self, policy):
        """Better to omit the number than to invent a denominator for it."""
        score = score_routing([("metal can", "metal can")], policy)

        assert score.labelled == 0
        assert score.end_to_end_accuracy == 0.0
        assert "of ALL labelled" not in score.report()

    def test_the_report_names_which_denominator_each_number_uses(self, policy):
        report = score_routing([("metal can", "metal can")] * 3, policy, labelled=12).report()

        assert "3 of 12 labelled (25.0%)" in report
        assert "(of matched)" in report
        assert "25.0%" in report


class TestLabelledInstances:
    def test_counts_objects_no_prediction_ever_reached(self):
        """Ultralytics' background row holds exactly the missed detections.

        Reading only the real-class rows would count matched objects twice
        over and report the holdout as smaller than it is.
        """
        size = len(NAMES)
        matrix = [[0] * (size + 1) for _ in range(size + 1)]
        matrix[0][0] = 4  # metal can, correctly predicted
        matrix[2][1] = 7  # plastic bottle, called plastic bag
        matrix[size][1] = 9  # plastic bottle, missed entirely

        assert evaluate.labelled_instances(matrix, NAMES) == 20

    def test_spurious_predictions_are_not_labelled_objects(self):
        """A false positive inflates no denominator: nothing was there."""
        size = len(NAMES)
        matrix = [[0] * (size + 1) for _ in range(size + 1)]
        matrix[0][0] = 3
        matrix[2][size] = 50  # 50 'plastic bag' boxes on background

        assert evaluate.labelled_instances(matrix, NAMES) == 3


def _holdout(root: Path, class_indices: list[int]) -> Path:
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    for i, index in enumerate(class_indices):
        (root / "images" / "val" / f"f{i}.jpg").write_bytes(b"\xff\xd8")
        (root / "labels" / "val" / f"f{i}.txt").write_text(
            f"{index} .5 .5 .2 .2\n", encoding="utf-8"
        )
    descriptor = root / "data.yaml"
    descriptor.write_text(
        f"path: {root}\ntrain: images/train\nval: images/val\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)),
        encoding="utf-8",
    )
    return descriptor


class TestClassScoping:
    def test_present_finds_only_the_classes_the_holdout_labels(self, tmp_path):
        descriptor = _holdout(tmp_path, [0, 1, 1, 0])

        assert evaluate.classes_present(descriptor, NAMES) == ["metal can", "plastic bottle"]

    def test_present_resolves_to_indices_excluding_the_absent_class(self, tmp_path):
        """WaRP holds no film, so `plastic bag` must not survive resolution."""
        descriptor = _holdout(tmp_path, [0, 1])

        assert evaluate.resolve_classes("present", descriptor, NAMES) == [0, 1]

    def test_named_classes_resolve_in_the_order_given(self, tmp_path):
        descriptor = _holdout(tmp_path, [0])

        assert evaluate.resolve_classes("plastic bag, metal can", descriptor, NAMES) == [2, 0]

    def test_no_flag_means_no_filtering(self, tmp_path):
        descriptor = _holdout(tmp_path, [0])

        assert evaluate.resolve_classes(None, descriptor, NAMES) is None

    def test_a_misspelled_class_fails_loudly_and_lists_the_real_ones(self, tmp_path):
        """Silently dropping it would scope the run to less than was asked."""
        descriptor = _holdout(tmp_path, [0])

        with pytest.raises(SystemExit) as caught:
            evaluate.resolve_classes("plastic bottel", descriptor, NAMES)

        assert "plastic bottel" in str(caught.value)
        assert "plastic bottle" in str(caught.value)

    def test_present_on_an_unreadable_holdout_says_so(self, tmp_path):
        descriptor = tmp_path / "data.yaml"
        descriptor.write_text(
            f"path: {tmp_path}\ntrain: images/train\nval: images/val\nnames:\n  0: metal can\n",
            encoding="utf-8",
        )

        with pytest.raises(SystemExit) as caught:
            evaluate.resolve_classes("present", descriptor, NAMES)

        assert "--classes" in str(caught.value)
