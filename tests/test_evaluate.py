"""Routing-aware scoring: class errors that matter, and those that do not."""

from __future__ import annotations

from recyclevision.evaluate import per_class_counts, score_routing
from recyclevision.models import BoundingBox


def box(x1, y1, x2, y2) -> BoundingBox:
    return BoundingBox(x1, y1, x2, y2)


class TestScoreRouting:
    def test_exact_matches_score_on_both(self, policy):
        score = score_routing([("metal can", "metal can")] * 3, policy)
        assert score.class_accuracy == 1.0
        assert score.routing_accuracy == 1.0
        assert score.forgiven == 0

    def test_a_class_error_within_one_bin_is_forgiven(self, policy):
        """The whole reason routing is scored separately.

        A metal can read as a plastic bottle is wrong, but both go in mixed
        recycling, so the destination — the thing the system exists to
        produce — is still right.
        """
        score = score_routing([("metal can", "plastic bottle")], policy)
        assert score.class_accuracy == 0.0
        assert score.routing_accuracy == 1.0
        assert score.forgiven == 1
        assert score.harmless_confusions[("metal can", "plastic bottle")] == 1
        assert not score.harmful_confusions

    def test_a_class_error_across_bins_is_not_forgiven(self, policy):
        score = score_routing([("glass jar", "empty drinking glass")], policy)
        assert score.routing_accuracy == 0.0
        assert score.harmful_confusions[("glass jar", "empty drinking glass")] == 1
        assert not score.harmless_confusions

    def test_routing_accuracy_is_never_below_class_accuracy(self, policy):
        pairs = [
            ("metal can", "metal can"),
            ("metal can", "plastic bottle"),
            ("glass jar", "empty drinking glass"),
            ("sheet of paper", "cardboard box"),
        ]
        score = score_routing(pairs, policy)
        assert score.routing_accuracy >= score.class_accuracy

    def test_prediction_of_a_class_the_policy_ignores_is_harmful(self, policy):
        """An unroutable prediction produces no bin, so it cannot be right."""
        score = score_routing([("person", "metal can")], policy)
        assert score.routing_accuracy == 0.0
        assert score.harmful_confusions

    def test_no_pairs_scores_zero_rather_than_dividing(self, policy):
        score = score_routing([], policy)
        assert score.matched == 0
        assert score.class_accuracy == 0.0
        assert score.routing_accuracy == 0.0

    def test_report_separates_harmful_from_harmless(self, policy):
        pairs = [("metal can", "plastic bottle"), ("glass jar", "empty drinking glass")]
        report = score_routing(pairs, policy).report()
        assert "confusions that changed the bin" in report
        assert "confusions that did not matter" in report


class TestIou:
    def test_identical_boxes(self):
        from recyclevision.evaluate import iou
        from recyclevision.models import BoundingBox

        assert iou(BoundingBox(0, 0, 10, 10), BoundingBox(0, 0, 10, 10)) == 1.0

    def test_disjoint_boxes(self):
        from recyclevision.evaluate import iou
        from recyclevision.models import BoundingBox

        assert iou(BoundingBox(0, 0, 10, 10), BoundingBox(20, 20, 30, 30)) == 0.0

    def test_touching_edges_do_not_overlap(self):
        from recyclevision.evaluate import iou
        from recyclevision.models import BoundingBox

        assert iou(BoundingBox(0, 0, 10, 10), BoundingBox(10, 0, 20, 10)) == 0.0

    def test_partial_overlap(self):
        import pytest

        from recyclevision.evaluate import iou
        from recyclevision.models import BoundingBox

        assert iou(BoundingBox(0, 0, 10, 10), BoundingBox(5, 0, 15, 10)) == pytest.approx(1 / 3)

    def test_degenerate_boxes_do_not_divide_by_zero(self):
        from recyclevision.evaluate import iou
        from recyclevision.models import BoundingBox

        assert iou(BoundingBox(5, 5, 5, 5), BoundingBox(5, 5, 5, 5)) == 0.0


class TestMatching:
    def _box(self, *corners):
        from recyclevision.models import BoundingBox

        return BoundingBox(*corners)

    def test_overlapping_prediction_matches_its_object(self):
        from recyclevision.evaluate import match_detections

        result = match_detections(
            [("metal can", self._box(0, 0, 10, 10), 0.9)],
            [("steel can", self._box(0, 0, 10, 10))],
        )
        assert result.pairs == [("metal can", "steel can")]
        assert result.precision == 1.0 and result.recall == 1.0

    def test_each_object_is_claimed_once(self):
        """Two boxes on one can is one match and one false positive."""
        from recyclevision.evaluate import match_detections

        result = match_detections(
            [
                ("metal can", self._box(0, 0, 10, 10), 0.9),
                ("metal can", self._box(1, 1, 11, 11), 0.5),
            ],
            [("steel can", self._box(0, 0, 10, 10))],
        )
        assert len(result.pairs) == 1
        assert result.false_positives == ["metal can"]

    def test_highest_confidence_prediction_claims_first(self):
        from recyclevision.evaluate import match_detections

        result = match_detections(
            [
                ("low", self._box(0, 0, 10, 10), 0.2),
                ("high", self._box(0, 0, 10, 10), 0.9),
            ],
            [("truth", self._box(0, 0, 10, 10))],
        )
        assert result.pairs == [("high", "truth")]

    def test_unmatched_prediction_is_a_false_positive(self):
        from recyclevision.evaluate import match_detections

        result = match_detections(
            [("ghost", self._box(50, 50, 60, 60), 0.9)],
            [("real", self._box(0, 0, 10, 10))],
        )
        assert result.pairs == []
        assert result.false_positives == ["ghost"]
        assert result.missed == ["real"]
        assert result.precision == 0.0 and result.recall == 0.0

    def test_no_predictions_means_everything_was_missed(self):
        from recyclevision.evaluate import match_detections

        result = match_detections([], [("a", self._box(0, 0, 5, 5))])
        assert result.missed == ["a"]
        assert result.recall == 0.0

    def test_empty_inputs_do_not_divide_by_zero(self):
        from recyclevision.evaluate import match_detections

        result = match_detections([], [])
        assert result.precision == 0.0 and result.recall == 0.0

    def test_threshold_governs_what_counts_as_a_match(self):
        from recyclevision.evaluate import match_detections

        args = ([("p", self._box(5, 0, 15, 10), 0.9)], [("t", self._box(0, 0, 10, 10))])
        assert match_detections(*args, threshold=0.9).pairs == []
        assert match_detections(*args, threshold=0.3).pairs == [("p", "t")]


class TestPerClassCounts:
    """Two label spaces meet in this tally and must not be added together.

    A ground-truth object is *found* by whatever prediction covers it, under
    any name; a prompt *fires* under its own name, possibly onto an object of
    a different class. An earlier report mixed the two per row and printed
    "4 predictions matched 140 objects" -- impossible, and believed anyway.
    """

    def test_a_correct_detection_counts_everywhere(self):
        counts = per_class_counts(
            [([("metal can", box(0, 0, 10, 10), 0.9)], [("metal can", box(0, 0, 10, 10))])]
        )
        c = counts["metal can"]
        assert (c.truth, c.found, c.named, c.fires) == (1, 1, 1, 1)
        assert c.recall == 1.0
        assert c.naming_accuracy == 1.0

    def test_a_misnamed_detection_is_found_but_not_named(self):
        # The box is right, the label is wrong: recall credits the truth
        # class, naming accuracy does not, and `fires` credits the wrong one.
        counts = per_class_counts(
            [([("metal can", box(0, 0, 10, 10), 0.9)], [("plastic bottle", box(0, 0, 10, 10))])]
        )
        assert counts["plastic bottle"].found == 1
        assert counts["plastic bottle"].named == 0
        assert counts["plastic bottle"].naming_accuracy == 0.0
        assert counts["plastic bottle"].fires == 0
        assert counts["metal can"].fires == 1
        assert counts["metal can"].truth == 0

    def test_found_never_exceeds_truth(self):
        # Five predictions on one object: only one can claim it.
        predictions = [("metal can", box(0, 0, 10, 10), 0.9 - i / 100) for i in range(5)]
        counts = per_class_counts([(predictions, [("metal can", box(0, 0, 10, 10))])])
        c = counts["metal can"]
        assert c.found <= c.truth
        assert c.named <= c.found
        assert c.fires == 5, "every prediction still counts as the prompt firing"

    def test_fires_may_exceed_truth(self):
        # The property that made the old single-column report impossible.
        predictions = [
            ("beverage carton", box(100 * i, 0, 100 * i + 10, 10), 0.9) for i in range(4)
        ]
        counts = per_class_counts([(predictions, [("beverage carton", box(0, 0, 10, 10))])])
        assert counts["beverage carton"].fires > counts["beverage carton"].truth

    def test_a_prompt_that_never_fires_is_visible(self):
        counts = per_class_counts([([], [("glass bottle", box(0, 0, 10, 10))])])
        c = counts["glass bottle"]
        assert c.fires == 0 and c.truth == 1 and c.found == 0
        assert c.recall == 0.0

    def test_a_class_only_ever_predicted_still_appears(self):
        # A prompt firing on something the dataset never labels is a real
        # finding; dropping the row would hide it.
        counts = per_class_counts([([("plastic bag", box(0, 0, 10, 10), 0.9)], [])])
        assert counts["plastic bag"].fires == 1
        assert counts["plastic bag"].truth == 0

    def test_empty_input_is_empty_not_an_error(self):
        assert per_class_counts([]) == {}

    def test_rates_are_zero_rather_than_dividing_by_zero(self):
        counts = per_class_counts([([("plastic bag", box(0, 0, 10, 10), 0.9)], [])])
        assert counts["plastic bag"].recall == 0.0
        assert counts["plastic bag"].naming_accuracy == 0.0
