"""Routing-aware scoring: class errors that matter, and those that do not."""

from __future__ import annotations

from recyclevision.evaluate import score_routing


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
