"""End-to-end pipeline behaviour, driven by a stub detector."""

from __future__ import annotations

import pytest

from recyclevision.detector import Detector, StubDetector
from recyclevision.models import SortResult
from recyclevision.pipeline import SortingPipeline
from tests.conftest import make_detection


def build(policy, labels_with_conf) -> SortingPipeline:
    detections = [make_detection(label, conf) for label, conf in labels_with_conf]
    return SortingPipeline(StubDetector(detections, name="Stub"), policy)


class TestSorting:
    def test_routes_detections_into_items(self, policy, image):
        result = build(policy, [("bottle", 0.9), ("pizza", 0.8)]).sort(image)
        assert result.total_items == 2
        assert set(result.counts_by_bin) == {"recycling", "compost"}

    def test_non_waste_is_recorded_but_not_counted(self, policy, image):
        result = build(policy, [("bottle", 0.9), ("person", 0.99), ("car", 0.95)]).sort(image)
        assert result.total_items == 1
        assert {d.label for d in result.ignored} == {"person", "car"}
        # A 99%-confident person must not drag the item average around.
        assert result.average_confidence == pytest.approx(0.9)

    def test_items_are_ordered_by_confidence(self, policy, image):
        result = build(policy, [("bottle", 0.4), ("pizza", 0.95), ("book", 0.7)]).sort(image)
        confidences = [item.confidence for item in result.items]
        assert confidences == sorted(confidences, reverse=True)

    def test_confidence_threshold_is_honoured(self, policy, image):
        pipeline = build(policy, [("bottle", 0.9), ("pizza", 0.3)])
        assert pipeline.sort(image, confidence=0.5).total_items == 1
        assert pipeline.sort(image, confidence=0.1).total_items == 2

    def test_result_records_provenance(self, policy, image):
        result = build(policy, [("bottle", 0.9)]).sort(image)
        assert result.model_name == "Stub"
        assert result.policy_name == policy.name

    def test_empty_image_yields_empty_result(self, policy, image):
        result = build(policy, []).sort(image)
        assert result.total_items == 0
        assert result.diversion_rate == 0.0
        assert result.contamination_rate == 0.0
        assert result.average_confidence == 0.0
        assert result.bins_used == []

    def test_image_of_only_non_waste_is_not_a_contaminated_stream(self, policy, image):
        """Zero items means "nothing to report", not "0% diverted"."""
        result = build(policy, [("person", 0.9), ("dog", 0.8)]).sort(image)
        assert result.total_items == 0
        assert result.contamination_rate == 0.0


class TestMetrics:
    def test_diversion_and_contamination_are_complementary(self, policy, image):
        result = build(policy, [("bottle", 0.9), ("pizza", 0.9), ("wine glass", 0.9)]).sort(image)
        assert result.diverted_count == 2
        assert result.diversion_rate == pytest.approx(2 / 3)
        assert result.contamination_rate == pytest.approx(1 / 3)

    def test_review_queue_collects_low_certainty_items(self, policy, image):
        result = build(policy, [("bottle", 0.9), ("bowl", 0.9), ("fork", 0.9)]).sort(image)
        assert {item.detection.label for item in result.items_for_review} == {"bowl", "fork"}

    def test_counts_by_bin_and_material(self, policy, image):
        result = build(policy, [("pizza", 0.9), ("banana", 0.9), ("bottle", 0.9)]).sort(image)
        assert result.counts_by_bin["compost"] == 2
        assert result.counts_by_material["organic"] == 2

    def test_items_in_filters_by_bin(self, policy, image):
        result = build(policy, [("pizza", 0.9), ("bottle", 0.9)]).sort(image)
        assert len(result.items_in("compost")) == 1
        assert result.items_in("nonexistent") == []

    def test_bins_used_preserves_first_seen_order_without_duplicates(self, policy, image):
        result = build(policy, [("pizza", 0.95), ("banana", 0.9), ("bottle", 0.85)]).sort(image)
        assert [b.key for b in result.bins_used] == ["compost", "recycling"]

    def test_len_matches_item_count(self, policy, image):
        result = build(policy, [("bottle", 0.9), ("pizza", 0.9)]).sort(image)
        assert len(result) == result.total_items == 2


class TestDetectorInterface:
    def test_stub_satisfies_the_protocol(self):
        assert isinstance(StubDetector(), Detector)

    def test_pipeline_accepts_any_detector(self, policy, image):
        """The seam that lets a conveyor-trained model drop in later."""

        class Fake:
            name = "Fake"

            def detect(self, image, confidence=0.25):
                return [make_detection("bottle", 0.9)]

        result = SortingPipeline(Fake(), policy).sort(image)
        assert isinstance(result, SortResult)
        assert result.total_items == 1

    def test_unroutable_classes_reports_policy_gaps(self, policy):
        class Fake:
            name = "Fake"
            class_names = ["bottle", "aardvark", "pizza"]

            def detect(self, image, confidence=0.25):
                return []

        assert SortingPipeline(Fake(), policy).unroutable_classes() == ["aardvark"]

    def test_unroutable_classes_is_empty_when_detector_lists_none(self, policy):
        assert SortingPipeline(StubDetector(), policy).unroutable_classes() == []
