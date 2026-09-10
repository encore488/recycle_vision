"""Aggregating results across images."""

from __future__ import annotations

import pytest

from recyclevision.detector import StubDetector
from recyclevision.pipeline import SortingPipeline
from recyclevision.session import Session
from tests.conftest import make_detection


def build(policy, image, *batches):
    session = Session()
    for i, labels in enumerate(batches):
        detections = [make_detection(label, 0.9) for label in labels]
        session.add(f"img{i}.jpg", SortingPipeline(StubDetector(detections), policy).sort(image))
    return session


class TestSession:
    def test_totals_span_every_image(self, policy, image):
        session = build(policy, image, ["bottle", "pizza"], ["bottle"])
        assert session.image_count == 2
        assert session.total_items == 3
        assert session.counts_by_bin["recycling"] == 2

    def test_diversion_is_computed_over_the_whole_batch(self, policy, image):
        session = build(policy, image, ["bottle"], ["wine glass"])
        assert session.diversion_rate == pytest.approx(0.5)
        assert session.contamination_rate == pytest.approx(0.5)

    def test_empty_session_is_zero_not_an_error(self):
        session = Session()
        assert session.total_items == 0
        assert session.diversion_rate == 0.0
        assert session.contamination_rate == 0.0
        assert session.average_confidence == 0.0
        assert session.bins_used == []
        assert session.as_result().total_items == 0

    def test_images_with_no_items_still_count_as_images(self, policy, image):
        session = build(policy, image, ["bottle"], ["person"])
        assert session.image_count == 2
        assert session.total_items == 1

    def test_composition_counts_plain_language_names(self, policy, image):
        session = build(policy, image, ["banana", "pizza"], ["bottle"])
        assert session.counts_by_item["Food waste"] == 2

    def test_review_queue_spans_images(self, policy, image):
        session = build(policy, image, ["bowl"], ["fork"])
        assert len(session.items_for_review) == 2

    def test_as_result_flattens_for_single_result_consumers(self, policy, image):
        session = build(policy, image, ["bottle", "pizza"], ["wine glass"])
        combined = session.as_result()
        assert combined.total_items == 3
        assert combined.diversion_rate == session.diversion_rate

    def test_as_result_keeps_ignored_detections(self, policy, image):
        session = build(policy, image, ["person"], ["car"])
        assert {d.label for d in session.as_result().ignored} == {"person", "car"}
