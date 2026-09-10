"""Export rows, CSV, and JSON payloads."""

from __future__ import annotations

import csv
import io
import json

from recyclevision.detector import StubDetector
from recyclevision.pipeline import SortingPipeline
from recyclevision.report import COLUMNS, detection_rows, result_payload, to_csv, to_json
from tests.conftest import make_detection


def sort(policy, labels, image):
    detections = [make_detection(label, conf) for label, conf in labels]
    return SortingPipeline(StubDetector(detections), policy).sort(image)


class TestDetectionRows:
    def test_one_row_per_item(self, policy, image):
        result = sort(policy, [("bottle", 0.9), ("pizza", 0.8)], image)
        assert len(detection_rows(result, "a.jpg")) == 2

    def test_carries_geometry(self, policy, image):
        result = sort(policy, [("bottle", 0.9)], image)
        row = detection_rows(result, "a.jpg")[0]
        assert row["width"] == 100 and row["height"] == 100
        assert row["area_px"] == 10_000

    def test_non_waste_is_not_exported(self, policy, image):
        result = sort(policy, [("bottle", 0.9), ("person", 0.99)], image)
        assert len(detection_rows(result, "a.jpg")) == 1

    def test_every_declared_column_is_populated(self, policy, image):
        row = detection_rows(sort(policy, [("bottle", 0.9)], image), "a.jpg")[0]
        assert set(row) == set(COLUMNS)


class TestCsv:
    def test_roundtrips_through_a_reader(self, policy, image):
        result = sort(policy, [("bottle", 0.9), ("pizza", 0.8)], image)
        text = to_csv(detection_rows(result, "a.jpg"))
        parsed = list(csv.DictReader(io.StringIO(text)))
        assert len(parsed) == 2
        assert parsed[0]["source"] == "a.jpg"

    def test_column_order_is_stable(self, policy, image):
        text = to_csv(detection_rows(sort(policy, [("bottle", 0.9)], image), "a.jpg"))
        assert text.splitlines()[0] == ",".join(COLUMNS)

    def test_empty_export_still_has_a_header(self):
        """A header with no rows reads as "nothing found"; an empty file reads
        as a failure."""
        assert to_csv([]).strip() == ",".join(COLUMNS)

    def test_commas_in_handling_text_do_not_break_the_csv(self, policy, image):
        result = sort(policy, [("bottle", 0.9)], image)
        rows = detection_rows(result, "a.jpg")
        rows[0]["handling"] = "Rinse, then flatten, then bin"
        parsed = list(csv.DictReader(io.StringIO(to_csv(rows))))
        assert parsed[0]["handling"] == "Rinse, then flatten, then bin"


class TestJson:
    def test_payload_is_serialisable_and_complete(self, policy, image):
        result = sort(policy, [("bottle", 0.9), ("wine glass", 0.8)], image)
        payload = result_payload(result, "a.jpg")
        parsed = json.loads(to_json([payload]))[0]

        assert parsed["total_items"] == 2
        assert parsed["diversion_rate"] == 0.5
        assert parsed["counts_by_bin"] == {"recycling": 1, "landfill": 1}
        assert len(parsed["items"]) == 2

    def test_records_ignored_classes_separately(self, policy, image):
        result = sort(policy, [("bottle", 0.9), ("person", 0.99)], image)
        payload = result_payload(result, "a.jpg")
        assert payload["ignored_classes"] == ["person"]
        assert payload["total_items"] == 1
