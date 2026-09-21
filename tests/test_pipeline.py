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


def test_the_old_trained_detector_name_still_resolves():
    # Renamed when train.py started producing detection weights; an existing
    # script or notebook importing the old name should not break.
    from recyclevision.detector import TrainedDetector, TrainedSegmentationDetector

    assert TrainedSegmentationDetector is TrainedDetector


class TestWeightsSelection:
    """Dropping a trained model into models/ must not redefine the baseline.

    `resolve_weights()` prefers models/best_model.pt, and the app's "Stock
    COCO (baseline)" entry used to call `YoloDetector()` with no argument. A
    single `cp` into models/ therefore turned the labelled baseline into the
    trained model, silently, destroying the one comparison it exists for.
    """

    def test_stock_weights_carry_the_coco_caveat(self):
        from recyclevision.weights import STOCK_WEIGHTS, WeightsChoice

        choice = WeightsChoice(path=STOCK_WEIGHTS, is_custom=False)
        assert choice.caveat
        assert "COCO" in choice.caveat

    def test_a_custom_model_is_silent_by_default(self):
        from recyclevision.weights import WeightsChoice

        assert WeightsChoice(path="models/best_model.pt", is_custom=True).caveat == ""

    def test_an_overridden_caveat_survives_being_custom(self):
        # "Not stock COCO" is not a clean bill of health, and a fine-tuned
        # model needs to be able to say so.
        from recyclevision.weights import WeightsChoice

        choice = WeightsChoice(path="x.pt", is_custom=True, caveat_override="only as general as")
        assert choice.caveat == "only as general as"

    def test_resolve_prefers_a_custom_model_when_present(self, tmp_path):
        from recyclevision.weights import resolve_weights

        custom = tmp_path / "best_model.pt"
        custom.write_bytes(b"not really weights")
        assert resolve_weights(custom).is_custom

    def test_resolve_falls_back_when_absent(self, tmp_path):
        from recyclevision.weights import STOCK_WEIGHTS, resolve_weights

        choice = resolve_weights(tmp_path / "absent.pt")
        assert not choice.is_custom
        assert choice.path == STOCK_WEIGHTS

    def test_the_app_pins_its_baseline_to_stock(self):
        # Read rather than imported: app.py is a Streamlit script and running
        # it here would need a session. The property under test is textual —
        # the COCO branch must not go through resolve_weights().
        import ast
        from pathlib import Path

        source = Path(__file__).resolve().parent.parent / "app.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        loader = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "load_detector"
        )
        body = ast.unparse(loader)
        assert "STOCK_WEIGHTS" in body, "the baseline must name stock weights explicitly"
        assert "YoloDetector()" not in body, "a bare YoloDetector() resolves to custom weights"


class TestLatestTrainedWeights:
    """Resolving weights so a timestamped run directory never has to be typed.

    Not a convenience: a hand-typed run path is a constant small source of
    error, and instructions that contain a placeholder for it break when
    pasted into a shell.
    """

    def test_the_newest_run_wins(self, tmp_path, monkeypatch):
        import os
        import time

        from recyclevision import weights as weights_module

        monkeypatch.setattr(weights_module, "RUNS_ROOT", tmp_path / "runs")
        monkeypatch.setattr(weights_module, "CUSTOM_WEIGHTS", tmp_path / "models" / "absent.pt")
        for name, age in (("old_run", 500), ("new_run", 10)):
            folder = tmp_path / "runs" / "detect" / name / "weights"
            folder.mkdir(parents=True)
            best = folder / "best.pt"
            best.write_bytes(b"x")
            os.utime(best, (time.time() - age, time.time() - age))

        assert weights_module.latest_trained_weights().parent.parent.name == "new_run"

    def test_a_stale_named_model_does_not_shadow_a_newer_run(self, tmp_path, monkeypatch):
        """The failure this replaced.

        models/best_model.pt used to win unconditionally, on the reasoning
        that putting weights there deliberately names a current model. True
        the day you do it; false a week later. A WaRP model copied there in
        one session shadowed every run afterwards, and an evaluation meant to
        score a fresh pool run re-scored the old one — returning numbers
        identical to the previous week's, which is the worst way to fail.
        """
        import os
        import time

        from recyclevision import weights as weights_module

        named = tmp_path / "models" / "best_model.pt"
        named.parent.mkdir(parents=True)
        named.write_bytes(b"x")
        os.utime(named, (time.time() - 9999,) * 2)

        run = tmp_path / "runs" / "detect" / "fresh" / "weights"
        run.mkdir(parents=True)
        (run / "best.pt").write_bytes(b"x")

        monkeypatch.setattr(weights_module, "RUNS_ROOT", tmp_path / "runs")
        monkeypatch.setattr(weights_module, "CUSTOM_WEIGHTS", named)

        chosen = weights_module.latest_trained_weights()
        assert chosen == run / "best.pt"
        assert named in weights_module.shadowed_by(chosen), "the loser must be reportable"

    def test_a_freshly_named_model_still_wins(self, tmp_path, monkeypatch):
        import os
        import time

        from recyclevision import weights as weights_module

        run = tmp_path / "runs" / "detect" / "old" / "weights"
        run.mkdir(parents=True)
        (run / "best.pt").write_bytes(b"x")
        os.utime(run / "best.pt", (time.time() - 9999,) * 2)

        named = tmp_path / "models" / "best_model.pt"
        named.parent.mkdir(parents=True)
        named.write_bytes(b"x")

        monkeypatch.setattr(weights_module, "RUNS_ROOT", tmp_path / "runs")
        monkeypatch.setattr(weights_module, "CUSTOM_WEIGHTS", named)
        assert weights_module.latest_trained_weights() == named

    def test_nothing_trained_resolves_to_nothing(self, tmp_path, monkeypatch):
        from recyclevision import weights as weights_module

        monkeypatch.setattr(weights_module, "RUNS_ROOT", tmp_path / "runs")
        monkeypatch.setattr(weights_module, "CUSTOM_WEIGHTS", tmp_path / "absent.pt")
        assert weights_module.latest_trained_weights() is None

    def test_a_runs_directory_with_no_weights_resolves_to_nothing(self, tmp_path, monkeypatch):
        from recyclevision import weights as weights_module

        (tmp_path / "runs" / "detect" / "r").mkdir(parents=True)
        monkeypatch.setattr(weights_module, "RUNS_ROOT", tmp_path / "runs")
        monkeypatch.setattr(weights_module, "CUSTOM_WEIGHTS", tmp_path / "absent.pt")
        assert weights_module.latest_trained_weights() is None
