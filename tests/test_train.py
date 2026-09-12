"""Choosing the starting weights and the device before a run begins.

Both of these exist to fail fast. Ultralytics discovers a model/dataset
mismatch only after caching every label, and answers a request for a CUDA
device on a Mac with a raw traceback -- in both cases minutes after the run
could have been rejected in a tenth of a second, with a usable explanation.

No torch and no weights here: only the decisions made before either loads.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import train
from recyclevision.external import BOXES, MIXED, POLYGONS

DATA = Path("datasets/warp/data.yaml")


def test_boxes_get_a_detection_model():
    assert train.choose_model(None, BOXES, DATA) == train.DETECT_MODEL


def test_polygons_get_a_segmentation_model():
    assert train.choose_model(None, POLYGONS, DATA) == train.SEGMENT_MODEL


def test_a_segmentation_model_on_boxes_is_refused_before_the_run():
    # The exact case WaRP hit: yolo11s-seg.pt against a detect-only dataset.
    with pytest.raises(SystemExit) as excinfo:
        train.choose_model("yolo11s-seg.pt", BOXES, DATA)
    message = str(excinfo.value)
    # The fix belongs in the message, not in a search: ultralytics' own error
    # says "supply a segment dataset", which blames the wrong half.
    assert train.DETECT_MODEL in message
    assert "--model" in message


def test_a_detection_model_on_polygons_is_allowed():
    # Ultralytics derives boxes from polygons happily, so this is a real
    # choice -- train detection on a segmentation dataset -- not a mistake.
    assert train.choose_model("yolo11s.pt", POLYGONS, DATA) == "yolo11s.pt"


def test_mixed_geometry_is_refused_whatever_the_model():
    for requested in (None, "yolo11s.pt", "yolo11s-seg.pt"):
        with pytest.raises(SystemExit):
            train.choose_model(requested, MIXED, DATA)


def test_an_explicit_model_is_otherwise_respected():
    assert train.choose_model("runs/detect/x/weights/best.pt", BOXES, DATA) == (
        "runs/detect/x/weights/best.pt"
    )


def test_describe_geometry_reads_a_dataset_on_disk(tmp_path):
    (tmp_path / "images" / "train").mkdir(parents=True)
    labels = tmp_path / "labels" / "train"
    labels.mkdir(parents=True)
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    data.write_text("path: .\ntrain: images/train\nnames: [x]\n", encoding="utf-8")

    geometry, labels_dir = train.describe_geometry(data)
    assert geometry == BOXES
    assert labels_dir == labels


class TestNmsBudget:
    """Ultralytics gives a whole NMS batch `2.0 + 0.05 * batch` seconds and,
    on timeout, breaks out of its per-image loop -- leaving every image it
    never reached with the empty tensor it was initialised with. Those images
    score as "predicted nothing", so validation understates the model, and
    validation is what picks best.pt and trips `patience`.
    """

    @staticmethod
    def _fake_ultralytics(monkeypatch):
        """Stand in for ultralytics.utils.nms, so this runs without torch."""
        calls = []

        def non_max_suppression(*args, **kwargs):
            calls.append(kwargs)
            return []

        nms = types.ModuleType("ultralytics.utils.nms")
        nms.non_max_suppression = non_max_suppression
        utils = types.ModuleType("ultralytics.utils")
        utils.nms = nms
        root = types.ModuleType("ultralytics")
        root.utils = utils
        for name, module in (
            ("ultralytics", root),
            ("ultralytics.utils", utils),
            ("ultralytics.utils.nms", nms),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        return nms, calls

    def test_the_budget_is_raised_on_mps(self, monkeypatch):
        nms, calls = self._fake_ultralytics(monkeypatch)
        train.relax_nms_time_limit("mps")
        nms.non_max_suppression("preds", 0.001)
        assert calls[0]["max_time_img"] == train.NMS_SECONDS_PER_IMAGE

    def test_cuda_is_left_alone(self, monkeypatch):
        # There the stock budget is ample, and quietly changing a CUDA run's
        # behaviour to fix an Apple Silicon problem would be its own surprise.
        nms, _calls = self._fake_ultralytics(monkeypatch)
        before = nms.non_max_suppression
        train.relax_nms_time_limit("0")
        assert nms.non_max_suppression is before

    def test_an_explicit_caller_still_wins(self, monkeypatch):
        nms, calls = self._fake_ultralytics(monkeypatch)
        train.relax_nms_time_limit("cpu")
        nms.non_max_suppression("preds", max_time_img=0.01)
        assert calls[0]["max_time_img"] == 0.01

    def test_patching_twice_does_not_stack_wrappers(self, monkeypatch):
        nms, _calls = self._fake_ultralytics(monkeypatch)
        train.relax_nms_time_limit("mps")
        once = nms.non_max_suppression
        train.relax_nms_time_limit("mps")
        assert nms.non_max_suppression is once

    def test_a_missing_ultralytics_is_not_an_error(self, monkeypatch):
        # train.py is importable without torch installed; this must stay true.
        for name in ("ultralytics", "ultralytics.utils", "ultralytics.utils.nms"):
            monkeypatch.setitem(sys.modules, name, None)
        train.relax_nms_time_limit("mps")  # must not raise

    def test_the_raised_budget_actually_clears_a_real_batch(self):
        # The arithmetic that matters: a 16-image validation batch was getting
        # 2.8s total, which is what the observed warning reported.
        stock = 2.0 + 0.05 * 16
        raised = 2.0 + train.NMS_SECONDS_PER_IMAGE * 16
        assert stock == pytest.approx(2.8)
        assert raised > 10 * stock
