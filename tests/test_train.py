"""Choosing the starting weights and the device before a run begins.

Both of these exist to fail fast. Ultralytics discovers a model/dataset
mismatch only after caching every label, and answers a request for a CUDA
device on a Mac with a raw traceback -- in both cases minutes after the run
could have been rejected in a tenth of a second, with a usable explanation.

No torch and no weights here: only the decisions made before either loads.
"""

from __future__ import annotations

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
