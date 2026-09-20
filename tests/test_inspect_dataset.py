"""Recognising an unfamiliar dataset's format before importing it.

Guessing wrong can half-succeed, which is worse than failing: an importer
that finds images but no labels writes a dataset of empty annotations, and
that teaches a model every image is background.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import inspect_dataset  # noqa: E402


def _coco(path: Path, categories, annotations) -> Path:
    path.write_text(
        json.dumps(
            {
                "images": [{"id": 0, "file_name": "a.jpg"}],
                "categories": categories,
                "annotations": annotations,
            }
        ),
        encoding="utf-8",
    )
    return path


class TestCocoDetection:
    def test_a_coco_file_is_recognised(self, tmp_path):
        path = _coco(tmp_path / "ann.json", [{"id": 1, "name": "cardboard"}], [])
        assert inspect_dataset.looks_like_coco(path) is not None

    def test_arbitrary_json_is_not_coco(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text('{"lr": 0.01, "epochs": 100}', encoding="utf-8")
        assert inspect_dataset.looks_like_coco(path) is None

    def test_malformed_json_does_not_raise(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json at all", encoding="utf-8")
        assert inspect_dataset.looks_like_coco(path) is None

    def test_a_missing_file_does_not_raise(self, tmp_path):
        assert inspect_dataset.looks_like_coco(tmp_path / "absent.json") is None

    def test_class_names_and_counts_are_reported(self, tmp_path):
        path = _coco(
            tmp_path / "ann.json",
            [{"id": 1, "name": "soft_plastic"}, {"id": 2, "name": "metal"}],
            [
                {"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 1, 1]},
                {"id": 1, "image_id": 0, "category_id": 1, "bbox": [0, 0, 1, 1]},
                {"id": 2, "image_id": 0, "category_id": 2, "bbox": [0, 0, 1, 1]},
            ],
        )
        lines = "\n".join(inspect_dataset.report_coco(path, inspect_dataset.looks_like_coco(path)))
        assert "soft_plastic" in lines and "metal" in lines
        assert "instances   3" in lines

    def test_polygons_are_distinguished_from_boxes(self, tmp_path):
        boxes = _coco(
            tmp_path / "b.json",
            [{"id": 1, "name": "x"}],
            [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 1, 1]}],
        )
        assert "boxes only" in "\n".join(
            inspect_dataset.report_coco(boxes, inspect_dataset.looks_like_coco(boxes))
        )

        polys = _coco(
            tmp_path / "p.json",
            [{"id": 1, "name": "x"}],
            [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "bbox": [0, 0, 1, 1],
                    "segmentation": [[1, 2, 3, 4, 5, 6]],
                }
            ],
        )
        assert "polygons" in "\n".join(
            inspect_dataset.report_coco(polys, inspect_dataset.looks_like_coco(polys))
        )


class TestYoloReport:
    def test_classes_are_listed_with_their_indices(self, tmp_path):
        path = tmp_path / "data.yaml"
        path.write_text("train: images/train\nnames:\n  0: bottle\n  1: can\n", encoding="utf-8")
        lines = "\n".join(inspect_dataset.report_yolo(path))
        assert "0  bottle" in lines and "1  can" in lines

    def test_a_list_of_names_works_too(self, tmp_path):
        path = tmp_path / "data.yaml"
        path.write_text("train: x\nnames: [bottle, can]\n", encoding="utf-8")
        assert "classes     2" in "\n".join(inspect_dataset.report_yolo(path))

    def test_broken_yaml_is_reported_not_raised(self, tmp_path):
        path = tmp_path / "data.yaml"
        path.write_text("names: [unclosed\n", encoding="utf-8")
        assert "unreadable" in "\n".join(inspect_dataset.report_yolo(path))


class TestFind:
    def test_nested_files_are_found(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        deep.mkdir(parents=True)
        (deep / "target.json").write_text("{}", encoding="utf-8")
        assert inspect_dataset.find(tmp_path, lambda p: p.suffix == ".json")

    def test_the_limit_is_respected(self, tmp_path):
        for i in range(20):
            (tmp_path / f"{i}.json").write_text("{}", encoding="utf-8")
        assert len(inspect_dataset.find(tmp_path, lambda p: p.suffix == ".json", limit=5)) == 5
