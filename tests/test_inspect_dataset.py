"""Recognising an unfamiliar dataset's format before importing it.

Guessing wrong can half-succeed, which is worse than failing: an importer
that finds images but no labels writes a dataset of empty annotations, and
that teaches a model every image is background.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
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


class TestCleanName:
    """Some converters bake the original category id into the class name.

    SortWaste's YOLO descriptor reads "0 pet", "1 pead", ... "4 ecal" — the
    original ids, kept inside the names, while the YOLO index beside them is
    a fresh 0-based renumbering. The originals skipped 3, so the two disagree
    from that point on.
    """

    def test_an_index_prefix_is_stripped(self):
        assert inspect_dataset.clean_name("0 pet") == "pet"
        assert inspect_dataset.clean_name("4 ecal") == "ecal"

    def test_a_clean_name_is_untouched(self):
        assert inspect_dataset.clean_name("cardboard") == "cardboard"

    def test_a_hyphenated_name_is_untouched(self):
        assert inspect_dataset.clean_name("bottle-blue") == "bottle-blue"

    def test_a_name_that_is_only_digits_survives(self):
        assert inspect_dataset.clean_name("42") == "42"

    def test_a_name_containing_digits_later_is_untouched(self):
        assert inspect_dataset.clean_name("bottle 5l") == "bottle 5l"


class TestCrossCheck:
    """A dataset shipping both YOLO and COCO describes the same images twice.
    If the totals disagree, the conversion reindexed something, and training
    on it means training on wrong labels while every metric looks plausible.
    """

    def test_agreement_is_reported_as_safe(self):
        counts = Counter({"pet": 10, "ecal": 9})
        lines = "\n".join(inspect_dataset.cross_check(counts, Counter(counts)))
        assert "agrees" in lines
        assert "MISMATCH" not in lines

    def test_a_shifted_reindex_is_caught(self):
        # The exact failure a sloppy converter produces: right totals,
        # attached to the wrong classes.
        yolo = Counter({"pet": 9, "pead": 10, "ecal": 5, "mixed_plastic_soft": 7})
        coco = Counter({"pet": 10, "pead": 7, "ecal": 9, "mixed_plastic_soft": 5})
        lines = "\n".join(inspect_dataset.cross_check(yolo, coco))
        assert "MISMATCH" in lines
        assert "do not train on it" in lines

    def test_a_class_missing_from_one_side_is_a_mismatch(self):
        lines = "\n".join(
            inspect_dataset.cross_check(Counter({"pet": 5}), Counter({"pet": 5, "glass": 3}))
        )
        assert "MISMATCH" in lines

    def test_counting_yolo_labels_uses_the_class_index(self, tmp_path):
        labels = tmp_path / "labels" / "train"
        labels.mkdir(parents=True)
        (labels / "a.txt").write_text("0 .5 .5 .2 .2\n1 .5 .5 .2 .2\n0 .5 .5 .2 .2\n")
        counts = inspect_dataset.count_yolo_labels(tmp_path, ["0 pet", "1 pead"])
        assert counts == Counter({"pet": 2, "pead": 1})

    def test_an_out_of_range_index_is_ignored_not_crashed(self, tmp_path):
        labels = tmp_path / "labels" / "train"
        labels.mkdir(parents=True)
        (labels / "a.txt").write_text("0 .5 .5 .2 .2\n99 .5 .5 .2 .2\n")
        assert inspect_dataset.count_yolo_labels(tmp_path, ["0 pet"]) == Counter({"pet": 1})
