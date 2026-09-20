"""Reading COCO annotations into this project's YOLO-shaped datasets.

Two things here are easy to get silently wrong, and both produce a dataset
that trains without complaint and teaches the wrong thing: the box geometry,
and which file on disk a `file_name` actually refers to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recyclevision.coco import (
    CocoError,
    index_images,
    locate,
    looks_like_mask,
    read_coco,
    to_yolo_line,
)


class TestGeometry:
    """COCO is [x_min, y_min, w, h] in pixels; YOLO is centre-x, centre-y, w,
    h normalised. A transposed pair or a missing divide writes labels that
    look fine in a file and sit nowhere near the objects.
    """

    def test_a_box_converts_to_normalised_centre_form(self):
        # (10,20) 100x50 in a 200x100 image -> centre (60,45).
        assert to_yolo_line(3, 10, 20, 100, 50, 200, 100) == (
            "3 0.300000 0.450000 0.500000 0.500000"
        )

    def test_a_full_frame_box_is_centred_and_whole(self):
        assert to_yolo_line(0, 0, 0, 640, 480, 640, 480) == (
            "0 0.500000 0.500000 1.000000 1.000000"
        )

    def test_the_class_id_is_first(self):
        assert to_yolo_line(15, 0, 0, 10, 10, 100, 100).startswith("15 ")

    def test_a_box_running_past_the_edge_is_clamped(self):
        # Ultralytics rejects a whole label file over a coordinate above 1.0,
        # and COCO boxes do occasionally overhang by a pixel or two.
        line = to_yolo_line(0, 0, 0, 700, 500, 640, 480)
        assert line is not None
        assert all(0.0 <= float(v) <= 1.0 for v in line.split()[1:])

    def test_a_zero_area_box_is_dropped(self):
        assert to_yolo_line(0, 10, 10, 0, 50, 200, 100) is None
        assert to_yolo_line(0, 10, 10, 50, 0, 200, 100) is None

    def test_an_image_with_no_size_is_refused_rather_than_divided_by_zero(self):
        assert to_yolo_line(0, 10, 10, 50, 50, 0, 100) is None
        assert to_yolo_line(0, 10, 10, 50, 50, 200, 0) is None


class TestReadCoco:
    @staticmethod
    def _write(path: Path, **overrides) -> Path:
        payload = {
            "images": [{"id": 0, "file_name": "a.png", "width": 100, "height": 50}],
            "categories": [{"id": 1, "name": "cardboard"}, {"id": 2, "name": "metal"}],
            "annotations": [
                {"id": 0, "image_id": 0, "category_id": 1, "bbox": [1, 2, 3, 4]},
                {"id": 1, "image_id": 0, "category_id": 2, "bbox": [5, 6, 7, 8]},
            ],
        }
        payload.update(overrides)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_categories_and_boxes_are_read(self, tmp_path):
        names, images = read_coco(self._write(tmp_path / "l.json"))
        assert names == ["cardboard", "metal"]
        assert len(images) == 1
        assert {b[0] for b in images[0].boxes} == {"cardboard", "metal"}

    def test_an_image_with_no_annotations_is_kept(self, tmp_path):
        # In YOLO an empty label file means "nothing here", which is exactly
        # what an unannotated frame is and a real training signal.
        path = self._write(tmp_path / "l.json", annotations=[])
        _names, images = read_coco(path)
        assert len(images) == 1 and images[0].boxes == []

    def test_an_annotation_for_an_unknown_image_is_skipped(self, tmp_path):
        path = self._write(
            tmp_path / "l.json",
            annotations=[{"id": 0, "image_id": 99, "category_id": 1, "bbox": [1, 2, 3, 4]}],
        )
        _names, images = read_coco(path)
        assert images[0].boxes == []

    def test_a_malformed_bbox_is_skipped(self, tmp_path):
        path = self._write(
            tmp_path / "l.json",
            annotations=[{"id": 0, "image_id": 0, "category_id": 1, "bbox": [1, 2]}],
        )
        _names, images = read_coco(path)
        assert images[0].boxes == []

    def test_a_missing_file_is_a_coco_error(self, tmp_path):
        with pytest.raises(CocoError, match="no COCO file"):
            read_coco(tmp_path / "absent.json")

    def test_json_that_is_not_coco_is_a_coco_error(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text('{"lr": 0.01}', encoding="utf-8")
        with pytest.raises(CocoError, match="annotations"):
            read_coco(path)


class TestLocatingImages:
    """ZeroWaste stores frames in data/ and segmentation masks in sem_seg/
    under the SAME filenames. A basename lookup that resolves to the mask
    imports a greyscale label map as a training image — it trains happily
    and teaches nothing.
    """

    @staticmethod
    def _tree(tmp_path: Path) -> Path:
        for folder in ("data", "sem_seg"):
            (tmp_path / folder).mkdir()
            (tmp_path / folder / "01_frame_000000.PNG").write_bytes(b"\x89PNG")
        return tmp_path

    def test_a_frame_wins_over_a_mask_of_the_same_name(self, tmp_path):
        index = index_images(self._tree(tmp_path))
        assert locate("01_frame_000000.PNG", index).parent.name == "data"

    def test_a_mask_directory_is_recognised(self):
        assert looks_like_mask(Path("val/sem_seg/f.png"))
        assert looks_like_mask(Path("x/masks/f.png"))
        assert not looks_like_mask(Path("val/data/f.png"))

    def test_an_explicit_relative_path_still_reaches_the_mask(self, tmp_path):
        # Asking for it by name is unambiguous and should be honoured.
        index = index_images(self._tree(tmp_path))
        assert locate("sem_seg/01_frame_000000.PNG", index).parent.name == "sem_seg"

    def test_a_leading_dot_slash_resolves(self, tmp_path):
        index = index_images(self._tree(tmp_path))
        assert locate("./01_frame_000000.PNG", index) is not None

    def test_an_unknown_name_resolves_to_nothing(self, tmp_path):
        assert locate("absent.png", index_images(self._tree(tmp_path))) is None

    def test_an_empty_name_resolves_to_nothing(self, tmp_path):
        assert locate("", index_images(self._tree(tmp_path))) is None
