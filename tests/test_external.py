"""Translating an outside dataset's classes into ours."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from recyclevision import external
from recyclevision.external import (
    ClassMapping,
    MappingError,
    build_index_map,
    read_yolo_data_yaml,
    remap_label_line,
)

MINIMAL = {
    "name": "Test dataset",
    "mapping": {"PET": "plastic bottle", "Metal": "metal can", "Other": None},
    "cannot_express": ["container glass vs drinking glass"],
}


@pytest.fixture
def mapping(tmp_path):
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(MINIMAL))
    return ClassMapping.load(path)


class TestLoading:
    def test_reads_rules_and_caveats(self, mapping):
        assert mapping.name == "Test dataset"
        assert mapping.translate("PET") == "plastic bottle"
        assert mapping.cannot_express

    def test_missing_file_is_a_mapping_error(self, tmp_path):
        with pytest.raises(MappingError, match="no mapping file"):
            ClassMapping.load(tmp_path / "absent.yaml")

    def test_invalid_yaml_is_a_mapping_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("mapping: [unclosed\n")
        with pytest.raises(MappingError, match="not valid YAML"):
            ClassMapping.load(bad)

    def test_file_without_a_mapping_section_is_rejected(self, tmp_path):
        bare = tmp_path / "bare.yaml"
        bare.write_text(yaml.safe_dump({"name": "nothing"}))
        with pytest.raises(MappingError, match="mapping"):
            ClassMapping.load(bare)


class TestTranslation:
    def test_null_means_deliberately_dropped(self, mapping):
        assert mapping.translate("Other") is None
        assert mapping.dropped() == ["Other"]

    def test_an_unmapped_class_raises_rather_than_dropping(self, mapping):
        """The distinction the whole module exists for.

        A dropped class is a decision; an unmapped one is an oversight that
        would teach the model those objects are background.
        """
        with pytest.raises(MappingError, match="no rule for"):
            mapping.translate("Glass")

    def test_unmapped_lists_what_is_missing(self, mapping):
        assert mapping.unmapped(["PET", "Glass", "Metal", "Foam"]) == ["Glass", "Foam"]
        assert mapping.unmapped(["PET", "Metal"]) == []

    def test_targets_outside_the_vocabulary_are_reported(self, mapping):
        assert mapping.unknown_targets(["plastic bottle"]) == ["metal can"]
        assert mapping.unknown_targets(["plastic bottle", "metal can"]) == []


class TestDataYaml:
    def _write(self, tmp_path, names, **extra):
        path = tmp_path / "data.yaml"
        path.write_text(yaml.safe_dump({"names": names, **extra}))
        return path

    def test_reads_names_as_an_index_mapping(self, tmp_path):
        path = self._write(tmp_path, {0: "PET", 1: "Metal"})
        _root, names = read_yolo_data_yaml(path)
        assert names == ["PET", "Metal"]

    def test_reads_names_as_a_list(self, tmp_path):
        path = self._write(tmp_path, ["PET", "Metal"])
        _root, names = read_yolo_data_yaml(path)
        assert names == ["PET", "Metal"]

    def test_index_order_is_respected_not_insertion_order(self, tmp_path):
        """Class index is the label file's only link to a name."""
        path = self._write(tmp_path, {2: "Third", 0: "First", 1: "Second"})
        _root, names = read_yolo_data_yaml(path)
        assert names == ["First", "Second", "Third"]

    def test_relative_path_resolves_against_the_descriptor(self, tmp_path):
        (tmp_path / "sub").mkdir()
        path = self._write(tmp_path, ["PET"], path="sub")
        root, _names = read_yolo_data_yaml(path)
        assert root == (tmp_path / "sub").resolve()

    def test_missing_names_is_rejected(self, tmp_path):
        path = tmp_path / "data.yaml"
        path.write_text(yaml.safe_dump({"train": "images/train"}))
        with pytest.raises(MappingError, match="names"):
            read_yolo_data_yaml(path)


class TestRemapping:
    def test_index_map_translates_through_the_vocabulary(self, mapping):
        index_map = build_index_map(
            ["PET", "Other", "Metal"], mapping, ["metal can", "plastic bottle"]
        )
        assert index_map == {0: 1, 1: None, 2: 0}

    def test_label_line_keeps_geometry_and_changes_only_the_class(self):
        line = remap_label_line("3 0.5 0.5 0.2 0.2", {3: 7})
        assert line == "7 0.5 0.5 0.2 0.2"

    def test_polygon_geometry_survives_untouched(self):
        line = remap_label_line("2 0.1 0.1 0.2 0.2 0.3 0.3", {2: 0})
        assert line == "0 0.1 0.1 0.2 0.2 0.3 0.3"

    def test_dropped_class_yields_no_line(self):
        assert remap_label_line("1 0.5 0.5 0.2 0.2", {1: None}) is None

    def test_class_absent_from_the_map_yields_no_line(self):
        assert remap_label_line("9 0.5 0.5 0.2 0.2", {0: 0}) is None

    def test_malformed_lines_are_skipped_not_crashed_on(self):
        assert remap_label_line("", {0: 0}) is None
        assert remap_label_line("0 0.5", {0: 0}) is None
        assert remap_label_line("notanumber 0.5 0.5 0.2 0.2", {0: 0}) is None


# --- dataset geometry -------------------------------------------------------
#
# Ultralytics refuses to train a segmentation model on a boxes-only dataset,
# but only after caching every label -- minutes into a run that was never
# going to start. These cover recognising that up front.


def _write_dataset(root: Path, *, train_lines: list[str], split: str = "train") -> Path:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    labels = root / "labels" / split
    labels.mkdir(parents=True, exist_ok=True)
    for i, line in enumerate(train_lines):
        (labels / f"frame_{i:03d}.txt").write_text(line + "\n", encoding="utf-8")
    data = root / "data.yaml"
    data.write_text(
        f"path: .\n{split}: images/{split}\nnames:\n  0: MixedRecycling\n", encoding="utf-8"
    )
    return data


def test_box_labels_read_as_boxes(tmp_path):
    _write_dataset(tmp_path, train_lines=["0 0.5 0.5 0.2 0.3"])
    assert external.inspect_label_geometry(tmp_path / "labels" / "train") == external.BOXES


def test_polygon_labels_read_as_polygons(tmp_path):
    _write_dataset(tmp_path, train_lines=["0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4"])
    assert external.inspect_label_geometry(tmp_path / "labels" / "train") == external.POLYGONS


def test_a_dataset_holding_both_is_mixed(tmp_path):
    _write_dataset(
        tmp_path,
        train_lines=["0 0.5 0.5 0.2 0.3", "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4"],
    )
    assert external.inspect_label_geometry(tmp_path / "labels" / "train") == external.MIXED


def test_empty_label_files_are_not_mistaken_for_boxes(tmp_path):
    # An empty .txt means "this image contains nothing", which is a real
    # training signal -- but a directory of only these has no geometry at all.
    _write_dataset(tmp_path, train_lines=[""])
    assert external.inspect_label_geometry(tmp_path / "labels" / "train") == external.EMPTY


def test_a_missing_label_directory_is_empty_not_an_error(tmp_path):
    assert external.inspect_label_geometry(tmp_path / "nowhere") == external.EMPTY


def test_label_dir_swaps_the_last_images_segment(tmp_path):
    # "images" appearing twice is the case that catches a naive replace().
    found = external.label_dir_for(Path("datasets/images/warp/images/train"))
    assert found == Path("datasets/images/warp/labels/train")


def test_label_dir_leaves_an_unconventional_layout_alone(tmp_path):
    # Better to find no labels and say so than to invent a path.
    assert external.label_dir_for(Path("datasets/warp/frames")) == Path("datasets/warp/frames")


def test_split_images_dir_resolves_against_the_descriptor(tmp_path):
    data = _write_dataset(tmp_path, train_lines=["0 0.5 0.5 0.2 0.3"])
    assert external.read_split_images_dir(data, "train") == (tmp_path / "images" / "train")


def test_split_images_dir_is_none_for_a_split_not_declared(tmp_path):
    data = _write_dataset(tmp_path, train_lines=["0 0.5 0.5 0.2 0.3"])
    assert external.read_split_images_dir(data, "val") is None


def test_split_images_dir_accepts_a_list_of_paths(tmp_path):
    # Ultralytics allows several directories per split.
    data = tmp_path / "data.yaml"
    data.write_text("path: .\ntrain:\n  - images/a\n  - images/b\nnames: [x]\n", encoding="utf-8")
    assert external.read_split_images_dir(data, "train") == (tmp_path / "images" / "a")
