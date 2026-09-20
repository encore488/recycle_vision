"""Merging several public datasets into one training set.

Every public waste dataset numbers its frames from 1, so importing two into
one directory used to overwrite the overlap silently: the second source's
`000001.jpg` replaced the first's, and the instance counts printed afterwards
looked perfectly healthy because they were counted before the write.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import import_dataset  # noqa: E402


class TestSlugify:
    def test_a_name_becomes_a_filesystem_safe_tag(self):
        assert import_dataset.slugify("WaRP (Warp-D)") == "warp-warp-d"

    def test_punctuation_does_not_produce_empty_segments(self):
        assert "--" not in import_dataset.slugify("A -- B // C")

    def test_a_long_name_is_bounded(self):
        assert len(import_dataset.slugify("x" * 200)) <= 32

    def test_distinct_datasets_get_distinct_tags(self):
        assert import_dataset.slugify("ZeroWaste-f") != import_dataset.slugify("TACO")


class TestSourceRegistry:
    def test_an_absent_registry_reads_as_empty(self, tmp_path):
        assert import_dataset.read_sources(tmp_path) == {}

    def test_a_written_registry_round_trips(self, tmp_path):
        record = {"warp-": {"mapping": "WaRP", "images": 2974}}
        import_dataset.write_sources(tmp_path, record)
        assert import_dataset.read_sources(tmp_path) == record

    def test_a_corrupt_registry_does_not_crash_the_import(self, tmp_path):
        # Better to treat provenance as unknown than to refuse to import.
        (tmp_path / "sources.json").write_text("{not json", encoding="utf-8")
        assert import_dataset.read_sources(tmp_path) == {}

    def test_the_registry_records_what_a_data_card_needs(self, tmp_path):
        import_dataset.write_sources(
            tmp_path,
            {
                "warp-": {
                    "mapping": "WaRP",
                    "source": "a/data.yaml",
                    "images": 2974,
                    "instances": {"plastic bottle": 8285},
                }
            },
        )
        entry = import_dataset.read_sources(tmp_path)["warp-"]
        assert {"mapping", "source", "images", "instances"} <= set(entry)

    def test_prefixes_identify_the_owning_source(self, tmp_path):
        # The check that makes a merge safe: a prefix belongs to one source.
        import_dataset.write_sources(
            tmp_path, {"plant-a-": {"mapping": "Plant A"}, "plant-b-": {"mapping": "Plant B"}}
        )
        sources = import_dataset.read_sources(tmp_path)
        assert sources["plant-a-"]["mapping"] != sources["plant-b-"]["mapping"]


class TestSplitLayouts:
    """YOLO does not mandate a directory layout; the descriptor declares one.

    WaRP is images/<split>/, SortWaste is <split>/images/, and both are
    valid. Assuming the first meant importing SortWaste found nothing, said
    "imported 0 image(s)", wrote a data.yaml, and printed the next command to
    run — a dataset that looks real and contains nothing.
    """

    @staticmethod
    def _descriptor(tmp_path, layout: str):
        import yaml

        paths = {
            "images_first": {"train": "images/train", "val": "images/val"},
            "split_first": {"train": "train/images", "val": "val/images"},
        }[layout]
        for split, rel in paths.items():
            (tmp_path / rel).mkdir(parents=True)
            labels = (
                Path(str(tmp_path / rel).replace("images", "labels", 1))
                if layout == "images_first"
                else tmp_path / split / "labels"
            )
            labels.mkdir(parents=True, exist_ok=True)
        descriptor = tmp_path / "data.yaml"
        descriptor.write_text(
            yaml.safe_dump({"path": ".", **paths, "names": {0: "x"}}, sort_keys=False),
            encoding="utf-8",
        )
        return descriptor

    def test_an_images_first_layout_resolves(self, tmp_path):
        from recyclevision.external import label_dir_for, read_split_images_dir

        descriptor = self._descriptor(tmp_path, "images_first")
        images = read_split_images_dir(descriptor, "train")
        assert images == tmp_path / "images" / "train"
        assert label_dir_for(images) == tmp_path / "labels" / "train"

    def test_a_split_first_layout_resolves(self, tmp_path):
        from recyclevision.external import label_dir_for, read_split_images_dir

        descriptor = self._descriptor(tmp_path, "split_first")
        images = read_split_images_dir(descriptor, "train")
        assert images == tmp_path / "train" / "images"
        assert label_dir_for(images) == tmp_path / "train" / "labels"

    def test_a_test_split_is_read_not_ignored(self, tmp_path):
        import yaml

        from recyclevision.external import read_split_images_dir

        (tmp_path / "test" / "images").mkdir(parents=True)
        descriptor = tmp_path / "data.yaml"
        descriptor.write_text(
            yaml.safe_dump({"path": ".", "test": "test/images", "names": {0: "x"}}),
            encoding="utf-8",
        )
        assert read_split_images_dir(descriptor, "test") == tmp_path / "test" / "images"
