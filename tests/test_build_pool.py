"""Assembling a balanced training pool from several imported datasets.

The failure this guards against is quiet: a pool whose labels no longer match
its images, or one that silently drops the rare classes it was built to
preserve. Both train without complaint.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_pool  # noqa: E402


class TestLabelCounts:
    def test_instances_are_counted_by_class_name(self, tmp_path):
        label = tmp_path / "a.txt"
        label.write_text("0 .5 .5 .2 .2\n1 .5 .5 .2 .2\n0 .5 .5 .2 .2\n", encoding="utf-8")
        assert build_pool.label_counts(label, ["can", "bottle"]) == Counter({"can": 2, "bottle": 1})

    def test_an_empty_label_file_counts_nothing(self, tmp_path):
        # A background image: legitimate, and it should not crash selection.
        label = tmp_path / "a.txt"
        label.write_text("", encoding="utf-8")
        assert build_pool.label_counts(label, ["can"]) == Counter()

    def test_an_out_of_range_index_is_ignored(self, tmp_path):
        label = tmp_path / "a.txt"
        label.write_text("0 .5 .5 .2 .2\n99 .5 .5 .2 .2\n", encoding="utf-8")
        assert build_pool.label_counts(label, ["can"]) == Counter({"can": 1})

    def test_a_truncated_line_is_ignored(self, tmp_path):
        label = tmp_path / "a.txt"
        label.write_text("0 .5 .5\n0 .5 .5 .2 .2\n", encoding="utf-8")
        assert build_pool.label_counts(label, ["can"]) == Counter({"can": 1})

    def test_a_missing_file_is_not_an_error(self, tmp_path):
        assert build_pool.label_counts(tmp_path / "absent.txt", ["can"]) == Counter()


class TestCollect:
    @staticmethod
    def _dataset(root: Path, split: str, count: int) -> Path:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(count):
            (root / "images" / split / f"f{i}.jpg").write_bytes(b"\xff\xd8")
            (root / "labels" / split / f"f{i}.txt").write_text("0 .5 .5 .2 .2\n", encoding="utf-8")
        return root

    def test_labelled_images_are_collected_with_their_contents(self, tmp_path):
        found = build_pool.collect(self._dataset(tmp_path, "train", 3), "train", ["can"])
        assert len(found) == 3
        assert all(counts == Counter({"can": 1}) for _image, counts in found)

    def test_an_image_without_a_label_is_skipped(self, tmp_path):
        root = self._dataset(tmp_path, "train", 3)
        (root / "labels" / "train" / "f0.txt").unlink()
        assert len(build_pool.collect(root, "train", ["can"])) == 2

    def test_a_missing_split_yields_nothing(self, tmp_path):
        assert build_pool.collect(tmp_path, "val", ["can"]) == []


class TestPlace:
    def test_the_label_travels_with_its_image(self, tmp_path):
        source = tmp_path / "src"
        (source / "images" / "train").mkdir(parents=True)
        (source / "labels" / "train").mkdir(parents=True)
        image = source / "images" / "train" / "f.jpg"
        image.write_bytes(b"\xff\xd8")
        label = source / "labels" / "train" / "f.txt"
        label.write_text("2 .5 .5 .2 .2\n", encoding="utf-8")

        out = tmp_path / "pool"
        for split in ("train", "val"):
            (out / "images" / split).mkdir(parents=True)
            (out / "labels" / split).mkdir(parents=True)
        build_pool.place(image, label, out, "train")

        assert (out / "images" / "train" / "f.jpg").is_file()
        assert (out / "labels" / "train" / "f.txt").read_text() == "2 .5 .5 .2 .2\n"

    def test_placing_twice_does_not_fail(self, tmp_path):
        # Rebuilding a pool over an existing one must be safe.
        source = tmp_path / "src"
        source.mkdir()
        image = source / "f.jpg"
        image.write_bytes(b"\xff\xd8")
        label = source / "f.txt"
        label.write_text("0 .5 .5 .2 .2\n", encoding="utf-8")
        out = tmp_path / "pool"
        (out / "images" / "train").mkdir(parents=True)
        (out / "labels" / "train").mkdir(parents=True)

        build_pool.place(image, label, out, "train")
        build_pool.place(image, label, out, "train")
        assert (out / "images" / "train" / "f.jpg").is_file()
