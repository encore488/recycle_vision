"""Checks that run before a night of training is spent.

Each one corresponds to a failure this project had, or to one that is cheap to
cause and expensive to notice. The severity split matters as much as the
detection: blocking on a warning teaches people to ignore the tool.
"""

from __future__ import annotations

from pathlib import Path

from recyclevision.preflight import check_dataset

NAMES = ["metal can", "plastic bottle", "plastic bag"]


def build(root: Path, splits: dict[str, dict[str, str]]) -> Path:
    """A dataset from {split: {stem: label text}}. An empty text means no label file."""
    for split, items in splits.items():
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem, text in items.items():
            (root / "images" / split / f"{stem}.jpg").write_bytes(b"\xff\xd8")
            if text:
                (root / "labels" / split / f"{stem}.txt").write_text(text, encoding="utf-8")
    return root


BOTH = "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.2 0.2\n"


def healthy(root: Path) -> Path:
    return build(
        root,
        {
            "train": {f"t{i}": BOTH for i in range(4)},
            "val": {f"v{i}": BOTH for i in range(2)},
        },
    )


def levels(result, check: str) -> list[str]:
    return [f.level for f in result.findings if f.check == check]


class TestHealthyDataset:
    def test_a_sound_dataset_blocks_nothing(self, tmp_path):
        result = check_dataset(healthy(tmp_path), NAMES[:2])

        assert result.blocking == []

    def test_it_reports_what_it_counted(self, tmp_path):
        result = check_dataset(healthy(tmp_path), NAMES[:2])

        assert "train: 4 images, 8 instances" in result.report()


class TestBlockingFindings:
    def test_an_image_with_no_label_blocks(self, tmp_path):
        """Ultralytics reads it as an image of nothing — the WaRP density trap."""
        root = healthy(tmp_path)
        (root / "images" / "train" / "extra.jpg").write_bytes(b"\xff\xd8")
        result = check_dataset(root, NAMES[:2])

        assert "block" in levels(result, "train")
        assert "background" in result.report()

    def test_the_same_image_in_both_splits_blocks(self, tmp_path):
        root = healthy(tmp_path)
        (root / "images" / "val" / "t0.jpg").write_bytes(b"\xff\xd8")
        (root / "labels" / "val" / "t0.txt").write_text(BOTH, encoding="utf-8")
        result = check_dataset(root, NAMES[:2])

        assert levels(result, "leakage") == ["block"]
        assert "memorisation" in result.report()

    def test_a_class_only_in_val_blocks(self, tmp_path):
        """Scored as a failure to generalise when it is a failure to have data."""
        root = healthy(tmp_path)
        (root / "labels" / "val" / "v0.txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        result = check_dataset(root, NAMES)

        assert levels(result, "coverage")[0] == "block"
        assert "plastic bag" in result.report()

    def test_a_nan_producing_label_blocks(self, tmp_path):
        root = healthy(tmp_path)
        (root / "labels" / "train" / "t0.txt").write_text("0 0.5 0.5 0.0 0.2\n", encoding="utf-8")
        result = check_dataset(root, NAMES[:2])

        assert "block" in levels(result, "train")
        assert "NaN" in result.report()

    def test_a_stale_label_cache_blocks(self, tmp_path):
        """Trap 2: the cache is keyed on byte size and paths, never contents."""
        root = healthy(tmp_path)
        cache = root / "labels" / "train.cache"
        cache.write_bytes(b"stale")
        import os

        old = cache.stat().st_mtime - 3600
        os.utime(cache, (old, old))
        result = check_dataset(root, NAMES[:2])

        assert levels(result, "cache") == ["block"]
        assert "OLD labels" in result.report()

    def test_a_fresh_cache_does_not_block(self, tmp_path):
        root = healthy(tmp_path)
        (root / "labels" / "train.cache").write_bytes(b"fresh")
        result = check_dataset(root, NAMES[:2])

        assert levels(result, "cache") == []

    def test_a_missing_split_blocks(self, tmp_path):
        root = build(tmp_path, {"train": {"t0": BOTH}})
        result = check_dataset(root, NAMES[:2])

        assert "block" in levels(result, "val")

    def test_the_exit_summary_counts_only_blocking(self, tmp_path):
        result = check_dataset(healthy(tmp_path), NAMES)

        assert result.blocking == []
        assert "nothing blocking" in result.report()


class TestWarnings:
    def test_a_declared_class_with_no_data_warns_but_does_not_block(self, tmp_path):
        """10 of 17 vocabulary classes are in this state; it must not block."""
        result = check_dataset(healthy(tmp_path), NAMES)

        assert levels(result, "coverage") == ["warn"]
        assert result.blocking == []

    def test_a_sub_pixel_box_warns(self, tmp_path):
        root = healthy(tmp_path)
        (root / "labels" / "train" / "t0.txt").write_text("0 0.5 0.5 0.001 0.2\n", encoding="utf-8")
        result = check_dataset(root, NAMES[:2], imgsz=960)

        assert "warn" in levels(result, "train")
        assert "under 2px" in result.report()

    def test_a_box_large_enough_at_this_resolution_is_not_flagged(self, tmp_path):
        """The threshold is in pixels, so it has to follow imgsz."""
        root = healthy(tmp_path)
        (root / "labels" / "train" / "t0.txt").write_text("0 0.5 0.5 0.01 0.2\n", encoding="utf-8")

        assert "under 2px" not in check_dataset(root, NAMES[:2], imgsz=960).report()
        assert "under 2px" in check_dataset(root, NAMES[:2], imgsz=100).report()

    def test_a_dominant_class_warns(self, tmp_path):
        root = build(
            tmp_path,
            {
                "train": {f"t{i}": "0 0.5 0.5 0.2 0.2\n" for i in range(4)},
                "val": {"v0": "0 0.5 0.5 0.2 0.2\n"},
            },
        )
        result = check_dataset(root, NAMES[:1])

        assert levels(result, "balance") == ["warn"]

    def test_an_orphan_label_warns_rather_than_blocks(self, tmp_path):
        """It is ignored by training, so it is untidy rather than harmful."""
        root = healthy(tmp_path)
        (root / "labels" / "train" / "ghost.txt").write_text(BOTH, encoding="utf-8")
        result = check_dataset(root, NAMES[:2])

        assert "warn" in levels(result, "train")
        assert result.blocking == []
