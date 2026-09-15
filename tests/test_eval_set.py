"""Reserving images for a hand-labelled evaluation set, without leaking them.

Leakage is the failure this file exists to prevent. Training on an image that
is also in the evaluation set makes every number reported afterwards
meaningless, and nothing in any output looks wrong when it happens — the
metrics simply come out better. It has to be caught here or not at all.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_eval_set  # noqa: E402


def _imported(root: Path, prefix: str, per_split: int = 5) -> Path:
    """A minimal stand-in for an imported dataset, same names in both splits."""
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(per_split):
            (root / "images" / split / f"{prefix}{i:03d}.jpg").write_bytes(b"\xff\xd8")
            (root / "labels" / split / f"{prefix}{i:03d}.txt").write_text("0 .5 .5 .2 .2\n")
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames: [x]\n")
    return root


class TestCollect:
    def test_both_splits_are_candidates(self, tmp_path):
        found = build_eval_set.collect(_imported(tmp_path, "a-", per_split=3))
        assert len(found) == 6
        assert {split for _image, split in found} == {"train", "val"}

    def test_an_image_with_no_label_is_not_a_candidate(self, tmp_path):
        root = _imported(tmp_path, "a-", per_split=2)
        (root / "labels" / "train" / "a-000.txt").unlink()
        assert len(build_eval_set.collect(root)) == 3

    def test_a_missing_dataset_yields_nothing(self, tmp_path):
        assert build_eval_set.collect(tmp_path / "absent") == []


class TestEvalName:
    """import_dataset prefixes by SOURCE, not by split, so one dataset can hold
    the same basename in train and val. Flattening them into one evaluation
    directory silently dropped one of the pair and shrank the set.
    """

    def test_an_unused_name_is_kept_as_is(self):
        assert build_eval_set.eval_name(Path("a/b/frame.jpg"), "train", set()) == "frame.jpg"

    def test_a_clash_is_disambiguated_by_split(self):
        name = build_eval_set.eval_name(Path("a/b/frame.jpg"), "val", {"frame.jpg"})
        assert name != "frame.jpg"
        assert "val" in name and name.endswith(".jpg")

    def test_disambiguation_survives_both_splits(self):
        taken: set[str] = set()
        first = build_eval_set.eval_name(Path("x/frame.jpg"), "train", taken)
        taken.add(first)
        second = build_eval_set.eval_name(Path("y/frame.jpg"), "val", taken)
        assert first != second


class TestNoLeakage:
    def test_the_manifest_names_every_reserved_image(self, tmp_path):
        source = _imported(tmp_path / "ds", "one-", per_split=6)
        out = tmp_path / "gold"
        build_eval_set.main(
            ["--from", str(source), "--out", str(out), "--per-source", "4", "--seed", "0"]
        )
        manifest = json.loads((out / "manifest.json").read_text())
        on_disk = {p.name for p in (out / "images" / "val").iterdir()}
        assert set(manifest["images"]) == on_disk

    def test_reserved_images_are_identified_by_output_name(self, tmp_path):
        # The identity that import_dataset matches on. If this drifts, the
        # exclusion silently stops excluding and every later metric is wrong.
        source = _imported(tmp_path / "ds", "one-", per_split=6)
        out = tmp_path / "gold"
        build_eval_set.main(["--from", str(source), "--out", str(out), "--per-source", "3"])
        manifest = json.loads((out / "manifest.json").read_text())
        for entry in manifest["images"].values():
            assert Path(entry["source_image"]).name.startswith("one-")

    def test_no_labels_are_written(self, tmp_path):
        # An empty .txt means "this image contains nothing" in YOLO, which
        # would be a lie about an image nobody has looked at yet.
        source = _imported(tmp_path / "ds", "one-", per_split=6)
        out = tmp_path / "gold"
        build_eval_set.main(["--from", str(source), "--out", str(out), "--per-source", "3"])
        assert list((out / "labels" / "val").iterdir()) == []

    def test_source_labels_are_kept_aside_for_later_comparison(self, tmp_path):
        source = _imported(tmp_path / "ds", "one-", per_split=6)
        out = tmp_path / "gold"
        build_eval_set.main(["--from", str(source), "--out", str(out), "--per-source", "3"])
        assert len(list((out / "reference").iterdir())) == 3

    def test_the_selection_is_reproducible(self, tmp_path):
        source = _imported(tmp_path / "ds", "one-", per_split=8)
        picked = []
        for run in ("a", "b"):
            out = tmp_path / run
            build_eval_set.main(
                ["--from", str(source), "--out", str(out), "--per-source", "5", "--seed", "42"]
            )
            picked.append(sorted(json.loads((out / "manifest.json").read_text())["images"]))
        assert picked[0] == picked[1]

    def test_the_manifest_says_it_is_not_for_training(self, tmp_path):
        source = _imported(tmp_path / "ds", "one-", per_split=4)
        out = tmp_path / "gold"
        build_eval_set.main(["--from", str(source), "--out", str(out), "--per-source", "2"])
        manifest = json.loads((out / "manifest.json").read_text())
        assert "never train" in manifest["purpose"].lower()
