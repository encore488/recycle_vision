"""Warning when a holdout is too sparsely annotated to measure precision.

WaRP labels 3.5 objects per image in frames holding dozens. A model scored
there is penalised for finding objects the dataset never labelled: measured
earlier in this project, its true precision lay somewhere between a raw 4.1%
and a fair 56.0%, and no amount of care narrows that.

A pool model scored 0.104 precision on it, which reads as a catastrophe and
is mostly an artefact. The number needs the caveat attached to it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import evaluate  # noqa: E402


def _dataset(root: Path, images: int, per_image: int) -> Path:
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    for i in range(images):
        (root / "images" / "val" / f"f{i}.jpg").write_bytes(b"\xff\xd8")
        (root / "labels" / "val" / f"f{i}.txt").write_text(
            "\n".join("0 .5 .5 .2 .2" for _ in range(per_image)) + "\n", encoding="utf-8"
        )
    descriptor = root / "data.yaml"
    descriptor.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames: [x]\n", encoding="utf-8"
    )
    return descriptor


def test_density_is_instances_per_image(tmp_path):
    assert evaluate.annotation_density(_dataset(tmp_path, 10, 4)) == 4.0


def test_a_warp_like_density_falls_below_the_threshold(tmp_path):
    assert evaluate.annotation_density(_dataset(tmp_path, 10, 3)) < evaluate.SPARSE_BELOW


def test_a_sortwaste_like_density_does_not(tmp_path):
    assert evaluate.annotation_density(_dataset(tmp_path, 10, 16)) > evaluate.SPARSE_BELOW


def test_empty_label_files_count_as_zero_not_as_missing(tmp_path):
    # A background image is a real annotation: "nothing here".
    root = tmp_path / "ds"
    descriptor = _dataset(root, 4, 8)
    (root / "labels" / "val" / "f0.txt").write_text("", encoding="utf-8")
    assert evaluate.annotation_density(descriptor) == 6.0


def test_a_truncated_line_is_not_counted(tmp_path):
    root = tmp_path / "ds"
    descriptor = _dataset(root, 1, 2)
    (root / "labels" / "val" / "f0.txt").write_text("0 .5 .5\n0 .5 .5 .2 .2\n", encoding="utf-8")
    assert evaluate.annotation_density(descriptor) == 1.0


def test_a_dataset_with_no_val_labels_reports_nothing(tmp_path):
    descriptor = tmp_path / "data.yaml"
    descriptor.write_text("path: .\nval: images/val\nnames: [x]\n", encoding="utf-8")
    assert evaluate.annotation_density(descriptor) is None


def test_an_unreadable_descriptor_reports_nothing(tmp_path):
    assert evaluate.annotation_density(tmp_path / "absent.yaml") is None
