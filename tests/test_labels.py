"""Labels that make a run produce NaN instead of a model.

A detector's loss divides by box area, so a zero-width box NaNs its batch.
Ultralytics then restores last.pt and re-runs the epoch, meets the same label,
and NaNs again — and its "attempt 1/3" counter resets on every successful
recovery, so the run never exhausts it. Two runs of this project were read as
fp16 overflow and then as a plateau before anyone read the labels.
"""

from __future__ import annotations

from pathlib import Path

from recyclevision.dataset import MIN_BOX_SIDE, box_line
from recyclevision.labels import scan_labels
from recyclevision.models import BoundingBox


def write(root: Path, name: str, lines: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


class TestBoxLine:
    def test_an_ordinary_box_survives(self):
        assert box_line(0, BoundingBox(10, 10, 110, 110), 640, 480).startswith("0 ")

    def test_a_zero_width_box_is_dropped(self):
        """The defect itself: w=0 divides by zero in the loss."""
        assert box_line(0, BoundingBox(10, 10, 10, 110), 640, 480) == ""

    def test_a_zero_height_box_is_dropped(self):
        assert box_line(0, BoundingBox(10, 10, 110, 10), 640, 480) == ""

    def test_a_box_that_rounds_to_zero_on_disk_is_dropped(self):
        """Six decimal places is what gets written; below that is zero."""
        sliver = BoundingBox(10, 10, 10 + 640 * MIN_BOX_SIDE / 2, 110)

        assert box_line(0, sliver, 640, 480) == ""

    def test_a_genuinely_thin_annotation_is_kept(self):
        """A 1px-tall box in a 1080px frame is 0.000926 — small, not degenerate.

        Dropping it would silently delete real supervision, which is the
        failure mode on the other side of this guard.
        """
        line = box_line(0, BoundingBox(10, 10, 110, 11), 1920, 1080)

        assert line != ""
        assert line.split()[4] == "0.000926"


class TestScanLabels:
    def test_clean_labels_report_clean(self, tmp_path):
        write(tmp_path, "a.txt", ["0 0.5 0.5 0.2 0.2", "1 0.3 0.3 0.1 0.1"])
        scan = scan_labels(tmp_path, classes=2)

        assert scan.instances == 2
        assert scan.defects == []
        assert scan.fatal == []
        assert "clean" in scan.report()

    def test_a_zero_area_box_is_fatal(self, tmp_path):
        write(tmp_path, "a.txt", ["0 0.5 0.5 0.000000 0.2"])
        scan = scan_labels(tmp_path, classes=2)

        assert [d.kind for d in scan.fatal] == ["zero-area"]
        assert "divides by zero" in scan.defects[0].detail
        assert "cannot converge" in scan.report()

    def test_nan_geometry_is_fatal(self, tmp_path):
        write(tmp_path, "a.txt", ["0 nan nan 0.2 0.2"])
        scan = scan_labels(tmp_path, classes=2)

        assert [d.kind for d in scan.fatal] == ["not-a-number"]

    def test_a_defect_names_its_file_and_line(self, tmp_path):
        write(tmp_path, "a.txt", ["0 0.5 0.5 0.2 0.2", "0 0.5 0.5 0.0 0.2"])
        scan = scan_labels(tmp_path, classes=2)

        assert scan.defects[0].line_number == 2
        assert scan.defects[0].path.name == "a.txt"

    def test_an_out_of_range_class_is_reported_but_not_fatal(self, tmp_path):
        """It trains, wrongly. NaN does not, which is a different severity."""
        write(tmp_path, "a.txt", ["7 0.5 0.5 0.2 0.2"])
        scan = scan_labels(tmp_path, classes=2)

        assert scan.counts()["bad-class"] == 1
        assert scan.fatal == []

    def test_duplicate_lines_are_counted_once_as_a_defect(self, tmp_path):
        """Ultralytics silently removes these; the scan says they were there."""
        write(tmp_path, "a.txt", ["0 0.5 0.5 0.2 0.2", "0 0.5 0.5 0.2 0.2"])
        scan = scan_labels(tmp_path, classes=2)

        assert scan.counts()["duplicate"] == 1

    def test_a_centre_outside_the_frame_is_reported(self, tmp_path):
        write(tmp_path, "a.txt", ["0 1.5 0.5 0.2 0.2"])
        scan = scan_labels(tmp_path, classes=2)

        assert scan.counts()["out-of-frame"] == 1

    def test_a_zero_area_box_outside_the_frame_reports_the_fatal_one(self, tmp_path):
        """Both are true; only one stops the run, so only one leads."""
        write(tmp_path, "a.txt", ["0 1.5 0.5 0.0 0.2"])
        scan = scan_labels(tmp_path, classes=2)

        assert [d.kind for d in scan.defects] == ["zero-area"]

    def test_a_malformed_line_does_not_stop_the_scan(self, tmp_path):
        write(tmp_path, "a.txt", ["garbage", "0 0.5 0.5 0.2 0.2"])
        scan = scan_labels(tmp_path, classes=2)

        assert scan.counts()["malformed"] == 1
        assert scan.instances == 2

    def test_blank_lines_are_not_instances(self, tmp_path):
        write(tmp_path, "a.txt", ["0 0.5 0.5 0.2 0.2", "", "   "])
        scan = scan_labels(tmp_path, classes=2)

        assert scan.instances == 1

    def test_a_missing_directory_is_not_a_crash(self, tmp_path):
        scan = scan_labels(tmp_path / "nope", classes=2)

        assert scan.files == 0
        assert scan.defects == []

    def test_every_file_under_the_tree_is_read(self, tmp_path):
        write(tmp_path, "a.txt", ["0 0.5 0.5 0.2 0.2"])
        write(tmp_path / "nested", "b.txt", ["0 0.5 0.5 0.0 0.2"])
        scan = scan_labels(tmp_path, classes=2)

        assert scan.files == 2
        assert len(scan.fatal) == 1
