"""QA tooling: contact sheets, verdict stubs, and scoring."""

from __future__ import annotations

import json

import pytest

from recyclevision.detector import StubDetector
from recyclevision.pipeline import SortingPipeline
from recyclevision.qa import UNGRADED, VERDICTS, contact_sheet, main, verdict_stub
from tests.conftest import make_detection


def sort(policy, labels, image):
    detections = [make_detection(label, conf) for label, conf in labels]
    return SortingPipeline(StubDetector(detections), policy).sort(image)


class TestContactSheet:
    def test_builds_a_tile_per_detection(self, policy, image):
        result = sort(policy, [("bottle", 0.9), ("pizza", 0.8)], image)
        sheet = contact_sheet(image, result)
        assert sheet is not None
        assert sheet.width > 0 and sheet.height > 0

    def test_returns_none_when_there_is_nothing_to_grade(self, policy, image):
        assert contact_sheet(image, sort(policy, [], image)) is None

    @pytest.mark.parametrize("box", [(0, 0, 20, 20), (620, 460, 640, 480), (0, 0, 640, 480)])
    def test_handles_detections_flush_against_the_image_edge(self, policy, image, box):
        """Crop padding must not run off the image in either direction."""
        detections = [make_detection("bottle", 0.9, box=box)]
        result = SortingPipeline(StubDetector(detections), policy).sort(image)
        assert contact_sheet(image, result) is not None

    def test_wraps_onto_multiple_rows(self, policy, image):
        few = contact_sheet(image, sort(policy, [("bottle", 0.9)], image))
        many = contact_sheet(
            image, sort(policy, [("bottle", 0.9 - i / 20) for i in range(7)], image)
        )
        assert few is not None and many is not None
        assert many.height > few.height


class TestVerdictStub:
    def test_one_ungraded_record_per_item(self, policy, image):
        result = sort(policy, [("bottle", 0.9), ("pizza", 0.8)], image)
        records = verdict_stub(result, "img.jpg")
        assert len(records) == 2
        assert all(r["verdict"] == UNGRADED for r in records)

    def test_records_what_a_grader_needs(self, policy, image):
        record = verdict_stub(sort(policy, [("bottle", 0.9)], image), "img.jpg")[0]
        assert record["detected_as"] == "bottle"
        assert record["routed_to"] == "recycling"
        assert record["source"] == "img.jpg"
        assert len(record["box"]) == 4


class TestScoring:
    def _write(self, tmp_path, verdicts):
        tmp_path.mkdir(parents=True, exist_ok=True)
        path = tmp_path / "verdicts.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "source": "i.jpg",
                        "index": i,
                        "detected_as": "cup",
                        "confidence": 0.5,
                        "routed_to": "landfill",
                        "policy_certainty": "low",
                        "box": [0, 0, 1, 1],
                        "verdict": v,
                        "actually_is": "steel can",
                        "should_be_bin": "recycling",
                    }
                    for i, v in enumerate(verdicts)
                ]
            )
        )
        return path

    def test_scores_a_graded_file(self, tmp_path, capsys):
        path = self._write(tmp_path, ["correct", "wrong_bin", "false_positive"])
        assert main(["score", str(path)]) == 0

        out = capsys.readouterr().out
        assert "Graded 3 detection(s)" in out
        # 2 of 3 boxes are on a real object; 1 of those 2 reaches the right bin.
        assert "67%" in out
        assert "50%" in out

    def test_reports_the_misreadings_behind_wrong_bins(self, tmp_path, capsys):
        path = self._write(tmp_path, ["wrong_bin", "wrong_bin"])
        main(["score", str(path)])
        out = capsys.readouterr().out
        assert "'cup' is really 'steel can'" in out
        assert "x2" in out

    def test_ungraded_file_is_an_error_not_a_crash(self, tmp_path, capsys):
        path = self._write(tmp_path, [UNGRADED, UNGRADED])
        assert main(["score", str(path)]) == 1
        assert "Nothing graded yet" in capsys.readouterr().out

    def test_unsure_is_excluded_from_the_rates(self, tmp_path, capsys):
        """An unsure grade is evidence of nothing and must not pad a denominator."""
        both = self._write(tmp_path / "a", ["correct", "false_positive"])
        (tmp_path / "a").mkdir(exist_ok=True)
        main(["score", str(both)])
        without = capsys.readouterr().out

        plus = self._write(tmp_path / "b", ["correct", "false_positive", "unsure"])
        (tmp_path / "b").mkdir(exist_ok=True)
        main(["score", str(plus)])
        with_unsure = capsys.readouterr().out

        assert "detection precision    50%" in without
        assert "detection precision    50%" in with_unsure
        assert "1 unsure detection(s) excluded" in with_unsure

    def test_all_unsure_scores_nothing(self, tmp_path, capsys):
        path = self._write(tmp_path, ["unsure", "unsure"])
        assert main(["score", str(path)]) == 0
        assert "nothing to score" in capsys.readouterr().out

    def test_partially_graded_file_says_how_many_remain(self, tmp_path, capsys):
        path = self._write(tmp_path, ["correct", UNGRADED])
        assert main(["score", str(path)]) == 0
        assert "1 still ungraded" in capsys.readouterr().out


class TestVocabulary:
    def test_verdicts_separate_detector_failures_from_policy_failures(self):
        """The taxonomy is the point: the two have different fixes."""
        assert "false_positive" in VERDICTS  # detector put a box on nothing
        assert "wrong_bin" in VERDICTS  # object is real, destination is wrong
        assert UNGRADED not in VERDICTS


class TestCli:
    def test_missing_image_is_a_clean_error(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["sheet", str(tmp_path / "nope.jpg")])


class TestVocabularyDefault:
    def test_default_vocabulary_has_embeddings_built(self):
        """The shipped default must be usable straight from a clone."""
        from recyclevision.vocabulary import DEFAULT_VOCAB, Vocabulary, discover

        assert DEFAULT_VOCAB in discover(), (
            f"{DEFAULT_VOCAB.name} has no cached embeddings; run scripts/build_vocab_embeddings.py"
        )
        assert Vocabulary.load(DEFAULT_VOCAB).classes
