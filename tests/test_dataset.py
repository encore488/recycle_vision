"""Frame sampling, splitting, and YOLO label formatting."""

from __future__ import annotations

from collections import Counter

import pytest
import yaml
from PIL import Image

from recyclevision.dataset import (
    DatasetStats,
    balanced_selection,
    block_split,
    box_line,
    frame_difference,
    is_novel,
    polygon_line,
    prepare_tree,
    spread_sample,
    thumbnail_of,
    write_data_yaml,
)
from recyclevision.models import BoundingBox


class TestFrameSampling:
    def test_identical_frames_are_not_novel(self):
        image = Image.new("RGB", (200, 200), "grey")
        signature = thumbnail_of(image)
        assert not is_novel(signature, signature, min_diff=6.0)

    def test_the_first_frame_is_always_kept(self):
        assert is_novel(thumbnail_of(Image.new("RGB", (10, 10))), None, min_diff=6.0)

    def test_a_clearly_different_frame_is_novel(self):
        black = thumbnail_of(Image.new("RGB", (200, 200), "black"))
        white = thumbnail_of(Image.new("RGB", (200, 200), "white"))
        assert frame_difference(black, white) > 200
        assert is_novel(white, black, min_diff=6.0)

    def test_zero_threshold_keeps_everything(self):
        signature = thumbnail_of(Image.new("RGB", (50, 50), "grey"))
        assert is_novel(signature, signature, min_diff=0.0)

    def test_mismatched_signatures_are_treated_as_different(self):
        assert frame_difference([1.0, 2.0], []) == 255.0


class TestBlockSplit:
    def test_validation_comes_from_the_end_not_at_random(self):
        """Random splitting of video frames leaks near-duplicates across sets."""
        train, val = block_split(10, 0.2)
        assert train == list(range(8))
        assert val == [8, 9]
        assert max(train) < min(val)

    def test_sets_are_disjoint_and_complete(self):
        train, val = block_split(37, 0.25)
        assert set(train) & set(val) == set()
        assert sorted(train + val) == list(range(37))

    def test_never_takes_every_frame_for_validation(self):
        train, val = block_split(2, 0.9)
        assert train and val

    def test_single_frame_goes_to_train(self):
        train, val = block_split(1, 0.2)
        assert train == [0] and val == []

    def test_no_frames_is_not_an_error(self):
        assert block_split(0, 0.2) == ([], [])

    def test_zero_fraction_keeps_everything_for_training(self):
        train, val = block_split(10, 0.0)
        assert len(train) == 10 and val == []


class TestLabelFormatting:
    def test_box_line_is_normalised_centre_and_size(self):
        line = box_line(3, BoundingBox(100, 50, 300, 250), width=400, height=400)
        parts = line.split()
        assert parts[0] == "3"
        assert [float(v) for v in parts[1:]] == pytest.approx([0.5, 0.375, 0.5, 0.5])

    def test_polygon_line_normalises_every_point(self):
        line = polygon_line(1, [(0, 0), (100, 0), (100, 100)], width=200, height=200)
        parts = line.split()
        assert parts[0] == "1"
        assert [float(v) for v in parts[1:]] == pytest.approx([0, 0, 0.5, 0, 0.5, 0.5])

    def test_degenerate_polygon_is_dropped_not_written(self):
        """YOLO rejects a label file containing a polygon with under 3 points."""
        assert polygon_line(0, [(1, 1), (2, 2)], 10, 10) == ""

    def test_coordinates_are_clamped_into_range(self):
        """A box may extend past the frame edge; a label may not."""
        line = box_line(0, BoundingBox(-50, -50, 500, 500), width=100, height=100)
        assert all(0.0 <= float(v) <= 1.0 for v in line.split()[1:])

    def test_polygon_coordinates_are_clamped(self):
        line = polygon_line(0, [(-5, -5), (200, 0), (100, 200)], width=100, height=100)
        assert all(0.0 <= float(v) <= 1.0 for v in line.split()[1:])


class TestDatasetScaffold:
    def test_writes_the_layout_ultralytics_expects(self, tmp_path):
        prepare_tree(tmp_path)
        for kind in ("images", "labels"):
            for split in ("train", "val"):
                assert (tmp_path / kind / split).is_dir()

    def test_data_yaml_names_classes_by_index(self, tmp_path):
        path = write_data_yaml(tmp_path, ["metal can", "glass jar"])
        parsed = yaml.safe_load(path.read_text())
        assert parsed["names"] == {0: "metal can", 1: "glass jar"}
        assert parsed["train"] == "images/train"


class TestStats:
    def test_warns_about_classes_too_thin_to_learn(self):
        stats = DatasetStats(images=10, instances=210, per_class={"metal can": 200, "foil": 10})
        report = stats.report()
        assert "too few to learn" in report
        assert "foil" in report
        assert "metal can" not in report.split("too few to learn")[1]

    def test_no_warning_when_every_class_is_well_covered(self):
        stats = DatasetStats(images=10, instances=200, per_class={"metal can": 100, "jar": 100})
        assert "too few" not in stats.report()

    def test_instances_per_image_survives_an_empty_dataset(self):
        assert DatasetStats().instances_per_image == 0.0


class TestLabelCaches:
    """Ultralytics caches parsed labels and keys the cache on total byte size
    plus file paths -- never on contents.

    Re-importing after a vocabulary change rewrites class indices in place. If
    the new index has as many digits as the old one, the total size is
    unchanged, the key still matches, and the run silently trains on the
    previous labels. That is exactly what `paper cup` 14 -> `beverage carton`
    15 does, and it reports nothing wrong.
    """

    def test_preparing_a_tree_clears_a_stale_cache(self, tmp_path):
        labels = tmp_path / "labels"
        labels.mkdir()
        cache = labels / "train.cache"
        cache.write_bytes(b"stale")

        removed = prepare_tree(tmp_path)

        assert not cache.exists()
        assert removed == [cache]

    def test_nested_caches_are_found_too(self, tmp_path):
        deep = tmp_path / "labels" / "train"
        deep.mkdir(parents=True)
        (deep / "sub.cache").write_bytes(b"stale")

        assert len(prepare_tree(tmp_path)) == 1

    def test_a_clean_tree_reports_nothing_removed(self, tmp_path):
        assert prepare_tree(tmp_path) == []

    def test_labels_themselves_survive(self, tmp_path):
        labels = tmp_path / "labels" / "train"
        labels.mkdir(parents=True)
        kept = labels / "frame.txt"
        kept.write_text("15 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (labels.parent / "train.cache").write_bytes(b"stale")

        prepare_tree(tmp_path)

        assert kept.read_text(encoding="utf-8") == "15 0.5 0.5 0.2 0.2\n"


class TestSpreadSample:
    """Selection for the hand-labelled evaluation set.

    Not a uniform random draw: these listings sort by filename, which for
    video-derived data means they sort by time. A clumped sample is a handful
    of moments rather than a survey, and an eval set built that way flatters
    or damns a model for reasons nobody can reconstruct.
    """

    def test_the_same_seed_gives_the_same_selection(self):
        items = list(range(500))
        assert spread_sample(items, 20, seed=7) == spread_sample(items, 20, seed=7)

    def test_a_different_seed_gives_a_different_selection(self):
        items = list(range(500))
        assert spread_sample(items, 20, seed=1) != spread_sample(items, 20, seed=2)

    def test_the_sample_spans_the_whole_sequence(self):
        picked = spread_sample(list(range(1000)), 10, seed=0)
        assert min(picked) < 150, "nothing from the start of the dataset"
        assert max(picked) > 850, "nothing from the end of the dataset"

    def test_no_item_is_picked_twice(self):
        picked = spread_sample(list(range(200)), 40, seed=3)
        assert len(picked) == len(set(picked))

    def test_asking_for_more_than_exists_returns_everything(self):
        assert spread_sample([1, 2, 3], 10) == [1, 2, 3]

    def test_asking_for_none_returns_none(self):
        assert spread_sample([1, 2, 3], 0) == []

    def test_an_empty_sequence_is_not_an_error(self):
        assert spread_sample([], 5) == []

    def test_the_requested_count_is_honoured(self):
        assert len(spread_sample(list(range(97)), 13, seed=0)) == 13


class TestBalancedSelection:
    """A random subset of an imbalanced pool keeps the imbalance.

    The real pool runs 99:1 — plastic bottle 42.5% of instances, glass bottle
    0.4% — so a random cut leaves the rare classes in numbers too small to
    learn, while most of the images taken teach the majority class something
    it already knows.
    """

    @staticmethod
    def _pool():
        items = [(f"bottle_{i}", Counter({"bottle": 5})) for i in range(1000)]
        items += [(f"glass_{i}", Counter({"glass": 2})) for i in range(20)]
        items += [(f"can_{i}", Counter({"can": 1})) for i in range(60)]
        return items

    def test_a_rare_class_survives_the_cut(self, tmp_path):
        chosen = balanced_selection(self._pool(), 150)
        assert sum(1 for k in chosen if k.startswith("glass")) == 20

    def test_a_random_cut_would_have_lost_it(self):
        # The comparison that justifies the method existing.
        random_cut = [k for k, _c in spread_sample(self._pool(), 150, 0)]
        assert sum(1 for k in random_cut if k.startswith("glass")) < 10

    def test_the_budget_is_respected(self):
        assert len(balanced_selection(self._pool(), 150)) == 150

    def test_no_image_is_chosen_twice(self):
        chosen = balanced_selection(self._pool(), 300)
        assert len(chosen) == len(set(chosen))

    def test_a_budget_larger_than_the_pool_returns_everything(self):
        pool = self._pool()
        assert len(balanced_selection(pool, 99999)) == len(pool)

    def test_an_empty_pool_is_not_an_error(self):
        assert balanced_selection([], 10) == []

    def test_a_zero_budget_returns_nothing(self):
        assert balanced_selection(self._pool(), 0) == []

    def test_images_with_no_labels_do_not_crash_it(self):
        items = [("empty_1", Counter()), ("a", Counter({"x": 1}))]
        assert len(balanced_selection(items, 1)) == 1

    def test_the_selection_is_reproducible(self):
        pool = self._pool()
        assert balanced_selection(pool, 100, seed=4) == balanced_selection(pool, 100, seed=4)

    def test_an_image_carrying_several_classes_advances_all_of_them(self):
        # Why exact balance is impossible: one frame can serve two quotas.
        items = [("both", Counter({"rare": 1, "common": 9}))]
        items += [(f"c_{i}", Counter({"common": 5})) for i in range(50)]
        assert "both" in balanced_selection(items, 5)
