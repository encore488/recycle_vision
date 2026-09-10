"""Routing policy: parsing, validation, and the rules themselves."""

from __future__ import annotations

import copy

import pytest

from recyclevision.models import Certainty
from recyclevision.policy import PolicyError, RoutingPolicy
from tests.conftest import MINIMAL_POLICY, make_detection


class TestLoading:
    def test_loads_shipped_policy(self, policy):
        assert policy.name
        assert {b.key for b in policy.bins} >= {"recycling", "compost", "landfill"}

    def test_landfill_is_the_only_undiverted_bin(self, policy):
        undiverted = {b.key for b in policy.bins if not b.diverted}
        assert undiverted == {"landfill"}

    def test_missing_file_is_a_policy_error(self, tmp_path):
        with pytest.raises(PolicyError, match="no policy file"):
            RoutingPolicy.load(tmp_path / "absent.yaml")

    def test_invalid_yaml_is_a_policy_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("rules: [unclosed\n")
        with pytest.raises(PolicyError, match="not valid YAML"):
            RoutingPolicy.load(bad)

    def test_rule_pointing_at_undefined_bin_is_rejected(self):
        raw = copy.deepcopy(MINIMAL_POLICY)
        raw["rules"]["bottle"]["bin"] = "nonexistent"
        with pytest.raises(PolicyError, match="undefined bins"):
            RoutingPolicy.from_dict(raw)

    def test_rule_without_a_bin_is_rejected(self):
        raw = copy.deepcopy(MINIMAL_POLICY)
        raw["rules"]["bottle"] = {"item": "Bottle"}
        with pytest.raises(PolicyError, match="missing a 'bin'"):
            RoutingPolicy.from_dict(raw)

    def test_unknown_certainty_is_rejected(self):
        raw = copy.deepcopy(MINIMAL_POLICY)
        raw["rules"]["bottle"]["certainty"] = "maybe"
        with pytest.raises(PolicyError, match="certainty"):
            RoutingPolicy.from_dict(raw)

    def test_empty_sections_are_rejected(self):
        with pytest.raises(PolicyError, match="bins"):
            RoutingPolicy.from_dict({"rules": {"a": {"bin": "b"}}})


class TestRouting:
    def test_routes_a_known_class(self, policy):
        item = policy.route(make_detection("bottle"))
        assert item is not None
        assert item.bin.key == "recycling"

    def test_label_matching_ignores_case_and_separators(self, policy):
        for variant in ("wine glass", "Wine Glass", "WINE_GLASS", "  wine-glass  "):
            item = policy.route(make_detection(variant))
            assert item is not None, variant
            assert item.bin.key == "landfill", variant

    def test_unlisted_class_is_ignored_by_default(self, policy):
        # COCO detects people and cars constantly; counting them as waste
        # would corrupt every metric on the dashboard.
        for label in ("person", "car", "dog", "couch"):
            assert policy.route(make_detection(label)) is None, label

    def test_non_ignore_default_routes_unlisted_classes(self):
        raw = copy.deepcopy(MINIMAL_POLICY)
        raw["default"] = "landfill"
        routed = RoutingPolicy.from_dict(raw).route(make_detection("mystery object"))
        assert routed is not None
        assert routed.bin.key == "landfill"
        # An unrecognised item is exactly what a human should double-check.
        assert routed.needs_review

    def test_falls_back_to_detector_label_when_rule_has_no_item_name(self, policy):
        item = policy.route(make_detection("toothbrush"))
        assert item is not None
        assert item.label

    def test_preserves_detection_geometry(self, policy):
        detection = make_detection("bottle", box=(1, 2, 30, 40))
        item = policy.route(detection)
        assert item is not None
        assert item.detection.box.area == pytest.approx(29 * 38)

    def test_covers_reports_rule_presence(self, policy):
        assert policy.covers("bottle")
        assert policy.covers("WINE_GLASS")
        assert not policy.covers("aardvark")

    def test_unknown_bin_lookup_raises(self, policy):
        with pytest.raises(PolicyError, match="no bin"):
            policy.bin("no-such-bin")


class TestShippedRules:
    """Guards on the domain knowledge itself.

    These are the counter-intuitive calls that a material-first design gets
    wrong. If someone "simplifies" the policy by routing on material, these
    fail -- which is the point.
    """

    @pytest.mark.parametrize(
        "label,expected_bin,why",
        [
            ("wine glass", "landfill", "drinking glass contaminates container glass"),
            ("vase", "landfill", "ceramics contaminate container glass"),
            ("cup", "landfill", "disposable cups are plastic-lined"),
            ("pizza", "compost", "food waste is organic, not recyclable"),
            ("bottle", "recycling", "the ordinary case still has to work"),
            ("book", "recycling", "paper"),
            ("laptop", "special", "e-waste is never kerbside"),
            ("cell phone", "special", "lithium batteries start fires in trucks"),
            ("scissors", "special", "sharps injure sorting-line workers"),
            ("backpack", "reuse", "textiles tangle sorting machinery"),
        ],
    )
    def test_destination(self, policy, label, expected_bin, why):
        item = policy.route(make_detection(label))
        assert item is not None, f"{label} should be routed"
        assert item.bin.key == expected_bin, why

    def test_glass_items_do_not_share_a_bin(self, policy):
        """Material is an attribute, not a destination -- the core thesis."""
        bottle = policy.route(make_detection("bottle"))
        glassware = policy.route(make_detection("wine glass"))
        assert bottle is not None and glassware is not None
        assert bottle.bin.key != glassware.bin.key

    @pytest.mark.parametrize("label", ["cup", "bowl", "fork", "knife", "spoon", "potted plant"])
    def test_genuinely_ambiguous_items_are_flagged_for_review(self, policy, label):
        item = policy.route(make_detection(label))
        assert item is not None
        assert item.certainty is Certainty.LOW
        assert item.rationale, "a low-certainty route must explain itself"

    def test_surprising_routes_carry_a_rationale(self, policy):
        for label in ("wine glass", "cup", "scissors"):
            item = policy.route(make_detection(label))
            assert item is not None and item.rationale, label

    def test_every_rule_names_a_handling_step_or_rationale(self, policy):
        """A bin with no explanation is a worse answer than no bin at all."""
        for label in ("bottle", "wine glass", "pizza", "laptop", "backpack"):
            item = policy.route(make_detection(label))
            assert item is not None
            assert item.handling or item.rationale, label


class TestShippedPolicySet:
    """Every policy in policies/ must load, and they must differ meaningfully."""

    @pytest.fixture
    def policy_paths(self):
        from recyclevision.pipeline import DEFAULT_POLICY

        return sorted(DEFAULT_POLICY.parent.glob("*.yaml"))

    def test_all_shipped_policies_load(self, policy_paths):
        assert policy_paths, "expected at least one shipped policy"
        for path in policy_paths:
            assert RoutingPolicy.load(path).bins, path

    def test_policy_names_are_unique(self, policy_paths):
        names = [RoutingPolicy.load(p).name for p in policy_paths]
        assert len(names) == len(set(names))

    def test_context_changes_the_destination(self, policy_paths):
        """The whole point of policies as data.

        A "cup" in a kitchen is tableware and goes to landfill. On a sorting
        line it is a can, and belongs with the containers. Same detector,
        same label, different answer.
        """
        by_name = {RoutingPolicy.load(p).name: RoutingPolicy.load(p) for p in policy_paths}
        household = by_name["Household Single-Stream"]
        mrf = by_name["MRF Sorting Line"]

        detection = make_detection("cup")
        home_route = household.route(detection)
        line_route = mrf.route(detection)

        assert home_route is not None and line_route is not None
        assert home_route.bin.key != line_route.bin.key
        assert not home_route.bin.diverted
        assert line_route.bin.diverted

    def test_every_policy_ignores_non_waste(self, policy_paths):
        for path in policy_paths:
            policy = RoutingPolicy.load(path)
            for label in ("person", "car", "dog"):
                assert policy.route(make_detection(label)) is None, f"{path.name}: {label}"


class TestVocabularyCoverage:
    """Every shipped policy must have a rule for every shipped vocabulary class.

    An uncovered class is not a loud failure: the policy's `default: ignore`
    silently drops the detection as non-waste, so the item vanishes from every
    count with nothing to show it was ever seen. This test exists because that
    is exactly what happened to `drinking glass` and `paper`.
    """

    def test_every_policy_covers_every_vocabulary(self):
        from recyclevision.pipeline import DEFAULT_POLICY
        from recyclevision.vocabulary import VOCAB_DIR, Vocabulary

        policies = sorted(DEFAULT_POLICY.parent.glob("*.yaml"))
        vocabularies = sorted(VOCAB_DIR.glob("*.yaml"))
        assert policies and vocabularies

        gaps = {}
        for policy_path in policies:
            policy = RoutingPolicy.load(policy_path)
            for vocab_path in vocabularies:
                vocab = Vocabulary.load(vocab_path)
                missing = [c for c in vocab.classes if not policy.covers(c)]
                if missing:
                    gaps[f"{policy_path.name} x {vocab_path.name}"] = missing

        assert not gaps, f"vocabulary classes with no routing rule: {gaps}"
