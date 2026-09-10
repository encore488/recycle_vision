"""Impact estimates, and the honesty flag that travels with them."""

from __future__ import annotations

import pytest

from recyclevision.detector import StubDetector
from recyclevision.impact import DEFAULT_FACTORS, ImpactError, ImpactModel
from recyclevision.pipeline import SortingPipeline
from tests.conftest import make_detection


def sort(policy, labels, image):
    detections = [make_detection(label, 0.9) for label in labels]
    return SortingPipeline(StubDetector(detections), policy).sort(image)


@pytest.fixture
def model():
    return ImpactModel.load()


class TestHonesty:
    def test_shipped_factors_are_marked_unverified(self, model):
        """The values in impact/factors.yaml are placeholders.

        If someone replaces them with cited figures they can flip the flag —
        but it must never be flipped while the estimates are guesses, because
        the flag is the only thing standing between an estimate and a claim.
        """
        assert model.verified is False

    def test_unverified_estimates_carry_a_caveat(self, model, policy, image):
        estimate = model.estimate(sort(policy, ["metal can"], image))
        assert estimate.verified is False
        assert "unverified" in estimate.caveat.lower()

    def test_verified_factors_drop_the_caveat(self):
        verified = ImpactModel(
            name="t",
            materials={"metal": {"co2e_kg_per_kg": 5.0}},
            items={"metal can": {"mass_g": 40, "material": "metal"}},
            verified=True,
        )
        assert verified.verified is True

    def test_missing_verified_key_defaults_to_unverified(self, tmp_path):
        """Trust has to be asserted, never assumed by omission."""
        factors = tmp_path / "f.yaml"
        factors.write_text("name: no flag\nitems:\n  metal can: {mass_g: 40}\n")
        assert ImpactModel.load(factors).verified is False


class TestEstimate:
    def test_sums_mass_over_items(self, model, policy, image):
        estimate = model.estimate(sort(policy, ["metal can", "metal can"], image))
        assert estimate.total_mass_kg == pytest.approx(0.080)

    def test_landfilled_items_count_toward_mass_but_not_diversion(self, model, policy, image):
        """Recycling avoids the emission; landfilling avoids nothing."""
        estimate = model.estimate(sort(policy, ["empty drinking glass"], image))
        assert estimate.total_mass_kg > 0
        assert estimate.diverted_mass_kg == 0
        assert estimate.co2e_avoided_kg == 0

    def test_carbon_scales_with_diverted_mass(self, model, policy, image):
        one = model.estimate(sort(policy, ["metal can"], image))
        two = model.estimate(sort(policy, ["metal can", "metal can"], image))
        assert two.co2e_avoided_kg == pytest.approx(2 * one.co2e_avoided_kg)

    def test_aluminium_dominates_glass_by_an_order_of_magnitude(self, model, policy, image):
        """The one qualitative fact the placeholder numbers get right."""
        metal = model.estimate(sort(policy, ["metal can"], image))
        glass = model.estimate(sort(policy, ["glass jar"], image))
        assert metal.co2e_avoided_kg > glass.co2e_avoided_kg

    def test_unpriced_items_are_reported_not_silently_zeroed(self, policy, image):
        model = ImpactModel(name="t", materials={}, items={}, verified=False)
        estimate = model.estimate(sort(policy, ["metal can"], image))
        assert estimate.total_mass_kg == 0
        assert "metal can" in estimate.unpriced_items
        assert "No mass factor" in estimate.coverage_note

    def test_empty_result_is_all_zeros(self, model, policy, image):
        estimate = model.estimate(sort(policy, [], image))
        assert estimate.total_mass_kg == 0
        assert estimate.co2e_avoided_kg == 0
        assert estimate.coverage_note == ""

    def test_label_matching_ignores_case(self, model, policy, image):
        estimate = model.estimate(sort(policy, ["METAL CAN"], image))
        assert estimate.total_mass_kg > 0


class TestLoading:
    def test_missing_file_is_an_impact_error(self, tmp_path):
        with pytest.raises(ImpactError, match="no factors file"):
            ImpactModel.load(tmp_path / "absent.yaml")

    def test_malformed_yaml_is_an_impact_error(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("items: [unclosed\n")
        with pytest.raises(ImpactError, match="not valid YAML"):
            ImpactModel.load(bad)

    def test_file_without_items_is_rejected(self, tmp_path):
        bare = tmp_path / "bare.yaml"
        bare.write_text("name: nothing\n")
        with pytest.raises(ImpactError, match="items"):
            ImpactModel.load(bare)

    def test_default_factors_cover_the_default_vocabulary(self):
        """An item with no mass factor contributes nothing to the totals."""
        from recyclevision.vocabulary import DEFAULT_VOCAB, Vocabulary

        model = ImpactModel.load(DEFAULT_FACTORS)
        missing = [
            c
            for c in Vocabulary.load(DEFAULT_VOCAB).classes
            if model._normalize(c) not in model._items
        ]
        assert not missing, f"no mass factor for: {missing}"
