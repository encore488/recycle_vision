"""Domain model invariants."""

from __future__ import annotations

import pytest

from recyclevision.models import BoundingBox, Certainty


class TestBoundingBox:
    def test_normalises_inverted_corners(self):
        """Detectors are not required to agree on corner order.

        An inverted box used to reach PIL, which raises "x1 must be greater
        than or equal to x0" and takes the whole render down.
        """
        box = BoundingBox(100, 80, 10, 20)
        assert (box.x1, box.y1, box.x2, box.y2) == (10, 20, 100, 80)

    def test_area_is_never_negative(self):
        for corners in [(0, 0, 10, 10), (10, 10, 0, 0), (5, 0, 0, 5)]:
            assert BoundingBox(*corners).area >= 0

    def test_degenerate_box_has_zero_area(self):
        assert BoundingBox(5, 5, 5, 5).area == 0

    def test_geometry(self):
        box = BoundingBox(10, 20, 110, 220)
        assert box.width == 100
        assert box.height == 200
        assert box.area == 20_000
        assert box.center == (60, 120)

    def test_negative_coordinates_are_preserved(self):
        box = BoundingBox(-10, -10, 10, 10)
        assert box.area == 400


class TestCertainty:
    def test_only_low_needs_review(self):
        assert Certainty.LOW.needs_review
        assert not Certainty.HIGH.needs_review

    def test_parses_from_string(self):
        assert Certainty("low") is Certainty.LOW
        with pytest.raises(ValueError):
            Certainty("maybe")
