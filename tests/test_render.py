"""Annotation rendering.

The v0.2 app passed ultralytics' BGR array straight to Streamlit, so every
annotated image displayed with red and blue swapped. These tests pin the
channel order down.
"""

from __future__ import annotations

import pytest
from PIL import Image

from recyclevision.detector import StubDetector
from recyclevision.pipeline import SortingPipeline
from recyclevision.render import _hex_to_rgb, _readable_text_color, annotate
from tests.conftest import make_detection


def unique_colors(image: Image.Image) -> set[tuple[int, int, int]]:
    """Every distinct colour present in an image."""
    # getcolors() over the full 24-bit space never returns None for RGB.
    return {color for _count, color in image.getcolors(1 << 24)}


def sort(policy, detections, image):
    return SortingPipeline(StubDetector(detections), policy).sort(image)


class TestColorHelpers:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("#1E6FD9", (30, 111, 217)),
            ("1E6FD9", (30, 111, 217)),
            ("#fff", (255, 255, 255)),
            ("#000", (0, 0, 0)),
        ],
    )
    def test_hex_parsing(self, value, expected):
        assert _hex_to_rgb(value) == expected

    def test_text_color_stays_legible(self):
        assert _readable_text_color((255, 255, 255)) == (0, 0, 0)
        assert _readable_text_color((0, 0, 0)) == (255, 255, 255)


class TestAnnotate:
    def test_returns_a_new_image_of_the_same_size(self, policy, image):
        result = sort(policy, [make_detection("bottle")], image)
        out = annotate(image, result)
        assert out.size == image.size
        assert out is not image

    def test_original_image_is_untouched(self, policy, image):
        before = unique_colors(image)
        annotate(image, sort(policy, [make_detection("bottle")], image))
        assert unique_colors(image) == before

    def test_boxes_are_drawn_in_the_bin_color_not_swapped(self, policy, image):
        """Guards the v0.2 BGR/RGB bug: recycling blue must render blue."""
        result = sort(policy, [make_detection("bottle", box=(50, 50, 300, 300))], image)
        out = annotate(image, result, show_labels=False)

        expected = _hex_to_rgb(policy.bin("recycling").color)
        assert expected[2] > expected[0], "fixture assumption: recycling blue is blue-dominant"

        pixels = unique_colors(out)
        assert expected in pixels, "box should be drawn in the bin's exact colour"
        # The BGR-swapped version of that colour must be absent.
        assert (expected[2], expected[1], expected[0]) not in pixels

    def test_different_bins_get_different_colors(self, policy, image):
        result = sort(
            policy,
            [
                make_detection("bottle", box=(10, 10, 200, 200)),
                make_detection("pizza", box=(300, 300, 500, 450)),
            ],
            image,
        )
        out = annotate(image, result, show_labels=False)
        pixels = unique_colors(out)
        assert _hex_to_rgb(policy.bin("recycling").color) in pixels
        assert _hex_to_rgb(policy.bin("compost").color) in pixels

    def test_toggles_actually_suppress_drawing(self, policy, image):
        """The v0.2 sidebar toggles were wired to nothing at all."""
        result = sort(policy, [make_detection("bottle")], image)
        blank = annotate(image, result, show_boxes=False, show_labels=False)
        assert unique_colors(blank) == {(255, 255, 255)}

        boxed = annotate(image, result, show_boxes=True, show_labels=False)
        assert unique_colors(boxed) != {(255, 255, 255)}

    def test_empty_result_leaves_the_image_clean(self, policy, image):
        out = annotate(image, sort(policy, [], image))
        assert unique_colors(out) == {(255, 255, 255)}

    def test_labels_near_the_top_edge_stay_on_canvas(self, policy, image):
        """A box at y=0 has no room above it for its label chip."""
        result = sort(policy, [make_detection("bottle", box=(0, 0, 200, 100))], image)
        out = annotate(image, result, show_labels=True)
        assert out.size == image.size
        assert unique_colors(out) != {(255, 255, 255)}

    def test_labels_near_the_right_edge_stay_on_canvas(self, policy, image):
        result = sort(policy, [make_detection("bottle", box=(600, 200, 639, 300))], image)
        annotate(image, result, show_labels=True)  # must not raise

    def test_handles_non_rgb_input(self, policy):
        grayscale = Image.new("L", (200, 200), 128)
        result = sort(policy, [make_detection("bottle", box=(10, 10, 100, 100))], grayscale)
        out = annotate(grayscale, result)
        assert out.mode == "RGB"
