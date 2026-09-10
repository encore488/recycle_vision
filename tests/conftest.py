"""Shared fixtures.

Everything here is deliberately weight-free and network-free: the whole
pipeline is testable through `StubDetector`, so CI needs neither torch nor a
model download.
"""

from __future__ import annotations

import pytest
from PIL import Image

from recyclevision.models import BoundingBox, Detection
from recyclevision.pipeline import DEFAULT_POLICY
from recyclevision.policy import RoutingPolicy


@pytest.fixture
def policy() -> RoutingPolicy:
    """The real shipped household policy, so tests guard the actual rules."""
    return RoutingPolicy.load(DEFAULT_POLICY)


@pytest.fixture
def image() -> Image.Image:
    return Image.new("RGB", (640, 480), "white")


def make_detection(label: str, confidence: float = 0.9, box=(10, 10, 110, 110)) -> Detection:
    return Detection(label=label, confidence=confidence, box=BoundingBox(*box))


@pytest.fixture
def detection_factory():
    return make_detection


MINIMAL_POLICY = {
    "name": "Test policy",
    "bins": [
        {"key": "recycling", "name": "Recycling", "color": "#1E6FD9", "diverted": True},
        {"key": "landfill", "name": "Landfill", "color": "#6B7280", "diverted": False},
    ],
    "rules": {
        "bottle": {"bin": "recycling", "item": "Bottle", "material": "plastic"},
        "toothbrush": {"bin": "landfill", "certainty": "low"},
    },
}
