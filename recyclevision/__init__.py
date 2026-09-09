"""RecycleVision AI -- tells you which bin each item belongs in, and why.

The public surface is small on purpose:

    from recyclevision import SortingPipeline, RoutingPolicy, annotate

Domain models carry no vision dependencies, so importing this package is
cheap; torch is only pulled in when a real detector is constructed.
"""

from .models import (
    Bin,
    BoundingBox,
    Certainty,
    Detection,
    RoutedItem,
    SortResult,
)
from .pipeline import DEFAULT_POLICY, SortingPipeline
from .policy import PolicyError, RoutingPolicy
from .render import annotate
from .weights import WeightsChoice, resolve_weights

__version__ = "0.3.0"

__all__ = [
    "Bin",
    "BoundingBox",
    "Certainty",
    "DEFAULT_POLICY",
    "Detection",
    "PolicyError",
    "RoutedItem",
    "RoutingPolicy",
    "SortResult",
    "SortingPipeline",
    "WeightsChoice",
    "annotate",
    "resolve_weights",
]
