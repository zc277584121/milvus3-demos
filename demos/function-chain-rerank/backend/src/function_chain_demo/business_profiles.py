"""Deterministic product-level operational profiles keyed only by synthetic item ID."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from function_chain_demo.synthetic_catalog import build_items

PROFILE_VERSION: Final = "synthetic-operational-profiles-v1"


@dataclass(frozen=True)
class OperationalProfile:
    rating_value: float
    inventory_ratio: float
    return_rate: float
    age_days: int


PROFILE_TEMPLATES: Final[dict[str, OperationalProfile]] = {
    "H": OperationalProfile(5.0, 0.95, 0.01, 36),
    "G": OperationalProfile(4.6, 0.82, 0.035, 131),
    "S": OperationalProfile(4.3, 0.65, 0.06, 234),
    "N": OperationalProfile(4.0, 0.50, 0.10, 365),
    "R": OperationalProfile(3.1, 0.02, 0.24, 642),
}

# The mapping contains no query identity, vector rank, or target order. These are stable,
# fictional merchant profiles attached to products regardless of which query retrieves them.
# Only the 36 strong intent items carry an explicit tier (six per intent type, matching the
# frozen strong-association layout); every other item falls back to the generic deterministic
# signal path in `dataset.simulated_signals`.
ITEM_PROFILE_NAMES: Final[dict[str, str]] = {
    str(item["id"]): str(item["profile"])
    for item in build_items()
    if item["selection_class"] == "strong"
}


def operational_profile(item_id: str) -> OperationalProfile | None:
    """Return an item-level profile without consulting query or ranking information."""
    profile_name = ITEM_PROFILE_NAMES.get(item_id)
    return PROFILE_TEMPLATES.get(profile_name) if profile_name else None
