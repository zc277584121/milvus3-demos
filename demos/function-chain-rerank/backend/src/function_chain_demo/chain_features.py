"""Single source of truth for the on-chain feature engineering formulas.

The Function Chain re-rank stage computes three business features from raw
collection columns at search time:

- ``popularity``  : ``num_combine(clicks_30d, sales_30d, weighted=[0.3, 0.7])``
- ``price_affinity`` : ``decay(display_price_usd, linear, origin=0, scale=500)``
- ``freshness``     : ``decay(release_epoch, exp, origin=AS_OF_EPOCH, scale=180d)``

The training pipeline MUST reproduce these formulas bit-for-bit in Python so the
XGBoost model learns against exactly the feature values Milvus will feed it at
query time. Every constant lives here, and both :mod:`dataset` (training) and
:mod:`service` (chain builder) import from this module rather than re-typing
numbers.

Formulas were verified against a running Milvus 3.0.0 instance:

- ``linearDecay`` = ``max(decay, 1 - (1-decay)/scale * max(0, |x-origin| - offset))``
- ``expDecay``     = ``exp(ln(decay)/scale * max(0, |x-origin| - offset))``
"""

from __future__ import annotations

import math
from typing import Final

# num_combine(weighted) for popularity.
# Clicks and sales are min-max normalized by their fixed generation bounds so the
# combined popularity lands in [0, 1], keeping feature magnitudes comparable to
# the other [0, 1] features. The weights are therefore 0.3/MAX_CLICKS and
# 0.7/MAX_SALES, so the chain still computes popularity on the fly.
POPULARITY_CLICKS_WEIGHT: Final = 0.3
POPULARITY_SALES_WEIGHT: Final = 0.7
POPULARITY_MAX_CLICKS: Final = 12_000
POPULARITY_MAX_SALES: Final = 850
POPULARITY_CLICKS_COEFF: Final = POPULARITY_CLICKS_WEIGHT / POPULARITY_MAX_CLICKS
POPULARITY_SALES_COEFF: Final = POPULARITY_SALES_WEIGHT / POPULARITY_MAX_SALES

# decay(linear) for price affinity: cheaper is better.
PRICE_DECAY_FUNCTION: Final = "linear"
PRICE_DECAY_ORIGIN: Final = 0.0
PRICE_DECAY_SCALE: Final = 500.0
PRICE_DECAY_OFFSET: Final = 0.0
PRICE_DECAY_VALUE: Final = 0.5

# decay(exp) for freshness: newer is better.
# origin is the dataset as-of date epoch (2026-08-20); scale is 180 days.
FRESHNESS_DECAY_FUNCTION: Final = "exp"
FRESHNESS_DECAY_ORIGIN: Final = 1787184000
FRESHNESS_DECAY_SCALE: Final = 180 * 86400
FRESHNESS_DECAY_OFFSET: Final = 0.0
FRESHNESS_DECAY_VALUE: Final = 0.5


def popularity(clicks_30d: float, sales_30d: float) -> float:
    """Mirror ``fn.num_combine(clicks_30d, sales_30d, weighted=[c, s])`` (min-max)."""
    return POPULARITY_CLICKS_COEFF * clicks_30d + POPULARITY_SALES_COEFF * sales_30d


def _adjusted_distance(value: float, origin: float, offset: float) -> float:
    return max(0.0, abs(value - origin) - offset)


def price_affinity(display_price_usd: float) -> float:
    """Mirror ``fn.decay(price, linear, origin=0, scale=500, decay=0.5)``."""
    slope = (1.0 - PRICE_DECAY_VALUE) / PRICE_DECAY_SCALE
    adjusted = _adjusted_distance(display_price_usd, PRICE_DECAY_ORIGIN, PRICE_DECAY_OFFSET)
    return max(PRICE_DECAY_VALUE, 1.0 - slope * adjusted)


def freshness(release_epoch: float) -> float:
    """Mirror ``fn.decay(release_epoch, exp, origin=AS_OF, scale=180d, decay=0.5)``."""
    lam = math.log(FRESHNESS_DECAY_VALUE) / FRESHNESS_DECAY_SCALE
    adjusted = _adjusted_distance(release_epoch, FRESHNESS_DECAY_ORIGIN, FRESHNESS_DECAY_OFFSET)
    return math.exp(lam * adjusted)
