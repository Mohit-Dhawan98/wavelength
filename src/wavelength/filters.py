"""Hard filters: the circuit breaker that runs before any scoring.

Why this is a filter and not a heavily weighted feature: a weighted average
smooths. A pair that agrees on eleven features and disagrees on whether to
have children still averages out to a high score, and the one disagreement
that decides the outcome is hidden inside it. No weight fixes that, because
enough agreement elsewhere can always outvote it. So dealbreakers are checked
first, and a pair that fails never reaches the scorer.

The four features and their maximum distances are the article's.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from wavelength.features import FEATURE_NAMES, FEATURES, MISSING

#: Maximum allowed distance per hard-filter feature. A pair passes when its
#: distance is less than or equal to the threshold. Timeline is tightest
#: because the article found it the most consistently predictive of rejection.
HARD_FILTERS: Mapping[str, float] = MappingProxyType(
    {
        "relationship_timeline": 0.2,
        "commitment_style": 0.3,
        "family_plans": 0.3,
        "trust_boundaries": 0.3,
    }
)


@dataclass(frozen=True)
class FilterViolation:
    """Why a pair was stopped. ``distance`` is None when a value was unknown."""

    feature: str
    distance: float | None
    threshold: float


def hard_filter_violations(
    a: Mapping[str, str | None], b: Mapping[str, str | None]
) -> list[FilterViolation]:
    """Every hard filter this pair fails; an empty list means the pair may be scored.

    An unknown value fails closed. The article does not say what production
    does when a dealbreaker has not been extracted yet; this implementation
    treats "we do not know" as "we cannot show this match", because the cost
    of surfacing a dealbreaker mismatch is much higher than the cost of
    waiting for one more answer.
    """
    violations: list[FilterViolation] = []
    for name, threshold in HARD_FILTERS.items():
        va, vb = a.get(name), b.get(name)
        if va is None or vb is None:
            violations.append(FilterViolation(name, None, threshold))
            continue
        d = FEATURES[name].distance(va, vb)
        if d > threshold:
            violations.append(FilterViolation(name, d, threshold))
    return violations


def passes_hard_filters(a: Mapping[str, str | None], b: Mapping[str, str | None]) -> bool:
    """True when the pair clears all four hard filters."""
    return not hard_filter_violations(a, b)


def eligibility_matrix(codes: np.ndarray) -> np.ndarray:
    """Vectorised hard filters for a whole population.

    ``codes`` is an ``(n_users, n_features)`` integer array in
    ``FEATURE_NAMES`` order. Returns a symmetric boolean matrix where
    ``[a, b]`` is True when the pair clears every hard filter. The diagonal
    is False: nobody is a candidate for themselves.
    """
    n = codes.shape[0]
    eligible = ~np.eye(n, dtype=bool)
    for name, threshold in HARD_FILTERS.items():
        column = codes[:, FEATURE_NAMES.index(name)]
        known = column != MISSING
        safe = np.where(known, column, 0)
        d = FEATURES[name].distances[safe[:, None], safe[None, :]]
        eligible &= known[:, None] & known[None, :] & (d <= threshold)
    return eligible
