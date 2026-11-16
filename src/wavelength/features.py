"""Feature taxonomy and graduated distance matrices.

Why categories and not 1-10 scales: "wants children soon" versus "childfree"
is not a point on a line, it is a pair of life plans, and how far apart two
plans are is a judgement about that specific pair. So every feature has a
small set of named values and an explicit distance for every pair of values,
from 0.0 (identical) to 1.0 (opposed).

Why graduated and not binary: "different" does not always mean
"incompatible". Someone who raises issues directly and someone who needs time
first both engage with conflict (distance 0.3); direct paired with avoidance
does not work nearly as well (0.9). A compatible/incompatible flag cannot
express that.

The article describes 26 features and 160 distance entries derived with LLM
analysis. This module carries a 12-feature subset. Only the
``conflict_resolution`` matrix and two ``commitment_style`` distances
(0.15 and 0.8) are quoted from the article; every other number here is an
illustrative value chosen for this reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np


class Group(Enum):
    """The article's three categories, ordered from fatal to flexible."""

    CORE_VALUES = "core_values"
    PERSONALITY = "personality"
    LIFESTYLE = "lifestyle"


@dataclass(frozen=True, eq=False)
class Feature:
    """One categorical feature and the distance between each pair of its values."""

    name: str
    group: Group
    values: tuple[str, ...]
    distances: np.ndarray  # shape (len(values), len(values)), symmetric, zero diagonal

    def code(self, value: str) -> int:
        """Integer code of a value; raises on anything outside the enum.

        Extraction in the article is validated against predefined enums for
        the same reason: a value the matrices do not know has no distance.
        """
        try:
            return self.values.index(value)
        except ValueError:
            raise ValueError(f"{value!r} is not a value of {self.name}: {self.values}") from None

    def distance(self, a: str, b: str) -> float:
        """Distance between two values, 0.0 (identical) to 1.0 (opposed)."""
        return float(self.distances[self.code(a), self.code(b)])


def _feature(
    name: str,
    group: Group,
    values: tuple[str, ...],
    pairs: Mapping[tuple[str, str], float],
) -> Feature:
    """Build a feature from its unordered pair distances.

    Each pair is written once and mirrored, so a matrix cannot be
    asymmetric by typo, and a missing pair is an error rather than a
    silent zero.
    """
    n = len(values)
    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 0.0)
    for (a, b), d in pairs.items():
        i, j = values.index(a), values.index(b)
        if i == j or not np.isnan(matrix[i, j]):
            raise ValueError(f"{name}: pair ({a}, {b}) is a self-pair or a duplicate")
        if not 0.0 <= d <= 1.0:
            raise ValueError(f"{name}: distance {d} for ({a}, {b}) is outside [0, 1]")
        matrix[i, j] = matrix[j, i] = d
    if np.isnan(matrix).any():
        raise ValueError(f"{name}: not every pair of values has a distance")
    matrix.setflags(write=False)
    return Feature(name=name, group=group, values=values, distances=matrix)


_CORE = Group.CORE_VALUES
_PERS = Group.PERSONALITY
_LIFE = Group.LIFESTYLE

_ALL_FEATURES: tuple[Feature, ...] = (
    # --- Core values: the features where incompatibility is fatal. ---
    _feature(
        "relationship_timeline",
        _CORE,
        ("marriage_focused", "long_term", "exploring", "casual"),
        {
            ("marriage_focused", "long_term"): 0.2,
            ("marriage_focused", "exploring"): 0.6,
            ("marriage_focused", "casual"): 1.0,
            ("long_term", "exploring"): 0.4,
            ("long_term", "casual"): 0.8,
            ("exploring", "casual"): 0.2,
        },
    ),
    _feature(
        "commitment_style",
        _CORE,
        ("requires_exclusivity", "prefers_monogamy", "flexible", "open_relationship"),
        {
            ("requires_exclusivity", "prefers_monogamy"): 0.15,  # from the article
            ("requires_exclusivity", "flexible"): 0.6,
            ("requires_exclusivity", "open_relationship"): 1.0,
            ("prefers_monogamy", "flexible"): 0.4,
            ("prefers_monogamy", "open_relationship"): 0.8,  # from the article
            ("flexible", "open_relationship"): 0.3,
        },
    ),
    _feature(
        "family_plans",
        _CORE,
        ("wants_children_soon", "wants_children_someday", "undecided", "childfree"),
        {
            ("wants_children_soon", "wants_children_someday"): 0.2,
            ("wants_children_soon", "undecided"): 0.5,
            ("wants_children_soon", "childfree"): 1.0,
            ("wants_children_someday", "undecided"): 0.3,
            ("wants_children_someday", "childfree"): 0.9,
            ("undecided", "childfree"): 0.4,
        },
    ),
    _feature(
        "trust_boundaries",
        _CORE,
        ("needs_security", "balanced", "very_flexible"),
        {
            ("needs_security", "balanced"): 0.3,
            ("needs_security", "very_flexible"): 0.9,
            ("balanced", "very_flexible"): 0.4,
        },
    ),
    _feature(
        "emotional_availability",
        _CORE,
        ("open", "guarded", "unavailable"),
        {
            ("open", "guarded"): 0.4,
            ("open", "unavailable"): 0.9,
            ("guarded", "unavailable"): 0.5,
        },
    ),
    # --- Personality: important but flexible. ---
    _feature(
        "intellectual_connection",
        _PERS,
        ("high", "moderate", "low"),
        {("high", "moderate"): 0.4, ("high", "low"): 0.9, ("moderate", "low"): 0.4},
    ),
    _feature(
        "emotional_expressiveness",
        _PERS,
        ("expressive", "measured", "reserved"),
        {
            ("expressive", "measured"): 0.3,
            ("expressive", "reserved"): 0.8,
            ("measured", "reserved"): 0.3,
        },
    ),
    _feature(
        "adventure_tolerance",
        _PERS,
        ("thrill_seeking", "moderate", "comfort_seeking"),
        {
            ("thrill_seeking", "moderate"): 0.4,
            ("thrill_seeking", "comfort_seeking"): 0.9,
            ("moderate", "comfort_seeking"): 0.3,
        },
    ),
    _feature(
        "social_energy",
        _PERS,
        ("extrovert", "ambivert", "introvert"),
        {
            # Opposite social energy is a difference, not an opposition:
            # plenty of such couples work, so it stops well short of 1.0.
            ("extrovert", "ambivert"): 0.2,
            ("extrovert", "introvert"): 0.6,
            ("ambivert", "introvert"): 0.2,
        },
    ),
    # --- Lifestyle: nice-to-haves. ---
    _feature(
        "conflict_resolution",
        _LIFE,
        ("direct", "time_first", "avoidance"),
        {
            # This matrix is the article's, verbatim.
            ("direct", "time_first"): 0.3,
            ("direct", "avoidance"): 0.9,
            ("time_first", "avoidance"): 0.7,
        },
    ),
    _feature(
        "professional_ambition",
        _LIFE,
        ("career_first", "balanced", "life_first"),
        {
            ("career_first", "balanced"): 0.4,
            ("career_first", "life_first"): 0.8,
            ("balanced", "life_first"): 0.2,
        },
    ),
    _feature(
        "travel_frequency",
        _LIFE,
        ("frequent", "occasional", "rare"),
        {("frequent", "occasional"): 0.3, ("frequent", "rare"): 0.8, ("occasional", "rare"): 0.3},
    ),
)

#: All features by name, in a fixed order. Array-based code relies on this
#: order: column ``j`` of any feature array is ``FEATURE_NAMES[j]``.
FEATURES: Mapping[str, Feature] = MappingProxyType({f.name: f for f in _ALL_FEATURES})
FEATURE_NAMES: tuple[str, ...] = tuple(FEATURES)

#: Code used in integer feature arrays for "not extracted yet".
MISSING = -1


def distance(feature: str, a: str, b: str) -> float:
    """Distance between two values of the named feature."""
    return FEATURES[feature].distance(a, b)


def encode(values: Mapping[str, str | None]) -> np.ndarray:
    """Encode a profile's feature values as integer codes (``MISSING`` where absent)."""
    codes = np.full(len(FEATURE_NAMES), MISSING, dtype=np.int64)
    for j, name in enumerate(FEATURE_NAMES):
        value = values.get(name)
        if value is not None:
            codes[j] = FEATURES[name].code(value)
    return codes
