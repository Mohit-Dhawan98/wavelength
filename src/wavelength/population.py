"""Who the simulated people really are.

This is the first half of the hidden truth: each person's side of the
market, what they want, how much each thing matters to them, how they come
across and how they live. ``decisions.py`` adds who they would say yes to.
No matcher reads any of it; a matcher sees only what ``observation.py``
derives from it.

Everything a person could argue about comes from the ``Scenario``. The few
constants below are the resolution of the model, not claims about the world.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from wavelength.features import FEATURE_NAMES, FEATURES, Group
from wavelength.scenario import CORE_FEATURES, DEALBREAKER_FEATURES, Scenario
from wavelength.weights import TIER_BANDS, Tier

#: Order used wherever a tier is stored as an integer code.
TIERS: tuple[Tier, ...] = (Tier.ALPHA, Tier.BETA, Tier.GAMMA)

#: People fall into this many loose "styles": ways of sounding and living
#: that shape their softer traits. Style is independent of archetype, which
#: is how two people can come across alike and want different lives.
N_STYLES = 6

#: Sizes of the hidden vibe and lifestyle vectors.
VIBE_DIM = 16
LIFESTYLE_DIM = 6

#: How tightly people cluster around their style, and how often a soft trait
#: follows the style's typical value instead of being drawn on its own.
STYLE_PULL = 1.0
STYLE_TRAIT_SHARE = 0.5

#: Among people for whom a feature is not a dealbreaker, the share who still
#: call it important (beta) and not merely nice-to-have (gamma).
IMPORTANT_SHARE_OF_REST: dict[Group, float] = {
    Group.CORE_VALUES: 0.70,
    Group.PERSONALITY: 0.50,
    Group.LIFESTYLE: 0.35,
}


@dataclass(frozen=True, eq=False)
class Population:
    """Ground truth about ``n`` people. Arrays are indexed by person."""

    scenario: Scenario
    majority: np.ndarray  # (n,) bool: True on the larger side of the market
    archetype: np.ndarray  # (n,) index into scenario.archetypes
    codes: np.ndarray  # (n, F) true feature values
    tiers: np.ndarray  # (n, F) index into TIERS: how much each feature matters
    weights: np.ndarray  # (n, F) true importance weights on the 0-10 scale
    vibe: np.ndarray  # (n, VIBE_DIM) unit vectors: how the person comes across
    lifestyle: np.ndarray  # (n, LIFESTYLE_DIM) unit vectors: how the person lives
    active: np.ndarray  # (n,) bool: responds to introductions at all
    appeal: np.ndarray  # (n,) general desirability, standard normal

    @property
    def n(self) -> int:
        return self.codes.shape[0]

    @property
    def across(self) -> np.ndarray:
        """``[a, b]`` is True when a and b are on opposite sides, so could be introduced."""
        return self.majority[:, None] != self.majority[None, :]


def unit_rows(x: np.ndarray) -> np.ndarray:
    """Scale each row to length 1."""
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _draw(rng: np.random.Generator, shares: Mapping[str, float], size: int) -> np.ndarray:
    """Draw value codes with the given population shares."""
    p = np.fromiter(shares.values(), dtype=float)
    return rng.choice(p.size, size=size, p=p / p.sum())


def _sides(rng: np.random.Generator, scenario: Scenario) -> np.ndarray:
    """Assign sides with the exact configured split, in random order."""
    n = scenario.pool_size
    n_majority = min(n - 1, max(1, round(scenario.majority_share * n)))
    majority = np.zeros(n, dtype=bool)
    majority[:n_majority] = True
    return rng.permutation(majority)


def _features(
    rng: np.random.Generator, scenario: Scenario, archetype: np.ndarray, style: np.ndarray
) -> np.ndarray:
    """True feature values.

    Dealbreaker features come from the person's archetype. Every other
    feature half follows the person's style and half is their own, both
    drawn from the scenario's population shares.
    """
    n = archetype.size
    codes = np.empty((n, len(FEATURE_NAMES)), dtype=np.int64)
    for j, name in enumerate(FEATURE_NAMES):
        if name in DEALBREAKER_FEATURES:
            for index, kind in enumerate(scenario.archetypes):
                members = archetype == index
                codes[members, j] = _draw(rng, kind.tendency(name), int(members.sum()))
            continue
        shares = scenario.trait_shares[name]
        typical = _draw(rng, shares, N_STYLES)[style]
        own = _draw(rng, shares, n)
        codes[:, j] = np.where(rng.random(n) < STYLE_TRAIT_SHARE, typical, own)
    return codes


def _dealbreaker_share(scenario: Scenario, feature: str) -> float:
    if feature in CORE_FEATURES:
        return scenario.dealbreaker_shares[feature]
    if FEATURES[feature].group is Group.PERSONALITY:
        return scenario.personality_dealbreaker_share
    return scenario.lifestyle_dealbreaker_share


def _importance(
    rng: np.random.Generator, scenario: Scenario, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """How much each feature matters to each person: a tier, and a weight inside its band."""
    tiers = np.empty((n, len(FEATURE_NAMES)), dtype=np.int64)
    for j, name in enumerate(FEATURE_NAMES):
        firm = _dealbreaker_share(scenario, name)
        important = (1.0 - firm) * IMPORTANT_SHARE_OF_REST[FEATURES[name].group]
        tiers[:, j] = rng.choice(3, size=n, p=[firm, important, 1.0 - firm - important])
    low = np.array([TIER_BANDS[t][0] for t in TIERS])[tiers]
    high = np.array([TIER_BANDS[t][1] for t in TIERS])[tiers]
    return tiers, rng.uniform(low, high)


def _clustered(rng: np.random.Generator, centres: np.ndarray) -> np.ndarray:
    """Unit vectors scattered around each person's centre."""
    scatter = rng.standard_normal(centres.shape) / np.sqrt(centres.shape[1])
    return unit_rows(STYLE_PULL * unit_rows(centres) + scatter)


def generate_population(rng: np.random.Generator, scenario: Scenario) -> Population:
    """Draw one pool of people from the scenario."""
    n = scenario.pool_size
    majority = _sides(rng, scenario)
    archetype = rng.choice(
        len(scenario.archetypes), size=n, p=[a.share for a in scenario.archetypes]
    )
    style = rng.integers(0, N_STYLES, size=n)
    codes = _features(rng, scenario, archetype, style)
    tiers, weights = _importance(rng, scenario, n)

    # How someone sounds is mostly their style; `intent_visible_in_bio`
    # says how much of what they want leaks into it.
    leak = scenario.intent_visible_in_bio
    style_voice = unit_rows(rng.standard_normal((N_STYLES, VIBE_DIM)))
    intent_voice = unit_rows(rng.standard_normal((len(scenario.archetypes), VIBE_DIM)))
    vibe = _clustered(rng, (1.0 - leak) * style_voice[style] + leak * intent_voice[archetype])
    style_life = unit_rows(rng.standard_normal((N_STYLES, LIFESTYLE_DIM)))
    lifestyle = _clustered(rng, style_life[style])

    return Population(
        scenario=scenario,
        majority=majority,
        archetype=archetype,
        codes=codes,
        tiers=tiers,
        weights=weights,
        vibe=vibe,
        lifestyle=lifestyle,
        active=rng.random(n) < scenario.active_share,
        appeal=rng.standard_normal(n),
    )
