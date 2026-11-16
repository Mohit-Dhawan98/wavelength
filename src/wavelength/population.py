"""Who the simulated people really are.

This is the first half of the hidden truth: each person's side of the
market, what they want, how much each thing matters to them, how they come
across in each channel, where they live, what their family expects and how
they use the app. ``decisions.py`` and ``key.py`` add how far every pair
would get. No rule reads any of it; a rule sees only what ``observation.py``
derives from it.

Everything a person could argue about comes from the ``Scenario``. The few
constants below are the resolution of the model, not claims about the world.

Three hidden traits feed three channels, and they are drawn independently:
what someone says (vibe, seen through text), how they live (lifestyle, seen
through photos) and how they talk (manner, heard in voice). Each has its own
loose "styles". Keeping them apart matters: if one hidden style drove all
three, any one channel would stand in for the rest and a rule that used
only text would look as good as one that used everything.
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

#: People fall into this many loose styles per channel.
N_STYLES = 6

#: Sizes of the hidden vectors.
VIBE_DIM = 16
LIFESTYLE_DIM = 6
MANNER_DIM = 8
FAMILY_DIM = 4

#: How tightly people cluster around a style, and how often a soft trait
#: follows a style's typical value instead of being drawn on its own.
#: Personality traits lean on the vibe style and lifestyle features on the
#: lifestyle style, loosely, so a channel hints at the features nearest to
#: it without giving them away.
STYLE_PULL = 1.0
STYLE_TRAIT_SHARE = 0.3

#: Among people for whom a feature is not a dealbreaker, the share who still
#: call it important (beta) and not merely nice-to-have (gamma).
IMPORTANT_SHARE_OF_REST: dict[Group, float] = {
    Group.CORE_VALUES: 0.70,
    Group.PERSONALITY: 0.50,
    Group.LIFESTYLE: 0.35,
}

#: How sharply a person's attention concentrates on what they rate highest,
#: at full ``preference_diversity``: importance goes as weight to this power.
IMPORTANCE_SHARPNESS = 3.0

#: Ages in the pool, in years, drawn evenly. Marriage pressure starts at the
#: first age and is complete at the second.
AGE_RANGE = (22, 40)
PRESSURE_AGES = (25, 35)

#: Spread between people in how far they will travel and how patient they are.
TOLERANCE_SPREAD = 0.4
PATIENCE_SPREAD = 0.4


@dataclass(frozen=True, eq=False)
class Population:
    """Ground truth about ``n`` people. Arrays are indexed by person."""

    scenario: Scenario
    majority: np.ndarray  # (n,) bool: True on the larger side of the market
    archetype: np.ndarray  # (n,) index into scenario.archetypes
    codes: np.ndarray  # (n, F) true feature values
    stated_codes: np.ndarray  # (n, F) what the person says: differs on timeline if not open
    open_about_intent: np.ndarray  # (n,) bool: states intent and constraints as they are
    tiers: np.ndarray  # (n, F) index into TIERS: how much each feature matters
    weights: np.ndarray  # (n, F) true importance weights on the 0-10 scale
    importance: np.ndarray  # (n, F) share of the person's attention on each feature, sums to 1
    vibe: np.ndarray  # (n, VIBE_DIM) unit vectors: what the person says
    lifestyle: np.ndarray  # (n, LIFESTYLE_DIM) unit vectors: how the person lives
    manner: np.ndarray  # (n, MANNER_DIM) unit vectors: how the person talks
    family: np.ndarray  # (n, FAMILY_DIM) unit vectors: what the person's family expects
    community: np.ndarray  # (n,) index into scenario.community_shares
    endogamous: np.ndarray  # (n,) bool: a partner outside the community is a dealbreaker
    family_required: np.ndarray  # (n,) bool: needs family approval
    age: np.ndarray  # (n,) years
    location: np.ndarray  # (n, 2) position in a unit square
    travel: np.ndarray  # (n,) how far the person will go to meet, same units
    initiates: np.ndarray  # (n,) bool: will make the first move
    habit: np.ndarray  # (n,) chance of opening the app in a given week
    patience: np.ndarray  # (n,) letdowns the person takes before giving up
    photos_hidden: np.ndarray  # (n,) bool: keeps photos hidden until later
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
    rng: np.random.Generator,
    scenario: Scenario,
    archetype: np.ndarray,
    styles: Mapping[Group, np.ndarray],
) -> np.ndarray:
    """True feature values, before age has any say.

    Dealbreaker features come from the person's archetype. Personality and
    lifestyle features lean loosely on the matching style; the rest are the
    person's own. All are drawn from the scenario's population shares.
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
        own = _draw(rng, shares, n)
        style = styles.get(FEATURES[name].group)
        if style is None:
            codes[:, j] = own
            continue
        typical = _draw(rng, shares, N_STYLES)[style]
        codes[:, j] = np.where(rng.random(n) < STYLE_TRAIT_SHARE, typical, own)
    return codes


def _apply_marriage_pressure(
    rng: np.random.Generator, scenario: Scenario, codes: np.ndarray, age: np.ndarray
) -> None:
    """Move some older people one step towards marriage, in place."""
    start, full = PRESSURE_AGES
    strength = scenario.marriage_pressure * np.clip((age - start) / (full - start), 0.0, 1.0)
    column = FEATURE_NAMES.index("relationship_timeline")
    # The feature's values run from marriage-focused (0) to casual; a step down is towards marriage.
    moved = rng.random(age.size) < strength
    codes[moved, column] = np.maximum(codes[moved, column] - 1, 0)


def _stated(
    rng: np.random.Generator, scenario: Scenario, codes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """What people say they are looking for.

    Someone who is not open about intent gives the answer most people in
    the pool give. Every other value is stated as it is.
    """
    n = codes.shape[0]
    open_about_intent = rng.random(n) < scenario.intent_openness
    column = FEATURE_NAMES.index("relationship_timeline")
    usual = int(np.bincount(codes[:, column]).argmax())
    stated = codes.copy()
    stated[~open_about_intent, column] = usual
    return stated, open_about_intent


def _dealbreaker_share(scenario: Scenario, feature: str) -> float:
    if feature in CORE_FEATURES:
        return scenario.dealbreaker_shares[feature]
    if FEATURES[feature].group is Group.PERSONALITY:
        return scenario.personality_dealbreaker_share
    return scenario.lifestyle_dealbreaker_share


def _weights(rng: np.random.Generator, scenario: Scenario, n: int) -> tuple[np.ndarray, np.ndarray]:
    """How much each feature matters to each person: a tier, and a weight inside its band."""
    tiers = np.empty((n, len(FEATURE_NAMES)), dtype=np.int64)
    for j, name in enumerate(FEATURE_NAMES):
        firm = _dealbreaker_share(scenario, name)
        important = (1.0 - firm) * IMPORTANT_SHARE_OF_REST[FEATURES[name].group]
        tiers[:, j] = rng.choice(3, size=n, p=[firm, important, 1.0 - firm - important])
    low = np.array([TIER_BANDS[t][0] for t in TIERS])[tiers]
    high = np.array([TIER_BANDS[t][1] for t in TIERS])[tiers]
    return tiers, rng.uniform(low, high)


def importance_from_weights(weights: np.ndarray, diversity: float) -> np.ndarray:
    """Turn 0-10 weights into each person's shares of attention.

    A person's own shares concentrate on what they rate highest. With no
    diversity everyone uses the pool's average shares; with full diversity
    everyone uses their own.
    """
    own = (weights / 10.0) ** IMPORTANCE_SHARPNESS
    own = own / own.sum(axis=1, keepdims=True)
    return (1.0 - diversity) * own.mean(axis=0, keepdims=True) + diversity * own


def _clustered(rng: np.random.Generator, centres: np.ndarray) -> np.ndarray:
    """Unit vectors scattered around each person's centre."""
    scatter = rng.standard_normal(centres.shape) / np.sqrt(centres.shape[1])
    return unit_rows(STYLE_PULL * unit_rows(centres) + scatter)


def _styled(rng: np.random.Generator, n: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """A style per person and the centre of that style in a ``dim``-dimensional space."""
    style = rng.integers(0, N_STYLES, size=n)
    return style, unit_rows(rng.standard_normal((N_STYLES, dim)))[style]


def _habits(rng: np.random.Generator, scenario: Scenario, n: int) -> np.ndarray:
    """Each person's weekly chance of opening the app.

    A beta distribution around ``open_rate``: no spread gives everyone the
    same habit, full spread gives people who always answer and people who
    never do.
    """
    spread = scenario.open_rate_spread
    if spread <= 0.0:
        return np.full(n, scenario.open_rate)
    if spread >= 1.0:
        return (rng.random(n) < scenario.open_rate).astype(float)
    concentration = 1.0 / spread - 1.0
    return rng.beta(
        scenario.open_rate * concentration, (1.0 - scenario.open_rate) * concentration, size=n
    )


def generate_population(rng: np.random.Generator, scenario: Scenario) -> Population:
    """Draw one pool of people from the scenario."""
    n = scenario.pool_size
    majority = _sides(rng, scenario)
    archetype = rng.choice(
        len(scenario.archetypes), size=n, p=[a.share for a in scenario.archetypes]
    )
    vibe_style, vibe_centre = _styled(rng, n, VIBE_DIM)
    life_style, life_centre = _styled(rng, n, LIFESTYLE_DIM)
    _, manner_centre = _styled(rng, n, MANNER_DIM)

    age = rng.integers(AGE_RANGE[0], AGE_RANGE[1] + 1, size=n)
    codes = _features(
        rng, scenario, archetype, {Group.PERSONALITY: vibe_style, Group.LIFESTYLE: life_style}
    )
    _apply_marriage_pressure(rng, scenario, codes, age)
    stated_codes, open_about_intent = _stated(rng, scenario, codes)
    tiers, weights = _weights(rng, scenario, n)

    # What someone says is mostly their own style; `intent_visible_in_bio`
    # says how much of what they want leaks into it.
    leak = scenario.intent_visible_in_bio
    intent_voice = unit_rows(rng.standard_normal((len(scenario.archetypes), VIBE_DIM)))
    vibe = _clustered(rng, (1.0 - leak) * vibe_centre + leak * intent_voice[archetype])

    community = _draw(rng, scenario.community_shares, n)
    family_centres = unit_rows(rng.standard_normal((len(scenario.community_shares), FAMILY_DIM)))
    initiates_rate = np.where(
        majority, scenario.larger_side_initiates, scenario.scarce_side_initiates
    )
    patience = scenario.patience * np.exp(PATIENCE_SPREAD * rng.standard_normal(n))

    return Population(
        scenario=scenario,
        majority=majority,
        archetype=archetype,
        codes=codes,
        stated_codes=stated_codes,
        open_about_intent=open_about_intent,
        tiers=tiers,
        weights=weights,
        importance=importance_from_weights(weights, scenario.preference_diversity),
        vibe=vibe,
        lifestyle=_clustered(rng, life_centre),
        manner=_clustered(rng, manner_centre),
        family=_clustered(rng, family_centres[community]),
        community=community,
        endogamous=rng.random(n) < scenario.endogamy_share,
        family_required=rng.random(n) < scenario.family_approval_share,
        age=age,
        location=rng.random((n, 2)),
        travel=scenario.distance_tolerance * np.exp(TOLERANCE_SPREAD * rng.standard_normal(n)),
        initiates=rng.random(n) < initiates_rate,
        habit=_habits(rng, scenario, n),
        patience=np.maximum(1, np.rint(patience)).astype(np.int64),
        photos_hidden=rng.random(n) < scenario.photos_hidden_share,
        appeal=rng.standard_normal(n),
    )
