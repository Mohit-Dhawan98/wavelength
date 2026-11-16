"""Who would really say yes to whom when introduced.

This is the first stage of the hidden answer key (``key.py`` has the rest),
and the part that keeps the simulation from being circular. If simulated
people accepted according to a matcher's own score, any matcher would look
perfect. So the decision is built from quantities no rule ever sees, and
differs from a matcher's score in form as well as inputs:

- it uses each person's true feature values and true priorities, not the
  extracted values and learned weights;
- a dealbreaker the person can see in the introduction is a refusal, not a
  large term in an average;
- part of attraction is the other person's general appeal;
- part is chemistry, some of it shared by the pair;
- part of any single decision is mood on the day.

A person says yes when what they feel clears their own bar. The bar is an
absolute level of attraction, fixed per person before any rule runs. It is
placed so that the person would say yes to their own ``yes_rate`` share of
eligible people picked at random. Being shown better-than-random people
therefore produces more yeses, and by how much is not set anywhere: it
follows from how well a rule's choices track true attraction.

Everything random here is drawn once, from the world's seed, before any
rule runs. Two rules introduced to the same pair get the same answer.

What a person can see in an introduction is what the other person states:
their four dealbreaker answers and their community. Dealbreakers that are
not in a profile (and intent that was not stated honestly) surface later,
in person; see ``key.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelength.features import FEATURE_NAMES, FEATURES
from wavelength.filters import eligibility_matrix
from wavelength.population import TIERS, Population
from wavelength.scenario import DEALBREAKER_FEATURES, Scenario
from wavelength.weights import Tier

#: A person for whom a feature is a true dealbreaker refuses anyone at least
#: this far from them on it. The matcher's hard-filter thresholds (0.2 and
#: 0.3) are tighter, so a pair that passes the filters on true values never
#: triggers a refusal on those four features.
REFUSAL_DISTANCE = 0.5

#: A person's bar is placed using their own eligible candidates when there
#: are at least this many of them, and the whole pool's otherwise.
MIN_REFERENCE = 8

#: Personal yes-rates are kept inside this range whatever the spread.
YES_RATE_LIMITS = (0.01, 0.95)

#: Weeks of app-opening luck drawn per person. A run can be at most this long.
MAX_ROUNDS = 52

_ALPHA = TIERS.index(Tier.ALPHA)


@dataclass(frozen=True, eq=False)
class Decisions:
    """What each person would do if introduced to each other person."""

    felt: np.ndarray  # (n, n) what row feels about column on the day: attraction plus mood
    would_accept: np.ndarray  # (n, n) bool: row's decision about column, if row answers
    yes_rate: np.ndarray  # (n,) each person's own yes-rate to a random eligible person
    bar: np.ndarray  # (n,) the level of felt attraction each person needs to say yes
    refuses: np.ndarray  # (n, n) bool: row refuses column on what column's profile states
    refuses_in_person: np.ndarray  # (n, n) bool: row has any true dealbreaker against column
    eligible: np.ndarray  # (n, n) bool: opposite sides and clear of the hard filters, truly
    profile: np.ndarray  # (n, n) the part of attraction a perfect read of profiles explains
    chemistry: np.ndarray  # (n, n) row's chemistry with column: shared part plus their own
    shared_chemistry: np.ndarray  # (n, n) symmetric: the part of chemistry both feel
    open_luck: np.ndarray  # (n, MAX_ROUNDS) uniform draws: does the person open the app that week
    habit: np.ndarray  # (n,) each person's chance of opening the app in a week

    @property
    def responds(self) -> np.ndarray:
        """Who answers in the first week, before anyone has been let down."""
        return self.open_luck[:, 0] < self.habit

    @property
    def accepts(self) -> np.ndarray:
        """``[a, b]`` is True when a answers in the first week and says yes to b."""
        return self.would_accept & self.responds[:, None]


def standardise(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rescale to mean 0 and spread 1 over the pairs that could be introduced."""
    values = x[mask]
    spread = values.std() if values.size else 0.0
    if spread == 0.0:
        return np.zeros_like(x, dtype=float)
    return (x - values.mean()) / spread


def feature_distances(codes_a: np.ndarray, codes_b: np.ndarray, name: str) -> np.ndarray:
    """Distance on one feature between every row person's value and every column person's."""
    j = FEATURE_NAMES.index(name)
    return FEATURES[name].distances[codes_a[:, j][:, None], codes_b[:, j][None, :]]


def fit_and_refusals(population: Population) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit by the viewer's true priorities, refusals on a profile, and refusals in person.

    A refusal on a profile needs a dealbreaker the profile shows: one of the
    four stated answers, clearly different from the viewer's own, or a
    different community when the viewer will not go outside theirs. A
    refusal in person uses every true dealbreaker and the other's true
    values, including whatever a profile did not show or did not say truly.
    """
    n = population.n
    distance_by_importance = np.zeros((n, n))
    on_profile = np.zeros((n, n), dtype=bool)
    in_person = np.zeros((n, n), dtype=bool)
    for j, name in enumerate(FEATURE_NAMES):
        true_gap = feature_distances(population.codes, population.codes, name)
        distance_by_importance += population.importance[:, j][:, None] * true_gap
        firm = (population.tiers[:, j] == _ALPHA)[:, None]
        in_person |= firm & (true_gap >= REFUSAL_DISTANCE)
        if name in DEALBREAKER_FEATURES:
            stated_gap = feature_distances(population.codes, population.stated_codes, name)
            on_profile |= firm & (stated_gap >= REFUSAL_DISTANCE)
    other_community = population.community[:, None] != population.community[None, :]
    outside = population.endogamous[:, None] & other_community
    return 1.0 - distance_by_importance, on_profile | outside, in_person | outside


def profile_attraction(
    scenario: Scenario, population: Population, fit: np.ndarray | None = None
) -> np.ndarray:
    """The part of attraction a perfect read of both profiles could explain.

    Personal fit is mixed from values, vibe, lifestyle and manner by
    ``attraction_mix``, then combined with the other person's general appeal
    by ``desirability_consensus``. Mean 0 and spread about 1 over pairs on
    opposite sides.
    """
    across = population.across
    if fit is None:
        fit, _, _ = fit_and_refusals(population)
    mix = scenario.attraction_mix
    parts = {
        "values": fit,
        "vibe": population.vibe @ population.vibe.T,
        "lifestyle": population.lifestyle @ population.lifestyle.T,
        "manner": population.manner @ population.manner.T,
    }
    personal = sum(np.sqrt(mix[name]) * standardise(part, across) for name, part in parts.items())
    consensus = scenario.desirability_consensus
    return (
        np.sqrt(1.0 - consensus) * standardise(personal, across)
        + np.sqrt(consensus) * population.appeal[None, :]
    )


def draw_chemistry(
    rng: np.random.Generator, scenario: Scenario, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Chemistry per ordered pair, and the part of it that both people feel.

    ``chemistry_mutuality`` of each person's chemistry is the pair's shared
    spark; the rest is their own.
    """
    upper = np.triu(rng.standard_normal((n, n)), k=1)
    shared = upper + upper.T
    own = rng.standard_normal((n, n))
    mutuality = scenario.chemistry_mutuality
    return np.sqrt(mutuality) * shared + np.sqrt(1.0 - mutuality) * own, shared


def _yes_rates(rng: np.random.Generator, scenario: Scenario, population: Population) -> np.ndarray:
    """Each person's own yes-rate: the scenario's, scaled by side, spread between people."""
    typical = np.where(
        population.majority, scenario.yes_rate, scenario.yes_rate * scenario.minority_yes_ratio
    )
    spread = scenario.yes_rate_spread
    # Log-normal with mean 1, so `yes_rate` stays the average whatever the spread.
    factor = np.exp(spread * rng.standard_normal(population.n) - spread**2 / 2.0)
    return np.clip(typical * factor, *YES_RATE_LIMITS)


def _bars(
    felt: np.ndarray, open_to: np.ndarray, eligible: np.ndarray, yes_rate: np.ndarray
) -> np.ndarray:
    """Each person's bar: the felt attraction above which they say yes.

    Placed so that the share of the person's eligible candidates they would
    accept equals their yes-rate. A refusal counts as a no, so someone with
    many dealbreakers has a lower bar for the people who survive them. With
    too few eligible candidates to place a bar, the pool-wide one is used.
    """
    n = felt.shape[0]
    pooled = np.sort(felt[eligible & open_to])
    pooled_open = (eligible & open_to).sum() / max(eligible.sum(), 1)
    bars = np.empty(n)
    for person in range(n):
        candidates = eligible[person]
        if candidates.sum() >= MIN_REFERENCE:
            reference = np.sort(felt[person, candidates & open_to[person]])
            open_share = reference.size / candidates.sum()
        else:
            reference, open_share = pooled, pooled_open
        # Among people not refused outright, the share that must clear the bar.
        needed = yes_rate[person] / open_share if open_share > 0 else 1.0
        if reference.size == 0 or needed >= 1.0:
            bars[person] = -np.inf
        else:
            bars[person] = np.quantile(reference, 1.0 - needed)
    return bars


def decide(rng: np.random.Generator, scenario: Scenario, population: Population) -> Decisions:
    """Draw every decision once, so a given introduction has one outcome whoever proposes it."""
    across = population.across
    fit, on_profile, in_person = fit_and_refusals(population)
    eligible = eligibility_matrix(population.codes) & across
    yes_rate = _yes_rates(rng, scenario, population)
    profile = profile_attraction(scenario, population, fit)
    chemistry, shared = draw_chemistry(rng, scenario, population.n)
    share, noise = scenario.chemistry_share, scenario.decision_noise
    lasting = np.sqrt(1.0 - share) * profile + np.sqrt(share) * chemistry
    mood = rng.standard_normal(lasting.shape)
    felt = np.sqrt(1.0 - noise) * lasting + np.sqrt(noise) * mood

    open_to = across & ~on_profile
    bar = _bars(felt, open_to, eligible, yes_rate)
    return Decisions(
        felt=felt,
        would_accept=open_to & (felt >= bar[:, None]),
        yes_rate=yes_rate,
        bar=bar,
        refuses=on_profile,
        refuses_in_person=in_person,
        eligible=eligible,
        profile=profile,
        chemistry=chemistry,
        shared_chemistry=shared,
        open_luck=rng.random((population.n, MAX_ROUNDS)),
        habit=population.habit,
    )
