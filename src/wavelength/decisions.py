"""Who would really say yes to whom.

This is the second half of the hidden truth, and the part that keeps the
simulation from being circular. If simulated people accepted according to
the matcher's own score, any matcher would look perfect. So the decision
rule here is built from the true quantities the matcher never sees, and
differs from the matcher's score in form as well as inputs:

- it uses each person's true feature values and true importance weights,
  not the extracted values and learned weights;
- a true dealbreaker is a refusal, not a large term in an average;
- part of attraction is the other person's general appeal, which this
  repository's matchers have no signal for;
- part is chemistry, some of it shared by the pair, which nothing can
  observe;
- part of any single decision is mood on the day.

A person says yes when what they feel clears their own bar. The bar is an
absolute level of attraction, fixed per person before any matcher runs. It
is placed so that the person would say yes to their own ``yes_rate`` share
of eligible people picked at random. Being shown better-than-random people
therefore produces more yeses, and by how much is not set anywhere: it
follows from how well a matcher's ranking tracks true attraction.

Responding is separate from deciding. ``would_accept`` is the decision each
person would make; ``responds`` says whether they answer at all.

The simulation shows what follows from these premises in a given world. It
cannot show that the premises are true.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelength.features import FEATURE_NAMES, FEATURES
from wavelength.filters import eligibility_matrix
from wavelength.population import TIERS, Population
from wavelength.scenario import Scenario
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

_ALPHA = TIERS.index(Tier.ALPHA)


@dataclass(frozen=True, eq=False)
class Decisions:
    """What each person would do if introduced to each other person."""

    felt: np.ndarray  # (n, n) what row feels about column on the day: attraction plus mood
    would_accept: np.ndarray  # (n, n) bool: row's decision about column, if row responds
    responds: np.ndarray  # (n,) bool: answers introductions at all
    yes_rate: np.ndarray  # (n,) each person's own yes-rate to a random eligible person
    bar: np.ndarray  # (n,) the level of felt attraction each person needs to say yes
    refuses: np.ndarray  # (n, n) bool: row has a true dealbreaker against column
    eligible: np.ndarray  # (n, n) bool: opposite sides and clear of the hard filters, truly

    @property
    def accepts(self) -> np.ndarray:
        """``[a, b]`` is True when a responds and says yes to b."""
        return self.would_accept & self.responds[:, None]


def _standardise(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rescale to mean 0 and spread 1 over the pairs that could be introduced."""
    values = x[mask]
    spread = values.std()
    if spread == 0.0:
        return np.zeros_like(x)
    return (x - values.mean()) / spread


def _fit_and_refusals(population: Population) -> tuple[np.ndarray, np.ndarray]:
    """Agreement on features by the viewer's true weights, and true dealbreaker hits."""
    n = population.n
    weighted_distance = np.zeros((n, n))
    refuses = np.zeros((n, n), dtype=bool)
    for j, name in enumerate(FEATURE_NAMES):
        column = population.codes[:, j]
        d = FEATURES[name].distances[column[:, None], column[None, :]]
        weighted_distance += population.weights[:, j][:, None] * d
        firm = (population.tiers[:, j] == _ALPHA)[:, None]
        refuses |= firm & (d >= REFUSAL_DISTANCE)
    fit = 1.0 - weighted_distance / population.weights.sum(axis=1, keepdims=True)
    return fit, refuses


def profile_attraction(
    scenario: Scenario, population: Population, fit: np.ndarray | None = None
) -> np.ndarray:
    """The part of attraction a perfect read of both profiles could explain.

    Personal fit is mixed from values, vibe and lifestyle by
    ``attraction_mix``, then combined with the other person's general appeal
    by ``desirability_consensus``. Mean 0 and spread about 1 over pairs on
    opposite sides.
    """
    across = population.across
    if fit is None:
        fit, _ = _fit_and_refusals(population)
    mix = scenario.attraction_mix
    personal = (
        np.sqrt(mix["values"]) * _standardise(fit, across)
        + np.sqrt(mix["vibe"]) * _standardise(population.vibe @ population.vibe.T, across)
        + np.sqrt(mix["lifestyle"])
        * _standardise(population.lifestyle @ population.lifestyle.T, across)
    )
    consensus = scenario.desirability_consensus
    return (
        np.sqrt(1.0 - consensus) * _standardise(personal, across)
        + np.sqrt(consensus) * population.appeal[None, :]
    )


def attraction(
    rng: np.random.Generator,
    scenario: Scenario,
    population: Population,
    fit: np.ndarray | None = None,
) -> np.ndarray:
    """How drawn each person (row) is to each other person (column), lastingly.

    ``profile_attraction`` plus chemistry, by ``chemistry_share``. Chemistry
    is drawn per pair; ``chemistry_mutuality`` of it is the same in both
    directions and the rest is each person's own.
    """
    n = population.n
    one_way = rng.standard_normal((n, n))
    upper = np.triu(rng.standard_normal((n, n)), k=1)
    both_ways = upper + upper.T
    mutuality = scenario.chemistry_mutuality
    chemistry = np.sqrt(mutuality) * both_ways + np.sqrt(1.0 - mutuality) * one_way
    share = scenario.chemistry_share
    explained = profile_attraction(scenario, population, fit)
    return np.sqrt(1.0 - share) * explained + np.sqrt(share) * chemistry


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
    fit, refuses = _fit_and_refusals(population)
    eligible = eligibility_matrix(population.codes) & across
    yes_rate = _yes_rates(rng, scenario, population)
    lasting = attraction(rng, scenario, population, fit)
    noise = scenario.decision_noise
    mood = rng.standard_normal(lasting.shape)
    felt = np.sqrt(1.0 - noise) * lasting + np.sqrt(noise) * mood

    open_to = across & ~refuses
    bar = _bars(felt, open_to, eligible, yes_rate)
    return Decisions(
        felt=felt,
        would_accept=open_to & (felt >= bar[:, None]),
        responds=population.active,
        yes_rate=yes_rate,
        bar=bar,
        refuses=refuses,
        eligible=eligible,
    )
