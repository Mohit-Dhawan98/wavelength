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
- part is chemistry, drawn per ordered pair, which nothing can observe;
- each person has their own bar, set so that they say yes to their own
  share of the other side.

The last point builds in the article's premise that people say yes to those
near the top of their own list. The simulation shows what follows from that
premise in a given world. It cannot show that the premise is true.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelength.features import FEATURE_NAMES, FEATURES
from wavelength.population import TIERS, Population
from wavelength.scenario import Scenario
from wavelength.weights import Tier

#: A person for whom a feature is a true dealbreaker refuses anyone at least
#: this far from them on it. The matcher's hard-filter thresholds (0.2 and
#: 0.3) are tighter, so a pair that passes the filters on true values never
#: triggers a refusal on those four features.
REFUSAL_DISTANCE = 0.5

_ALPHA = TIERS.index(Tier.ALPHA)


@dataclass(frozen=True, eq=False)
class Decisions:
    """What each person would do if introduced to each other person."""

    accepts: np.ndarray  # (n, n) bool: row would say yes to column
    yes_share: np.ndarray  # (n,) share of the other side each person would accept
    refuses: np.ndarray  # (n, n) bool: row has a true dealbreaker against column


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


def attraction(rng: np.random.Generator, scenario: Scenario, population: Population) -> np.ndarray:
    """How drawn each person (row) is to each other person (column).

    Built in layers, each a share of the variance of the layer above:
    personal fit is mixed from values, vibe and lifestyle by
    ``attraction_mix``; observable attraction is personal fit plus general
    appeal by ``desirability_consensus``; total attraction is that plus
    chemistry by ``chemistry_share``.
    """
    across = population.across
    fit, _ = _fit_and_refusals(population)
    mix = scenario.attraction_mix
    personal = (
        np.sqrt(mix["values"]) * _standardise(fit, across)
        + np.sqrt(mix["vibe"]) * _standardise(population.vibe @ population.vibe.T, across)
        + np.sqrt(mix["lifestyle"])
        * _standardise(population.lifestyle @ population.lifestyle.T, across)
    )
    consensus, chemistry = scenario.desirability_consensus, scenario.chemistry_share
    observable = (
        np.sqrt(1.0 - consensus) * _standardise(personal, across)
        + np.sqrt(consensus) * population.appeal[None, :]
    )
    spark = rng.standard_normal((population.n, population.n))
    return np.sqrt(1.0 - chemistry) * observable + np.sqrt(chemistry) * spark


def _yes_shares(rng: np.random.Generator, scenario: Scenario, population: Population) -> np.ndarray:
    """Each person's own yes-rate: typical pickiness, scaled by side, spread between people."""
    typical = np.where(
        population.majority, scenario.pickiness, scenario.pickiness * scenario.minority_yes_ratio
    )
    spread = scenario.pickiness_spread
    # Log-normal with mean 1, so `pickiness` stays the average whatever the spread.
    factor = np.exp(spread * rng.standard_normal(population.n) - spread**2 / 2.0)
    return np.clip(typical * factor, 0.0, 0.95)


def decide(rng: np.random.Generator, scenario: Scenario, population: Population) -> Decisions:
    """Draw every decision once, so a given introduction has one outcome whoever proposes it."""
    across = population.across
    _, refuses = _fit_and_refusals(population)
    yes_share = _yes_shares(rng, scenario, population)
    drawn = attraction(rng, scenario, population)

    # Each person says yes to the top `yes_share` of the other side by their
    # own attraction, never to someone they have a true dealbreaker against,
    # and not at all if they are inactive.
    open_to = across & ~refuses
    ranked = np.where(open_to, drawn, -np.inf)
    n_across = across.sum(axis=1)
    n_yes = np.clip(np.rint(yes_share * n_across).astype(int), 1, np.maximum(n_across, 1))
    by_attraction = -np.sort(-ranked, axis=1)
    bar = by_attraction[np.arange(population.n), n_yes - 1]
    accepts = open_to & (ranked >= bar[:, None]) & population.active[:, None]
    return Decisions(accepts=accepts, yes_share=yes_share, refuses=refuses)
