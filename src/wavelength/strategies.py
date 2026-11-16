"""The article's three strategies, as functions of what the app can see.

Each takes ``Observations`` and proposes one round of introductions. None of
them can reach the hidden truth: the type they are given does not contain it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from wavelength.baselines import one_directional_greedy, similarity_scores
from wavelength.features import FEATURE_NAMES
from wavelength.matching import RankCurve, bidirectional_match
from wavelength.observation import Observations
from wavelength.scoring import score_matrix
from wavelength.weights import GLOBAL_WEIGHTS


@dataclass(frozen=True)
class Proposal:
    """One proposed introduction.

    ``predicted`` is P(mutual) and the two ranks are each person's position
    in the other's list, when the strategy has them (ranks are 0 otherwise).
    """

    a: int
    b: int
    predicted: float | None = None
    rank_of_b_for_a: int = 0
    rank_of_a_for_b: int = 0


#: A strategy maps (observations, visiting order, top_n) to one round of proposals.
Strategy = Callable[[Observations, np.ndarray, int], list[Proposal]]


def similarity_only(obs: Observations, order: np.ndarray, top_n: int) -> list[Proposal]:
    """Stage 1: text similarity, each user's best available. No features, no hard filters.

    The only restriction is the side of the market, which the article's
    Stage 1 also applied as a metadata filter.
    """
    pairs = one_directional_greedy(similarity_scores(obs.text_embeddings), obs.across, order)
    return [Proposal(a, b) for a, b in pairs]


def global_scores(obs: Observations) -> np.ndarray:
    """Tri-signal scores with the same global weights for every user."""
    weights = np.tile([GLOBAL_WEIGHTS[f] for f in FEATURE_NAMES], (obs.codes.shape[0], 1))
    return score_matrix(obs.codes, weights, obs.text_embeddings, obs.visual_embeddings)


def personal_scores(obs: Observations) -> np.ndarray:
    """Tri-signal scores with each user's own learned weights."""
    return score_matrix(obs.codes, obs.weights, obs.text_embeddings, obs.visual_embeddings)


def observed_eligibility(obs: Observations) -> np.ndarray:
    """Pairs on opposite sides that break no dealbreaker the app knows about."""
    return obs.known_ok


def filtered_global(obs: Observations, order: np.ndarray, top_n: int) -> list[Proposal]:
    """Stage 2: hard filters, then tri-signal scores with global weights, one-directional."""
    pairs = one_directional_greedy(global_scores(obs), observed_eligibility(obs), order)
    return [Proposal(a, b) for a, b in pairs]


def bidirectional(
    obs: Observations, order: np.ndarray, top_n: int, curve: RankCurve | None = None
) -> list[Proposal]:
    """Stage 3: hard filters, per-user weights, mutual top-N, assignment by P(mutual).

    ``curve`` replaces the article's ``P(rank)`` for both the order of
    assignment and the predicted probability.
    """
    assigned = bidirectional_match(personal_scores(obs), observed_eligibility(obs), top_n, curve)
    return [
        Proposal(p.a, p.b, p.probability, p.rank_of_b_for_a, p.rank_of_a_for_b) for p in assigned
    ]


#: The three strategies, keyed by a short name, with the label used in reports.
STRATEGIES: dict[str, tuple[str, Strategy]] = {
    "similarity": ("1 similarity only", similarity_only),
    "filtered_global": ("2 features + hard filters, global weights", filtered_global),
    "bidirectional": ("3 per-user weights + bidirectional", bidirectional),
}

#: The extra row a calibrated run adds: strategy 3 again, with the fitted curve.
FITTED_KEY = "bidirectional_fitted"
FITTED_LABEL = "3 again, with the fitted P(rank)"

#: Strategies that apply the hard filters, so must never propose a violating pair.
FILTERED_STRATEGIES: frozenset[str] = frozenset({"filtered_global", "bidirectional", FITTED_KEY})
