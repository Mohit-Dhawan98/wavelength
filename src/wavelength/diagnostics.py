"""Facts about a simulated world that a reader needs before reading any score.

A strategy's numbers mean little without knowing the world they came from.
Two things in particular can make a comparison empty, and neither shows up
in an acceptance rate:

- If a typical person has only twenty eligible candidates, their "top 10"
  is half of everyone they could be shown, and mutual top-N selects almost
  nothing.
- If the matcher's score barely tracks what people actually feel, no way of
  using that score can do much, and differences between strategies are
  mostly noise.

So every run measures both, along with whether people in the generated
world say yes as often as the scenario asked for. These are properties of
the world and of what the app can see in it, not of any strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelength.strategies import global_scores, observed_eligibility, personal_scores
from wavelength.view import Board
from wavelength.world import World

#: A person needs at least this many candidates for a rank correlation to mean anything.
_MIN_CANDIDATES = 5


@dataclass(frozen=True)
class WorldCheck:
    """Measured properties of one world."""

    yes_rate_realised: float
    """Share of truly eligible pairs in which the row person would say yes: the
    realised yes-rate to a random eligible introduction, averaged over both sides."""

    responding_share: float
    """Share of people who respond to introductions at all."""

    matchable_share: float
    """Share of people with at least one candidate after the hard filters, on what the app knows."""

    candidates_majority: float
    """Median number of eligible candidates, on what the app knows, for a matchable
    person on the larger side."""

    candidates_minority: float
    """The same for a matchable person on the scarce side."""

    score_tracks_feeling: float
    """Mean per-person rank correlation between the matcher's score (per-user
    weights) and what the person actually feels, over their eligible candidates."""

    global_score_tracks_feeling: float
    """The same for the score with global weights."""

    text_tracks_feeling: float
    """The same for the text channel alone."""

    photo_tracks_feeling: float
    """The same for the photo channel alone."""

    voice_tracks_feeling: float
    """The same for the voice channel alone."""

    features_personal_track_feeling: float
    """The same for agreement on structured features by each person's learned weights."""

    features_global_track_feeling: float
    """The same for agreement on structured features by one global set of weights."""

    profile_tracks_feeling: float
    """The same for a perfect read of both profiles including general appeal: the
    most any score built from profiles could reach in this world."""


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size)
    ranks[order] = np.arange(values.size)
    return ranks


def mean_rank_correlation(score: np.ndarray, truth: np.ndarray, candidates: np.ndarray) -> float:
    """Mean over people of the Spearman correlation of two rows, on each person's candidates."""
    correlations = []
    for person in range(score.shape[0]):
        chosen = candidates[person]
        if chosen.sum() < _MIN_CANDIDATES:
            continue
        a, b = _ranks(score[person, chosen]), _ranks(truth[person, chosen])
        if a.std() == 0.0 or b.std() == 0.0:
            continue
        correlations.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(correlations)) if correlations else float("nan")


def check_world(world: World) -> WorldCheck:
    """Measure one world."""
    population, decisions, obs = world.population, world.decisions, world.observations
    seen_eligible = observed_eligibility(obs)
    n_candidates = seen_eligible.sum(axis=1)
    matchable = n_candidates > 0

    def median_candidates(side: np.ndarray) -> float:
        chosen = n_candidates[side & matchable]
        return float(np.median(chosen)) if chosen.size else 0.0

    truly_eligible = decisions.eligible
    felt = decisions.felt
    board = Board(obs.copy())

    def tracks(score: np.ndarray) -> float:
        return mean_rank_correlation(score, felt, seen_eligible)

    return WorldCheck(
        yes_rate_realised=float(
            (decisions.would_accept & truly_eligible).sum() / max(truly_eligible.sum(), 1)
        ),
        responding_share=float(decisions.responds.mean()),
        matchable_share=float(matchable.mean()),
        candidates_majority=median_candidates(population.majority),
        candidates_minority=median_candidates(~population.majority),
        score_tracks_feeling=mean_rank_correlation(personal_scores(obs), felt, seen_eligible),
        global_score_tracks_feeling=mean_rank_correlation(global_scores(obs), felt, seen_eligible),
        text_tracks_feeling=tracks(board.text),
        photo_tracks_feeling=tracks(board.photo),
        voice_tracks_feeling=tracks(board.voice),
        features_personal_track_feeling=tracks(board.feature_similarity(obs.weights)),
        features_global_track_feeling=tracks(board.feature_similarity(board.global_weights)),
        profile_tracks_feeling=tracks(decisions.profile),
    )
