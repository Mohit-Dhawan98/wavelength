"""Bidirectional matching: optimise for both people saying yes.

A ranking answers "who is best for A?". A match needs a different question
answered: "where are A and B each other's best?". A's top-ranked candidate is
a wasted introduction if A sits far down that candidate's own list, and
being ignored after saying yes is a worse experience than no match at all.

The procedure, from the article:

1. Rank every user's candidates from that user's own perspective.
2. Keep only pairs where each is in the other's top N (N = 10).
3. Give each kept pair a success probability,
   ``P(mutual) = P(A accepts) x P(B accepts)``, where
   ``P(rank) = 1 - 0.5 x (rank - 1) / (N - 1)``.
4. Assign pairs in descending probability; each user is matched at most
   once per round.

This is not Gale-Shapley: there is no propose-and-reject cycle and no
stability guarantee. It is a greedy approximation to maximum-weight matching
over the mutual top-N graph.

Two modelling assumptions are worth keeping in view. The product in step 3
treats the two decisions as independent, which real acceptance is not. And
the decay in ``P(rank)`` is a single hand-set curve, not a fitted one, so its
calibration has to be checked against real outcomes before it is trusted.

For that reason the curve is replaceable. Every function that turns ranks
into probabilities takes an optional ``curve``: a sequence of N values where
``curve[r - 1]`` is P(accept) at rank ``r``. Leaving it out gives the
article's formula. ``calibration.py`` fits such a curve to observed outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

TOP_N = 10

#: P(accept) by rank: ``curve[r - 1]`` is the probability at rank ``r``.
RankCurve = Sequence[float]


@dataclass(frozen=True)
class MutualPair:
    """Two users who each rank the other in their top N."""

    a: int
    b: int
    rank_of_b_for_a: int
    rank_of_a_for_b: int
    probability: float


def acceptance_probability(rank: int, top_n: int = TOP_N) -> float:
    """P(a user accepts the candidate at this rank in their list).

    Linear decay from 1.0 at rank 1 to 0.5 at rank N. Outside the top N the
    model says 0.0: not "this could never work" but "we will not bet an
    introduction on it".
    """
    if top_n < 2:
        raise ValueError("top_n must be at least 2")
    if rank < 1 or rank > top_n:
        return 0.0
    return 1.0 - 0.5 * (rank - 1) / (top_n - 1)


def article_curve(top_n: int = TOP_N) -> tuple[float, ...]:
    """The article's hand-set ``P(rank)`` as a curve, one value per rank."""
    return tuple(acceptance_probability(rank, top_n) for rank in range(1, top_n + 1))


def _checked_curve(curve: RankCurve | None, top_n: int) -> RankCurve:
    if curve is None:
        return article_curve(top_n)
    if len(curve) != top_n:
        raise ValueError(f"a curve needs one value per rank: expected {top_n}, got {len(curve)}")
    if any(not 0.0 <= p <= 1.0 for p in curve):
        raise ValueError("curve values are probabilities and must lie in [0, 1]")
    return curve


def _curve_value(curve: RankCurve, rank: int) -> float:
    return float(curve[rank - 1]) if 1 <= rank <= len(curve) else 0.0


def mutual_probability(
    rank_of_b_for_a: int,
    rank_of_a_for_b: int,
    top_n: int = TOP_N,
    curve: RankCurve | None = None,
) -> float:
    """P(both accept), assuming the two decisions are independent.

    ``curve`` replaces the article's ``P(rank)``; see the module docstring.
    """
    if curve is None:
        return acceptance_probability(rank_of_b_for_a, top_n) * acceptance_probability(
            rank_of_a_for_b, top_n
        )
    curve = _checked_curve(curve, top_n)
    return _curve_value(curve, rank_of_b_for_a) * _curve_value(curve, rank_of_a_for_b)


def rank_matrix(scores: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Per-user rankings as a matrix.

    ``scores[a, b]`` is Score(a -> b). ``ranks[a, b]`` is the 1-based position
    of ``b`` in ``a``'s list, or 0 when ``b`` is not a candidate for ``a``
    (failed a hard filter, or is ``a``). Ties break towards the lower index
    so the result is deterministic.
    """
    n = scores.shape[0]
    masked = np.where(eligible, scores, -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=np.int64)
    ranks[np.arange(n)[:, None], order] = np.arange(1, n + 1)[None, :]
    ranks[~eligible] = 0
    return ranks


def mutual_top_n(
    ranks: np.ndarray, top_n: int = TOP_N, curve: RankCurve | None = None
) -> list[MutualPair]:
    """Every pair where each user is in the other's top N, with its probability.

    A pair where A ranks B first and B does not rank A in the top N is
    dropped here, however much A would like the match. ``curve`` replaces
    the article's ``P(rank)`` when the probability is computed.
    """
    if curve is not None:
        curve = _checked_curve(curve, top_n)
    in_top = (ranks >= 1) & (ranks <= top_n)
    mutual = np.triu(in_top & in_top.T, k=1)
    pairs = []
    for a, b in np.argwhere(mutual):
        r_ab, r_ba = int(ranks[a, b]), int(ranks[b, a])
        probability = mutual_probability(r_ab, r_ba, top_n, curve)
        pairs.append(MutualPair(int(a), int(b), r_ab, r_ba, probability))
    return pairs


def assign_by_probability(
    pairs: Iterable[MutualPair], curve: RankCurve | None = None
) -> list[MutualPair]:
    """Greedy assignment: best pair first, each user used at most once.

    Sorting by probability first is what makes this differ from handing each
    user their own favourite. A user can lose their #1 to someone for whom
    the feeling is mutual, and be given the best pairing still available.

    With ``curve``, each pair's probability is recomputed from its two ranks
    by that curve before sorting, so a fitted curve changes the order of
    assignment and not only the number reported.
    """
    if curve is not None:
        top_n = len(curve)
        pairs = [
            replace(
                p,
                probability=mutual_probability(p.rank_of_b_for_a, p.rank_of_a_for_b, top_n, curve),
            )
            for p in pairs
        ]
    matched: set[int] = set()
    assigned: list[MutualPair] = []
    for pair in sorted(pairs, key=lambda p: (-p.probability, p.a, p.b)):
        if pair.a in matched or pair.b in matched:
            continue
        matched.update((pair.a, pair.b))
        assigned.append(pair)
    return assigned


def bidirectional_match(
    scores: np.ndarray,
    eligible: np.ndarray,
    top_n: int = TOP_N,
    curve: RankCurve | None = None,
) -> list[MutualPair]:
    """One round of matching: rank, intersect, score by probability, assign.

    Users with no mutual top-N pair are left unmatched this round. That is
    deliberate: the alternative is an introduction the model expects to fail.
    """
    return assign_by_probability(mutual_top_n(rank_matrix(scores, eligible), top_n, curve))
