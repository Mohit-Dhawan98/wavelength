"""Bidirectional matching: ranks, mutual top-N, probability, assignment."""

import numpy as np
import pytest

from wavelength.matching import (
    TOP_N,
    MutualPair,
    acceptance_probability,
    assign_by_probability,
    bidirectional_match,
    mutual_probability,
    mutual_top_n,
    rank_matrix,
)


def _no_self(n: int) -> np.ndarray:
    return ~np.eye(n, dtype=bool)


def test_top_n_is_ten():
    assert TOP_N == 10


def test_rank_decay_endpoints_and_formula():
    assert acceptance_probability(1) == 1.0
    assert acceptance_probability(10) == 0.5
    for rank in range(1, 11):
        assert acceptance_probability(rank) == pytest.approx(1 - 0.5 * (rank - 1) / 9)


def test_outside_top_n_has_zero_probability():
    assert acceptance_probability(11) == 0.0
    assert acceptance_probability(47) == 0.0
    assert acceptance_probability(0) == 0.0  # 0 means "not a candidate"
    assert mutual_probability(1, 11) == 0.0


def test_articles_examples():
    # Both rank each other #1.
    assert mutual_probability(1, 1) == 1.00
    # A ranks B #1, B ranks A #4.
    assert acceptance_probability(4) == pytest.approx(0.83, abs=0.005)
    assert mutual_probability(1, 4) == pytest.approx(0.83, abs=0.005)
    # The opening "Scenario B": #3 one way, #2 the other, about 84%.
    assert mutual_probability(3, 2) == pytest.approx(0.84, abs=0.005)


def test_mutual_probability_is_symmetric_in_the_pair():
    assert mutual_probability(2, 7) == mutual_probability(7, 2)


def test_rank_matrix_orders_each_users_own_list():
    scores = np.array(
        [
            [0.0, 0.9, 0.5, 0.7],
            [0.2, 0.0, 0.8, 0.4],
            [0.6, 0.6, 0.0, 0.1],
            [0.3, 0.2, 0.1, 0.0],
        ]
    )
    eligible = _no_self(4)
    eligible[0, 3] = eligible[3, 0] = False  # a hard-filter failure
    ranks = rank_matrix(scores, eligible)
    assert ranks.tolist() == [
        [0, 1, 2, 0],
        [3, 0, 1, 2],
        [1, 2, 0, 3],  # tie between 0 and 1 breaks to the lower index
        [0, 1, 2, 0],
    ]


def test_one_sided_interest_is_not_a_mutual_pair():
    # 0 ranks 1 first, but 1 ranks 0 last. With top_n=2 that is not mutual.
    scores = np.array(
        [
            [0.0, 0.9, 0.1, 0.2],
            [0.1, 0.0, 0.9, 0.8],
            [0.1, 0.9, 0.0, 0.8],
            [0.1, 0.8, 0.9, 0.0],
        ]
    )
    pairs = mutual_top_n(rank_matrix(scores, _no_self(4)), top_n=2)
    assert all(0 not in (p.a, p.b) for p in pairs)
    assert {(p.a, p.b) for p in pairs} == {(1, 2), (1, 3), (2, 3)}
    best = next(p for p in pairs if (p.a, p.b) == (1, 2))
    assert (best.rank_of_b_for_a, best.rank_of_a_for_b, best.probability) == (1, 1, 1.0)


def test_assignment_takes_highest_probability_first_and_uses_each_user_once():
    pairs = [
        MutualPair(0, 1, 1, 4, mutual_probability(1, 4)),  # 0.83
        MutualPair(1, 2, 1, 1, mutual_probability(1, 1)),  # 1.00, takes user 1
        MutualPair(0, 3, 2, 2, mutual_probability(2, 2)),  # 0.89
        MutualPair(2, 3, 3, 3, mutual_probability(3, 3)),  # blocked: 2 and 3 both used
    ]
    assigned = assign_by_probability(pairs)
    assert [(p.a, p.b) for p in assigned] == [(1, 2), (0, 3)]


def test_bidirectional_prefers_mutual_fit_over_one_sided_enthusiasm():
    """0's favourite is 1, but 1 and 2 are each other's favourite. 0 gets 3 instead."""
    scores = np.array(
        [
            [0.0, 0.95, 0.20, 0.88],
            [0.30, 0.0, 0.90, 0.10],
            [0.20, 0.90, 0.0, 0.10],
            [0.85, 0.20, 0.10, 0.0],
        ]
    )
    assigned = bidirectional_match(scores, _no_self(4), top_n=2)
    assert {(p.a, p.b) for p in assigned} == {(1, 2), (0, 3)}


def test_ineligible_pairs_are_never_matched():
    scores = np.array([[0.0, 0.99], [0.99, 0.0]])
    assert bidirectional_match(scores, np.zeros((2, 2), dtype=bool)) == []
