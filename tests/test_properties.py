"""Property checks on random inputs: invariants that must hold for any scores."""

import numpy as np
import pytest

from wavelength.baselines import one_directional_greedy
from wavelength.matching import bidirectional_match, rank_matrix


def random_round(seed: int):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 60))
    scores = rng.random((n, n))  # asymmetric on purpose: Score(a->b) != Score(b->a)
    eligible = rng.random((n, n)) < rng.uniform(0.05, 0.9)
    eligible = eligible & eligible.T  # a hard filter passes or fails for the pair
    np.fill_diagonal(eligible, False)
    top_n = int(rng.integers(2, 12))
    return scores, eligible, top_n


@pytest.mark.parametrize("seed", range(40))
def test_bidirectional_never_breaks_a_filter_or_double_books(seed):
    scores, eligible, top_n = random_round(seed)
    ranks = rank_matrix(scores, eligible)
    seen: set[int] = set()
    for pair in bidirectional_match(scores, eligible, top_n):
        assert eligible[pair.a, pair.b]
        assert 1 <= ranks[pair.a, pair.b] <= top_n
        assert 1 <= ranks[pair.b, pair.a] <= top_n
        assert pair.a not in seen and pair.b not in seen
        seen.update((pair.a, pair.b))


@pytest.mark.parametrize("seed", range(40))
def test_bidirectional_assigns_in_descending_probability(seed):
    scores, eligible, top_n = random_round(seed)
    probabilities = [p.probability for p in bidirectional_match(scores, eligible, top_n)]
    assert probabilities == sorted(probabilities, reverse=True)


@pytest.mark.parametrize("seed", range(40))
def test_one_directional_never_breaks_a_filter_or_double_books(seed):
    scores, eligible, _ = random_round(seed)
    order = np.random.default_rng(seed).permutation(scores.shape[0]).tolist()
    seen: set[int] = set()
    for a, b in one_directional_greedy(scores, eligible, order):
        assert eligible[a, b]
        assert a not in seen and b not in seen
        seen.update((a, b))
