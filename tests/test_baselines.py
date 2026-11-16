"""The baselines behave as the naive strategies they stand in for."""

import numpy as np

from wavelength.baselines import one_directional_greedy, similarity_scores


def test_similarity_scores_are_symmetric_and_bounded():
    embeddings = np.random.default_rng(0).standard_normal((12, 5))
    s = similarity_scores(embeddings)
    assert np.allclose(s, s.T)
    assert np.allclose(np.diag(s), 1.0)
    assert s.min() >= 0.0 and s.max() <= 1.0 + 1e-12


def test_greedy_gives_the_first_user_their_favourite_regardless_of_reciprocity():
    # 0 adores 1; 1 would much rather have 2. Greedy pairs 0 with 1 anyway.
    scores = np.array(
        [
            [0.0, 0.95, 0.20, 0.88],
            [0.30, 0.0, 0.90, 0.10],
            [0.20, 0.90, 0.0, 0.10],
            [0.85, 0.20, 0.10, 0.0],
        ]
    )
    pairs = one_directional_greedy(scores, ~np.eye(4, dtype=bool), order=[0, 1, 2, 3])
    assert pairs == [(0, 1), (2, 3)]


def test_greedy_respects_eligibility_and_matches_each_user_once():
    rng = np.random.default_rng(3)
    n = 31
    scores = rng.random((n, n))
    eligible = rng.random((n, n)) < 0.3
    eligible = eligible & eligible.T & ~np.eye(n, dtype=bool)
    pairs = one_directional_greedy(scores, eligible, order=rng.permutation(n))
    users = [u for pair in pairs for u in pair]
    assert len(users) == len(set(users))
    assert all(eligible[a, b] for a, b in pairs)
