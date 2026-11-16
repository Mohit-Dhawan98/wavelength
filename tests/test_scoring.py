"""Tri-signal scoring and its asymmetry."""

import numpy as np
import pytest

from wavelength.features import FEATURE_NAMES, FEATURES, encode
from wavelength.scoring import (
    FEATURE_WEIGHT,
    NEUTRAL,
    TEXT_WEIGHT,
    VISUAL_WEIGHT,
    Profile,
    cosine_similarity01,
    directed_score,
    feature_similarity,
    score_matrix,
    tri_signal_score,
)
from wavelength.weights import GLOBAL_WEIGHTS

from conftest import make_profile


def test_signal_mix_is_the_articles():
    assert (TEXT_WEIGHT, VISUAL_WEIGHT, FEATURE_WEIGHT) == (0.25, 0.15, 0.60)
    assert tri_signal_score(1.0, 0.0, 0.0) == 0.25
    assert tri_signal_score(0.0, 1.0, 0.0) == 0.15
    assert tri_signal_score(0.0, 0.0, 1.0) == 0.60
    assert tri_signal_score(1.0, 1.0, 1.0) == pytest.approx(1.0)


def test_cosine_is_rescaled_to_unit_interval():
    u = np.array([1.0, 0.0])
    assert cosine_similarity01(u, u) == pytest.approx(1.0)
    assert cosine_similarity01(u, np.array([0.0, 3.0])) == pytest.approx(0.5)
    assert cosine_similarity01(u, -u) == pytest.approx(0.0)


def test_identical_profiles_score_one():
    a = make_profile("a")
    assert directed_score(a, make_profile("b"), GLOBAL_WEIGHTS) == pytest.approx(1.0)


def test_feature_similarity_is_one_minus_weighted_mean_distance(base_features):
    other = {**base_features, "conflict_resolution": "avoidance"}  # distance 0.9
    weights = dict(GLOBAL_WEIGHTS)
    expected = 1.0 - weights["conflict_resolution"] * 0.9 / sum(weights.values())
    assert feature_similarity(weights, base_features, other) == pytest.approx(expected)


def test_missing_features_are_skipped_not_penalised(base_features):
    partial = {**base_features, "travel_frequency": None}
    assert feature_similarity(GLOBAL_WEIGHTS, base_features, partial) == pytest.approx(1.0)
    nothing = dict.fromkeys(base_features)
    assert feature_similarity(GLOBAL_WEIGHTS, base_features, nothing) == NEUTRAL


def test_same_gap_different_weights_gives_asymmetric_scores():
    """The article's example: A weights intellectual connection 9.9, B weights it 2.0."""
    a = make_profile("a", intellectual_connection="high")
    b = make_profile("b", intellectual_connection="moderate")
    weights_a = {**GLOBAL_WEIGHTS, "intellectual_connection": 9.9}
    weights_b = {**GLOBAL_WEIGHTS, "intellectual_connection": 2.0}

    a_to_b = directed_score(a, b, weights_a)
    b_to_a = directed_score(b, a, weights_b)

    assert a_to_b < b_to_a, "the gap costs A more because A cares about it more"
    # Swapping who is viewer and who is candidate changes nothing by itself:
    # the asymmetry comes from the weights alone.
    assert directed_score(a, b, weights_a) == pytest.approx(directed_score(b, a, weights_a))


def test_score_matrix_matches_the_scalar_reference():
    rng = np.random.default_rng(1)
    n = 25
    profiles, weights = [], rng.uniform(1.0, 10.0, size=(n, len(FEATURE_NAMES)))
    for i in range(n):
        features = {
            name: (str(rng.choice(FEATURES[name].values)) if rng.random() < 0.8 else None)
            for name in FEATURE_NAMES
        }
        profiles.append(
            Profile(str(i), features, rng.standard_normal(8), rng.standard_normal(4))
        )

    matrix = score_matrix(
        np.array([encode(p.features) for p in profiles]),
        weights,
        np.array([p.text_embedding for p in profiles]),
        np.array([p.visual_embedding for p in profiles]),
    )

    for i in range(n):
        w = dict(zip(FEATURE_NAMES, weights[i]))
        for j in range(n):
            if i != j:
                assert matrix[i, j] == pytest.approx(directed_score(profiles[i], profiles[j], w))
    assert not np.allclose(matrix, matrix.T), "per-user weights make the matrix asymmetric"
