"""The taxonomy and its distance matrices."""

import numpy as np
import pytest

from wavelength.features import FEATURE_NAMES, FEATURES, MISSING, Group, distance, encode
from wavelength.filters import HARD_FILTERS


def test_three_groups_are_all_populated():
    groups = {f.group for f in FEATURES.values()}
    assert groups == {Group.CORE_VALUES, Group.PERSONALITY, Group.LIFESTYLE}
    assert 10 <= len(FEATURES) <= 12


def test_hard_filter_features_are_core_values():
    for name in HARD_FILTERS:
        assert FEATURES[name].group is Group.CORE_VALUES


@pytest.mark.parametrize("name", FEATURE_NAMES)
def test_every_matrix_is_a_valid_graduated_distance(name):
    m = FEATURES[name].distances
    assert m.shape == (len(FEATURES[name].values),) * 2
    assert np.array_equal(m, m.T), "distance must not depend on who is asked"
    assert np.all(np.diag(m) == 0.0), "identical values are at distance 0"
    off_diagonal = m[~np.eye(len(m), dtype=bool)]
    assert np.all((off_diagonal > 0.0) & (off_diagonal <= 1.0))


def test_conflict_resolution_matrix_is_the_articles():
    f = FEATURES["conflict_resolution"]
    assert f.values == ("direct", "time_first", "avoidance")
    expected = np.array([[0.0, 0.3, 0.9], [0.3, 0.0, 0.7], [0.9, 0.7, 0.0]])
    assert np.array_equal(f.distances, expected)


def test_different_is_not_the_same_as_incompatible():
    # Complementary styles sit close; engaging versus avoiding sits far.
    assert distance("conflict_resolution", "direct", "time_first") == 0.3
    assert distance("conflict_resolution", "direct", "avoidance") == 0.9


def test_commitment_distances_quoted_in_the_article():
    assert distance("commitment_style", "prefers_monogamy", "requires_exclusivity") == 0.15
    assert distance("commitment_style", "prefers_monogamy", "open_relationship") == 0.8


def test_unknown_value_is_rejected():
    with pytest.raises(ValueError):
        distance("family_plans", "wants_children_soon", "maybe")


def test_matrices_are_read_only():
    with pytest.raises(ValueError):
        FEATURES["family_plans"].distances[0, 1] = 0.0


def test_encode_marks_absent_features_as_missing():
    codes = encode({"family_plans": "childfree", "social_energy": None})
    assert codes[FEATURE_NAMES.index("family_plans")] == 3
    assert codes[FEATURE_NAMES.index("social_energy")] == MISSING
    assert (codes == MISSING).sum() == len(FEATURE_NAMES) - 1
