"""Hard filters stop a pair before it is scored."""

import numpy as np
import pytest

from wavelength.features import FEATURE_NAMES, FEATURES, encode
from wavelength.filters import (
    HARD_FILTERS,
    eligibility_matrix,
    hard_filter_violations,
    passes_hard_filters,
)
from wavelength.scoring import directed_score
from wavelength.weights import GLOBAL_WEIGHTS

from conftest import make_profile


def test_thresholds_are_the_articles():
    assert dict(HARD_FILTERS) == {
        "relationship_timeline": 0.2,
        "commitment_style": 0.3,
        "family_plans": 0.3,
        "trust_boundaries": 0.3,
    }


def test_identical_profiles_pass(base_features):
    assert passes_hard_filters(base_features, base_features)


@pytest.mark.parametrize(
    "feature, a, b",
    [
        ("relationship_timeline", "marriage_focused", "casual"),
        ("commitment_style", "requires_exclusivity", "open_relationship"),
        ("commitment_style", "prefers_monogamy", "open_relationship"),
        ("family_plans", "wants_children_soon", "childfree"),
        ("trust_boundaries", "needs_security", "very_flexible"),
    ],
)
def test_the_articles_dealbreaker_examples_fail(base_features, feature, a, b):
    left = {**base_features, feature: a}
    right = {**base_features, feature: b}
    violations = hard_filter_violations(left, right)
    assert [v.feature for v in violations] == [feature]
    assert violations[0].distance > violations[0].threshold
    assert not passes_hard_filters(right, left), "filters are symmetric"


def test_close_enough_values_pass(base_features):
    # "Prefers monogamy" and "requires exclusivity" are 0.15 apart: inside 0.3.
    other = {**base_features, "commitment_style": "prefers_monogamy"}
    assert passes_hard_filters(base_features, other)


def test_threshold_is_inclusive(base_features):
    # marriage_focused / long_term sits exactly on the 0.2 timeline threshold.
    other = {**base_features, "relationship_timeline": "long_term"}
    assert FEATURES["relationship_timeline"].distance("marriage_focused", "long_term") == 0.2
    assert passes_hard_filters(base_features, other)


def test_unknown_dealbreaker_fails_closed(base_features):
    other = {**base_features, "family_plans": None}
    violations = hard_filter_violations(base_features, other)
    assert [(v.feature, v.distance) for v in violations] == [("family_plans", None)]


def test_soft_features_never_trigger_a_filter(base_features):
    other = {**base_features, "conflict_resolution": "avoidance", "travel_frequency": "rare"}
    assert passes_hard_filters(base_features, other)


def test_a_high_score_does_not_rescue_a_dealbreaker():
    """The article's lawyer and consultant: alike in everything but kids and monogamy."""
    lawyer = make_profile("lawyer")
    consultant = make_profile(
        "consultant", family_plans="childfree", commitment_style="open_relationship"
    )
    assert directed_score(lawyer, consultant, GLOBAL_WEIGHTS) > 0.8
    failed = {v.feature for v in hard_filter_violations(lawyer.features, consultant.features)}
    assert failed == {"family_plans", "commitment_style"}


def test_eligibility_matrix_agrees_with_the_scalar_rule():
    rng = np.random.default_rng(0)
    profiles = [
        {name: str(rng.choice(FEATURES[name].values)) for name in FEATURE_NAMES} for _ in range(40)
    ]
    profiles[3]["trust_boundaries"] = None
    eligible = eligibility_matrix(np.array([encode(p) for p in profiles]))
    assert np.array_equal(eligible, eligible.T)
    assert not eligible.diagonal().any()
    for i, a in enumerate(profiles):
        for j, b in enumerate(profiles):
            if i != j:
                assert eligible[i, j] == passes_hard_filters(a, b)
