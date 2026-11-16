"""Per-user weights: tiers, source priority, and provenance."""

import itertools

import pytest

from wavelength.weights import (
    GLOBAL_WEIGHTS,
    TIER_BANDS,
    Source,
    Tier,
    WeightProfile,
    WeightSignal,
)

FEATURE = "intellectual_connection"

WEEK_1 = WeightSignal(FEATURE, 5.0, Tier.BETA, Source.ONBOARDING, 0.8, 1, "I like smart people")
WEEK_3 = WeightSignal(
    FEATURE, 7.5, Tier.BETA, Source.MATCH_FEEDBACK, 0.8, 3, "They weren't intellectually stimulating"
)
WEEK_8 = WeightSignal(
    FEATURE,
    9.0,
    Tier.ALPHA,
    Source.POST_DATE_FEEDBACK,
    0.8,
    8,
    "The intellectual connection was everything",
)


def test_tier_bands_are_the_articles():
    assert TIER_BANDS[Tier.ALPHA] == (9.5, 10.0)
    assert TIER_BANDS[Tier.BETA] == (5.0, 7.0)
    assert TIER_BANDS[Tier.GAMMA] == (1.0, 3.0)


def test_source_priority_order():
    assert Source.ONBOARDING < Source.MATCH_FEEDBACK < Source.POST_DATE_FEEDBACK


def test_the_articles_worked_example_week_by_week():
    profile = WeightProfile()
    assert profile.weight(FEATURE) == GLOBAL_WEIGHTS[FEATURE] == 7.0  # cold start

    profile.record(WEEK_1)
    assert profile.weight(FEATURE) == 5.0
    profile.record(WEEK_3)
    assert profile.weight(FEATURE) == 7.5
    profile.record(WEEK_8)
    assert profile.weight(FEATURE) == 9.0

    winner = profile.resolved(FEATURE)
    assert winner.source is Source.POST_DATE_FEEDBACK
    assert winner.tier is Tier.ALPHA
    assert winner.evidence == "The intellectual connection was everything"


@pytest.mark.parametrize("order", list(itertools.permutations([WEEK_1, WEEK_3, WEEK_8])))
def test_resolution_does_not_depend_on_arrival_order(order):
    profile = WeightProfile()
    for signal in order:
        profile.record(signal)
    assert profile.weight(FEATURE) == 9.0


def test_a_later_weaker_source_does_not_override_a_stronger_one():
    profile = WeightProfile()
    profile.record(WEEK_8)
    profile.record(
        WeightSignal(FEATURE, 2.0, Tier.GAMMA, Source.ONBOARDING, 1.0, 20, "not that important")
    )
    assert profile.weight(FEATURE) == 9.0


def test_within_a_source_higher_confidence_wins_over_recency():
    profile = WeightProfile()
    profile.record(WeightSignal(FEATURE, 6.0, Tier.BETA, Source.MATCH_FEEDBACK, 0.9, 2))
    profile.record(WeightSignal(FEATURE, 2.0, Tier.GAMMA, Source.MATCH_FEEDBACK, 0.4, 5))
    assert profile.weight(FEATURE) == 6.0


def test_within_equal_confidence_most_recent_wins():
    profile = WeightProfile()
    profile.record(WeightSignal(FEATURE, 6.0, Tier.BETA, Source.MATCH_FEEDBACK, 0.7, 2))
    profile.record(WeightSignal(FEATURE, 9.8, Tier.ALPHA, Source.MATCH_FEEDBACK, 0.7, 5))
    assert profile.weight(FEATURE) == 9.8


def test_every_weight_is_traceable_to_its_signal():
    profile = WeightProfile()
    for signal in (WEEK_1, WEEK_3, WEEK_8):
        profile.record(signal)

    # Nothing is discarded when a signal loses.
    assert profile.history(FEATURE) == (WEEK_1, WEEK_3, WEEK_8)

    audit = profile.explain(FEATURE)
    assert audit.splitlines()[0] == f"{FEATURE}: 9.0 (set by post_date_feedback)"
    winning_line = next(line for line in audit.splitlines() if "->" in line)
    assert "The intellectual connection was everything" in winning_line
    assert "I like smart people" in audit


def test_features_without_signals_fall_back_to_global_defaults():
    profile = WeightProfile()
    profile.record(WEEK_8)
    weights = profile.as_dict()
    assert weights[FEATURE] == 9.0
    assert weights["adventure_tolerance"] == GLOBAL_WEIGHTS["adventure_tolerance"] == 4.5
    assert "global default" in profile.explain("adventure_tolerance")


def test_invalid_signals_are_rejected():
    with pytest.raises(ValueError):
        WeightSignal("shoe_size", 5.0, Tier.BETA, Source.ONBOARDING)
    with pytest.raises(ValueError):
        WeightSignal(FEATURE, 11.0, Tier.ALPHA, Source.ONBOARDING)
    with pytest.raises(ValueError):
        WeightSignal(FEATURE, 5.0, Tier.BETA, Source.ONBOARDING, confidence=1.5)
