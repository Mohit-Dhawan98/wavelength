"""Rank curves: the curve argument in matching, the fit, and the held-out check."""

import dataclasses

import numpy as np
import pytest

from wavelength.calibration import RankOutcomes, fit_rank_curve
from wavelength.diagnostics import check_world, mean_rank_correlation
from wavelength.matching import (
    TOP_N,
    MutualPair,
    acceptance_probability,
    article_curve,
    assign_by_probability,
    bidirectional_match,
    mutual_probability,
    mutual_top_n,
    rank_matrix,
)
from wavelength.presets import PRESETS
from wavelength.scenario import Scenario
from wavelength.simulate import FITTED_KEY, observe_rank_outcomes, run_simulation, trial_seeds
from wavelength.strategies import bidirectional
from wavelength.world import generate_world

FLAT = (0.5,) * TOP_N
STEEP = tuple(1.0 if rank == 1 else 0.1 for rank in range(1, TOP_N + 1))


# --- the curve argument in matching ------------------------------------


def test_article_curve_is_the_articles_formula():
    assert article_curve() == tuple(acceptance_probability(r) for r in range(1, TOP_N + 1))
    assert article_curve()[0] == 1.0 and article_curve()[-1] == 0.5


def test_no_curve_and_the_article_curve_agree_everywhere():
    for a in range(0, TOP_N + 2):
        for b in range(0, TOP_N + 2):
            assert mutual_probability(a, b) == pytest.approx(
                mutual_probability(a, b, curve=article_curve())
            )


def test_a_curve_replaces_the_formula_and_is_zero_outside_top_n():
    assert mutual_probability(1, 3, curve=FLAT) == 0.25
    assert mutual_probability(1, 2, curve=STEEP) == pytest.approx(0.1)
    assert mutual_probability(1, TOP_N + 1, curve=FLAT) == 0.0


@pytest.mark.parametrize("bad", [(0.5,) * 3, (1.5,) + (0.5,) * (TOP_N - 1)])
def test_a_malformed_curve_is_rejected(bad):
    with pytest.raises(ValueError):
        mutual_probability(1, 1, curve=bad)
    with pytest.raises(ValueError):
        mutual_top_n(np.zeros((2, 2), dtype=int), curve=bad)


def _ranks():
    rng = np.random.default_rng(0)
    scores = rng.random((30, 30))
    return scores, ~np.eye(30, dtype=bool)


def test_mutual_top_n_uses_the_curve_for_probabilities():
    scores, eligible = _ranks()
    ranks = rank_matrix(scores, eligible)
    default, flat = mutual_top_n(ranks), mutual_top_n(ranks, curve=FLAT)
    assert [(p.a, p.b) for p in default] == [(p.a, p.b) for p in flat]
    assert all(p.probability == 0.25 for p in flat)
    assert any(p.probability != 0.25 for p in default)


def test_assign_by_probability_reorders_by_the_curve():
    # Under the article's curve (2, 2) beats (1, 5); under a curve where only
    # first place counts for much, (1, 5) ... still loses to a mutual first.
    pairs = [
        MutualPair(0, 1, 2, 2, mutual_probability(2, 2)),
        MutualPair(0, 2, 1, 5, mutual_probability(1, 5)),
    ]
    assert assign_by_probability(pairs)[0].b == 1
    first_matters = (1.0, 0.2, 0.2, 0.2, 0.9) + (0.1,) * (TOP_N - 5)
    chosen = assign_by_probability(pairs, curve=first_matters)
    assert chosen[0].b == 2 and chosen[0].probability == pytest.approx(0.9)


def test_bidirectional_match_passes_the_curve_through():
    scores, eligible = _ranks()
    flat = bidirectional_match(scores, eligible, curve=FLAT)
    assert flat and all(p.probability == 0.25 for p in flat)
    assert bidirectional_match(scores, eligible) == bidirectional_match(
        scores, eligible, curve=article_curve()
    )


def test_the_simulated_strategy_reports_the_curves_probabilities():
    world = generate_world(1, dataclasses.replace(Scenario(), pool_size=200))
    proposals = bidirectional(world.observations, np.arange(200), TOP_N, FLAT)
    assert proposals and all(p.predicted == 0.25 for p in proposals)
    assert all(
        1 <= p.rank_of_b_for_a <= TOP_N and 1 <= p.rank_of_a_for_b <= TOP_N for p in proposals
    )


# --- fitting ------------------------------------------------------------


def _outcomes(seen, yes):
    outcomes = RankOutcomes(len(seen))
    outcomes.seen[:] = seen
    outcomes.yes[:] = yes
    return outcomes


def test_fit_recovers_a_known_curve_from_enough_decisions():
    truth = np.linspace(0.8, 0.2, TOP_N)
    rng = np.random.default_rng(0)
    outcomes = RankOutcomes()
    for rank in range(1, TOP_N + 1):
        for accepted in rng.random(4000) < truth[rank - 1]:
            outcomes.add(rank, bool(accepted))
    assert np.abs(np.array(fit_rank_curve(outcomes)) - truth).max() < 0.03


def test_fit_never_rises_with_rank_and_stays_a_probability():
    fitted = fit_rank_curve(_outcomes([50, 50, 50, 50], [10, 40, 5, 30]))
    assert all(a >= b for a, b in zip(fitted, fitted[1:]))
    assert all(0.0 < p < 1.0 for p in fitted)


def test_fit_fills_unseen_ranks_from_the_nearest_seen_one():
    fitted = fit_rank_curve(_outcomes([100, 0, 0, 100], [80, 0, 0, 20]))
    assert fitted[1] == fitted[0] and fitted[2] == fitted[3]


def test_fit_with_no_decisions_returns_the_article_curve():
    assert fit_rank_curve(RankOutcomes()) == article_curve()


def test_rank_outcomes_reject_ranks_outside_the_list():
    with pytest.raises(ValueError):
        RankOutcomes().add(TOP_N + 1, True)


# --- fit on some pools, check on others ---------------------------------


def test_fitting_pools_are_not_the_pools_the_curve_is_checked_on():
    assert trial_seeds(0, 0) != trial_seeds(0, 0, fitting=True)
    assert trial_seeds(0, 0, fitting=True) != trial_seeds(0, 1, fitting=True)


def test_only_decisions_of_people_who_answered_are_fitted_on():
    scenario = dataclasses.replace(Scenario(), pool_size=300, active_share=0.5)
    outcomes = observe_rank_outcomes(scenario, seed=0, trials=1)
    world = generate_world(trial_seeds(0, 0, fitting=True)[0], scenario)
    proposals = bidirectional(world.observations, np.arange(300), TOP_N)
    answered = sum(
        int(world.decisions.responds[p.a]) + int(world.decisions.responds[p.b]) for p in proposals
    )
    assert outcomes.seen.sum() == answered < 2 * len(proposals)


@pytest.mark.parametrize("name", ["today", "casual", "serious"])
def test_a_fitted_curve_is_close_to_outcomes_on_held_out_pools(name):
    """A property of fitting, not a comparison of strategies.

    A curve fitted to decisions in one set of pools should predict the
    both-yes rate in other pools from the same scenario to within a loose
    bound. The hand-set curve is not expected to, in a world it was not set
    for; that is the point of fitting.
    """
    report = run_simulation(PRESETS[name].scenario, name=name, users=700, trials=3, calibrate=True)
    fitted = report.calibrations[FITTED_KEY]
    assert fitted.pairs == report.strategies[FITTED_KEY].responded > 100
    assert fitted.gap < 0.12
    assert report.fit_outcomes.seen.sum() > 0
    assert fitted.curve != report.calibrations["bidirectional"].curve


def test_an_uncalibrated_run_has_only_the_article_curve_and_the_same_main_rows():
    scenario = dataclasses.replace(Scenario(), pool_size=300)
    plain = run_simulation(scenario, trials=2)
    fitted = run_simulation(scenario, trials=2, calibrate=True)
    assert set(plain.calibrations) == {"bidirectional"} and not plain.calibrated
    assert plain.calibrations["bidirectional"].curve == article_curve()
    for key, stats in plain.strategies.items():
        assert stats.mutual == fitted.strategies[key].mutual
        assert stats.pairs == fitted.strategies[key].pairs


def test_independence_check_compares_both_yes_with_the_product_of_the_two_sides():
    report = run_simulation(dataclasses.replace(Scenario(), pool_size=300), trials=2)
    for b in report.calibrations["bidirectional"].bins:
        if b.pairs:
            expected = (b.majority_yes / b.pairs) * (b.minority_yes / b.pairs)
            assert b.if_independent == pytest.approx(expected)
            assert b.mutual <= min(b.majority_yes, b.minority_yes)


# --- facts about the world ----------------------------------------------


def test_rank_correlation_is_one_for_identical_rankings_and_minus_one_for_reversed():
    rng = np.random.default_rng(0)
    x = rng.random((6, 12))
    everyone = np.ones((6, 12), dtype=bool)
    assert mean_rank_correlation(x, 2 * x + 1, everyone) == pytest.approx(1.0)
    assert mean_rank_correlation(x, -x, everyone) == pytest.approx(-1.0)
    assert np.isnan(mean_rank_correlation(x, x, np.zeros((6, 12), dtype=bool)))


def test_world_check_reports_what_the_scenario_implies():
    scenario = dataclasses.replace(Scenario(), pool_size=600)
    check = check_world(generate_world(0, scenario))
    assert check.responding_share == pytest.approx(scenario.active_share, abs=0.06)
    assert 0.15 < check.yes_rate_realised < scenario.yes_rate
    assert check.candidates_majority > 0 and check.candidates_minority > 0
    # A perfect read of profiles tracks feeling better than any noisy score can,
    # and a score made from the truth-bearing signals is better than nothing.
    assert check.profile_tracks_feeling > check.score_tracks_feeling > 0.0


def test_better_signals_make_the_score_track_feeling_more_closely():
    def tracking(**changes):
        scenario = dataclasses.replace(Scenario(), pool_size=500, **changes)
        return check_world(generate_world(0, scenario)).score_tracks_feeling

    blind = tracking(text_signal_quality=0.0, photo_signal_quality=0.0, feature_coverage=0.34)
    sharp = tracking(
        text_signal_quality=1.0,
        photo_signal_quality=1.0,
        feature_coverage=1.0,
        full_profile_share=1.0,
        dealbreakers_known_share=1.0,
        extraction_accuracy=1.0,
    )
    assert sharp > blind
    assert tracking(chemistry_share=0.9) < tracking(chemistry_share=0.1)
