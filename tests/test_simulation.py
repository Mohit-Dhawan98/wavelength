"""The simulated world and the run: shares, determinism, invariants and the CLI.

Nothing here asserts that one strategy does better than another. What a
strategy scores in a made-up world is an output of that world's assumptions,
not a property of the code to be pinned by a test.
"""

import dataclasses

import numpy as np
import pytest

from wavelength.cli import main
from wavelength.decisions import REFUSAL_DISTANCE, decide
from wavelength.features import FEATURE_NAMES, FEATURES, MISSING
from wavelength.filters import HARD_FILTERS, eligibility_matrix
from wavelength.observation import observe
from wavelength.population import TIERS, generate_population
from wavelength.presets import PRESETS
from wavelength.report import format_comparison, format_report
from wavelength.scenario import CORE_FEATURES, DEALBREAKER_FEATURES, Scenario
from wavelength.simulate import (
    FILTERED_STRATEGIES,
    STRATEGIES,
    check_round,
    generate_world,
    run_simulation,
    trial_seeds,
)
from wavelength.weights import Tier

BIG = 6000  # people, for checks on shares; no pairwise work is done at this size
SMALL = dict(users=160, trials=2)


def _column(name):
    return FEATURE_NAMES.index(name)


def _population(name, n=BIG, seed=0):
    scenario = dataclasses.replace(PRESETS[name].scenario, pool_size=n)
    return scenario, generate_population(np.random.default_rng(seed), scenario)


# --- the population matches the scenario --------------------------------


@pytest.mark.parametrize("name", list(PRESETS))
def test_dealbreaker_values_match_the_configured_shares(name):
    scenario, people = _population(name)
    for feature in DEALBREAKER_FEATURES:
        expected = scenario.value_shares(feature)
        for code, value in enumerate(FEATURES[feature].values):
            observed = np.mean(people.codes[:, _column(feature)] == code)
            assert observed == pytest.approx(expected[value], abs=0.025), (feature, value)


@pytest.mark.parametrize("name", list(PRESETS))
def test_archetype_and_side_shares(name):
    scenario, people = _population(name)
    for index, kind in enumerate(scenario.archetypes):
        assert np.mean(people.archetype == index) == pytest.approx(kind.share, abs=0.025)
    assert people.majority.mean() == pytest.approx(scenario.majority_share, abs=1 / BIG)
    assert people.active.mean() == pytest.approx(scenario.active_share, abs=0.025)


def test_dealbreakers_are_correlated_through_archetypes():
    _, people = _population("today")
    timeline = people.codes[:, _column("relationship_timeline")]
    family = people.codes[:, _column("family_plans")]
    soon = FEATURES["family_plans"].code("wants_children_soon")
    marriage = timeline == FEATURES["relationship_timeline"].code("marriage_focused")
    casual = timeline == FEATURES["relationship_timeline"].code("casual")
    assert np.mean(family[marriage] == soon) > np.mean(family[casual] == soon) + 0.2


@pytest.mark.parametrize("name", list(PRESETS))
def test_true_dealbreaker_shares_match_the_scenario(name):
    scenario, people = _population(name)
    alpha = TIERS.index(Tier.ALPHA)
    for feature in CORE_FEATURES:
        observed = np.mean(people.tiers[:, _column(feature)] == alpha)
        assert observed == pytest.approx(scenario.dealbreaker_shares[feature], abs=0.025)


def test_people_say_yes_at_the_configured_rate():
    scenario = dataclasses.replace(
        Scenario(), pool_size=800, pickiness_spread=0.0, active_share=1.0
    )
    people = generate_population(np.random.default_rng(0), scenario)
    decisions = decide(np.random.default_rng(1), scenario, people)
    rate = decisions.accepts.sum(axis=1) / people.across.sum(axis=1)
    assert rate[people.majority].mean() == pytest.approx(scenario.pickiness, abs=0.01)
    assert rate[~people.majority].mean() == pytest.approx(
        scenario.pickiness * scenario.minority_yes_ratio, abs=0.01
    )


def test_decisions_respect_sides_dealbreakers_and_inactivity():
    world = generate_world(3, dataclasses.replace(PRESETS["serious"].scenario, pool_size=300))
    people, accepts = world.population, world.decisions.accepts
    assert not accepts[~people.across].any()
    assert not accepts[world.decisions.refuses].any()
    assert not accepts[~people.active].any()
    # A refusal is a true dealbreaker held by the refuser and a clear gap on it.
    a, b = np.argwhere(world.decisions.refuses)[0]
    assert any(
        TIERS[people.tiers[a, j]] is Tier.ALPHA
        and FEATURES[name].distances[people.codes[a, j], people.codes[b, j]] >= REFUSAL_DISTANCE
        for j, name in enumerate(FEATURE_NAMES)
    )


def test_refusal_distance_is_looser_than_every_hard_filter():
    assert all(threshold < REFUSAL_DISTANCE for threshold in HARD_FILTERS.values())


# --- what the matcher sees ----------------------------------------------


@pytest.mark.parametrize("name", list(PRESETS))
def test_observed_coverage_matches_the_scenario(name):
    scenario, people = _population(name)
    seen = observe(
        np.random.default_rng(1),
        dataclasses.replace(
            scenario, onboarding_reach=0.0, match_feedback_reach=0.0, post_date_reach=0.0
        ),
        people,
    )
    hard = [_column(f) for f in DEALBREAKER_FEATURES]
    soft = [j for j in range(len(FEATURE_NAMES)) if j not in hard]
    assert np.mean(seen.codes[:, soft] != MISSING) == pytest.approx(
        scenario.feature_coverage, abs=0.02
    )
    all_known = (seen.codes[:, hard] != MISSING).all(axis=1)
    assert all_known.mean() == pytest.approx(scenario.dealbreakers_known_share, abs=0.02)
    complete = (seen.codes != MISSING).all(axis=1)
    assert complete.mean() >= scenario.full_profile_share - 0.02


def test_observed_dealbreakers_are_never_wrong_only_missing():
    scenario, people = _population("dystopia", n=2000)
    seen = observe(
        np.random.default_rng(1),
        dataclasses.replace(
            scenario, onboarding_reach=0.0, match_feedback_reach=0.0, post_date_reach=0.0
        ),
        people,
    )
    for feature in DEALBREAKER_FEATURES:
        observed, true = seen.codes[:, _column(feature)], people.codes[:, _column(feature)]
        known = observed != MISSING
        assert (observed[known] == true[known]).all()


def test_better_sources_override_stated_importance():
    scenario = dataclasses.replace(
        Scenario(),
        pool_size=300,
        self_knowledge=0.0,
        onboarding_reach=1.0,
        post_date_reach=1.0,
        post_date_accuracy=1.0,
    )
    people = generate_population(np.random.default_rng(0), scenario)
    seen = observe(np.random.default_rng(1), scenario, people)
    resolved = seen.weight_profiles[0].resolved("family_plans")
    assert resolved.source.name == "POST_DATE_FEEDBACK"
    assert resolved.tier is TIERS[people.tiers[0, _column("family_plans")]]


# --- determinism --------------------------------------------------------


def _counts(report):
    return {
        key: (s.users, s.pairs, s.mutual, s.one_sided, s.violations)
        for key, s in report.strategies.items()
    }


def test_same_seed_same_world_and_same_report():
    scenario = dataclasses.replace(Scenario(), pool_size=150)
    first, second = generate_world(7, scenario), generate_world(7, scenario)
    assert np.array_equal(first.population.codes, second.population.codes)
    assert np.array_equal(first.decisions.accepts, second.decisions.accepts)
    assert np.array_equal(first.observations.codes, second.observations.codes)
    assert np.array_equal(first.observations.weights, second.observations.weights)
    a = run_simulation(scenario, seed=5, trials=2)
    b = run_simulation(scenario, seed=5, trials=2)
    assert _counts(a) == _counts(b)
    assert format_report(a) == format_report(b)


def test_different_seeds_give_different_worlds():
    scenario = dataclasses.replace(Scenario(), pool_size=150)
    assert not np.array_equal(
        generate_world(1, scenario).population.codes, generate_world(2, scenario).population.codes
    )


# --- invariants, in every preset world ----------------------------------


@pytest.mark.parametrize("name", list(PRESETS))
def test_no_strategy_assigns_anyone_twice_or_within_a_side(name):
    scenario = dataclasses.replace(PRESETS[name].scenario, pool_size=200)
    for trial in range(2):
        world_seed, order_seed = trial_seeds(0, trial)
        world = generate_world(world_seed, scenario)
        order = np.random.default_rng(order_seed).permutation(scenario.pool_size)
        for _, strategy in STRATEGIES.values():
            proposals = strategy(world.observations, order, 10)
            check_round(proposals)
            users = [u for p in proposals for u in (p.a, p.b)]
            assert len(users) == len(set(users))
            assert all(world.population.across[p.a, p.b] for p in proposals)


@pytest.mark.parametrize("name", list(PRESETS))
def test_filtered_strategies_never_break_a_hard_filter_on_true_values(name):
    scenario = dataclasses.replace(PRESETS[name].scenario, pool_size=200)
    for trial in range(2):
        world_seed, order_seed = trial_seeds(0, trial)
        world = generate_world(world_seed, scenario)
        truly_eligible = eligibility_matrix(world.population.codes)
        order = np.random.default_rng(order_seed).permutation(scenario.pool_size)
        for key in FILTERED_STRATEGIES:
            for p in STRATEGIES[key][1](world.observations, order, 10):
                assert truly_eligible[p.a, p.b]


@pytest.mark.parametrize("name", list(PRESETS))
def test_report_counts_are_consistent(name):
    report = run_simulation(PRESETS[name].scenario, name=name, **SMALL)
    assert report.scenario.pool_size == 160
    assert set(report.strategies) == set(STRATEGIES)
    for key, s in report.strategies.items():
        assert s.users == 2 * 160
        assert 0 <= s.mutual + s.one_sided <= s.pairs <= s.users // 2
        assert 0.0 <= s.unproposed_share <= 1.0
        assert s.matches_per_100_users == pytest.approx(100 * s.mutual / s.users)
        if key in FILTERED_STRATEGIES:
            assert s.violations == 0
    predicted = sum(b.pairs for b in report.calibration)
    assert predicted == report.strategies["bidirectional"].pairs
    for b in report.calibration:
        if b.pairs:
            assert b.low <= b.mean_predicted <= b.high and 0.0 <= b.observed <= 1.0


def test_check_round_catches_a_double_assignment():
    from wavelength.simulate import Proposal

    with pytest.raises(AssertionError):
        check_round([Proposal(0, 1), Proposal(1, 2)])


def test_run_rejects_zero_trials():
    with pytest.raises(ValueError):
        run_simulation(Scenario(), trials=0)


# --- reports and the command line ---------------------------------------


def test_reports_say_what_they_are():
    reports = [run_simulation(PRESETS[n].scenario, name=n, **SMALL) for n in ("today", "casual")]
    single, both = format_report(reports[0]), format_comparison(reports)
    for text in (single, both):
        assert "Not the article's production results" in text
        for label, _ in STRATEGIES.values():
            assert label in text
    assert "Calibration of predicted P(mutual)" in single
    assert "today" in both and "casual" in both


def test_cli_scenarios_lists_every_preset(capsys):
    assert main(["scenarios"]) == 0
    out = capsys.readouterr().out
    for p in PRESETS.values():
        assert p.name in out and p.summary in out


@pytest.mark.parametrize("name", list(PRESETS))
def test_cli_describe_prints_every_field(name, capsys):
    assert main(["describe", name]) == 0
    out = capsys.readouterr().out
    for field in dataclasses.fields(Scenario):
        assert field.name in out
    assert "meaning:" in out and "sensible range:" in out and "basis of the default:" in out
    for reason in PRESETS[name].reasons.values():
        assert reason.split()[0] in out


def test_cli_describe_reflects_set(capsys):
    assert main(["describe", "today", "--set", "pickiness=0.08"]) == 0
    assert "pickiness = 0.08" in capsys.readouterr().out


def test_cli_simulate_runs_and_is_deterministic(capsys):
    args = ["simulate", "--scenario", "today", "--seed", "3", "--users", "120", "--trials", "1"]
    assert main(args) == 0
    first = capsys.readouterr().out
    assert main(args) == 0
    assert capsys.readouterr().out == first
    assert 'scenario "today"' in first and "users/trial=120" in first


def test_cli_simulate_with_set_and_compare(capsys):
    args = [
        "simulate",
        "--scenario",
        "today",
        "--set",
        "pickiness=0.08",
        "--users",
        "120",
        "--trials",
        "1",
        "--compare",
        "casual,serious,dystopia",
    ]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "side by side" in out
    for name in PRESETS:
        assert f"{name} (modified)" in out


def test_cli_simulate_accepts_a_scenario_file(tmp_path, capsys):
    path = tmp_path / "tiny.toml"
    path.write_text('base = "casual"\npool_size = 100\n')
    assert main(["simulate", "--scenario", str(path), "--trials", "1"]) == 0
    out = capsys.readouterr().out
    assert 'scenario "tiny"' in out and "users/trial=100" in out


def test_cli_reports_bad_input_without_a_traceback(capsys):
    assert main(["simulate", "--set", "nope=1"]) == 2
    assert "not a valid key" in capsys.readouterr().err
    assert main(["describe", "utopia"]) == 2
    assert "neither a preset" in capsys.readouterr().err
