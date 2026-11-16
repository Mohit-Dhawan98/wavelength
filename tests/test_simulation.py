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
from wavelength.simulate import check_round, run_simulation, trial_seeds
from wavelength.strategies import FILTERED_STRATEGIES, STRATEGIES, Proposal
from wavelength.weights import Tier
from wavelength.world import generate_world

BIG = 6000  # people, for checks on shares; no pairwise work is done at this size
SMALL = dict(users=160, trials=2)


def _column(name):
    return FEATURE_NAMES.index(name)


def _population(name, n=BIG, seed=0, **changes):
    scenario = dataclasses.replace(PRESETS[name].scenario, pool_size=n, **changes)
    return scenario, generate_population(np.random.default_rng(seed), scenario)


# --- the population matches the scenario --------------------------------


@pytest.mark.parametrize("name", list(PRESETS))
def test_dealbreaker_values_match_the_configured_shares(name):
    scenario, people = _population(name, marriage_pressure=0.0)
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
    assert people.habit.mean() == pytest.approx(scenario.open_rate, abs=0.025)


def test_dealbreakers_are_correlated_through_archetypes():
    _, people = _population("today", marriage_pressure=0.0)
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


def _yes_rate_to_random_eligible(decisions, side):
    eligible = decisions.eligible[side]
    return (decisions.would_accept[side] & eligible).sum() / eligible.sum()


@pytest.mark.parametrize("name", list(PRESETS))
def test_realised_yes_rate_to_random_eligible_people_matches_the_scenario(name):
    scenario = dataclasses.replace(PRESETS[name].scenario, pool_size=900)
    people = generate_population(np.random.default_rng(0), scenario)
    decisions = decide(np.random.default_rng(1), scenario, people)
    majority = _yes_rate_to_random_eligible(decisions, people.majority)
    minority = _yes_rate_to_random_eligible(decisions, ~people.majority)
    # Personal rates are clipped to a range, which pulls wide spreads in slightly.
    assert majority == pytest.approx(scenario.yes_rate, rel=0.15)
    assert minority == pytest.approx(scenario.yes_rate * scenario.minority_yes_ratio, rel=0.15)


def test_each_person_meets_their_own_yes_rate():
    scenario = dataclasses.replace(Scenario(), pool_size=900)
    people = generate_population(np.random.default_rng(0), scenario)
    decisions = decide(np.random.default_rng(1), scenario, people)
    n_eligible = decisions.eligible.sum(axis=1)
    enough = n_eligible >= 30
    yeses = (decisions.would_accept & decisions.eligible).sum(axis=1)
    realised = yeses[enough] / n_eligible[enough]
    # Nobody can say yes to more people than survive their own dealbreakers.
    not_refused = (decisions.eligible & ~decisions.refuses).sum(axis=1)[enough] / n_eligible[enough]
    target = np.minimum(decisions.yes_rate[enough], not_refused)
    assert np.abs(realised - target).max() < 0.05
    assert decisions.yes_rate.std() > 0.05  # people differ


def test_the_bar_is_absolute_so_better_candidates_get_more_yeses():
    scenario = dataclasses.replace(Scenario(), pool_size=600)
    people = generate_population(np.random.default_rng(0), scenario)
    decisions = decide(np.random.default_rng(1), scenario, people)
    # A yes is exactly "felt attraction clears the person's bar and no dealbreaker is hit".
    expected = people.across & ~decisions.refuses & (decisions.felt >= decisions.bar[:, None])
    assert np.array_equal(decisions.would_accept, expected)
    # So among eligible candidates, the ones a person feels most for are accepted most.
    some = decisions.eligible.any(axis=1)
    felt = np.where(decisions.eligible, decisions.felt, np.nan)[some]
    above_median = felt >= np.nanmedian(felt, axis=1, keepdims=True)
    eligible, would_accept = decisions.eligible[some], decisions.would_accept[some]
    top = would_accept[eligible & above_median].mean()
    bottom = would_accept[eligible & ~above_median].mean()
    assert bottom < scenario.yes_rate < top


def test_mood_changes_single_decisions_and_shared_chemistry_links_the_pair():
    def both_and_product(**changes):
        scenario = dataclasses.replace(Scenario(), pool_size=700, yes_rate_spread=0.0, **changes)
        people = generate_population(np.random.default_rng(0), scenario)
        d = decide(np.random.default_rng(1), scenario, people)
        pairs = d.eligible & people.majority[:, None]
        yes, back = d.would_accept[pairs], d.would_accept.T[pairs]
        return (yes & back).mean(), yes.mean() * back.mean(), d

    both, product, calm = both_and_product(chemistry_share=0.9, chemistry_mutuality=1.0)
    assert both > 1.5 * product  # shared chemistry: the two decisions move together
    both, product, _ = both_and_product(chemistry_share=0.9, chemistry_mutuality=0.0)
    assert both < 1.5 * product
    _, _, moody = both_and_product(chemistry_share=0.9, chemistry_mutuality=1.0, decision_noise=0.4)
    assert (calm.would_accept != moody.would_accept).any()


def test_not_responding_is_separate_from_saying_no():
    world = generate_world(3, dataclasses.replace(PRESETS["serious"].scenario, pool_size=300))
    people, decisions = world.population, world.decisions
    assert np.array_equal(decisions.responds, decisions.open_luck[:, 0] < people.habit)
    silent = ~decisions.responds
    assert silent.any() and decisions.would_accept[silent].any()  # they had an opinion
    assert not decisions.accepts[silent].any()  # but nobody heard it
    assert np.array_equal(decisions.accepts[~silent], decisions.would_accept[~silent])


def test_decisions_respect_sides_and_dealbreakers():
    world = generate_world(3, dataclasses.replace(PRESETS["serious"].scenario, pool_size=300))
    people, would_accept = world.population, world.decisions.would_accept
    assert not would_accept[~people.across].any()
    assert not would_accept[world.decisions.refuses].any()
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


def test_observed_dealbreakers_are_as_stated_or_missing():
    scenario, people = _population("dystopia", n=2000)
    seen = observe(
        np.random.default_rng(1),
        dataclasses.replace(
            scenario, onboarding_reach=0.0, match_feedback_reach=0.0, post_date_reach=0.0
        ),
        people,
    )
    for feature in DEALBREAKER_FEATURES:
        observed, stated = seen.codes[:, _column(feature)], people.stated_codes[:, _column(feature)]
        known = observed != MISSING
        assert (observed[known] == stated[known]).all()
    # What people state is the truth for everyone who is open about intent.
    open_ = people.open_about_intent
    assert np.array_equal(people.stated_codes[open_], people.codes[open_])
    assert (people.stated_codes[~open_] != people.codes[~open_]).any()


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
        key: (s.users, s.pairs, s.responded, s.mutual, s.one_sided, s.violations)
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
def test_filtered_strategies_never_break_a_known_dealbreaker(name):
    scenario = dataclasses.replace(PRESETS[name].scenario, pool_size=200)
    for trial in range(2):
        world_seed, order_seed = trial_seeds(0, trial)
        world = generate_world(world_seed, scenario)
        known_ok = world.observations.known_ok
        order = np.random.default_rng(order_seed).permutation(scenario.pool_size)
        for key in FILTERED_STRATEGIES & set(STRATEGIES):
            for p in STRATEGIES[key][1](world.observations, order, 10):
                assert known_ok[p.a, p.b]


def test_with_everyone_open_about_intent_known_dealbreakers_are_the_true_ones():
    scenario = dataclasses.replace(
        Scenario(), pool_size=300, intent_openness=1.0, endogamy_share=0.0
    )
    world = generate_world(0, scenario)
    truly_eligible = eligibility_matrix(world.population.codes)
    assert not (world.observations.known_ok & ~truly_eligible).any()
    hidden = dataclasses.replace(scenario, intent_openness=0.5)
    world = generate_world(0, hidden)
    truly_eligible = eligibility_matrix(world.population.codes)
    assert (world.observations.known_ok & ~truly_eligible).any()


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
    calibration = report.calibrations["bidirectional"]
    assert calibration.pairs == report.strategies["bidirectional"].responded
    for b in calibration.bins:
        if b.pairs:
            assert b.low <= b.mean_predicted <= b.high and 0.0 <= b.observed <= 1.0


@pytest.mark.parametrize("name", list(PRESETS))
def test_answered_and_all_pairs_rates_are_reported_separately(name):
    report = run_simulation(PRESETS[name].scenario, name=name, users=300, trials=2)
    for s in report.strategies.values():
        assert 0 < s.responded < s.pairs  # some introductions expire in every preset
        assert s.mutual <= s.responded
        assert s.mutual_rate == pytest.approx(s.mutual / s.pairs)
        assert s.mutual_rate_responded == pytest.approx(s.mutual / s.responded)
        assert s.mutual_rate_responded >= s.mutual_rate
        assert s.expired_rate == pytest.approx(1 - s.responded / s.pairs)


def test_everyone_answering_makes_the_two_rates_equal():
    scenario = dataclasses.replace(Scenario(), open_rate=1.0, open_rate_spread=0.0)
    report = run_simulation(scenario, users=200, trials=1)
    for s in report.strategies.values():
        assert s.responded == s.pairs and s.mutual_rate == s.mutual_rate_responded


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_world_is_informative_at_default_size(name):
    """A floor on the world, not a verdict on any strategy.

    A preset in which almost nobody ever matches cannot tell strategies
    apart, however it is read. Each one must give at least one strategy a
    workable number of mutual matches in a default run, and leave a typical
    person more eligible candidates than the top-N cutoff.
    """
    report = run_simulation(PRESETS[name].scenario, name=name)
    assert max(s.mutual for s in report.strategies.values()) >= 15
    assert min(report.world.candidates_majority, report.world.candidates_minority) > report.top_n


def test_check_round_catches_a_double_assignment():
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
    assert "about this world" in single and "about each world" in both
    for text in (single, both):
        assert "both yes (all)" in text and "both yes (answered)" in text
        assert "fair test of the matcher" in text and "what the product delivers" in text
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
    assert main(["describe", "today", "--set", "yes_rate=0.08"]) == 0
    assert "yes_rate = 0.08" in capsys.readouterr().out


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
        "yes_rate=0.08",
        "--users",
        "120",
        "--trials",
        "1",
        "--compare",
        "casual,serious,swipe,dystopia",
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


def test_cli_calibrate_shows_both_curves_side_by_side(capsys):
    args = ["simulate", "--scenario", "today", "--users", "300", "--trials", "2", "--calibrate"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "P(accept) by rank" in out
    assert "article's curve" in out and "fitted curve" in out
    assert "P(mutual) from the article's curve" in out and "P(mutual) from the fitted curve" in out
    assert "if independent" in out
    assert "3 again, with the fitted P(rank)" in out
