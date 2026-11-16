"""The game: best-possible matching, the engine, the View, rules, tournament and web surface."""

import ast
import dataclasses
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from wavelength import web
from wavelength.best import maximum_matching, most_couples
from wavelength.cli import main
from wavelength.decisions import Decisions
from wavelength.engine import Arena
from wavelength.key import LAST_STAGE, AnswerKey
from wavelength.matching import TOP_N, bidirectional_match
from wavelength.play import Rule, RuleError, load_rules, rule, run
from wavelength.population import Population
from wavelength.presets import PRESETS, load_preset_file, load_scenario_file
from wavelength.scenario import Scenario
from wavelength.scoring import FEATURE_WEIGHT, TEXT_WEIGHT, VISUAL_WEIGHT
from wavelength.tournament import format_leaderboard, rules_from, run_tournament
from wavelength.view import deferred_acceptance
from wavelength.world import World

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "rules"
SMALL = dataclasses.replace(Scenario(), pool_size=200)
DEMO = {r.name: r for file in sorted(RULES.glob("*.py")) for r in load_rules(file)}
ARTICLE = DEMO["the article's rule"]


@pytest.fixture(scope="module")
def arena():
    return Arena(SMALL, seed=0)


def capture(arena, rounds=1):
    """Play a do-nothing rule and keep the views it was shown."""
    seen = []

    def watcher(v):
        seen.append(v)
        return []

    arena.play(watcher, rounds)
    return seen


# --- the most couples possible ------------------------------------------


def _brute_force(adjacency, n_right):
    edges = [(u, v) for u, row in enumerate(adjacency) for v in row]
    for size in range(min(len(adjacency), n_right), 0, -1):
        for chosen in itertools.combinations(edges, size):
            if len({u for u, _ in chosen}) == size and len({v for _, v in chosen}) == size:
                return size
    return 0


def test_maximum_matching_agrees_with_brute_force_on_small_graphs():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n_left, n_right = rng.integers(1, 7, size=2)
        dense = rng.random((n_left, n_right)) < rng.uniform(0.1, 0.7)
        adjacency = [np.flatnonzero(row).tolist() for row in dense]
        matching = maximum_matching(adjacency, int(n_right))
        assert len(matching) == _brute_force(adjacency, int(n_right))
        assert all(dense[u, v] for u, v in matching)
        assert len({u for u, _ in matching}) == len(matching) == len({v for _, v in matching})


def test_maximum_matching_finds_the_augmenting_path_greedy_misses():
    # Left 0 can take either right vertex; left 1 only the first. Greedy on 0 blocks 1.
    assert len(maximum_matching([[0, 1], [0]], 2)) == 2
    assert maximum_matching([[], []], 3) == []
    assert len(maximum_matching([[0], [0], [0]], 1)) == 1


def test_most_couples_only_counts_pairs_across_the_two_sides():
    majority = np.array([True, True, False, False])
    pairs = np.zeros((4, 4), dtype=bool)
    pairs[0, 1] = pairs[1, 0] = True  # same side: not a couple
    assert most_couples(pairs, majority) == 0
    pairs[0, 2] = pairs[2, 0] = pairs[1, 2] = pairs[2, 1] = True
    assert most_couples(pairs, majority) == 1
    pairs[1, 3] = pairs[3, 1] = True
    assert most_couples(pairs, majority) == 2


def test_best_possible_shrinks_with_each_stage_and_bounds_every_rule(arena):
    best = [arena.best_possible(k) for k in range(1, LAST_STAGE + 1)]
    assert best == sorted(best, reverse=True) and best[-1] > 0
    for chosen in (ARTICLE, DEMO["random introductions"]):
        play = arena.play(chosen, 6)
        for k in range(1, LAST_STAGE + 1):
            assert arena.couples(play, k) <= arena.best_possible(k)
    informed = arena.informed_play(6)
    assert arena.couples(informed, LAST_STAGE) <= arena.best_possible(LAST_STAGE)
    assert arena.couples(informed, LAST_STAGE) > arena.couples(
        arena.play(DEMO["random introductions"], 6), LAST_STAGE
    )


# --- the same luck for every rule ---------------------------------------


def test_a_run_is_repeatable(arena):
    first, second = arena.play(ARTICLE, 5), arena.play(ARTICLE, 5)
    assert first.introductions and [dataclasses.astuple(i) for i in first.introductions] == [
        dataclasses.astuple(i) for i in second.introductions
    ]
    again = Arena(SMALL, seed=0).play(ARTICLE, 5)
    assert [dataclasses.astuple(i) for i in again.introductions] == [
        dataclasses.astuple(i) for i in first.introductions
    ]
    other = Arena(SMALL, seed=1).play(ARTICLE, 5)
    assert [dataclasses.astuple(i) for i in other.introductions] != [
        dataclasses.astuple(i) for i in first.introductions
    ]


def test_two_rules_introducing_the_same_pair_first_get_the_same_outcome(arena):
    """Common random numbers: luck belongs to the people and the week, not to the rule."""
    shown = capture(arena)[0]
    a, b = (int(x) for x in np.argwhere(np.triu(shown.ok))[0])

    def just_them(v):
        return [(a, b)] if v.round == 0 else []

    def them_and_others(v):
        if v.round > 0:
            return []
        rest = v.ok.copy()
        rest[[a, b], :] = rest[:, [a, b]] = False
        return [(a, b), *v.pair_best(v.text, among=rest)]

    alone = arena.play(just_them, 1).introductions[0]
    in_company = arena.play(them_and_others, 1).introductions[0]
    assert dataclasses.astuple(alone) == dataclasses.astuple(in_company)
    assert len(arena.play(them_and_others, 1).introductions) > 1


def test_outcomes_come_from_the_answer_key_and_the_predrawn_weeks(arena):
    world = arena._world
    play = arena.play(ARTICLE, 1)
    opens = world.decisions.open_luck[:, 0] < world.population.habit
    for i in play.introductions:
        assert (i.a_answered, i.b_answered) == (bool(opens[i.a]), bool(opens[i.b]))
        both_yes = (
            i.a_answered
            and i.b_answered
            and world.decisions.would_accept[i.a, i.b]
            and world.decisions.would_accept[i.b, i.a]
        )
        assert i.stage == (int(world.key.stage[i.a, i.b]) if both_yes else 0)


# --- attention has causes -----------------------------------------------


def _answered_late(**changes):
    arena = Arena(dataclasses.replace(SMALL, **changes), seed=0)
    play = arena.play(DEMO["random introductions"], 8)
    late = [i for i in play.introductions if i.round >= 5]
    return sum(i.a_answered + i.b_answered for i in late) / (2 * len(late)), play


def test_letdowns_wear_attention_down_and_people_give_up():
    fresh, calm = _answered_late(fatigue=0.0, patience=50.0)
    tired, weary = _answered_late(fatigue=0.4, patience=50.0)
    assert tired < fresh - 0.1
    assert not calm.gave_up_in and not weary.gave_up_in
    _, quitters = _answered_late(fatigue=0.0, patience=2.0)
    assert len(quitters.gave_up_in) > 20
    # Someone who has given up never answers again.
    for i in quitters.introductions:
        for person, answered in ((i.a, i.a_answered), (i.b, i.b_answered)):
            if person in quitters.gave_up_in and i.round > quitters.gave_up_in[person]:
                assert not answered


def test_couples_leave_the_pool_at_the_chosen_stage():
    def leavers(stage):
        arena = Arena(dataclasses.replace(SMALL, leave_after_stage=stage), seed=0)
        play = arena.play(ARTICLE, 6)
        for person, week in play.left_in.items():
            assert not any(person in (i.a, i.b) and i.round > week for i in play.introductions)
        reached = {p for i in play.introductions if i.stage >= stage for p in (i.a, i.b)}
        assert set(play.left_in) == reached
        return len(play.left_in)

    assert leavers(1) > leavers(4) > 0


def test_feedback_between_rounds_updates_the_learned_weights():
    arena = Arena(dataclasses.replace(SMALL, match_feedback_reach=0.9), seed=0)
    personal = []

    def watcher(v):
        personal.append(v.features("personal").copy())
        return v.pair_best(v.text)

    arena.play(watcher, 3)
    assert not np.allclose(personal[0], personal[2])
    quiet = Arena(dataclasses.replace(SMALL, match_feedback_reach=0.0, post_date_reach=0.0), seed=0)
    personal.clear()
    quiet.play(watcher, 3)
    assert np.allclose(personal[0], personal[2])


# --- what a rule can see ------------------------------------------------


def _reachable(start):
    """Every object reachable from ``start`` through attributes, containers and array bases."""
    seen, stack, found = set(), [start], []
    while stack:
        thing = stack.pop()
        if id(thing) in seen or isinstance(thing, (str, bytes, int, float, bool, type(None))):
            continue
        seen.add(id(thing))
        found.append(thing)
        if isinstance(thing, np.ndarray):
            if thing.base is not None:
                stack.append(thing.base)
            continue
        if isinstance(thing, dict):
            stack.extend(thing.keys())
            stack.extend(thing.values())
        elif isinstance(thing, (list, tuple, set, frozenset)):
            stack.extend(thing)
        for klass in type(thing).__mro__:
            for name in getattr(klass, "__slots__", ()):
                if hasattr(thing, name):
                    stack.append(getattr(thing, name))
        if hasattr(thing, "__dict__"):
            stack.extend(vars(thing).values())
    return found


def _arrays_in(*holders):
    return [x for holder in holders for x in vars(holder).values() if isinstance(x, np.ndarray)]


def test_nothing_reachable_from_a_view_is_or_shares_memory_with_the_hidden_truth(arena):
    world = arena._world
    secret = _arrays_in(world.population, world.decisions, world.key)
    assert len(secret) > 30
    for view in capture(arena, rounds=2):
        reachable = _reachable(view)
        assert len(reachable) > 20
        for thing in reachable:
            assert not isinstance(thing, (World, Population, Decisions, AnswerKey, Arena))
            if isinstance(thing, np.ndarray):
                assert not any(np.shares_memory(thing, hidden) for hidden in secret)
        # Not even the observations held by the world are handed over: the board has its own copy.
        observed = _arrays_in(world.observations)
        arrays = [t for t in reachable if isinstance(t, np.ndarray)]
        assert not any(np.shares_memory(a, o) for a in arrays for o in observed)


def test_a_view_has_no_attribute_for_any_hidden_quantity(arena):
    view = capture(arena)[0]
    public = {name for name in dir(view) if not name.startswith("_")}
    hidden = set(vars(arena._world.population)) | set(vars(arena._world.decisions))
    hidden |= set(vars(arena._world.key)) | {"key", "truth", "world", "population", "decisions"}
    # Age is on every profile, and "family" on a view is the app's own noisy read of it.
    assert not (public & (hidden - {"age", "family"}))
    assert not hasattr(view, "__dict__")  # slots only: a rule cannot be handed extras by accident


def test_view_arrays_cannot_be_changed_by_a_rule(arena):
    def vandal(v):
        for name in ("text", "photo", "voice", "ok", "open", "available", "distance"):
            with pytest.raises(ValueError):
                getattr(v, name)[0, ...] = 1
        with pytest.raises(ValueError):
            v.history.introduced[0, 0] = 5
        return []

    arena.play(vandal, 2)


def test_a_view_shows_what_a_profile_shows(arena):
    view = capture(arena)[0]
    n = view.n
    assert n == 200 and view.round == 0 and view.rounds == 1
    for name in ("text", "photo", "voice", "family", "distance", "ok", "open", "same_community"):
        assert getattr(view, name).shape == (n, n)
    for signal in (view.text, view.photo, view.voice, view.features(), view.features("global")):
        assert signal.min() >= 0.0 and signal.max() <= 1.0 + 1e-9
    assert np.array_equal(view.open, view.larger_side[:, None] != view.larger_side[None, :])
    assert not (view.ok & ~view.open).any() and view.ok.sum() < view.open.sum()
    assert np.array_equal(view.ok, arena._world.observations.known_ok)
    custom = view.features({"travel_frequency": 1.0})
    assert custom.shape == (n, n) and not np.allclose(custom, view.features("global"))
    with pytest.raises(ValueError, match="unknown feature"):
        view.features({"height": 1.0})
    with pytest.raises(ValueError, match="personal"):
        view.features("mine")


def test_history_records_what_the_app_would_have_logged():
    arena = Arena(SMALL, seed=0)
    views = []

    def watcher(v):
        views.append(v)
        return v.pair_best(v.text)

    play = arena.play(watcher, 3)
    first, last = views[0].history, views[-1].history
    assert (first.introduced == -1).all() and not first.introductions.any()
    before_last = [i for i in play.introductions if i.round < 2]
    for i in before_last:
        assert last.introduced[i.a, i.b] == last.introduced[i.b, i.a] == i.round
        assert last.answered[i.a, i.b] == i.a_answered and last.said_yes[i.a, i.b] == i.a_yes
        assert not views[-1].open[i.a, i.b]  # nobody is offered the same person twice
    assert last.introductions.sum() == 2 * len(before_last)
    assert last.unanswered.sum() == sum(
        (not i.a_answered) + (not i.b_answered) for i in before_last
    )
    assert (last.yeses + last.declines + last.unanswered == last.introductions).all()
    assert (last.recent_declines <= last.declines).all()


# --- helpers ------------------------------------------------------------


def test_pair_helpers_return_valid_pairs_within_what_is_allowed(arena):
    view = capture(arena)[0]
    score = TEXT_WEIGHT * view.text + VISUAL_WEIGHT * view.photo + FEATURE_WEIGHT * view.features()
    for pairs in (view.pair_best(score), view.pair_mutual(score), view.pair_stable(score)):
        people = [p for pair in pairs for p in pair]
        assert pairs and len(people) == len(set(people))
        assert all(view.ok[a, b] for a, b in pairs)
    everyone = view.pair_best(score, among=view.open)
    assert len(everyone) > len(view.pair_best(score))
    assert not all(view.ok[a, b] for a, b in everyone)
    ranks = view.rank(score)
    assert ranks.shape == (view.n, view.n) and (ranks[~view.ok] == 0).all()
    assert set(np.unique(ranks[0][view.ok[0]])) == set(range(1, int(view.ok[0].sum()) + 1))


def test_pair_mutual_is_the_articles_method(arena):
    view = capture(arena)[0]
    score = TEXT_WEIGHT * view.text + VISUAL_WEIGHT * view.photo + FEATURE_WEIGHT * view.features()
    expected = [(p.a, p.b) for p in bidirectional_match(score, view.ok, TOP_N)]
    assert view.pair_mutual(score, top=10) == expected == ARTICLE(view)
    flat = view.pair_mutual(score, top=10, curve=(0.5,) * 10)
    assert {frozenset(p) for p in flat} != {frozenset(p) for p in expected} or flat != expected


def test_deferred_acceptance_is_stable_on_the_scores_given():
    rng = np.random.default_rng(0)
    n = 30
    proposers = np.arange(n) < 18
    scores = rng.random((n, n))
    allowed = (proposers[:, None] != proposers[None, :]) & (rng.random((n, n)) < 0.6)
    allowed &= allowed.T
    pairs = deferred_acceptance(scores, allowed, proposers)
    partner = {a: b for a, b in pairs} | {b: a for a, b in pairs}
    assert len(partner) == 2 * len(pairs) and all(allowed[a, b] for a, b in pairs)
    for a, b in np.argwhere(allowed):
        if not proposers[a]:
            continue
        a_prefers = a not in partner or scores[a, b] > scores[a, partner[a]]
        b_prefers = b not in partner or scores[b, a] > scores[b, partner[b]]
        assert not (a_prefers and b_prefers), "a blocking pair"


# --- rules and validation -----------------------------------------------


def test_the_rule_decorator_names_a_function():
    @rule("  my rule ")
    def mine(v):
        """What it does."""
        return []

    assert isinstance(mine, Rule) and mine.name == "my rule" and mine.description == "What it does."
    with pytest.raises(ValueError):
        rule("")
    with pytest.raises(TypeError):
        rule("x")(3)


def _same_side(view):
    return tuple(int(p) for p in np.flatnonzero(view.larger_side)[:2])


@pytest.mark.parametrize(
    "bad, message",
    [
        (lambda v: None, "returned nothing"),
        (lambda v: 5, "returned int"),
        (lambda v: "0,1", "returned str"),
        (lambda v: v.text, "returned ndarray"),
        (lambda v: [(0, 1, 2)], "not a pair"),
        (lambda v: [(0.5, 1)], "not a pair"),
        (lambda v: [(0, 10**6)], "does not exist"),
        (lambda v: [_same_side(v)], "same side"),
        (lambda v: v.pair_best(v.text)[:1] * 2, "is in two pairs"),
    ],
)
def test_bad_output_is_refused_with_a_message_that_names_the_problem(arena, bad, message):
    with pytest.raises(RuleError, match=message) as raised:
        arena.play(bad, 1)
    assert "round 0" in str(raised.value)


def test_repeats_and_departed_people_are_refused(arena):
    first = capture(arena)[0].pair_best(capture(arena)[0].text)[:1]
    with pytest.raises(RuleError, match="already introduced in round 0"):
        arena.play(lambda v: first, 2)
    leaving = Arena(dataclasses.replace(SMALL, leave_after_stage=1), seed=0)
    world = leaving._world
    yes = world.decisions.would_accept
    opens = world.decisions.open_luck[:, 0] < world.population.habit
    a, b = next((int(a), int(b)) for a, b in np.argwhere(yes & yes.T) if opens[a] and opens[b])
    others = [
        int(p) for p in np.flatnonzero(world.population.majority != world.population.majority[a])
    ]
    other = next(p for p in others if p != b)

    def clingy(v):
        return [(a, b)] if v.round == 0 else [(a, other)]

    with pytest.raises(RuleError, match=f"person {a} has left the pool"):
        leaving.play(clingy, 2)


def test_rounds_must_be_sensible(arena):
    for rounds in (0, 53):
        with pytest.raises(ValueError, match="rounds"):
            arena.play(ARTICLE, rounds)


def test_the_seven_demo_rules_exist_and_run():
    assert set(DEMO) == {
        "random introductions",
        "most similar",
        "best for me",
        "dealbreakers then greedy",
        "the article's rule",
        "stable matching",
        "spread attention",
    }
    assert len(list(RULES.glob("*.py"))) == 7
    arena = Arena(SMALL, seed=0)
    for chosen in DEMO.values():
        play = arena.play(chosen, 4)
        assert play.introductions and chosen.description
    for file in RULES.glob("*.py"):
        body = [
            line
            for line in file.read_text().splitlines()
            if line.strip() and not line.startswith('"""')
        ]
        assert len(body) <= 12, f"{file.name} should stay a few lines long"


def test_rules_that_check_dealbreakers_never_break_a_known_one():
    arena = Arena(SMALL, seed=0)
    for name in (
        "dealbreakers then greedy",
        "the article's rule",
        "stable matching",
        "spread attention",
    ):
        assert all(i.known_ok for i in arena.play(DEMO[name], 5).introductions)
    assert not all(i.known_ok for i in arena.play(DEMO["most similar"], 5).introductions)


# --- the scorecard ------------------------------------------------------


def test_scorecard_is_plain_json_with_the_promised_parts():
    card = run(ARTICLE, "today", seed=0, people=200, rounds=5)
    assert json.loads(json.dumps(card)) == card
    assert {
        "rule",
        "world",
        "seed",
        "people",
        "rounds",
        "score",
        "stages",
        "introductions",
        "people_outcomes",
        "per_round",
        "cast",
        "events",
    } <= set(card)
    assert card["rule"] == "the article's rule" and card["people"] == 200 and card["rounds"] == 5
    assert [s["stage"] for s in card["stages"]] == [1, 2, 3, 4]
    for s in card["stages"]:
        assert 0 <= s["couples"] <= s["best_possible"]
        assert s["share_of_best"] == pytest.approx(s["couples"] / s["best_possible"], abs=1e-4)
        assert 0.0 <= s["informed_share_of_best"] <= 1.0
    assert card["score"] == card["stages"][-1]["share_of_best"]
    couples = [s["couples"] for s in card["stages"]]
    assert couples == sorted(couples, reverse=True)
    made = card["introductions"]
    assert (
        made["made"]
        == made["answered"] + made["expired"]
        == sum(r["introductions"] for r in card["per_round"])
    )
    assert made["wasted"] <= made["made"] and made["known_dealbreaker_violations"] == 0
    assert len(card["per_round"]) == 5 and [r["round"] for r in card["per_round"]] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert card["people_outcomes"]["hit_it_off"] == 2 * card["stages"][-1]["couples"]
    followed = {p["id"] for p in card["cast"]}
    assert len(followed) == card["people"] and {p["side"] for p in card["cast"]} == {"larger", "scarce"}
    assert set(card["cast"][0]) == {"id", "side", "age", "community", "x", "y", "possible"}
    assert set(card["fates"]) == {"gave_up", "left_as_couple"}
    assert card["events"] and all(e["a"] in followed or e["b"] in followed for e in card["events"])
    assert all(0 <= e["stage"] <= 4 and 1 <= e["round"] <= 5 for e in card["events"])


def test_run_accepts_a_preset_a_scenario_or_a_plain_function():
    by_name = run(ARTICLE, "casual", seed=1, people=150, rounds=2)
    by_scenario = run(ARTICLE, PRESETS["casual"].scenario, seed=1, people=150, rounds=2)
    assert by_name["stages"] == by_scenario["stages"] and by_name["world"] == "casual"

    def nobody(v):
        return []

    card = run(nobody, people=150, rounds=2)
    assert card["rule"] == "nobody" and card["score"] == 0.0
    assert card["people_outcomes"]["never_introduced"] == 150


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_world_can_be_played_and_leaves_couples_to_find(name):
    """A floor on the world: with a perfect read of profiles, couples can be formed."""
    arena = Arena(dataclasses.replace(PRESETS[name].scenario, pool_size=600), seed=0)
    assert arena.best_possible(LAST_STAGE) >= 50
    assert arena.couples(arena.informed_play(8), LAST_STAGE) >= 15


# --- tournament ---------------------------------------------------------


def test_tournament_output_has_the_documented_shape():
    result = run_tournament(rules_from(RULES), ["today", "casual"], seeds=2, people=150, rounds=3)
    assert json.loads(json.dumps(result)) == result
    assert result["worlds"] == ["today", "casual"] and result["seeds"] == 2
    assert {"best_possible_couples", "informed_benchmark"} == set(result["benchmarks"]["today"])
    board = result["leaderboard"]
    assert {row["rule"] for row in board} == set(DEMO)
    assert [row["overall"] for row in board] == sorted(
        (row["overall"] for row in board), reverse=True
    )
    for row in board:
        assert {"rule", "file", "description", "overall", "worlds"} == set(row)
        for world in result["worlds"]:
            entry = row["worlds"][world]
            assert {"mean", "spread", "low", "high"} == set(entry["score"])
            assert entry["score"]["low"] <= entry["score"]["mean"] <= entry["score"]["high"]
            assert set(entry["stages"]) == {"1", "2", "3", "4"}
    text = format_leaderboard(result)
    assert "not evidence about real users" in text and all(name in text for name in DEMO)


def test_a_broken_rule_is_recorded_and_does_not_stop_the_tournament():
    @rule("broken")
    def broken(v):
        return [(0, 0)]

    result = run_tournament(
        [("broken.py", broken), ("article.py", ARTICLE)], ["today"], seeds=1, people=120, rounds=2
    )
    rows = {row["rule"]: row for row in result["leaderboard"]}
    assert (
        "same side" in rows["broken"]["worlds"]["today"]["error"]
        and rows["broken"]["overall"] == 0.0
    )
    assert "score" in rows["the article's rule"]["worlds"]["today"]
    assert "broken failed in today" in format_leaderboard(result)


def test_rule_names_must_be_unique_and_files_must_hold_a_rule(tmp_path):
    (tmp_path / "one.py").write_text(
        'from wavelength.play import rule\n@rule("x")\ndef f(v): return []\n'
    )
    (tmp_path / "two.py").write_text(
        'from wavelength.play import rule\n@rule("x")\ndef g(v): return []\n'
    )
    with pytest.raises(ValueError, match="share a name"):
        rules_from(tmp_path)
    (tmp_path / "two.py").write_text("x = 1\n")
    with pytest.raises(ValueError, match="no rule found"):
        rules_from(tmp_path)


def test_cli_play_and_tournament(tmp_path, capsys):
    assert main(["play", str(RULES / "article.py"), "--people", "150", "--rounds", "3"]) == 0
    out = capsys.readouterr().out
    assert "of the couples that were possible" in out and "best possible" in out
    assert (
        main(["play", str(RULES / "article.py"), "--people", "150", "--rounds", "2", "--json"]) == 0
    )
    assert json.loads(capsys.readouterr().out)["rounds"] == 2
    board = tmp_path / "leaderboard.json"
    args = [
        "tournament",
        str(RULES),
        "--worlds",
        "today,swipe",
        "--seeds",
        "1",
        "--people",
        "120",
        "--rounds",
        "2",
        "--out",
        str(board),
    ]
    assert main(args) == 0
    assert "random introductions" in capsys.readouterr().out
    assert len(json.loads(board.read_text())["leaderboard"]) == 7
    assert main(["play", str(tmp_path / "missing.py")]) == 2
    assert capsys.readouterr().err.startswith("wavelength:")


# --- the browser surface ------------------------------------------------

GOOD = (
    '@rule("mine")\ndef mine(v):\n    return v.pair_mutual(0.5 * v.text + 0.5 * v.voice, top=5)\n'
)


def test_describe_settings_is_enough_to_draw_every_control():
    settings = web.describe_settings()
    assert json.loads(json.dumps(settings)) == settings
    keys = [s["key"] for s in settings]
    assert len(keys) == len(set(keys))
    scalar = {f.name for f in dataclasses.fields(Scenario) if f.type in ("float", "int")}
    assert scalar <= set(keys)
    assert {
        "archetypes.casual.share",
        "dealbreaker_shares.family_plans",
        "attraction_mix.manner",
        "community_shares.first",
    } <= set(keys)
    for s in settings:
        assert {
            "key",
            "group",
            "label",
            "meaning",
            "sensible",
            "basis",
            "type",
            "min",
            "max",
            "step",
            "default",
        } <= set(s)
        assert s["min"] <= s["default"] <= s["max"] and s["step"] > 0 and s["meaning"]
    # Every default, sent straight back, is accepted and changes nothing.
    echo = web.run({s["key"]: s["default"] for s in settings}, GOOD, people=120, rounds=2)
    plain = web.run({}, GOOD, people=120, rounds=2)
    assert "error" not in echo and echo["stages"] == plain["stages"]


def test_presets_and_rules_are_listed_as_data():
    presets = web.list_presets()
    assert [p["name"] for p in presets] == list(PRESETS)
    keys = {s["key"] for s in web.describe_settings()}
    for p in presets:
        assert set(p["settings"]) == keys and p["product"] and p["description"]
    assert json.loads(json.dumps(presets)) == presets
    listed = web.list_rules()
    assert {r["name"] for r in listed} == set(DEMO) and all("@rule" in r["source"] for r in listed)


def test_web_run_returns_a_scorecard_for_source_or_a_name():
    card = web.run({"preset": "casual", "yes_rate": 0.5}, GOOD, seed=2, people=150, rounds=3)
    assert card["rule"] == "mine" and card["world"] == "casual (modified)" and card["people"] == 150
    assert json.loads(json.dumps(card)) == card
    named = web.run({"preset": "today"}, "the article's rule", people=150, rounds=3)
    assert named == {**run(ARTICLE, "today", people=150, rounds=3), "world": "today"}
    with_import = "from wavelength.play import rule\nimport numpy\n" + GOOD
    assert web.run({}, with_import, people=120, rounds=1)["rule"] == "mine"


@pytest.mark.parametrize(
    "source, line, message",
    [
        ('@rule("x")\ndef f(v):\n    return v.text +\n', 3, "SyntaxError"),
        ('@rule("x")\ndef f(v):\n    s = v.text\n    return v.nope\n', 4, "AttributeError"),
        ("boom = 1 / 0\n", 1, "ZeroDivisionError"),
        ("x = 1\n", None, "no rule found"),
        ("", None, "name or its source"),
        ('@rule("x")\ndef f(v):\n    return [(0, 0)]\n', None, "same side"),
        ('@rule("x")\ndef f(v):\n    return np.zeros(3)[7]\n', 3, "IndexError"),
    ],
)
def test_web_run_reports_rule_errors_as_data(source, line, message):
    result = web.run({}, source, people=120, rounds=2)
    assert set(result) == {"error"} and result["error"]["where"] == "rule"
    assert result["error"]["line"] == line and message in result["error"]["message"]


@pytest.mark.parametrize(
    "settings, people, rounds, message",
    [
        ({"preset": "utopia"}, 120, 2, "unknown preset"),
        ({"yes_rate": 7}, 120, 2, "yes_rate"),
        ({"no_such_setting": 1}, 120, 2, "not a valid key"),
        ({"feature_coverage": 0.1}, 120, 2, "full_profile_share"),
        ({}, 5, 2, "people"),
        ({}, 10**6, 2, "people"),
        ({}, 120, 0, "rounds"),
    ],
)
def test_web_run_reports_setting_errors_as_data(settings, people, rounds, message):
    result = web.run(settings, GOOD, people=people, rounds=rounds)
    assert set(result) == {"error"} and result["error"]["where"] == "settings"
    assert message in result["error"]["message"]


def test_a_submitted_rule_gets_only_rule_numpy_and_builtins():
    probe = (
        '@rule("probe")\ndef probe(v):\n'
        '    names = sorted(k for k in globals() if not k.startswith("__"))\n'
        '    raise RuntimeError(",".join(names))\n'
    )
    assert (
        web.run({}, probe, people=120, rounds=1)["error"]["message"]
        == "RuntimeError: np,probe,rule"
    )
    # Each submission gets a fresh namespace: nothing carries over from the last one.
    web.run({}, "leftover = 1\n" + GOOD, people=120, rounds=1)
    peek = '@rule("peek")\ndef peek(v):\n    return leftover\n'
    assert "NameError" in web.run({}, peek, people=120, rounds=1)["error"]["message"]


def test_compare_plays_several_rules_against_the_same_world():
    result = web.compare(
        {"preset": "today"}, ["most similar", GOOD, "def broken(:"], seed=1, people=150, rounds=3
    )
    assert json.loads(json.dumps(result)) == result
    assert result["world"] == "today" and result["people"] == 150 and len(result["results"]) == 3
    similar, mine, broken = result["results"]
    assert similar["rule"] == "most similar" and mine["rule"] == "mine" and "error" in broken
    assert [s["best_possible"] for s in similar["stages"]] == [
        s["best_possible"] for s in mine["stages"]
    ]
    assert similar["cast"] == mine["cast"]
    assert web.compare({"preset": "nowhere"}, [GOOD])["error"]["where"] == "settings"


# --- preset files with citations ----------------------------------------


def test_a_preset_file_with_a_basis_per_field_loads(tmp_path):
    path = tmp_path / "somewhere.toml"
    path.write_text(
        'name = "somewhere"\nproduct = "introductions"\nsummary = "A test world."\n'
        'description = "Made up for a test."\nbase = "serious"\n\n'
        '[fields.endogamy_share]\nvalue = 0.6\nbasis = "Source: a made-up table."\n\n'
        "[fields.community_shares]\nvalue = { north = 0.7, south = 0.3 }\n"
        'basis = "Source: another made-up table."\n'
    )
    loaded = load_preset_file(path)
    assert loaded.name == "somewhere" and loaded.reasons["endogamy_share"].startswith("Source")
    scenario = loaded.scenario
    assert scenario.endogamy_share == 0.6 and dict(scenario.community_shares) == {
        "north": 0.7,
        "south": 0.3,
    }
    assert scenario.yes_rate == PRESETS["serious"].scenario.yes_rate  # inherited from the base
    assert load_scenario_file(path) == scenario
    assert run(ARTICLE, str(path), people=120, rounds=2)["world"] == "somewhere"


@pytest.mark.parametrize(
    "body, message",
    [
        ("[fields.endogamy_share]\nvalue = 0.6\n", "needs exactly 'value' and 'basis'"),
        ('[fields.endogamy_share]\nvalue = 0.6\nbasis = "  "\n', "has no basis"),
        ('[fields.endogamy_share]\nvalue = 6\nbasis = "x"\n', "endogamy_share"),
        ('[fields.no_such]\nvalue = 1\nbasis = "x"\n', "unknown scenario field"),
        ('colour = "red"\n[fields]\n', "unknown key"),
    ],
)
def test_a_preset_file_without_citations_or_with_bad_values_is_refused(tmp_path, body, message):
    path = tmp_path / "bad.toml"
    path.write_text(body)
    with pytest.raises(ValueError, match=message):
        load_preset_file(path)


def test_no_country_presets_or_statistics_ship_with_the_repository():
    assert not list(ROOT.glob("presets/*")) and not list(ROOT.glob("countries/*"))


# --- runs where the browser runs ----------------------------------------

STDLIB_ALLOWED = {
    "__future__",
    "argparse",
    "builtins",
    "collections",
    "dataclasses",
    "enum",
    "json",
    "math",
    "pathlib",
    "statistics",
    "sys",
    "textwrap",
    "tomllib",
    "traceback",
    "types",
    "typing",
}


def test_the_package_imports_only_numpy_and_a_short_list_of_stdlib_modules():
    """Nothing deprecated or removed by Python 3.14, no C extensions, no threads, no network."""
    for file in sorted((ROOT / "src" / "wavelength").glob("*.py")):
        tree = ast.parse(file.read_text(), filename=str(file), feature_version=(3, 12))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            for name in names:
                top = name.split(".")[0]
                assert top in STDLIB_ALLOWED | {"numpy", "wavelength"}, (
                    f"{file.name} imports {name}"
                )
    assert sys.version_info >= (3, 12)


def test_the_engine_path_does_not_touch_files_or_the_network():
    for name in (
        "engine",
        "view",
        "key",
        "decisions",
        "population",
        "observation",
        "best",
        "world",
    ):
        source = (ROOT / "src" / "wavelength" / f"{name}.py").read_text()
        for banned in ("open(", "pathlib", "socket", "urllib", "threading", "subprocess", "os."):
            assert banned not in source, f"{name}.py uses {banned}"
