"""The answer key and the world behind it: determinism, stages, channels, culture."""

import dataclasses

import numpy as np
import pytest

from wavelength.diagnostics import mean_rank_correlation
from wavelength.features import FEATURE_NAMES, FEATURES
from wavelength.key import LAST_STAGE, STAGES
from wavelength.population import generate_population, importance_from_weights
from wavelength.presets import PRESETS
from wavelength.scenario import Scenario
from wavelength.scoring import FEATURE_WEIGHT, TEXT_WEIGHT, VISUAL_WEIGHT, score_matrix
from wavelength.view import Board
from wavelength.world import generate_world

SMALL = dataclasses.replace(Scenario(), pool_size=300)
TIMELINE = FEATURE_NAMES.index("relationship_timeline")


def _world(seed=0, **changes):
    return generate_world(seed, dataclasses.replace(SMALL, **changes))


# --- the key ------------------------------------------------------------


def test_the_key_is_fixed_by_the_seed():
    first, second = _world(4), _world(4)
    assert np.array_equal(first.key.stage, second.key.stage)
    assert np.array_equal(first.key.informed, second.key.informed)
    assert not np.array_equal(first.key.stage, _world(5).key.stage)


@pytest.mark.parametrize("name", list(PRESETS))
def test_stages_are_nested_symmetric_and_only_across_sides(name):
    world = generate_world(1, dataclasses.replace(PRESETS[name].scenario, pool_size=300))
    stage, across = world.key.stage, world.population.across
    assert stage.min() >= 0 and stage.max() <= LAST_STAGE == len(STAGES) - 1
    assert np.array_equal(stage, stage.T)
    assert not stage[~across].any()
    counts = [int(world.key.reaches(k).sum()) for k in range(1, LAST_STAGE + 1)]
    assert counts == sorted(counts, reverse=True) and counts[-1] > 0
    for k in range(2, LAST_STAGE + 1):  # nobody reaches a stage without the one before
        assert not (world.key.reaches(k) & ~world.key.reaches(k - 1)).any()


def test_stage_one_is_exactly_both_saying_yes():
    world = _world()
    yes = world.decisions.would_accept
    assert np.array_equal(world.key.reaches(1), yes & yes.T)


def test_each_later_stage_needs_its_stated_condition():
    world = _world(
        family_approval_share=0.5,
        distance_tolerance=0.2,
        scarce_side_initiates=0.2,
        larger_side_initiates=0.2,
    )
    key, decisions = world.key, world.decisions
    assert not (key.reaches(2) & ~key.conversation_possible).any()
    assert not (key.reaches(3) & ~key.within_reach).any()
    assert not (key.reaches(4) & ~key.families_fit).any()
    broken = decisions.refuses_in_person | decisions.refuses_in_person.T
    assert not (key.reaches(4) & broken).any()
    # ... and each condition does rule out pairs that had got that far.
    assert (key.reaches(1) & ~key.conversation_possible).any()
    assert (key.reaches(2) & ~key.within_reach).any()
    assert (key.reaches(3) & broken).any()


def test_stage_rates_set_how_many_pairs_pass_not_which():
    def reaching(k, **changes):
        return int(_world(**changes).key.reaches(k).sum())

    assert reaching(2, conversation_rate=0.9) > reaching(2, conversation_rate=0.2)
    assert reaching(3, date_rate=0.9) > reaching(3, date_rate=0.2)
    assert reaching(4, second_date_rate=0.9) > reaching(4, second_date_rate=0.2)
    # Changing a later stage's rate leaves the earlier stages exactly as they were.
    assert reaching(1, second_date_rate=0.9) == reaching(1, second_date_rate=0.2)
    assert reaching(2, date_rate=0.9) == reaching(2, date_rate=0.2)


def test_better_fitting_pairs_get_further():
    world = _world()
    eligible = np.triu(world.decisions.eligible)
    informed, stage = world.key.informed[eligible], world.key.stage[eligible]
    likely = informed >= np.quantile(informed, 0.9)
    assert (stage[likely] == LAST_STAGE).mean() > 5 * (stage[~likely] == LAST_STAGE).mean()
    assert 0.0 <= informed.min() and informed.max() <= 1.0
    assert np.allclose(world.key.informed, world.key.informed.T)


def test_what_the_app_observes_does_not_change_the_truth_or_the_key():
    sharp, blind = _world(text_signal_quality=1.0), _world(text_signal_quality=0.0)
    assert np.array_equal(sharp.key.stage, blind.key.stage)
    assert np.array_equal(sharp.population.codes, blind.population.codes)
    assert not np.array_equal(
        sharp.observations.text_embeddings, blind.observations.text_embeddings
    )


# --- channels -----------------------------------------------------------


def _tracking(world, score):
    return mean_rank_correlation(score, world.decisions.felt, world.observations.known_ok)


def test_each_channel_carries_its_own_information():
    world = generate_world(0, dataclasses.replace(Scenario(), pool_size=500))
    board = Board(world.observations.copy())
    alone = {name: _tracking(world, getattr(board, name)) for name in ("text", "photo", "voice")}
    assert all(value > 0.05 for value in alone.values())
    ok = world.observations.known_ok

    def scaled(x):  # the channels have different spreads; put them on one scale first
        return (x - x[ok].mean()) / x[ok].std()

    together = _tracking(world, sum(scaled(getattr(board, name)) for name in alone))
    assert together > max(alone.values()) + 0.03  # no one channel stands in for the others
    # The hidden traits behind the channels are unrelated to each other.
    people = world.population
    vibe, manner = people.vibe @ people.vibe.T, people.manner @ people.manner.T
    assert abs(np.corrcoef(vibe[people.across], manner[people.across])[0, 1]) < 0.1


@pytest.mark.parametrize("channel", ["text", "photo", "voice"])
def test_a_channel_with_no_quality_says_nothing(channel):
    setting = {
        "text": "text_signal_quality",
        "photo": "photo_signal_quality",
        "voice": "voice_signal_quality",
    }[channel]
    world = generate_world(0, dataclasses.replace(Scenario(), pool_size=500, **{setting: 0.0}))
    assert abs(_tracking(world, getattr(Board(world.observations.copy()), channel))) < 0.04


def test_hidden_photos_give_a_neutral_photo_signal():
    world = _world(photos_hidden_share=0.5)
    board = Board(world.observations.copy())
    hidden = world.observations.photos_hidden
    assert (
        hidden.any()
        and (board.photo[hidden] == 0.5).all()
        and (board.photo[:, hidden] == 0.5).all()
    )


def test_board_feature_similarity_is_the_scorers_feature_term():
    world = _world()
    obs = world.observations
    board = Board(obs.copy())
    whole = score_matrix(obs.codes, obs.weights, obs.text_embeddings, obs.visual_embeddings)
    photo = (
        unit := obs.visual_embeddings / np.linalg.norm(obs.visual_embeddings, axis=1, keepdims=True)
    ) @ unit.T
    rebuilt = (
        TEXT_WEIGHT * board.text
        + VISUAL_WEIGHT * (photo + 1) / 2
        + FEATURE_WEIGHT * board.feature_similarity(obs.weights)
    )
    assert np.allclose(whole, rebuilt)


# --- preference diversity ----------------------------------------------


def test_importance_is_shared_without_diversity_and_personal_with_it():
    weights = np.random.default_rng(0).uniform(1, 10, size=(50, len(FEATURE_NAMES)))
    alike = importance_from_weights(weights, 0.0)
    own = importance_from_weights(weights, 1.0)
    assert np.allclose(alike, alike[0]) and np.allclose(alike.sum(axis=1), 1.0)
    assert np.allclose(own.sum(axis=1), 1.0) and own.std(axis=0).mean() > 0.02
    assert own.max(axis=1).mean() > alike.max(axis=1).mean()  # attention concentrates


def test_knowing_true_weights_matters_more_as_preferences_diverge():
    """With true values known, personal weights beat global ones more as diversity rises."""

    def advantage(diversity):
        gaps = []
        for seed in range(3):
            world = generate_world(
                seed, dataclasses.replace(Scenario(), pool_size=400, preference_diversity=diversity)
            )
            truth = dataclasses.replace(
                world.observations.copy(), codes=world.population.codes.copy()
            )
            board = Board(truth)
            personal = _tracking(world, board.feature_similarity(world.population.weights))
            shared = _tracking(world, board.feature_similarity(board.global_weights))
            gaps.append(personal - shared)
        return float(np.mean(gaps))

    assert advantage(1.0) > advantage(0.0) + 0.02
    assert advantage(1.0) > 0.02


# --- culture and attention ----------------------------------------------


def _people(n=6000, seed=0, **changes):
    scenario = dataclasses.replace(Scenario(), pool_size=n, **changes)
    return scenario, generate_population(np.random.default_rng(seed), scenario)


def test_culture_shares_are_realised():
    scenario, people = _people(
        endogamy_share=0.3, family_approval_share=0.4, photos_hidden_share=0.2, intent_openness=0.7
    )
    assert people.endogamous.mean() == pytest.approx(0.3, abs=0.025)
    assert people.family_required.mean() == pytest.approx(0.4, abs=0.025)
    assert people.photos_hidden.mean() == pytest.approx(0.2, abs=0.025)
    assert people.open_about_intent.mean() == pytest.approx(0.7, abs=0.025)
    for index, share in enumerate(scenario.community_shares.values()):
        assert np.mean(people.community == index) == pytest.approx(share, abs=0.025)
    larger, scarce = people.initiates[people.majority], people.initiates[~people.majority]
    assert larger.mean() == pytest.approx(scenario.larger_side_initiates, abs=0.03)
    assert scarce.mean() == pytest.approx(scenario.scarce_side_initiates, abs=0.03)
    assert np.median(people.travel) == pytest.approx(scenario.distance_tolerance, rel=0.1)
    assert np.median(people.patience) == pytest.approx(scenario.patience, abs=1)


def test_marriage_pressure_moves_older_people_towards_marriage():
    marriage = FEATURES["relationship_timeline"].code("marriage_focused")

    def share_wanting_marriage(pressure, older):
        _, people = _people(marriage_pressure=pressure)
        chosen = people.age >= 34 if older else people.age <= 25
        return np.mean(people.codes[chosen, TIMELINE] == marriage)

    assert share_wanting_marriage(1.0, older=True) > share_wanting_marriage(0.0, older=True) + 0.2
    assert share_wanting_marriage(1.0, older=False) == pytest.approx(
        share_wanting_marriage(0.0, older=False), abs=0.03
    )


def test_people_who_are_not_open_state_the_usual_answer_and_hide_community_rules():
    world = _world(intent_openness=0.5, endogamy_share=0.5)
    people, obs = world.population, world.observations
    quiet = ~people.open_about_intent
    usual = np.bincount(people.codes[:, TIMELINE]).argmax()
    assert (people.stated_codes[quiet, TIMELINE] == usual).all()
    assert not obs.community_required[quiet].any()
    assert np.array_equal(obs.community_required, people.endogamous & people.open_about_intent)
    # A stated community rule is a known dealbreaker; an unstated one is only a refusal.
    other = people.community[:, None] != people.community[None, :]
    assert not (obs.known_ok & obs.community_required[:, None] & other).any()
    assert not (world.decisions.would_accept & people.endogamous[:, None] & other).any()


def test_endogamy_family_and_distance_cut_the_number_of_couples():
    def couples(**changes):
        return int(_world(**changes).key.reaches(LAST_STAGE).sum())

    assert couples(endogamy_share=0.9) < couples(endogamy_share=0.0)
    assert couples(family_approval_share=0.9) < couples(family_approval_share=0.0)
    assert couples(distance_tolerance=0.05) < couples(distance_tolerance=1.5)
    assert couples(larger_side_initiates=0.3, scarce_side_initiates=0.05) < couples(
        larger_side_initiates=1.0, scarce_side_initiates=1.0
    )


def test_family_visibility_sets_how_much_the_app_sees_of_family_fit():
    def seen(visibility):
        world = _world(family_visibility=visibility)
        truth = world.population.family @ world.population.family.T
        across = world.population.across
        return np.corrcoef(Board(world.observations.copy()).family[across], truth[across])[0, 1]

    assert seen(1.0) > 0.99 and abs(seen(0.0)) < 0.1


def test_open_habits_follow_the_rate_and_the_spread():
    _, even = _people(open_rate=0.6, open_rate_spread=0.0)
    _, split = _people(open_rate=0.6, open_rate_spread=1.0)
    _, mixed = _people(open_rate=0.6, open_rate_spread=0.3)
    assert (even.habit == 0.6).all()
    assert set(np.unique(split.habit)) == {0.0, 1.0} and split.habit.mean() == pytest.approx(
        0.6, abs=0.02
    )
    assert mixed.habit.mean() == pytest.approx(0.6, abs=0.02) and 0.1 < mixed.habit.std() < 0.45
