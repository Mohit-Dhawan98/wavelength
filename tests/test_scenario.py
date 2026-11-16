"""Scenarios: validation, presets, plain-data round trips, files and --set overrides."""

import dataclasses
import json

import pytest

from wavelength.features import FEATURES
from wavelength.presets import PRESETS, load_scenario_file, preset, resolve_scenario
from wavelength.scenario import (
    ATTRACTION_PARTS,
    CORE_FEATURES,
    DEALBREAKER_FEATURES,
    TRAIT_FEATURES,
    Archetype,
    Scenario,
    scenario_fields,
)
from wavelength.scenario_io import from_dict, to_dict, with_overrides


def _archetype(**changes):
    spec = dict(
        name="x",
        description="",
        share=1.0,
        relationship_timeline={"casual": 1.0},
        commitment_style={"flexible": 1.0},
        family_plans={"undecided": 1.0},
        trust_boundaries={"balanced": 1.0},
    )
    return Archetype(**{**spec, **changes})


# --- validation ---------------------------------------------------------


def test_default_scenario_is_valid_and_fully_documented():
    Scenario()
    for f in scenario_fields():
        for key in ("group", "meaning", "sensible", "basis"):
            assert f.metadata[key].strip(), f"{f.name} lacks {key}"
        assert f.metadata["basis"].startswith(("assumption", "source", "direction from a source"))


def test_archetype_shares_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        Scenario(archetypes=(_archetype(share=0.6), _archetype(name="y", share=0.6)))


def test_archetype_tendencies_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        _archetype(relationship_timeline={"casual": 0.5})


def test_archetype_rejects_unknown_value_and_negative_share():
    with pytest.raises(ValueError, match="unknown value"):
        _archetype(family_plans={"wants_ten_children": 1.0})
    with pytest.raises(ValueError, match="negative"):
        _archetype(family_plans={"undecided": 1.5, "childfree": -0.5})


def test_archetype_names_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        Scenario(archetypes=(_archetype(share=0.5), _archetype(share=0.5)))


def test_omitted_values_are_zero_and_order_is_fixed():
    kind = _archetype()
    assert tuple(kind.relationship_timeline) == FEATURES["relationship_timeline"].values
    assert kind.relationship_timeline["marriage_focused"] == 0.0


@pytest.mark.parametrize(
    "field, bad",
    [
        ("yes_rate", 0.0),
        ("yes_rate", 1.5),
        ("majority_share", 0.4),
        ("majority_share", 1.0),
        ("active_share", -0.1),
        ("chemistry_share", 1.1),
        ("self_knowledge", 2.0),
        ("yes_rate_spread", -1.0),
        ("pool_size", 5),
        ("pool_size", 300.5),
        ("yes_rate", "high"),
    ],
)
def test_out_of_range_values_are_rejected(field, bad):
    with pytest.raises(ValueError, match=field):
        dataclasses.replace(Scenario(), **{field: bad})


def test_mix_and_trait_shares_must_sum_to_one():
    with pytest.raises(ValueError, match="attraction_mix"):
        Scenario(attraction_mix={"values": 0.5, "vibe": 0.5, "lifestyle": 0.5})
    traits = {name: dict(shares) for name, shares in Scenario().trait_shares.items()}
    traits["social_energy"] = {"extrovert": 0.9}
    with pytest.raises(ValueError, match="trait_shares.social_energy"):
        Scenario(trait_shares=traits)


def test_dealbreaker_shares_cover_exactly_the_core_features():
    with pytest.raises(ValueError, match="dealbreaker_shares"):
        Scenario(dealbreaker_shares={"relationship_timeline": 0.5})
    assert set(Scenario().dealbreaker_shares) == set(CORE_FEATURES)


def test_profile_completeness_fields_must_be_consistent():
    with pytest.raises(ValueError, match="full_profile_share"):
        Scenario(feature_coverage=0.3)  # below the default full_profile_share
    with pytest.raises(ValueError, match="dealbreakers_known_share"):
        Scenario(dealbreakers_known_share=0.2)


def test_implied_value_shares_sum_to_one():
    for p in PRESETS.values():
        for feature in (*DEALBREAKER_FEATURES, *TRAIT_FEATURES):
            assert sum(p.scenario.value_shares(feature).values()) == pytest.approx(1.0)


def test_replace_changes_one_field_and_leaves_the_original():
    base = Scenario()
    changed = dataclasses.replace(base, yes_rate=0.08)
    assert changed.yes_rate == 0.08 and base.yes_rate != 0.08
    assert changed.archetypes == base.archetypes


# --- presets ------------------------------------------------------------


def test_the_four_named_presets_exist_and_load():
    assert list(PRESETS) == ["today", "casual", "serious", "swipe", "dystopia"]
    assert {p.product for p in PRESETS.values()} == {"introductions", "swipe feed"}
    for p in PRESETS.values():
        assert isinstance(p.scenario, Scenario)
        assert len(p.description) > 100
        assert set(p.overrides) == set(p.reasons)


def test_today_is_the_defaults():
    assert preset("today").scenario == Scenario()


def test_unknown_preset_is_an_error():
    with pytest.raises(ValueError, match="unknown preset"):
        preset("utopia")


def _timeline(name):
    return PRESETS[name].scenario.value_shares("relationship_timeline")


def test_casual_and_serious_differ_in_what_people_want():
    casual, serious, today = _timeline("casual"), _timeline("serious"), _timeline("today")
    assert casual["casual"] > today["casual"] > serious["casual"]
    assert serious["marriage_focused"] > today["marriage_focused"] > casual["marriage_focused"]
    assert casual["casual"] + casual["exploring"] > 0.5
    assert serious["marriage_focused"] + serious["long_term"] > 0.8


def test_casual_and_serious_differ_in_dealbreakers_and_yes_rate():
    casual, serious, today = (PRESETS[n].scenario for n in ("casual", "serious", "today"))
    for feature in CORE_FEATURES:
        assert (
            casual.dealbreaker_shares[feature]
            < today.dealbreaker_shares[feature]
            < serious.dealbreaker_shares[feature]
        )
    assert casual.yes_rate > today.yes_rate > serious.yes_rate
    assert serious.active_share > today.active_share


def test_dystopia_is_worse_than_today_on_every_input_it_touches():
    dystopia, today = PRESETS["dystopia"].scenario, PRESETS["today"].scenario
    assert dystopia.majority_share > today.majority_share
    assert dystopia.minority_yes_ratio < today.minority_yes_ratio
    assert dystopia.decision_noise > today.decision_noise
    assert dystopia.yes_rate_spread > today.yes_rate_spread
    assert dystopia.desirability_consensus > today.desirability_consensus
    assert dystopia.self_knowledge < today.self_knowledge
    assert dystopia.feature_coverage < today.feature_coverage
    assert dystopia.dealbreakers_known_share < today.dealbreakers_known_share
    assert dystopia.active_share < today.active_share
    assert dystopia.text_signal_quality < today.text_signal_quality
    assert dystopia.archetypes == today.archetypes  # intentions deliberately unchanged


# --- plain data, files and overrides ------------------------------------


@pytest.mark.parametrize("name", list(PRESETS))
def test_plain_data_round_trip(name):
    scenario = PRESETS[name].scenario
    data = to_dict(scenario)
    assert from_dict(json.loads(json.dumps(data))) == scenario


def test_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown scenario field"):
        from_dict({"pickyness": 0.1})


def test_from_dict_merges_per_feature_and_replaces_distributions():
    scenario = from_dict(
        {
            "dealbreaker_shares": {"family_plans": 0.9},
            "archetypes": {"casual": {"relationship_timeline": {"casual": 1.0}}},
        }
    )
    assert scenario.dealbreaker_shares["family_plans"] == 0.9
    assert (
        scenario.dealbreaker_shares["trust_boundaries"]
        == Scenario().dealbreaker_shares["trust_boundaries"]
    )
    assert scenario.archetype("casual").relationship_timeline["casual"] == 1.0
    assert scenario.archetype("casual").relationship_timeline["exploring"] == 0.0


def test_from_dict_can_add_an_archetype_and_demands_a_full_one():
    shares = {
        n: {"share": 0.2} for n in ("settling_down", "relationship_seeking", "exploring", "casual")
    }
    new = {
        "share": 0.2,
        "relationship_timeline": {"long_term": 1.0},
        "commitment_style": {"open_relationship": 1.0},
        "family_plans": {"childfree": 1.0},
        "trust_boundaries": {"very_flexible": 1.0},
    }
    scenario = from_dict({"archetypes": {**shares, "poly": new}})
    assert len(scenario.archetypes) == 5 and scenario.archetype("poly").share == 0.2
    with pytest.raises(ValueError, match="new archetype also needs"):
        from_dict({"archetypes": {**shares, "poly": {"share": 0.2}}})


def test_toml_file_builds_on_a_named_preset(tmp_path):
    path = tmp_path / "small_town.toml"
    path.write_text(
        'base = "serious"\n'
        "pool_size = 120\n"
        "yes_rate = 0.15\n"
        "[dealbreaker_shares]\n"
        "family_plans = 0.95\n"
        "[archetypes.settling_down]\n"
        "share = 0.60\n"
        "[archetypes.relationship_seeking]\n"
        "share = 0.30\n"
    )
    scenario = load_scenario_file(path)
    serious = PRESETS["serious"].scenario
    assert scenario.pool_size == 120 and scenario.yes_rate == 0.15
    assert scenario.dealbreaker_shares["family_plans"] == 0.95
    assert scenario.archetype("settling_down").share == 0.60
    assert scenario.active_share == serious.active_share  # inherited from the base
    assert resolve_scenario(str(path)) == ("small_town", scenario)


def test_json_file_defaults_to_today(tmp_path):
    path = tmp_path / "picky.json"
    path.write_text(json.dumps({"yes_rate": 0.03}))
    assert load_scenario_file(path) == dataclasses.replace(Scenario(), yes_rate=0.03)


def test_bad_scenario_files_are_errors(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[archetypes.casual]\nshare = 0.9\n")  # shares no longer sum to 1
    with pytest.raises(ValueError, match="sum to 1"):
        load_scenario_file(bad)
    with pytest.raises(ValueError, match="neither a preset"):
        resolve_scenario("scenario.yaml")
    with pytest.raises(ValueError, match="not found"):
        resolve_scenario(str(tmp_path / "missing.toml"))


def test_set_overrides_a_single_field():
    base = Scenario()
    changed = with_overrides(base, ["yes_rate=0.08", "pool_size=300"])
    assert changed == dataclasses.replace(base, yes_rate=0.08, pool_size=300)
    assert with_overrides(base, []) == base


def test_set_reaches_into_mappings():
    changed = with_overrides(Scenario(), ["dealbreaker_shares.family_plans=0.9"])
    assert changed.dealbreaker_shares["family_plans"] == 0.9
    assert (
        changed.dealbreaker_shares["commitment_style"]
        == Scenario().dealbreaker_shares["commitment_style"]
    )


def test_set_on_an_archetype_share_rescales_the_others():
    base = Scenario()
    changed = with_overrides(base, ["archetypes.casual.share=0.5"])
    assert changed.archetype("casual").share == pytest.approx(0.5)
    assert sum(a.share for a in changed.archetypes) == pytest.approx(1.0)
    ratio = base.archetype("exploring").share / base.archetype("settling_down").share
    assert changed.archetype("exploring").share / changed.archetype(
        "settling_down"
    ).share == pytest.approx(ratio)


def test_set_on_a_tendency_or_mix_rescales_its_siblings():
    changed = with_overrides(
        Scenario(),
        ["archetypes.casual.relationship_timeline.casual=0.9", "attraction_mix.values=0.8"],
    )
    timeline = changed.archetype("casual").relationship_timeline
    assert timeline["casual"] == pytest.approx(0.9) and timeline["exploring"] == pytest.approx(0.1)
    assert changed.attraction_mix["values"] == pytest.approx(0.8)
    assert set(changed.attraction_mix) == set(ATTRACTION_PARTS)


@pytest.mark.parametrize(
    "assignment, message",
    [
        ("yes_rate", "key=value"),
        ("nope=1", "not a valid key"),
        ("archetypes.nobody.share=0.5", "not in archetypes"),
        ("yes_rate=7", "yes_rate"),
        ("pickiness=0.1", "pickiness is now yes_rate"),
        ("archetypes.casual.share=1.5", "between 0 and 1"),
    ],
)
def test_bad_set_overrides_are_errors(assignment, message):
    with pytest.raises(ValueError, match=message):
        with_overrides(Scenario(), [assignment])


def test_swipe_differs_from_today_only_in_product_not_in_people():
    swipe, today = PRESETS["swipe"].scenario, PRESETS["today"].scenario
    assert PRESETS["swipe"].product == "swipe feed" and PRESETS["today"].product == "introductions"
    assert swipe.yes_rate < today.yes_rate / 2
    assert swipe.decision_noise > today.decision_noise
    assert swipe.feature_coverage < today.feature_coverage
    assert swipe.match_feedback_reach < today.match_feedback_reach
    assert swipe.archetypes == today.archetypes
    assert swipe.dealbreaker_shares == today.dealbreaker_shares


def test_every_preset_description_names_its_product():
    for p in PRESETS.values():
        word = "swipe feed" if p.product == "swipe feed" else "introductions"
        assert word in p.description


def test_renamed_fields_get_a_pointer_in_files_too():
    with pytest.raises(ValueError, match="pickiness is now yes_rate"):
        from_dict({"pickiness": 0.1})
