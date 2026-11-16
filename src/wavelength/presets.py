"""Named worlds: each preset is a short list of overrides and the reason for each.

``today`` is ``Scenario()`` with nothing changed; the reasons for its numbers
are the ``basis`` notes on the fields themselves. Every other preset changes
only the fields that make its world different, in the same plain-data form
as a scenario file, and gives one reason per changed field.

Each preset also says which kind of product it assumes, because that
decides what a yes means.

In an introductions product there is no feed. A person talks at length with
the app when they join, and is then offered a small number of chosen matches
one at a time, each with the reasons for it. They answer yes or no; two
people are introduced only when both have said yes; an offer nobody answers
expires after a couple of days. After a date, people come back and talk
about how it went, which is where post-date signals come from.

In a swipe feed a person judges a long stream of profiles in seconds, a yes
costs nothing, and nobody is asked for a reason.

The same person says yes far more readily to a considered introduction than
to a profile in a feed. The article describes an introductions product, so
that is what every preset except ``swipe`` assumes.

None of these numbers was tuned to produce any particular outcome, and in
particular none was set by working backwards from the acceptance rates
reported in the article. They are positions to argue with. Unless a reason
names a source, it is an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from wavelength.scenario import Scenario
from wavelength.scenario_io import from_dict, read_scenario_file

INTRODUCTIONS = "introductions"
SWIPE_FEED = "swipe feed"


@dataclass(frozen=True)
class Preset:
    """A named world.

    ``product`` is the kind of product the world assumes. ``overrides`` is
    what differs from ``Scenario()``; ``reasons`` has one entry per
    overridden field saying why that number was chosen.
    """

    name: str
    product: str
    summary: str
    description: str
    overrides: Mapping[str, Any]
    reasons: Mapping[str, str]
    base: Scenario | None = None  # what the overrides are applied to; None means the defaults

    def __post_init__(self) -> None:
        if set(self.overrides) != set(self.reasons):
            raise ValueError(f"preset {self.name}: every override needs exactly one reason")

    @property
    def scenario(self) -> Scenario:
        """The full scenario: the base with this preset's overrides applied."""
        return from_dict(self.overrides, base=self.base)


def _archetype_shares(
    settling_down: float, relationship_seeking: float, exploring: float, casual: float
) -> dict[str, dict[str, float]]:
    """Move share between the default archetypes, keeping their tendencies."""
    return {
        "settling_down": {"share": settling_down},
        "relationship_seeking": {"share": relationship_seeking},
        "exploring": {"share": exploring},
        "casual": {"share": casual},
    }


TODAY = Preset(
    name="today",
    product=INTRODUCTIONS,
    summary="Today's mix of people, on an introductions product.",
    description=(
        "The people of a present-day general-purpose app in a large city, "
        "using an introductions product: each person is sent a few considered "
        "matches and asked for a yes or no. Intentions are mixed: a little "
        "over half the pool leans towards a lasting relationship and the rest "
        "are exploring or want something casual. There are somewhat more "
        "people on one side than the other. A typical person would say yes to "
        "about three in ten eligible strangers, knows themselves imperfectly, "
        "and about a third of profiles are complete. This preset is the "
        "defaults; each field's own 'basis' note says where its number comes "
        "from."
    ),
    overrides={},
    reasons={},
)

CASUAL = Preset(
    name="casual",
    product=INTRODUCTIONS,
    summary="Most people want something casual or are exploring; few dealbreakers.",
    description=(
        "An introductions product that people use for dates, not for a "
        "partner. Four in five are exploring or want something casual, and "
        "because little is at stake in one evening, few people hold firm "
        "dealbreakers and most say yes more readily. What draws people is "
        "less about shared life plans and more about how someone comes across "
        "and broad appeal. The market itself (size, balance, engagement) is "
        "the same as today's, so any difference comes from what people want."
    ),
    overrides={
        "archetypes": _archetype_shares(0.05, 0.15, 0.40, 0.40),
        "dealbreaker_shares": {
            "relationship_timeline": 0.25,
            "commitment_style": 0.20,
            "family_plans": 0.10,
            "trust_boundaries": 0.15,
            "emotional_availability": 0.15,
        },
        "yes_rate": 0.40,
        "attraction_mix": {"values": 0.25, "vibe": 0.30, "lifestyle": 0.20, "manner": 0.25},
        "desirability_consensus": 0.40,
    },
    reasons={
        "archetypes": (
            "The defining change: 80% exploring or casual, 20% wanting a lasting "
            "relationship. Assumption."
        ),
        "dealbreaker_shares": (
            "Roughly half of today's or less. Family plans falls furthest, since "
            "children matter little to a casual date. Assumption."
        ),
        "yes_rate": (
            "Four in ten eligible strangers, against three in ten today: a date "
            "costs less than a commitment. Assumption."
        ),
        "attraction_mix": (
            "Life plans carry less of the attraction; how someone sounds and "
            "lives carries more. Assumption."
        ),
        "desirability_consensus": (
            "Higher than today: with less weight on personal fit, people agree "
            "more on who is appealing. Assumption."
        ),
    },
)

SERIOUS = Preset(
    name="serious",
    product=INTRODUCTIONS,
    summary="Marriage-minded, many firm dealbreakers, high intent.",
    description=(
        "A matrimony-style introductions service. Nine in ten want marriage "
        "or a lasting relationship and most hold firm dealbreakers on "
        "timeline, commitment and children. People say yes less readily "
        "because the decision is larger, but they arrive knowing what they "
        "want, answer the questions they are asked, and respond to "
        "introductions. The two sides are close to balanced."
    ),
    overrides={
        "archetypes": _archetype_shares(0.55, 0.35, 0.08, 0.02),
        "dealbreaker_shares": {
            "relationship_timeline": 0.85,
            "commitment_style": 0.80,
            "family_plans": 0.75,
            "trust_boundaries": 0.50,
            "emotional_availability": 0.40,
        },
        "yes_rate": 0.20,
        "majority_share": 0.55,
        "open_rate": 0.85,
        "family_approval_share": 0.40,
        "family_visibility": 0.70,
        "endogamy_share": 0.40,
        "marriage_pressure": 0.50,
        "intent_openness": 0.95,
        "self_knowledge": 0.65,
        "onboarding_reach": 0.90,
        "dealbreakers_known_share": 0.98,
        "full_profile_share": 0.50,
        "feature_coverage": 0.90,
        "attraction_mix": {"values": 0.60, "vibe": 0.15, "lifestyle": 0.10, "manner": 0.15},
    },
    reasons={
        "archetypes": (
            "The defining change: 90% settling down or relationship-seeking. Assumption."
        ),
        "dealbreaker_shares": (
            "Most people here will not bend on timeline, commitment or children. Assumption."
        ),
        "yes_rate": (
            "Two in ten eligible strangers, against three in ten today: the "
            "stakes are higher. Assumption."
        ),
        "majority_share": "Closer to balanced than a mainstream app. Assumption.",
        "open_rate": "People who sign up to find a spouse mostly answer. Assumption.",
        "family_approval_share": (
            "Four in ten need their family's approval: marriage here is partly "
            "a family matter. Assumption; not a figure for any real population."
        ),
        "family_visibility": "The service asks about family in detail. Assumption.",
        "endogamy_share": (
            "Four in ten will only marry within their community. Assumption; "
            "not a figure for any real population."
        ),
        "marriage_pressure": "Age pushes people towards marriage more than today. Assumption.",
        "intent_openness": "People come to say what they want. Assumption.",
        "self_knowledge": (
            "Somewhat better than today: people have thought about what they "
            "need, though still imperfectly. Assumption."
        ),
        "onboarding_reach": "A long, structured intake that people complete. Assumption.",
        "dealbreakers_known_share": "The service asks the four questions up front. Assumption.",
        "full_profile_share": "Half the profiles are complete, against a third today. Assumption.",
        "feature_coverage": "Slightly above the article's conversational figure. Assumption.",
        "attraction_mix": "Shared life plans carry most of the attraction. Assumption.",
    },
)

SWIPE = Preset(
    name="swipe",
    product=SWIPE_FEED,
    summary="Today's people on a high-volume swipe feed: quick judgements, low yes-rate.",
    description=(
        "The same people as today, on a different product: a swipe feed. "
        "Each person judges a long stream of profiles in seconds, so a "
        "typical person says yes to about one in ten eligible strangers, "
        "decisions are noisier, and a yes costs nothing and promises little. "
        "Profiles are a few photos and a short form, not a conversation, so "
        "the app knows less about what anyone wants and learns little from "
        "reasons, because nobody is asked for one. More accounts are idle. "
        "The strategies in this repository were designed for the other kind "
        "of product; this preset is here to show what changes when that "
        "assumption is dropped."
    ),
    overrides={
        "yes_rate": 0.10,
        "decision_noise": 0.25,
        "desirability_consensus": 0.45,
        "open_rate": 0.60,
        "fatigue": 0.15,
        "intent_openness": 0.70,
        "voice_signal_quality": 0.0,
        "family_visibility": 0.0,
        "feature_coverage": 0.30,
        "full_profile_share": 0.05,
        "dealbreakers_known_share": 0.70,
        "onboarding_reach": 0.40,
        "match_feedback_reach": 0.20,
        "post_date_reach": 0.05,
        "text_signal_quality": 0.40,
    },
    reasons={
        "yes_rate": (
            "The defining change: one in ten eligible strangers, a third of "
            "the introductions figure. A feed "
            "invites selectivity because there is always another profile. "
            "Assumption."
        ),
        "decision_noise": "Decisions made in seconds are less consistent. Assumption.",
        "desirability_consensus": (
            "Photo-first judgement leans more on general appeal than on fit. Assumption."
        ),
        "open_rate": "A typical person opens the app six weeks in ten. Assumption.",
        "fatigue": "A feed of misses wears people down faster. Assumption.",
        "intent_openness": (
            "A profile field invites the safe answer more than a conversation does. Assumption."
        ),
        "voice_signal_quality": "A swipe feed has no voice.",
        "family_visibility": "A swipe feed does not ask about family.",
        "feature_coverage": (
            "Source for the figure: the article's survey completion rate, the "
            "nearest thing it reports to a form-based profile."
        ),
        "full_profile_share": "Few people complete an optional form. Assumption.",
        "dealbreakers_known_share": (
            "Some dealbreaker questions are optional and skipped. Assumption."
        ),
        "onboarding_reach": "A short signup says little about what matters. Assumption.",
        "match_feedback_reach": (
            "A swipe carries no stated reason, so behaviour reveals less. Assumption."
        ),
        "post_date_reach": "Nobody is asked how the date went. Assumption.",
        "text_signal_quality": "Short bios say less than a conversation. Assumption.",
    },
)

DYSTOPIA = Preset(
    name="dystopia",
    product=INTRODUCTIONS,
    summary="Lopsided market, attention piled onto a few, thin profiles, low honesty.",
    description=(
        "Today's introductions product after everything has gone wrong. "
        "Three people are seeking for every one sought. Most of what "
        "attracts people is the same for everyone, so attention piles onto a "
        "few. People on the crowded side are as willing as today's, but the "
        "scarce side accepts far less often and yes-rates vary wildly, with "
        "some people accepting almost nobody. People open the app barely one "
        "week in two, tire quickly and give up early; most profiles are "
        "nearly empty, what people say they "
        "want is a poor guide to what they respond to, and almost nobody "
        "stays long enough for behaviour to correct it. What people want is "
        "left exactly as in today's world, so any difference comes from the "
        "market and the data, not from intentions."
    ),
    overrides={
        "majority_share": 0.75,
        "minority_yes_ratio": 0.50,
        "yes_rate_spread": 1.00,
        "desirability_consensus": 0.60,
        "decision_noise": 0.20,
        "open_rate": 0.55,
        "fatigue": 0.20,
        "patience": 4.0,
        "intent_openness": 0.60,
        "self_knowledge": 0.35,
        "onboarding_reach": 0.50,
        "match_feedback_reach": 0.15,
        "post_date_reach": 0.03,
        "feature_coverage": 0.30,
        "full_profile_share": 0.05,
        "dealbreakers_known_share": 0.75,
        "extraction_accuracy": 0.75,
        "text_signal_quality": 0.30,
        "photo_signal_quality": 0.30,
    },
    reasons={
        "majority_share": "Three seeking for every one sought. Assumption.",
        "minority_yes_ratio": (
            "The scarce side accepts half as often as the crowded side. Assumption."
        ),
        "yes_rate_spread": "A long tail: some accept almost nobody. Assumption.",
        "desirability_consensus": (
            "Most of what can be explained about attraction is general appeal, "
            "so attention concentrates on a small share of people. Assumption."
        ),
        "decision_noise": "Tired, distrustful users decide less consistently. Assumption.",
        "open_rate": "A typical person opens the app barely one week in two. Assumption.",
        "fatigue": "Each letdown costs twice the willingness it does today. Assumption.",
        "patience": "People give up after about four letdowns, not eight. Assumption.",
        "intent_openness": "Four in ten say what they think will be accepted. Assumption.",
        "self_knowledge": (
            "Barely better than guessing among three levels: stated preferences "
            "are aspirational or performed. Assumption."
        ),
        "onboarding_reach": "A short signup most people rush. Assumption.",
        "match_feedback_reach": "Few people stay long enough to give feedback. Assumption.",
        "post_date_reach": "Dates are rare, so post-date feedback hardly exists. Assumption.",
        "feature_coverage": (
            "Source for the figure: the article's survey completion rate, used "
            "here as what a thin, form-based profile looks like."
        ),
        "full_profile_share": "Complete profiles are rare. Assumption.",
        "dealbreakers_known_share": "One in four never answers all four questions. Assumption.",
        "extraction_accuracy": "Sparse text is misread more often. Assumption.",
        "text_signal_quality": "One-line bios say little. Assumption.",
        "photo_signal_quality": "Few, heavily curated photos say little. Assumption.",
    },
)

#: All presets by name, in the order they are listed.
PRESETS: Mapping[str, Preset] = MappingProxyType(
    {preset.name: preset for preset in (TODAY, CASUAL, SERIOUS, SWIPE, DYSTOPIA)}
)


def preset(name: str) -> Preset:
    """Look up a preset by name."""
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; choose from {list(PRESETS)}") from None


def load_scenario_file(path: str | Path) -> Scenario:
    """Load a TOML or JSON scenario file.

    The optional top-level key ``base`` names the preset to start from
    (default ``today``); every other key overrides one field of it. A file
    with a ``fields`` table is a preset file; see ``load_preset_file``.
    """
    data = dict(read_scenario_file(path))
    if "fields" in data:
        return load_preset_file(path).scenario
    base = preset(str(data.pop("base", "today"))).scenario
    return from_dict(data, base=base)


_PRESET_FILE_KEYS = {"name", "product", "summary", "description", "base", "fields"}


def load_preset_file(path: str | Path) -> Preset:
    """Load a preset whose every number carries its own citation.

    This is the format for worlds built from sourced figures, such as a
    particular country. Each changed field gives a ``value`` and a ``basis``
    saying where the value comes from; a field without a basis is refused::

        name = "somewhere"
        product = "introductions"
        summary = "One line."
        description = "A paragraph."
        base = "today"

        [fields.endogamy_share]
        value = 0.4
        basis = "Source: author, title, year, table, URL."

        [fields.community_shares]
        value = { first = 0.8, second = 0.2 }
        basis = "Source: ..."

    No such file ships with this repository.
    """
    path = Path(path)
    data = dict(read_scenario_file(path))
    unknown = sorted(set(data) - _PRESET_FILE_KEYS)
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) {unknown}; a preset file has {sorted(_PRESET_FILE_KEYS)}"
        )
    fields = data.get("fields", {})
    if not isinstance(fields, Mapping):
        raise ValueError(f"{path}: 'fields' must be a table of field name to value and basis")
    overrides: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    for name, entry in fields.items():
        if not isinstance(entry, Mapping) or set(entry) != {"value", "basis"}:
            raise ValueError(f"{path}: fields.{name} needs exactly 'value' and 'basis'")
        basis = str(entry["basis"]).strip()
        if not basis:
            raise ValueError(f"{path}: fields.{name} has no basis; say where the value comes from")
        overrides[name], reasons[name] = entry["value"], basis
    loaded = Preset(
        name=str(data.get("name", path.stem)),
        product=str(data.get("product", INTRODUCTIONS)),
        summary=str(data.get("summary", "")),
        description=str(data.get("description", "")),
        overrides=overrides,
        reasons=reasons,
        base=preset(str(data.get("base", "today"))).scenario,
    )
    loaded.scenario  # build it now, so a bad value is reported when the file is loaded
    return loaded


def load_preset_directory(directory: str | Path) -> dict[str, Preset]:
    """Every preset file (``.toml`` or ``.json``) in a directory, by preset name."""
    found: dict[str, Preset] = {}
    for path in sorted(Path(directory).iterdir()):
        if path.suffix in (".toml", ".json"):
            loaded = load_preset_file(path)
            found[loaded.name] = loaded
    return found


def resolve_scenario(name_or_path: str) -> tuple[str, Scenario]:
    """Turn a preset name or a scenario file path into ``(label, scenario)``."""
    if name_or_path in PRESETS:
        return name_or_path, PRESETS[name_or_path].scenario
    path = Path(name_or_path)
    if path.suffix in (".toml", ".json"):
        if not path.is_file():
            raise ValueError(f"scenario file not found: {name_or_path}")
        return path.stem, load_scenario_file(path)
    raise ValueError(
        f"{name_or_path!r} is neither a preset ({', '.join(PRESETS)}) nor a .toml/.json file"
    )
