"""Named worlds: each preset is a short list of overrides and the reason for each.

``today`` is ``Scenario()`` with nothing changed; the reasons for its numbers
are the ``basis`` notes on the fields themselves. Every other preset changes
only the fields that make its world different, in the same plain-data form
as a scenario file, and gives one reason per changed field.

None of these numbers was tuned to produce any particular outcome, and in
particular none was tuned to reproduce the acceptance rates reported in the
article. They are positions to argue with. Unless a reason names a source,
it is an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from wavelength.scenario import Scenario
from wavelength.scenario_io import from_dict, read_scenario_file


@dataclass(frozen=True)
class Preset:
    """A named world.

    ``overrides`` is what differs from ``Scenario()``; ``reasons`` has one
    entry per overridden field saying why that number was chosen.
    """

    name: str
    summary: str
    description: str
    overrides: Mapping[str, Any]
    reasons: Mapping[str, str]

    def __post_init__(self) -> None:
        if set(self.overrides) != set(self.reasons):
            raise ValueError(f"preset {self.name}: every override needs exactly one reason")

    @property
    def scenario(self) -> Scenario:
        """The full scenario: the defaults with this preset's overrides applied."""
        return from_dict(self.overrides)


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
    summary="A mainstream dating app resembling now.",
    description=(
        "A general-purpose app in a large city. Intentions are mixed: a little "
        "over half the pool leans towards a lasting relationship and the rest "
        "are exploring or want something casual. There are somewhat more "
        "people on one side than the other. People are moderately picky, "
        "know themselves imperfectly, and about a third of profiles are "
        "complete. This preset is the defaults; each field's own 'basis' note "
        "says where its number comes from."
    ),
    overrides={},
    reasons={},
)

CASUAL = Preset(
    name="casual",
    summary="Most people want something casual or are exploring; few dealbreakers.",
    description=(
        "An app people open for dates, not for a partner. Four in five are "
        "exploring or want something casual, and because little is at stake "
        "in one evening, few people hold firm dealbreakers and most say yes "
        "more readily. What draws people is less about shared life plans and "
        "more about how someone comes across and broad appeal. The market "
        "itself (size, balance, engagement) is the same as today's, so any "
        "difference comes from what people want."
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
        "pickiness": 0.20,
        "attraction_mix": {"values": 0.30, "vibe": 0.40, "lifestyle": 0.30},
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
        "pickiness": "Twice today's yes-rate: a date costs less than a commitment. Assumption.",
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
    summary="Marriage-minded, many firm dealbreakers, high intent.",
    description=(
        "A matrimony-style service. Nine in ten want marriage or a lasting "
        "relationship and most hold firm dealbreakers on timeline, commitment "
        "and children. People are pickier because the decision is larger, but "
        "they arrive knowing what they want, answer the questions they are "
        "asked, and respond to introductions. The two sides are close to "
        "balanced."
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
        "pickiness": 0.06,
        "majority_share": 0.55,
        "active_share": 0.85,
        "self_knowledge": 0.65,
        "onboarding_reach": 0.90,
        "dealbreakers_known_share": 0.98,
        "full_profile_share": 0.50,
        "feature_coverage": 0.90,
        "attraction_mix": {"values": 0.65, "vibe": 0.20, "lifestyle": 0.15},
    },
    reasons={
        "archetypes": (
            "The defining change: 90% settling down or relationship-seeking. Assumption."
        ),
        "dealbreaker_shares": (
            "Most people here will not bend on timeline, commitment or children. Assumption."
        ),
        "pickiness": "A bit over half of today's yes-rate: the stakes are higher. Assumption.",
        "majority_share": "Closer to balanced than a mainstream app. Assumption.",
        "active_share": "People who sign up to find a spouse mostly answer. Assumption.",
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

DYSTOPIA = Preset(
    name="dystopia",
    summary="Lopsided market, attention piled onto a few, thin profiles, low honesty.",
    description=(
        "Today's app after everything has gone wrong. Four people are seeking "
        "for every one sought. Everyone wants the same few people, so those "
        "few can refuse nearly everybody, and pickiness varies wildly from "
        "person to person. Most profiles are abandoned or nearly empty, what "
        "people say they want is a poor guide to what they respond to, and "
        "almost nobody stays long enough for behaviour to correct it. What "
        "people want is left exactly as in today's world, so any difference "
        "comes from the market and the data, not from intentions."
    ),
    overrides={
        "majority_share": 0.80,
        "minority_yes_ratio": 0.25,
        "pickiness": 0.05,
        "pickiness_spread": 1.00,
        "desirability_consensus": 0.70,
        "active_share": 0.35,
        "self_knowledge": 0.35,
        "onboarding_reach": 0.50,
        "match_feedback_reach": 0.15,
        "post_date_reach": 0.03,
        "feature_coverage": 0.30,
        "full_profile_share": 0.05,
        "dealbreakers_known_share": 0.60,
        "extraction_accuracy": 0.75,
        "text_signal_quality": 0.30,
        "photo_signal_quality": 0.30,
    },
    reasons={
        "majority_share": "Four seeking for every one sought. Assumption.",
        "minority_yes_ratio": (
            "The scarce side accepts a quarter as often as the crowded side. Assumption."
        ),
        "pickiness": "Half of today's yes-rate even on the crowded side. Assumption.",
        "pickiness_spread": "A long tail: some accept almost nobody. Assumption.",
        "desirability_consensus": (
            "Most of attraction is general appeal, so attention concentrates on "
            "a small share of people. Assumption."
        ),
        "active_share": "Two in three profiles never answer. Assumption.",
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
        "dealbreakers_known_share": "Four in ten never answer all four questions. Assumption.",
        "extraction_accuracy": "Sparse text is misread more often. Assumption.",
        "text_signal_quality": "One-line bios say little. Assumption.",
        "photo_signal_quality": "Few, heavily curated photos say little. Assumption.",
    },
)

#: All presets by name, in the order they are listed.
PRESETS: Mapping[str, Preset] = MappingProxyType(
    {preset.name: preset for preset in (TODAY, CASUAL, SERIOUS, DYSTOPIA)}
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
    (default ``today``); every other key overrides one field of it.
    """
    data = dict(read_scenario_file(path))
    base = preset(str(data.pop("base", "today"))).scenario
    return from_dict(data, base=base)


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
