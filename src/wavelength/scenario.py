"""A scenario: the assumptions about a world, as named, arguable numbers.

A simulation is only as good as its premises, so the premises should be the
part that is easiest to read and to change. Every field of ``Scenario`` is a
quantity about people or about a dating market that someone could dispute
without reading any code: what share of the pool wants something casual, how
many people a typical person says yes to, how often what people say they
want matches what they respond to.

Each field carries three pieces of documentation in its metadata, which
``wavelength describe`` prints:

- ``meaning``   what the number is, in the world
- ``sensible``  the range worth trying, and what the ends look like
- ``basis``     where the default comes from. It starts with "assumption"
                when it is a judgement call and with "source" when a public
                source informs it. Most are assumptions, and say so.

The defaults describe the ``today`` preset; the other presets in
``presets.py`` are small sets of overrides on top of it.
"""

from __future__ import annotations

from dataclasses import Field, dataclass, field, fields
from types import MappingProxyType
from typing import Any, Mapping

from wavelength.features import FEATURE_NAMES, FEATURES, Group
from wavelength.filters import HARD_FILTERS

#: The four features the matcher hard-filters on. A scenario says how the
#: population is spread over their values.
DEALBREAKER_FEATURES: tuple[str, ...] = tuple(HARD_FILTERS)

#: Every other feature. Their population shares live in ``Scenario.trait_shares``.
TRAIT_FEATURES: tuple[str, ...] = tuple(n for n in FEATURE_NAMES if n not in HARD_FILTERS)

#: The article's "core values" group: the features people most often refuse
#: to compromise on.
CORE_FEATURES: tuple[str, ...] = tuple(
    n for n in FEATURE_NAMES if FEATURES[n].group is Group.CORE_VALUES
)

#: What observable attraction is made of; see ``Scenario.attraction_mix``.
ATTRACTION_PARTS: tuple[str, ...] = ("values", "vibe", "lifestyle", "manner")

_TOLERANCE = 1e-6


def _distribution(
    where: str, shares: Mapping[str, float], allowed: tuple[str, ...]
) -> Mapping[str, float]:
    """Check a set of shares and return it complete, in a fixed order, read-only.

    Values left out are taken as zero, so a distribution can be written by
    naming only the values that occur.
    """
    unknown = sorted(set(shares) - set(allowed))
    if unknown:
        raise ValueError(f"{where}: unknown value(s) {unknown}; expected some of {list(allowed)}")
    full = {value: float(shares.get(value, 0.0)) for value in allowed}
    if any(share < 0.0 for share in full.values()):
        raise ValueError(f"{where}: shares cannot be negative")
    total = sum(full.values())
    if abs(total - 1.0) > _TOLERANCE:
        raise ValueError(f"{where}: shares must sum to 1, got {total:.6g}")
    return MappingProxyType(full)


def _share(where: str, value: float, low: float = 0.0, high: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a number, got {value!r}")
    if not low <= value <= high:
        raise ValueError(f"{where} must be between {low:g} and {high:g}, got {value:g}")
    return float(value)


@dataclass(frozen=True)
class Archetype:
    """A kind of person, defined by what they tend to want.

    Archetypes are how a scenario makes dealbreakers correlated. Drawing
    each dealbreaker independently would produce as many marriage-minded
    people who never want children as the two shares multiply out to; real
    populations are lumpier than that. So a person first belongs to an
    archetype, and the archetype sets the odds for each of the four
    dealbreaker features. Within an archetype the four are drawn
    independently.

    ``share`` is the archetype's share of the pool. Each of the four
    mappings gives the share of the archetype's members holding each value;
    values left out are zero.
    """

    name: str
    description: str
    share: float
    relationship_timeline: Mapping[str, float]
    commitment_style: Mapping[str, float]
    family_plans: Mapping[str, float]
    trust_boundaries: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("an archetype needs a name")
        object.__setattr__(self, "share", _share(f"archetype {self.name}: share", self.share))
        for feature in DEALBREAKER_FEATURES:
            checked = _distribution(
                f"archetype {self.name}: {feature}",
                getattr(self, feature),
                FEATURES[feature].values,
            )
            object.__setattr__(self, feature, checked)

    def tendency(self, feature: str) -> Mapping[str, float]:
        """This archetype's distribution over the values of one dealbreaker feature."""
        if feature not in DEALBREAKER_FEATURES:
            raise KeyError(f"{feature!r} is not a dealbreaker feature")
        return getattr(self, feature)


def _knob(
    default: Any,
    *,
    group: str,
    meaning: str,
    sensible: str,
    basis: str,
    low: float = 0.0,
    high: float = 1.0,
) -> Any:
    """A scenario field with its real-world documentation attached.

    ``low`` and ``high`` are the hard limits for a number: construction
    rejects anything outside them, and a user interface can build a slider
    from them. ``sensible`` is the narrower range worth trying, in words.
    """
    metadata = {
        "group": group,
        "meaning": meaning,
        "sensible": sensible,
        "basis": basis,
        "low": low,
        "high": high,
    }
    if isinstance(default, (dict, tuple)):
        return field(default_factory=lambda: default, metadata=metadata)
    return field(default=default, metadata=metadata)


_MARKET = "The market"
_WANTS = "What people want"
_CHOICE = "How people choose"
_KNOWN = "What the app knows"
_CULTURE = "Culture"
_ATTENTION = "Attention over time"
_LATER = "After the yes"

_DEFAULT_ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        name="settling_down",
        description="Wants marriage and, mostly, children; expects exclusivity.",
        share=0.20,
        relationship_timeline={"marriage_focused": 0.70, "long_term": 0.30},
        commitment_style={"requires_exclusivity": 0.60, "prefers_monogamy": 0.35, "flexible": 0.05},
        family_plans={
            "wants_children_soon": 0.55,
            "wants_children_someday": 0.35,
            "undecided": 0.08,
            "childfree": 0.02,
        },
        trust_boundaries={"needs_security": 0.50, "balanced": 0.45, "very_flexible": 0.05},
    ),
    Archetype(
        name="relationship_seeking",
        description="Wants a lasting relationship without a deadline on it.",
        share=0.35,
        relationship_timeline={"marriage_focused": 0.10, "long_term": 0.70, "exploring": 0.20},
        commitment_style={
            "requires_exclusivity": 0.35,
            "prefers_monogamy": 0.50,
            "flexible": 0.12,
            "open_relationship": 0.03,
        },
        family_plans={
            "wants_children_soon": 0.10,
            "wants_children_someday": 0.50,
            "undecided": 0.30,
            "childfree": 0.10,
        },
        trust_boundaries={"needs_security": 0.35, "balanced": 0.55, "very_flexible": 0.10},
    ),
    Archetype(
        name="exploring",
        description="Open to something serious, not looking for it; seeing what happens.",
        share=0.30,
        relationship_timeline={"long_term": 0.15, "exploring": 0.65, "casual": 0.20},
        commitment_style={
            "requires_exclusivity": 0.10,
            "prefers_monogamy": 0.45,
            "flexible": 0.35,
            "open_relationship": 0.10,
        },
        family_plans={
            "wants_children_soon": 0.02,
            "wants_children_someday": 0.33,
            "undecided": 0.45,
            "childfree": 0.20,
        },
        trust_boundaries={"needs_security": 0.20, "balanced": 0.55, "very_flexible": 0.25},
    ),
    Archetype(
        name="casual",
        description="Wants dates and company, not a shared future.",
        share=0.15,
        relationship_timeline={"exploring": 0.25, "casual": 0.75},
        commitment_style={
            "requires_exclusivity": 0.03,
            "prefers_monogamy": 0.22,
            "flexible": 0.45,
            "open_relationship": 0.30,
        },
        family_plans={"wants_children_someday": 0.15, "undecided": 0.45, "childfree": 0.40},
        trust_boundaries={"needs_security": 0.10, "balanced": 0.45, "very_flexible": 0.45},
    ),
)

_DEFAULT_TRAIT_SHARES: dict[str, dict[str, float]] = {
    "emotional_availability": {"open": 0.50, "guarded": 0.40, "unavailable": 0.10},
    "intellectual_connection": {"high": 0.30, "moderate": 0.50, "low": 0.20},
    "emotional_expressiveness": {"expressive": 0.30, "measured": 0.45, "reserved": 0.25},
    "adventure_tolerance": {"thrill_seeking": 0.20, "moderate": 0.50, "comfort_seeking": 0.30},
    "social_energy": {"extrovert": 0.30, "ambivert": 0.40, "introvert": 0.30},
    "conflict_resolution": {"direct": 0.30, "time_first": 0.40, "avoidance": 0.30},
    "professional_ambition": {"career_first": 0.30, "balanced": 0.50, "life_first": 0.20},
    "travel_frequency": {"frequent": 0.20, "occasional": 0.55, "rare": 0.25},
}

_DEFAULT_DEALBREAKER_SHARES: dict[str, float] = {
    "relationship_timeline": 0.55,
    "commitment_style": 0.50,
    "family_plans": 0.45,
    "trust_boundaries": 0.30,
    "emotional_availability": 0.30,
}

_DEFAULT_ATTRACTION_MIX: dict[str, float] = {
    "values": 0.40,
    "vibe": 0.25,
    "lifestyle": 0.15,
    "manner": 0.20,
}

_DEFAULT_COMMUNITY_SHARES: dict[str, float] = {"first": 0.60, "second": 0.30, "third": 0.10}


@dataclass(frozen=True)
class Scenario:
    """Every assumption about one simulated world.

    Change a field with ``dataclasses.replace(scenario, yes_rate=0.2)``,
    with a scenario file, or with ``--set`` on the command line; see
    ``scenario_io``. Construction validates everything, so an impossible
    world (shares that do not sum to 1, a probability above 1) is an error
    at the point it is written down.
    """

    # --- The market -------------------------------------------------------

    pool_size: int = _knob(
        1200,
        low=20,
        high=20000,
        group=_MARKET,
        meaning=(
            "People in one matching pool: everyone who could in principle be "
            "introduced to each other, such as one city's users in one age band."
        ),
        sensible=(
            "300 (a small town or a new app) to 3000. Memory and time grow with "
            "its square. Below a few hundred, a person's top 10 is most of the "
            "people they are eligible for, and 'near the top of the list' stops "
            "meaning much; `wavelength simulate` prints the eligible-pool size."
        ),
        basis=(
            "assumption. Large enough that a typical person has several dozen "
            "eligible candidates, small enough that a run takes seconds. The "
            "article's app had 100K+ users across many such pools."
        ),
    )
    majority_share: float = _knob(
        0.60,
        low=0.5,
        high=0.95,
        group=_MARKET,
        meaning=(
            "The pool has two sides and people are only introduced across them. "
            "This is the share of the pool on the larger side; the rest are the "
            "people the larger side is competing for."
        ),
        sensible="0.50 (balanced) to 0.85 (about six seeking every one sought).",
        basis=(
            "assumption. Mainstream apps are widely described as having more men "
            "than women; no specific figure is taken from a source."
        ),
    )
    # --- What people want -------------------------------------------------

    archetypes: tuple[Archetype, ...] = _knob(
        _DEFAULT_ARCHETYPES,
        group=_WANTS,
        meaning=(
            "The mix of what people want. Each archetype is a kind of person "
            "with a share of the pool and its own odds for each dealbreaker "
            "feature (relationship timeline, commitment style, family plans, "
            "trust boundaries). The population share of any single value, such "
            "as 'casual', follows from the archetype shares and tendencies."
        ),
        sensible=(
            "Any number of archetypes whose shares sum to 1. Move share between "
            "them to make a pool more casual or more marriage-minded."
        ),
        basis=(
            "assumption. Four archetypes with roughly as much of the pool "
            "leaning towards a lasting relationship as away from one. The "
            "tendencies encode ordinary correlations (people who want marriage "
            "more often want children and exclusivity); the exact shares are "
            "not taken from a survey."
        ),
    )
    trait_shares: Mapping[str, Mapping[str, float]] = _knob(
        _DEFAULT_TRAIT_SHARES,
        group=_WANTS,
        meaning=(
            "Population share of each value of the eight features that are not "
            "hard-filtered: personality, lifestyle and emotional availability."
        ),
        sensible="Any shares summing to 1 per feature.",
        basis="assumption. The middle value of each trait is the most common.",
    )
    dealbreaker_shares: Mapping[str, float] = _knob(
        _DEFAULT_DEALBREAKER_SHARES,
        group=_WANTS,
        meaning=(
            "For each core-values feature, the share of people for whom it is a "
            "true dealbreaker: they refuse anyone clearly different on it, "
            "whatever else is right. For everyone else it is a preference that "
            "can be outweighed."
        ),
        sensible="0.10 (almost everything is negotiable) to 0.90 (almost nothing is).",
        basis=(
            "assumption. The order follows the article, which found relationship "
            "timeline the most consistently predictive of rejection; the sizes "
            "are judgement calls."
        ),
    )
    personality_dealbreaker_share: float = _knob(
        0.10,
        group=_WANTS,
        meaning=(
            "Share of people who treat any one personality trait (for example "
            "intellectual connection) as a dealbreaker. This is the article's "
            "'if there is no intellectual spark, I am out' user."
        ),
        sensible="0.0 to 0.3.",
        basis="assumption.",
    )
    lifestyle_dealbreaker_share: float = _knob(
        0.03,
        group=_WANTS,
        meaning="Share of people who treat any one lifestyle feature as a dealbreaker.",
        sensible="0.0 to 0.15.",
        basis="assumption. The article calls these nice-to-haves.",
    )

    # --- How people choose ------------------------------------------------

    yes_rate: float = _knob(
        0.30,
        low=0.01,
        high=0.95,
        group=_CHOICE,
        meaning=(
            "Share of eligible, randomly chosen people on the other side that a "
            "typical person would say yes to if introduced. 'Eligible' means "
            "clearing the four hard filters; 'randomly chosen' means before any "
            "matcher has picked well. It describes the person and the product, "
            "not the matcher: someone shown better-than-random candidates says "
            "yes more often than this."
        ),
        sensible=(
            "0.03 to 0.6. It depends on the product: a person swiping a long "
            "feed says yes to few; a person sent a handful of considered "
            "introductions, each with a real person waiting, says yes to many more."
        ),
        basis=(
            "assumption, for an introductions product: about three in ten "
            "eligible strangers. It is not a measurement and was not derived "
            "from any reported acceptance rate. Set it from your own data if "
            "you have any; see the `swipe` preset for the feed case."
        ),
    )
    yes_rate_spread: float = _knob(
        0.50,
        high=2.0,
        group=_CHOICE,
        meaning=(
            "How much people differ in their own yes-rate. 0 means everyone has "
            "the same; at 0.5 most people are within about 0.6x to 1.6x of "
            "typical; at 1.0 there is a long tail of people who accept almost "
            "nobody and a few who accept many."
        ),
        sensible="0.0 to 1.2.",
        basis="assumption.",
    )
    minority_yes_ratio: float = _knob(
        0.70,
        low=0.01,
        high=2.0,
        group=_CHOICE,
        meaning=(
            "How often people on the scarce side say yes, relative to people on "
            "the larger side. 1.0 means both sides have the same yes-rate; 0.5 "
            "means the scarce side accepts half as often."
        ),
        sensible="0.2 (the scarce side can afford to refuse nearly everyone) to 1.0.",
        basis=(
            "direction from a source, size an assumption. On the site studied by "
            "Hitsch, Hortacsu and Ariely (2010), men were more likely than women "
            "to contact a profile they had browsed, so the two sides of a market "
            "need not be equally picky. That the choosier side is also the "
            "scarce side is this scenario's assumption."
        ),
    )
    desirability_consensus: float = _knob(
        0.25,
        group=_CHOICE,
        meaning=(
            "How much people agree on who is desirable. At 0 attraction is all "
            "personal fit and everyone has a different favourite; at 1 everyone "
            "ranks the other side the same way and attention piles onto a few "
            "people. Formally: the share of non-chemistry attraction that is a "
            "person's general appeal rather than fit with the viewer. The "
            "matchers in this repository have no signal for general appeal."
        ),
        sensible="0.0 to 0.8.",
        basis=(
            "assumption for the size. That preferences have a shared component "
            "alongside a similarity component is a finding of Hitsch, Hortacsu "
            "and Ariely (2010)."
        ),
    )
    chemistry_share: float = _knob(
        0.35,
        group=_CHOICE,
        meaning=(
            "Share of attraction that no system can observe: the part that is "
            "particular to two people and shows up in no profile, photo or "
            "stated preference. It sets a ceiling on any matcher."
        ),
        sensible="0.1 (people are predictable from their profiles) to 0.7 (mostly spark).",
        basis="assumption.",
    )
    chemistry_mutuality: float = _knob(
        0.50,
        group=_CHOICE,
        meaning=(
            "How much of chemistry is felt by both people. At 0 each person's "
            "spark for the other is their own and the two decisions are "
            "unrelated beyond what profiles explain; at 1 chemistry is fully "
            "shared, so when one says yes the other is more likely to as well. "
            "This is what makes the two decisions in a pair not independent."
        ),
        sensible="0.0 to 1.0.",
        basis=(
            "assumption. The article notes that real decisions are correlated and gives no size."
        ),
    )
    decision_noise: float = _knob(
        0.10,
        group=_CHOICE,
        meaning=(
            "Mood on the day: the share of a single yes/no decision that has "
            "nothing to do with the other person. The same introduction on a "
            "different day could go the other way. Unlike chemistry it is "
            "never shared between the two people."
        ),
        sensible="0.0 (people are perfectly consistent) to 0.4 (snap judgements).",
        basis="assumption.",
    )
    attraction_mix: Mapping[str, float] = _knob(
        _DEFAULT_ATTRACTION_MIX,
        group=_CHOICE,
        meaning=(
            "What the explainable part of personal fit is made of: 'values' "
            "(agreement on the structured features, weighted by what the person "
            "cares about), 'vibe' (what two people say and how alike it is), "
            "'lifestyle' (how alike their daily lives are) and 'manner' (how "
            "they talk: pace, warmth, humour). Approximate shares of variance. "
            "Each part is a separate hidden trait, so no one signal carries "
            "the others."
        ),
        sensible="Any four shares summing to 1.",
        basis=(
            "assumption. Deliberately not the article's 0.60 / 0.25 / 0.15 "
            "scoring mix, which was fitted on its own production data."
        ),
    )
    preference_diversity: float = _knob(
        0.70,
        group=_CHOICE,
        meaning=(
            "How much people differ in what matters to them. At 0 everyone "
            "weighs the features alike, so one set of weights describes "
            "everybody. At 1 each person's attention is concentrated on their "
            "own few features, so one person's dealbreaker is another's shrug "
            "and a single global weighting misreads most people. It governs "
            "graded preference; who refuses outright is set by the dealbreaker "
            "shares."
        ),
        sensible="0.0 to 1.0.",
        basis=(
            "assumption. Set high because the article reports that the same "
            "feature was decisive for some users and irrelevant to others, "
            "across most of its features; it gives no measure."
        ),
    )

    # --- After the yes ----------------------------------------------------

    conversation_rate: float = _knob(
        0.50,
        low=0.01,
        group=_LATER,
        meaning=(
            "Of eligible, randomly chosen pairs, the share whose conversation "
            "would keep going if both said yes. Which pairs these are is not "
            "random: it follows from how alike they sound, how well their ways "
            "of talking fit, and whether either will make the first move."
        ),
        sensible="0.2 to 0.9.",
        basis="assumption.",
    )
    date_rate: float = _knob(
        0.40,
        low=0.01,
        group=_LATER,
        meaning=(
            "Of eligible, randomly chosen pairs, the share who would go on to "
            "meet if their conversation kept going. Which pairs follows from shared "
            "chemistry, how alike their lives are, and whether they live close "
            "enough."
        ),
        sensible="0.2 to 0.9.",
        basis="assumption.",
    )
    second_date_rate: float = _knob(
        0.40,
        low=0.01,
        group=_LATER,
        meaning=(
            "Of eligible, randomly chosen pairs, the share in which both would "
            "want to meet again if they met, before counting dealbreakers that "
            "only surface in person and family approval. Which pairs follows "
            "from what each feels for the other, chemistry included."
        ),
        sensible="0.1 to 0.8.",
        basis="assumption.",
    )
    leave_after_stage: int = _knob(
        4,
        low=1,
        high=4,
        group=_LATER,
        meaning=(
            "How far a couple must get before both stop taking introductions: "
            "1 both said yes, 2 the conversation keeps going, 3 they met, 4 "
            "both want to meet again."
        ),
        sensible="3 or 4. Lower values take people out of the pool on thin evidence.",
        basis="assumption.",
    )

    # --- Attention over time ----------------------------------------------

    open_rate: float = _knob(
        0.70,
        low=0.05,
        group=_ATTENTION,
        meaning=(
            "How often a typical person opens the app in a given week, and so "
            "answers what they were sent before it expires. An introduction "
            "that expires is not a no."
        ),
        sensible="0.3 (most weeks pass unopened) to 0.95 (a paid, high-intent service).",
        basis="assumption.",
    )
    open_rate_spread: float = _knob(
        0.30,
        group=_ATTENTION,
        meaning=(
            "How much people differ in that habit. At 0 everyone opens the app "
            "equally often. At 1 the pool splits into people who always answer "
            "and people who never do."
        ),
        sensible="0.0 to 1.0.",
        basis="assumption.",
    )
    fatigue: float = _knob(
        0.10,
        group=_ATTENTION,
        meaning=(
            "Share of a person's willingness to open the app that is lost with "
            "each letdown: an introduction they turned down, or one where they "
            "said yes and the other person did not. A matcher that wastes "
            "people's attention makes them harder to reach later."
        ),
        sensible="0.0 (people never tire) to 0.4.",
        basis="assumption.",
    )
    patience: float = _knob(
        8.0,
        low=1.0,
        high=50.0,
        group=_ATTENTION,
        meaning=(
            "How many letdowns a typical person takes before giving up on the "
            "app altogether. People differ around this number."
        ),
        sensible="2 to 15.",
        basis="assumption.",
    )

    # --- Culture ----------------------------------------------------------

    community_shares: Mapping[str, float] = _knob(
        _DEFAULT_COMMUNITY_SHARES,
        group=_CULTURE,
        meaning=(
            "Shares of the pool belonging to each community. A community stands "
            "for whatever people in this world marry within when they marry "
            "within anything: religion, language, caste, region. The names are "
            "placeholders."
        ),
        sensible="Any number of communities with shares summing to 1.",
        basis="assumption. Three placeholder communities; no real population is meant.",
    )
    endogamy_share: float = _knob(
        0.15,
        group=_CULTURE,
        meaning=(
            "Share of people for whom a partner from their own community is a "
            "dealbreaker. They say no to anyone outside it."
        ),
        sensible="0.0 to 0.9.",
        basis="assumption. Not taken from any survey.",
    )
    family_approval_share: float = _knob(
        0.10,
        group=_CULTURE,
        meaning=(
            "Share of people who need their family's approval before a "
            "relationship can continue. Their family judges a partner by how "
            "well the two families' expectations fit, and can end a match that "
            "the two people themselves were happy with."
        ),
        sensible="0.0 to 0.9.",
        basis="assumption. Not taken from any survey.",
    )
    family_visibility: float = _knob(
        0.30,
        group=_CULTURE,
        meaning=(
            "How well the app can see family expectations. 1.0 is a product "
            "that asks about family in detail; 0.0 is one that never asks."
        ),
        sensible="0.0 to 0.9.",
        basis="assumption.",
    )
    marriage_pressure: float = _knob(
        0.20,
        group=_CULTURE,
        meaning=(
            "How strongly age pushes people towards wanting marriage. At 0 what "
            "people want does not depend on age. At 1 nearly everyone in their "
            "mid-thirties has moved a step towards marriage from where their "
            "own inclination would put them."
        ),
        sensible="0.0 to 1.0.",
        basis="assumption. Not taken from any survey.",
    )
    intent_openness: float = _knob(
        0.85,
        group=_CULTURE,
        meaning=(
            "Share of people who state what they are looking for, and what they "
            "will not accept, as it really is. The rest give the answer most "
            "people give and keep a community requirement to themselves, so "
            "the mismatch only surfaces in person. This is the one place where "
            "what the app knows about a dealbreaker can be wrong."
        ),
        sensible="0.5 to 1.0.",
        basis="assumption.",
    )
    larger_side_initiates: float = _knob(
        0.80,
        group=_CULTURE,
        meaning=(
            "Share of people on the larger side who will make the first move "
            "once both have said yes. If neither person in a pair will, the "
            "conversation never starts."
        ),
        sensible="0.3 to 1.0.",
        basis="assumption.",
    )
    scarce_side_initiates: float = _knob(
        0.50,
        group=_CULTURE,
        meaning="The same for people on the scarce side.",
        sensible="0.05 to 1.0.",
        basis="assumption.",
    )
    distance_tolerance: float = _knob(
        0.50,
        low=0.05,
        high=2.0,
        group=_CULTURE,
        meaning=(
            "How far a typical person will travel to meet, as a share of the "
            "width of the area the pool covers. Two people meet only if the "
            "distance between them is within what they will travel between "
            "them. People differ around this number."
        ),
        sensible="0.1 (a neighbourhood) to 1.5 (anywhere in the area).",
        basis="assumption.",
    )
    photos_hidden_share: float = _knob(
        0.05,
        group=_CULTURE,
        meaning=(
            "Share of people who keep their photos hidden until later. For "
            "them the app has no photo signal at all."
        ),
        sensible="0.0 to 0.9.",
        basis="assumption.",
    )

    # --- What the app knows -----------------------------------------------

    self_knowledge: float = _knob(
        0.55,
        group=_KNOWN,
        meaning=(
            "How often the importance a person states at onboarding (dealbreaker, "
            "important, nice-to-have) is the importance they act on. Low values "
            "mean people misjudge or misreport what matters to them."
        ),
        sensible="0.33 (no better than a guess among three levels) to 0.9.",
        basis=(
            "assumption. The article says stated preferences often do not match "
            "what people respond to and gives no rate."
        ),
    )
    onboarding_reach: float = _knob(
        0.70,
        group=_KNOWN,
        meaning="Share of features whose importance a person said anything about at onboarding.",
        sensible="0.2 (a two-question signup) to 0.95 (a long guided conversation).",
        basis="assumption.",
    )
    match_feedback_reach: float = _knob(
        0.40,
        group=_KNOWN,
        meaning=(
            "Share of features whose importance a person reveals each time they "
            "explain a yes or a no. People arrive with one such explanation "
            "behind them and add another with every introduction they answer."
        ),
        sensible="0.0 (everyone is new) to 0.8.",
        basis="assumption.",
    )
    match_feedback_accuracy: float = _knob(
        0.80,
        group=_KNOWN,
        meaning="How often importance learned from match feedback is the true importance.",
        sensible="0.5 to 0.95. The article's premise is that it exceeds self_knowledge.",
        basis="assumption. Only the ordering (above stated preference) comes from the article.",
    )
    post_date_reach: float = _knob(
        0.15,
        group=_KNOWN,
        meaning=(
            "Share of features whose importance a person reveals each time they "
            "talk through a real date. People arrive with one such talk behind "
            "them and add another each time they meet someone."
        ),
        sensible="0.0 to 0.5.",
        basis="assumption.",
    )
    post_date_accuracy: float = _knob(
        0.95,
        group=_KNOWN,
        meaning="How often importance learned from post-date feedback is the true importance.",
        sensible="0.6 to 1.0. The article's premise is that it is the best signal there is.",
        basis="assumption. Only the ordering (above match feedback) comes from the article.",
    )
    feature_coverage: float = _knob(
        0.85,
        group=_KNOWN,
        meaning=(
            "Profile completeness: across all users, the share of non-dealbreaker "
            "features the app has extracted a value for."
        ),
        sensible=(
            "0.30 to 0.95. Documented examples from the article: 0.85 for "
            "extraction from conversation, 0.30 for survey completion."
        ),
        basis="source: the article's reported coverage from conversational extraction.",
    )
    full_profile_share: float = _knob(
        0.34,
        group=_KNOWN,
        meaning=(
            "Share of users with every feature extracted. The remaining users "
            "have gaps, sized so that overall coverage equals feature_coverage."
        ),
        sensible="0.0 to feature_coverage (it cannot exceed it).",
        basis="source: the article reports about 34% of users with full feature extraction.",
    )
    dealbreakers_known_share: float = _knob(
        0.90,
        group=_KNOWN,
        meaning=(
            "Share of users for whom all four hard-filter values are known. The "
            "filters fail closed, so everyone else cannot be matched by a "
            "filtering strategy until they answer."
        ),
        sensible="full_profile_share (it cannot be lower) to 1.0.",
        basis="assumption. High because an app can ask these four questions directly.",
    )
    extraction_accuracy: float = _knob(
        0.90,
        group=_KNOWN,
        meaning=(
            "How often an extracted non-dealbreaker value is the person's true "
            "value. Dealbreaker values are modelled as either known correctly or "
            "not known."
        ),
        sensible="0.6 to 1.0.",
        basis="assumption.",
    )
    text_signal_quality: float = _knob(
        0.60,
        group=_KNOWN,
        meaning=(
            "How well the text signal reflects how a person really comes across. "
            "1.0 is a perfect read of their vibe; 0.0 is noise. In a product "
            "built on long onboarding conversations the text is a transcript, "
            "which says far more than a written bio."
        ),
        sensible="0.2 (one-line bios) to 0.9 (a long conversation, well summarised).",
        basis="assumption.",
    )
    photo_signal_quality: float = _knob(
        0.50,
        group=_KNOWN,
        meaning=(
            "How well the photo signal reflects how a person really lives. 1.0 is "
            "a perfect read of their lifestyle; 0.0 is noise."
        ),
        sensible="0.1 (one curated headshot) to 0.9.",
        basis=(
            "assumption. That photos carry lifestyle signal at all is the "
            "article's point, citing Bai et al. (2025), 'Beauty vs. Vibe'; no "
            "number is taken from that study."
        ),
    )
    voice_signal_quality: float = _knob(
        0.50,
        group=_KNOWN,
        meaning=(
            "How well the voice signal reflects how a person really talks: "
            "pace, warmth, humour, the things a transcript loses. 1.0 is a "
            "perfect read; 0.0 is noise, or a product with no voice at all."
        ),
        sensible="0.0 (no voice in the product) to 0.9.",
        basis="assumption.",
    )
    intent_visible_in_bio: float = _knob(
        0.20,
        group=_KNOWN,
        meaning=(
            "How much the way people describe themselves gives away what they "
            "want. At 0, two people can sound identical and want opposite lives; "
            "at 1, sounding alike means wanting the same things."
        ),
        sensible="0.0 to 0.8.",
        basis=(
            "assumption. Low, following the article's account of similar-sounding "
            "profiles with opposite dealbreakers. This is a premise, not a result."
        ),
    )

    def __post_init__(self) -> None:
        if isinstance(self.pool_size, bool) or not isinstance(self.pool_size, int):
            raise ValueError(f"pool_size must be a whole number, got {self.pool_size!r}")
        if not 20 <= self.pool_size <= 20000:
            raise ValueError(f"pool_size must be between 20 and 20000, got {self.pool_size}")
        self._check_archetypes()
        self._check_mappings()
        for f in fields(self):
            if f.type == "float":
                value = _share(f.name, getattr(self, f.name), f.metadata["low"], f.metadata["high"])
                object.__setattr__(self, f.name, value)
        if isinstance(self.leave_after_stage, bool) or self.leave_after_stage not in (1, 2, 3, 4):
            raise ValueError(
                f"leave_after_stage must be 1, 2, 3 or 4, got {self.leave_after_stage!r}"
            )
        if self.full_profile_share > self.feature_coverage:
            raise ValueError("full_profile_share cannot exceed feature_coverage")
        if self.dealbreakers_known_share < self.full_profile_share:
            raise ValueError("dealbreakers_known_share cannot be below full_profile_share")

    def _check_archetypes(self) -> None:
        archetypes = tuple(self.archetypes)
        if not archetypes or not all(isinstance(a, Archetype) for a in archetypes):
            raise ValueError("archetypes must be a non-empty sequence of Archetype")
        names = [a.name for a in archetypes]
        if len(names) != len(set(names)):
            raise ValueError(f"archetype names must be unique, got {names}")
        total = sum(a.share for a in archetypes)
        if abs(total - 1.0) > _TOLERANCE:
            raise ValueError(f"archetypes: shares must sum to 1, got {total:.6g}")
        object.__setattr__(self, "archetypes", archetypes)

    def _check_mappings(self) -> None:
        if set(self.trait_shares) != set(TRAIT_FEATURES):
            raise ValueError(f"trait_shares needs exactly these features: {list(TRAIT_FEATURES)}")
        traits = {
            name: _distribution(
                f"trait_shares.{name}", self.trait_shares[name], FEATURES[name].values
            )
            for name in TRAIT_FEATURES
        }
        object.__setattr__(self, "trait_shares", MappingProxyType(traits))

        if set(self.dealbreaker_shares) != set(CORE_FEATURES):
            raise ValueError(
                f"dealbreaker_shares needs exactly these features: {list(CORE_FEATURES)}"
            )
        firm = {
            name: _share(f"dealbreaker_shares.{name}", self.dealbreaker_shares[name])
            for name in CORE_FEATURES
        }
        object.__setattr__(self, "dealbreaker_shares", MappingProxyType(firm))

        if set(self.attraction_mix) != set(ATTRACTION_PARTS):
            raise ValueError(f"attraction_mix needs exactly these parts: {list(ATTRACTION_PARTS)}")
        mix = _distribution("attraction_mix", self.attraction_mix, ATTRACTION_PARTS)
        object.__setattr__(self, "attraction_mix", mix)

        if not self.community_shares:
            raise ValueError("community_shares needs at least one community")
        communities = _distribution(
            "community_shares", self.community_shares, tuple(self.community_shares)
        )
        object.__setattr__(self, "community_shares", communities)

    def archetype(self, name: str) -> Archetype:
        """The archetype with this name."""
        for candidate in self.archetypes:
            if candidate.name == name:
                return candidate
        raise KeyError(f"no archetype named {name!r}; have {[a.name for a in self.archetypes]}")

    def value_shares(self, feature: str) -> dict[str, float]:
        """Population share of each value of a feature, implied by the scenario.

        For a dealbreaker feature this is the archetype tendencies averaged
        by archetype share; for any other feature it is ``trait_shares``.
        """
        if feature in DEALBREAKER_FEATURES:
            return {
                value: sum(a.share * a.tendency(feature)[value] for a in self.archetypes)
                for value in FEATURES[feature].values
            }
        return dict(self.trait_shares[feature])


def scenario_fields() -> tuple[Field[Any], ...]:
    """The scenario's fields, in declaration order, with their documentation metadata."""
    return fields(Scenario)
