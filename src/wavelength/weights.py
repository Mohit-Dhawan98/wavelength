"""Per-user importance weights, with provenance.

Why per-user: the same feature is one person's dealbreaker and another
person's shrug. A single global weight is too low for the first and too high
for the second, so it is wrong for both, in opposite directions.

Why provenance: weights are learned from signals of very different quality.
What someone says at signup is a guess about themselves; what they say after
rejecting a match is evidence; what they say after a real date is the best
evidence there is. When signals disagree the better source has to win, and
it has to be possible to ask "why is this weight 9.0?" and get back the exact
signal that set it. So a profile never stores a bare number. It stores every
signal it has seen and derives the current weight from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Mapping

from wavelength.features import FEATURE_NAMES, FEATURES


class Tier(Enum):
    """How much a feature matters to one user."""

    ALPHA = "alpha"  # dealbreaker, non-negotiable
    BETA = "beta"  # important, strong preference
    GAMMA = "gamma"  # nice-to-have, flexible


#: Weight band per tier, from the article. The bands are where a weight lands
#: when a signal only tells us the tier. They are not enforced as validation:
#: the article's own worked example has a beta signal at 7.5 and an alpha
#: signal at 9.0, both outside these bands, so a signal may carry any weight
#: on the 0-10 scale alongside its tier label.
TIER_BANDS: Mapping[Tier, tuple[float, float]] = MappingProxyType(
    {Tier.ALPHA: (9.5, 10.0), Tier.BETA: (5.0, 7.0), Tier.GAMMA: (1.0, 3.0)}
)


class Source(IntEnum):
    """Where a signal came from. Higher value means higher priority.

    Ordered by how much real experience stands behind the signal:
    stated preference < decision about a real match < reflection on a real date.
    """

    ONBOARDING = 1
    MATCH_FEEDBACK = 2
    POST_DATE_FEEDBACK = 3


@dataclass(frozen=True)
class WeightSignal:
    """One piece of evidence about how much one feature matters to one user.

    ``observed_at`` is any ordinal that increases with time (a week number in
    the tests, a timestamp in a real system). ``evidence`` is the human-readable
    reason, kept so that an audit ends at something a person can read.
    """

    feature: str
    weight: float
    tier: Tier
    source: Source
    confidence: float = 1.0
    observed_at: int = 0
    evidence: str = ""

    def __post_init__(self) -> None:
        if self.feature not in FEATURES:
            raise ValueError(f"unknown feature {self.feature!r}")
        if not 0.0 < self.weight <= 10.0:
            raise ValueError(f"weight {self.weight} is outside (0, 10]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} is outside [0, 1]")

    @property
    def priority(self) -> tuple[int, float, int]:
        """Sort key for conflict resolution: source, then confidence, then recency."""
        return (int(self.source), self.confidence, self.observed_at)


#: Weights used when nothing is known about a user (cold start) and by the
#: non-personalised baseline. The article quotes three global weights
#: (intellectual connection 7.0, adventure tolerance 4.5, emotional
#: expressiveness 6.0); the rest are illustrative.
GLOBAL_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "relationship_timeline": 9.0,
        "commitment_style": 9.0,
        "family_plans": 9.0,
        "trust_boundaries": 8.0,
        "emotional_availability": 7.5,
        "intellectual_connection": 7.0,
        "emotional_expressiveness": 6.0,
        "adventure_tolerance": 4.5,
        "social_energy": 5.0,
        "conflict_resolution": 4.0,
        "professional_ambition": 4.0,
        "travel_frequency": 3.0,
    }
)
if set(GLOBAL_WEIGHTS) != set(FEATURE_NAMES):
    raise RuntimeError("GLOBAL_WEIGHTS must have exactly one weight per feature")


class WeightProfile:
    """One user's weights, derived from the full history of signals.

    The history is append-only. The current weight for a feature is a pure
    function of that history, so the order signals arrive in does not matter
    and nothing is lost when a stronger signal overrides a weaker one.
    """

    def __init__(self, defaults: Mapping[str, float] = GLOBAL_WEIGHTS) -> None:
        self._defaults = defaults
        self._history: dict[str, list[WeightSignal]] = {}

    def record(self, signal: WeightSignal) -> None:
        """Add a signal. Nothing is overwritten; resolution happens on read."""
        self._history.setdefault(signal.feature, []).append(signal)

    def history(self, feature: str) -> tuple[WeightSignal, ...]:
        """Every signal recorded for a feature, oldest first."""
        return tuple(sorted(self._history.get(feature, ()), key=lambda s: s.observed_at))

    def resolved(self, feature: str) -> WeightSignal | None:
        """The signal that currently sets this feature's weight, if any.

        Higher-priority source wins; within a source, higher confidence wins;
        within equal confidence, the most recent wins.
        """
        signals = self._history.get(feature)
        if not signals:
            return None
        return max(signals, key=lambda s: s.priority)

    def weight(self, feature: str) -> float:
        """Current weight: the winning signal's, or the global default at cold start."""
        winner = self.resolved(feature)
        return winner.weight if winner is not None else self._defaults[feature]

    def as_dict(self) -> dict[str, float]:
        """Current weight for every feature, in ``FEATURE_NAMES`` order."""
        return {name: self.weight(name) for name in FEATURE_NAMES}

    def explain(self, feature: str) -> str:
        """A readable audit trail for one feature's weight."""
        winner = self.resolved(feature)
        if winner is None:
            return f"{feature}: {self._defaults[feature]} (global default, no signals yet)"
        lines = [f"{feature}: {winner.weight} (set by {winner.source.name.lower()})"]
        for s in self.history(feature):
            marker = "->" if s is winner else "  "
            lines.append(
                f" {marker} t={s.observed_at} {s.source.name.lower()} "
                f"tier={s.tier.value} weight={s.weight} confidence={s.confidence} "
                f'"{s.evidence}"'
            )
        return "\n".join(lines)
