"""The answer key: how far every pair would get, fixed before any rule runs.

A matcher can only be graded against something it did not write. So for a
seeded world the engine first works out, from the hidden truth alone, what
would happen to every possible introduction if both people answered it:

    0  at least one says no
    1  both say yes
    2  the conversation keeps going
    3  they meet
    4  both want to meet again

Each stage needs the one before it, and each depends on different hidden
things, stated here so they can be argued with:

- **Yes** (``decisions.py``): what each feels on the day clears their own
  bar, and nothing in the other's profile is a dealbreaker.
- **Conversation**: how alike what they say is, how well their ways of
  talking fit, and how close they are on the features about communicating
  (conflict style, expressiveness, social energy). At least one of them
  must be willing to make the first move.
- **Meeting**: the chemistry they share, how alike their lives are, and how
  close they are on the features about how time is spent (travel, ambition,
  adventure). They must live within reach of each other.
- **Meeting again**: what each feels for the other, chemistry counting for
  half, and both must feel it. No true dealbreaker of either may be broken,
  including the ones no profile showed and intent that was not stated
  truly. Where a family's approval is required, the two families'
  expectations must fit.

The rates in the scenario (``conversation_rate``, ``date_rate``,
``second_date_rate``) say how many random eligible pairs pass each test.
They do not pick which pairs: that is decided by the pairs' hidden fit plus
luck drawn once from the seed. Better-chosen pairs pass more often, and by
how much is an output.

``informed`` is the chance a pair reaches the last stage as judged by a
reader with a perfect view of both profiles (true values, priorities,
traits, family, location) and no view of chemistry, mood or luck. It is
what "limited by information" means in the scorecard.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from wavelength.decisions import Decisions, feature_distances, standardise
from wavelength.population import Population
from wavelength.scenario import Scenario

STAGES: tuple[str, ...] = (
    "at least one said no",
    "both said yes",
    "conversation kept going",
    "they met",
    "both want to meet again",
)
LAST_STAGE = len(STAGES) - 1

#: What a conversation is made of, as shares of its explainable part.
CONVERSATION_MIX = {"vibe": 0.4, "manner": 0.3, "communication": 0.3}
COMMUNICATION_FEATURES = ("conflict_resolution", "emotional_expressiveness", "social_energy")

#: What gets two people to meet. Chemistry here is the part both feel.
DATE_MIX = {"chemistry": 0.5, "lifestyle": 0.3, "time": 0.2}
TIME_FEATURES = ("travel_frequency", "professional_ambition", "adventure_tolerance")

#: How much of wanting to meet again is chemistry; the rest is profile attraction.
AGAIN_CHEMISTRY = 0.5

#: A family approves a partner whose family's expectations are at least this
#: close to theirs, in standard units: roughly the nearer six in ten.
FAMILY_BAR = -0.25


@dataclass(frozen=True, eq=False)
class AnswerKey:
    """How far each pair would get if introduced and both answered."""

    stage: np.ndarray  # (n, n) int8, symmetric: 0 to 4; 0 for pairs on the same side
    informed: np.ndarray  # (n, n) symmetric: P(stage 4) from profiles alone
    conversation_possible: np.ndarray  # (n, n) bool: someone would make the first move
    within_reach: np.ndarray  # (n, n) bool: they live close enough to meet
    families_fit: np.ndarray  # (n, n) bool: every family whose approval is needed gives it

    def reaches(self, stage: int) -> np.ndarray:
        """Pairs that would get at least this far."""
        return self.stage >= stage


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF, by the Abramowitz and Stegun 7.1.26 approximation of erf.

    Accurate to about 1e-7, which is far finer than anything it is used for,
    and needs nothing beyond numpy.
    """
    x = np.abs(z) / sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * x)
    series = t * (
        0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    return 0.5 * (1.0 + np.sign(z) * (1.0 - series * np.exp(-x * x)))


def _chance_to_clear(known: np.ndarray, bar: np.ndarray | float, hidden_share: float) -> np.ndarray:
    """P(known + unseen >= bar) when the unseen part has variance ``hidden_share``."""
    if hidden_share <= 1e-9:
        return (known >= bar).astype(float)
    return _normal_cdf((known - bar) / sqrt(hidden_share))


def _closeness(population: Population, names: tuple[str, ...]) -> np.ndarray:
    """Minus the mean distance between two people on a few named features."""
    gaps = [feature_distances(population.codes, population.codes, name) for name in names]
    return -np.mean(gaps, axis=0)


def _symmetric_luck(rng: np.random.Generator, n: int) -> np.ndarray:
    upper = np.triu(rng.standard_normal((n, n)), k=1)
    return upper + upper.T


def _bar_for(scores: np.ndarray, eligible: np.ndarray, rate: float) -> float:
    """The level that the given share of eligible pairs clears."""
    values = scores[eligible]
    return float(np.quantile(values, 1.0 - rate)) if values.size else float("inf")


def build_key(
    rng: np.random.Generator, scenario: Scenario, population: Population, decisions: Decisions
) -> AnswerKey:
    """Work out every pair's path from the hidden truth and luck drawn once from the seed."""
    n, across, eligible = population.n, population.across, decisions.eligible
    noise = scenario.decision_noise
    steady, luck = sqrt(1.0 - noise), sqrt(noise)

    # Stage 1: both say yes.
    stage1 = decisions.would_accept & decisions.would_accept.T
    seen_share = (1.0 - noise) * (1.0 - scenario.chemistry_share)
    says_yes = _chance_to_clear(
        sqrt(seen_share) * decisions.profile, decisions.bar[:, None], 1.0 - seen_share
    ) * (across & ~decisions.refuses)

    # Stage 2: the conversation keeps going.
    talk = (
        sqrt(CONVERSATION_MIX["vibe"]) * standardise(population.vibe @ population.vibe.T, across)
        + sqrt(CONVERSATION_MIX["manner"])
        * standardise(population.manner @ population.manner.T, across)
        + sqrt(CONVERSATION_MIX["communication"])
        * standardise(_closeness(population, COMMUNICATION_FEATURES), across)
    )
    talk = standardise(talk, across)
    conversation = steady * talk + luck * _symmetric_luck(rng, n)
    talk_bar = _bar_for(conversation, eligible, scenario.conversation_rate)
    first_move = population.initiates[:, None] | population.initiates[None, :]
    stage2 = stage1 & first_move & (conversation >= talk_bar)
    keeps_talking = _chance_to_clear(steady * talk, talk_bar, noise) * first_move

    # Stage 3: they meet.
    life = standardise(
        sqrt(DATE_MIX["lifestyle"])
        * standardise(population.lifestyle @ population.lifestyle.T, across)
        + sqrt(DATE_MIX["time"]) * standardise(_closeness(population, TIME_FEATURES), across),
        across,
    )
    known_share = 1.0 - DATE_MIX["chemistry"]
    pull = sqrt(DATE_MIX["chemistry"]) * decisions.shared_chemistry + sqrt(known_share) * life
    date = steady * pull + luck * _symmetric_luck(rng, n)
    date_bar = _bar_for(date, eligible, scenario.date_rate)
    apart = np.linalg.norm(
        population.location[:, None, :] - population.location[None, :, :], axis=2
    )
    within_reach = apart <= population.travel[:, None] + population.travel[None, :]
    stage3 = stage2 & within_reach & (date >= date_bar)
    meets = (
        _chance_to_clear(
            steady * sqrt(known_share) * life, date_bar, 1.0 - (1.0 - noise) * known_share
        )
        * within_reach
    )

    # Stage 4: both want to meet again.
    wants = (
        sqrt(AGAIN_CHEMISTRY) * decisions.chemistry
        + sqrt(1.0 - AGAIN_CHEMISTRY) * decisions.profile
    )
    wants_on_the_day = steady * wants + luck * rng.standard_normal((n, n))
    again_bar = _bar_for(
        np.minimum(wants_on_the_day, wants_on_the_day.T), eligible, scenario.second_date_rate
    )
    no_dealbreaker = ~(decisions.refuses_in_person | decisions.refuses_in_person.T)
    family_fit = standardise(population.family @ population.family.T, across) >= FAMILY_BAR
    needs_family = population.family_required[:, None] | population.family_required[None, :]
    families_fit = family_fit | ~needs_family
    one_wants = wants_on_the_day >= again_bar
    stage4 = stage3 & one_wants & one_wants.T & no_dealbreaker & families_fit
    seen_again = (1.0 - noise) * (1.0 - AGAIN_CHEMISTRY)
    one_would = _chance_to_clear(sqrt(seen_again) * decisions.profile, again_bar, 1.0 - seen_again)
    both_would = one_would * one_would.T * (no_dealbreaker & families_fit)

    stage = (
        stage1.astype(np.int8)
        + stage2.astype(np.int8)
        + stage3.astype(np.int8)
        + stage4.astype(np.int8)
    )
    informed = says_yes * says_yes.T * keeps_talking * meets * both_would
    return AnswerKey(
        stage=stage,
        informed=informed,
        conversation_possible=first_move,
        within_reach=within_reach,
        families_fit=families_fit,
    )
