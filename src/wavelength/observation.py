"""What the app has managed to learn about the simulated people.

A matcher works from this and nothing else: feature values with gaps and
mistakes, text and photo signals of limited quality, and importance weights
resolved from onboarding, match-feedback and post-date signals by the same
``WeightProfile`` a real system would use.

The app records what people state. The four hard-filter answers are either
recorded as stated or not known at all; extraction never garbles them. What
people state is their true answer except where ``intent_openness`` says
otherwise: someone who is not open about what they want gives the usual
answer, and keeps a community requirement to themselves. So a rule can obey
every dealbreaker the app knows about and still introduce two people who
turn out, in person, to want different things. The scorecard counts
violations of *known* dealbreakers, which are always the rule's doing.

Four channels carry four different hidden traits: structured features,
text (what people say), photos (how they live) and voice (how they talk).
Each has its own quality setting, and none stands in for another.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from wavelength.features import FEATURE_NAMES, FEATURES, MISSING
from wavelength.filters import eligibility_matrix
from wavelength.population import TIERS, Population, unit_rows
from wavelength.scenario import DEALBREAKER_FEATURES, Scenario
from wavelength.weights import TIER_BANDS, Source, WeightProfile, WeightSignal

#: Weeks in which each kind of signal can arrive. Only used for provenance.
_WEEKS: dict[Source, tuple[int, int]] = {
    Source.ONBOARDING: (1, 1),
    Source.MATCH_FEEDBACK: (2, 8),
    Source.POST_DATE_FEEDBACK: (4, 12),
}


@dataclass(frozen=True, eq=False)
class Observations:
    """Everything a matcher is allowed to see."""

    majority: np.ndarray  # (n,) bool: on the larger side; the app knows who is seeking whom
    codes: np.ndarray  # (n, F) int, as stated; MISSING where a feature was not extracted
    text_embeddings: np.ndarray  # (n, VIBE_DIM)
    visual_embeddings: np.ndarray  # (n, LIFESTYLE_DIM); noise for people whose photos are hidden
    voice_embeddings: np.ndarray  # (n, MANNER_DIM)
    family_embeddings: np.ndarray  # (n, FAMILY_DIM): what the app has gathered about family
    photos_hidden: np.ndarray  # (n,) bool: no photo signal for this person
    community: np.ndarray  # (n,) which community each person belongs to
    community_required: np.ndarray  # (n,) bool: has told the app they will not go outside it
    age: np.ndarray  # (n,) years
    location: np.ndarray  # (n, 2) where each person lives, in a unit square
    weight_profiles: tuple[WeightProfile, ...] = field(repr=False)

    def copy(self) -> "Observations":
        """A deep copy: new arrays and new weight profiles, sharing nothing with this one."""
        arrays = {
            f.name: getattr(self, f.name).copy()
            for f in fields(self)
            if f.name != "weight_profiles"
        }
        return Observations(**arrays, weight_profiles=tuple(p.copy() for p in self.weight_profiles))

    @property
    def across(self) -> np.ndarray:
        """``[a, b]`` is True when a and b are on opposite sides."""
        return self.majority[:, None] != self.majority[None, :]

    @property
    def known_ok(self) -> np.ndarray:
        """Pairs on opposite sides that break no dealbreaker the app knows about.

        The four hard filters on recorded values (failing closed on a gap),
        and a stated requirement to stay within one's community.
        """
        other_community = self.community[:, None] != self.community[None, :]
        barred = (
            self.community_required[:, None] | self.community_required[None, :]
        ) & other_community
        return eligibility_matrix(self.codes) & self.across & ~barred

    @property
    def weights(self) -> np.ndarray:
        """Resolved per-user weights as an ``(n, F)`` array."""
        return np.array([[p.weight(f) for f in FEATURE_NAMES] for p in self.weight_profiles])


def _share_among_incomplete(overall: float, complete: float) -> float:
    """The rate among users without a full profile that makes the overall rate come out right."""
    if complete >= 1.0:
        return 1.0
    return (overall - complete) / (1.0 - complete)


def _observe_features(
    rng: np.random.Generator, scenario: Scenario, codes: np.ndarray, stated: np.ndarray
) -> np.ndarray:
    n = codes.shape[0]
    full = rng.random(n) < scenario.full_profile_share
    observed = stated.copy()

    trait_rate = _share_among_incomplete(scenario.feature_coverage, scenario.full_profile_share)
    for j, name in enumerate(FEATURE_NAMES):
        if name in DEALBREAKER_FEATURES:
            continue
        k = len(FEATURES[name].values)
        wrong = rng.random(n) >= scenario.extraction_accuracy
        other = (codes[:, j] + rng.integers(1, k, size=n)) % k  # any value but the true one
        observed[:, j] = np.where(wrong, other, codes[:, j])
        observed[~full & (rng.random(n) >= trait_rate), j] = MISSING

    # Users who have not answered all four dealbreaker questions are missing
    # one of them, chosen at random. One gap is enough to fail closed.
    known_rate = _share_among_incomplete(
        scenario.dealbreakers_known_share, scenario.full_profile_share
    )
    unanswered = ~full & (rng.random(n) >= known_rate)
    columns = [FEATURE_NAMES.index(name) for name in DEALBREAKER_FEATURES]
    gap = rng.choice(columns, size=n)
    observed[unanswered, gap[unanswered]] = MISSING
    return observed


def _observe_weights(
    rng: np.random.Generator, scenario: Scenario, tiers: np.ndarray
) -> tuple[WeightProfile, ...]:
    """Generate importance signals per user and let ``WeightProfile`` resolve them.

    A signal reports the true tier with the source's accuracy and one of the
    other two tiers otherwise.
    """
    sources: dict[Source, tuple[float, float]] = {
        Source.ONBOARDING: (scenario.onboarding_reach, scenario.self_knowledge),
        Source.MATCH_FEEDBACK: (scenario.match_feedback_reach, scenario.match_feedback_accuracy),
        Source.POST_DATE_FEEDBACK: (scenario.post_date_reach, scenario.post_date_accuracy),
    }
    n, n_features = tiers.shape
    profiles = tuple(WeightProfile() for _ in range(n))
    for source, (reach, accuracy) in sources.items():
        exists = rng.random((n, n_features)) < reach
        correct = rng.random((n, n_features)) < accuracy
        reported = np.where(correct, tiers, (tiers + rng.integers(1, 3, size=tiers.shape)) % 3)
        position = rng.random((n, n_features))
        confidence = rng.uniform(0.5, 1.0, size=(n, n_features))
        week = rng.integers(_WEEKS[source][0], _WEEKS[source][1] + 1, size=(n, n_features))
        for user, j in np.argwhere(exists):
            tier = TIERS[reported[user, j]]
            low, high = TIER_BANDS[tier]
            profiles[user].record(
                WeightSignal(
                    feature=FEATURE_NAMES[j],
                    weight=float(low + (high - low) * position[user, j]),
                    tier=tier,
                    source=source,
                    confidence=float(confidence[user, j]),
                    observed_at=int(week[user, j]),
                    evidence=f"simulated {source.name.lower()} signal",
                )
            )
    return profiles


def _observe_vectors(rng: np.random.Generator, truth: np.ndarray, quality: float) -> np.ndarray:
    """Blend the true vectors with noise: quality 1 returns the truth, 0 returns noise."""
    noise = rng.standard_normal(truth.shape) / np.sqrt(truth.shape[1])
    return unit_rows(quality * truth + (1.0 - quality) * noise)


def observe(rng: np.random.Generator, scenario: Scenario, population: Population) -> Observations:
    """Derive the matcher's view of a population.

    The weights reflect onboarding plus one earlier round of match and
    post-date feedback: people arrive with a little history. The engine
    adds more as introductions are answered.
    """
    photos = _observe_vectors(rng, population.lifestyle, scenario.photo_signal_quality)
    no_photos = _observe_vectors(rng, population.lifestyle, 0.0)
    return Observations(
        majority=population.majority.copy(),
        codes=_observe_features(rng, scenario, population.codes, population.stated_codes),
        text_embeddings=_observe_vectors(rng, population.vibe, scenario.text_signal_quality),
        visual_embeddings=np.where(population.photos_hidden[:, None], no_photos, photos),
        voice_embeddings=_observe_vectors(rng, population.manner, scenario.voice_signal_quality),
        family_embeddings=_observe_vectors(rng, population.family, scenario.family_visibility),
        photos_hidden=population.photos_hidden.copy(),
        community=population.community.copy(),
        community_required=population.endogamous & population.open_about_intent,
        age=population.age.copy(),
        location=population.location.copy(),
        weight_profiles=_observe_weights(rng, scenario, population.tiers),
    )
