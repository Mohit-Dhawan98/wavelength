"""Shared fixtures: small, fully specified profiles."""

import numpy as np
import pytest

from wavelength.scoring import Profile

BASE_FEATURES = {
    "relationship_timeline": "marriage_focused",
    "commitment_style": "requires_exclusivity",
    "family_plans": "wants_children_soon",
    "trust_boundaries": "needs_security",
    "emotional_availability": "open",
    "intellectual_connection": "high",
    "emotional_expressiveness": "expressive",
    "adventure_tolerance": "moderate",
    "social_energy": "ambivert",
    "conflict_resolution": "direct",
    "professional_ambition": "balanced",
    "travel_frequency": "occasional",
}


def make_profile(user_id="u", text=(1.0, 0.0), visual=(1.0, 0.0), **overrides) -> Profile:
    return Profile(
        user_id=user_id,
        features={**BASE_FEATURES, **overrides},
        text_embedding=np.array(text),
        visual_embedding=np.array(visual),
    )


@pytest.fixture
def base_features():
    return dict(BASE_FEATURES)
