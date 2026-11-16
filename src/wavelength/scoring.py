"""Tri-signal, per-viewer scoring.

Three signals, because none of them sees the whole person:

- text      how someone describes themselves (the "vibe")
- visual    how they appear to live, as opposed to what they say
- features  structured preferences, weighted by what the viewer cares about

The mix is the article's: ``0.25 x text + 0.15 x visual + 0.60 x features``.
In production the text signal is an embedding similarity and the visual
signal comes from features extracted from photos by a multimodal model. Here
both are cosine similarities between vectors the caller supplies. They stand
in for the real signals; they are not derived from any text or image.

The score is directed. ``directed_score(a, b, weights_of_a)`` answers "how
good is b for a, by a's priorities". The feature distances between two people
are the same in both directions; the weights are not. That is the only
source of asymmetry, and it is enough to make Score(A->B) and Score(B->A)
disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from wavelength.features import FEATURE_NAMES, FEATURES, MISSING

TEXT_WEIGHT = 0.25
VISUAL_WEIGHT = 0.15
FEATURE_WEIGHT = 0.60

#: Feature similarity when two profiles share no extracted feature. Neutral,
#: because the absence of evidence is neither a match nor a mismatch.
NEUTRAL = 0.5


@dataclass(frozen=True, eq=False)
class Profile:
    """What the matcher knows about one user."""

    user_id: str
    features: Mapping[str, str | None]
    text_embedding: np.ndarray = field(repr=False)
    visual_embedding: np.ndarray = field(repr=False)


def cosine_similarity01(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity rescaled from [-1, 1] to [0, 1], so all three signals share a scale."""
    cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return (cos + 1.0) / 2.0


def feature_similarity(
    viewer_weights: Mapping[str, float],
    viewer: Mapping[str, str | None],
    candidate: Mapping[str, str | None],
) -> float:
    """One minus the weighted mean distance, using the viewer's weights.

    Features missing on either side are skipped and the remaining weights
    renormalised, so a partly extracted profile is scored on what is known
    rather than penalised for what is not.
    """
    weighted_distance = 0.0
    total_weight = 0.0
    for name in FEATURE_NAMES:
        a, b = viewer.get(name), candidate.get(name)
        if a is None or b is None:
            continue
        w = viewer_weights[name]
        weighted_distance += w * FEATURES[name].distance(a, b)
        total_weight += w
    if total_weight == 0.0:
        return NEUTRAL
    return 1.0 - weighted_distance / total_weight


def tri_signal_score(text: float, visual: float, features: float) -> float:
    """Combine the three signals, each already on a 0-1 scale."""
    return TEXT_WEIGHT * text + VISUAL_WEIGHT * visual + FEATURE_WEIGHT * features


def directed_score(
    viewer: Profile, candidate: Profile, viewer_weights: Mapping[str, float]
) -> float:
    """Score(viewer -> candidate): how good the candidate is by the viewer's priorities.

    Call this only for pairs that have passed the hard filters. The score
    does not know about dealbreakers and will happily average one away.
    """
    return tri_signal_score(
        text=cosine_similarity01(viewer.text_embedding, candidate.text_embedding),
        visual=cosine_similarity01(viewer.visual_embedding, candidate.visual_embedding),
        features=feature_similarity(viewer_weights, viewer.features, candidate.features),
    )


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """All-pairs ``cosine_similarity01`` for an ``(n, d)`` array of embeddings."""
    unit = _unit_rows(embeddings)
    return (unit @ unit.T + 1.0) / 2.0


def score_matrix(
    codes: np.ndarray,
    weights: np.ndarray,
    text_embeddings: np.ndarray,
    visual_embeddings: np.ndarray,
) -> np.ndarray:
    """All-pairs ``directed_score``; row is the viewer, column the candidate.

    ``codes`` and ``weights`` are ``(n_users, n_features)`` arrays in
    ``FEATURE_NAMES`` order (``MISSING`` marks an unextracted feature). This is
    the same arithmetic as the scalar functions above, vectorised so a whole
    population can be scored at once; a test pins the two together.
    The result is not symmetric unless every user has the same weights.
    """
    n = codes.shape[0]
    weighted_distance = np.zeros((n, n))
    total_weight = np.zeros((n, n))
    for j, name in enumerate(FEATURE_NAMES):
        column = codes[:, j]
        known = column != MISSING
        both = known[:, None] & known[None, :]
        safe = np.where(known, column, 0)
        d = FEATURES[name].distances[safe[:, None], safe[None, :]]
        w = weights[:, j][:, None]  # the viewer's weight, broadcast across candidates
        weighted_distance += np.where(both, w * d, 0.0)
        total_weight += np.where(both, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        features = np.where(total_weight > 0, 1.0 - weighted_distance / total_weight, NEUTRAL)
    return tri_signal_score(
        similarity_matrix(text_embeddings), similarity_matrix(visual_embeddings), features
    )
