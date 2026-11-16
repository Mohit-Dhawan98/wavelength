"""Baselines: the strategies the article moved away from.

They exist so the bidirectional matcher has something honest to be compared
against on the same population. Both use the same assignment rule, "give
each user their own best available candidate"; they differ only in what
"best" is computed from.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from wavelength.scoring import similarity_matrix


def similarity_scores(text_embeddings: np.ndarray) -> np.ndarray:
    """Stage 1 scores: text-embedding similarity and nothing else.

    Symmetric by construction, and blind to dealbreakers: two profiles that
    sound alike score highly whatever they want from life.
    """
    return similarity_matrix(text_embeddings)


def one_directional_greedy(
    scores: np.ndarray, eligible: np.ndarray, order: Sequence[int]
) -> list[tuple[int, int]]:
    """Give each user, in turn, their highest-scored candidate who is still free.

    This is "find A's best match" with the one concession needed to compare
    it with the bidirectional matcher: each user is matched at most once per
    round. Nothing here asks what the candidate thinks of the user. In each
    returned pair the first element is the user the match was chosen for.
    """
    matched = np.zeros(scores.shape[0], dtype=bool)
    pairs: list[tuple[int, int]] = []
    for a in order:
        if matched[a]:
            continue
        available = eligible[a] & ~matched
        available[a] = False
        if not available.any():
            continue
        b = int(np.argmax(np.where(available, scores[a], -np.inf)))
        matched[a] = matched[b] = True
        pairs.append((int(a), b))
    return pairs
