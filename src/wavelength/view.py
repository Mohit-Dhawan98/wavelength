"""What a rule is allowed to see, and the helpers it can pair people with.

A ``View`` is handed to a rule once per round. It is built only from a
copy of what the app has observed plus the history of the run so far: who
was introduced, who answered, who said yes, and what the app has learned
about priorities from feedback. Nothing in it refers back to the hidden
truth or the answer key, and its arrays are read-only copies, so a rule
cannot see or change anything it should not. Tests check both.

All pair matrices are ``n x n`` and read "me, then them": ``s[i, j]`` is how
good ``j`` looks for ``i``. Scores are on a 0 to 1 scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from wavelength.baselines import one_directional_greedy
from wavelength.features import FEATURE_NAMES, FEATURES, MISSING
from wavelength.matching import TOP_N, RankCurve, bidirectional_match, rank_matrix
from wavelength.observation import Observations
from wavelength.scoring import NEUTRAL, similarity_matrix
from wavelength.weights import GLOBAL_WEIGHTS

Pairs = list[tuple[int, int]]


def _frozen(array: np.ndarray) -> np.ndarray:
    array = np.array(array)  # always a fresh copy
    array.setflags(write=False)
    return array


class Board:
    """Everything the app can see that does not change from round to round.

    Built once per run from a copy of the observations. Feature similarity
    is prepared in a form that makes "one minus the weighted mean distance"
    a pair of small matrix products, so it can be recomputed every round as
    the learned weights change.
    """

    def __init__(self, observations: Observations) -> None:
        obs = observations
        self.n = obs.codes.shape[0]
        self.larger_side = _frozen(obs.majority)
        self.across = _frozen(obs.across)
        self.known_ok = _frozen(obs.known_ok)
        self.text = _frozen(similarity_matrix(obs.text_embeddings))
        self.voice = _frozen(similarity_matrix(obs.voice_embeddings))
        self.family = _frozen(similarity_matrix(obs.family_embeddings))
        photo = similarity_matrix(obs.visual_embeddings)
        hidden = obs.photos_hidden[:, None] | obs.photos_hidden[None, :]
        self.photo = _frozen(np.where(hidden, NEUTRAL, photo))
        gap = obs.location[:, None, :] - obs.location[None, :, :]
        self.distance = _frozen(np.linalg.norm(gap, axis=2))
        self.age = _frozen(obs.age)
        self.same_community = _frozen(obs.community[:, None] == obs.community[None, :])

        # One column per (feature, value); a row has a 1 where the person holds that value.
        starts = np.cumsum([0] + [len(FEATURES[name].values) for name in FEATURE_NAMES])
        self._feature_of_column = np.repeat(np.arange(len(FEATURE_NAMES)), np.diff(starts))
        self._holds = np.zeros((self.n, starts[-1]))
        self._distances = np.zeros((starts[-1], starts[-1]))
        for j, name in enumerate(FEATURE_NAMES):
            known = np.flatnonzero(obs.codes[:, j] != MISSING)
            self._holds[known, starts[j] + obs.codes[known, j]] = 1.0
            self._distances[starts[j] : starts[j + 1], starts[j] : starts[j + 1]] = FEATURES[
                name
            ].distances
        self._known = (obs.codes != MISSING).astype(float)
        self._gap_to = self._distances @ self._holds.T  # (columns, n)
        self.global_weights = np.tile([GLOBAL_WEIGHTS[f] for f in FEATURE_NAMES], (self.n, 1))

    def feature_similarity(self, weights: np.ndarray) -> np.ndarray:
        """One minus the weighted mean distance, by each viewer's weights.

        The same arithmetic as ``scoring.score_matrix``'s feature term:
        features unknown on either side are skipped and the rest
        renormalised; with nothing in common the result is neutral.
        """
        weighted = (self._holds * weights[:, self._feature_of_column]) @ self._gap_to
        total = (weights * self._known) @ self._known.T
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(total > 0, 1.0 - weighted / total, NEUTRAL)


@dataclass(frozen=True, eq=False)
class History:
    """What the app has logged so far. All arrays are read-only copies.

    Pair matrices are ``n x n``; ``answered[i, j]`` and ``said_yes[i, j]``
    are about person ``i``'s response to being introduced to ``j``.
    """

    introduced: np.ndarray  # (n, n) int: round in which the pair was introduced, -1 if never
    answered: np.ndarray  # (n, n) bool
    said_yes: np.ndarray  # (n, n) bool
    introductions: np.ndarray  # (n,) how many introductions each person has been sent
    unanswered: np.ndarray  # (n,) how many of those they let expire
    yeses: np.ndarray  # (n,) how many they said yes to
    declines: np.ndarray  # (n,) how many they said no to
    recent_declines: np.ndarray  # (n,) the same, over the last two rounds only
    idle_rounds: np.ndarray  # (n,) introductions in a row they have let expire


class View:
    """What the app can see this round, and ways to turn a score into pairs."""

    __slots__ = (
        "round",
        "rounds",
        "n",
        "larger_side",
        "available",
        "open",
        "ok",
        "text",
        "photo",
        "voice",
        "family",
        "distance",
        "age",
        "same_community",
        "history",
        "_board",
        "_weights",
        "_order",
        "_seed",
    )

    def __init__(
        self,
        board: Board,
        *,
        round: int,
        rounds: int,
        available: np.ndarray,
        weights: np.ndarray,
        history: History,
        order: np.ndarray,
        seed: int = 0,
    ) -> None:
        self.round = round
        self.rounds = rounds
        self.n = board.n
        self.larger_side = board.larger_side
        self.available = _frozen(available)
        unmet = history.introduced < 0
        open_pairs = board.across & available[:, None] & available[None, :] & unmet
        self.open = _frozen(open_pairs)
        self.ok = _frozen(open_pairs & board.known_ok)
        self.text = board.text
        self.photo = board.photo
        self.voice = board.voice
        self.family = board.family
        self.distance = board.distance
        self.age = board.age
        self.same_community = board.same_community
        self.history = history
        self._board = board
        self._weights = _frozen(weights)
        self._order = _frozen(order)
        self._seed = seed

    # --- signals ---------------------------------------------------------

    def features(self, weights: str | Mapping[str, float] = "personal") -> np.ndarray:
        """Agreement on structured features, 0 to 1, by the viewer's weights.

        ``"personal"`` uses what the app has learned about each person so
        far; ``"global"`` uses one set of weights for everyone. A mapping
        from feature name to weight uses your own global weights.
        """
        if isinstance(weights, str):
            if weights == "personal":
                return self._board.feature_similarity(self._weights)
            if weights == "global":
                return self._board.feature_similarity(self._board.global_weights)
            raise ValueError(f'weights must be "personal", "global" or a mapping, got {weights!r}')
        unknown = sorted(set(weights) - set(FEATURE_NAMES))
        if unknown:
            raise ValueError(f"unknown feature(s) {unknown}; features are {list(FEATURE_NAMES)}")
        row = [float(weights.get(name, 0.0)) for name in FEATURE_NAMES]
        return self._board.feature_similarity(np.tile(row, (self.n, 1)))

    def random(self) -> np.ndarray:
        """An ``n x n`` matrix of random numbers between 0 and 1.

        Seeded by the world and the round, so it is the same for every rule
        and every rerun: use it for a random baseline or to break ties.
        """
        return np.random.default_rng([self._seed, self.round]).random((self.n, self.n))

    # --- turning a score into pairs --------------------------------------

    def _among(self, among: np.ndarray | None) -> np.ndarray:
        return self.ok if among is None else (np.asarray(among, dtype=bool) & self.open)

    def rank(self, scores: np.ndarray, among: np.ndarray | None = None) -> np.ndarray:
        """Each person's ranking of their candidates: 1 is best, 0 means not a candidate."""
        return rank_matrix(np.asarray(scores, dtype=float), self._among(among))

    def pair_best(self, scores: np.ndarray, among: np.ndarray | None = None) -> Pairs:
        """Give each person, in turn, their best-scored candidate who is still free.

        One-directional: nobody asks what the candidate thinks. The order
        people are visited in is random and the same for every rule.
        """
        scores = np.asarray(scores, dtype=float)
        return one_directional_greedy(scores, self._among(among), self._order.tolist())

    def pair_mutual(
        self,
        scores: np.ndarray,
        top: int = TOP_N,
        curve: RankCurve | None = None,
        among: np.ndarray | None = None,
    ) -> Pairs:
        """Pair people who are each in the other's top ``top``, most promising first.

        The article's method: mutual top-N, then assignment by the product
        of the two acceptance probabilities. ``curve`` replaces the
        article's ``P(rank)``. People with no mutual pair get nobody.
        """
        scores = np.asarray(scores, dtype=float)
        return [(p.a, p.b) for p in bidirectional_match(scores, self._among(among), top, curve)]

    def pair_stable(self, scores: np.ndarray, among: np.ndarray | None = None) -> Pairs:
        """A stable matching on the scores (deferred acceptance, larger side proposing).

        Stable with respect to the scores given: no two candidates would
        both rather have each other than the partner they were given.
        Whether that is stable in fact depends on how good the scores are.
        """
        scores = np.asarray(scores, dtype=float)
        return deferred_acceptance(scores, self._among(among), self.larger_side)


def deferred_acceptance(scores: np.ndarray, allowed: np.ndarray, proposers: np.ndarray) -> Pairs:
    """Gale-Shapley with incomplete lists: proposers work down their own ranking.

    ``scores[i, j]`` is how ``i`` rates ``j``. A receiver holds the best
    proposal so far by their own scores and releases the rest.
    """
    wish_lists = {
        int(p): [int(c) for c in np.argsort(-scores[p], kind="stable") if allowed[p, c]]
        for p in np.flatnonzero(proposers)
    }
    next_choice = dict.fromkeys(wish_lists, 0)
    holding: dict[int, int] = {}
    free = [p for p in wish_lists if wish_lists[p]]
    while free:
        proposer = free.pop()
        wishes = wish_lists[proposer]
        if next_choice[proposer] >= len(wishes):
            continue
        receiver = wishes[next_choice[proposer]]
        next_choice[proposer] += 1
        current = holding.get(receiver)
        if current is None:
            holding[receiver] = proposer
        elif scores[receiver, proposer] > scores[receiver, current]:
            holding[receiver] = proposer
            free.append(current)
        else:
            free.append(proposer)
    return sorted((proposer, receiver) for receiver, proposer in holding.items())


class RuleError(Exception):
    """A rule returned something the engine cannot use. The message says what and where."""


def checked_pairs(result: object, view: View) -> Pairs:
    """Turn a rule's return value into pairs, or say plainly what is wrong with it."""
    where = f"round {view.round}"
    if result is None:
        raise RuleError(f"{where}: the rule returned nothing; return a list of (a, b) pairs")
    if not isinstance(result, Iterable) or isinstance(result, (str, bytes, np.ndarray, Mapping)):
        raise RuleError(
            f"{where}: the rule returned {type(result).__name__}; return a list of (a, b) pairs"
        )
    pairs: Pairs = []
    used: dict[int, tuple[int, int]] = {}
    for item in result:
        ends: Sequence[object] | None = None
        if hasattr(item, "a") and hasattr(item, "b"):
            ends = (item.a, item.b)
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            ends = item
        if ends is None or not all(isinstance(e, (int, np.integer)) for e in ends):
            raise RuleError(f"{where}: {item!r} is not a pair of two person numbers")
        a, b = int(ends[0]), int(ends[1])
        for person in (a, b):
            if not 0 <= person < view.n:
                raise RuleError(
                    f"{where}: person {person} does not exist (people are 0 to {view.n - 1})"
                )
        if view.larger_side[a] == view.larger_side[b]:
            raise RuleError(f"{where}: pair ({a}, {b}) puts two people from the same side together")
        for person in (a, b):
            if not view.available[person]:
                raise RuleError(f"{where}: person {person} has left the pool; check v.available")
            if person in used:
                raise RuleError(
                    f"{where}: person {person} is in two pairs, {used[person]} and ({a}, {b})"
                )
        if view.history.introduced[a, b] >= 0:
            raise RuleError(
                f"{where}: pair ({a}, {b}) was already introduced in round "
                f"{int(view.history.introduced[a, b])}; use v.open or v.ok to avoid repeats"
            )
        used[a] = used[b] = (a, b)
        pairs.append((a, b))
    return pairs
