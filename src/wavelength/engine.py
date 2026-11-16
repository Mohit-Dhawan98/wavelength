"""Play a rule against a world for several rounds and score it against the answer key.

An ``Arena`` is one seeded world prepared for play: the hidden truth and its
answer key on one side, and on the other a ``Board`` built from a copy of
the observations. A rule only ever receives a ``View`` made from the board.

Each round ("week") the rule proposes introductions among the people still
available. What happens next is read off things fixed before the rule ran:

- Whether a person answers depends on their own habit of opening the app,
  worn down by every letdown so far, against a number drawn for that person
  and that week when the world was made. An introduction a person does not
  answer expires.
- If both answer, how far the pair gets is the answer key's entry for it.
- A letdown is an introduction the person turned down, or one they said
  yes to that the other person did not. Enough of them and the person gives
  up for good.
- A couple that gets as far as ``leave_after_stage`` leaves the pool.
- Answering reveals something about what matters to the person, with the
  scenario's reach and accuracy, and the app's learned weights update.

All luck is drawn from the seed in advance, per person and per week, never
per rule. Two rules in the same arena differ only in what they chose to do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from wavelength.best import maximum_matching, most_couples
from wavelength.decisions import MAX_ROUNDS
from wavelength.features import FEATURE_NAMES
from wavelength.key import LAST_STAGE
from wavelength.population import TIERS
from wavelength.scenario import Scenario
from wavelength.view import Board, History, Pairs, View, checked_pairs
from wavelength.weights import TIER_BANDS, Source, WeightProfile, WeightSignal
from wavelength.world import generate_world

DEFAULT_ROUNDS = 8

#: How many people from each side are followed in the event log.
CAST_PER_SIDE = 400

#: Rounds that count as "recent" in ``History.recent_declines``.
RECENT = 2

RuleFunction = Callable[[View], object]


@dataclass(eq=False)
class Introduction:
    """One introduction and how it went."""

    round: int
    a: int
    b: int
    a_answered: bool
    b_answered: bool
    a_yes: bool
    b_yes: bool
    stage: int  # how far the pair got: 0 if it expired or someone said no
    known_ok: bool  # broke no dealbreaker the app knew about


@dataclass(eq=False)
class Play:
    """Everything that happened in one run of one rule."""

    rounds: int
    introductions: list[Introduction] = field(default_factory=list)
    gave_up_in: dict[int, int] = field(default_factory=dict)  # person -> round
    left_in: dict[int, int] = field(default_factory=dict)  # person -> round


class _Feedback:
    """Luck for what people reveal when they explain themselves, drawn per person and week."""

    def __init__(self, seed: int, n: int, rounds: int) -> None:
        rng = np.random.default_rng([seed, 0xFEED])
        shape = (2, rounds, n, len(FEATURE_NAMES))
        self.reveal = rng.random(shape)
        self.correct = rng.random(shape)
        self.other_tier = rng.integers(1, 3, size=shape)
        self.position = rng.random(shape)
        self.confidence = rng.uniform(0.5, 1.0, size=shape)

    def signals(
        self,
        kind: int,
        round: int,
        person: int,
        true_tiers: np.ndarray,
        reach: float,
        accuracy: float,
    ) -> list[WeightSignal]:
        source = (Source.MATCH_FEEDBACK, Source.POST_DATE_FEEDBACK)[kind]
        found = []
        for j in np.flatnonzero(self.reveal[kind, round, person] < reach):
            tier_code = int(true_tiers[j])
            if self.correct[kind, round, person, j] >= accuracy:
                tier_code = (tier_code + int(self.other_tier[kind, round, person, j])) % 3
            tier = TIERS[tier_code]
            low, high = TIER_BANDS[tier]
            found.append(
                WeightSignal(
                    feature=FEATURE_NAMES[j],
                    weight=float(low + (high - low) * self.position[kind, round, person, j]),
                    tier=tier,
                    source=source,
                    confidence=float(self.confidence[kind, round, person, j]),
                    observed_at=100 + round,  # after everything people arrived with
                    evidence=f"simulated {source.name.lower()} in week {round + 1}",
                )
            )
        return found


class Arena:
    """One seeded world, ready for any number of rules to be played against it."""

    def __init__(self, scenario: Scenario, seed: int = 0) -> None:
        self.scenario = scenario
        self.seed = seed
        self._world = generate_world(seed, scenario)
        self._observations = self._world.observations.copy()
        self._board = Board(self._observations.copy())
        self._best: dict[int, int] = {}
        self._possible: np.ndarray | None = None
        self._informed: dict[int, Play] = {}

    @property
    def n(self) -> int:
        return self._world.population.n

    def best_possible(self, stage: int) -> int:
        """The most disjoint couples that could reach at least ``stage`` in this world."""
        if stage not in self._best:
            world = self._world
            self._best[stage] = most_couples(world.key.reaches(stage), world.population.majority)
        return self._best[stage]

    def couples(self, play: Play, stage: int) -> int:
        """Disjoint couples a run formed that reached at least ``stage``."""
        return _disjoint([(i.a, i.b) for i in play.introductions if i.stage >= stage])

    def informed_play(self, rounds: int = DEFAULT_ROUNDS) -> Play:
        """A benchmark run by a reader of perfect profiles who cannot see chemistry or luck.

        Each round it introduces the pairs most likely to go all the way as
        judged from the profile-only chances in the answer key, most likely
        first. It is greedy, so it is a strong benchmark, not a strict upper
        bound; ``best_possible`` is the strict one.
        """
        if rounds not in self._informed:
            informed = self._world.key.informed

            def benchmark(view: View) -> Pairs:
                rows, cols = np.nonzero(np.triu(view.open) & (informed > 0))
                taken = np.zeros(view.n, dtype=bool)
                pairs = []
                for k in np.argsort(-informed[rows, cols], kind="stable"):
                    a, b = int(rows[k]), int(cols[k])
                    if not taken[a] and not taken[b]:
                        taken[a] = taken[b] = True
                        pairs.append((a, b))
                return pairs

            self._informed[rounds] = self.play(benchmark, rounds)
        return self._informed[rounds]

    def cast(self) -> list[int]:
        """The people whose story the event log follows: everyone, up to a cap per side."""
        larger = np.flatnonzero(self._board.larger_side)[:CAST_PER_SIDE]
        scarce = np.flatnonzero(~self._board.larger_side)[:CAST_PER_SIDE]
        return sorted(int(p) for p in (*larger, *scarce))

    def possible_partners(self) -> np.ndarray:
        """For each person, how many people the answer key says they would hit it off with.

        This is hidden truth. It is for the scorecard after a run, never for a rule.
        """
        if self._possible is None:
            reaches = self._world.key.reaches(LAST_STAGE)
            self._possible = (reaches | reaches.T).sum(axis=1).astype(int)
        return self._possible

    def describe_person(self, person: int) -> dict[str, object]:
        """What the app shows about one person: nothing hidden."""
        obs = self._observations
        communities = list(self.scenario.community_shares)
        return {
            "id": person,
            "side": "larger" if obs.majority[person] else "scarce",
            "age": int(obs.age[person]),
            "community": communities[int(obs.community[person])],
            "x": round(float(obs.location[person, 0]), 3),
            "y": round(float(obs.location[person, 1]), 3),
            "possible": int(self.possible_partners()[person]),
        }

    def play(self, rule: RuleFunction, rounds: int = DEFAULT_ROUNDS) -> Play:
        """Run one rule for ``rounds`` rounds and record what happened."""
        if not 1 <= rounds <= MAX_ROUNDS:
            raise ValueError(f"rounds must be between 1 and {MAX_ROUNDS}, got {rounds}")
        world, scenario, n = self._world, self.scenario, self.n
        truth, decisions, key = world.population, world.decisions, world.key
        board = self._board
        profiles: list[WeightProfile] = [p.copy() for p in self._observations.weight_profiles]
        weights = np.array([[p.weight(f) for f in FEATURE_NAMES] for p in profiles])
        feedback = _Feedback(self.seed, n, rounds)
        order_rng = np.random.default_rng([self.seed, 0x0DE5])

        introduced = np.full((n, n), -1, dtype=np.int16)
        answered = np.zeros((n, n), dtype=bool)
        said_yes = np.zeros((n, n), dtype=bool)
        sent = np.zeros(n, dtype=np.int64)
        expired = np.zeros(n, dtype=np.int64)
        yeses = np.zeros(n, dtype=np.int64)
        declines = np.zeros(n, dtype=np.int64)
        declines_by_round = np.zeros((rounds, n), dtype=np.int64)
        idle = np.zeros(n, dtype=np.int64)
        letdowns = np.zeros(n, dtype=np.int64)
        in_couple = np.zeros(n, dtype=bool)
        gave_up = np.zeros(n, dtype=bool)
        play = Play(rounds)

        for week in range(rounds):
            history = History(
                *(
                    _readonly(x)
                    for x in (
                        introduced,
                        answered,
                        said_yes,
                        sent,
                        expired,
                        yeses,
                        declines,
                        declines_by_round[max(0, week - RECENT) : week].sum(axis=0),
                        idle,
                    )
                )
            )
            view = View(
                board,
                round=week,
                rounds=rounds,
                available=~in_couple,
                weights=weights,
                history=history,
                order=order_rng.permutation(n),
                seed=self.seed,
            )
            pairs = checked_pairs(rule(view), view)

            willing = truth.habit * (1.0 - scenario.fatigue) ** letdowns
            opens = (decisions.open_luck[:, week] < willing) & ~gave_up
            for a, b in pairs:
                a_answered, b_answered = bool(opens[a]), bool(opens[b])
                a_yes = a_answered and bool(decisions.would_accept[a, b])
                b_yes = b_answered and bool(decisions.would_accept[b, a])
                stage = int(key.stage[a, b]) if (a_yes and b_yes) else 0
                introduced[a, b] = introduced[b, a] = week
                play.introductions.append(
                    Introduction(
                        week,
                        a,
                        b,
                        a_answered,
                        b_answered,
                        a_yes,
                        b_yes,
                        stage,
                        bool(board.known_ok[a, b]),
                    )
                )
                for me, them, i_answered, i_yes, they_yes in (
                    (a, b, a_answered, a_yes, b_yes),
                    (b, a, b_answered, b_yes, a_yes),
                ):
                    sent[me] += 1
                    if not i_answered:
                        expired[me] += 1
                        idle[me] += 1
                        continue
                    idle[me] = 0
                    answered[me, them] = True
                    said_yes[me, them] = i_yes
                    yeses[me] += i_yes
                    if not i_yes:
                        declines[me] += 1
                        declines_by_round[week, me] += 1
                    if not i_yes or not they_yes:
                        letdowns[me] += 1
                    kinds = [(0, scenario.match_feedback_reach, scenario.match_feedback_accuracy)]
                    if stage >= 3:
                        kinds.append((1, scenario.post_date_reach, scenario.post_date_accuracy))
                    for kind, reach, accuracy in kinds:
                        for signal in feedback.signals(
                            kind, week, me, truth.tiers[me], reach, accuracy
                        ):
                            profiles[me].record(signal)
                            j = FEATURE_NAMES.index(signal.feature)
                            weights[me, j] = profiles[me].weight(signal.feature)
                if stage >= scenario.leave_after_stage:
                    in_couple[a] = in_couple[b] = True
                    play.left_in[a] = play.left_in[b] = week

            quitting = (letdowns >= truth.patience) & ~gave_up & ~in_couple
            for person in np.flatnonzero(quitting):
                play.gave_up_in[int(person)] = week
            gave_up |= quitting
        return play


def _readonly(array: np.ndarray) -> np.ndarray:
    copy_ = array.copy()
    copy_.setflags(write=False)
    return copy_


def _disjoint(pairs: list[tuple[int, int]]) -> int:
    """The largest number of pairs that share nobody (pairs always cross the two sides)."""
    if not pairs:
        return 0
    left = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    index = {person: k for k, person in enumerate(left)}
    # Orient every pair the same way round by splitting people into two colours.
    colour: dict[int, int] = {}
    neighbours: dict[int, list[int]] = {person: [] for person in left}
    for a, b in pairs:
        neighbours[a].append(b)
        neighbours[b].append(a)
    for start in left:
        if start in colour:
            continue
        colour[start] = 0
        stack = [start]
        while stack:
            person = stack.pop()
            for other in neighbours[person]:
                if other not in colour:
                    colour[other] = 1 - colour[person]
                    stack.append(other)
    adjacency: list[list[int]] = [[] for _ in left]
    for a, b in pairs:
        u, v = (a, b) if colour[a] == 0 else (b, a)
        adjacency[index[u]].append(index[v])
    return len(maximum_matching(adjacency, len(left)))


__all__ = ["Arena", "DEFAULT_ROUNDS", "Introduction", "LAST_STAGE", "Play", "RuleFunction"]
