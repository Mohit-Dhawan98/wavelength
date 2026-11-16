"""Run the article's three strategies on the same simulated people and keep score.

Each strategy sees only ``Observations`` and proposes one round of
introductions. The hidden truth then says who would have accepted whom.
Because every decision was drawn once, before any strategy ran, the same
introduction has the same outcome whichever strategy proposes it.

Two things are kept apart in the score. Someone who never answers has not
said no, so acceptance is counted twice: over every proposed pair (what the
product delivers) and over pairs in which both people responded (what the
matcher can be held to).

With ``calibrate=True`` the run has two phases. A ``P(accept | rank)`` curve
is fitted on one set of pools, then every strategy is scored on different
pools, with the bidirectional matcher run once with the article's curve and
once with the fitted one.

What comes out describes the scenario that went in. It is not evidence
about real people and it is not the article's production measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from wavelength.calibration import RankOutcomes, fit_rank_curve
from wavelength.diagnostics import WorldCheck, check_world
from wavelength.matching import TOP_N, article_curve
from wavelength.observation import Observations
from wavelength.scenario import Scenario
from wavelength.strategies import (
    FILTERED_STRATEGIES,
    FITTED_KEY,
    FITTED_LABEL,
    STRATEGIES,
    Proposal,
    Strategy,
    bidirectional,
)
from wavelength.world import World, generate_world

__all__ = [
    "FILTERED_STRATEGIES",
    "FITTED_KEY",
    "STRATEGIES",
    "SimulationReport",
    "generate_world",
    "run_simulation",
]

DEFAULT_TRIALS = 5


#: Edges of the predicted-probability bins in the calibration tables.
CALIBRATION_EDGES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass
class StrategyStats:
    """Outcome counts for one strategy, pooled over trials."""

    key: str
    label: str
    users: int = 0
    pairs: int = 0
    responded: int = 0  # pairs in which both people answered
    mutual: int = 0  # pairs in which both answered yes
    one_sided: int = 0  # pairs with exactly one yes, whatever the other did
    violations: int = 0

    @staticmethod
    def _rate(count: int, out_of: int) -> float:
        return count / out_of if out_of else 0.0

    @property
    def mutual_rate(self) -> float:
        """Share of all proposed pairs accepted by both people."""
        return self._rate(self.mutual, self.pairs)

    @property
    def mutual_rate_responded(self) -> float:
        """Share accepted by both, among pairs in which both people responded."""
        return self._rate(self.mutual, self.responded)

    @property
    def one_sided_rate(self) -> float:
        """Share of all proposed pairs with exactly one yes."""
        return self._rate(self.one_sided, self.pairs)

    @property
    def expired_rate(self) -> float:
        """Share of all proposed pairs that expired because at least one person never answered."""
        return self._rate(self.pairs - self.responded, self.pairs)

    @property
    def violation_rate(self) -> float:
        """Share of proposed pairs that break a hard filter on the true values."""
        return self._rate(self.violations, self.pairs)

    @property
    def unproposed_share(self) -> float:
        """Share of users left without a proposal this round."""
        return 1.0 - 2 * self.pairs / self.users if self.users else 0.0

    @property
    def matches_per_100_users(self) -> float:
        """Mutually accepted pairs per 100 users: acceptance and coverage in one number."""
        return 100 * self.mutual / self.users if self.users else 0.0


@dataclass
class CalibrationBin:
    """Responded pairs whose predicted P(mutual) fell in one range, and how they turned out."""

    low: float
    high: float
    pairs: int = 0
    predicted_sum: float = 0.0
    mutual: int = 0
    majority_yes: int = 0  # the person on the larger side said yes
    minority_yes: int = 0  # the person on the scarce side said yes

    @property
    def mean_predicted(self) -> float:
        return self.predicted_sum / self.pairs if self.pairs else float("nan")

    @property
    def observed(self) -> float:
        return self.mutual / self.pairs if self.pairs else float("nan")

    @property
    def if_independent(self) -> float:
        """The both-accept rate these pairs would have if the two decisions were unrelated.

        The product of the two observed yes-rates in the bin. When
        ``observed`` is above it, the two people's decisions move together.
        """
        if not self.pairs:
            return float("nan")
        return (self.majority_yes / self.pairs) * (self.minority_yes / self.pairs)


@dataclass
class Calibration:
    """How one ``P(rank)`` curve's predictions compared with outcomes, on responded pairs."""

    name: str
    curve: tuple[float, ...]
    bins: list[CalibrationBin] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.bins:
            edges = zip(CALIBRATION_EDGES, CALIBRATION_EDGES[1:])
            self.bins = [CalibrationBin(low, high) for low, high in edges]

    def add(self, predicted: float, majority_yes: bool, minority_yes: bool) -> None:
        chosen = self.bins[-1]
        for candidate in self.bins[:-1]:
            if predicted < candidate.high:
                chosen = candidate
                break
        chosen.pairs += 1
        chosen.predicted_sum += predicted
        chosen.mutual += majority_yes and minority_yes
        chosen.majority_yes += majority_yes
        chosen.minority_yes += minority_yes

    @property
    def pairs(self) -> int:
        return sum(b.pairs for b in self.bins)

    @property
    def gap(self) -> float:
        """Pair-weighted mean distance between predicted and observed, over the bins."""
        filled = [b for b in self.bins if b.pairs]
        if not filled:
            return float("nan")
        total = sum(b.pairs * abs(b.mean_predicted - b.observed) for b in filled)
        return total / sum(b.pairs for b in filled)

    @property
    def independence_gap(self) -> float:
        """Pair-weighted mean of observed minus the independent product, over the bins."""
        filled = [b for b in self.bins if b.pairs]
        if not filled:
            return float("nan")
        total = sum(b.pairs * (b.observed - b.if_independent) for b in filled)
        return total / sum(b.pairs for b in filled)


@dataclass
class SimulationReport:
    """Everything one run produced, for one scenario."""

    scenario_name: str
    scenario: Scenario
    seed: int
    trials: int
    top_n: int
    strategies: dict[str, StrategyStats]
    calibrations: dict[str, Calibration]
    world: WorldCheck  # measured facts about the first pool
    fit_outcomes: RankOutcomes | None = None  # decisions the fitted curve was fitted on

    @property
    def calibrated(self) -> bool:
        return FITTED_KEY in self.strategies


def check_round(proposals: list[Proposal]) -> None:
    """A round may use each user at most once; a strategy that breaks this is a bug."""
    seen = [user for p in proposals for user in (p.a, p.b)]
    if len(seen) != len(set(seen)):
        raise AssertionError("a strategy matched the same user twice in one round")


def trial_seeds(seed: int, trial: int, fitting: bool = False) -> tuple[int, int]:
    """The (world seed, visiting-order seed) for one trial of a run.

    Fitting pools use a separate seed stream, so a curve is never checked on
    the pools it was fitted on.
    """
    entropy = [seed, trial, 1] if fitting else [seed, trial]
    world_seed, order_seed = np.random.SeedSequence(entropy).generate_state(2)
    return int(world_seed), int(order_seed)


def observe_rank_outcomes(
    scenario: Scenario, *, seed: int, trials: int, top_n: int = TOP_N
) -> RankOutcomes:
    """Run the bidirectional matcher on fitting pools and record decisions by rank.

    Only what a real system could log is used: for each introduction made,
    and each of its two people who responded, the rank they held the other
    at and whether they said yes.
    """
    outcomes = RankOutcomes(top_n)
    for trial in range(trials):
        world_seed, order_seed = trial_seeds(seed, trial, fitting=True)
        world = generate_world(world_seed, scenario)
        order = np.random.default_rng(order_seed).permutation(scenario.pool_size)
        decisions = world.decisions
        for p in bidirectional(world.observations, order, top_n):
            if decisions.responds[p.a]:
                outcomes.add(p.rank_of_b_for_a, decisions.would_accept[p.a, p.b])
            if decisions.responds[p.b]:
                outcomes.add(p.rank_of_a_for_b, decisions.would_accept[p.b, p.a])
    return outcomes


def _score(
    world: World, proposals: list[Proposal], tally: StrategyStats, calibration: Calibration | None
) -> None:
    """Look up how each proposal would really have gone and add it to the counts."""
    decisions, majority = world.decisions, world.population.majority
    truly_eligible = decisions.eligible  # proposals always cross sides
    tally.users += world.population.n
    responds, would_accept = decisions.responds, decisions.would_accept
    for p in proposals:
        a_yes = bool(responds[p.a] and would_accept[p.a, p.b])
        b_yes = bool(responds[p.b] and would_accept[p.b, p.a])
        both_responded = bool(responds[p.a] and responds[p.b])
        tally.pairs += 1
        tally.responded += both_responded
        tally.mutual += a_yes and b_yes
        tally.one_sided += a_yes != b_yes
        tally.violations += not truly_eligible[p.a, p.b]
        if calibration is not None and p.predicted is not None and both_responded:
            first_yes, second_yes = (a_yes, b_yes) if majority[p.a] else (b_yes, a_yes)
            calibration.add(p.predicted, first_yes, second_yes)


def run_simulation(
    scenario: Scenario,
    *,
    seed: int = 0,
    trials: int = DEFAULT_TRIALS,
    top_n: int = TOP_N,
    users: int | None = None,
    name: str = "custom",
    calibrate: bool = False,
) -> SimulationReport:
    """Run every strategy over ``trials`` independent pools and pool the counts.

    ``users`` overrides the scenario's ``pool_size`` for this run. With
    ``calibrate``, a ``P(rank)`` curve is first fitted on ``trials`` separate
    pools and the bidirectional matcher is scored a second time using it.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if users is not None:
        scenario = replace(scenario, pool_size=users)

    strategies: dict[str, tuple[str, Strategy]] = dict(STRATEGIES)
    calibrations = {"bidirectional": Calibration("the article's curve", article_curve(top_n))}
    fit_outcomes = None
    if calibrate:
        fit_outcomes = observe_rank_outcomes(scenario, seed=seed, trials=trials, top_n=top_n)
        fitted = fit_rank_curve(fit_outcomes)

        def with_fitted_curve(obs: Observations, order: np.ndarray, n: int) -> list[Proposal]:
            return bidirectional(obs, order, n, fitted)

        strategies[FITTED_KEY] = (FITTED_LABEL, with_fitted_curve)
        calibrations[FITTED_KEY] = Calibration("the fitted curve", fitted)

    stats = {key: StrategyStats(key, label) for key, (label, _) in strategies.items()}
    world_check = None
    for trial in range(trials):
        world_seed, order_seed = trial_seeds(seed, trial)
        world = generate_world(world_seed, scenario)
        if world_check is None:  # one pool of people is enough to describe the world
            world_check = check_world(world)
        # One visiting order per world, shared by the one-directional strategies.
        order = np.random.default_rng(order_seed).permutation(scenario.pool_size)
        for key, (_, strategy) in strategies.items():
            proposals = strategy(world.observations, order, top_n)
            check_round(proposals)
            _score(world, proposals, stats[key], calibrations.get(key))

    return SimulationReport(
        name,
        scenario,
        seed,
        trials,
        top_n,
        stats,
        calibrations,
        world_check,
        fit_outcomes,
    )
