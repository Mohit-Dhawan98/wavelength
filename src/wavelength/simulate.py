"""Run the article's three strategies on the same simulated people and keep score.

Each strategy sees only ``Observations`` and proposes one round of
introductions. The hidden truth then says who would have accepted whom.
Because every decision was drawn once, before any strategy ran, the same
introduction has the same outcome whichever strategy proposes it.

What comes out describes the scenario that went in. It is not evidence
about real people and it is not the article's production measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from wavelength.baselines import one_directional_greedy, similarity_scores
from wavelength.decisions import Decisions, decide
from wavelength.features import FEATURE_NAMES
from wavelength.filters import eligibility_matrix
from wavelength.matching import TOP_N, bidirectional_match
from wavelength.observation import Observations, observe
from wavelength.population import Population, generate_population
from wavelength.scenario import Scenario
from wavelength.scoring import score_matrix
from wavelength.weights import GLOBAL_WEIGHTS

DEFAULT_TRIALS = 5


@dataclass(frozen=True, eq=False)
class World:
    """One simulated pool: the hidden truth and the matcher's view of it."""

    population: Population
    decisions: Decisions
    observations: Observations


def generate_world(seed: int, scenario: Scenario) -> World:
    """Build one world from a seed. The same seed and scenario give the same world."""
    truth_rng, decision_rng, observation_rng = np.random.default_rng(seed).spawn(3)
    population = generate_population(truth_rng, scenario)
    return World(
        population=population,
        decisions=decide(decision_rng, scenario, population),
        observations=observe(observation_rng, scenario, population),
    )


@dataclass(frozen=True)
class Proposal:
    """One proposed introduction. ``predicted`` is P(mutual) when the strategy has one."""

    a: int
    b: int
    predicted: float | None = None


#: A strategy maps (observations, visiting order, top_n) to one round of proposals.
Strategy = Callable[[Observations, np.ndarray, int], list[Proposal]]


def similarity_only(obs: Observations, order: np.ndarray, top_n: int) -> list[Proposal]:
    """Stage 1: text similarity, each user's best available. No features, no hard filters.

    The only restriction is the side of the market, which the article's
    Stage 1 also applied as a metadata filter.
    """
    pairs = one_directional_greedy(similarity_scores(obs.text_embeddings), obs.across, order)
    return [Proposal(a, b) for a, b in pairs]


def filtered_global(obs: Observations, order: np.ndarray, top_n: int) -> list[Proposal]:
    """Stage 2: hard filters, then tri-signal scores with global weights, one-directional."""
    weights = np.tile([GLOBAL_WEIGHTS[f] for f in FEATURE_NAMES], (obs.codes.shape[0], 1))
    scores = score_matrix(obs.codes, weights, obs.text_embeddings, obs.visual_embeddings)
    eligible = eligibility_matrix(obs.codes) & obs.across
    return [Proposal(a, b) for a, b in one_directional_greedy(scores, eligible, order)]


def bidirectional(obs: Observations, order: np.ndarray, top_n: int) -> list[Proposal]:
    """Stage 3: hard filters, per-user weights, mutual top-N, assignment by P(mutual)."""
    scores = score_matrix(obs.codes, obs.weights, obs.text_embeddings, obs.visual_embeddings)
    eligible = eligibility_matrix(obs.codes) & obs.across
    assigned = bidirectional_match(scores, eligible, top_n)
    return [Proposal(p.a, p.b, p.probability) for p in assigned]


#: The three strategies, keyed by a short name, with the label used in reports.
STRATEGIES: dict[str, tuple[str, Strategy]] = {
    "similarity": ("1 similarity only", similarity_only),
    "filtered_global": ("2 features + hard filters, global weights", filtered_global),
    "bidirectional": ("3 per-user weights + bidirectional", bidirectional),
}

#: Strategies that apply the hard filters, so must never propose a violating pair.
FILTERED_STRATEGIES: frozenset[str] = frozenset({"filtered_global", "bidirectional"})

#: Edges of the predicted-probability bins in the calibration table. The
#: lowest possible P(mutual) for a mutual top-N pair is 0.5 x 0.5 = 0.25.
CALIBRATION_EDGES: tuple[float, ...] = (0.25, 0.40, 0.55, 0.70, 0.85, 1.0)


@dataclass
class StrategyStats:
    """Outcome counts for one strategy, pooled over trials."""

    key: str
    label: str
    users: int = 0
    pairs: int = 0
    mutual: int = 0
    one_sided: int = 0
    violations: int = 0

    def _per_pair(self, count: int) -> float:
        return count / self.pairs if self.pairs else 0.0

    @property
    def mutual_rate(self) -> float:
        """Share of proposed pairs accepted by both people."""
        return self._per_pair(self.mutual)

    @property
    def one_sided_rate(self) -> float:
        """Share of proposed pairs accepted by exactly one person."""
        return self._per_pair(self.one_sided)

    @property
    def violation_rate(self) -> float:
        """Share of proposed pairs that break a hard filter on the true values."""
        return self._per_pair(self.violations)

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
    """Proposals whose predicted P(mutual) fell in one range, and how they turned out."""

    low: float
    high: float
    pairs: int = 0
    predicted_sum: float = 0.0
    mutual: int = 0

    @property
    def mean_predicted(self) -> float:
        return self.predicted_sum / self.pairs if self.pairs else float("nan")

    @property
    def observed(self) -> float:
        return self.mutual / self.pairs if self.pairs else float("nan")


@dataclass
class SimulationReport:
    """Everything one run produced, for one scenario."""

    scenario_name: str
    scenario: Scenario
    seed: int
    trials: int
    top_n: int
    strategies: dict[str, StrategyStats]
    calibration: list[CalibrationBin] = field(default_factory=list)

    @property
    def calibration_error(self) -> float:
        """Pair-weighted mean gap between predicted and observed mutual acceptance."""
        filled = [b for b in self.calibration if b.pairs]
        if not filled:
            return float("nan")
        gap = sum(b.pairs * abs(b.mean_predicted - b.observed) for b in filled)
        return gap / sum(b.pairs for b in filled)


def _bin_for(bins: list[CalibrationBin], predicted: float) -> CalibrationBin:
    for candidate in bins[:-1]:
        if predicted < candidate.high:
            return candidate
    return bins[-1]


def check_round(proposals: list[Proposal]) -> None:
    """A round may use each user at most once; a strategy that breaks this is a bug."""
    seen = [user for p in proposals for user in (p.a, p.b)]
    if len(seen) != len(set(seen)):
        raise AssertionError("a strategy matched the same user twice in one round")


def trial_seeds(seed: int, trial: int) -> tuple[int, int]:
    """The (world seed, visiting-order seed) for one trial of a run."""
    world_seed, order_seed = np.random.SeedSequence([seed, trial]).generate_state(2)
    return int(world_seed), int(order_seed)


def run_simulation(
    scenario: Scenario,
    *,
    seed: int = 0,
    trials: int = DEFAULT_TRIALS,
    top_n: int = TOP_N,
    users: int | None = None,
    name: str = "custom",
) -> SimulationReport:
    """Run every strategy over ``trials`` independent pools and pool the counts.

    ``users`` overrides the scenario's ``pool_size`` for this run.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if users is not None:
        scenario = replace(scenario, pool_size=users)
    stats = {key: StrategyStats(key, label) for key, (label, _) in STRATEGIES.items()}
    bins = [CalibrationBin(lo, hi) for lo, hi in zip(CALIBRATION_EDGES, CALIBRATION_EDGES[1:])]

    for trial in range(trials):
        world_seed, order_seed = trial_seeds(seed, trial)
        world = generate_world(world_seed, scenario)
        accepts = world.decisions.accepts
        truly_eligible = eligibility_matrix(world.population.codes)
        # One visiting order per world, shared by the one-directional strategies.
        order = np.random.default_rng(order_seed).permutation(scenario.pool_size)
        for key, (_, strategy) in STRATEGIES.items():
            proposals = strategy(world.observations, order, top_n)
            check_round(proposals)
            tally = stats[key]
            tally.users += scenario.pool_size
            for p in proposals:
                a_yes, b_yes = bool(accepts[p.a, p.b]), bool(accepts[p.b, p.a])
                tally.pairs += 1
                tally.mutual += a_yes and b_yes
                tally.one_sided += a_yes != b_yes
                tally.violations += not truly_eligible[p.a, p.b]
                if p.predicted is not None:
                    bucket = _bin_for(bins, p.predicted)
                    bucket.pairs += 1
                    bucket.predicted_sum += p.predicted
                    bucket.mutual += a_yes and b_yes

    return SimulationReport(name, scenario, seed, trials, top_n, stats, bins)
