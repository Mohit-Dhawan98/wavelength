"""One simulated pool: the hidden truth, its answer key, and what the app can see."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelength.decisions import Decisions, decide
from wavelength.key import AnswerKey, build_key
from wavelength.observation import Observations, observe
from wavelength.population import Population, generate_population
from wavelength.scenario import Scenario


@dataclass(frozen=True, eq=False)
class World:
    """One simulated pool.

    ``population``, ``decisions`` and ``key`` are the hidden truth.
    ``observations`` is the only part a rule is ever built from.
    """

    seed: int
    population: Population
    decisions: Decisions
    key: AnswerKey
    observations: Observations


def generate_world(seed: int, scenario: Scenario) -> World:
    """Build one world from a seed. The same seed and scenario give the same world.

    Each part draws from its own stream, so changing what the app observes
    never changes who the people are or how any pair would turn out.
    """
    truth_rng, decision_rng, observation_rng, key_rng = np.random.default_rng(seed).spawn(4)
    population = generate_population(truth_rng, scenario)
    decisions = decide(decision_rng, scenario, population)
    return World(
        seed=seed,
        population=population,
        decisions=decisions,
        key=build_key(key_rng, scenario, population, decisions),
        observations=observe(observation_rng, scenario, population),
    )
