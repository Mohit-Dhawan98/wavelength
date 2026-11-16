"""One simulated pool: the hidden truth and the matcher's view of it, from one seed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wavelength.decisions import Decisions, decide
from wavelength.observation import Observations, observe
from wavelength.population import Population, generate_population
from wavelength.scenario import Scenario


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
