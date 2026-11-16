"""Write a matching rule in a few lines and score it against a hidden answer key.

    from wavelength.play import rule

    @rule("vibes first")
    def vibes_first(v):
        s = 0.5 * v.text + 0.3 * v.voice + 0.2 * v.features()
        return v.pair_mutual(s, top=10)

A rule is a function of one argument, a ``View`` of what the app can see
this round, returning the pairs to introduce. ``run`` plays it for several
rounds in a seeded world and returns a scorecard: how many couples it
formed at each stage as a share of the most that anyone could have formed.

The world is made, and every pair's fate decided, before the rule is
called. The rule cannot see any of that and cannot change it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from wavelength.engine import DEFAULT_ROUNDS, Arena, Play, RuleFunction
from wavelength.key import LAST_STAGE, STAGES
from wavelength.presets import resolve_scenario
from wavelength.scenario import Scenario
from wavelength.view import RuleError, View

DEFAULT_PEOPLE = 600

__all__ = ["Rule", "RuleError", "View", "load_rules", "rule", "run", "scorecard"]


@dataclass(frozen=True)
class Rule:
    """A named matching rule. Call it with a ``View`` to get pairs."""

    name: str
    function: RuleFunction
    description: str = ""

    def __call__(self, view: View) -> object:
        return self.function(view)


def rule(name: str) -> Callable[[RuleFunction], Rule]:
    """Decorator: give a function a name and make it a rule.

    The function's docstring, if it has one, becomes the rule's description.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError('a rule needs a name: @rule("my rule")')

    def wrap(function: RuleFunction) -> Rule:
        if not callable(function):
            raise TypeError("@rule goes on a function that takes a view and returns pairs")
        return Rule(name.strip(), function, (function.__doc__ or "").strip())

    return wrap


def rules_in(namespace: Mapping[str, Any]) -> list[Rule]:
    """Every rule defined in a namespace, in the order they were defined."""
    found: list[Rule] = []
    for value in namespace.values():
        if isinstance(value, Rule) and value not in found:
            found.append(value)
    return found


def load_rules(path: str | Path) -> list[Rule]:
    """Run a rule file and return the rules it defines."""
    path = Path(path)
    namespace: dict[str, Any] = {"__name__": f"rule_file_{path.stem}", "__file__": str(path)}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    found = rules_in(namespace)
    if not found:
        raise ValueError(f'{path}: no rule found; define one with @rule("name")')
    return found


def _scenario_for(world: str | Scenario, people: int | None) -> tuple[str, Scenario]:
    name, scenario = ("custom", world) if isinstance(world, Scenario) else resolve_scenario(world)
    return name, scenario if people is None else replace(scenario, pool_size=people)


def _share(count: int, best: int) -> float:
    return round(count / best, 4) if best else 0.0


def scorecard(
    arena: Arena, rule_name: str, play: Play, world_name: str = "custom"
) -> dict[str, Any]:
    """Grade one run against the arena's answer key, as plain JSON-ready data."""
    rounds = play.rounds
    stages = range(1, LAST_STAGE + 1)
    best = {k: arena.best_possible(k) for k in stages}
    couples = {k: arena.couples(play, k) for k in stages}
    informed = {k: arena.couples(arena.informed_play(rounds), k) for k in stages}
    made = play.introductions
    both_answered = [i for i in made if i.a_answered and i.b_answered]
    introduced = {person for i in made for person in (i.a, i.b)}
    cast = arena.cast()
    followed = set(cast)

    per_round = []
    for week in range(rounds):
        this_week = [i for i in made if i.round == week]
        per_round.append(
            {
                "round": week + 1,
                "introductions": len(this_week),
                "answered": sum(i.a_answered and i.b_answered for i in this_week),
                "both_said_yes": sum(i.stage >= 1 for i in this_week),
                "hit_it_off": sum(i.stage >= LAST_STAGE for i in this_week),
                "gave_up": sum(r == week for r in play.gave_up_in.values()),
            }
        )

    return {
        "rule": rule_name,
        "world": world_name,
        "seed": arena.seed,
        "people": arena.n,
        "rounds": rounds,
        "score": _share(couples[LAST_STAGE], best[LAST_STAGE]),
        "stages": [
            {
                "stage": k,
                "meaning": STAGES[k],
                "couples": couples[k],
                "best_possible": best[k],
                "share_of_best": _share(couples[k], best[k]),
                "informed_benchmark": informed[k],
                "informed_share_of_best": _share(informed[k], best[k]),
            }
            for k in stages
        ],
        "introductions": {
            "made": len(made),
            "answered": len(both_answered),
            "expired": len(made) - len(both_answered),
            "wasted": sum(i.stage == 0 for i in made),
            "one_sided": sum(i.a_yes != i.b_yes for i in made),
            "known_dealbreaker_violations": sum(not i.known_ok for i in made),
        },
        "people_outcomes": {
            "hit_it_off": 2 * couples[LAST_STAGE],
            "left_as_a_couple": len(play.left_in),
            "gave_up": len(play.gave_up_in),
            "never_introduced": arena.n - len(introduced),
        },
        "per_round": per_round,
        "cast": [arena.describe_person(person) for person in cast],
        "fates": {
            "gave_up": {str(p): week + 1 for p, week in sorted(play.gave_up_in.items())},
            "left_as_couple": {str(p): week + 1 for p, week in sorted(play.left_in.items())},
        },
        "events": [
            {
                "round": i.round + 1,
                "a": i.a,
                "b": i.b,
                "a_answered": i.a_answered,
                "b_answered": i.b_answered,
                "a_yes": i.a_yes,
                "b_yes": i.b_yes,
                "stage": i.stage,
                "known_ok": i.known_ok,
            }
            for i in made
            if i.a in followed or i.b in followed
        ],
    }


def run(
    rule: Rule | RuleFunction,
    world: str | Scenario = "today",
    seed: int = 0,
    people: int | None = DEFAULT_PEOPLE,
    rounds: int = DEFAULT_ROUNDS,
) -> dict[str, Any]:
    """Play one rule in one seeded world and return its scorecard.

    ``world`` is a preset name, a scenario file or a ``Scenario``.
    ``people`` overrides the pool size (``None`` keeps the scenario's).
    """
    world_name, scenario = _scenario_for(world, people)
    arena = Arena(scenario, seed)
    name = rule.name if isinstance(rule, Rule) else getattr(rule, "__name__", "rule")
    return scorecard(arena, name, arena.play(rule, rounds), world_name)
