"""Score many rules against the same worlds, the way Axelrod scored strategies.

Every rule plays every world for several seeds. Within one world and seed
all rules face the same people, the same answer key and the same luck, so
differences in score are differences between rules. The leaderboard gives
each rule's mean share of the best possible and how much it moved from
seed to seed; a lead smaller than that spread is not a lead.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Sequence

from wavelength.engine import DEFAULT_ROUNDS, Arena
from wavelength.key import LAST_STAGE
from wavelength.play import DEFAULT_PEOPLE, Rule, load_rules, scorecard
from wavelength.presets import resolve_scenario
from wavelength.view import RuleError


def rules_from(path: str | Path) -> list[tuple[str, Rule]]:
    """Rules from one file or every ``.py`` file in a directory, as ``(file name, rule)``."""
    path = Path(path)
    files = sorted(path.glob("*.py")) if path.is_dir() else [path]
    found = [(file.name, rule) for file in files for rule in load_rules(file)]
    if not found:
        raise ValueError(f"no rule files found in {path}")
    names = [rule.name for _, rule in found]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"two rules share a name: {repeated}")
    return found


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": round(fmean(values), 4),
        "spread": round(pstdev(values), 4) if len(values) > 1 else 0.0,
        "low": round(min(values), 4),
        "high": round(max(values), 4),
    }


def run_tournament(
    rules: Sequence[tuple[str, Rule]],
    worlds: Sequence[str],
    *,
    seeds: int = 5,
    people: int | None = DEFAULT_PEOPLE,
    rounds: int = DEFAULT_ROUNDS,
) -> dict[str, Any]:
    """Play every rule in every world for ``seeds`` seeds and build the leaderboard.

    A rule that fails in a world is recorded with its error there and
    scores zero for it, so one bad entry cannot stop a tournament.
    """
    if seeds < 1:
        raise ValueError("seeds must be at least 1")
    scores: dict[str, dict[str, list[dict[str, Any]]]] = {r.name: {} for _, r in rules}
    errors: dict[str, dict[str, str]] = {r.name: {} for _, r in rules}
    benchmarks: dict[str, dict[str, Any]] = {}
    for world in worlds:
        world_name, scenario = resolve_scenario(world)
        if people is not None:
            scenario = replace(scenario, pool_size=people)
        best, informed = [], []
        for seed in range(seeds):
            arena = Arena(scenario, seed)
            best.append(arena.best_possible(LAST_STAGE))
            informed_couples = arena.couples(arena.informed_play(rounds), LAST_STAGE)
            informed.append(informed_couples / best[-1] if best[-1] else 0.0)
            for _, rule in rules:
                if world_name in errors[rule.name]:
                    continue
                try:
                    card = scorecard(arena, rule.name, arena.play(rule, rounds), world_name)
                except RuleError as problem:
                    errors[rule.name][world_name] = str(problem)
                except Exception as problem:  # a submitted rule may fail in any way
                    errors[rule.name][world_name] = f"{type(problem).__name__}: {problem}"
                else:
                    scores[rule.name].setdefault(world_name, []).append(card)
        benchmarks[world_name] = {
            "best_possible_couples": _summary(best),
            "informed_benchmark": _summary(informed),
        }

    world_names = list(benchmarks)
    table = []
    for file, rule in rules:
        by_world: dict[str, Any] = {}
        for world_name in world_names:
            if world_name in errors[rule.name]:
                by_world[world_name] = {"error": errors[rule.name][world_name]}
                continue
            cards = scores[rule.name][world_name]
            by_world[world_name] = {
                "score": _summary([c["score"] for c in cards]),
                "stages": {
                    str(k): _summary([c["stages"][k - 1]["share_of_best"] for c in cards])
                    for k in range(1, LAST_STAGE + 1)
                },
                "gave_up": _summary([c["people_outcomes"]["gave_up"] for c in cards]),
                "never_introduced": _summary(
                    [c["people_outcomes"]["never_introduced"] for c in cards]
                ),
                "known_dealbreaker_violations": _summary(
                    [c["introductions"]["known_dealbreaker_violations"] for c in cards]
                ),
            }
        means = [
            by_world[w]["score"]["mean"] if "score" in by_world[w] else 0.0 for w in world_names
        ]
        table.append(
            {
                "rule": rule.name,
                "file": file,
                "description": rule.description,
                "overall": round(fmean(means), 4),
                "worlds": by_world,
            }
        )
    table.sort(key=lambda row: (-row["overall"], row["rule"]))
    return {
        "score": "couples who both want to meet again, as a share of the most possible",
        "worlds": world_names,
        "seeds": seeds,
        "people": people,
        "rounds": rounds,
        "benchmarks": benchmarks,
        "leaderboard": table,
    }


def format_leaderboard(result: dict[str, Any]) -> str:
    """The leaderboard as a plain-text table: mean score, with spread across seeds."""
    worlds = result["worlds"]
    header = ["rule", "overall", *worlds]
    rows = []
    for row in result["leaderboard"]:
        cells = [row["rule"], f"{row['overall']:.1%}"]
        for world in worlds:
            entry = row["worlds"][world]
            cells.append(
                "error"
                if "error" in entry
                else f"{entry['score']['mean']:.1%} ±{entry['score']['spread']:.1%}"
            )
        rows.append(cells)
    benchmark = ["(perfect read of profiles)", ""]
    benchmark += [f"{result['benchmarks'][w]['informed_benchmark']['mean']:.1%}" for w in worlds]
    rows.append(benchmark)
    widths = [max(len(r[i]) for r in (header, *rows)) for i in range(len(header))]
    lines = [
        "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths)))
        for r in (header, *rows)
    ]
    lines += [
        "",
        f"Score: {result['score']}.",
        f"Mean over {result['seeds']} seeds, ± spread across seeds; "
        f"{result['people']} people, {result['rounds']} rounds.",
        "Simulated people under each world's stated assumptions; not evidence about real users.",
    ]
    for row in result["leaderboard"]:
        for world, entry in row["worlds"].items():
            if "error" in entry:
                lines.append(f"{row['rule']} failed in {world}: {entry['error']}")
    return "\n".join(lines)
