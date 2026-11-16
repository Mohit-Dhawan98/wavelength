"""A plain-data surface for a browser front end.

Everything here takes and returns JSON-ready values (dicts, lists, strings,
numbers) and never raises for bad input: a broken rule or an impossible
setting comes back as ``{"error": {...}}``. Nothing in this module reads or
writes files beyond the rule files shipped with the package, and nothing
uses the network, threads or anything outside numpy and the standard
library, so it runs unchanged under Pyodide.
"""

from __future__ import annotations

import builtins
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from wavelength.engine import DEFAULT_ROUNDS, Arena
from wavelength.play import DEFAULT_PEOPLE, Rule, rule, rules_in, scorecard
from wavelength.presets import PRESETS
from wavelength.scenario import Scenario, scenario_fields
from wavelength.scenario_io import to_dict, with_values
from wavelength.view import RuleError

#: The name a submitted rule's source is compiled under, so errors can be traced to its lines.
RULE_FILE = "<rule>"

#: The largest pool a browser run will accept.
MAX_PEOPLE = 2000

#: Mapping-valued settings exposed as one slider per entry, and whether they must sum to 1.
_SLIDER_GROUPS = {
    "dealbreaker_shares": False,
    "attraction_mix": True,
    "community_shares": True,
}


def _error(message: str, *, where: str, line: int | None = None) -> dict[str, Any]:
    return {"error": {"where": where, "line": line, "message": message}}


def _label(name: str) -> str:
    return name.replace("_", " ").replace(".", ": ")


def describe_settings() -> list[dict[str, Any]]:
    """Every world setting, with enough to draw a control for it.

    Each entry has ``key`` (the name to send back), ``group``, ``label``,
    ``meaning``, ``sensible``, ``basis``, ``type``, ``min``, ``max``,
    ``step`` and ``default``. Shares that must sum to 1 carry ``sums_with``,
    the key prefix they share; setting one rescales the others.
    """
    defaults = to_dict(Scenario())
    described: list[dict[str, Any]] = []
    for f in scenario_fields():
        about = {
            "group": f.metadata["group"],
            "meaning": f.metadata["meaning"],
            "sensible": f.metadata["sensible"],
            "basis": f.metadata["basis"],
        }
        if f.type in ("float", "int"):
            whole = f.type == "int"
            described.append(
                {
                    "key": f.name,
                    "label": _label(f.name),
                    **about,
                    "type": "integer" if whole else "number",
                    "min": f.metadata["low"],
                    "max": min(f.metadata["high"], MAX_PEOPLE)
                    if f.name == "pool_size"
                    else f.metadata["high"],
                    "step": 1 if whole else 0.01,
                    "default": defaults[f.name],
                }
            )
        elif f.name == "archetypes":
            for name, spec in defaults["archetypes"].items():
                described.append(
                    {
                        "key": f"archetypes.{name}.share",
                        "label": f"share of people who are {_label(name)}",
                        **about,
                        "meaning": f"{spec['description']} {about['meaning']}",
                        "type": "number",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "default": spec["share"],
                        "sums_with": "archetypes",
                    }
                )
        elif f.name in _SLIDER_GROUPS:
            for name, value in defaults[f.name].items():
                entry = {
                    "key": f"{f.name}.{name}",
                    "label": f"{_label(f.name)}: {_label(name)}",
                    **about,
                    "type": "number",
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "default": value,
                }
                if _SLIDER_GROUPS[f.name]:
                    entry["sums_with"] = f.name
                described.append(entry)
    return described


def _slider_values(scenario: Scenario) -> dict[str, Any]:
    data = to_dict(scenario)
    values: dict[str, Any] = {}
    for entry in describe_settings():
        node: Any = data
        for part in entry["key"].split("."):
            node = node[part]
        values[entry["key"]] = node
    return values


def list_presets() -> list[dict[str, Any]]:
    """The preset worlds: name, product, description, and the value of every setting."""
    return [
        {
            "name": p.name,
            "product": p.product,
            "summary": p.summary,
            "description": p.description,
            "changes": list(p.overrides),
            "reasons": dict(p.reasons),
            "settings": _slider_values(p.scenario),
        }
        for p in PRESETS.values()
    ]


def _rules_directory() -> Path | None:
    here = Path(__file__).resolve().parent
    for candidate in (here / "_rules", here.parent.parent / "rules"):
        if candidate.is_dir():
            return candidate
    return None


def list_rules() -> list[dict[str, str]]:
    """The demo rules shipped with the package: name, description and source text."""
    directory = _rules_directory()
    listed = []
    for path in sorted(directory.glob("*.py")) if directory else []:
        source = path.read_text(encoding="utf-8")
        for found in _compile(source):
            listed.append({"name": found.name, "description": found.description, "source": source})
    return listed


def _compile(source: str) -> list[Rule]:
    """Run rule source in a fresh namespace holding only ``rule``, ``np`` and the builtins."""
    namespace: dict[str, Any] = {
        "__builtins__": builtins,
        "__name__": "submitted_rule",
        "rule": rule,
        "np": np,
    }
    exec(compile(source, RULE_FILE, "exec"), namespace)
    return rules_in(namespace)


def _line_in_rule(problem: BaseException) -> int | None:
    """The last line of the submitted source that an exception passed through."""
    if isinstance(problem, SyntaxError) and problem.filename == RULE_FILE:
        return problem.lineno
    lines = [
        f.lineno for f in traceback.extract_tb(problem.__traceback__) if f.filename == RULE_FILE
    ]
    return lines[-1] if lines else None


def _resolve_rule(rule_source_or_name: str) -> Rule | dict[str, Any]:
    if not isinstance(rule_source_or_name, str) or not rule_source_or_name.strip():
        return _error("give a rule's name or its source text", where="rule")
    for listed in list_rules():
        if listed["name"] == rule_source_or_name.strip():
            return _compile(listed["source"])[0]
    try:
        found = _compile(rule_source_or_name)
    except Exception as problem:  # anything a visitor can type
        message = f"{type(problem).__name__}: {problem}"
        if isinstance(problem, SyntaxError):
            message = f"SyntaxError: {problem.msg}"
        return _error(message, where="rule", line=_line_in_rule(problem))
    if not found:
        return _error('no rule found; define one with @rule("name")', where="rule")
    return found[0]


def _resolve_world(
    settings: Mapping[str, Any] | None, people: int | None
) -> tuple[str, Scenario] | dict[str, Any]:
    values = dict(settings or {})
    name = str(values.pop("preset", "today"))
    if name not in PRESETS:
        return _error(f"unknown preset {name!r}; choose from {list(PRESETS)}", where="settings")
    try:
        scenario = with_values(PRESETS[name].scenario, values)
        if people is not None:
            if not 20 <= int(people) <= MAX_PEOPLE:
                raise ValueError(f"people must be between 20 and {MAX_PEOPLE}, got {people}")
            scenario = replace(scenario, pool_size=int(people))
    except (ValueError, TypeError, KeyError) as problem:
        return _error(str(problem), where="settings")
    return (name if not values else f"{name} (modified)"), scenario


def _play(arena: Arena, chosen: Rule, rounds: int, world_name: str) -> dict[str, Any]:
    try:
        return scorecard(arena, chosen.name, arena.play(chosen, rounds), world_name)
    except RuleError as problem:
        return _error(str(problem), where="rule")
    except Exception as problem:  # a submitted rule may fail in any way
        return _error(
            f"{type(problem).__name__}: {problem}", where="rule", line=_line_in_rule(problem)
        )


def run(
    settings: Mapping[str, Any] | None = None,
    rule_source_or_name: str = "the article's rule",
    seed: int = 0,
    people: int | None = DEFAULT_PEOPLE,
    rounds: int = DEFAULT_ROUNDS,
) -> dict[str, Any]:
    """Play one rule in one world and return its scorecard, or ``{"error": ...}``.

    ``settings`` is ``{"preset": name, key: value, ...}`` with keys from
    ``describe_settings``. ``rule_source_or_name`` is the name of a demo
    rule or the text of a rule file.
    """
    result = compare(settings, [rule_source_or_name], seed, people, rounds)
    return result if "error" in result else result["results"][0]


def compare(
    settings: Mapping[str, Any] | None,
    rules: Sequence[str],
    seed: int = 0,
    people: int | None = DEFAULT_PEOPLE,
    rounds: int = DEFAULT_ROUNDS,
) -> dict[str, Any]:
    """Play several rules against the same world, people and luck.

    Returns ``{"world": ..., "results": [...]}`` with one scorecard or error
    per rule, in the order given. A problem with the settings is a single
    top-level error.
    """
    world = _resolve_world(settings, people)
    if isinstance(world, dict):
        return world
    world_name, scenario = world
    try:
        arena = Arena(scenario, int(seed))
        if not 1 <= int(rounds) <= 52:
            raise ValueError(f"rounds must be between 1 and 52, got {rounds}")
    except (ValueError, TypeError) as problem:
        return _error(str(problem), where="settings")
    results = []
    for entry in rules:
        chosen = _resolve_rule(entry)
        results.append(
            chosen if isinstance(chosen, dict) else _play(arena, chosen, int(rounds), world_name)
        )
    return {"world": world_name, "seed": int(seed), "people": arena.n, "results": results}
