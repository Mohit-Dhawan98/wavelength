"""Scenarios as plain data: dictionaries, TOML or JSON files, and ``--set`` overrides.

A scenario has one plain-data form, used everywhere:

- top-level keys are ``Scenario`` field names;
- ``archetypes`` is a table keyed by archetype name;
- anything left out keeps the value of the scenario being built on.

The presets are written in this form, so a scenario file looks exactly like
a preset. Two rules keep partial overrides predictable:

- In a file or dictionary, a distribution (shares that must sum to 1) is
  replaced whole. Writing ``relationship_timeline = {casual = 1.0}`` means
  exactly that, not "merge with what was there".
- With ``--set``, which changes one number at a time, setting one share of a
  distribution rescales its siblings proportionally so the total stays 1.
  ``--set archetypes.casual.share=0.5`` shrinks the other archetypes to fit.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from wavelength.scenario import DEALBREAKER_FEATURES, Archetype, Scenario, scenario_fields

_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in scenario_fields())
_INT_FIELDS = frozenset(f.name for f in scenario_fields() if f.type == "int")


def to_dict(scenario: Scenario) -> dict[str, Any]:
    """The scenario as nested dictionaries of plain values."""
    data: dict[str, Any] = {}
    for name in _FIELD_NAMES:
        value = getattr(scenario, name)
        if name == "archetypes":
            data[name] = {
                a.name: {
                    "description": a.description,
                    "share": a.share,
                    **{feature: dict(a.tendency(feature)) for feature in DEALBREAKER_FEATURES},
                }
                for a in value
            }
        elif name == "trait_shares":
            data[name] = {feature: dict(shares) for feature, shares in value.items()}
        elif isinstance(value, Mapping):
            data[name] = dict(value)
        else:
            data[name] = value
    return data


def _merge_archetypes(current: dict[str, Any], changes: Mapping[str, Any]) -> tuple[Archetype, ...]:
    """Apply per-archetype changes; a new name adds an archetype."""
    if not isinstance(changes, Mapping):
        raise ValueError("archetypes must be a table keyed by archetype name")
    merged = {name: dict(spec) for name, spec in current.items()}
    for name, spec in changes.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"archetypes.{name} must be a table")
        allowed = {"description", "share", *DEALBREAKER_FEATURES}
        unknown = sorted(set(spec) - allowed)
        if unknown:
            raise ValueError(
                f"archetypes.{name}: unknown key(s) {unknown}; expected {sorted(allowed)}"
            )
        merged.setdefault(name, {"description": ""}).update(spec)
    archetypes = []
    for name, spec in merged.items():
        missing = sorted({"share", *DEALBREAKER_FEATURES} - set(spec))
        if missing:
            raise ValueError(f"archetypes.{name}: a new archetype also needs {missing}")
        archetypes.append(Archetype(name=name, **spec))
    return tuple(archetypes)


#: Fields that were renamed, so an old file or command gets a useful message.
_RENAMED = {"pickiness": "yes_rate", "pickiness_spread": "yes_rate_spread"}


def _renamed_hint(names: Sequence[str]) -> str:
    hints = [f"{old} is now {_RENAMED[old]}" for old in names if old in _RENAMED]
    return f" ({'; '.join(hints)})" if hints else ""


def from_dict(data: Mapping[str, Any], base: Scenario | None = None) -> Scenario:
    """Build a scenario from plain data, starting from ``base`` (default: ``Scenario()``)."""
    unknown = sorted(set(data) - set(_FIELD_NAMES))
    if unknown and _renamed_hint(unknown):
        raise ValueError(f"unknown scenario field(s) {unknown}{_renamed_hint(unknown)}")
    if unknown:
        raise ValueError(f"unknown scenario field(s) {unknown}; valid fields: {list(_FIELD_NAMES)}")
    current = to_dict(base if base is not None else Scenario())
    values: dict[str, Any] = {}
    for name in _FIELD_NAMES:
        if name not in data:
            values[name] = current[name]
        elif name == "archetypes":
            values[name] = data[name]
        elif name in ("trait_shares", "dealbreaker_shares"):
            if not isinstance(data[name], Mapping):
                raise ValueError(f"{name} must be a table")
            values[name] = {**current[name], **data[name]}
        else:
            values[name] = data[name]
    values["archetypes"] = _merge_archetypes(current["archetypes"], values["archetypes"])
    return Scenario(**values)


def read_scenario_file(path: str | Path) -> dict[str, Any]:
    """Read a ``.toml`` or ``.json`` scenario file into plain data, unvalidated."""
    path = Path(path)
    if path.suffix == ".toml":
        with path.open("rb") as handle:
            return tomllib.load(handle)
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: a scenario file must hold a table of fields")
        return data
    raise ValueError(f"{path}: scenario files must end in .toml or .json")


def _parse_value(text: str) -> Any:
    """Read a ``--set`` value: a number, true/false, JSON, or else a plain string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _is_distribution(path: Sequence[str]) -> bool:
    """True when the path names one share of a set that must sum to 1."""
    if path[0] == "archetypes":
        return len(path) == 4 or (len(path) == 3 and path[2] == "share")
    if path[0] == "trait_shares":
        return len(path) == 3
    return path[0] == "attraction_mix" and len(path) == 2


def _rescale_siblings(key: str, siblings: dict[str, float], own: str, value: Any) -> None:
    """Scale the other shares so that, with ``own`` fixed at ``value``, the total is 1."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{key}: a share must be a number between 0 and 1, got {value!r}")
    others = [name for name in siblings if name != own]
    rest = sum(siblings[name] for name in others)
    if rest <= 0.0:
        if value < 1.0:
            raise ValueError(f"{key}: the other shares are all zero, so they cannot be rescaled")
        return
    for name in others:
        siblings[name] *= (1.0 - value) / rest


def _set_path(data: dict[str, Any], key: str, value: Any) -> None:
    path = key.split(".")
    node: Any = data
    for depth, part in enumerate(path[:-1]):
        if not isinstance(node, dict) or part not in node:
            where = ".".join(path[:depth]) or "the scenario"
            options = sorted(node) if isinstance(node, dict) else []
            raise ValueError(f"--set {key}: {part!r} is not in {where}; valid here: {options}")
        node = node[part]
    leaf = path[-1]
    if not isinstance(node, dict) or leaf not in node:
        options = sorted(node) if isinstance(node, dict) else []
        if leaf in _RENAMED:
            raise ValueError(f"--set {key}: {leaf} is now {_RENAMED[leaf]}")
        raise ValueError(f"--set {key}: {leaf!r} is not a valid key; valid here: {options}")

    if path[0] == "archetypes" and len(path) == 3 and leaf == "share":
        shares = {name: spec["share"] for name, spec in data["archetypes"].items()}
        _rescale_siblings(key, shares, path[1], value)
        for name, spec in data["archetypes"].items():
            spec["share"] = shares[name]
    elif _is_distribution(path):
        _rescale_siblings(key, node, leaf, value)
    if len(path) == 1 and leaf in _INT_FIELDS and isinstance(value, float) and value.is_integer():
        value = int(value)
    node[leaf] = value


def with_overrides(scenario: Scenario, assignments: Sequence[str]) -> Scenario:
    """Apply ``key=value`` overrides, as given to ``--set``, in order.

    Keys are dotted paths into the plain-data form: ``yes_rate``,
    ``dealbreaker_shares.family_plans``, ``archetypes.casual.share``,
    ``archetypes.casual.relationship_timeline.casual``.
    """
    data = to_dict(scenario)
    for assignment in assignments:
        key, separator, text = assignment.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"--set expects key=value, got {assignment!r}")
        _set_path(data, key.strip(), _parse_value(text.strip()))
    return from_dict(data, base=scenario)
