"""Plain-text rendering: preset lists, scenario descriptions and simulation reports.

Kept apart from the simulation so the numbers can be used from Python
without going through text, and so wording changes never touch the model.
"""

from __future__ import annotations

import textwrap
from typing import Any, Callable, Mapping, Sequence

from wavelength.presets import PRESETS, Preset
from wavelength.scenario import DEALBREAKER_FEATURES, Scenario, scenario_fields
from wavelength.simulate import SimulationReport, StrategyStats

_WIDTH = 88
_CAVEAT = (
    "Simulated people under the scenario's stated assumptions.\n"
    "Not the article's production results, and not evidence about real users."
)


def _wrap(text: str, indent: str) -> list[str]:
    return textwrap.wrap(text, width=_WIDTH, initial_indent=indent, subsequent_indent=indent)


def _shares(shares: Mapping[str, float]) -> str:
    return ", ".join(f"{name} {share:.0%}" for name, share in shares.items() if share > 0)


def format_presets() -> str:
    """The list of presets with the world each one depicts."""
    lines: list[str] = []
    for preset in PRESETS.values():
        lines.append(f"{preset.name}: {preset.summary}")
        lines += _wrap(preset.description, "    ")
        if preset.overrides:
            lines += _wrap("changes from today: " + ", ".join(preset.overrides), "    ")
        lines.append("")
    lines.append("Run `wavelength describe <preset>` for every field, its value and its meaning.")
    return "\n".join(lines)


def _format_value(name: str, value: Any) -> list[str]:
    """One field's value as indented lines."""
    if name == "archetypes":
        lines = []
        for kind in value:
            lines.append(f"    {kind.name}: {kind.share:.0%} of the pool. {kind.description}")
            for feature in DEALBREAKER_FEATURES:
                lines.append(f"        {feature}: {_shares(kind.tendency(feature))}")
        return lines
    if name == "trait_shares":
        return [f"    {feature}: {_shares(shares)}" for feature, shares in value.items()]
    if isinstance(value, Mapping):
        return [f"    {key}: {share:g}" for key, share in value.items()]
    return []


def format_scenario(name: str, scenario: Scenario, preset: Preset | None = None) -> str:
    """Every field of a scenario: its value, what it means and where the number comes from."""
    lines = [f"Scenario: {name}"]
    if preset is not None:
        lines += _wrap(preset.description, "")
    group = None
    for f in scenario_fields():
        if f.metadata["group"] != group:
            group = f.metadata["group"]
            lines += ["", group, "-" * len(group)]
        value = getattr(scenario, f.name)
        nested = _format_value(f.name, value)
        lines.append(f"{f.name}" + ("" if nested else f" = {value:g}"))
        lines += nested
        lines += _wrap(f"meaning: {f.metadata['meaning']}", "    ")
        lines += _wrap(f"sensible range: {f.metadata['sensible']}", "    ")
        lines += _wrap(f"basis of the default: {f.metadata['basis']}", "    ")
        if preset is not None and f.name in preset.reasons:
            lines += _wrap(f"why {name} changes it: {preset.reasons[f.name]}", "    ")
        if f.name == "archetypes":
            lines.append("    implied population shares:")
            for feature in DEALBREAKER_FEATURES:
                lines.append(f"        {feature}: {_shares(scenario.value_shares(feature))}")
    return "\n".join(lines)


#: Column heading, how to read it off a strategy's stats, and how to print it.
_METRICS: tuple[tuple[str, Callable[[StrategyStats], float], str], ...] = (
    ("both accepted", lambda s: s.mutual_rate, "{:.1%}"),
    ("one accepted", lambda s: s.one_sided_rate, "{:.1%}"),
    ("no proposal", lambda s: s.unproposed_share, "{:.1%}"),
    ("matches/100 users", lambda s: s.matches_per_100_users, "{:.1f}"),
    ("filter violations", lambda s: s.violation_rate, "{:.1%}"),
)

_LEGEND = (
    "both accepted / one accepted: share of proposed pairs accepted by both / by exactly one.",
    "no proposal: share of users the strategy left without an introduction this round.",
    "matches/100 users: pairs accepted by both, per 100 users in the pool.",
    "filter violations: share of proposed pairs that break a hard filter on the true values.",
)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    widths = [max(len(row[i]) for row in (header, *rows)) for i in range(len(header))]
    return [
        "  ".join(
            cell.ljust(width) if i == 0 else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(row, widths))
        )
        for row in (header, *rows)
    ]


def _run_line(report: SimulationReport) -> str:
    return (
        f"seed={report.seed}  trials={report.trials}  "
        f"users/trial={report.scenario.pool_size}  top_n={report.top_n}"
    )


def format_report(report: SimulationReport) -> str:
    """One scenario: a row per strategy, then the calibration table."""
    rows = [
        [s.label, str(s.pairs), *(fmt.format(read(s)) for _, read, fmt in _METRICS)]
        for s in report.strategies.values()
    ]
    lines = [
        f'Wavelength simulation, scenario "{report.scenario_name}"',
        _CAVEAT,
        _run_line(report),
        "",
        *_table(["strategy", "pairs", *(heading for heading, _, _ in _METRICS)], rows),
        "",
        "pairs: introductions proposed, summed over trials.",
        *_LEGEND,
        "",
        "Calibration of predicted P(mutual), strategy 3",
    ]
    bins = [
        [
            f"{b.low:.2f}-{b.high:.2f}",
            str(b.pairs),
            f"{b.mean_predicted:.2f}" if b.pairs else "-",
            f"{b.observed:.2f}" if b.pairs else "-",
        ]
        for b in report.calibration
    ]
    lines += _table(["predicted", "pairs", "mean predicted", "observed both accepted"], bins)
    lines.append(f"mean gap between predicted and observed: {report.calibration_error:.2f}")
    return "\n".join(lines)


def format_comparison(reports: Sequence[SimulationReport]) -> str:
    """Several scenarios side by side: one table per measure, a column per scenario."""
    first = reports[0]
    names = [r.scenario_name for r in reports]
    lines = [
        "Wavelength simulation, scenarios side by side: " + ", ".join(names),
        _CAVEAT,
        f"seed={first.seed}  trials={first.trials}  top_n={first.top_n}  users/trial: "
        + ", ".join(f"{r.scenario_name} {r.scenario.pool_size}" for r in reports),
    ]
    for heading, read, fmt in _METRICS:
        rows = [
            [first.strategies[key].label, *(fmt.format(read(r.strategies[key])) for r in reports)]
            for key in first.strategies
        ]
        lines += ["", *_table([heading, *names], rows)]
    gap = ["mean gap, predicted vs observed", *(f"{r.calibration_error:.2f}" for r in reports)]
    lines += ["", *_table(["calibration of P(mutual), strategy 3", *names], [gap])]
    lines += ["", *_LEGEND]
    return "\n".join(lines)
