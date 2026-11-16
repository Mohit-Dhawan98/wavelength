"""Plain-text rendering: preset lists, scenario descriptions and simulation reports.

Kept apart from the simulation so the numbers can be used from Python
without going through text, and so wording changes never touch the model.
"""

from __future__ import annotations

import textwrap
from typing import Any, Callable, Mapping, Sequence

from wavelength.presets import PRESETS, Preset
from wavelength.scenario import DEALBREAKER_FEATURES, Scenario, scenario_fields
from wavelength.simulate import FITTED_KEY, Calibration, SimulationReport, StrategyStats

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
        lines.append(f"    product: {preset.product}")
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
        lines.append(f"Product: {preset.product}")
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
    ("both yes (all)", lambda s: s.mutual_rate, "{:.1%}"),
    ("both yes (answered)", lambda s: s.mutual_rate_responded, "{:.1%}"),
    ("one yes", lambda s: s.one_sided_rate, "{:.1%}"),
    ("expired", lambda s: s.expired_rate, "{:.1%}"),
    ("no proposal", lambda s: s.unproposed_share, "{:.1%}"),
    ("matches/100", lambda s: s.matches_per_100_users, "{:.1f}"),
    ("violations", lambda s: s.violation_rate, "{:.1%}"),
)

_LEGEND = (
    "both yes (all): share of all proposed pairs in which both people said yes. A pair that",
    "    expired counts against it. This is what the product delivers.",
    "both yes (answered): the same, among pairs in which both people answered. Not answering",
    "    is not a no, so this is the fair test of the matcher.",
    "one yes: share of all proposed pairs with exactly one yes (the other declined or was silent).",
    "expired: share of all proposed pairs in which at least one person never answered.",
    "no proposal: share of users the strategy left without an introduction this round.",
    "matches/100: pairs in which both said yes, per 100 users in the pool.",
    "violations: share of proposed pairs that break a hard filter on the true values.",
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


def _share_of_candidates(top_n: int, candidates: float) -> str:
    return f"{min(1.0, top_n / candidates):.0%}" if candidates > 0 else "-"


#: Row label and how to print it, for the facts about a world.
_WORLD_ROWS: tuple[tuple[str, Callable[[SimulationReport], str]], ...] = (
    ("yes-rate to a random eligible introduction", lambda r: f"{r.world.yes_rate_realised:.1%}"),
    ("people who answer before expiry", lambda r: f"{r.world.responding_share:.1%}"),
    ("people with any eligible candidate", lambda r: f"{r.world.matchable_share:.1%}"),
    ("eligible candidates, larger side (median)", lambda r: f"{r.world.candidates_majority:.0f}"),
    ("eligible candidates, scarce side (median)", lambda r: f"{r.world.candidates_minority:.0f}"),
    (
        "top N as a share of those, larger side",
        lambda r: _share_of_candidates(r.top_n, r.world.candidates_majority),
    ),
    (
        "top N as a share of those, scarce side",
        lambda r: _share_of_candidates(r.top_n, r.world.candidates_minority),
    ),
    ("score vs feeling, per-user weights", lambda r: f"{r.world.score_tracks_feeling:.2f}"),
    ("score vs feeling, global weights", lambda r: f"{r.world.global_score_tracks_feeling:.2f}"),
    (
        "score vs feeling, perfect read of profiles",
        lambda r: f"{r.world.profile_tracks_feeling:.2f}",
    ),
)

_WORLD_LEGEND = (
    "Measured on the first pool, before any strategy runs. Eligible candidates are counted",
    "on what the app knows. 'Score vs feeling' is the rank correlation, per person, between a",
    "score and what that person actually feels about each eligible candidate (0 unrelated, 1",
    "identical); the last row is the most a score built from profiles could reach.",
)


def _world_table(reports: Sequence[SimulationReport]) -> list[str]:
    header = ["about each world" if len(reports) > 1 else "about this world"]
    header += [r.scenario_name for r in reports] if len(reports) > 1 else ["measured"]
    rows = [[label, *(read(r) for r in reports)] for label, read in _WORLD_ROWS]
    return [*_table(header, rows), *_WORLD_LEGEND]


def _calibration_table(calibration: Calibration) -> list[str]:
    rows = [
        [
            f"{b.low:.1f}-{b.high:.1f}",
            str(b.pairs),
            f"{b.mean_predicted:.2f}" if b.pairs else "-",
            f"{b.observed:.2f}" if b.pairs else "-",
            f"{b.if_independent:.2f}" if b.pairs else "-",
        ]
        for b in calibration.bins
    ]
    header = ["predicted", "pairs", "mean predicted", "both yes", "if independent"]
    return [
        f"P(mutual) from {calibration.name}",
        *_table(header, rows),
        f"mean gap between predicted and observed: {calibration.gap:.2f}",
        f"observed minus independent product: {calibration.independence_gap:+.2f}",
    ]


def _curve_table(report: SimulationReport) -> list[str]:
    fitted = report.calibrations[FITTED_KEY].curve
    article = report.calibrations["bidirectional"].curve
    seen = report.fit_outcomes.seen if report.fit_outcomes is not None else [0] * len(fitted)
    rows = [
        [str(rank), f"{article[rank - 1]:.2f}", f"{fitted[rank - 1]:.2f}", str(int(seen[rank - 1]))]
        for rank in range(1, len(fitted) + 1)
    ]
    return [
        "P(accept) by rank",
        *_table(["rank", "article's curve", "fitted curve", "decisions fitted on"], rows),
        "The fitted curve comes from separate pools; all else here is from pools it never saw.",
    ]


_CALIBRATION_LEGEND = (
    "Calibration uses pairs in which both people answered. 'both yes' is the observed share;",
    "a calibrated curve has it close to 'mean predicted'. 'if independent' is the product of",
    "the two sides' observed yes-rates in the bin: when 'both yes' is above it, the two",
    "decisions move together and P(A) x P(B) understates the pair, however good the curve.",
)


def format_report(report: SimulationReport) -> str:
    """One scenario: the world's measured facts, a row per strategy, then calibration."""
    rows = [
        [s.label, str(s.pairs), *(fmt.format(read(s)) for _, read, fmt in _METRICS)]
        for s in report.strategies.values()
    ]
    lines = [
        f'Wavelength simulation, scenario "{report.scenario_name}"',
        _CAVEAT,
        _run_line(report),
        "",
        *_world_table([report]),
        "",
        *_table(["strategy", "pairs", *(heading for heading, _, _ in _METRICS)], rows),
        "",
        "pairs: introductions proposed, summed over trials.",
        *_LEGEND,
        "",
        "Calibration of predicted P(mutual), strategy 3",
    ]
    if report.calibrated:
        lines += ["", *_curve_table(report)]
    for calibration in report.calibrations.values():
        lines += ["", *_calibration_table(calibration)]
    lines += ["", *_CALIBRATION_LEGEND]
    if not report.calibrated:
        lines.append(
            "Run with --calibrate to fit P(accept) by rank and check it on held-out pools."
        )
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
        "",
        *_world_table(reports),
    ]
    for heading, read, fmt in _METRICS:
        rows = [
            [first.strategies[key].label, *(fmt.format(read(r.strategies[key])) for r in reports)]
            for key in first.strategies
        ]
        lines += ["", *_table([heading, *names], rows)]
    for key, calibration in first.calibrations.items():
        rows = [
            [
                "mean gap, predicted vs observed",
                *(f"{r.calibrations[key].gap:.2f}" for r in reports),
            ],
            [
                "observed minus independent product",
                *(f"{r.calibrations[key].independence_gap:+.2f}" for r in reports),
            ],
        ]
        lines += ["", *_table([f"P(mutual) from {calibration.name}", *names], rows)]
    lines += ["", *_LEGEND, *_CALIBRATION_LEGEND]
    return "\n".join(lines)
