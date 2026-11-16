"""Command-line entry point.

``wavelength scenarios``            list the preset worlds
``wavelength describe <scenario>``  every field, its value and its meaning
``wavelength simulate``             run the three strategies in one or more worlds
``wavelength simulate --calibrate`` also fit P(accept) by rank and check it held out

A scenario is a preset name or the path of a ``.toml`` / ``.json`` scenario
file. ``--set key=value`` then overrides single fields.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from wavelength.matching import TOP_N
from wavelength.presets import PRESETS, resolve_scenario
from wavelength.report import format_comparison, format_presets, format_report, format_scenario
from wavelength.scenario import Scenario
from wavelength.scenario_io import with_overrides
from wavelength.simulate import DEFAULT_TRIALS, run_simulation


def _add_scenario_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "override one field, e.g. --set yes_rate=0.2 or "
            "--set archetypes.casual.share=0.5 (repeatable; setting one share of a "
            "distribution rescales the others)"
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wavelength",
        description="Mutual-match matchmaking on simulated people, under assumptions you set.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("scenarios", help="list the preset worlds and what each depicts")

    describe = commands.add_parser(
        "describe", help="print every field of a scenario, its value and its meaning"
    )
    describe.add_argument("scenario", help="preset name or scenario file (.toml / .json)")
    _add_scenario_options(describe)

    simulate = commands.add_parser(
        "simulate",
        help="run the three matching strategies on the same simulated people",
        description=(
            "Generate people from a scenario, run each strategy on the same pool, and "
            "report how the hidden ground truth says each introduction would have gone."
        ),
    )
    simulate.add_argument(
        "--scenario",
        default="today",
        help=f"preset ({', '.join(PRESETS)}) or scenario file (default: today)",
    )
    _add_scenario_options(simulate)
    simulate.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    simulate.add_argument(
        "--users",
        type=int,
        default=None,
        help="people per pool (default: the scenario's pool_size)",
    )
    simulate.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help=f"independent pools to draw and add up (default: {DEFAULT_TRIALS})",
    )
    simulate.add_argument(
        "--top-n", type=int, default=TOP_N, help=f"mutual top-N cutoff (default: {TOP_N})"
    )
    simulate.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "fit P(accept) by rank on separate pools, then report strategy 3 with both "
            "the article's curve and the fitted one on held-out pools"
        ),
    )
    simulate.add_argument(
        "--compare",
        default="",
        metavar="A,B,...",
        help=(
            "also run these scenarios and print them side by side; --set, --users, "
            "--seed, --trials and --calibrate apply to every scenario"
        ),
    )
    return parser


def _resolve(name_or_path: str, overrides: Sequence[str]) -> tuple[str, Scenario]:
    name, scenario = resolve_scenario(name_or_path)
    if not overrides:
        return name, scenario
    return f"{name} (modified)", with_overrides(scenario, overrides)


def _describe(args: argparse.Namespace) -> str:
    name, scenario = _resolve(args.scenario, args.overrides)
    # A preset's reasons only describe the preset as written.
    preset = PRESETS.get(args.scenario) if not args.overrides else None
    return format_scenario(name, scenario, preset)


def _simulate(args: argparse.Namespace) -> str:
    requested = [args.scenario, *(part.strip() for part in args.compare.split(",") if part.strip())]
    reports = []
    for name_or_path in dict.fromkeys(requested):  # drop repeats, keep order
        name, scenario = _resolve(name_or_path, args.overrides)
        reports.append(
            run_simulation(
                scenario,
                seed=args.seed,
                trials=args.trials,
                top_n=args.top_n,
                users=args.users,
                name=name,
                calibrate=args.calibrate,
            )
        )
    return format_report(reports[0]) if len(reports) == 1 else format_comparison(reports)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "scenarios":
            output = format_presets()
        elif args.command == "describe":
            output = _describe(args)
        else:
            output = _simulate(args)
    except ValueError as error:
        print(f"wavelength: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
