"""CLI entry point — run with ``python -m emergent_constitution``.

v1 functions (build_parser, run_simulation, main) retained for backward compat.
v2 functions (build_parser_v2, run_simulation_v2, main_v2) add DSGE-HA flags
and support --benchmark mode per REQ-035.
"""

from __future__ import annotations

import argparse
import logging
import sys

from pydantic import ValidationError

from emergent_constitution import __version__
from emergent_constitution.config import SimulationConfig, SimulationConfigV2
from emergent_constitution.lead import Lead, LeadV2
from emergent_constitution.logging import configure_logging
from emergent_constitution.reporter import generate_json_v2, generate_report, generate_report_v2


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        Configured ArgumentParser with all CLI flags.
    """
    parser = argparse.ArgumentParser(
        prog="emergent-constitution",
        description="The Emergent Constitution — agent-based political economy simulation",
    )
    parser.add_argument(
        "-n", "--agents", type=int, default=50, help="Number of agents (default: 50)"
    )
    parser.add_argument("-t", "--ticks", type=int, default=100, help="Max ticks (default: 100)")
    parser.add_argument("-s", "--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--proposal-interval", type=int, default=5, help="Proposal interval (default: 5)"
    )
    parser.add_argument(
        "--observer-interval", type=int, default=5, help="Observer interval (default: 5)"
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file path")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output raw JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress logging")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def run_simulation(args: argparse.Namespace) -> str:
    """Create config from CLI args, run the simulation, and format output.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Formatted output string (JSON or Markdown).

    Raises:
        ValidationError: If args produce an invalid SimulationConfig.
    """
    config = SimulationConfig(
        num_agents=args.agents,
        max_ticks=args.ticks,
        seed=args.seed,
        proposal_interval=args.proposal_interval,
        observer_interval=args.observer_interval,
    )
    lead = Lead(config)
    output = lead.run()

    if args.json_output:
        return output.model_dump_json(indent=2) + "\n"
    return generate_report(output)


def main() -> None:
    """Parse arguments, run simulation, and write output."""
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.ERROR if args.quiet else logging.INFO
    configure_logging(level=log_level)

    try:
        result = run_simulation(args)
    except ValidationError as exc:
        print(f"Error: invalid configuration — {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print(result, end="")


# ============================================================================
# v2 CLI (DSGE-HA)
# ============================================================================


def build_parser_v2() -> argparse.ArgumentParser:
    """Build the v2 argument parser with DSGE-HA flags.

    Returns:
        Configured ArgumentParser with all v2 CLI flags per REQ-035.
    """
    parser = argparse.ArgumentParser(
        prog="emergent-constitution-v2",
        description="The Emergent Constitution v2 — DSGE-HA political economy simulation",
    )

    # Core
    parser.add_argument(
        "-n", "--agents", type=int, default=50, help="Number of agents (default: 50, min: 20)"
    )
    parser.add_argument(
        "-t", "--periods", type=int, default=100, help="Max periods (default: 100)"
    )
    parser.add_argument("-s", "--seed", type=int, default=42, help="RNG seed (default: 42)")

    # Shock parameters
    parser.add_argument("--rho-z", type=float, default=0.9, help="Idiosyncratic persistence")
    parser.add_argument("--sigma-z", type=float, default=0.2, help="Idiosyncratic volatility")
    parser.add_argument(
        "--rho-A", type=float, default=0.95, dest="rho_a", help="Aggregate TFP persistence"
    )
    parser.add_argument(
        "--sigma-A", type=float, default=0.01, dest="sigma_a", help="Aggregate TFP volatility"
    )

    # Production
    parser.add_argument("--alpha", type=float, default=0.33, help="Capital share (default: 0.33)")
    parser.add_argument(
        "--delta", type=float, default=0.1, help="Depreciation rate (default: 0.1)"
    )

    # Household
    parser.add_argument("--a-min", type=float, default=0.0, help="Borrowing limit (default: 0.0)")

    # Intervals
    parser.add_argument(
        "--proposal-interval", type=int, default=5, help="Proposal interval (default: 5)"
    )
    parser.add_argument(
        "--observer-interval", type=int, default=5, help="Observer interval (default: 5)"
    )

    # Execution mode
    parser.add_argument(
        "--benchmark", action="store_true", help="Numerical-only mode (no LLM calls)"
    )

    # LLM settings
    parser.add_argument("--llm-provider", type=str, default="anthropic", help="LLM provider")
    parser.add_argument(
        "--llm-model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="LLM model ID",
    )

    # Output
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file path")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output raw JSON")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress logging")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")

    return parser


def run_simulation_v2(args: argparse.Namespace) -> str:
    """Create v2 config from CLI args, run the simulation, and format output.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Formatted output string (JSON or Markdown).

    Raises:
        ValidationError: If args produce an invalid SimulationConfigV2.
    """
    config = SimulationConfigV2(
        num_agents=args.agents,
        max_periods=args.periods,
        seed=args.seed,
        rho_z=args.rho_z,
        sigma_z=args.sigma_z,
        rho_a=args.rho_a,
        sigma_a=args.sigma_a,
        alpha=args.alpha,
        delta=args.delta,
        a_min=args.a_min,
        proposal_interval=args.proposal_interval,
        observer_interval=args.observer_interval,
        benchmark_mode=args.benchmark,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )
    lead = LeadV2(config)
    output = lead.run()

    if args.json_output:
        return generate_json_v2(output) + "\n"
    return generate_report_v2(output)


def main_v2() -> None:
    """Parse v2 arguments, run simulation, and write output."""
    parser = build_parser_v2()
    args = parser.parse_args()

    log_level = logging.ERROR if args.quiet else logging.INFO
    configure_logging(level=log_level)

    try:
        result = run_simulation_v2(args)
    except ValidationError as exc:
        print(f"Error: invalid configuration — {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
