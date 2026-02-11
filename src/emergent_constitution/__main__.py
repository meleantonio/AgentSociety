"""CLI entry point — run with ``python -m emergent_constitution``."""

from __future__ import annotations

import argparse
import logging
import sys

from pydantic import ValidationError

from emergent_constitution import __version__
from emergent_constitution.config import SimulationConfig
from emergent_constitution.lead import Lead
from emergent_constitution.logging import configure_logging
from emergent_constitution.reporter import generate_report


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
        return output.model_dump_json(indent=2)
    return generate_report(output)


def main() -> None:
    """Parse arguments, run simulation, and write output."""
    parser = build_parser()
    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    configure_logging(level=log_level)

    try:
        result = run_simulation(args)
    except ValidationError as exc:
        print(f"Error: invalid configuration — {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
