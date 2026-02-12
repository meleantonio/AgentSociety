"""Tests for the CLI entry point."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from emergent_constitution.__main__ import build_parser, run_simulation

# Subprocess tests need PYTHONPATH to find the package under src/.
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
_SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": _SRC_DIR}


class TestBuildParser:
    """Tests for the argument parser."""

    def test_default_values(self):
        """Defaults match SimulationConfig defaults."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.agents == 50
        assert args.ticks == 100
        assert args.seed == 42
        assert args.proposal_interval == 5
        assert args.observer_interval == 5
        assert args.output is None
        assert args.json_output is False
        assert args.quiet is False

    def test_custom_values(self):
        """Custom flags are parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["-n", "10", "-t", "20", "-s", "99", "--json", "-q"])
        assert args.agents == 10
        assert args.ticks == 20
        assert args.seed == 99
        assert args.json_output is True
        assert args.quiet is True

    def test_short_flags(self):
        """Short flags work correctly."""
        parser = build_parser()
        args = parser.parse_args(["-n", "5", "-t", "10", "-s", "7", "-o", "out.md"])
        assert args.agents == 5
        assert args.ticks == 10
        assert args.seed == 7
        assert args.output == "out.md"

    def test_long_interval_flags(self):
        """Long interval flags work correctly."""
        parser = build_parser()
        args = parser.parse_args(["--proposal-interval", "3", "--observer-interval", "7"])
        assert args.proposal_interval == 3
        assert args.observer_interval == 7


class TestRunSimulation:
    """Tests for run_simulation."""

    def test_json_output_is_valid(self):
        """JSON output is parseable and has expected keys."""
        parser = build_parser()
        args = parser.parse_args(["-n", "3", "-t", "5", "-s", "42", "--json", "-q"])
        result = run_simulation(args)
        data = json.loads(result)
        assert "constitution" in data
        assert "history" in data
        assert "final_agent_states" in data
        assert "seed" in data
        assert data["seed"] == 42
        assert data["total_ticks"] == 5

    def test_markdown_output_has_headers(self):
        """Markdown output contains expected section headers."""
        parser = build_parser()
        args = parser.parse_args(["-n", "3", "-t", "5", "-q"])
        result = run_simulation(args)
        assert "# The Emergent Constitution" in result
        assert "## Final Constitution" in result
        assert "## Wealth Distribution" in result

    def test_deterministic_output(self):
        """Same seed produces identical output."""
        parser = build_parser()
        args = parser.parse_args(["-n", "5", "-t", "10", "-s", "123", "--json", "-q"])
        result1 = run_simulation(args)
        args2 = parser.parse_args(["-n", "5", "-t", "10", "-s", "123", "--json", "-q"])
        result2 = run_simulation(args2)
        assert result1 == result2

    def test_invalid_config_raises(self):
        """Invalid config (e.g. 0 agents) raises ValidationError."""
        from pydantic import ValidationError

        parser = build_parser()
        args = parser.parse_args(["-n", "0", "-t", "5", "-q"])
        with pytest.raises(ValidationError):
            run_simulation(args)


class TestCLIEndToEnd:
    """End-to-end tests via subprocess."""

    def test_module_runs(self):
        """python -m emergent_constitution executes successfully."""
        result = subprocess.run(
            [sys.executable, "-m", "emergent_constitution", "-n", "3", "-t", "5", "-q"],
            capture_output=True,
            text=True,
            timeout=30,
            env=_SUBPROCESS_ENV,
        )
        assert result.returncode == 0
        assert "# The Emergent Constitution" in result.stdout

    def test_json_flag(self):
        """--json flag produces valid JSON output."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "emergent_constitution",
                "-n",
                "3",
                "-t",
                "5",
                "--json",
                "-q",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=_SUBPROCESS_ENV,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "constitution" in data

    def test_output_file(self, tmp_path: Path):
        """--output flag writes to a file."""
        out_file = tmp_path / "report.md"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "emergent_constitution",
                "-n",
                "3",
                "-t",
                "5",
                "-o",
                str(out_file),
                "-q",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=_SUBPROCESS_ENV,
        )
        assert result.returncode == 0
        content = out_file.read_text()
        assert "# The Emergent Constitution" in content

    def test_version_flag(self):
        """--version prints the version string."""
        result = subprocess.run(
            [sys.executable, "-m", "emergent_constitution", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            env=_SUBPROCESS_ENV,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_invalid_config_exits_nonzero(self):
        """Invalid config (e.g. 0 agents) exits with code 1 and stderr message."""
        result = subprocess.run(
            [sys.executable, "-m", "emergent_constitution", "-n", "0"],
            capture_output=True,
            text=True,
            timeout=10,
            env=_SUBPROCESS_ENV,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr
