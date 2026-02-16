"""Tests for the v2 CLI entry point."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from emergent_constitution.__main__ import build_parser_v2, run_simulation_v2

# Subprocess tests need PYTHONPATH to find the package under src/.
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
_SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": _SRC_DIR}

# Use mock LLM provider for fast tests (benchmark_mode uses VFI which is slow).
_FAST_FLAGS = ["-n", "20", "-t", "3", "-q", "--llm-provider", "mock"]


class TestBuildParserV2:
    """Tests for the v2 argument parser."""

    def test_default_values(self):
        parser = build_parser_v2()
        args = parser.parse_args([])
        assert args.agents == 50
        assert args.periods == 100
        assert args.seed == 42
        assert args.rho_z == 0.9
        assert args.sigma_z == 0.2
        assert args.rho_a == 0.95
        assert args.sigma_a == 0.01
        assert args.alpha == 0.33
        assert args.delta == 0.1
        assert args.a_min == 0.0
        assert args.proposal_interval == 5
        assert args.observer_interval == 5
        assert args.benchmark is False
        assert args.llm_provider == "local"
        assert args.llm_model == "deepseek/deepseek-r1-0528-qwen3-8b"
        assert args.llm_base_url == "http://localhost:1234/v1"
        assert args.output is None
        assert args.json_output is False
        assert args.quiet is False

    def test_custom_core_values(self):
        parser = build_parser_v2()
        args = parser.parse_args(["-n", "30", "-t", "50", "-s", "99"])
        assert args.agents == 30
        assert args.periods == 50
        assert args.seed == 99

    def test_shock_parameters(self):
        parser = build_parser_v2()
        args = parser.parse_args(["--rho-z", "0.8", "--sigma-z", "0.3"])
        assert args.rho_z == 0.8
        assert args.sigma_z == 0.3

    def test_aggregate_shock_parameters(self):
        parser = build_parser_v2()
        args = parser.parse_args(["--rho-A", "0.9", "--sigma-A", "0.02"])
        assert args.rho_a == 0.9
        assert args.sigma_a == 0.02

    def test_production_parameters(self):
        parser = build_parser_v2()
        args = parser.parse_args(["--alpha", "0.4", "--delta", "0.05"])
        assert args.alpha == 0.4
        assert args.delta == 0.05

    def test_borrowing_limit(self):
        parser = build_parser_v2()
        args = parser.parse_args(["--a-min", "5.0"])
        assert args.a_min == 5.0

    def test_benchmark_flag(self):
        parser = build_parser_v2()
        args = parser.parse_args(["--benchmark"])
        assert args.benchmark is True

    def test_llm_flags(self):
        parser = build_parser_v2()
        args = parser.parse_args(["--llm-provider", "openai", "--llm-model", "gpt-4"])
        assert args.llm_provider == "openai"
        assert args.llm_model == "gpt-4"

    def test_output_flags(self):
        parser = build_parser_v2()
        args = parser.parse_args(["--json", "-q", "-o", "out.json"])
        assert args.json_output is True
        assert args.quiet is True
        assert args.output == "out.json"

    def test_interval_flags(self):
        parser = build_parser_v2()
        args = parser.parse_args(["--proposal-interval", "3", "--observer-interval", "7"])
        assert args.proposal_interval == 3
        assert args.observer_interval == 7


class TestRunSimulationV2:
    """Tests for run_simulation_v2 (uses mock LLM for speed)."""

    def test_mock_produces_markdown(self):
        """Mock LLM mode produces Markdown output."""
        parser = build_parser_v2()
        args = parser.parse_args(_FAST_FLAGS)
        result = run_simulation_v2(args)
        assert "# The Emergent Constitution" in result
        assert "(v2)" in result

    def test_json_output_is_valid(self):
        """JSON output is parseable and has expected keys."""
        parser = build_parser_v2()
        args = parser.parse_args([*_FAST_FLAGS, "--json"])
        result = run_simulation_v2(args)
        data = json.loads(result)
        assert "constitution" in data
        assert "history" in data
        assert "final_households" in data
        assert "final_firms" in data
        assert "welfare_summary" in data
        assert "seed" in data
        assert data["seed"] == 42
        assert data["total_periods"] == 3

    def test_markdown_has_v2_sections(self):
        """Markdown output contains v2-specific sections."""
        parser = build_parser_v2()
        args = parser.parse_args(_FAST_FLAGS)
        result = run_simulation_v2(args)
        assert "## Firm Summary" in result
        assert "## Welfare Summary" in result

    def test_deterministic_output(self):
        """Same seed produces identical output."""
        parser = build_parser_v2()
        flags = [*_FAST_FLAGS, "-s", "123", "--json"]
        args1 = parser.parse_args(flags)
        args2 = parser.parse_args(flags)
        result1 = run_simulation_v2(args1)
        result2 = run_simulation_v2(args2)
        assert result1 == result2

    def test_invalid_config_raises(self):
        """Invalid config (e.g. too few agents) raises ValidationError."""
        parser = build_parser_v2()
        args = parser.parse_args(["-n", "5", "-t", "3", "-q", "--llm-provider", "mock"])
        with pytest.raises(ValidationError):
            run_simulation_v2(args)

    def test_custom_seed_in_output(self):
        """Custom seed appears in JSON output."""
        parser = build_parser_v2()
        args = parser.parse_args([*_FAST_FLAGS, "-s", "999", "--json"])
        result = run_simulation_v2(args)
        data = json.loads(result)
        assert data["seed"] == 999


class TestCLIV2EndToEnd:
    """End-to-end tests via subprocess using main_v2."""

    def test_version_flag(self):
        """--version prints the version string."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from emergent_constitution.__main__ import main_v2; main_v2()",
                "--version",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env=_SUBPROCESS_ENV,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout

    def test_mock_llm_e2e(self):
        """Mock LLM produces Markdown output via subprocess."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from emergent_constitution.__main__ import main_v2; main_v2()",
                *_FAST_FLAGS,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=_SUBPROCESS_ENV,
        )
        assert result.returncode == 0
        assert "# The Emergent Constitution" in result.stdout

    def test_json_e2e(self):
        """--json produces valid JSON via subprocess."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from emergent_constitution.__main__ import main_v2; main_v2()",
                *_FAST_FLAGS,
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=_SUBPROCESS_ENV,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "constitution" in data

    def test_invalid_config_exits_nonzero(self):
        """Invalid config exits with code 1 and stderr message."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from emergent_constitution.__main__ import main_v2; main_v2()",
                "-n",
                "5",
                "--llm-provider",
                "mock",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env=_SUBPROCESS_ENV,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr

    def test_output_file_e2e(self, tmp_path: Path):
        """--output flag writes to a file."""
        out_file = tmp_path / "report_v2.md"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from emergent_constitution.__main__ import main_v2; main_v2()",
                *_FAST_FLAGS,
                "-o",
                str(out_file),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=_SUBPROCESS_ENV,
        )
        assert result.returncode == 0
        content = out_file.read_text()
        assert "# The Emergent Constitution" in content
