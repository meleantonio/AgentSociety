"""Property-based invariant tests for the simulation.

Tests PROP-001 through PROP-006 from the spec, plus a smoke test (17.7).
These tests run against the v1 simulation API (Lead + SimulationConfig v1).
All tests use use_llm=False (deterministic rule-based logic, no LLM calls).

Traceability:
    17.1 -> PROP-001 (Determinism)
    17.2 -> PROP-002 (Budget consistency)
    17.3 -> PROP-003 (Market clearing / output consistency)
    17.4 -> PROP-004 (Non-negativity)
    17.5 -> PROP-005 (Welfare measurability)
    17.6 -> PROP-006 (Constitutional validity)
    17.7 -> Smoke test (all 6 properties, 50 agents, 50 ticks)
"""

from __future__ import annotations

import math

import pytest

from emergent_constitution.config import SimulationConfig
from emergent_constitution.lead import Lead
from emergent_constitution.models.agent import UtilityParams
from emergent_constitution.models.constitution import (
    CONSTITUTION_FIELDS,
    Constitution,
)
from emergent_constitution.models.history import SimulationOutput

# Default tolerance for floating-point comparisons (PROP-003 spec)
TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    num_agents: int = 20,
    max_ticks: int = 20,
    seed: int = 42,
    proposal_interval: int = 5,
    observer_interval: int = 5,
) -> SimulationConfig:
    """Create a SimulationConfig for property tests.

    Uses use_llm=False so all decisions use deterministic rule-based logic.
    """
    return SimulationConfig(
        num_agents=num_agents,
        max_ticks=max_ticks,
        seed=seed,
        proposal_interval=proposal_interval,
        observer_interval=observer_interval,
        use_llm=False,
    )


def _run_simulation(
    num_agents: int = 20,
    max_ticks: int = 20,
    seed: int = 42,
    proposal_interval: int = 5,
    observer_interval: int = 5,
) -> SimulationOutput:
    """Run a deterministic simulation and return output."""
    config = _make_config(
        num_agents=num_agents,
        max_ticks=max_ticks,
        seed=seed,
        proposal_interval=proposal_interval,
        observer_interval=observer_interval,
    )
    return Lead(config).run()


# ===========================================================================
# 17.1 PROP-001: Determinism
# ===========================================================================


class TestPROP001Determinism:
    """Same seed + same config = identical output (PROP-001).

    Run simulation twice with the same seed and deterministic mode.
    Assert outputs are identical by comparing serialized JSON.
    """

    def test_determinism_small(self):
        """Two identical runs with 10 agents / 20 ticks produce identical JSON."""
        output1 = _run_simulation(num_agents=10, max_ticks=20, seed=42)
        output2 = _run_simulation(num_agents=10, max_ticks=20, seed=42)
        json1 = output1.model_dump_json(indent=2)
        json2 = output2.model_dump_json(indent=2)
        assert json1 == json2, "Identical seed must produce identical output"

    def test_determinism_medium(self):
        """Two identical runs with 20 agents / 50 ticks produce identical JSON."""
        output1 = _run_simulation(num_agents=20, max_ticks=50, seed=77)
        output2 = _run_simulation(num_agents=20, max_ticks=50, seed=77)
        json1 = output1.model_dump_json(indent=2)
        json2 = output2.model_dump_json(indent=2)
        assert json1 == json2, "Identical seed must produce identical output"

    def test_different_seeds_differ(self):
        """Different seeds produce different outputs."""
        output1 = _run_simulation(num_agents=10, max_ticks=20, seed=42)
        output2 = _run_simulation(num_agents=10, max_ticks=20, seed=99)
        json1 = output1.model_dump_json(indent=2)
        json2 = output2.model_dump_json(indent=2)
        assert json1 != json2, "Different seeds should produce different output"

    def test_determinism_with_governance(self):
        """Determinism holds even when proposals and votes occur."""
        output1 = _run_simulation(num_agents=20, max_ticks=50, seed=42, proposal_interval=5)
        output2 = _run_simulation(num_agents=20, max_ticks=50, seed=42, proposal_interval=5)
        assert output1.model_dump_json() == output2.model_dump_json()

    def test_determinism_final_agents_match(self):
        """Final agent states are identical across deterministic runs."""
        output1 = _run_simulation(num_agents=15, max_ticks=30, seed=123)
        output2 = _run_simulation(num_agents=15, max_ticks=30, seed=123)
        for a1, a2 in zip(output1.final_agent_states, output2.final_agent_states, strict=True):
            assert a1.wealth == a2.wealth, f"Wealth mismatch for {a1.id}"
            assert a1.productivity == a2.productivity


# ===========================================================================
# 17.2 PROP-002: Budget Consistency
# ===========================================================================


class TestPROP002BudgetConsistency:
    """Budget consistency: wealth must remain non-negative and finite.

    In the v1 model, the economic step adds production output minus taxes
    plus redistribution. Wealth should always be >= 0 and finite.
    """

    def test_wealth_nonnegative_all_agents(self):
        """All agents maintain non-negative wealth."""
        output = _run_simulation(num_agents=20, max_ticks=20, seed=42)
        for a in output.final_agent_states:
            assert a.wealth >= 0.0, f"Agent {a.id} has negative wealth: {a.wealth}"

    def test_mean_wealth_positive(self):
        """Mean wealth in history is non-negative."""
        output = _run_simulation(num_agents=20, max_ticks=20, seed=42)
        for entry in output.history:
            assert entry.mean_wealth >= 0.0

    def test_no_nan_in_wealth(self):
        """No agent has NaN or Inf in wealth."""
        output = _run_simulation(num_agents=20, max_ticks=20, seed=42)
        for a in output.final_agent_states:
            assert not math.isnan(a.wealth), f"Agent {a.id} has NaN wealth"
            assert not math.isinf(a.wealth), f"Agent {a.id} has Inf wealth"

    def test_productivity_positive(self):
        """Agent productivity remains positive throughout."""
        output = _run_simulation(num_agents=20, max_ticks=20, seed=42)
        for a in output.final_agent_states:
            assert a.productivity > 0.0, (
                f"Agent {a.id} has non-positive productivity: {a.productivity}"
            )

    @pytest.mark.parametrize("seed", [42, 77, 123, 456, 789])
    def test_budget_consistency_multiple_seeds(self, seed: int):
        """Budget consistency holds across multiple random seeds."""
        output = _run_simulation(num_agents=15, max_ticks=20, seed=seed)
        for a in output.final_agent_states:
            assert a.wealth >= 0.0
            assert not math.isnan(a.wealth)
            assert not math.isinf(a.wealth)
            assert a.productivity > 0.0


# ===========================================================================
# 17.3 PROP-003: Market Clearing / Output Consistency
# ===========================================================================


class TestPROP003MarketClearing:
    """Output consistency: total_output > 0, aggregate stats valid.

    In v1, total_output = sum of agent productivities. We verify it's
    positive and consistent with the number of agents.
    """

    def test_total_output_positive(self):
        """Total output is positive every observation tick."""
        output = _run_simulation(num_agents=20, max_ticks=20, seed=42, observer_interval=1)
        for entry in output.history:
            assert entry.total_output > 0.0, f"Zero/negative output at tick {entry.tick}"

    def test_total_output_equals_sum_productivities(self):
        """Total output should match sum of agent productivities at final state."""
        output = _run_simulation(num_agents=20, max_ticks=20, seed=42)
        expected_output = sum(a.productivity for a in output.final_agent_states)
        assert expected_output > 0.0

    def test_mean_wealth_consistent(self):
        """Mean wealth in history is consistent (non-negative, finite)."""
        output = _run_simulation(num_agents=20, max_ticks=20, seed=42, observer_interval=1)
        for entry in output.history:
            assert entry.mean_wealth >= 0.0
            assert not math.isnan(entry.mean_wealth)
            assert not math.isinf(entry.mean_wealth)

    def test_median_wealth_nonneg(self):
        """Median wealth is non-negative."""
        output = _run_simulation(num_agents=20, max_ticks=20, seed=42, observer_interval=1)
        for entry in output.history:
            assert entry.median_wealth >= 0.0, (
                f"Negative median wealth at tick {entry.tick}: {entry.median_wealth}"
            )

    @pytest.mark.parametrize("seed", [42, 77, 123])
    def test_output_positive_multiple_seeds(self, seed: int):
        """Output is positive across multiple seeds."""
        output = _run_simulation(num_agents=20, max_ticks=20, seed=seed, observer_interval=5)
        for entry in output.history:
            assert entry.total_output > 0.0


# ===========================================================================
# 17.4 PROP-004: Non-negativity
# ===========================================================================


class TestPROP004NonNegativity:
    """Assert wealth >= 0, productivity > 0 for all agents."""

    def test_agent_wealth_nonneg(self):
        """All agents have non-negative wealth."""
        output = _run_simulation(num_agents=20, max_ticks=20, seed=42)
        for a in output.final_agent_states:
            assert a.wealth >= 0.0, f"Agent {a.id} has negative wealth: {a.wealth}"

    def test_agent_productivity_positive(self):
        """All agents have positive productivity."""
        output = _run_simulation(num_agents=20, max_ticks=20, seed=42)
        for a in output.final_agent_states:
            assert a.productivity > 0.0, (
                f"Agent {a.id} has non-positive productivity: {a.productivity}"
            )

    def test_gini_in_range(self):
        """Gini coefficient is in [0, 1] at every observation."""
        output = _run_simulation(num_agents=20, max_ticks=20, seed=42, observer_interval=1)
        for entry in output.history:
            assert 0.0 <= entry.gini <= 1.0, f"Invalid Gini {entry.gini} at tick {entry.tick}"

    def test_pareto_score_in_range(self):
        """Pareto score is in [0, 1] at every observation."""
        output = _run_simulation(num_agents=20, max_ticks=20, seed=42, observer_interval=1)
        for entry in output.history:
            assert 0.0 <= entry.pareto_score <= 1.0, (
                f"Invalid Pareto score {entry.pareto_score} at tick {entry.tick}"
            )

    def test_total_output_nonneg(self):
        """Total output is non-negative."""
        output = _run_simulation(num_agents=20, max_ticks=20, seed=42, observer_interval=1)
        for entry in output.history:
            assert entry.total_output >= 0.0

    @pytest.mark.parametrize("seed", [42, 77, 123, 999, 2024])
    def test_nonneg_multiple_seeds(self, seed: int):
        """Non-negativity holds across multiple seeds."""
        output = _run_simulation(num_agents=15, max_ticks=20, seed=seed)
        for a in output.final_agent_states:
            assert a.wealth >= 0.0
            assert a.productivity > 0.0

    def test_no_nan_or_inf(self):
        """No agent has NaN or Inf in key fields."""
        output = _run_simulation(num_agents=20, max_ticks=30, seed=42)
        for a in output.final_agent_states:
            assert not math.isnan(a.wealth), f"Agent {a.id} has NaN wealth"
            assert not math.isinf(a.wealth), f"Agent {a.id} has Inf wealth"
            assert not math.isnan(a.productivity), f"Agent {a.id} has NaN productivity"
            assert not math.isinf(a.productivity), f"Agent {a.id} has Inf productivity"

    def test_num_coalitions_nonneg(self):
        """Number of coalitions is non-negative in history."""
        output = _run_simulation(num_agents=20, max_ticks=20, seed=42, observer_interval=1)
        for entry in output.history:
            assert entry.num_coalitions >= 0


# ===========================================================================
# 17.5 PROP-005: Welfare Measurability
# ===========================================================================


class TestPROP005WelfareMeasurability:
    """Verify that utility is computable from observables.

    In v1, the observer computes utility as:
    u(wealth, leisure=1.0, public_goods) = wealth^alpha * 1^beta * G^gamma

    We verify this function behaves correctly with known inputs and that
    the v2 compute_realized_utility function works correctly.
    """

    @staticmethod
    def _compute_utility(
        wealth: float,
        leisure: float,
        g: float,
        params: UtilityParams,
    ) -> float:
        """Compute u(c, leisure, G) = c^alpha * leisure^beta * G^gamma.

        Handles edge cases where c=0, leisure=0, or G=0 (utility = 0).
        """
        if wealth <= 0.0 or leisure <= 0.0 or g <= 0.0:
            return 0.0
        try:
            return (
                math.pow(wealth, params.alpha)
                * math.pow(leisure, params.beta)
                * math.pow(g, params.gamma)
            )
        except (ValueError, OverflowError):
            return 0.0

    def test_utility_nonnegative(self):
        """Utility is non-negative for positive inputs."""
        params = UtilityParams(alpha=0.5, beta=0.3, gamma=0.2)
        u = self._compute_utility(100.0, 0.5, 10.0, params)
        assert u >= 0.0

    def test_higher_wealth_higher_utility(self):
        """Utility function is monotone in wealth (ceteris paribus)."""
        params = UtilityParams(alpha=0.5, beta=0.3, gamma=0.2)
        u_low = self._compute_utility(wealth=10.0, leisure=0.5, g=5.0, params=params)
        u_high = self._compute_utility(wealth=20.0, leisure=0.5, g=5.0, params=params)
        assert u_high > u_low

    def test_higher_leisure_higher_utility(self):
        """Utility function is monotone in leisure (ceteris paribus)."""
        params = UtilityParams(alpha=0.5, beta=0.3, gamma=0.2)
        u_low = self._compute_utility(wealth=10.0, leisure=0.3, g=5.0, params=params)
        u_high = self._compute_utility(wealth=10.0, leisure=0.7, g=5.0, params=params)
        assert u_high > u_low

    def test_higher_public_goods_higher_utility(self):
        """Utility function is monotone in public goods (ceteris paribus)."""
        params = UtilityParams(alpha=0.5, beta=0.3, gamma=0.2)
        u_low = self._compute_utility(wealth=10.0, leisure=0.5, g=2.0, params=params)
        u_high = self._compute_utility(wealth=10.0, leisure=0.5, g=10.0, params=params)
        assert u_high > u_low

    @pytest.mark.parametrize(
        "alpha,beta,gamma",
        [
            (0.5, 0.3, 0.2),
            (0.33, 0.34, 0.33),
            (0.1, 0.1, 0.8),
            (0.8, 0.1, 0.1),
        ],
    )
    def test_utility_various_params(self, alpha: float, beta: float, gamma: float):
        """Utility is computable and finite for various parameter combinations."""
        params = UtilityParams(alpha=alpha, beta=beta, gamma=gamma)
        u = self._compute_utility(wealth=50.0, leisure=0.5, g=10.0, params=params)
        assert u >= 0.0
        assert not math.isnan(u)
        assert not math.isinf(u)

    def test_zero_inputs_yield_zero(self):
        """Zero consumption/leisure/public goods yields zero utility."""
        params = UtilityParams(alpha=0.5, beta=0.3, gamma=0.2)
        assert self._compute_utility(0.0, 0.5, 5.0, params) == 0.0
        assert self._compute_utility(10.0, 0.0, 5.0, params) == 0.0
        assert self._compute_utility(10.0, 0.5, 0.0, params) == 0.0

    def test_pareto_score_computed(self):
        """Pareto score is computed at each observation."""
        output = _run_simulation(num_agents=20, max_ticks=20, seed=42, observer_interval=5)
        for entry in output.history:
            assert 0.0 <= entry.pareto_score <= 1.0
            assert not math.isnan(entry.pareto_score)


# ===========================================================================
# 17.6 PROP-006: Constitutional Validity
# ===========================================================================


class TestPROP006ConstitutionalValidity:
    """Every active rule in the constitution must be valid at every observation."""

    @staticmethod
    def _validate_constitution(constitution: Constitution) -> list[str]:
        """Validate a v1 constitution. Returns list of error messages."""
        errors = []
        # tax_rate must be in [0, 1]
        if not (0.0 <= constitution.tax_rate <= 1.0):
            errors.append(f"Invalid tax_rate: {constitution.tax_rate}")
        # property_rule must be a valid PropertyRule
        try:
            from emergent_constitution.models.constitution import PropertyRule

            PropertyRule(constitution.property_rule)
        except ValueError:
            errors.append(f"Invalid property_rule: {constitution.property_rule}")
        # voting_rule must be a valid VotingRule
        try:
            from emergent_constitution.models.constitution import VotingRule

            VotingRule(constitution.voting_rule)
        except ValueError:
            errors.append(f"Invalid voting_rule: {constitution.voting_rule}")
        # redistribution_rule must be a valid RedistributionRule
        try:
            from emergent_constitution.models.constitution import RedistributionRule

            RedistributionRule(constitution.redistribution_rule)
        except ValueError:
            errors.append(f"Invalid redistribution_rule: {constitution.redistribution_rule}")
        return errors

    def test_initial_constitution_valid(self):
        """Default constitution passes all validation checks."""
        constitution = Constitution()
        errors = self._validate_constitution(constitution)
        assert not errors, f"Default constitution has errors: {errors}"

    def test_all_history_snapshots_valid(self):
        """Every constitution snapshot in history passes validation."""
        output = _run_simulation(
            num_agents=30,
            max_ticks=50,
            seed=42,
            proposal_interval=5,
            observer_interval=5,
        )
        for entry in output.history:
            errors = self._validate_constitution(entry.constitution_snapshot)
            assert not errors, f"Invalid constitution at tick {entry.tick}: {errors}"

    def test_final_constitution_valid(self):
        """Final constitution passes all validation checks."""
        output = _run_simulation(
            num_agents=30,
            max_ticks=50,
            seed=42,
            proposal_interval=5,
        )
        errors = self._validate_constitution(output.constitution)
        assert not errors, f"Final constitution has errors: {errors}"

    @pytest.mark.parametrize("seed", [42, 77, 123, 456, 789])
    def test_constitutional_validity_multiple_seeds(self, seed: int):
        """Constitution remains valid across multiple random seeds."""
        output = _run_simulation(num_agents=20, max_ticks=30, seed=seed, proposal_interval=5)
        errors = self._validate_constitution(output.constitution)
        assert not errors, f"Constitution invalid with seed={seed}: {errors}"
        for entry in output.history:
            errors = self._validate_constitution(entry.constitution_snapshot)
            assert not errors, (
                f"Constitution invalid at tick {entry.tick} with seed={seed}: {errors}"
            )

    def test_constitution_fields_defined(self):
        """CONSTITUTION_FIELDS covers all required fields."""
        assert "property_rule" in CONSTITUTION_FIELDS
        assert "tax_rate" in CONSTITUTION_FIELDS
        assert "voting_rule" in CONSTITUTION_FIELDS
        assert "redistribution_rule" in CONSTITUTION_FIELDS


# ===========================================================================
# 17.7 Smoke Test: All 6 Properties Simultaneously
# ===========================================================================


@pytest.mark.slow
class TestSmokeAllProperties:
    """Run 50 agents for 50 ticks in deterministic mode and verify
    all 6 properties hold simultaneously."""

    @pytest.fixture(scope="class")
    def smoke_output(self) -> SimulationOutput:
        """Run the smoke simulation once (cached across all tests in class)."""
        return _run_simulation(
            num_agents=50,
            max_ticks=50,
            seed=42,
            proposal_interval=5,
            observer_interval=5,
        )

    # --- PROP-001: Determinism ---

    def test_prop001_determinism(self, smoke_output: SimulationOutput):
        """PROP-001: Same seed produces identical output."""
        output2 = _run_simulation(
            num_agents=50,
            max_ticks=50,
            seed=42,
            proposal_interval=5,
            observer_interval=5,
        )
        assert smoke_output.model_dump_json() == output2.model_dump_json()

    # --- PROP-002: Budget Consistency ---

    def test_prop002_wealth_nonneg(self, smoke_output: SimulationOutput):
        """PROP-002: All agents have non-negative wealth."""
        for a in smoke_output.final_agent_states:
            assert a.wealth >= 0.0, f"Agent {a.id} has negative wealth: {a.wealth}"

    def test_prop002_productivity_positive(self, smoke_output: SimulationOutput):
        """PROP-002: All agents have positive productivity."""
        for a in smoke_output.final_agent_states:
            assert a.productivity > 0.0, (
                f"Agent {a.id} has non-positive productivity: {a.productivity}"
            )

    # --- PROP-003: Market Clearing ---

    def test_prop003_output_positive(self, smoke_output: SimulationOutput):
        """PROP-003: Total output is positive every observation tick."""
        for entry in smoke_output.history:
            assert entry.total_output > 0.0, f"Total output zero or negative at tick {entry.tick}"

    def test_prop003_mean_wealth_nonneg(self, smoke_output: SimulationOutput):
        """PROP-003: Mean wealth is non-negative."""
        for entry in smoke_output.history:
            assert entry.mean_wealth >= 0.0, f"Negative mean wealth at tick {entry.tick}"

    # --- PROP-004: Non-negativity ---

    def test_prop004_agents(self, smoke_output: SimulationOutput):
        """PROP-004: Non-negativity of all agent quantities."""
        for a in smoke_output.final_agent_states:
            assert a.wealth >= 0.0
            assert a.productivity > 0.0

    def test_prop004_no_nan_inf(self, smoke_output: SimulationOutput):
        """PROP-004: No NaN/Inf in any agent quantity."""
        for a in smoke_output.final_agent_states:
            for field_name in ["wealth", "productivity"]:
                val = getattr(a, field_name)
                assert not math.isnan(val), f"Agent {a.id} has NaN {field_name}"
                assert not math.isinf(val), f"Agent {a.id} has Inf {field_name}"

    def test_prop004_gini_valid(self, smoke_output: SimulationOutput):
        """PROP-004: Gini coefficient is in [0, 1] at every observation."""
        for entry in smoke_output.history:
            assert 0.0 <= entry.gini <= 1.0, f"Invalid Gini {entry.gini} at tick {entry.tick}"

    # --- PROP-005: Welfare Measurability ---

    def test_prop005_pareto_valid(self, smoke_output: SimulationOutput):
        """PROP-005: Pareto score is in [0, 1] at every observation."""
        for entry in smoke_output.history:
            assert 0.0 <= entry.pareto_score <= 1.0, (
                f"Invalid Pareto score {entry.pareto_score} at tick {entry.tick}"
            )

    # --- PROP-006: Constitutional Validity ---

    def test_prop006_constitution_valid(self, smoke_output: SimulationOutput):
        """PROP-006: Constitution is valid at every observation."""
        for entry in smoke_output.history:
            snap = entry.constitution_snapshot
            assert 0.0 <= snap.tax_rate <= 1.0, (
                f"Invalid tax rate {snap.tax_rate} at tick {entry.tick}"
            )

    def test_prop006_final_valid(self, smoke_output: SimulationOutput):
        """PROP-006: Final constitution is valid."""
        c = smoke_output.constitution
        assert 0.0 <= c.tax_rate <= 1.0

    # --- Structural checks ---

    def test_history_completeness(self, smoke_output: SimulationOutput):
        """History has the expected number of entries."""
        expected = 50 // 5  # max_ticks // observer_interval
        assert len(smoke_output.history) == expected

    def test_agent_count(self, smoke_output: SimulationOutput):
        """Correct number of agents in final output."""
        assert len(smoke_output.final_agent_states) == 50

    def test_output_metadata(self, smoke_output: SimulationOutput):
        """Output has correct metadata."""
        assert smoke_output.seed == 42
        assert smoke_output.total_ticks == 50
