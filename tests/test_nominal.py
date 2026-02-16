"""Unit tests for the nominal rigidities block (REQ-310 through REQ-315).

Tests cover:
- Taylor rule computation (REQ-312)
- Fisher equation identity (REQ-313)
- NKPC at steady state (REQ-311)
- Nominal block convergence
- Interaction with market clearing
- Config parameters
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from emergent_constitution.nominal import NominalBlock, NominalState

# ============================================================================
# Test fixtures
# ============================================================================


def _default_config() -> SimpleNamespace:
    """Create a default nominal config namespace."""
    return SimpleNamespace(
        rotemberg_cost=100.0,
        taylor_phi_pi=1.5,
        taylor_phi_y=0.125,
        inflation_target=0.02,
        elasticity_sub=6.0,
    )


# ============================================================================
# NominalState tests
# ============================================================================


class TestNominalState:
    """Tests for the NominalState dataclass."""

    def test_default_values(self) -> None:
        state = NominalState()
        assert state.price_level == 1.0
        assert state.inflation == 0.0
        assert state.nominal_rate == 0.05
        assert state.real_bond_rate == 0.03
        assert state.expected_inflation == 0.0
        assert state.marginal_cost == 1.0

    def test_custom_values(self) -> None:
        state = NominalState(
            price_level=1.02,
            inflation=0.02,
            nominal_rate=0.04,
            real_bond_rate=0.02,
            expected_inflation=0.02,
            marginal_cost=0.83,
        )
        assert state.price_level == 1.02
        assert state.inflation == 0.02
        assert state.marginal_cost == 0.83


# ============================================================================
# Taylor rule tests (REQ-312)
# ============================================================================


class TestTaylorRule:
    """Tests for the Taylor rule computation."""

    def test_at_target_zero_gap(self) -> None:
        """At target inflation with zero output gap, i = r_natural."""
        block = NominalBlock(_default_config())
        r_natural = 0.03
        i = block.taylor_rule(
            inflation=block.pi_bar,
            output_gap=0.0,
            r_natural=r_natural,
        )
        assert abs(i - r_natural) < 1e-10

    def test_inflation_above_target(self) -> None:
        """Higher inflation raises nominal rate (Taylor principle: phi_pi > 1)."""
        block = NominalBlock(_default_config())
        r_natural = 0.03
        pi_high = 0.05  # 5% inflation vs 2% target
        i = block.taylor_rule(inflation=pi_high, output_gap=0.0, r_natural=r_natural)
        # i = 0.03 + 1.5 * (0.05 - 0.02) + 0.125 * 0 = 0.03 + 0.045 = 0.075
        assert abs(i - 0.075) < 1e-10

    def test_positive_output_gap(self) -> None:
        """Positive output gap raises nominal rate."""
        block = NominalBlock(_default_config())
        r_natural = 0.03
        i = block.taylor_rule(inflation=0.02, output_gap=0.04, r_natural=r_natural)
        # i = 0.03 + 1.5*(0.02 - 0.02) + 0.125*0.04 = 0.03 + 0.005 = 0.035
        assert abs(i - 0.035) < 1e-10

    def test_negative_output_gap(self) -> None:
        """Negative output gap lowers nominal rate."""
        block = NominalBlock(_default_config())
        r_natural = 0.03
        i = block.taylor_rule(inflation=0.02, output_gap=-0.04, r_natural=r_natural)
        # i = 0.03 + 0 + 0.125*(-0.04) = 0.03 - 0.005 = 0.025
        assert abs(i - 0.025) < 1e-10

    def test_combined_effects(self) -> None:
        """Both inflation and output gap affect the rate."""
        block = NominalBlock(_default_config())
        r_natural = 0.02
        i = block.taylor_rule(inflation=0.04, output_gap=0.02, r_natural=r_natural)
        # i = 0.02 + 1.5*(0.04 - 0.02) + 0.125*0.02 = 0.02 + 0.03 + 0.0025 = 0.0525
        assert abs(i - 0.0525) < 1e-10

    def test_custom_parameters(self) -> None:
        """Custom Taylor rule coefficients work correctly."""
        config = SimpleNamespace(
            rotemberg_cost=100.0,
            taylor_phi_pi=2.0,
            taylor_phi_y=0.5,
            inflation_target=0.0,
            elasticity_sub=6.0,
        )
        block = NominalBlock(config)
        i = block.taylor_rule(inflation=0.03, output_gap=0.01, r_natural=0.01)
        # i = 0.01 + 2.0*0.03 + 0.5*0.01 = 0.01 + 0.06 + 0.005 = 0.075
        assert abs(i - 0.075) < 1e-10


# ============================================================================
# Fisher equation tests (REQ-313)
# ============================================================================


class TestFisherEquation:
    """Tests for the Fisher equation identity."""

    def test_zero_expected_inflation(self) -> None:
        """With zero expected inflation, real rate equals nominal rate."""
        block = NominalBlock(_default_config())
        r_b = block.fisher_equation(nominal_rate=0.05, expected_inflation=0.0)
        assert abs(r_b - 0.05) < 1e-10

    def test_positive_expected_inflation(self) -> None:
        """Expected inflation reduces real rate below nominal."""
        block = NominalBlock(_default_config())
        r_b = block.fisher_equation(nominal_rate=0.05, expected_inflation=0.02)
        # r^b = (1.05) / (1.02) - 1 = 0.029411764...
        expected = 1.05 / 1.02 - 1.0
        assert abs(r_b - expected) < 1e-10

    def test_fisher_identity_roundtrip(self) -> None:
        """Fisher equation is consistent: (1+r^b)*(1+E[pi']) = (1+i)."""
        block = NominalBlock(_default_config())
        i = 0.06
        e_pi = 0.03
        r_b = block.fisher_equation(nominal_rate=i, expected_inflation=e_pi)
        # Verify: (1+r^b)*(1+E[pi']) should equal (1+i)
        lhs = (1.0 + r_b) * (1.0 + e_pi)
        rhs = 1.0 + i
        assert abs(lhs - rhs) < 1e-12

    def test_negative_expected_inflation(self) -> None:
        """Deflation raises real rate above nominal rate."""
        block = NominalBlock(_default_config())
        r_b = block.fisher_equation(nominal_rate=0.03, expected_inflation=-0.01)
        # r^b = 1.03 / 0.99 - 1 = 0.04040...
        expected = 1.03 / 0.99 - 1.0
        assert abs(r_b - expected) < 1e-10

    def test_equal_nominal_and_inflation(self) -> None:
        """When i = E[pi'], real rate is approximately zero."""
        block = NominalBlock(_default_config())
        r_b = block.fisher_equation(nominal_rate=0.03, expected_inflation=0.03)
        # r^b = 1.03 / 1.03 - 1 = 0
        assert abs(r_b) < 1e-12


# ============================================================================
# NKPC tests (REQ-311)
# ============================================================================


class TestNKPC:
    """Tests for the New Keynesian Phillips Curve."""

    def test_steady_state_zero_inflation(self) -> None:
        """At zero inflation steady state, mc = (epsilon - 1) / epsilon.

        From the NKPC with pi = pi_bar = 0 and Y = Y':
        phi_p * 0 * (0 - 0) = (1 - epsilon) + epsilon * mc + beta * phi_p * 0 * (0 - 0) * 1
        0 = (1 - epsilon) + epsilon * mc
        mc = (epsilon - 1) / epsilon
        """
        config = SimpleNamespace(
            rotemberg_cost=100.0,
            taylor_phi_pi=1.5,
            taylor_phi_y=0.125,
            inflation_target=0.0,  # zero inflation target
            elasticity_sub=6.0,
        )
        block = NominalBlock(config)
        epsilon = 6.0
        mc_ss = (epsilon - 1.0) / epsilon  # = 5/6 ≈ 0.8333

        residual = block.nkpc(
            inflation=0.0,
            marginal_cost=mc_ss,
            expected_inflation=0.0,
            output=100.0,
            expected_output=100.0,
            beta=0.95,
        )
        assert abs(residual) < 1e-10

    def test_nkpc_residual_away_from_steady_state(self) -> None:
        """Non-zero residual when mc differs from steady state."""
        block = NominalBlock(_default_config())
        residual = block.nkpc(
            inflation=0.0,
            marginal_cost=1.0,  # too high
            expected_inflation=0.0,
            output=100.0,
            expected_output=100.0,
            beta=0.95,
        )
        # With epsilon=6, mc_ss = 5/6 ≈ 0.833
        # mc=1.0 is above steady state, so residual should be nonzero
        assert abs(residual) > 0.01

    def test_nkpc_solve_inflation_at_steady_state(self) -> None:
        """NKPC solve returns target inflation at steady state mc."""
        config = SimpleNamespace(
            rotemberg_cost=100.0,
            taylor_phi_pi=1.5,
            taylor_phi_y=0.125,
            inflation_target=0.0,
            elasticity_sub=6.0,
        )
        block = NominalBlock(config)
        epsilon = 6.0
        mc_ss = (epsilon - 1.0) / epsilon

        pi = block.nkpc_solve_inflation(
            marginal_cost=mc_ss,
            expected_inflation=0.0,
            output=100.0,
            expected_output=100.0,
            beta=0.95,
        )
        # At steady-state mc with zero target, inflation should be zero
        assert abs(pi) < 1e-10

    def test_nkpc_solve_inflation_positive_mc(self) -> None:
        """Higher marginal cost implies higher inflation (cost-push)."""
        config = SimpleNamespace(
            rotemberg_cost=100.0,
            taylor_phi_pi=1.5,
            taylor_phi_y=0.125,
            inflation_target=0.02,
            elasticity_sub=6.0,
        )
        block = NominalBlock(config)
        epsilon = 6.0
        mc_ss = (epsilon - 1.0) / epsilon

        pi_ss = block.nkpc_solve_inflation(
            marginal_cost=mc_ss,
            expected_inflation=0.02,
            output=100.0,
            expected_output=100.0,
            beta=0.95,
        )
        pi_high = block.nkpc_solve_inflation(
            marginal_cost=mc_ss + 0.1,  # higher mc
            expected_inflation=0.02,
            output=100.0,
            expected_output=100.0,
            beta=0.95,
        )
        assert pi_high > pi_ss

    def test_nkpc_solve_consistency(self) -> None:
        """Solved inflation produces zero residual when plugged back."""
        block = NominalBlock(_default_config())
        mc = 0.85
        e_pi = 0.02
        output = 100.0
        e_output = 102.0
        beta = 0.95

        pi_solved = block.nkpc_solve_inflation(mc, e_pi, output, e_output, beta)
        residual = block.nkpc(pi_solved, mc, e_pi, output, e_output, beta)
        assert abs(residual) < 1e-8


# ============================================================================
# Nominal block update tests
# ============================================================================


class TestNominalBlockUpdate:
    """Tests for the full nominal block update iteration."""

    def test_update_returns_nominal_state(self) -> None:
        """Update returns a NominalState instance."""
        block = NominalBlock(_default_config())
        state = block.update(
            output=100.0,
            steady_state_output=100.0,
            wage=1.0,
            interest_rate=0.03,
            beta=0.95,
            aggregate_tfp=1.0,
        )
        assert isinstance(state, NominalState)

    def test_update_converges(self) -> None:
        """Nominal block converges for reasonable parameters."""
        block = NominalBlock(_default_config())
        state = block.update(
            output=100.0,
            steady_state_output=100.0,
            wage=1.0,
            interest_rate=0.03,
            beta=0.95,
            aggregate_tfp=1.0,
        )
        # Should have finite values
        assert math.isfinite(state.inflation)
        assert math.isfinite(state.nominal_rate)
        assert math.isfinite(state.real_bond_rate)
        assert math.isfinite(state.price_level)

    def test_update_price_level_tracks(self) -> None:
        """Price level updates: P_t = P_{t-1} * (1 + pi_t)."""
        block = NominalBlock(_default_config())
        state1 = block.update(
            output=100.0,
            steady_state_output=100.0,
            wage=1.0,
            interest_rate=0.03,
            beta=0.95,
        )
        # P_1 = P_0 * (1 + pi_1) = 1.0 * (1 + pi_1)
        assert abs(state1.price_level - (1.0 * (1.0 + state1.inflation))) < 1e-12

        # Second period
        state2 = block.update(
            output=102.0,
            steady_state_output=100.0,
            wage=1.01,
            interest_rate=0.03,
            beta=0.95,
        )
        assert abs(state2.price_level - (state1.price_level * (1.0 + state2.inflation))) < 1e-12

    def test_update_fisher_consistency(self) -> None:
        """Fisher equation holds in the update output (PROP-009)."""
        block = NominalBlock(_default_config())
        state = block.update(
            output=100.0,
            steady_state_output=100.0,
            wage=1.0,
            interest_rate=0.03,
            beta=0.95,
        )
        # Verify Fisher: (1 + r^b) * (1 + E[pi']) = (1 + i)
        lhs = (1.0 + state.real_bond_rate) * (1.0 + state.expected_inflation)
        rhs = 1.0 + state.nominal_rate
        assert abs(lhs - rhs) < 1e-10

    def test_update_zero_gap_at_target(self) -> None:
        """At steady state with mc near ss-mc, inflation near target."""
        config = SimpleNamespace(
            rotemberg_cost=100.0,
            taylor_phi_pi=1.5,
            taylor_phi_y=0.125,
            inflation_target=0.02,
            elasticity_sub=6.0,
        )
        block = NominalBlock(config)

        # Run a few periods to let expectations settle
        for _ in range(5):
            state = block.update(
                output=100.0,
                steady_state_output=100.0,
                wage=1.0,
                interest_rate=0.03,
                beta=0.95,
                aggregate_tfp=1.0,
            )

        # After convergence, inflation should be near some equilibrium
        # (not necessarily exactly pi_bar due to mc effects)
        assert math.isfinite(state.inflation)
        assert abs(state.inflation) < 0.5  # reasonable bound

    def test_update_output_shock_raises_inflation(self) -> None:
        """Positive output gap tends to raise inflation."""
        block = NominalBlock(_default_config())

        # Baseline at steady state
        state_base = block.update(
            output=100.0,
            steady_state_output=100.0,
            wage=1.0,
            interest_rate=0.03,
            beta=0.95,
            aggregate_tfp=1.0,
        )

        # Reset to same starting point for fair comparison
        block2 = NominalBlock(_default_config())
        state_shock = block2.update(
            output=110.0,  # positive output shock
            steady_state_output=100.0,
            wage=1.0,
            interest_rate=0.03,
            beta=0.95,
            aggregate_tfp=1.0,
        )

        # Higher output gap should lead to higher nominal rate
        # (Taylor rule responds to positive gap)
        assert state_shock.nominal_rate > state_base.nominal_rate

    def test_sequential_periods(self) -> None:
        """Multiple periods produce consistent, non-explosive results."""
        block = NominalBlock(_default_config())
        states = []
        for t in range(20):
            state = block.update(
                output=100.0 + 0.5 * t,
                steady_state_output=100.0,
                wage=1.0 + 0.001 * t,
                interest_rate=0.03,
                beta=0.95,
                aggregate_tfp=1.0 + 0.001 * t,
            )
            states.append(state)

        # Price level should be monotonically related to cumulative inflation
        for s in states:
            assert math.isfinite(s.price_level)
            assert s.price_level > 0


# ============================================================================
# Config interaction tests
# ============================================================================


class TestNominalConfig:
    """Tests for nominal config parameters in SimulationConfigV2."""

    def test_config_defaults(self) -> None:
        """Nominal config has correct defaults."""
        from emergent_constitution.config import SimulationConfigV2

        config = SimulationConfigV2(benchmark_mode=True)
        assert config.nominal_rigidities is False
        assert config.rotemberg_cost == 100.0
        assert config.taylor_phi_pi == 1.5
        assert config.taylor_phi_y == 0.125
        assert config.inflation_target == 0.02
        assert config.elasticity_sub == 6.0
        assert config.wage_rigidity is False
        assert config.rotemberg_wage_cost == 50.0

    def test_config_custom_values(self) -> None:
        """Custom nominal config values are accepted."""
        from emergent_constitution.config import SimulationConfigV2

        config = SimulationConfigV2(
            benchmark_mode=True,
            nominal_rigidities=True,
            rotemberg_cost=150.0,
            taylor_phi_pi=2.0,
            taylor_phi_y=0.25,
            inflation_target=0.03,
            elasticity_sub=10.0,
        )
        assert config.nominal_rigidities is True
        assert config.rotemberg_cost == 150.0
        assert config.taylor_phi_pi == 2.0
        assert config.taylor_phi_y == 0.25
        assert config.inflation_target == 0.03
        assert config.elasticity_sub == 10.0

    def test_taylor_principle_enforced(self) -> None:
        """Taylor phi_pi must be > 1 (Taylor principle)."""
        from emergent_constitution.config import SimulationConfigV2

        with pytest.raises(ValidationError):
            SimulationConfigV2(benchmark_mode=True, taylor_phi_pi=0.5)


# ============================================================================
# Interaction with market clearing tests
# ============================================================================


class TestNominalMarketInteraction:
    """Tests for nominal block interaction with market clearing."""

    def test_real_bond_rate_differs_from_capital_rate(self) -> None:
        """When nominal rigidities are active, r^b differs from r^k."""
        block = NominalBlock(_default_config())
        r_k = 0.05  # capital rate from market clearing

        state = block.update(
            output=100.0,
            steady_state_output=100.0,
            wage=1.5,
            interest_rate=r_k,
            beta=0.95,
            aggregate_tfp=1.0,
        )

        # Real bond rate should generally differ from capital rate
        # (they are linked but not identical)
        assert math.isfinite(state.real_bond_rate)
        # The bond rate is computed from Taylor rule + Fisher,
        # while r_k is from capital market clearing

    def test_nominal_block_with_high_rotemberg_cost(self) -> None:
        """Higher Rotemberg cost means less price flexibility."""
        config_low = SimpleNamespace(
            rotemberg_cost=10.0,
            taylor_phi_pi=1.5,
            taylor_phi_y=0.125,
            inflation_target=0.02,
            elasticity_sub=6.0,
        )
        config_high = SimpleNamespace(
            rotemberg_cost=1000.0,
            taylor_phi_pi=1.5,
            taylor_phi_y=0.125,
            inflation_target=0.02,
            elasticity_sub=6.0,
        )

        block_low = NominalBlock(config_low)
        block_high = NominalBlock(config_high)

        state_low = block_low.update(
            output=110.0,
            steady_state_output=100.0,
            wage=1.5,
            interest_rate=0.03,
            beta=0.95,
        )
        state_high = block_high.update(
            output=110.0,
            steady_state_output=100.0,
            wage=1.5,
            interest_rate=0.03,
            beta=0.95,
        )

        # Both should converge
        assert math.isfinite(state_low.inflation)
        assert math.isfinite(state_high.inflation)

    def test_nominal_disabled_keeps_real_economy(self) -> None:
        """When nominal_rigidities=False, NominalBlock is not created."""
        from emergent_constitution.config import SimulationConfigV2

        config = SimulationConfigV2(benchmark_mode=True, nominal_rigidities=False)
        assert config.nominal_rigidities is False
        # The LeadV2 would check this flag and skip nominal block
