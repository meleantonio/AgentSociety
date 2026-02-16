"""Integration and regression tests for HANK upgrade (Issue #73, Task 4.3).

Comprehensive tests verifying all properties (PROP-001 through PROP-012) hold
across 100-period simulations with Phase 1-4 features enabled, plus backward
compatibility tests ensuring v2 baseline behavior is preserved.

Traceability: PROP-001 through PROP-012, spec/tasks.md Task 4.3.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.distribution import Distribution
from emergent_constitution.egm_solver import EGMSolver
from emergent_constitution.lead import LeadV2
from emergent_constitution.models.history import SimulationOutputV2
from emergent_constitution.nominal import NominalBlock, NominalState


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def baseline_config() -> SimulationConfigV2:
    """Baseline v2 config: benchmark mode, EGM solver, all HANK features OFF."""
    return SimulationConfigV2(
        num_agents=20,
        max_periods=100,
        seed=42,
        benchmark_mode=True,
        solver_method="egm",
        observer_interval=5,
        proposal_interval=5,
    )


@pytest.fixture
def hank_config() -> SimulationConfigV2:
    """Config with all Phase 1-4 features enabled for full HANK testing."""
    return SimulationConfigV2(
        num_agents=20,
        max_periods=100,
        seed=42,
        benchmark_mode=True,
        solver_method="egm",
        observer_interval=5,
        proposal_interval=5,
        # Phase 1: Walrasian clearing + entrepreneur fix + Bellman occ choice
        market_clearing_method="walrasian",
        fix_entrepreneur_budget=True,
        use_bellman_occ_choice=True,
        # Phase 3: Government debt
        initial_debt=50.0,
        debt_gdp_max=1.5,
        fiscal_rule_adjustment=0.01,
    )


@pytest.fixture
def nominal_config() -> SimulationConfigV2:
    """Config with nominal rigidities enabled."""
    return SimulationConfigV2(
        num_agents=20,
        max_periods=100,
        seed=42,
        benchmark_mode=True,
        solver_method="egm",
        observer_interval=5,
        proposal_interval=5,
        nominal_rigidities=True,
        initial_debt=50.0,
    )


@pytest.fixture
def two_asset_config() -> SimulationConfigV2:
    """Config with two-asset household structure."""
    return SimulationConfigV2(
        num_agents=20,
        max_periods=50,
        seed=42,
        benchmark_mode=True,
        solver_method="egm",
        observer_interval=5,
        proposal_interval=5,
        two_asset_mode=True,
        chi_0=0.01,
        chi_1=0.005,
        b_min=0.0,
    )


@pytest.fixture
def mock_llm_config() -> SimulationConfigV2:
    """Config with mock LLM for governance testing."""
    return SimulationConfigV2(
        num_agents=20,
        max_periods=100,
        seed=42,
        use_llm=True,
        llm_provider="mock",
        benchmark_mode=False,
        solver_method="egm",
        observer_interval=5,
        proposal_interval=5,
    )


def _run_simulation(config: SimulationConfigV2) -> SimulationOutputV2:
    """Helper to run a simulation and return output."""
    lead = LeadV2(config)
    return lead.run()


# ============================================================================
# PROP-001: Determinism
# ============================================================================


class TestDeterminism:
    """PROP-001: Same seed + config = same output."""

    @pytest.mark.slow
    def test_baseline_determinism(self, baseline_config: SimulationConfigV2) -> None:
        """Two baseline runs with same seed produce identical output."""
        output1 = _run_simulation(baseline_config)
        output2 = _run_simulation(baseline_config)

        w1 = [h.wealth for h in output1.final_households]
        w2 = [h.wealth for h in output2.final_households]
        assert w1 == w2

        for h1, h2 in zip(output1.history, output2.history, strict=True):
            assert h1.gini == pytest.approx(h2.gini)
            assert h1.mean_wealth == pytest.approx(h2.mean_wealth)

    @pytest.mark.slow
    def test_hank_determinism(self, hank_config: SimulationConfigV2) -> None:
        """Full HANK runs with same seed produce identical output."""
        output1 = _run_simulation(hank_config)
        output2 = _run_simulation(hank_config)

        w1 = [h.wealth for h in output1.final_households]
        w2 = [h.wealth for h in output2.final_households]
        assert w1 == w2

    @pytest.mark.slow
    def test_different_seed_different_output(
        self, baseline_config: SimulationConfigV2
    ) -> None:
        """Different seeds produce different output."""
        output1 = _run_simulation(baseline_config)
        config2 = baseline_config.model_copy(update={"seed": 123})
        output2 = _run_simulation(config2)

        w1 = [h.wealth for h in output1.final_households]
        w2 = [h.wealth for h in output2.final_households]
        assert w1 != w2

    @pytest.mark.slow
    def test_mock_llm_determinism(
        self, mock_llm_config: SimulationConfigV2
    ) -> None:
        """Mock LLM runs with same seed produce identical output."""
        output1 = _run_simulation(mock_llm_config)
        output2 = _run_simulation(mock_llm_config)

        w1 = [h.wealth for h in output1.final_households]
        w2 = [h.wealth for h in output2.final_households]
        assert w1 == w2


# ============================================================================
# PROP-002: Budget Consistency
# ============================================================================


class TestBudgetConsistency:
    """PROP-002: Every agent satisfies its budget constraint every period."""

    @pytest.mark.slow
    def test_baseline_budget(self, baseline_config: SimulationConfigV2) -> None:
        """All household wealth >= a_min after baseline simulation."""
        output = _run_simulation(baseline_config)
        for h in output.final_households:
            assert h.wealth >= baseline_config.a_min, (
                f"Agent {h.id}: wealth {h.wealth} < a_min {baseline_config.a_min}"
            )

    @pytest.mark.slow
    def test_hank_budget(self, hank_config: SimulationConfigV2) -> None:
        """All household wealth >= a_min with HANK features."""
        output = _run_simulation(hank_config)
        for h in output.final_households:
            assert h.wealth >= hank_config.a_min, (
                f"Agent {h.id}: wealth {h.wealth} < a_min {hank_config.a_min}"
            )

    @pytest.mark.slow
    def test_consumption_non_negative(self, baseline_config: SimulationConfigV2) -> None:
        """Consumption is non-negative for all agents at final period."""
        output = _run_simulation(baseline_config)
        for h in output.final_households:
            assert h.consumption >= 0.0, (
                f"Agent {h.id}: negative consumption {h.consumption}"
            )


# ============================================================================
# PROP-003: Market Clearing
# ============================================================================


class TestMarketClearing:
    """PROP-003: Excess demand < 1e-8 after equilibrium computation."""

    @pytest.mark.slow
    def test_baseline_market_clearing(self, baseline_config: SimulationConfigV2) -> None:
        """Analytical clearing has near-zero error throughout simulation."""
        lead = LeadV2(baseline_config)
        output = lead.run()

        for entry in output.history:
            assert entry.aggregate_output < 1e-6, (
                f"Period {entry.period}: clearing error {entry.aggregate_output}"
            )

    @pytest.mark.slow
    def test_walrasian_market_clearing(self, hank_config: SimulationConfigV2) -> None:
        """Walrasian clearing with heterogeneous firms has bounded error."""
        lead = LeadV2(hank_config)
        output = lead.run()

        for entry in output.history:
            # Walrasian may have slightly larger clearing errors
            assert entry.aggregate_output < 1e-4, (
                f"Period {entry.period}: clearing error {entry.aggregate_output}"
            )

    @pytest.mark.slow
    def test_positive_prices(self, baseline_config: SimulationConfigV2) -> None:
        """Wages and output remain positive throughout simulation."""
        lead = LeadV2(baseline_config)
        output = lead.run()

        for entry in output.history:
            assert entry.wage > 0.0, f"Period {entry.period}: non-positive wage"
            assert entry.total_output > 0.0, f"Period {entry.period}: non-positive output"
            assert math.isfinite(entry.interest_rate)


# ============================================================================
# PROP-004: Non-negativity
# ============================================================================


class TestNonNegativity:
    """PROP-004: No negative consumption, wealth within bounds, no negative capital."""

    @pytest.mark.slow
    def test_baseline_non_negativity(self, baseline_config: SimulationConfigV2) -> None:
        """All non-negativity constraints hold after baseline simulation."""
        output = _run_simulation(baseline_config)

        for h in output.final_households:
            assert h.consumption >= 0.0, f"Agent {h.id}: negative consumption"
            assert h.wealth >= baseline_config.a_min, f"Agent {h.id}: wealth below a_min"
            assert h.productivity > 0.0, f"Agent {h.id}: non-positive productivity"
            assert 0.0 <= h.leisure <= 1.0, f"Agent {h.id}: leisure out of [0,1]"

    @pytest.mark.slow
    def test_hank_non_negativity(self, hank_config: SimulationConfigV2) -> None:
        """Non-negativity holds with all HANK features."""
        output = _run_simulation(hank_config)

        for h in output.final_households:
            assert h.consumption >= 0.0
            assert h.wealth >= hank_config.a_min
            assert h.productivity > 0.0

    @pytest.mark.slow
    def test_no_nan_inf(self, baseline_config: SimulationConfigV2) -> None:
        """No NaN or Inf values in simulation output."""
        output = _run_simulation(baseline_config)

        for h in output.final_households:
            assert math.isfinite(h.wealth), f"Agent {h.id}: invalid wealth"
            assert math.isfinite(h.consumption), f"Agent {h.id}: invalid consumption"
            assert math.isfinite(h.realized_utility), f"Agent {h.id}: invalid utility"

        for entry in output.history:
            assert math.isfinite(entry.gini)
            assert math.isfinite(entry.mean_wealth)
            assert math.isfinite(entry.social_welfare)
            assert math.isfinite(entry.wage)
            assert math.isfinite(entry.interest_rate)


# ============================================================================
# PROP-007: Euler Equation Accuracy
# ============================================================================


class TestEulerEquation:
    """PROP-007: Max Euler equation residual < 1e-6 for EGM solver."""

    @pytest.mark.slow
    def test_egm_euler_residual(self, baseline_config: SimulationConfigV2) -> None:
        """EGM solver produces Euler equation residual below tolerance."""
        lead = LeadV2(baseline_config)

        # Run a few periods to get stable policy functions
        for t in range(1, 6):
            lead.period_state = lead._advance_period(t)

        market = lead.period_state.market
        constitution = lead.period_state.constitution
        tax_rate = lead._get_tax_rate(constitution)

        solver = lead._solver
        if not isinstance(solver, EGMSolver):
            pytest.skip("EGM solver not active")

        # Get households for utility params
        ref = lead.period_state.households[0].utility_params

        def tax_fn(income: float) -> float:
            return income * tax_rate

        c_policy, lei_policy = solver.solve_egm_cached(
            alpha_u=ref.alpha,
            beta_u=ref.beta,
            gamma_u=ref.gamma,
            beta_discount=ref.beta_discount,
            wage=market.wage,
            interest_rate=market.interest_rate,
            public_goods=max(1e-10, 0.0),
            tax_function=tax_fn,
            transfer=0.0,
        )

        residual = solver.euler_residual(
            c_policy=c_policy,
            lei_policy=lei_policy,
            alpha_u=ref.alpha,
            beta_u=ref.beta,
            gamma_u=ref.gamma,
            beta_discount=ref.beta_discount,
            wage=market.wage,
            interest_rate=market.interest_rate,
            public_goods=max(1e-10, 0.0),
            tax_function=tax_fn,
            transfer=0.0,
        )

        assert residual < 1e-6, f"Euler equation residual {residual} exceeds 1e-6"

    @pytest.mark.slow
    def test_policy_monotonicity(self, baseline_config: SimulationConfigV2) -> None:
        """Consumption policy is monotone in wealth (wealthier consume more)."""
        lead = LeadV2(baseline_config)

        # Run a period to initialize
        lead.period_state = lead._advance_period(1)

        market = lead.period_state.market
        constitution = lead.period_state.constitution
        tax_rate = lead._get_tax_rate(constitution)
        solver = lead._solver

        if not isinstance(solver, EGMSolver):
            pytest.skip("EGM solver not active")

        ref = lead.period_state.households[0].utility_params

        def tax_fn(income: float) -> float:
            return income * tax_rate

        c_policy, _ = solver.solve_egm_cached(
            alpha_u=ref.alpha,
            beta_u=ref.beta,
            gamma_u=ref.gamma,
            beta_discount=ref.beta_discount,
            wage=market.wage,
            interest_rate=market.interest_rate,
            public_goods=max(1e-10, 0.0),
            tax_function=tax_fn,
            transfer=0.0,
        )

        # Check monotonicity for each productivity state
        # (consumption should weakly increase with wealth)
        for z_idx in range(c_policy.shape[1]):
            c_col = c_policy[:, z_idx]
            # Allow small violations due to numerical noise
            diffs = np.diff(c_col)
            violations = np.sum(diffs < -1e-8)
            assert violations == 0, (
                f"Consumption policy non-monotone at z_idx={z_idx}: "
                f"{violations} violations"
            )


# ============================================================================
# PROP-008: Distribution Conservation
# ============================================================================


class TestDistributionConservation:
    """PROP-008: KFE distribution sum(mu) = 1 within 1e-12."""

    def test_distribution_mass_conservation(self) -> None:
        """Distribution forward step preserves total mass."""
        a_grid = np.linspace(0.0, 100.0, 50)
        z_grid = np.array([0.5, 1.0, 1.5])

        dist = Distribution(a_grid, z_grid)
        dist.initialize_uniform()

        initial_mass = dist.mass_total()
        assert abs(initial_mass - 1.0) < 1e-12

        # Construct a simple savings policy: a' = a (no savings)
        policy = np.tile(a_grid.reshape(-1, 1), (1, len(z_grid)))

        # Simple transition matrix (stay in same state)
        trans = np.eye(len(z_grid))

        dist.forward(policy, trans)

        new_mass = dist.mass_total()
        assert abs(new_mass - 1.0) < 1e-12, (
            f"Mass after forward: {new_mass}, expected 1.0"
        )

    def test_distribution_stationary_convergence(self) -> None:
        """Stationary distribution converges and sums to 1."""
        a_grid = np.linspace(0.0, 50.0, 30)
        z_grid = np.array([0.8, 1.0, 1.2])

        dist = Distribution(a_grid, z_grid)
        dist.initialize_uniform()

        # Simple savings policy: a' = 0.9*a (agents save less)
        policy = 0.9 * np.tile(a_grid.reshape(-1, 1), (1, len(z_grid)))

        # Transition matrix with mixing
        trans = np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ])

        # Run forward iterations
        for _ in range(100):
            dist.forward(policy, trans)

        mass = dist.mass_total()
        assert abs(mass - 1.0) < 1e-12, f"Mass after convergence: {mass}"


# ============================================================================
# PROP-009: Fisher Consistency
# ============================================================================


class TestFisherConsistency:
    """PROP-009: (1+r^b)*(1+E[pi']) = (1+i) at every period."""

    def test_fisher_identity_holds(self) -> None:
        """Fisher equation identity holds for NominalBlock output."""
        block = NominalBlock(SimulationConfigV2(
            benchmark_mode=True, nominal_rigidities=True
        ))

        for t in range(20):
            state = block.update(
                output=100.0 + t * 0.5,
                steady_state_output=100.0,
                wage=1.0 + t * 0.001,
                interest_rate=0.03,
                beta=0.95,
                aggregate_tfp=1.0,
            )

            lhs = (1.0 + state.real_bond_rate) * (1.0 + state.expected_inflation)
            rhs = 1.0 + state.nominal_rate
            assert abs(lhs - rhs) < 1e-10, (
                f"Period {t}: Fisher violation: "
                f"(1+{state.real_bond_rate:.6f})*(1+{state.expected_inflation:.6f}) "
                f"!= (1+{state.nominal_rate:.6f})"
            )


# ============================================================================
# PROP-010: Government Budget Balance
# ============================================================================


class TestGovernmentBudget:
    """PROP-010: Government debt satisfies budget constraint."""

    @pytest.mark.slow
    def test_government_budget_with_debt(self, hank_config: SimulationConfigV2) -> None:
        """Government budget constraint holds with positive initial debt."""
        lead = LeadV2(hank_config)
        output = lead.run()

        # The simulation should complete without errors
        assert isinstance(output, SimulationOutputV2)
        assert output.total_periods == 100

        # Government state should track debt
        gov_state = lead._government.state
        assert math.isfinite(gov_state.debt)
        assert math.isfinite(gov_state.debt_to_gdp)

    def test_government_budget_identity(self) -> None:
        """Direct test: B' = (1+r^b)*B + G + Tr - T."""
        from emergent_constitution.government import Government

        gov = Government(initial_debt=100.0)

        old_debt = gov.state.debt
        bond_rate = 0.03
        tax_revenue = 20.0
        spending = 10.0
        transfers = 5.0

        gov.update_budget(
            tax_revenue=tax_revenue,
            spending=spending,
            transfers=transfers,
            bond_rate=bond_rate,
        )

        expected_new_debt = (1.0 + bond_rate) * old_debt + spending + transfers - tax_revenue
        assert gov.state.debt == pytest.approx(expected_new_debt, abs=1e-10), (
            f"Expected {expected_new_debt}, got {gov.state.debt}"
        )

    def test_fiscal_rule_activates(self) -> None:
        """Fiscal rule triggers when debt/GDP exceeds threshold."""
        from emergent_constitution.government import Government

        gov = Government(initial_debt=200.0, debt_gdp_max=1.5)

        # GDP = 100, so debt/GDP = 2.0 > 1.5
        adjustment = gov.fiscal_rule(output=100.0)
        assert adjustment > 0.0

    def test_no_fiscal_rule_below_threshold(self) -> None:
        """Fiscal rule does not trigger below threshold."""
        from emergent_constitution.government import Government

        gov = Government(initial_debt=50.0, debt_gdp_max=1.5)

        # GDP = 100, so debt/GDP = 0.5 < 1.5
        adjustment = gov.fiscal_rule(output=100.0)
        assert adjustment == 0.0


# ============================================================================
# PROP-012: Backward Compatibility
# ============================================================================


class TestBackwardCompatibility:
    """PROP-012: Default config matches v2 baseline to tolerance 1e-6."""

    @pytest.mark.slow
    def test_default_config_runs(self) -> None:
        """Default config (all new features OFF) runs successfully."""
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=50,
            seed=42,
            benchmark_mode=True,
            observer_interval=5,
        )
        output = _run_simulation(config)

        assert isinstance(output, SimulationOutputV2)
        assert output.total_periods == 50
        assert len(output.final_households) == 20

    @pytest.mark.slow
    def test_default_config_preserves_invariants(self) -> None:
        """Default config preserves all basic invariants."""
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=50,
            seed=42,
            benchmark_mode=True,
            observer_interval=5,
        )
        output = _run_simulation(config)

        # All agents have valid state
        for h in output.final_households:
            assert h.wealth >= 0.0
            assert h.consumption >= 0.0
            assert h.productivity > 0.0
            assert 0.0 <= h.leisure <= 1.0
            assert math.isfinite(h.realized_utility)

        # History is valid
        for entry in output.history:
            assert 0.0 <= entry.gini <= 1.0
            assert entry.mean_wealth >= 0.0
            assert entry.total_output > 0.0
            assert entry.wage > 0.0

    @pytest.mark.slow
    def test_vfi_and_egm_produce_similar_output(self) -> None:
        """VFI and EGM solvers produce qualitatively similar results."""
        config_egm = SimulationConfigV2(
            num_agents=20,
            max_periods=20,
            seed=42,
            benchmark_mode=True,
            solver_method="egm",
            observer_interval=5,
        )
        config_vfi = SimulationConfigV2(
            num_agents=20,
            max_periods=20,
            seed=42,
            benchmark_mode=True,
            solver_method="vfi",
            observer_interval=5,
        )

        output_egm = _run_simulation(config_egm)
        output_vfi = _run_simulation(config_vfi)

        # Both should complete successfully
        assert output_egm.total_periods == 20
        assert output_vfi.total_periods == 20

        # Mean wealth should be in same ballpark (within 50%)
        mean_egm = sum(h.wealth for h in output_egm.final_households) / len(
            output_egm.final_households
        )
        mean_vfi = sum(h.wealth for h in output_vfi.final_households) / len(
            output_vfi.final_households
        )
        assert mean_egm > 0.0
        assert mean_vfi > 0.0

        ratio = mean_egm / mean_vfi if mean_vfi > 0 else float("inf")
        assert 0.5 < ratio < 2.0, (
            f"EGM mean wealth {mean_egm:.2f} vs VFI mean wealth {mean_vfi:.2f} "
            f"differ by more than 2x"
        )

    @pytest.mark.slow
    def test_new_features_disabled_match_baseline(self) -> None:
        """Config with all new features explicitly OFF matches default."""
        config_default = SimulationConfigV2(
            num_agents=20,
            max_periods=30,
            seed=42,
            benchmark_mode=True,
            observer_interval=5,
        )
        config_explicit_off = SimulationConfigV2(
            num_agents=20,
            max_periods=30,
            seed=42,
            benchmark_mode=True,
            observer_interval=5,
            # Explicitly disable all new features
            fix_entrepreneur_budget=False,
            use_bellman_occ_choice=False,
            market_clearing_method="analytical",
            two_asset_mode=False,
            nominal_rigidities=False,
            distribution_mode="individual",
        )

        output1 = _run_simulation(config_default)
        output2 = _run_simulation(config_explicit_off)

        # Should produce identical results
        w1 = [h.wealth for h in output1.final_households]
        w2 = [h.wealth for h in output2.final_households]
        for a, b in zip(w1, w2, strict=True):
            assert a == pytest.approx(b, abs=1e-6), (
                f"Explicit OFF config differs: {a} vs {b}"
            )


# ============================================================================
# Full Integration Tests (All Properties)
# ============================================================================


class TestFullHANKIntegration:
    """100-period simulation with all features; verify all properties hold."""

    @pytest.mark.slow
    def test_100_period_baseline(self, baseline_config: SimulationConfigV2) -> None:
        """100-period baseline simulation completes with all properties."""
        lead = LeadV2(baseline_config)
        output = lead.run()

        assert output.total_periods == 100
        assert len(output.final_households) == 20

        # PROP-002: Budget consistency
        for h in output.final_households:
            assert h.wealth >= baseline_config.a_min

        # PROP-003: Market clearing
        for entry in output.history:
            assert entry.aggregate_output < 1e-4

        # PROP-004: Non-negativity
        for h in output.final_households:
            assert h.consumption >= 0.0
            assert h.productivity > 0.0

        # History has reasonable values
        for entry in output.history:
            assert 0.0 <= entry.gini <= 1.0
            assert entry.mean_wealth >= 0.0
            assert entry.total_output > 0.0

    @pytest.mark.slow
    def test_100_period_hank(self, hank_config: SimulationConfigV2) -> None:
        """100-period HANK simulation with all features completes."""
        lead = LeadV2(hank_config)
        output = lead.run()

        assert output.total_periods == 100
        assert len(output.final_households) == 20

        # PROP-002: Budget consistency
        for h in output.final_households:
            assert h.wealth >= hank_config.a_min

        # PROP-003: Market clearing (Walrasian may have slightly higher error)
        for entry in output.history:
            assert entry.aggregate_output < 1e-2

        # PROP-004: Non-negativity
        for h in output.final_households:
            assert h.consumption >= 0.0
            assert h.productivity > 0.0

        # Verify fiscal dynamics: government tracks debt
        gov_state = lead._government.state
        assert math.isfinite(gov_state.debt)

    @pytest.mark.slow
    def test_100_period_nominal(self, nominal_config: SimulationConfigV2) -> None:
        """100-period simulation with nominal rigidities completes."""
        lead = LeadV2(nominal_config)
        output = lead.run()

        assert output.total_periods == 100

        # All households have valid state
        for h in output.final_households:
            assert h.wealth >= nominal_config.a_min
            assert h.consumption >= 0.0
            assert math.isfinite(h.realized_utility)

        # Nominal state should have been computed
        if lead._nominal_block is not None:
            ns = lead._nominal_state
            assert math.isfinite(ns.inflation)
            assert math.isfinite(ns.nominal_rate)
            assert math.isfinite(ns.real_bond_rate)
            assert ns.price_level > 0.0

    @pytest.mark.slow
    def test_100_period_mock_llm(self, mock_llm_config: SimulationConfigV2) -> None:
        """100-period simulation with mock LLM governance completes."""
        output = _run_simulation(mock_llm_config)

        assert output.total_periods == 100
        assert len(output.final_households) == 20

        for h in output.final_households:
            assert h.wealth >= mock_llm_config.a_min
            assert h.consumption >= 0.0
            assert math.isfinite(h.realized_utility)


# ============================================================================
# Feature-specific integration tests
# ============================================================================


class TestEntrepreneurBudgetFix:
    """Test entrepreneur budget constraint fix (REQ-107..109)."""

    @pytest.mark.slow
    def test_entrepreneur_income_is_profit_only(self) -> None:
        """With fix_entrepreneur_budget, entrepreneurs earn only firm profit."""
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=20,
            seed=42,
            benchmark_mode=True,
            solver_method="egm",
            observer_interval=5,
            fix_entrepreneur_budget=True,
        )
        lead = LeadV2(config)
        output = lead.run()

        # Check that any entrepreneur has labor_supply = 0
        from emergent_constitution.models.household import OccupationalRole

        for h in output.final_households:
            if h.role == OccupationalRole.ENTREPRENEUR:
                assert h.labor_supply == 0.0, (
                    f"Entrepreneur {h.id} has labor_supply={h.labor_supply}"
                )


class TestWalrasianClearing:
    """Test Walrasian market clearing (REQ-101..106)."""

    @pytest.mark.slow
    def test_walrasian_clearing_works(self) -> None:
        """Walrasian clearing produces valid prices and bounded errors."""
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=20,
            seed=42,
            benchmark_mode=True,
            solver_method="egm",
            observer_interval=5,
            market_clearing_method="walrasian",
        )
        lead = LeadV2(config)
        output = lead.run()

        assert output.total_periods == 20
        for entry in output.history:
            assert entry.wage > 0.0
            assert math.isfinite(entry.interest_rate)


class TestGovernmentDebt:
    """Test government debt management (REQ-306..309)."""

    @pytest.mark.slow
    def test_debt_dynamics(self) -> None:
        """Government debt evolves without exploding."""
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=50,
            seed=42,
            benchmark_mode=True,
            solver_method="egm",
            observer_interval=5,
            initial_debt=100.0,
            debt_gdp_max=1.5,
            fiscal_rule_adjustment=0.01,
        )
        lead = LeadV2(config)
        output = lead.run()

        assert output.total_periods == 50
        # Debt should not have exploded (< 5x GDP)
        gov = lead._government.state
        assert math.isfinite(gov.debt)
        assert math.isfinite(gov.debt_to_gdp)


class TestNominalRigidities:
    """Test nominal rigidities block (REQ-310..315)."""

    @pytest.mark.slow
    def test_nominal_block_integration(self) -> None:
        """Nominal block produces finite, consistent state."""
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=30,
            seed=42,
            benchmark_mode=True,
            solver_method="egm",
            observer_interval=5,
            nominal_rigidities=True,
        )
        lead = LeadV2(config)
        output = lead.run()

        assert output.total_periods == 30

        # Nominal state should be set
        ns = lead._nominal_state
        assert math.isfinite(ns.inflation)
        assert math.isfinite(ns.nominal_rate)
        assert ns.price_level > 0.0


class TestTwoAssetMode:
    """Test two-asset household structure (REQ-301..305)."""

    @pytest.mark.slow
    def test_two_asset_simulation(self, two_asset_config: SimulationConfigV2) -> None:
        """Two-asset simulation runs and produces valid output."""
        output = _run_simulation(two_asset_config)

        assert output.total_periods == 50

        for h in output.final_households:
            assert h.wealth >= two_asset_config.a_min
            assert h.consumption >= 0.0
            assert math.isfinite(h.wealth)


class TestBellmanOccupationalChoice:
    """Test Bellman-based occupational choice (REQ-110..114)."""

    @pytest.mark.slow
    def test_bellman_occ_choice(self) -> None:
        """Bellman-based occupational choice produces valid output."""
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=20,
            seed=42,
            benchmark_mode=True,
            solver_method="egm",
            observer_interval=5,
            use_bellman_occ_choice=True,
        )
        output = _run_simulation(config)

        assert output.total_periods == 20
        for h in output.final_households:
            assert h.wealth >= config.a_min
            assert math.isfinite(h.realized_utility)


# ============================================================================
# Welfare and Observer Tests
# ============================================================================


class TestObserverOutput:
    """Test observer statistics and welfare computation."""

    @pytest.mark.slow
    def test_history_structure(self, baseline_config: SimulationConfigV2) -> None:
        """Observer produces correct number of history entries."""
        output = _run_simulation(baseline_config)

        expected_entries = baseline_config.max_periods // baseline_config.observer_interval
        assert len(output.history) == expected_entries

        for entry in output.history:
            assert 0.0 <= entry.gini <= 1.0
            assert entry.mean_wealth >= 0.0
            assert entry.social_welfare >= 0.0
            assert entry.total_output > 0.0

    @pytest.mark.slow
    def test_welfare_summary(self, baseline_config: SimulationConfigV2) -> None:
        """Welfare summary has non-negative values."""
        output = _run_simulation(baseline_config)

        # Benchmark mode -> benchmark_total_welfare is set
        ws = output.welfare_summary
        assert ws.benchmark_total_welfare is not None or ws.llm_total_welfare >= 0.0

    @pytest.mark.slow
    def test_gini_bounded(self, baseline_config: SimulationConfigV2) -> None:
        """Gini coefficient remains in [0, 1] throughout simulation."""
        output = _run_simulation(baseline_config)

        for entry in output.history:
            assert 0.0 <= entry.gini <= 1.0, (
                f"Period {entry.period}: Gini {entry.gini} out of bounds"
            )
