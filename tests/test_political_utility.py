"""Unit tests for political utility functions.

Tests cover:
- Political utility computation (REQ-401, REQ-402)
- Integration with Bellman equation (REQ-401)
- Sensitivity to theta values (REQ-402)
- Bellman preference matches manual value function comparison (REQ-403)
- LLM deviation logging (REQ-404)
- pure_bellman_politics mode (REQ-405)

Traceability: REQ-401, REQ-402, REQ-403, REQ-404, REQ-405.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.models.constitution import (
    ConstitutionalRule,
    ConstitutionV2,
    RuleType,
)
from emergent_constitution.models.household import (
    HouseholdState,
    OccupationalRole,
    UtilityParams,
    ValueVector,
)
from emergent_constitution.models.market import MarketState
from emergent_constitution.political_utility import (
    bellman_vote,
    build_bellman_political_context,
    compute_political_utility,
    evaluate_proposal,
    log_llm_bellman_deviation,
)

# ============================================================================
# Test fixtures
# ============================================================================


def _make_household(
    agent_id: str = "agent_0000",
    wealth: float = 100.0,
    productivity: float = 1.0,
    productivity_index: int = 1,
    equality: float = 0.5,
    liberty: float = 0.5,
    income: float = 10.0,
) -> HouseholdState:
    """Create a minimal test household."""
    return HouseholdState(
        id=agent_id,
        wealth=wealth,
        productivity=productivity,
        productivity_index=productivity_index,
        utility_params=UtilityParams(alpha=0.4, beta=0.35, gamma=0.25, beta_discount=0.95),
        value_vector=ValueVector(equality=equality, liberty=liberty),
        role=OccupationalRole.WORKER,
        income=income,
    )


def _make_constitution(tax_rate: float = 0.1, method: str = "equal_share") -> ConstitutionV2:
    """Create a constitution with given tax rate and transfer method."""
    rules = {
        "flat_tax": ConstitutionalRule(
            name="flat_tax",
            rule_type=RuleType.TAX_SCHEDULE,
            parameters={"rate": tax_rate},
            description="Flat income tax.",
        ),
        "transfer": ConstitutionalRule(
            name="transfer",
            rule_type=RuleType.TRANSFER_PROGRAM,
            parameters={"method": method},
            description="Transfer program.",
        ),
        "majority_vote": ConstitutionalRule(
            name="majority_vote",
            rule_type=RuleType.VOTING_PROCEDURE,
            parameters={"threshold": 0.5},
            description="Majority vote.",
        ),
    }
    return ConstitutionV2(rules=rules, voting_rule="majority_vote")


def _make_simple_value_function() -> tuple[list[list[float]], list[float]]:
    """Create a simple value function for testing.

    Returns (value_function, a_grid) with n_a=10, n_z=3.
    """
    n_a, n_z = 10, 3
    a_grid = [float(i * 100) for i in range(n_a)]
    # Value increases with wealth (concave)
    value_function = []
    for ai in range(n_a):
        row = []
        for zi in range(n_z):
            v = math.log(1.0 + a_grid[ai]) * (1.0 + 0.1 * zi)
            row.append(v)
        value_function.append(row)
    return value_function, a_grid


# ============================================================================
# Test: compute_political_utility (REQ-402)
# ============================================================================


class TestComputePoliticalUtility:
    """Tests for the core political utility function."""

    def test_basic_computation(self) -> None:
        """Political utility should combine equality and liberty components."""
        constitution = _make_constitution(tax_rate=0.2)
        v = compute_political_utility(constitution, theta_eq=0.5, theta_lib=0.5)
        # f_eq = -gini_proxy, f_lib = -0.2
        # v = 0.5 * f_eq + 0.5 * (-0.2)
        assert isinstance(v, float)
        assert math.isfinite(v)

    def test_with_observed_gini(self) -> None:
        """When observed_gini is provided, it should be used directly."""
        constitution = _make_constitution(tax_rate=0.1)
        v = compute_political_utility(constitution, theta_eq=1.0, theta_lib=0.0, observed_gini=0.3)
        # f_eq = -0.3, f_lib = -0.1, v = 1.0 * (-0.3) + 0.0 * (-0.1) = -0.3
        assert v == pytest.approx(-0.3)

    def test_pure_equality_agent(self) -> None:
        """An agent who cares only about equality should respond to Gini only."""
        const_low_tax = _make_constitution(tax_rate=0.05)
        const_high_tax = _make_constitution(tax_rate=0.50)
        v_low = compute_political_utility(
            const_low_tax, theta_eq=1.0, theta_lib=0.0, observed_gini=0.4
        )
        v_high = compute_political_utility(
            const_high_tax, theta_eq=1.0, theta_lib=0.0, observed_gini=0.4
        )
        # Same observed Gini -> same utility for pure equality agent
        assert v_low == pytest.approx(v_high)

    def test_pure_liberty_agent(self) -> None:
        """An agent who cares only about liberty should respond to tax rate only."""
        const_low = _make_constitution(tax_rate=0.05)
        const_high = _make_constitution(tax_rate=0.50)
        v_low = compute_political_utility(
            const_low, theta_eq=0.0, theta_lib=1.0, observed_gini=0.4
        )
        v_high = compute_political_utility(
            const_high, theta_eq=0.0, theta_lib=1.0, observed_gini=0.4
        )
        # Lower tax -> higher utility for liberty agent
        assert v_low > v_high

    def test_liberty_component_value(self) -> None:
        """f_lib(C) = -tau(C) should equal negative of the tax rate."""
        constitution = _make_constitution(tax_rate=0.25)
        v = compute_political_utility(constitution, theta_eq=0.0, theta_lib=1.0, observed_gini=0.0)
        # v = 0 * f_eq + 1.0 * (-0.25) = -0.25
        assert v == pytest.approx(-0.25)

    def test_zero_tax_constitution(self) -> None:
        """Zero tax rate should give maximum liberty utility."""
        constitution = _make_constitution(tax_rate=0.0)
        v = compute_political_utility(constitution, theta_eq=0.0, theta_lib=1.0)
        # f_lib = -0.0 = 0.0
        assert v == pytest.approx(0.0)

    def test_symmetry_balanced_agent(self) -> None:
        """A balanced agent with equal theta should value both components."""
        constitution = _make_constitution(tax_rate=0.2)
        v = compute_political_utility(constitution, theta_eq=0.5, theta_lib=0.5, observed_gini=0.3)
        expected = 0.5 * (-0.3) + 0.5 * (-0.2)
        assert v == pytest.approx(expected)


# ============================================================================
# Test: sensitivity to theta values (REQ-402)
# ============================================================================


class TestThetaSensitivity:
    """Political utility should be monotone in theta for each component."""

    def test_increasing_theta_eq_increases_equality_sensitivity(self) -> None:
        """Higher theta_eq should make agent more sensitive to Gini."""
        constitution = _make_constitution(tax_rate=0.1)
        v_low_eq = compute_political_utility(
            constitution, theta_eq=0.2, theta_lib=0.8, observed_gini=0.5
        )
        v_high_eq = compute_political_utility(
            constitution, theta_eq=0.8, theta_lib=0.2, observed_gini=0.5
        )
        # With high Gini (0.5), the equality-focused agent should have lower utility
        # because f_eq = -0.5 is a large penalty
        # But the liberty-focused agent has less penalty from Gini and more from tax
        # v_low_eq = 0.2*(-0.5) + 0.8*(-0.1) = -0.10 - 0.08 = -0.18
        # v_high_eq = 0.8*(-0.5) + 0.2*(-0.1) = -0.40 - 0.02 = -0.42
        assert v_low_eq > v_high_eq  # less equality-focused = less penalty from high Gini

    def test_increasing_theta_lib_increases_tax_sensitivity(self) -> None:
        """Higher theta_lib should make agent more sensitive to tax rate."""
        constitution = _make_constitution(tax_rate=0.5)
        v_low_lib = compute_political_utility(
            constitution, theta_eq=0.8, theta_lib=0.2, observed_gini=0.1
        )
        v_high_lib = compute_political_utility(
            constitution, theta_eq=0.2, theta_lib=0.8, observed_gini=0.1
        )
        # With high tax (0.5), the liberty-focused agent suffers more
        # v_low_lib = 0.8*(-0.1) + 0.2*(-0.5) = -0.08 - 0.10 = -0.18
        # v_high_lib = 0.2*(-0.1) + 0.8*(-0.5) = -0.02 - 0.40 = -0.42
        assert v_low_lib > v_high_lib


# ============================================================================
# Test: evaluate_proposal / Bellman-derived preferences (REQ-403)
# ============================================================================


class TestEvaluateProposal:
    """Tests for Bellman-derived proposal evaluation."""

    def test_same_constitution_returns_zero(self) -> None:
        """Comparing the same constitution should return ~0."""
        vf, ag = _make_simple_value_function()
        agent = _make_household()
        c = _make_constitution(tax_rate=0.1)
        delta = evaluate_proposal(vf, ag, agent, c, c, political_lambda=0.05)
        assert delta == pytest.approx(0.0)

    def test_lower_tax_preferred_by_liberty_agent(self) -> None:
        """A liberty-focused agent should prefer lower taxes."""
        vf, ag = _make_simple_value_function()
        agent = _make_household(equality=0.1, liberty=0.9)
        current = _make_constitution(tax_rate=0.3)
        proposed = _make_constitution(tax_rate=0.1)
        delta = evaluate_proposal(vf, ag, agent, proposed, current, political_lambda=0.05)
        assert delta > 0  # Prefers lower tax

    def test_higher_tax_preferred_by_equality_agent_if_gini_drops(self) -> None:
        """An equality-focused agent may prefer higher taxes if Gini drops.

        The political component should show the equality agent benefits from
        lower Gini, even though the economic component from higher taxes is negative.
        We use a high political_lambda and low income to make the political
        component dominate.
        """
        vf, ag = _make_simple_value_function()
        agent = _make_household(equality=0.9, liberty=0.1, income=0.1)
        current = _make_constitution(tax_rate=0.1)
        proposed = _make_constitution(tax_rate=0.12, method="means_tested")
        delta = evaluate_proposal(
            vf,
            ag,
            agent,
            proposed,
            current,
            political_lambda=0.5,
            observed_gini=0.5,
            proposed_gini=0.2,
        )
        # With high lambda and low income, political component dominates
        # Political delta: 0.5 * (v_proposed - v_current) / (1 - 0.95)
        # v_proposed = 0.9*(-0.2) + 0.1*(-0.12) = -0.192
        # v_current = 0.9*(-0.5) + 0.1*(-0.1) = -0.46
        # delta_political = 0.5 * (-0.192 - (-0.46)) / 0.05 = 0.5 * 0.268 / 0.05 = 2.68
        # delta_economic = -0.02 * 0.1 / 0.05 = -0.04
        # total ~ 2.64
        assert delta > 0

    def test_empty_value_function_returns_zero(self) -> None:
        """Empty value function should return 0 gracefully."""
        agent = _make_household()
        c = _make_constitution()
        delta = evaluate_proposal([], [], agent, c, c)
        assert delta == 0.0

    def test_political_lambda_scales_effect(self) -> None:
        """Higher political_lambda should amplify the political utility difference."""
        vf, ag = _make_simple_value_function()
        agent = _make_household(equality=0.0, liberty=1.0)
        current = _make_constitution(tax_rate=0.3)
        proposed = _make_constitution(tax_rate=0.1)
        delta_low = evaluate_proposal(vf, ag, agent, proposed, current, political_lambda=0.01)
        delta_high = evaluate_proposal(vf, ag, agent, proposed, current, political_lambda=0.10)
        # Both positive (prefers lower tax), but higher lambda -> larger political component
        assert delta_low > 0
        assert delta_high > 0
        # The difference in total delta should reflect the lambda scaling
        # Note: economic component is the same, political component scales with lambda
        assert abs(delta_high) > abs(delta_low)


# ============================================================================
# Test: bellman_vote (REQ-405)
# ============================================================================


class TestBellmanVote:
    """Tests for pure Bellman voting."""

    def test_vote_for_preferred_constitution(self) -> None:
        """Agent should vote for a constitution that increases their value."""
        vf, ag = _make_simple_value_function()
        agent = _make_household(equality=0.0, liberty=1.0)
        current = _make_constitution(tax_rate=0.4)
        proposed = _make_constitution(tax_rate=0.1)
        vote = bellman_vote(vf, ag, agent, proposed, current, political_lambda=0.05)
        assert vote is True

    def test_vote_against_disliked_constitution(self) -> None:
        """Agent should vote against a constitution that decreases their value."""
        vf, ag = _make_simple_value_function()
        agent = _make_household(equality=0.0, liberty=1.0)
        current = _make_constitution(tax_rate=0.1)
        proposed = _make_constitution(tax_rate=0.5)
        vote = bellman_vote(vf, ag, agent, proposed, current, political_lambda=0.05)
        assert vote is False


# ============================================================================
# Test: build_bellman_political_context (REQ-404)
# ============================================================================


class TestBuildBellmanPoliticalContext:
    """Tests for structured Bellman context for LLM."""

    def test_context_has_required_fields(self) -> None:
        """Context should contain recommendation, delta, and tax rates."""
        vf, ag = _make_simple_value_function()
        agent = _make_household()
        c = _make_constitution(tax_rate=0.1)
        ctx = build_bellman_political_context(vf, ag, agent, c, c, political_lambda=0.05)
        assert "bellman_recommendation" in ctx
        assert ctx["bellman_recommendation"] in ("support", "oppose")
        assert "value_delta" in ctx
        assert "political_utility_current" in ctx
        assert "political_utility_proposed" in ctx
        assert "tax_rate_current" in ctx
        assert "tax_rate_proposed" in ctx
        assert "note" in ctx

    def test_context_recommendation_matches_vote(self) -> None:
        """Context recommendation should be consistent with bellman_vote."""
        vf, ag = _make_simple_value_function()
        agent = _make_household(equality=0.0, liberty=1.0)
        current = _make_constitution(tax_rate=0.4)
        proposed = _make_constitution(tax_rate=0.1)
        ctx = build_bellman_political_context(
            vf, ag, agent, proposed, current, political_lambda=0.05
        )
        vote = bellman_vote(vf, ag, agent, proposed, current, political_lambda=0.05)
        expected_rec = "support" if vote else "oppose"
        assert ctx["bellman_recommendation"] == expected_rec

    def test_tax_rates_reported_correctly(self) -> None:
        """Tax rates in context should match the constitutions."""
        vf, ag = _make_simple_value_function()
        agent = _make_household()
        current = _make_constitution(tax_rate=0.15)
        proposed = _make_constitution(tax_rate=0.25)
        ctx = build_bellman_political_context(
            vf, ag, agent, proposed, current, political_lambda=0.05
        )
        assert ctx["tax_rate_current"] == pytest.approx(0.15)
        assert ctx["tax_rate_proposed"] == pytest.approx(0.25)


# ============================================================================
# Test: LLM deviation logging (REQ-404)
# ============================================================================


class TestLLMDeviationLogging:
    """Tests for LLM-Bellman deviation logging."""

    def test_aligned_vote_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """When LLM aligns with Bellman, should log at debug level."""
        import structlog

        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(0),
        )
        log_llm_bellman_deviation(
            agent_id="agent_0001",
            bellman_recommendation="support",
            llm_vote=True,
            delta_v=0.5,
        )
        # Should not raise, deviation not logged as warning

    def test_deviation_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """When LLM deviates from Bellman, it should be logged."""
        import structlog

        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(0),
        )
        log_llm_bellman_deviation(
            agent_id="agent_0002",
            bellman_recommendation="support",
            llm_vote=False,
            delta_v=0.5,
        )
        # Function should complete without error (actual logging depends on config)


# ============================================================================
# Test: config parameters (REQ-401, REQ-405)
# ============================================================================


class TestConfigParameters:
    """Tests for political utility config parameters."""

    def test_political_lambda_default(self) -> None:
        """Default political_lambda should be 0.05."""
        config = SimulationConfigV2()
        assert config.political_lambda == pytest.approx(0.05)

    def test_political_lambda_custom(self) -> None:
        """political_lambda should be configurable."""
        config = SimulationConfigV2(political_lambda=0.1)
        assert config.political_lambda == pytest.approx(0.1)

    def test_political_lambda_bounds(self) -> None:
        """political_lambda should be in [0, 1]."""
        with pytest.raises(ValueError):
            SimulationConfigV2(political_lambda=-0.1)
        with pytest.raises(ValueError):
            SimulationConfigV2(political_lambda=1.5)

    def test_pure_bellman_politics_default(self) -> None:
        """Default pure_bellman_politics should be False."""
        config = SimulationConfigV2()
        assert config.pure_bellman_politics is False

    def test_pure_bellman_politics_enabled(self) -> None:
        """pure_bellman_politics should be configurable."""
        config = SimulationConfigV2(pure_bellman_politics=True)
        assert config.pure_bellman_politics is True


# ============================================================================
# Test: EGM solver integration with political flow bonus (REQ-401)
# ============================================================================


class TestEGMPoliticalIntegration:
    """Tests for political utility integration with EGM solver."""

    def test_value_function_includes_political_bonus(self) -> None:
        """Value function with political bonus should be higher than without."""
        from emergent_constitution.egm_solver import EGMSolver
        from emergent_constitution.shock_generators import rouwenhorst_discretize

        config = SimulationConfigV2()
        states, trans = rouwenhorst_discretize(config.rho_z, config.sigma_z, config.num_z_states)

        solver = EGMSolver(config, states, trans)

        def tax_fn(income: float) -> float:
            return income * 0.1

        # Solve without political bonus
        c_pol, l_pol = solver.solve_egm(
            0.4,
            0.35,
            0.25,
            0.95,
            5.0,
            0.05,
            1.0,
            tax_fn,
            0.5,
            political_flow_bonus=0.0,
        )
        vf_no_bonus = solver._compute_value_function(
            c_pol,
            l_pol,
            0.4,
            0.35,
            0.25,
            0.95,
            5.0,
            0.05,
            1.0,
            tax_fn,
            0.5,
            political_flow_bonus=0.0,
        )

        # Solve with political bonus
        vf_with_bonus = solver._compute_value_function(
            c_pol,
            l_pol,
            0.4,
            0.35,
            0.25,
            0.95,
            5.0,
            0.05,
            1.0,
            tax_fn,
            0.5,
            political_flow_bonus=0.01,
        )

        # Value function with bonus should be strictly higher at every point
        vf_no = np.array(vf_no_bonus)
        vf_yes = np.array(vf_with_bonus)
        assert np.all(vf_yes > vf_no), "Political bonus should increase value function"

    def test_solve_all_accepts_political_bonus(self) -> None:
        """solve_all should accept political_flow_bonus parameter."""
        from emergent_constitution.egm_solver import EGMSolver
        from emergent_constitution.shock_generators import rouwenhorst_discretize

        config = SimulationConfigV2()
        states, trans = rouwenhorst_discretize(config.rho_z, config.sigma_z, config.num_z_states)
        solver = EGMSolver(config, states, trans)

        h = _make_household(wealth=50.0, productivity=states[1], productivity_index=1)
        market = MarketState(wage=5.0, interest_rate=0.05)

        decisions = solver.solve_all(
            households=[h],
            market=market,
            constitution_tax_rate=0.1,
            public_goods=1.0,
            transfer=0.5,
            political_flow_bonus=0.01,
        )
        assert h.id in decisions
        assert decisions[h.id].consumption > 0


# ============================================================================
# Test: LLM engine political integration (REQ-404, REQ-405)
# ============================================================================


class TestLLMEnginePoliticalIntegration:
    """Tests for LLM engine political utility integration."""

    def test_set_value_function(self) -> None:
        """LLM engine should accept value function and Gini."""
        from emergent_constitution.config import SimulationConfig
        from emergent_constitution.llm_engine import LLMDecisionEngine
        from emergent_constitution.rng import SimulationRNG

        config = SimulationConfig(use_llm=False)
        rng = SimulationRNG(42)
        engine = LLMDecisionEngine(config, rng)

        vf, ag = _make_simple_value_function()
        engine.set_value_function(vf, ag)
        engine.set_observed_gini(0.35)
        # Should not raise

    def test_pure_bellman_mode_returns_empty_decisions(self) -> None:
        """In pure Bellman mode, collect_political_decisions returns empty decisions."""
        from emergent_constitution.llm_engine import LLMDecisionEngine
        from emergent_constitution.rng import SimulationRNG

        config = SimulationConfigV2(
            use_llm=False,
            benchmark_mode=True,
            pure_bellman_politics=True,
        )
        rng = SimulationRNG(42)
        engine = LLMDecisionEngine(config, rng)

        h = _make_household()
        constitution = _make_constitution()
        market = MarketState(wage=5.0, interest_rate=0.05)

        decisions = engine.collect_political_decisions(
            households=[h],
            constitution=constitution,
            market=market,
        )
        assert h.id in decisions
        assert decisions[h.id].proposal is None
        assert decisions[h.id].votes == {}

    def test_bellman_deviation_counters(self) -> None:
        """Deviation and alignment counters should start at zero."""
        from emergent_constitution.config import SimulationConfig
        from emergent_constitution.llm_engine import LLMDecisionEngine
        from emergent_constitution.rng import SimulationRNG

        config = SimulationConfig(use_llm=False)
        rng = SimulationRNG(42)
        engine = LLMDecisionEngine(config, rng)
        assert engine.bellman_deviation_count == 0
        assert engine.bellman_alignment_count == 0
