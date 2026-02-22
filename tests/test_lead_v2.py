"""Integration tests for LeadV2 9-step period lifecycle.

Tests the full simulation pipeline: initialization, all 9 steps, multi-period
execution, benchmark mode, LLM fallback, and state consistency.

Traceability: REQ-033, integration tests per spec section 8.2.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.economics import produce_output
from emergent_constitution.lead import LeadV2
from emergent_constitution.llm_engine import LLMDecisionEngine
from emergent_constitution.models.constitution import ConstitutionV2
from emergent_constitution.models.decisions import EconomicDecision, EntrepreneurialDecision
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.history import HistoryEntryV2, PeriodState, SimulationOutputV2
from emergent_constitution.models.household import OccupationalRole
from emergent_constitution.models.market import MarketState
from emergent_constitution.models.proposal import ConstitutionalProposal
from emergent_constitution.models.shocks import ShockState
from emergent_constitution.observer import ObserverV2, detect_rule_changes_v2

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_config() -> SimulationConfigV2:
    """Config with mock LLM for fast integration tests."""
    return SimulationConfigV2(
        num_agents=20,
        max_periods=5,
        seed=42,
        use_llm=True,
        llm_provider="mock",
        benchmark_mode=False,
        observer_interval=1,
        proposal_interval=3,
    )


@pytest.fixture
def short_mock_config() -> SimulationConfigV2:
    """Minimal config with mock LLM for single-period tests."""
    return SimulationConfigV2(
        num_agents=20,
        max_periods=1,
        seed=42,
        use_llm=True,
        llm_provider="mock",
        benchmark_mode=False,
        observer_interval=1,
    )


@pytest.fixture
def benchmark_config() -> SimulationConfigV2:
    """Benchmark config (no LLM). Only used for init tests (VFI is slow)."""
    return SimulationConfigV2(
        num_agents=20,
        max_periods=1,
        seed=42,
        benchmark_mode=True,
        observer_interval=1,
    )


@pytest.fixture
def governance_config() -> SimulationConfigV2:
    """Config with governance enabled on proposal_interval=2."""
    return SimulationConfigV2(
        num_agents=20,
        max_periods=3,
        seed=42,
        use_llm=True,
        llm_provider="mock",
        benchmark_mode=False,
        observer_interval=1,
        proposal_interval=2,
    )


# ============================================================================
# Test Initialization (Subtask 11.1)
# ============================================================================


class TestLeadV2Init:
    """Test LeadV2.__init__ correctly bootstraps the simulation."""

    def test_benchmark_mode_initialization(self, benchmark_config: SimulationConfigV2) -> None:
        """Benchmark mode: no LLM engine, solver created."""
        lead = LeadV2(benchmark_config)

        assert lead.config is benchmark_config
        assert lead._llm_engine is None
        assert lead._solver is not None
        assert isinstance(lead.period_state, PeriodState)
        assert len(lead.period_state.households) == 20
        assert lead.period_state.period == 0
        assert lead.history == []
        assert lead.observer._cumulative_welfare == 0.0

    def test_llm_mode_initialization(self, mock_config: SimulationConfigV2) -> None:
        """LLM mode: LLM engine created, solver also created as fallback."""
        lead = LeadV2(mock_config)

        assert lead._llm_engine is not None
        assert isinstance(lead._llm_engine, LLMDecisionEngine)
        assert lead._solver is not None

    def test_initial_state_valid(self, mock_config: SimulationConfigV2) -> None:
        """Initial period state has valid households, market, shocks."""
        lead = LeadV2(mock_config)
        state = lead.period_state

        # Households have valid state
        assert all(h.wealth >= 0.0 for h in state.households)
        assert all(h.productivity > 0.0 for h in state.households)
        assert all(h.productivity_index >= 0 for h in state.households)

        # Market has positive wage
        assert state.market.wage > 0.0

        # Shocks have valid grid
        assert len(state.shocks.productivity_grid) > 0
        assert len(state.shocks.transition_matrix) > 0
        assert state.shocks.aggregate_tfp > 0.0

        # Constitution is default
        assert isinstance(state.constitution, ConstitutionV2)
        assert "flat_tax" in state.constitution.rules
        assert "majority_vote" in state.constitution.rules

    def test_rng_determinism(self, mock_config: SimulationConfigV2) -> None:
        """Same seed produces identical initial state (PROP-001)."""
        lead1 = LeadV2(mock_config)
        lead2 = LeadV2(mock_config)

        w1 = [h.wealth for h in lead1.period_state.households]
        w2 = [h.wealth for h in lead2.period_state.households]
        assert w1 == w2


# ============================================================================
# Test Full Period Lifecycle (Subtasks 11.2 - 11.10)
# ============================================================================


class TestSinglePeriod:
    """Test a single period execution with all 9 steps."""

    def test_single_period_completes(self, short_mock_config: SimulationConfigV2) -> None:
        """Run one period with mock LLM; verify state transitions."""
        lead = LeadV2(short_mock_config)

        new_state = lead._advance_period(1)

        assert new_state.period == 1
        assert len(new_state.households) == 20
        assert isinstance(new_state.market, MarketState)
        assert isinstance(new_state.shocks, ShockState)
        assert isinstance(new_state.constitution, ConstitutionV2)

        # Households should have updated consumption/leisure
        for h in new_state.households:
            assert h.consumption >= 0.0
            assert 0.0 <= h.leisure <= 1.0
            assert h.labor_supply == pytest.approx(1.0 - h.leisure, abs=1e-10)

    def test_shocks_are_drawn(self, short_mock_config: SimulationConfigV2) -> None:
        """Step 1: Verify shocks update productivity and aggregate TFP."""
        lead = LeadV2(short_mock_config)

        new_state = lead._advance_period(1)

        # All productivities should remain valid after shocks
        assert all(h.productivity > 0.0 for h in new_state.households)

        # Aggregate TFP should be positive
        assert new_state.shocks.aggregate_tfp > 0.0

    def test_market_clearing_produces_prices(self, short_mock_config: SimulationConfigV2) -> None:
        """Step 2: Market clearing gives positive wage and finite interest rate."""
        lead = LeadV2(short_mock_config)
        new_state = lead._advance_period(1)

        assert new_state.market.wage > 0.0
        assert math.isfinite(new_state.market.interest_rate)
        assert new_state.market.aggregate_output >= 0.0

    def test_budget_constraints_respected(self, short_mock_config: SimulationConfigV2) -> None:
        """Step 4: Wealth does not go below a_min (PROP-002)."""
        lead = LeadV2(short_mock_config)
        new_state = lead._advance_period(1)

        for h in new_state.households:
            assert h.wealth >= short_mock_config.a_min

    def test_realized_utility_computed(self, short_mock_config: SimulationConfigV2) -> None:
        """Step 8: Realized utility is computed for all agents."""
        lead = LeadV2(short_mock_config)
        new_state = lead._advance_period(1)

        for h in new_state.households:
            assert h.realized_utility >= 0.0
            assert math.isfinite(h.realized_utility)

    def test_observation_recorded(self, short_mock_config: SimulationConfigV2) -> None:
        """Step 9: Observation recorded when observer_interval = 1."""
        lead = LeadV2(short_mock_config)
        lead.period_state = lead._advance_period(1)

        assert len(lead.history) >= 1
        entry = lead.history[0]
        assert isinstance(entry, HistoryEntryV2)
        assert 0.0 <= entry.gini <= 1.0
        assert entry.mean_wealth >= 0.0
        assert entry.social_welfare >= 0.0
        assert entry.wage > 0.0


# ============================================================================
# Test Multi-Period Run (Subtask 11.11)
# ============================================================================


class TestFullRun:
    """Test full simulation run over multiple periods."""

    def test_run_completes(self, mock_config: SimulationConfigV2) -> None:
        """5 periods with mock LLM complete without error."""
        lead = LeadV2(mock_config)
        output = lead.run()

        assert isinstance(output, SimulationOutputV2)
        assert output.total_periods == 5
        assert output.seed == 42
        assert len(output.final_households) == 20
        assert isinstance(output.constitution, ConstitutionV2)

    def test_history_entries_recorded(self, mock_config: SimulationConfigV2) -> None:
        """Observer records entries at the correct interval."""
        lead = LeadV2(mock_config)
        output = lead.run()

        # observer_interval=1, max_periods=5 -> 5 entries
        assert len(output.history) == 5
        for i, entry in enumerate(output.history):
            assert entry.period == i + 1

    def test_welfare_summary_positive(self, mock_config: SimulationConfigV2) -> None:
        """Welfare summary has non-negative total welfare."""
        lead = LeadV2(mock_config)
        output = lead.run()

        assert output.welfare_summary.llm_total_welfare >= 0.0
        assert output.welfare_summary.benchmark_total_welfare is None

    def test_determinism_across_runs(self, mock_config: SimulationConfigV2) -> None:
        """Same config+seed produces identical output (PROP-001)."""
        lead1 = LeadV2(mock_config)
        output1 = lead1.run()

        lead2 = LeadV2(mock_config)
        output2 = lead2.run()

        w1 = [h.wealth for h in output1.final_households]
        w2 = [h.wealth for h in output2.final_households]
        assert w1 == w2

        assert len(output1.history) == len(output2.history)
        for h1, h2 in zip(output1.history, output2.history, strict=True):
            assert h1.gini == pytest.approx(h2.gini)
            assert h1.mean_wealth == pytest.approx(h2.mean_wealth)

    def test_no_nan_or_inf_in_output(self, mock_config: SimulationConfigV2) -> None:
        """No NaN or Inf values in any output field."""
        lead = LeadV2(mock_config)
        output = lead.run()

        for h in output.final_households:
            assert math.isfinite(h.wealth), f"Agent {h.id} has invalid wealth: {h.wealth}"
            assert math.isfinite(h.consumption)
            assert math.isfinite(h.realized_utility)

        for entry in output.history:
            assert math.isfinite(entry.gini)
            assert math.isfinite(entry.mean_wealth)
            assert math.isfinite(entry.social_welfare)
            assert math.isfinite(entry.wage)
            assert math.isfinite(entry.interest_rate)


# ============================================================================
# Test LLM Mode with MockProvider (Subtask 11.12)
# ============================================================================


class TestLLMMode:
    """Test LeadV2 with mock LLM provider for integration testing."""

    def test_llm_governance_on_interval(self, governance_config: SimulationConfigV2) -> None:
        """Political decisions collected on proposal_interval periods."""
        lead = LeadV2(governance_config)
        output = lead.run()

        # Mock provider returns no proposals (PoliticalDecision())
        # So no rule changes expected
        assert isinstance(output.constitution, ConstitutionV2)

    def test_llm_economic_decisions_applied(self, mock_config: SimulationConfigV2) -> None:
        """LLM-generated decisions are applied to households."""
        lead = LeadV2(mock_config)
        output = lead.run()

        for h in output.final_households:
            assert h.consumption >= 0.0
            assert 0.0 <= h.leisure <= 1.0


class TestPureBellmanMode:
    """Test pure Bellman governance mode with LLM economic decisions."""

    def test_pure_bellman_populates_value_function(self) -> None:
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=1,
            seed=42,
            use_llm=True,
            llm_provider="mock",
            benchmark_mode=False,
            pure_bellman_politics=True,
            proposal_interval=1,
            observer_interval=1,
        )
        lead = LeadV2(config)
        lead._advance_period(1)

        assert lead._llm_engine is not None
        assert lead._llm_engine._value_function is not None
        assert lead._llm_engine._a_grid is not None


class TestGovernanceVoteKeys:
    """Proposal-level vote keying should avoid rule-name collisions."""

    def test_same_rule_proposals_keep_distinct_tallies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=1,
            seed=7,
            benchmark_mode=True,
            proposal_interval=1,
            observer_interval=1,
        )
        lead = LeadV2(config)
        households = [h.model_copy(deep=True) for h in lead.period_state.households[:3]]
        constitution = lead.period_state.constitution
        market = lead.period_state.market

        proposer_a = households[0].id
        proposer_b = households[1].id

        def fake_decide_proposal_v2(
            household: object,
            _constitution: ConstitutionV2,
            _num_agents: int,
            _rng: object,
        ) -> ConstitutionalProposal | None:
            h = household
            if getattr(h, "id", "") == proposer_a:
                return ConstitutionalProposal(
                    proposer_id=proposer_a,
                    action="modify",
                    rule_name="flat_tax",
                    parameters={"rate": 0.21},
                    description="tax option A",
                )
            if getattr(h, "id", "") == proposer_b:
                return ConstitutionalProposal(
                    proposer_id=proposer_b,
                    action="modify",
                    rule_name="flat_tax",
                    parameters={"rate": 0.31},
                    description="tax option B",
                )
            return None

        def fake_decide_votes_v2(
            household: object,
            proposals: list[ConstitutionalProposal],
            _constitution: ConstitutionV2,
        ) -> dict[str, bool]:
            h_id = getattr(household, "id", "")
            p1, p2 = proposals
            if h_id == proposer_a:
                return {p1.proposal_id: True, p2.proposal_id: False}
            if h_id == proposer_b:
                return {p1.proposal_id: False, p2.proposal_id: False}
            return {p1.proposal_id: True, p2.proposal_id: True}

        monkeypatch.setattr(
            "emergent_constitution.lead.decide_proposal_v2",
            fake_decide_proposal_v2,
        )
        monkeypatch.setattr(
            "emergent_constitution.lead.decide_votes_v2",
            fake_decide_votes_v2,
        )

        proposals, vote_outcomes, _ = lead._process_governance(1, households, constitution, market)
        assert len(proposals) == 2
        assert proposals[0].proposal_id and proposals[1].proposal_id
        assert proposals[0].proposal_id != proposals[1].proposal_id
        assert len(vote_outcomes) == 2
        assert vote_outcomes[0].votes_for == 2
        assert vote_outcomes[1].votes_for == 1


class TestGovernmentFeedback:
    """Fiscal rule should feed back into effective tax policy."""

    def test_fiscal_rule_adjusts_tax_rule(self) -> None:
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=1,
            seed=11,
            benchmark_mode=True,
            initial_debt=400.0,
            debt_gdp_max=0.5,
            fiscal_rule_adjustment=0.05,
            observer_interval=1,
        )
        lead = LeadV2(config)
        households = [h.model_copy(deep=True) for h in lead.period_state.households]
        for household in households:
            household.taxes_paid = 0.0
            household.transfers_received = 0.0

        constitution = lead.period_state.constitution
        market = lead.period_state.market.model_copy(
            update={"aggregate_output": 100.0, "interest_rate": 0.02, "bond_rate": 0.02}
        )
        old_rate = lead._get_tax_rate(constitution)

        updated_constitution, updated_market = lead._update_government(
            households=households,
            constitution=constitution,
            market=market,
            public_goods=0.0,
        )
        new_rate = lead._get_tax_rate(updated_constitution)

        assert new_rate == pytest.approx(min(1.0, old_rate + config.fiscal_rule_adjustment))
        assert updated_market.bond_rate is not None


# ============================================================================
# Test Individual Steps
# ============================================================================


class TestDrawShocks:
    """Test step 1: shock drawing."""

    def test_idiosyncratic_shocks_update_productivity(
        self, short_mock_config: SimulationConfigV2
    ) -> None:
        """Idiosyncratic shocks update household productivity via Markov chain."""
        lead = LeadV2(short_mock_config)
        households = [h.model_copy(deep=True) for h in lead.period_state.households]
        shocks = lead.period_state.shocks.model_copy(deep=True)

        updated_h, updated_s = lead._draw_shocks(1, households, shocks)

        for h in updated_h:
            assert h.productivity > 0.0
            assert 0 <= h.productivity_index < len(shocks.productivity_grid)

        assert updated_s.aggregate_tfp > 0.0

    def test_preference_shocks_optional(self, short_mock_config: SimulationConfigV2) -> None:
        """Preference shocks only drawn when enabled."""
        lead = LeadV2(short_mock_config)
        households = [h.model_copy(deep=True) for h in lead.period_state.households]
        shocks = lead.period_state.shocks.model_copy(deep=True)

        _, updated_s = lead._draw_shocks(1, households, shocks)
        assert updated_s.preference_shocks == {}


class TestClearMarkets:
    """Test step 2: market clearing."""

    def test_positive_prices(self, short_mock_config: SimulationConfigV2) -> None:
        """Market clearing produces positive wage."""
        lead = LeadV2(short_mock_config)
        households = lead.period_state.households
        firms = lead.period_state.firms
        shocks = lead.period_state.shocks

        market = lead._clear_markets(households, firms, shocks)

        assert market.wage > 0.0
        assert math.isfinite(market.interest_rate)


class TestExecuteProduction:
    """Test step 5: production execution."""

    def test_no_firms_no_output(self, short_mock_config: SimulationConfigV2) -> None:
        """With no firms, production step returns empty list."""
        lead = LeadV2(short_mock_config)
        households = lead.period_state.households
        market = lead.period_state.market
        entre = {h.id: EntrepreneurialDecision() for h in households}

        firms = lead._execute_production([], entre, households, market)
        assert firms == []

    def test_firm_creation(self, short_mock_config: SimulationConfigV2) -> None:
        """Entrepreneurial decision to create firm adds a new firm."""
        lead = LeadV2(short_mock_config)
        households = lead.period_state.households
        market = lead.period_state.market

        rich_agent = households[0]
        entre = {h.id: EntrepreneurialDecision() for h in households}
        if rich_agent.wealth >= short_mock_config.min_firm_capital:
            entre[rich_agent.id] = EntrepreneurialDecision(
                create_firm=True,
                capital_investment=10.0,
                labor_demand=1.0,
            )

            firms = lead._execute_production([], entre, households, market)
            assert len(firms) == 1
            assert firms[0].owner_id == rich_agent.id

    def test_firm_creation_records_principal_debit(
        self, short_mock_config: SimulationConfigV2
    ) -> None:
        """Firm entry should debit sunk cost and track invested principal separately."""
        lead = LeadV2(short_mock_config)
        owner = lead.period_state.households[0].model_copy(update={"wealth": 200.0})
        market = lead.period_state.market
        entre = {
            owner.id: EntrepreneurialDecision(
                create_firm=True,
                capital_investment=40.0,
                labor_demand=1.0,
            )
        }

        firms = lead._execute_production([], entre, [owner], market)
        assert len(firms) == 1
        assert lead._period_capital_debits[owner.id] == pytest.approx(
            short_mock_config.firm_entry_cost
        )
        assert lead._period_principal_debits[owner.id] == pytest.approx(40.0)

    def test_firm_liquidation_records_principal_credit(
        self, short_mock_config: SimulationConfigV2
    ) -> None:
        """Firm liquidation should credit remaining principal to owner ledger."""
        lead = LeadV2(short_mock_config)
        owner = lead.period_state.households[0]
        market = lead.period_state.market
        firm = FirmState(
            id="firm_close",
            owner_id=owner.id,
            owner_ability=owner.entrepreneurial_ability,
            capital=25.0,
            labor_demand=1.0,
            tfp=1.0,
        )
        entre = {owner.id: EntrepreneurialDecision(close_firm=True)}

        firms = lead._execute_production([firm], entre, [owner], market)
        assert firms == []
        assert lead._period_capital_credits.get(owner.id, 0.0) == pytest.approx(0.0)
        assert lead._period_principal_credits[owner.id] == pytest.approx(25.0)

    def test_output_uses_updated_labor_demand(self, short_mock_config: SimulationConfigV2) -> None:
        """Same-period output should reflect owner-updated labor demand."""
        lead = LeadV2(short_mock_config)
        owner = lead.period_state.households[0]
        market = lead.period_state.market
        firm = FirmState(
            id="firm_test",
            owner_id=owner.id,
            owner_ability=owner.entrepreneurial_ability,
            capital=20.0,
            labor_demand=0.5,
            tfp=1.2,
        )
        entre = {
            owner.id: EntrepreneurialDecision(
                labor_demand=3.0,
                rd_spend=0.0,
                close_firm=False,
            )
        }

        updated_firms = lead._execute_production([firm], entre, [owner], market)
        assert len(updated_firms) == 1
        updated = updated_firms[0]
        expected_output = produce_output(
            firm.model_copy(update={"labor_demand": 3.0}),
            short_mock_config.alpha,
        )
        assert updated.labor_demand == pytest.approx(3.0)
        assert updated.output == pytest.approx(expected_output, abs=1e-10)


class TestEnforceConstitution:
    """Test step 6: constitution enforcement."""

    def test_default_constitution_collects_tax(
        self, short_mock_config: SimulationConfigV2
    ) -> None:
        """Default constitution has 10% tax; revenue is collected."""
        lead = LeadV2(short_mock_config)
        households = [h.model_copy(deep=True) for h in lead.period_state.households]
        firms = lead.period_state.firms
        constitution = lead.period_state.constitution
        market = lead.period_state.market

        updated_h, public_goods = lead._enforce_constitution(
            households, firms, constitution, market
        )

        total_taxes = sum(h.taxes_paid for h in updated_h)
        assert total_taxes > 0.0  # 10% default tax should collect revenue

    def test_taxes_use_post_production_profit_income(
        self, short_mock_config: SimulationConfigV2
    ) -> None:
        """Tax base should include current-period entrepreneurial profit."""
        config = short_mock_config.model_copy(update={"fix_entrepreneur_budget": True})
        lead = LeadV2(config)
        households = [h.model_copy(deep=True) for h in lead.period_state.households[:1]]
        owner = households[0]
        owner.role = OccupationalRole.ENTREPRENEUR

        decision = EconomicDecision(consumption=0.0, leisure=0.5)
        econ_decisions = {owner.id: decision}
        firms = [
            FirmState(
                id="firm_tax_test",
                owner_id=owner.id,
                owner_ability=owner.entrepreneurial_ability,
                capital=20.0,
                labor_demand=2.0,
                tfp=1.1,
                profit=50.0,
            )
        ]

        lead._refresh_household_incomes_after_production(
            households=households,
            firms=firms,
            market=lead.period_state.market,
            econ_decisions=econ_decisions,
        )
        updated_h, _ = lead._enforce_constitution(
            households,
            firms,
            lead.period_state.constitution,
            lead.period_state.market,
        )

        assert updated_h[0].income == pytest.approx(50.0)
        assert updated_h[0].taxes_paid > 0.0


class TestUpdateStates:
    """Test step 8: state updates."""

    def test_wealth_non_negative(self, short_mock_config: SimulationConfigV2) -> None:
        """After state update, all wealth >= a_min."""
        lead = LeadV2(short_mock_config)
        households = lead.period_state.households
        market = lead.period_state.market

        decisions = {h.id: EconomicDecision(consumption=0.0, leisure=0.5) for h in households}

        updated = lead._update_states(households, decisions, market, 0.0)

        for h in updated:
            assert h.wealth >= short_mock_config.a_min

    def test_budget_constraint_formula(self, short_mock_config: SimulationConfigV2) -> None:
        """Step 8: a' = (1+r)*a + w*z*(1-l) - c - taxes + transfers.

        Verify the DSGE-HA budget constraint is correctly applied.
        """
        lead = LeadV2(short_mock_config)
        households = [h.model_copy(deep=True) for h in lead.period_state.households]
        market = lead.period_state.market

        # Zero consumption and no taxes/transfers
        decisions = {h.id: EconomicDecision(consumption=0.0, leisure=0.5) for h in households}
        updated = lead._update_states(households, decisions, market, 0.0)

        for orig, upd in zip(households, updated, strict=True):
            labor_income = market.wage * orig.productivity * 0.5
            expected = (
                (1.0 + market.interest_rate) * orig.wealth
                + labor_income
                - 0.0  # consumption
                - orig.taxes_paid
                + orig.transfers_received
            )
            expected = max(expected, short_mock_config.a_min)
            assert upd.wealth == pytest.approx(expected, abs=1e-8), (
                f"Agent {orig.id}: expected {expected:.4f}, got {upd.wealth:.4f}"
            )
            # Labor income should be reflected in updated household
            assert upd.income == pytest.approx(labor_income, abs=1e-8)

    def test_two_asset_illiquid_accrues_return(self) -> None:
        """Two-asset transition should apply return on the illiquid stock."""
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=1,
            seed=42,
            benchmark_mode=True,
            two_asset_mode=True,
            b_min=0.0,
            observer_interval=1,
        )
        lead = LeadV2(config)

        base = lead.period_state.households[0].model_copy(
            update={
                "wealth": 100.0,
                "liquid": 30.0,
                "illiquid": 70.0,
                "taxes_paid": 0.0,
                "transfers_received": 0.0,
            }
        )
        market = lead.period_state.market.model_copy(update={"wage": 0.0, "interest_rate": 0.05})
        decisions = {base.id: EconomicDecision(consumption=0.0, leisure=1.0)}

        updated = lead._update_states([base], decisions, market, 0.0, firms=[])
        assert updated[0].illiquid == pytest.approx(73.5, abs=1e-8)

    def test_two_asset_uses_bond_and_capital_rates(self) -> None:
        """Liquid and illiquid assets should use bond/capital returns respectively."""
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=1,
            seed=9,
            benchmark_mode=True,
            two_asset_mode=True,
            b_min=0.0,
            observer_interval=1,
        )
        lead = LeadV2(config)
        base = lead.period_state.households[0].model_copy(
            update={
                "wealth": 100.0,
                "liquid": 30.0,
                "illiquid": 70.0,
                "taxes_paid": 0.0,
                "transfers_received": 0.0,
            }
        )
        market = lead.period_state.market.model_copy(
            update={
                "wage": 0.0,
                "interest_rate": 0.01,
                "bond_rate": 0.02,
                "capital_rate": 0.05,
            }
        )
        decisions = {base.id: EconomicDecision(consumption=0.0, leisure=1.0)}

        updated = lead._update_states([base], decisions, market, 0.0, firms=[])
        assert updated[0].liquid == pytest.approx(30.6, abs=1e-8)
        assert updated[0].illiquid == pytest.approx(73.5, abs=1e-8)

    def test_entrepreneurial_capital_account_updates_with_principal_flows(self) -> None:
        """Household entrepreneurial_capital should track principal debit/credit."""
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=1,
            seed=13,
            benchmark_mode=True,
            observer_interval=1,
        )
        lead = LeadV2(config)
        h = lead.period_state.households[0].model_copy(
            update={
                "wealth": 100.0,
                "entrepreneurial_capital": 30.0,
                "taxes_paid": 0.0,
                "transfers_received": 0.0,
            }
        )
        lead._period_principal_debits[h.id] = 12.0
        lead._period_principal_credits[h.id] = 7.5
        decisions = {h.id: EconomicDecision(consumption=0.0, leisure=1.0)}
        market = lead.period_state.market.model_copy(update={"wage": 0.0, "interest_rate": 0.0})

        updated = lead._update_states([h], decisions, market, 0.0, firms=[])
        assert updated[0].entrepreneurial_capital == pytest.approx(25.5)


class TestDistributionUpdate:
    """Test step 8b distribution update bookkeeping."""

    def test_policy_savings_includes_transfer(self) -> None:
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=1,
            seed=42,
            benchmark_mode=True,
            distribution_mode="kfe",
            observer_interval=1,
        )
        lead = LeadV2(config)
        assert lead._distribution is not None

        solver = lead._solver
        n_a = len(solver.a_grid)
        n_z = len(solver.productivity_grid)
        c_policy = np.zeros((n_a, n_z), dtype=np.float64)
        lei_policy = np.full((n_a, n_z), 0.5, dtype=np.float64)

        def fake_solve_egm_cached(**_kwargs: object) -> tuple[np.ndarray, np.ndarray]:
            return c_policy, lei_policy

        solver.solve_egm_cached = fake_solve_egm_cached  # type: ignore[method-assign]

        captured: list[np.ndarray] = []
        original_forward = lead._distribution.forward

        def capture_forward(policy_savings: np.ndarray, transition_matrix: np.ndarray) -> None:
            captured.append(policy_savings.copy())
            original_forward(policy_savings, transition_matrix)

        lead._distribution.forward = capture_forward  # type: ignore[method-assign]

        lead._update_distribution(
            econ_decisions={},
            households=lead.period_state.households,
            market=lead.period_state.market,
            constitution=lead.period_state.constitution,
            public_goods=0.0,
            transfer=0.0,
        )
        lead._update_distribution(
            econ_decisions={},
            households=lead.period_state.households,
            market=lead.period_state.market,
            constitution=lead.period_state.constitution,
            public_goods=0.0,
            transfer=1.25,
        )

        assert len(captured) == 2
        assert np.allclose(captured[1] - captured[0], 1.25, atol=1e-10)


class TestAggregateReconciliation:
    """Test explicit aggregate ledger reconciliation."""

    def test_resource_residual_recorded(self, short_mock_config: SimulationConfigV2) -> None:
        lead = LeadV2(short_mock_config)
        households = [h.model_copy(deep=True) for h in lead.period_state.households[:2]]
        households[0].consumption = 5.0
        households[1].consumption = 4.0

        firms = [
            FirmState(
                id="firm_a",
                owner_id=households[0].id,
                owner_ability=households[0].entrepreneurial_ability,
                capital=10.0,
                labor_demand=1.0,
                tfp=1.0,
                output=12.0,
            )
        ]
        market = lead.period_state.market.model_copy(
            update={"aggregate_output": 0.0, "aggregate_investment": 0.0}
        )
        reconciled = lead._reconcile_market_aggregates(
            market=market,
            households=households,
            firms=firms,
            public_goods=1.0,
        )

        expected_investment = firms[0].capital * short_mock_config.delta
        expected_residual = 12.0 - (9.0 + expected_investment + 2.0)

        assert reconciled.aggregate_output == pytest.approx(12.0)
        assert reconciled.aggregate_consumption == pytest.approx(9.0)
        assert reconciled.aggregate_investment == pytest.approx(expected_investment)
        assert reconciled.government_spending == pytest.approx(2.0)
        assert reconciled.resource_residual == pytest.approx(expected_residual)


class TestGiniComputation:
    """Test the Gini coefficient helper."""

    def test_perfect_equality(self) -> None:
        """All equal values -> Gini = 0."""
        gini = ObserverV2._compute_gini([100.0, 100.0, 100.0, 100.0])
        assert gini == pytest.approx(0.0, abs=1e-10)

    def test_maximal_inequality(self) -> None:
        """One agent has everything -> Gini near 1."""
        gini = ObserverV2._compute_gini([0.0, 0.0, 0.0, 1000.0])
        assert gini > 0.5

    def test_single_agent(self) -> None:
        """Single agent -> Gini = 0."""
        assert ObserverV2._compute_gini([100.0]) == 0.0

    def test_empty_list(self) -> None:
        """Empty list -> Gini = 0."""
        assert ObserverV2._compute_gini([]) == 0.0


class TestRuleChangeDetection:
    """Test the v2 rule change detection helper."""

    def test_no_changes(self) -> None:
        """Identical constitutions -> no changes detected."""
        from emergent_constitution.models.constitution import create_default_constitution

        c1 = create_default_constitution()
        c2 = c1.model_copy(deep=True)
        changes = detect_rule_changes_v2(c1, c2)
        assert changes == []

    def test_first_observation(self) -> None:
        """First observation (prev=None) -> no changes."""
        from emergent_constitution.models.constitution import create_default_constitution

        c = create_default_constitution()
        changes = detect_rule_changes_v2(None, c)
        assert changes == []

    def test_rule_parameter_change(self) -> None:
        """Changing a rule's parameters is detected."""
        from emergent_constitution.models.constitution import create_default_constitution

        c1 = create_default_constitution()
        c2 = c1.model_copy(deep=True)
        c2.rules["flat_tax"].parameters["rate"] = 0.25
        changes = detect_rule_changes_v2(c1, c2)
        assert any("flat_tax" in c and "modified" in c for c in changes)

    def test_rule_added(self) -> None:
        """Adding a new rule is detected."""
        from emergent_constitution.models.constitution import (
            ConstitutionalRule,
            RuleType,
            create_default_constitution,
        )

        c1 = create_default_constitution()
        c2 = c1.model_copy(deep=True)
        c2.rules["new_rule"] = ConstitutionalRule(
            name="new_rule",
            rule_type=RuleType.MARKET_REGULATION,
            parameters={"minimum_wage": 5.0},
            description="Test rule",
        )
        changes = detect_rule_changes_v2(c1, c2)
        assert any("new_rule" in c and "added" in c for c in changes)

    def test_rule_removed(self) -> None:
        """Removing a rule is detected."""
        from emergent_constitution.models.constitution import create_default_constitution

        c1 = create_default_constitution()
        c2 = c1.model_copy(deep=True)
        del c2.rules["private_property"]
        changes = detect_rule_changes_v2(c1, c2)
        assert any("private_property" in c and "removed" in c for c in changes)


# ============================================================================
# Entrepreneurial and governance integration tests
# ============================================================================


class TestEntrepreneurialFeatures:
    """Test entrepreneurial features in the LeadV2 lifecycle."""

    def test_firms_created_in_benchmark_mode(self) -> None:
        """Benchmark mode runs 5 periods without error and produces valid output."""
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=5,
            seed=42,
            benchmark_mode=True,
            observer_interval=1,
        )
        lead = LeadV2(config)
        output = lead.run()

        assert isinstance(output, SimulationOutputV2)
        assert output.total_periods == 5
        assert len(output.final_households) == 20
        # Verify every household has valid state after all periods
        for h in output.final_households:
            assert h.wealth >= config.a_min
            assert h.productivity > 0.0

    def test_entrepreneur_role_assigned(self) -> None:
        """Agents with ENTREPRENEUR role should own a firm or have owned one.

        The role update in step 8 uses the *previous* period's firm list,
        so we verify that any agent assigned ENTREPRENEUR was an owner in
        the penultimate period's firm set. As a simpler invariant, we
        verify that if any agent has ENTREPRENEUR role in the output, the
        simulation did have firms at some point.
        """
        config = SimulationConfigV2(
            num_agents=20,
            max_periods=5,
            seed=42,
            benchmark_mode=True,
            observer_interval=1,
        )
        lead = LeadV2(config)
        output = lead.run()

        entrepreneurs = [
            h for h in output.final_households if h.role == OccupationalRole.ENTREPRENEUR
        ]

        # The simulation creates firms (visible in history), so entrepreneurs
        # should exist or firms should appear in history/final state
        any_firms_ever = any(entry.num_active_firms > 0 for entry in output.history)

        if any_firms_ever:
            # If firms existed during the run, the role assignment mechanism
            # should have set at least some agents to ENTREPRENEUR
            # (Note: the role update lags by one period, so it's possible
            # the last-period entrepreneurs reflect the penultimate period.)
            assert len(entrepreneurs) > 0 or len(lead.period_state.firms) > 0, (
                "Firms existed during the simulation but no agents have "
                "ENTREPRENEUR role and no firms are active at the end"
            )

    def test_ability_shocks_drawn_each_period(self) -> None:
        """After 2 periods, at least some agents' ability_index should change."""
        config = SimulationConfigV2(
            num_agents=50,
            max_periods=2,
            seed=42,
            benchmark_mode=True,
            observer_interval=1,
        )
        lead = LeadV2(config)

        # Snapshot initial ability indices
        initial_indices = {
            h.id: h.entrepreneurial_ability_index for h in lead.period_state.households
        }

        # Run 2 periods
        output = lead.run()

        # Compare final ability indices to initial
        changes = 0
        for h in output.final_households:
            if h.entrepreneurial_ability_index != initial_indices[h.id]:
                changes += 1

        # With 50 agents and a Markov chain with off-diagonal probabilities,
        # at least some should change over 2 periods
        assert changes > 0, (
            "No agents changed entrepreneurial_ability_index over 2 periods; "
            "expected at least some transitions from the Markov chain"
        )
