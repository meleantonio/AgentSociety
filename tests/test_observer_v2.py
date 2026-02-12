"""Unit tests for ObserverV2 — DSGE-HA statistics and welfare computation.

Tests: Gini, quantiles, unemployment rate, firm stats, social welfare,
cumulative discounted welfare, Pareto efficiency, rule change detection,
finalize output.

Traceability: REQ-030, REQ-031, REQ-032.
"""

from __future__ import annotations

import pytest

from emergent_constitution.config import SimulationConfigV2
from emergent_constitution.models.constitution import (
    ConstitutionalRule,
    ConstitutionV2,
    RuleType,
    create_default_constitution,
)
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.history import PeriodState, SimulationOutputV2
from emergent_constitution.models.household import (
    HouseholdState,
    OccupationalRole,
    UtilityParams,
    ValueVector,
)
from emergent_constitution.models.market import MarketState
from emergent_constitution.models.shocks import ShockState
from emergent_constitution.observer import (
    ObserverV2,
    detect_rule_changes_v2,
)

# ============================================================================
# Helpers
# ============================================================================


def _make_household(
    agent_id: str,
    wealth: float = 100.0,
    consumption: float = 10.0,
    leisure: float = 0.3,
    realized_utility: float = 1.0,
    role: OccupationalRole = OccupationalRole.WORKER,
    productivity: float = 1.0,
    beta_discount: float = 0.95,
) -> HouseholdState:
    return HouseholdState(
        id=agent_id,
        wealth=wealth,
        productivity=productivity,
        productivity_index=0,
        utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2, beta_discount=beta_discount),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
        role=role,
        consumption=consumption,
        leisure=leisure,
        labor_supply=1.0 - leisure,
        realized_utility=realized_utility,
    )


def _make_firm(
    firm_id: str,
    worker_ids: list[str] | None = None,
    rd_spend: float = 0.0,
) -> FirmState:
    return FirmState(
        id=firm_id,
        owner_id="agent_0000",
        capital=100.0,
        labor_demand=1.0,
        tfp=1.0,
        worker_ids=worker_ids or [],
        rd_spend=rd_spend,
    )


def _make_period_state(
    period: int = 1,
    households: list[HouseholdState] | None = None,
    firms: list[FirmState] | None = None,
    constitution: ConstitutionV2 | None = None,
) -> PeriodState:
    if households is None:
        households = [_make_household(f"agent_{i:04d}") for i in range(5)]
    if constitution is None:
        constitution = create_default_constitution()
    return PeriodState(
        period=period,
        households=households,
        firms=firms or [],
        market=MarketState(
            wage=1.0,
            interest_rate=0.04,
            aggregate_output=100.0,
            aggregate_investment=20.0,
        ),
        shocks=ShockState(
            productivity_grid=[0.5, 1.0, 1.5],
            transition_matrix=[[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]],
        ),
        constitution=constitution,
    )


@pytest.fixture
def config() -> SimulationConfigV2:
    return SimulationConfigV2(
        num_agents=20,
        max_periods=3,
        seed=42,
        use_llm=True,
        llm_provider="mock",
        benchmark_mode=False,
        observer_interval=1,
        proposal_interval=5,
    )


# ============================================================================
# Tests: ObserverV2.observe() — Gini
# ============================================================================


class TestObserverV2Gini:
    """Test Gini coefficient computation via observe()."""

    def test_equal_wealth_gini_zero(self, config: SimulationConfigV2) -> None:
        """All equal wealth -> Gini = 0."""
        households = [_make_household(f"a{i}", wealth=100.0) for i in range(5)]
        ps = _make_period_state(households=households)
        observer = ObserverV2(config)
        entry = observer.observe(ps)
        assert entry.gini == pytest.approx(0.0, abs=1e-10)

    def test_unequal_wealth_gini_positive(self, config: SimulationConfigV2) -> None:
        """Unequal wealth -> Gini > 0."""
        households = [_make_household(f"a{i}", wealth=float(i * 100)) for i in range(5)]
        ps = _make_period_state(households=households)
        observer = ObserverV2(config)
        entry = observer.observe(ps)
        assert entry.gini > 0.0

    def test_single_agent_gini_zero(self, config: SimulationConfigV2) -> None:
        """Single agent -> Gini = 0."""
        households = [_make_household("a0")]
        ps = _make_period_state(households=households)
        observer = ObserverV2(config)
        entry = observer.observe(ps)
        assert entry.gini == 0.0


# ============================================================================
# Tests: Wealth quantiles
# ============================================================================


class TestObserverV2Quantiles:
    """Test wealth quantile computation."""

    def test_quantiles_five_agents(self, config: SimulationConfigV2) -> None:
        """Five agents with distinct wealth -> 5 quantile values."""
        households = [_make_household(f"a{i}", wealth=float((i + 1) * 100)) for i in range(5)]
        ps = _make_period_state(households=households)
        observer = ObserverV2(config)
        entry = observer.observe(ps)
        assert len(entry.wealth_quantiles) == 5
        # Quantiles should be non-decreasing
        for k in range(len(entry.wealth_quantiles) - 1):
            assert entry.wealth_quantiles[k] <= entry.wealth_quantiles[k + 1]

    def test_quantiles_equal_wealth(self, config: SimulationConfigV2) -> None:
        """All equal wealth -> all quantiles equal."""
        households = [_make_household(f"a{i}", wealth=50.0) for i in range(10)]
        ps = _make_period_state(households=households)
        observer = ObserverV2(config)
        entry = observer.observe(ps)
        assert all(q == pytest.approx(50.0) for q in entry.wealth_quantiles)


# ============================================================================
# Tests: Unemployment rate
# ============================================================================


class TestObserverV2Unemployment:
    """Test unemployment rate computation."""

    def test_no_unemployed(self, config: SimulationConfigV2) -> None:
        """All workers -> unemployment = 0."""
        households = [_make_household(f"a{i}", role=OccupationalRole.WORKER) for i in range(5)]
        ps = _make_period_state(households=households)
        observer = ObserverV2(config)
        entry = observer.observe(ps)
        assert entry.unemployment_rate == pytest.approx(0.0)

    def test_half_unemployed(self, config: SimulationConfigV2) -> None:
        """Half unemployed -> rate = 0.5."""
        households = [_make_household(f"a{i}", role=OccupationalRole.WORKER) for i in range(5)] + [
            _make_household(f"u{i}", role=OccupationalRole.UNEMPLOYED) for i in range(5)
        ]
        ps = _make_period_state(households=households)
        observer = ObserverV2(config)
        entry = observer.observe(ps)
        assert entry.unemployment_rate == pytest.approx(0.5)

    def test_all_unemployed(self, config: SimulationConfigV2) -> None:
        """All unemployed -> rate = 1.0."""
        households = [_make_household(f"a{i}", role=OccupationalRole.UNEMPLOYED) for i in range(5)]
        ps = _make_period_state(households=households)
        observer = ObserverV2(config)
        entry = observer.observe(ps)
        assert entry.unemployment_rate == pytest.approx(1.0)


# ============================================================================
# Tests: Firm statistics
# ============================================================================


class TestObserverV2FirmStats:
    """Test firm statistics computation."""

    def test_no_firms(self, config: SimulationConfigV2) -> None:
        """No firms -> zero stats."""
        ps = _make_period_state(firms=[])
        observer = ObserverV2(config)
        entry = observer.observe(ps)
        assert entry.num_active_firms == 0
        assert entry.mean_firm_size == 0.0
        assert entry.aggregate_rd_spend == 0.0

    def test_firm_count_and_size(self, config: SimulationConfigV2) -> None:
        """Two firms with workers -> correct count and mean size."""
        firms = [
            _make_firm("f0", worker_ids=["a0", "a1", "a2"]),
            _make_firm("f1", worker_ids=["a3"]),
        ]
        ps = _make_period_state(firms=firms)
        observer = ObserverV2(config)
        entry = observer.observe(ps)
        assert entry.num_active_firms == 2
        assert entry.mean_firm_size == pytest.approx(2.0)  # (3+1)/2

    def test_aggregate_rd_spend(self, config: SimulationConfigV2) -> None:
        """Total R&D aggregated across firms."""
        firms = [
            _make_firm("f0", rd_spend=10.0),
            _make_firm("f1", rd_spend=25.0),
        ]
        ps = _make_period_state(firms=firms)
        observer = ObserverV2(config)
        entry = observer.observe(ps)
        assert entry.aggregate_rd_spend == pytest.approx(35.0)


# ============================================================================
# Tests: Social welfare and cumulative welfare (REQ-031)
# ============================================================================


class TestObserverV2Welfare:
    """Test social welfare and cumulative discounted welfare."""

    def test_social_welfare_sum(self, config: SimulationConfigV2) -> None:
        """Social welfare = sum of realized utilities."""
        households = [_make_household(f"a{i}", realized_utility=float(i + 1)) for i in range(5)]
        ps = _make_period_state(households=households)
        observer = ObserverV2(config)
        entry = observer.observe(ps)
        assert entry.social_welfare == pytest.approx(15.0)  # 1+2+3+4+5

    def test_cumulative_welfare_discounting(self, config: SimulationConfigV2) -> None:
        """Cumulative welfare applies discount factor across periods."""
        households = [
            _make_household(f"a{i}", realized_utility=10.0, beta_discount=0.9) for i in range(5)
        ]
        observer = ObserverV2(config)

        ps1 = _make_period_state(period=1, households=households)
        entry1 = observer.observe(ps1)
        # First period: cumulative = 0 * 0.9 + 50 = 50
        assert entry1.cumulative_welfare == pytest.approx(50.0)

        ps2 = _make_period_state(period=2, households=households)
        entry2 = observer.observe(ps2)
        # Second period: cumulative = 50 * 0.9 + 50 = 95
        assert entry2.cumulative_welfare == pytest.approx(95.0)

    def test_zero_utility(self, config: SimulationConfigV2) -> None:
        """All zero utilities -> welfare = 0."""
        households = [_make_household(f"a{i}", realized_utility=0.0) for i in range(5)]
        ps = _make_period_state(households=households)
        observer = ObserverV2(config)
        entry = observer.observe(ps)
        assert entry.social_welfare == pytest.approx(0.0)


# ============================================================================
# Tests: Pareto efficiency
# ============================================================================


class TestObserverV2Pareto:
    """Test v2 Pareto efficiency computation."""

    def test_single_agent_is_pareto_efficient(self) -> None:
        """Single agent is always Pareto efficient."""
        households = [_make_household("a0")]
        score = ObserverV2.compute_pareto_efficiency_v2(households)
        assert score == 1.0

    def test_equal_wealth_is_efficient(self) -> None:
        """Equal wealth -> high Pareto score."""
        households = [_make_household(f"a{i}", wealth=100.0, consumption=10.0) for i in range(5)]
        score = ObserverV2.compute_pareto_efficiency_v2(households)
        assert score >= 0.8  # Nearly or fully efficient

    def test_score_in_valid_range(self) -> None:
        """Pareto score always in [0, 1]."""
        households = [
            _make_household(f"a{i}", wealth=float(i * 100), consumption=float(i * 10))
            for i in range(10)
        ]
        score = ObserverV2.compute_pareto_efficiency_v2(households)
        assert 0.0 <= score <= 1.0


# ============================================================================
# Tests: Market data in observation
# ============================================================================


class TestObserverV2MarketData:
    """Test market prices and aggregates in observation."""

    def test_wage_and_interest_rate(self, config: SimulationConfigV2) -> None:
        """Wage and interest rate from MarketState are recorded."""
        ps = _make_period_state()
        observer = ObserverV2(config)
        entry = observer.observe(ps)
        assert entry.wage == 1.0
        assert entry.interest_rate == 0.04

    def test_aggregate_output_and_investment(self, config: SimulationConfigV2) -> None:
        """Y and I from MarketState are recorded."""
        ps = _make_period_state()
        observer = ObserverV2(config)
        entry = observer.observe(ps)
        assert entry.aggregate_output == 100.0
        assert entry.aggregate_investment == 20.0

    def test_aggregate_consumption(self, config: SimulationConfigV2) -> None:
        """Aggregate consumption = sum of household consumption."""
        households = [_make_household(f"a{i}", consumption=float(i + 1)) for i in range(5)]
        ps = _make_period_state(households=households)
        observer = ObserverV2(config)
        entry = observer.observe(ps)
        assert entry.aggregate_consumption == pytest.approx(15.0)


# ============================================================================
# Tests: Rule change detection (v2)
# ============================================================================


class TestDetectRuleChangesV2:
    """Test detect_rule_changes_v2 function."""

    def test_no_prev(self) -> None:
        """First observation -> empty changes."""
        c = create_default_constitution()
        assert detect_rule_changes_v2(None, c) == []

    def test_no_changes(self) -> None:
        """Identical constitutions -> no changes."""
        c1 = create_default_constitution()
        c2 = c1.model_copy(deep=True)
        assert detect_rule_changes_v2(c1, c2) == []

    def test_rule_added(self) -> None:
        """Adding a rule is detected."""
        c1 = create_default_constitution()
        c2 = c1.model_copy(deep=True)
        c2.rules["new_rule"] = ConstitutionalRule(
            name="new_rule",
            rule_type=RuleType.MARKET_REGULATION,
            parameters={"min_wage": 5.0},
            description="New test rule",
        )
        changes = detect_rule_changes_v2(c1, c2)
        assert any("rule added" in c for c in changes)

    def test_rule_removed(self) -> None:
        """Removing a rule is detected."""
        c1 = create_default_constitution()
        c2 = c1.model_copy(deep=True)
        del c2.rules["private_property"]
        changes = detect_rule_changes_v2(c1, c2)
        assert any("rule removed" in c for c in changes)

    def test_parameter_change(self) -> None:
        """Changing rule parameters is detected."""
        c1 = create_default_constitution()
        c2 = c1.model_copy(deep=True)
        c2.rules["flat_tax"].parameters["rate"] = 0.3
        changes = detect_rule_changes_v2(c1, c2)
        assert any("rule modified" in c for c in changes)


# ============================================================================
# Tests: finalize() — REQ-032
# ============================================================================


class TestObserverV2Finalize:
    """Test finalize() produces correct SimulationOutputV2."""

    def test_finalize_output_type(self, config: SimulationConfigV2) -> None:
        """finalize() returns SimulationOutputV2."""
        observer = ObserverV2(config)
        ps = _make_period_state(period=3)
        observer.observe(ps)
        output = observer.finalize(ps)
        assert isinstance(output, SimulationOutputV2)

    def test_finalize_includes_history(self, config: SimulationConfigV2) -> None:
        """finalize() includes all recorded history entries."""
        observer = ObserverV2(config)
        for t in range(1, 4):
            ps = _make_period_state(period=t)
            observer.observe(ps)
        ps_final = _make_period_state(period=3)
        output = observer.finalize(ps_final)
        assert len(output.history) == 3

    def test_finalize_welfare_summary(self, config: SimulationConfigV2) -> None:
        """finalize() welfare summary has total welfare from all periods."""
        households = [_make_household(f"a{i}", realized_utility=10.0) for i in range(5)]
        observer = ObserverV2(config)
        for t in range(1, 4):
            ps = _make_period_state(period=t, households=households)
            observer.observe(ps)
        output = observer.finalize(_make_period_state(period=3, households=households))
        # 5 agents * 10 utility * 3 periods = 150
        assert output.welfare_summary.llm_total_welfare == pytest.approx(150.0)

    def test_finalize_seed_and_periods(self, config: SimulationConfigV2) -> None:
        """finalize() records seed from config and period from state."""
        observer = ObserverV2(config)
        ps = _make_period_state(period=5)
        observer.observe(ps)
        output = observer.finalize(ps)
        assert output.seed == config.seed
        assert output.total_periods == 5

    def test_finalize_deep_copies(self, config: SimulationConfigV2) -> None:
        """finalize() output is independent of input state."""
        observer = ObserverV2(config)
        ps = _make_period_state(period=1)
        observer.observe(ps)
        output = observer.finalize(ps)

        # Mutate original household
        ps.households[0].wealth = 999999.0
        assert output.final_households[0].wealth != 999999.0


# ============================================================================
# Tests: History accumulation
# ============================================================================


class TestObserverV2History:
    """Test that history entries accumulate correctly."""

    def test_history_length(self, config: SimulationConfigV2) -> None:
        """Each observe() appends one entry."""
        observer = ObserverV2(config)
        for t in range(1, 6):
            ps = _make_period_state(period=t)
            observer.observe(ps)
        assert len(observer.history) == 5

    def test_history_period_numbers(self, config: SimulationConfigV2) -> None:
        """History entries have correct period numbers."""
        observer = ObserverV2(config)
        for t in [1, 3, 5]:
            ps = _make_period_state(period=t)
            observer.observe(ps)
        periods = [e.period for e in observer.history]
        assert periods == [1, 3, 5]

    def test_constitution_snapshot_is_deep_copy(self, config: SimulationConfigV2) -> None:
        """Constitution snapshot in entry is independent of original."""
        c = create_default_constitution()
        ps = _make_period_state(constitution=c)
        observer = ObserverV2(config)
        entry = observer.observe(ps)

        # Mutate original constitution
        c.rules["flat_tax"].parameters["rate"] = 0.99
        assert entry.constitution_snapshot.rules["flat_tax"].parameters.get("rate") != 0.99
