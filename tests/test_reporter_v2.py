"""Tests for the v2 reporter module."""

from __future__ import annotations

import json

from emergent_constitution.models.constitution import (
    ConstitutionV2,
    create_default_constitution,
)
from emergent_constitution.models.firm import FirmState
from emergent_constitution.models.history import (
    HistoryEntryV2,
    SimulationOutputV2,
    WelfareSummary,
)
from emergent_constitution.models.household import (
    HouseholdState,
    OccupationalRole,
    UtilityParams,
    ValueVector,
)
from emergent_constitution.reporter import (
    _format_constitutional_timeline_v2,
    _format_final_constitution_v2,
    _format_firm_summary,
    _format_statistics_evolution_v2,
    _format_wealth_distribution_v2,
    _format_welfare_summary,
    generate_json_v2,
    generate_report_v2,
)


def _make_household(
    agent_id: str,
    wealth: float = 100.0,
    consumption: float = 10.0,
    role: OccupationalRole = OccupationalRole.WORKER,
    realized_utility: float = 5.0,
) -> HouseholdState:
    """Helper to create a v2 household with minimal fields."""
    return HouseholdState(
        id=agent_id,
        wealth=wealth,
        productivity=1.0,
        productivity_index=0,
        utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
        role=role,
        consumption=consumption,
        leisure=0.3,
        labor_supply=0.7,
        realized_utility=realized_utility,
    )


def _make_firm(
    firm_id: str = "firm_0000",
    owner_id: str = "agent_0000",
    workers: int = 3,
    rd_spend: float = 5.0,
    output: float = 50.0,
    profit: float = 10.0,
) -> FirmState:
    """Helper to create a firm with sensible defaults."""
    return FirmState(
        id=firm_id,
        owner_id=owner_id,
        capital=100.0,
        labor_demand=float(workers),
        tfp=1.0,
        worker_ids=[f"agent_{i:04d}" for i in range(workers)],
        rd_spend=rd_spend,
        output=output,
        profit=profit,
    )


def _make_history_entry_v2(
    period: int,
    gini: float = 0.3,
    aggregate_output: float = 500.0,
    mean_wealth: float = 100.0,
    median_wealth: float = 95.0,
    rule_changes: list[str] | None = None,
    social_welfare: float = 50.0,
    cumulative_welfare: float = 100.0,
    wage: float = 1.0,
    interest_rate: float = 0.05,
) -> HistoryEntryV2:
    """Helper to create a v2 history entry."""
    return HistoryEntryV2(
        period=period,
        gini=gini,
        pareto_score=0.9,
        aggregate_output=aggregate_output,
        aggregate_consumption=400.0,
        aggregate_investment=100.0,
        mean_wealth=mean_wealth,
        median_wealth=median_wealth,
        wealth_quantiles=[20.0, 50.0, 95.0, 150.0, 200.0],
        unemployment_rate=0.1,
        num_active_firms=5,
        mean_firm_size=4.0,
        aggregate_rd_spend=25.0,
        social_welfare=social_welfare,
        cumulative_welfare=cumulative_welfare,
        wage=wage,
        interest_rate=interest_rate,
        rule_changes=rule_changes or [],
        constitution_snapshot=create_default_constitution(),
    )


def _make_output(
    num_households: int = 10,
    num_firms: int = 2,
    num_history: int = 3,
) -> SimulationOutputV2:
    """Helper to create a complete v2 simulation output."""
    households = [
        _make_household(f"agent_{i:04d}", wealth=50.0 + i * 10) for i in range(num_households)
    ]
    firms = [_make_firm(f"firm_{i:04d}", f"agent_{i:04d}") for i in range(num_firms)]
    history = [_make_history_entry_v2(period=i) for i in range(num_history)]
    welfare = WelfareSummary(
        llm_total_welfare=150.0,
        benchmark_total_welfare=140.0,
    )
    return SimulationOutputV2(
        constitution=create_default_constitution(),
        history=history,
        final_households=households,
        final_firms=firms,
        welfare_summary=welfare,
        seed=42,
        total_periods=100,
    )


# ============================================================================
# generate_report_v2
# ============================================================================


class TestGenerateReportV2:
    """Tests for the top-level generate_report_v2 function."""

    def test_returns_nonempty_string(self):
        output = _make_output()
        report = generate_report_v2(output)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_contains_all_section_headers(self):
        output = _make_output()
        report = generate_report_v2(output)
        assert "# The Emergent Constitution" in report
        assert "## Final Constitution" in report
        assert "## Wealth Distribution" in report
        assert "## Firm Summary" in report
        assert "## Welfare Summary" in report
        assert "## Constitutional Timeline" in report
        assert "## Statistics Over Time" in report

    def test_contains_v2_metadata(self):
        output = _make_output(num_households=10, num_firms=2)
        report = generate_report_v2(output)
        assert "**Seed:** 42" in report
        assert "**Total periods:** 100" in report
        assert "**Number of households:** 10" in report
        assert "**Number of firms:** 2" in report


# ============================================================================
# Final Constitution (v2)
# ============================================================================


class TestFormatFinalConstitutionV2:
    """Tests for v2 constitution formatting."""

    def test_displays_voting_rule(self):
        constitution = create_default_constitution()
        text = _format_final_constitution_v2(constitution)
        assert "majority_vote" in text

    def test_displays_all_rules(self):
        constitution = create_default_constitution()
        text = _format_final_constitution_v2(constitution)
        assert "flat_tax" in text
        assert "flat_transfer" in text
        assert "public_goods_provision" in text
        assert "majority_vote" in text
        assert "private_property" in text

    def test_displays_rule_types(self):
        constitution = create_default_constitution()
        text = _format_final_constitution_v2(constitution)
        assert "tax_schedule" in text
        assert "transfer_program" in text
        assert "voting_procedure" in text

    def test_displays_parameters(self):
        constitution = create_default_constitution()
        text = _format_final_constitution_v2(constitution)
        assert "rate=0" in text  # flat_tax rate
        assert "threshold=0.5" in text  # majority_vote threshold

    def test_empty_constitution(self):
        constitution = ConstitutionV2()
        text = _format_final_constitution_v2(constitution)
        assert "Final Constitution" in text


# ============================================================================
# Wealth Distribution (v2)
# ============================================================================


class TestFormatWealthDistributionV2:
    """Tests for v2 wealth distribution formatting."""

    def test_shows_statistics(self):
        households = [_make_household("a0", 50.0), _make_household("a1", 150.0)]
        text = _format_wealth_distribution_v2(households)
        assert "Min:" in text
        assert "Max:" in text
        assert "Mean:" in text
        assert "Median:" in text
        assert "Std dev:" in text
        assert "Gini coefficient:" in text

    def test_quantiles_with_enough_agents(self):
        households = [_make_household(f"a{i}", float(i * 10)) for i in range(10)]
        text = _format_wealth_distribution_v2(households)
        assert "Wealth Quantiles" in text
        assert "p10" in text
        assert "p25" in text
        assert "p50" in text
        assert "p75" in text
        assert "p90" in text

    def test_quintile_breakdown_with_enough_agents(self):
        households = [_make_household(f"a{i}", float(i * 10)) for i in range(10)]
        text = _format_wealth_distribution_v2(households)
        assert "Quintile Breakdown" in text
        assert "Bottom 20%" in text
        assert "Top 20%" in text

    def test_no_quantiles_with_few_agents(self):
        households = [_make_household(f"a{i}") for i in range(3)]
        text = _format_wealth_distribution_v2(households)
        assert "Wealth Quantiles" not in text

    def test_empty_households(self):
        text = _format_wealth_distribution_v2([])
        assert "No households" in text

    def test_equal_wealth_gini_zero(self):
        households = [_make_household(f"a{i}", 100.0) for i in range(5)]
        text = _format_wealth_distribution_v2(households)
        assert "0.0000" in text


# ============================================================================
# Firm Summary
# ============================================================================


class TestFormatFirmSummary:
    """Tests for firm summary formatting."""

    def test_no_firms(self):
        output = _make_output(num_firms=0)
        text = _format_firm_summary(output)
        assert "No active firms" in text

    def test_firm_count(self):
        output = _make_output(num_firms=3)
        text = _format_firm_summary(output)
        assert "Number of firms:** 3" in text

    def test_firm_table_headers(self):
        output = _make_output(num_firms=2)
        text = _format_firm_summary(output)
        assert "Firm" in text
        assert "Owner" in text
        assert "Workers" in text
        assert "Capital" in text
        assert "TFP" in text
        assert "Output" in text
        assert "Profit" in text
        assert "R&D" in text

    def test_firm_details_in_table(self):
        output = _make_output(num_firms=1)
        text = _format_firm_summary(output)
        assert "firm_0000" in text
        assert "agent_0000" in text

    def test_aggregate_stats(self):
        output = _make_output(num_firms=2)
        text = _format_firm_summary(output)
        assert "Total R&D spending:" in text
        assert "Total firm output:" in text
        assert "Total firm profit:" in text
        assert "Mean firm size:" in text


# ============================================================================
# Welfare Summary
# ============================================================================


class TestFormatWelfareSummary:
    """Tests for welfare summary formatting."""

    def test_llm_welfare_shown(self):
        welfare = WelfareSummary(llm_total_welfare=150.0)
        text = _format_welfare_summary(welfare)
        assert "LLM total welfare:" in text
        assert "150.0000" in text

    def test_benchmark_shown_when_present(self):
        welfare = WelfareSummary(
            llm_total_welfare=150.0,
            benchmark_total_welfare=140.0,
        )
        text = _format_welfare_summary(welfare)
        assert "Benchmark total welfare:" in text
        assert "140.0000" in text
        assert "LLM / Benchmark ratio:" in text

    def test_no_benchmark_when_none(self):
        welfare = WelfareSummary(llm_total_welfare=150.0)
        text = _format_welfare_summary(welfare)
        assert "Benchmark" not in text

    def test_per_agent_comparison(self):
        welfare = WelfareSummary(
            llm_total_welfare=100.0,
            per_agent_comparison={
                "agent_0000": {"llm": 50.0, "benchmark": 45.0},
                "agent_0001": {"llm": 50.0, "benchmark": 55.0},
            },
        )
        text = _format_welfare_summary(welfare)
        assert "Per-Agent Comparison" in text
        assert "agent_0000" in text
        assert "agent_0001" in text


# ============================================================================
# Constitutional Timeline (v2)
# ============================================================================


class TestFormatConstitutionalTimelineV2:
    """Tests for v2 constitutional timeline."""

    def test_no_history(self):
        text = _format_constitutional_timeline_v2([])
        assert "No observations recorded" in text

    def test_no_changes(self):
        history = [_make_history_entry_v2(0), _make_history_entry_v2(5)]
        text = _format_constitutional_timeline_v2(history)
        assert "No governance activity" in text

    def test_with_changes(self):
        history = [
            _make_history_entry_v2(5, rule_changes=["rule added: carbon_tax"]),
            _make_history_entry_v2(
                10, rule_changes=["voting_rule: majority_vote -> supermajority"]
            ),
        ]
        text = _format_constitutional_timeline_v2(history)
        assert "Period 5" in text
        assert "carbon_tax" in text
        assert "Period 10" in text
        assert "supermajority" in text


# ============================================================================
# Statistics Evolution (v2)
# ============================================================================


class TestFormatStatisticsEvolutionV2:
    """Tests for v2 statistics evolution table."""

    def test_no_history(self):
        text = _format_statistics_evolution_v2([])
        assert "No observations recorded" in text

    def test_table_has_all_columns(self):
        history = [_make_history_entry_v2(0)]
        text = _format_statistics_evolution_v2(history)
        for col in [
            "Period",
            "Gini",
            "Pareto",
            "Output",
            "Consumption",
            "Investment",
            "Mean Wealth",
            "Median Wealth",
            "Unemployment",
            "Firms",
            "Welfare",
            "Cum. Welfare",
            "Wage",
            "Rate",
        ]:
            assert col in text

    def test_values_in_output(self):
        history = [
            _make_history_entry_v2(
                period=5,
                gini=0.35,
                aggregate_output=500.0,
                mean_wealth=100.0,
                median_wealth=90.0,
                social_welfare=50.0,
                wage=1.2,
                interest_rate=0.05,
            ),
        ]
        text = _format_statistics_evolution_v2(history)
        assert "0.3500" in text  # gini
        assert "500.00" in text  # output
        assert "100.00" in text  # mean wealth
        assert "90.00" in text  # median wealth
        assert "50.00" in text  # welfare
        assert "1.2000" in text  # wage
        assert "0.0500" in text  # interest rate


# ============================================================================
# JSON Serialization (v2)
# ============================================================================


class TestGenerateJsonV2:
    """Tests for generate_json_v2."""

    def test_returns_valid_json(self):
        output = _make_output()
        result = generate_json_v2(output)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_has_top_level_keys(self):
        output = _make_output()
        result = generate_json_v2(output)
        parsed = json.loads(result)
        assert "constitution" in parsed
        assert "history" in parsed
        assert "final_households" in parsed
        assert "final_firms" in parsed
        assert "welfare_summary" in parsed
        assert "seed" in parsed
        assert "total_periods" in parsed

    def test_constitution_has_rules(self):
        output = _make_output()
        parsed = json.loads(generate_json_v2(output))
        assert "rules" in parsed["constitution"]
        assert "voting_rule" in parsed["constitution"]

    def test_history_entries_have_all_fields(self):
        output = _make_output(num_history=2)
        parsed = json.loads(generate_json_v2(output))
        assert len(parsed["history"]) == 2
        entry = parsed["history"][0]
        for field in [
            "period",
            "gini",
            "pareto_score",
            "aggregate_output",
            "aggregate_consumption",
            "aggregate_investment",
            "mean_wealth",
            "median_wealth",
            "wealth_quantiles",
            "unemployment_rate",
            "num_active_firms",
            "mean_firm_size",
            "aggregate_rd_spend",
            "social_welfare",
            "cumulative_welfare",
            "wage",
            "interest_rate",
            "rule_changes",
            "constitution_snapshot",
        ]:
            assert field in entry, f"Missing field: {field}"

    def test_households_serialized(self):
        output = _make_output(num_households=5)
        parsed = json.loads(generate_json_v2(output))
        assert len(parsed["final_households"]) == 5
        h = parsed["final_households"][0]
        assert "id" in h
        assert "wealth" in h
        assert "utility_params" in h
        assert "role" in h

    def test_firms_serialized(self):
        output = _make_output(num_firms=3)
        parsed = json.loads(generate_json_v2(output))
        assert len(parsed["final_firms"]) == 3
        f = parsed["final_firms"][0]
        assert "id" in f
        assert "owner_id" in f
        assert "worker_ids" in f
        assert "rd_spend" in f

    def test_welfare_summary_serialized(self):
        output = _make_output()
        parsed = json.loads(generate_json_v2(output))
        ws = parsed["welfare_summary"]
        assert "llm_total_welfare" in ws
        assert "benchmark_total_welfare" in ws

    def test_roundtrip_seed(self):
        output = _make_output()
        parsed = json.loads(generate_json_v2(output))
        assert parsed["seed"] == 42
        assert parsed["total_periods"] == 100

    def test_empty_firms_serialized(self):
        output = _make_output(num_firms=0)
        parsed = json.loads(generate_json_v2(output))
        assert parsed["final_firms"] == []
