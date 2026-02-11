"""Tests for the reporter module."""

from __future__ import annotations

from emergent_constitution.models.agent import AgentState, UtilityParams, ValueVector
from emergent_constitution.models.constitution import (
    Constitution,
    PropertyRule,
    RedistributionRule,
    VotingRule,
)
from emergent_constitution.models.history import HistoryEntry, SimulationOutput
from emergent_constitution.reporter import (
    _format_constitutional_timeline,
    _format_final_constitution,
    _format_statistics_evolution,
    _format_wealth_distribution,
    generate_report,
)


def _make_agent(agent_id: str, wealth: float = 100.0) -> AgentState:
    """Helper to create an agent with minimal fields."""
    return AgentState(
        id=agent_id,
        wealth=wealth,
        productivity=10.0,
        utility_params=UtilityParams(alpha=0.5, beta=0.3, gamma=0.2),
        value_vector=ValueVector(equality=0.5, liberty=0.5),
    )


def _make_history_entry(
    tick: int,
    gini: float = 0.3,
    total_output: float = 50.0,
    mean_wealth: float = 100.0,
    median_wealth: float = 95.0,
    rule_changes: list[str] | None = None,
) -> HistoryEntry:
    """Helper to create a history entry with sensible defaults."""
    return HistoryEntry(
        tick=tick,
        gini=gini,
        total_output=total_output,
        mean_wealth=mean_wealth,
        median_wealth=median_wealth,
        rule_changes=rule_changes or [],
        constitution_snapshot=Constitution(),
    )


class TestGenerateReport:
    """Tests for the top-level generate_report function."""

    def test_returns_nonempty_string(self):
        """Report is a non-empty string."""
        output = SimulationOutput(
            constitution=Constitution(),
            history=[],
            final_agent_states=[_make_agent("a0")],
            seed=42,
            total_ticks=10,
        )
        report = generate_report(output)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_contains_all_section_headers(self):
        """Report includes all five expected section headers."""
        agents = [_make_agent(f"a{i}", wealth=50.0 + i * 10) for i in range(5)]
        output = SimulationOutput(
            constitution=Constitution(),
            history=[_make_history_entry(5)],
            final_agent_states=agents,
            seed=123,
            total_ticks=20,
        )
        report = generate_report(output)
        assert "# The Emergent Constitution" in report
        assert "## Final Constitution" in report
        assert "## Wealth Distribution" in report
        assert "## Constitutional Timeline" in report
        assert "## Statistics Over Time" in report

    def test_contains_metadata(self):
        """Report includes seed, tick count, and agent count."""
        agents = [_make_agent(f"a{i}") for i in range(3)]
        output = SimulationOutput(
            constitution=Constitution(),
            history=[],
            final_agent_states=agents,
            seed=99,
            total_ticks=50,
        )
        report = generate_report(output)
        assert "**Seed:** 99" in report
        assert "**Total ticks:** 50" in report
        assert "**Number of agents:** 3" in report


class TestFormatFinalConstitution:
    """Tests for constitution formatting."""

    def test_displays_all_four_rules(self):
        """All constitutional rules appear in the output."""
        text = _format_final_constitution(Constitution())
        assert "Property rule" in text
        assert "Tax rate" in text
        assert "Voting rule" in text
        assert "Redistribution rule" in text

    def test_tax_as_decimal_and_percent(self):
        """Tax rate is shown as both decimal and percentage."""
        c = Constitution(tax_rate=0.25)
        text = _format_final_constitution(c)
        assert "0.2500" in text
        assert "25%" in text

    def test_default_constitution_values(self):
        """Default constitution shows expected values."""
        text = _format_final_constitution(Constitution())
        assert "private" in text
        assert "0.0000" in text
        assert "0%" in text
        assert "majority" in text
        assert "flat" in text

    def test_non_default_constitution(self):
        """Non-default constitution displays correctly."""
        c = Constitution(
            property_rule=PropertyRule.COMMUNAL,
            tax_rate=0.6,
            voting_rule=VotingRule.SUPERMAJORITY,
            redistribution_rule=RedistributionRule.PROGRESSIVE,
        )
        text = _format_final_constitution(c)
        assert "communal" in text
        assert "60%" in text
        assert "supermajority" in text
        assert "progressive" in text


class TestFormatWealthDistribution:
    """Tests for wealth distribution formatting."""

    def test_shows_statistics(self):
        """Output contains min, max, mean, median, stdev, Gini."""
        agents = [_make_agent("a0", 50.0), _make_agent("a1", 150.0)]
        text = _format_wealth_distribution(agents)
        assert "Min:" in text
        assert "Max:" in text
        assert "Mean:" in text
        assert "Median:" in text
        assert "Std dev:" in text
        assert "Gini coefficient:" in text

    def test_all_equal_wealth(self):
        """All-equal wealth produces Gini of 0."""
        agents = [_make_agent(f"a{i}", 100.0) for i in range(5)]
        text = _format_wealth_distribution(agents)
        assert "0.0000" in text

    def test_single_agent_stdev_na(self):
        """Single agent shows N/A for std dev."""
        agents = [_make_agent("a0")]
        text = _format_wealth_distribution(agents)
        assert "N/A" in text

    def test_quintile_breakdown_with_enough_agents(self):
        """Quintile table appears when there are 5+ agents."""
        agents = [_make_agent(f"a{i}", wealth=float(i * 10)) for i in range(10)]
        text = _format_wealth_distribution(agents)
        assert "Quintile" in text
        assert "Bottom 20%" in text
        assert "Top 20%" in text

    def test_no_quintile_with_few_agents(self):
        """Quintile table does not appear with fewer than 5 agents."""
        agents = [_make_agent(f"a{i}") for i in range(3)]
        text = _format_wealth_distribution(agents)
        assert "Quintile" not in text

    def test_empty_agents(self):
        """Empty agents list produces a graceful message."""
        text = _format_wealth_distribution([])
        assert "No agents" in text


class TestFormatConstitutionalTimeline:
    """Tests for constitutional timeline formatting."""

    def test_no_history(self):
        """Empty history produces 'No observations' message."""
        text = _format_constitutional_timeline([])
        assert "No observations recorded" in text

    def test_no_changes(self):
        """History with no rule changes says so."""
        history = [_make_history_entry(5), _make_history_entry(10)]
        text = _format_constitutional_timeline(history)
        assert "No constitutional changes" in text

    def test_with_changes(self):
        """Rule changes appear with tick numbers."""
        history = [
            _make_history_entry(5, rule_changes=["tax_rate: 0.0 → 0.25"]),
            _make_history_entry(10, rule_changes=["voting_rule: majority → supermajority"]),
        ]
        text = _format_constitutional_timeline(history)
        assert "Tick 5" in text
        assert "tax_rate" in text
        assert "Tick 10" in text
        assert "voting_rule" in text


class TestFormatStatisticsEvolution:
    """Tests for statistics evolution table."""

    def test_no_history(self):
        """Empty history produces 'No observations' message."""
        text = _format_statistics_evolution([])
        assert "No observations recorded" in text

    def test_table_header(self):
        """Table has the expected column headers."""
        history = [_make_history_entry(5)]
        text = _format_statistics_evolution(history)
        assert "Tick" in text
        assert "Gini" in text
        assert "Total Output" in text
        assert "Mean Wealth" in text
        assert "Median Wealth" in text

    def test_correct_values(self):
        """Table contains the exact values from the history entry."""
        history = [
            _make_history_entry(
                10,
                gini=0.35,
                total_output=500.0,
                mean_wealth=100.0,
                median_wealth=90.0,
            ),
        ]
        text = _format_statistics_evolution(history)
        assert "0.3500" in text
        assert "500.00" in text
        assert "100.00" in text
        assert "90.00" in text
