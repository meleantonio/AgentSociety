"""Unit tests for dashboard chart builder functions."""

from __future__ import annotations

import pytest

from emergent_constitution.dashboard.charts import (
    make_bar_chart,
    make_dual_axis,
    make_fan_chart,
    make_scatter_timeline,
    make_time_series,
)


class TestMakeTimeSeries:
    """Tests for the standard line chart builder."""

    def test_single_trace(self) -> None:
        fig = make_time_series(
            {"GDP": [100.0, 110.0, 120.0]},
            [1, 2, 3],
            "GDP Over Time",
            "Value",
        )
        assert len(fig.data) == 1
        assert fig.data[0].name == "GDP"
        assert list(fig.data[0].x) == [1, 2, 3]

    def test_multiple_traces(self) -> None:
        fig = make_time_series(
            {"A": [1.0, 2.0], "B": [3.0, 4.0]},
            [5, 10],
            "Test",
        )
        assert len(fig.data) == 2

    def test_population_average_dashed(self) -> None:
        fig = make_time_series(
            {"Agent 1": [1.0], "Population Average": [2.0]},
            [1],
            "Test",
        )
        assert fig.data[1].line.dash == "dash"


class TestMakeDualAxis:
    """Tests for dual-axis chart builder."""

    def test_dual_traces(self) -> None:
        fig = make_dual_axis(
            [1, 2],
            left_data={"Wage": [10.0, 11.0]},
            right_data={"Rate": [0.05, 0.06]},
            title="Prices",
        )
        assert len(fig.data) == 2

    def test_right_axis_dotted(self) -> None:
        fig = make_dual_axis(
            [1],
            left_data={"L": [1.0]},
            right_data={"R": [2.0]},
            title="T",
        )
        assert fig.data[1].line.dash == "dot"


class TestMakeFanChart:
    """Tests for the quantile fan chart builder."""

    def test_with_quantiles(self) -> None:
        quantiles = [[10, 25, 50, 75, 90], [12, 28, 55, 80, 95]]
        fig = make_fan_chart([1, 2], quantiles, "Wealth")
        # Should have 3 traces: p10-p90 band, p25-p75 band, median line
        assert len(fig.data) == 3

    def test_empty_quantiles(self) -> None:
        fig = make_fan_chart([1, 2], [[], []], "Wealth")
        # Should have annotation about no data
        assert len(fig.layout.annotations) > 0

    def test_no_data(self) -> None:
        fig = make_fan_chart([], [], "Wealth")
        assert fig is not None


class TestMakeBarChart:
    """Tests for the grouped bar chart builder."""

    def test_bar_chart(self) -> None:
        fig = make_bar_chart(
            [5, 10],
            {"Submitted": [3, 5], "Passed": [1, 2]},
            "Governance Activity",
        )
        assert len(fig.data) == 2
        assert fig.data[0].name == "Submitted"


class TestMakeScatterTimeline:
    """Tests for the scatter timeline builder."""

    def test_scatter(self) -> None:
        fig = make_scatter_timeline(
            [5, 10, 15],
            ["A", "M", "R"],
            ["green", "orange", "red"],
            ["Added X", "Modified Y", "Removed Z"],
            "Timeline",
        )
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == 3
