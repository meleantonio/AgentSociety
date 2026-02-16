"""Shared Plotly chart builder utilities for the dashboard."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

_TEMPLATE = "plotly_white"
_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def make_time_series(
    data: dict[str, list[float]],
    periods: list[int],
    title: str,
    y_label: str = "",
) -> go.Figure:
    """Standard multi-trace line chart.

    Args:
        data: Mapping of trace name to list of y-values.
        periods: X-axis period numbers.
        title: Chart title.
        y_label: Y-axis label.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    for i, (name, values) in enumerate(data.items()):
        dash = "dash" if name == "Population Average" else None
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=values,
                mode="lines",
                name=name,
                line={"color": _COLORS[i % len(_COLORS)], "dash": dash},
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title=y_label,
        template=_TEMPLATE,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"t": 50, "b": 40, "l": 50, "r": 20},
    )
    return fig


def make_dual_axis(
    periods: list[int],
    left_data: dict[str, list[float]],
    right_data: dict[str, list[float]],
    title: str,
    left_label: str = "",
    right_label: str = "",
) -> go.Figure:
    """Line chart with two y-axes.

    Args:
        periods: X-axis period numbers.
        left_data: Traces for left y-axis.
        right_data: Traces for right y-axis.
        title: Chart title.
        left_label: Left y-axis label.
        right_label: Right y-axis label.

    Returns:
        Plotly Figure with dual axes.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    color_idx = 0
    for name, values in left_data.items():
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=values,
                mode="lines",
                name=name,
                line={"color": _COLORS[color_idx % len(_COLORS)]},
            ),
            secondary_y=False,
        )
        color_idx += 1
    for name, values in right_data.items():
        fig.add_trace(
            go.Scatter(
                x=periods,
                y=values,
                mode="lines",
                name=name,
                line={"color": _COLORS[color_idx % len(_COLORS)], "dash": "dot"},
            ),
            secondary_y=True,
        )
        color_idx += 1
    fig.update_layout(
        title=title,
        xaxis_title="Period",
        template=_TEMPLATE,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"t": 50, "b": 40, "l": 50, "r": 50},
    )
    fig.update_yaxes(title_text=left_label, secondary_y=False)
    fig.update_yaxes(title_text=right_label, secondary_y=True)
    return fig


def make_fan_chart(
    periods: list[int],
    quantiles: list[list[float]],
    title: str,
) -> go.Figure:
    """Filled area chart showing quantile bands (p10-p90).

    Args:
        periods: X-axis period numbers.
        quantiles: List of [p10, p25, p50, p75, p90] per period.
        title: Chart title.

    Returns:
        Plotly Figure with shaded bands.
    """
    if not quantiles or not quantiles[0]:
        fig = go.Figure()
        fig.update_layout(title=title, template=_TEMPLATE)
        fig.add_annotation(text="No quantile data available", showarrow=False, font={"size": 14})
        return fig

    p10 = [q[0] if len(q) > 0 else 0 for q in quantiles]
    p25 = [q[1] if len(q) > 1 else 0 for q in quantiles]
    p50 = [q[2] if len(q) > 2 else 0 for q in quantiles]
    p75 = [q[3] if len(q) > 3 else 0 for q in quantiles]
    p90 = [q[4] if len(q) > 4 else 0 for q in quantiles]

    fig = go.Figure()
    # p10-p90 band
    fig.add_trace(
        go.Scatter(
            x=periods + periods[::-1],
            y=p90 + p10[::-1],
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.1)",
            line={"color": "rgba(255,255,255,0)"},
            name="p10-p90",
            showlegend=True,
        )
    )
    # p25-p75 band
    fig.add_trace(
        go.Scatter(
            x=periods + periods[::-1],
            y=p75 + p25[::-1],
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.25)",
            line={"color": "rgba(255,255,255,0)"},
            name="p25-p75",
            showlegend=True,
        )
    )
    # Median line
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=p50,
            mode="lines",
            name="Median",
            line={"color": "#636EFA", "width": 2},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title="Wealth",
        template=_TEMPLATE,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"t": 50, "b": 40, "l": 50, "r": 20},
    )
    return fig


def make_bar_chart(
    periods: list[int],
    data: dict[str, list[Any]],
    title: str,
    y_label: str = "",
) -> go.Figure:
    """Grouped bar chart.

    Args:
        periods: X-axis categories.
        data: Mapping of trace name to y-values.
        title: Chart title.
        y_label: Y-axis label.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    for i, (name, values) in enumerate(data.items()):
        fig.add_trace(
            go.Bar(
                x=periods,
                y=values,
                name=name,
                marker_color=_COLORS[i % len(_COLORS)],
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis_title=y_label,
        barmode="group",
        template=_TEMPLATE,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"t": 50, "b": 40, "l": 50, "r": 20},
    )
    return fig


def make_scatter_timeline(
    periods: list[int],
    labels: list[str],
    colors: list[str],
    hover_texts: list[str],
    title: str,
) -> go.Figure:
    """Scatter plot timeline for constitutional events.

    Args:
        periods: X positions (periods).
        labels: Marker labels.
        colors: Marker colors per point.
        hover_texts: Hover text per point.
        title: Chart title.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=[1] * len(periods),
            mode="markers+text",
            text=labels,
            textposition="top center",
            hovertext=hover_texts,
            hoverinfo="text",
            marker={"color": colors, "size": 12, "symbol": "diamond"},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Period",
        yaxis={"visible": False},
        template=_TEMPLATE,
        showlegend=False,
        margin={"t": 50, "b": 40, "l": 50, "r": 20},
    )
    return fig
