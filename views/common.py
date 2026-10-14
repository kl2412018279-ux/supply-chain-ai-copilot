"""Shared UI helpers: palette, chart styling and formatting."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Reference categorical palette (validated for CVD and both themes): slot 1 and 2.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRID = "rgba(128,128,128,0.16)"
MUTED = "rgba(128,128,128,0.55)"

COMPANY = "Muara Logistics Sdn Bhd"


def rm(value: float, decimals: int = 2) -> str:
    return f"RM{value:,.{decimals}f}"


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f"## {title}")
    st.caption(subtitle)


def style(fig: go.Figure, *, height: int = 320, y_title: str = "", x_title: str = "") -> go.Figure:
    """Quiet chart chrome: transparent background, light grid, no zero line, legend on top."""
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=12, t=8, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(font_size=13),
        bargap=0.4,
    )
    fig.update_xaxes(title=x_title or None, showgrid=False, zeroline=False, ticks="")
    fig.update_yaxes(title=y_title or None, gridcolor=GRID, zeroline=False, ticks="")
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
