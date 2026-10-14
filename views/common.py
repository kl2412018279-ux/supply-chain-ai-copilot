"""Shared UI helpers: palette, page chrome, chart styling and formatting."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Reference categorical palette (validated for CVD and both themes): slot 1 and 2.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRID = "rgba(128,128,128,0.16)"
MUTED = "rgba(128,128,128,0.55)"
# Status colours: one meaning each, everywhere.
GOOD = "#1baf7a"
WARN = "#eda100"
BAD = "#e34948"

COMPANY = "Muara Logistics Sdn Bhd"

_CSS = """
<style>
[data-testid="stMainBlockContainer"] {
  padding-top: 4.4rem !important; padding-bottom: 3rem; max-width: 1440px; }
h2 { letter-spacing: -0.02em; }

/* page header */
.ph-eyebrow { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; opacity: 0.55; margin-bottom: -0.35rem; }
.ph-sub { opacity: 0.68; font-size: 0.93rem; margin-top: -0.4rem; max-width: 62rem; }

/* section titles */
.sec { display: flex; align-items: baseline; gap: 0.6rem; margin: 1.4rem 0 0.2rem; }
.sec b { font-size: 1.05rem; }
.sec span { opacity: 0.6; font-size: 0.85rem; }

/* metric cards */
[data-testid="stMetric"] { padding: 0.9rem 1rem 0.7rem; }
[data-testid="stMetricLabel"] p { font-size: 0.8rem; opacity: 0.75; }

/* pills */
.pill { display: inline-block; padding: 0.08rem 0.55rem; border-radius: 999px;
  font-size: 0.78rem; font-weight: 600; line-height: 1.5; white-space: nowrap; }
.pill.good { background: rgba(27,175,122,0.14); color: #148a60; }
.pill.warn { background: rgba(237,161,0,0.16); color: #a66f00; }
.pill.bad  { background: rgba(227,73,72,0.14); color: #c23a39; }
.pill.info { background: rgba(42,120,214,0.13); color: #1f62b5; }
@media (prefers-color-scheme: dark) {
  .pill.good { color: #4fd3a0; } .pill.warn { color: #f4bf4f; }
  .pill.bad { color: #ff7d7c; } .pill.info { color: #7fb0f0; }
}

/* sidebar brand */
.brand { display: flex; gap: 0.7rem; align-items: center; margin: 0.2rem 0 0.4rem; }
.brand .mark { width: 2.1rem; height: 2.1rem; border-radius: 0.5rem; background: #2a78d6;
  color: #fff; display: grid; place-items: center; font-weight: 700; font-size: 1.05rem; }
.brand .name { font-weight: 700; line-height: 1.15; }
.brand .tag { font-size: 0.76rem; opacity: 0.6; }
.kv { display: flex; justify-content: space-between; font-size: 0.86rem; padding: 0.18rem 0; }
.kv span:first-child { opacity: 0.62; }
.kv span:last-child { font-weight: 600; }
</style>
"""


def inject_css() -> None:
    st.html(_CSS)


def rm(value: float, decimals: int = 2) -> str:
    return f"RM{value:,.{decimals}f}"


def rm_short(value: float) -> str:
    """RM1.2m / RM84.3k / RM950 for headline numbers."""
    a = abs(value)
    if a >= 1e6:
        return f"RM{value / 1e6:,.2f}m"
    if a >= 1e4:
        return f"RM{value / 1e3:,.1f}k"
    return f"RM{value:,.0f}"


def page_header(title: str, subtitle: str, eyebrow: str | None = None) -> None:
    if eyebrow:
        st.html(f'<div class="ph-eyebrow">{eyebrow}</div>')
    st.markdown(f"## {title}")
    st.html(f'<div class="ph-sub">{subtitle}</div>')


def section(title: str, note: str = "") -> None:
    st.html(f'<div class="sec"><b>{title}</b><span>{note}</span></div>')


def pill(text: str, tone: str = "info") -> str:
    return f'<span class="pill {tone}">{text}</span>'


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
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
