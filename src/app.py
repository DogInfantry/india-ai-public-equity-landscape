"""Streamlit front end for the AI-augmented wealth idea engine."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import build_ai_india_dataset, cache_stats_path
from src.reporting import archetype_output_table, format_pct
from src.scoring import ARCHETYPE_CONFIGS, attach_rationales, compute_factor_scores, score_for_archetype


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total == 0:
        return weights
    return {name: value / total for name, value in weights.items()}


@st.cache_data(ttl=3600, show_spinner=False)
def load_base_dataset(use_cached_stats: bool) -> pd.DataFrame:
    """Load the scoring base table from cache or from live public market data."""

    stats_path = cache_stats_path()
    if use_cached_stats and stats_path.exists():
        base_stats = pd.read_csv(stats_path)
    else:
        _, _, _, base_stats = build_ai_india_dataset(start="2021-01-01")
        base_stats = compute_factor_scores(base_stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        base_stats.to_csv(stats_path, index=False)
    return compute_factor_scores(base_stats)


def format_score(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.2f}"


def _render_theme_map(stats: pd.DataFrame) -> None:
    """Render an interactive AI-purity vs market-cap scatter plot."""

    required = {"ai_purity_score", "market_cap", "segment", "name", "ticker"}
    missing = required.difference(stats.columns)
    if missing:
        st.info(f"Theme Map unavailable — missing columns: {', '.join(sorted(missing))}")
        return

    plot_df = stats.dropna(subset=["ai_purity_score", "market_cap"]).copy()
    if plot_df.empty:
        st.info("No data available for Theme Map yet. Run a live data refresh first.")
        return

    plot_df["market_cap_cr"] = plot_df["market_cap"] / 1e7  # convert INR to Cr

    size_col = None
    for candidate in ("sharpe", "return_1y", "cagr_3y"):
        if candidate in plot_df.columns and plot_df[candidate].notna().any():
            numeric = pd.to_numeric(plot_df[candidate], errors="coerce")
            if numeric.notna().any():
                plot_df["_size"] = (numeric - numeric.min() + 0.01).clip(lower=0.01)
                size_col = "_size"
                break

    fig = px.scatter(
        plot_df,
        x="market_cap_cr",
        y="ai_purity_score",
        color="segment",
        hover_name="name",
        size=size_col,
        size_max=30,
        text="ticker",
        log_x=True,
        title="AI-Purity vs Market Cap — India AI Universe",
        labels={
            "market_cap_cr": "Market Cap (\u20b9 Cr, log scale)",
            "ai_purity_score": "AI-Purity Score (0–1)",
            "segment": "Segment",
        },
        template="plotly_white",
        height=520,
    )
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(
        yaxis=dict(range=[0, 1.05], tickformat=".2f"),
        legend_title_text="Segment",
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Bubble size → Sharpe ratio (or 1Y return when Sharpe unavailable). "
        "Market cap on log scale. AI-purity score derived from segment taxonomy and keyword boosts."
    )


def main() -> None:
    st.set_page_config(page_title="AI x India Idea Engine", page_icon=":chart_with_upwards_trend:", layout="wide")
    st.title("AI x India Idea Engine")
    st.caption(
        "Educational prototype for Morgan Stanley-style wealth workflows. "
        "Scores use live public-market data plus deterministic factor rules and should not be treated as investment advice."
    )

    selected_archetype = st.sidebar.selectbox("Client archetype", list(ARCHETYPE_CONFIGS))
    use_cached_stats = st.sidebar.checkbox("Prefer cached stats if available", value=True)
    top_n = st.sidebar.slider("Ideas to display", min_value=5, max_value=10, value=5)

    base_config = ARCHETYPE_CONFIGS[selected_archetype]
    st.sidebar.markdown("### Override factor mix")
    growth_weight = st.sidebar.slider("Growth", 0.0, 1.0, float(base_config["weights"]["growth_score"]), 0.01)
    momentum_weight = st.sidebar.slider("Momentum", 0.0, 1.0, float(base_config["weights"]["momentum_score"]), 0.01)
    volatility_weight = st.sidebar.slider("Volatility", 0.0, 1.0, float(base_config["weights"]["volatility_score"]), 0.01)
    yield_weight = st.sidebar.slider("Yield", 0.0, 1.0, float(base_config["weights"]["yield_score"]), 0.01)
    ai_purity_weight = st.sidebar.slider("AI purity", 0.0, 1.0, float(base_config["weights"]["ai_purity_score"]), 0.01)

    if st.sidebar.button("🔄 Reset to archetype defaults"):
        st.rerun()

    if st.sidebar.button("Refresh live data"):
        load_base_dataset.clear()

    try:
        with st.spinner("Loading public market data and scoring the universe..."):
            stats = load_base_dataset(use_cached_stats=use_cached_stats)
    except Exception as exc:  # pragma: no cover - Streamlit runtime behavior
        st.error(f"Could not load the AI x India dataset: {exc}")
        st.stop()

    normalized_weights = normalize_weights(
        {
            "growth_score": growth_weight,
            "momentum_score": momentum_weight,
            "volatility_score": volatility_weight,
            "yield_score": yield_weight,
            "ai_purity_score": ai_purity_weight,
        }
    )
    custom_config = {**base_config, "weights": normalized_weights}
    ranked = score_for_archetype(stats, custom_config)
    ranked = attach_rationales(ranked.head(top_n), selected_archetype)

    # ------------------------------------------------------------------ tabs
    tab_ideas, tab_map, tab_universe = st.tabs(["Top Ideas", "Theme Map", "Full Universe"])

    with tab_ideas:
        st.subheader(selected_archetype)
        st.write(base_config["description"])

        meta_columns = st.columns(5)
        meta_columns[0].metric("Growth", base_config["growth_priority"].title())
        meta_columns[1].metric("Risk", base_config["risk_tolerance"].title())
        meta_columns[2].metric("Yield", base_config["dividend_preference"].title())
        meta_columns[3].metric("Concentration", base_config["concentration_preference"].title())
        meta_columns[4].metric("AI Purity", base_config["ai_purity_preference"].title())

        st.markdown("### Normalized factor weights")
        st.dataframe(
            pd.DataFrame(
                {
                    "Factor": [name.replace("_score", "").replace("_", " ").title() for name in normalized_weights],
                    "Weight": [f"{value:.0%}" for value in normalized_weights.values()],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        display_columns = [
            "ticker",
            "name",
            "total_score",
            "growth_score",
            "momentum_score",
            "volatility_score",
            "yield_score",
            "ai_purity_score",
            "rationale_text",
        ]
        st.markdown("### Top ideas")
        st.dataframe(archetype_output_table(ranked[display_columns]), use_container_width=True, hide_index=True)

        st.markdown("### Advisor talking points")
        for row in ranked.itertuples(index=False):
            with st.container(border=True):
                st.markdown(f"**{row.name} ({row.ticker})**")
                st.write(row.rationale_text)
                metrics = st.columns(4)
                metrics[0].metric("Total score", format_score(row.total_score))
                metrics[1].metric("3Y CAGR", format_pct(getattr(row, "cagr_3y", None)))
                metrics[2].metric("Volatility", format_pct(getattr(row, "annualized_volatility", None)))
                metrics[3].metric("Yield", format_pct(getattr(row, "dividend_yield", None)))

    with tab_map:
        st.subheader("AI-Purity vs Market Cap — Theme Map")
        st.caption(
            "Each bubble is one listed AI/ML company. "
            "X = market cap (log scale), Y = AI-purity score, colour = segment, size = Sharpe ratio."
        )
        _render_theme_map(stats)

    with tab_universe:
        st.subheader("Full Scored Universe")
        all_scored = score_for_archetype(stats, custom_config)
        universe_cols = [
            c for c in [
                "ticker", "name", "segment",
                "total_score", "ai_purity_score", "growth_score",
                "momentum_score", "volatility_score", "yield_score",
                "return_1y", "cagr_3y", "annualized_volatility", "dividend_yield",
            ] if c in all_scored.columns
        ]
        st.dataframe(all_scored[universe_cols], use_container_width=True, hide_index=True)

    st.caption(
        "Data comes from yfinance/Yahoo Finance with REST fallbacks, using a cached local stats file when available. "
        "A production deployment would require licensed market data, compliance controls, and auditable content review."
    )


if __name__ == "__main__":
    main()
