"""Shared analytics pipeline for the AI x India work samples."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import download_price_history, get_basic_fundamentals, load_universe

TRADING_DAYS = 252
RISK_FREE_ANNUAL = 0.065  # India 10Y G-Sec approximate
DEFAULT_BENCHMARK_TICKER = "^NSEI"


def trailing_return(series: pd.Series, lookback_days: int) -> float | None:
    """Compute a trailing total return using the nearest observation before the cutoff."""

    clean = series.dropna().sort_index()
    if clean.empty:
        return None
    last_date = clean.index.max()
    last_price = float(clean.iloc[-1])
    cutoff = last_date - pd.Timedelta(days=lookback_days)
    base = clean.loc[clean.index <= cutoff]
    if base.empty:
        return None
    first_price = float(base.iloc[-1])
    if first_price == 0:
        return None
    return (last_price / first_price) - 1.0


def trailing_cagr(series: pd.Series, lookback_years: int) -> float | None:
    """Compute CAGR for a trailing multi-year window when enough history exists."""

    total_return = trailing_return(series, lookback_days=int(365.25 * lookback_years))
    if total_return is None:
        return None
    return (1.0 + total_return) ** (1.0 / lookback_years) - 1.0


def annualized_volatility(series: pd.Series) -> float | None:
    """Annualized volatility using daily close-to-close returns."""

    returns = series.dropna().pct_change().dropna()
    if len(returns) < 30:
        return None
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def max_drawdown(series: pd.Series) -> float | None:
    """Max drawdown across the full sample."""

    clean = series.dropna()
    if clean.empty:
        return None
    running_peak = clean.cummax()
    drawdowns = clean / running_peak - 1.0
    return float(drawdowns.min())


def summarize_price_history(
    universe: pd.DataFrame,
    price_history: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """Combine metadata, trailing performance, and fundamentals into one table."""

    results: list[dict[str, object]] = []
    fundamentals_by_ticker = fundamentals.set_index("ticker").to_dict(orient="index")

    for row in universe.itertuples(index=False):
        history = price_history.loc[price_history["ticker"] == row.ticker].copy()
        history["date"] = pd.to_datetime(history["date"])
        history = history.sort_values("date")
        closes = history.set_index("date")["adj_close"].dropna()
        history_years = None
        if not closes.empty:
            history_years = (closes.index.max() - closes.index.min()).days / 365.25

        metrics = {
            "ticker": row.ticker,
            "name": row.name,
            "segment": row.segment,
            "notes": row.notes,
            "history_years": history_years,
            "last_close": float(closes.iloc[-1]) if not closes.empty else None,
            "return_1y": trailing_return(closes, 365),
            "return_3y_total": trailing_return(closes, int(365.25 * 3)),
            "cagr_3y": trailing_cagr(closes, 3),
            "momentum_6m": trailing_return(closes, 182),
            "momentum_12m": trailing_return(closes, 365),
            "annualized_volatility": annualized_volatility(closes),
            "max_drawdown": max_drawdown(closes),
            "data_source": ", ".join(sorted(history["source"].dropna().unique())) if not history.empty else None,
        }
        # --- Sharpe and Sortino ---
        _sharpe = None
        _sortino = None
        if not closes.empty:
            _daily = closes.pct_change().dropna()
            if len(_daily) >= 30:
                _ann_ret = float(_daily.mean() * TRADING_DAYS)
                _rf_daily = RISK_FREE_ANNUAL / TRADING_DAYS
                _excess = _daily - _rf_daily
                _vol = float(_daily.std() * np.sqrt(TRADING_DAYS))
                if _vol > 0:
                    _sharpe = float((_excess.mean() * TRADING_DAYS) / _vol)
                _downside = np.minimum(_daily.values, 0.0)
                _dd_dev = float(np.sqrt(np.mean(_downside ** 2)) * np.sqrt(TRADING_DAYS))
                if _dd_dev > 0:
                    _sortino = float((_ann_ret - RISK_FREE_ANNUAL) / _dd_dev)
        metrics["sharpe"] = _sharpe
        metrics["sortino"] = _sortino
        metrics.update(fundamentals_by_ticker.get(row.ticker, {}))
        results.append(metrics)

    df = pd.DataFrame(results)
    # Preserve Python None (not np.nan) for sharpe/sortino so callers can do `is None` checks.
    for _col in ("sharpe", "sortino"):
        if _col in df.columns:
            df[_col] = df[_col].astype(object).where(df[_col].notna(), other=None)
    return df


def fetch_fundamentals(universe: pd.DataFrame) -> pd.DataFrame:
    """Fetch fundamentals for every company in the curated universe."""

    rows = [get_basic_fundamentals(ticker) for ticker in universe["ticker"]]
    return pd.DataFrame(rows)


def build_ai_india_dataset(
    universe_path: str = "data/ai_india_universe.csv",
    start: str | pd.Timestamp = "2021-01-01",
    end: str | pd.Timestamp | None = None,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the full analysis data set used by the notebooks and Streamlit app."""

    end_date = pd.Timestamp.today().normalize() if end is None else pd.Timestamp(end)
    universe = load_universe(universe_path)
    price_history = download_price_history(universe["ticker"], start=start, end=end_date)
    benchmark_history = download_price_history([benchmark_ticker], start=start, end=end_date)
    fundamentals = fetch_fundamentals(universe)
    stats = summarize_price_history(universe=universe, price_history=price_history, fundamentals=fundamentals)
    stats["as_of_date"] = end_date.date().isoformat()
    return universe, price_history, benchmark_history, stats


def indexed_price_frame(
    price_history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
    tickers: Iterable[str],
    label_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build an indexed-price comparison frame starting at 100."""

    label_map = label_map or {}
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        subset = price_history.loc[price_history["ticker"] == ticker, ["date", "adj_close"]].dropna()
        if subset.empty:
            continue
        indexed = subset.copy()
        indexed["indexed_price"] = indexed["adj_close"] / indexed["adj_close"].iloc[0] * 100.0
        indexed["series"] = label_map.get(ticker, ticker)
        frames.append(indexed[["date", "series", "indexed_price"]])

    benchmark = benchmark_history.loc[:, ["date", "ticker", "adj_close"]].dropna()
    if not benchmark.empty:
        benchmark["indexed_price"] = benchmark["adj_close"] / benchmark["adj_close"].iloc[0] * 100.0
        benchmark["series"] = label_map.get(benchmark["ticker"].iloc[0], benchmark["ticker"].iloc[0])
        frames.append(benchmark[["date", "series", "indexed_price"]])

    if not frames:
        return pd.DataFrame(columns=["date", "series", "indexed_price"])
    return pd.concat(frames, ignore_index=True)


def make_indexed_performance_chart(indexed_frame: pd.DataFrame, title: str) -> go.Figure:
    figure = px.line(
        indexed_frame,
        x="date",
        y="indexed_price",
        color="series",
        title=title,
        labels={"indexed_price": "Indexed Price (Start = 100)", "date": "Date", "series": "Series"},
    )
    figure.update_layout(template="plotly_white", legend_title_text="")
    return figure


def make_return_bar_chart(stats: pd.DataFrame, title: str, return_column: str = "return_1y") -> go.Figure:
    chart_data = stats.sort_values(return_column, ascending=False).copy()
    chart_data["label"] = chart_data["name"]
    figure = px.bar(
        chart_data,
        x="label",
        y=return_column,
        color="segment",
        title=title,
        labels={return_column: return_column.replace("_", " ").title(), "label": "Company"},
    )
    figure.update_layout(template="plotly_white", xaxis_tickangle=-45, showlegend=True)
    figure.update_yaxes(tickformat=".0%")
    return figure


def make_valuation_scatter(
    stats: pd.DataFrame,
    valuation_column: str = "trailing_pe",
    return_column: str = "cagr_3y",
    title: str = "Valuation vs. 3Y CAGR",
    x_label: str | None = None,
    y_label: str | None = None,
) -> go.Figure:
    chart_data = stats.dropna(subset=[valuation_column, return_column]).copy()
    default_labels = {
        "trailing_pe": "Trailing P/E",
        "price_to_book": "Price / Book",
        "ai_purity_score": "AI Purity Score",
        "cagr_3y": "3Y CAGR",
        "return_1y": "1Y Return",
        "annualized_volatility": "Annualized Volatility",
    }
    figure = px.scatter(
        chart_data,
        x=valuation_column,
        y=return_column,
        color="segment",
        hover_name="name",
        size="market_cap",
        title=title,
        labels={
            valuation_column: x_label or default_labels.get(valuation_column, valuation_column.replace("_", " ").title()),
            return_column: y_label or default_labels.get(return_column, return_column.replace("_", " ").title()),
            "market_cap": "Market Cap",
        },
    )
    figure.update_layout(template="plotly_white")
    figure.update_yaxes(tickformat=".0%")
    return figure


def build_hypothetical_ideas(stats: pd.DataFrame) -> list[dict[str, str]]:
    """Create plausible but explicitly hypothetical M&A and partnership ideas."""

    by_ticker = stats.set_index("ticker")
    templates = [
        (
            "NETWEB.NS",
            "Hypothetical partnership: a global hyperscaler or sovereign cloud JV partner teaming with Netweb to deepen India-resident AI compute capacity.",
            "Netweb offers listed exposure to domestic HPC and AI infrastructure; its smaller scale versus mega-cap IT peers could make partnerships more practical than full M&A.",
        ),
        (
            "RATEGAIN.NS",
            "Hypothetical acquisition: a global travel-tech consolidator acquiring RateGain to strengthen pricing and revenue-management analytics in Asia.",
            "RateGain sits at the intersection of vertical SaaS and AI-driven revenue optimization, a profile that often attracts strategic buyers focused on data-rich niches.",
        ),
        (
            "AFFLE.NS",
            "Hypothetical acquisition or strategic stake: an ad-tech, martech, or telecom-data platform adding Affle for mobile consumer intelligence and ML-led campaign monetization.",
            "Affle combines applied machine learning with consumer data and fraud-control tooling, making it relevant to both strategic ad-tech buyers and large digital ecosystems.",
        ),
        (
            "KPITTECH.NS",
            "Hypothetical partnership: a global automotive Tier-1 supplier or software-defined-vehicle platform expanding in India through KPIT.",
            "KPIT is already positioned around ADAS and vehicle software, so a partnership-led route could help global players localize engineering and accelerate OEM programs.",
        ),
        (
            "PERSISTENT.NS",
            "Hypothetical strategic investment: a global enterprise software vendor using Persistent to deepen India-centric AI engineering delivery and product modernization work.",
            "Persistent has strong AI and data engineering credibility, but its larger scale makes a minority partnership or capability alliance more plausible than outright acquisition.",
        ),
    ]

    ideas: list[dict[str, str]] = []
    for ticker, title, rationale in templates:
        if ticker not in by_ticker.index:
            continue
        company = by_ticker.loc[ticker]
        market_cap = company.get("market_cap")
        cagr = company.get("cagr_3y")
        suffix = []
        if pd.notna(market_cap):
            suffix.append(f"Market cap screen: roughly INR {market_cap / 1e9:,.0f}bn.")
        if pd.notna(cagr):
            suffix.append(f"Trailing 3Y CAGR: {cagr:.1%}.")
        ideas.append(
            {
                "company": company["name"],
                "ticker": ticker,
                "idea": title,
                "rationale": f"{rationale} {' '.join(suffix)}".strip(),
            }
        )
    return ideas


def cache_stats_path() -> Path:
    """Location for the optional latest stats cache."""

    return Path(__file__).resolve().parents[1] / "data" / "ai_india_stats_latest.csv"
