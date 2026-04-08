"""Shared analytics pipeline for the AI x India work samples."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
import yfinance as yf

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


def compute_excess_returns(
    stock_prices: pd.Series,
    benchmark_prices: pd.Series,
) -> pd.Series:
    """Compute daily excess returns of a stock over a benchmark.

    Both inputs are price series indexed by date.  The function aligns them on
    their shared dates before computing returns so callers do not need to
    pre-align.

    Parameters
    ----------
    stock_prices:
        Price series for the stock (any numeric dtype, date index).
    benchmark_prices:
        Price series for the benchmark (e.g. NIFTY IT, NIFTY 500).

    Returns
    -------
    pd.Series
        Daily excess return series (stock daily return minus benchmark daily
        return) aligned on the intersection of both date indices.  Returns an
        empty Series when fewer than two shared dates exist.
    """
    stock = pd.to_numeric(stock_prices, errors="coerce").dropna()
    bench = pd.to_numeric(benchmark_prices, errors="coerce").dropna()

    shared_index = stock.index.intersection(bench.index).sort_values()
    if len(shared_index) < 2:
        return pd.Series(dtype="float64")

    stock_aligned = stock.loc[shared_index]
    bench_aligned = bench.loc[shared_index]

    stock_returns = stock_aligned.pct_change().dropna()
    bench_returns = bench_aligned.pct_change().dropna()

    common = stock_returns.index.intersection(bench_returns.index)
    return (stock_returns.loc[common] - bench_returns.loc[common]).rename("excess_return")


def compute_information_ratio(excess_returns: pd.Series) -> float | None:
    """Compute annualised Information Ratio from a daily excess-return series.

    IR = annualised mean excess return / annualised tracking error.

    Uses TRADING_DAYS (252) for annualisation.

    Parameters
    ----------
    excess_returns:
        Daily excess return series produced by :func:`compute_excess_returns`
        or any equivalent source.

    Returns
    -------
    float or None
        Annualised IR, or ``None`` when fewer than two observations are
        available or tracking error is zero.
    """
    clean = pd.to_numeric(excess_returns, errors="coerce").dropna()
    if len(clean) < 2:
        return None

    ann_mean = float(clean.mean() * TRADING_DAYS)
    tracking_error = float(clean.std() * np.sqrt(TRADING_DAYS))

    if tracking_error == 0.0 or np.isclose(tracking_error, 0.0):
        return None

    return ann_mean / tracking_error


def compute_correlation_matrix(price_history: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation of daily returns.

    Input:  Long-format DataFrame with columns ticker, date, adj_close.
    Output: Square DataFrame (tickers x tickers), values in [-1, 1].
            Tickers with < 30 overlapping return days produce NaN cells.
    """
    wide = (
        price_history
        .assign(date=lambda df: pd.to_datetime(df["date"]))
        .pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
        .sort_index()
    )
    daily_returns = wide.pct_change().dropna(how="all")
    return daily_returns.corr(method="pearson", min_periods=30)


def compute_valuation_percentile(
    ticker: str,
    metric: str,
    lookback_years: int = 3,
) -> float | None:
    """
    Returns percentile (0-100) of the current trailing multiple within its own history.
    Returns None if fewer than 4 quarters of data are available.
    Caches raw quarterly data to data/valuation_cache/ to avoid rate limits.

    metric must be 'pe' or 'pb'. Raises ValueError otherwise.
    """
    if metric not in ("pe", "pb"):
        raise ValueError(f"metric must be 'pe' or 'pb', got {metric!r}")

    cache_dir = Path(__file__).resolve().parents[1] / "data" / "valuation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker}_{metric}.csv"

    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            try:
                cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if len(cached) >= 4:
                    current = float(cached.iloc[-1, 0])
                    historical = cached.iloc[:, 0].dropna()
                    return float((historical < current).sum() / len(historical) * 100)
            except Exception:
                pass

    for attempt, delay in enumerate([0, 1, 2, 4]):
        try:
            if delay:
                time.sleep(delay)
            t = yf.Ticker(ticker)
            if metric == "pe":
                quarterly = t.quarterly_financials
                if quarterly is None or quarterly.empty:
                    return None
                fin_t = quarterly.T
                if "Basic EPS" not in fin_t.columns:
                    return None
                eps = fin_t["Basic EPS"].dropna()
                if len(eps) < 4:
                    return None
                prices = t.history(
                    start=(pd.Timestamp.today() - pd.Timedelta(days=lookback_years * 365 + 30)).strftime("%Y-%m-%d"),
                    interval="1mo",
                )
                if prices.empty:
                    return None
                quarterly_price = prices["Close"].resample("QE").last().dropna()
                aligned = pd.concat([quarterly_price.rename("price"), eps.rename("eps")], axis=1).dropna()
                if len(aligned) < 4:
                    return None
                pe_series = aligned["price"] / aligned["eps"]
                pe_series = pe_series.replace([np.inf, -np.inf], np.nan).dropna()
                if len(pe_series) < 4:
                    return None
                pe_series.to_frame("pe").to_csv(cache_file)
                current_pe = float(pe_series.iloc[-1])
                return float((pe_series < current_pe).sum() / len(pe_series) * 100)
            else:  # pb
                bs = t.quarterly_balance_sheet
                if bs is None or bs.empty:
                    return None
                bs_t = bs.T
                bvps_col = next((c for c in bs_t.columns if "book value" in c.lower() or "stockholders" in c.lower()), None)
                if bvps_col is None:
                    return None
                bvps = bs_t[bvps_col].dropna()
                if len(bvps) < 4:
                    return None
                prices = t.history(
                    start=(pd.Timestamp.today() - pd.Timedelta(days=lookback_years * 365 + 30)).strftime("%Y-%m-%d"),
                    interval="1mo",
                )
                if prices.empty:
                    return None
                quarterly_price = prices["Close"].resample("QE").last().dropna()
                info = t.info or {}
                shares = info.get("sharesOutstanding")
                if not shares:
                    return None
                bvps_per_share = bvps / shares
                aligned = pd.concat([quarterly_price.rename("price"), bvps_per_share.rename("bvps")], axis=1).dropna()
                if len(aligned) < 4:
                    return None
                pb_series = aligned["price"] / aligned["bvps"]
                pb_series = pb_series.replace([np.inf, -np.inf], np.nan).dropna()
                if len(pb_series) < 4:
                    return None
                pb_series.to_frame("pb").to_csv(cache_file)
                current_pb = float(pb_series.iloc[-1])
                return float((pb_series < current_pb).sum() / len(pb_series) * 100)
        except Exception:
            if attempt == 3:
                return None
            continue
    return None


def compute_efficient_frontier(
    price_history: pd.DataFrame,
    min_history_days: int = 252,
) -> dict:
    """
    Compute efficient frontier via Monte Carlo scatter + scipy.optimize exact portfolios.

    Returns dict with keys:
      simulated       - ndarray (5000, 3): [volatility, return, sharpe]
      max_sharpe_weights, max_sharpe_stats
      min_vol_weights, min_vol_stats
      tickers_used
    """
    wide = (
        price_history
        .assign(date=lambda df: pd.to_datetime(df["date"]))
        .pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last")
        .sort_index()
    )
    history_counts = wide.count()
    eligible = history_counts[history_counts >= min_history_days].index.tolist()
    wide = wide[eligible].dropna()

    daily_returns = wide.pct_change().dropna()
    mean_returns = daily_returns.mean().values * TRADING_DAYS
    cov_matrix = daily_returns.cov().values * TRADING_DAYS

    n = len(eligible)
    rng = np.random.default_rng(42)

    weights_mc = rng.dirichlet(np.ones(n), size=5000)
    port_returns = weights_mc @ mean_returns
    port_vols = np.sqrt(np.einsum("ij,jk,ik->i", weights_mc, cov_matrix, weights_mc))
    port_sharpes = (port_returns - RISK_FREE_ANNUAL) / np.where(port_vols > 0, port_vols, np.nan)
    simulated = np.column_stack([port_vols, port_returns, port_sharpes])

    def _portfolio_stats(w: np.ndarray) -> tuple[float, float, float]:
        ret = float(w @ mean_returns)
        vol = float(np.sqrt(w @ cov_matrix @ w))
        sharpe = (ret - RISK_FREE_ANNUAL) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    def _weights_to_dict(w: np.ndarray) -> dict[str, float]:
        w = np.where(w < 0.02, 0.0, w)
        total = w.sum()
        if total == 0:
            return {}
        w = w / total
        return {ticker: float(w[i]) for i, ticker in enumerate(eligible) if w[i] > 0}

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    max_sharpe_weights = None
    max_sharpe_stats = None
    try:
        def _neg_sharpe(w):
            ret, vol, _ = _portfolio_stats(w)
            return -(ret - RISK_FREE_ANNUAL) / vol if vol > 0 else 0.0

        res = minimize(_neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            max_sharpe_weights = _weights_to_dict(res.x)
            w_norm = np.array([max_sharpe_weights.get(t, 0.0) for t in eligible])
            ret, vol, sharpe = _portfolio_stats(w_norm)
            max_sharpe_stats = {"return": ret, "volatility": vol, "sharpe": sharpe}
    except Exception:
        pass

    min_vol_weights = None
    min_vol_stats = None
    try:
        def _portfolio_variance(w):
            return float(w @ cov_matrix @ w)

        res = minimize(_portfolio_variance, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        if res.success:
            min_vol_weights = _weights_to_dict(res.x)
            w_norm = np.array([min_vol_weights.get(t, 0.0) for t in eligible])
            ret, vol, sharpe = _portfolio_stats(w_norm)
            min_vol_stats = {"return": ret, "volatility": vol, "sharpe": sharpe}
    except Exception:
        pass

    return {
        "simulated": simulated,
        "max_sharpe_weights": max_sharpe_weights,
        "max_sharpe_stats": max_sharpe_stats,
        "min_vol_weights": min_vol_weights,
        "min_vol_stats": min_vol_stats,
        "tickers_used": eligible,
    }


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
        # --- Drawdown duration ---
        _dd_start = None
        _dd_trough = None
        _dd_recovery_days = None
        if not closes.empty and len(closes) >= 2:
            _running_peak = closes.cummax()
            _dd_series = closes / _running_peak - 1.0
            _trough_date = _dd_series.idxmin()
            _peak_price = float(_running_peak.loc[_trough_date])
            _pre_trough = closes.loc[closes.index <= _trough_date]
            _peak_matches = _pre_trough[_pre_trough >= _peak_price]
            if not _peak_matches.empty:
                _dd_start = _peak_matches.index[-1]
            _dd_trough = _trough_date
            _post_trough = closes.loc[closes.index > _trough_date]
            _recovered = _post_trough[_post_trough >= _peak_price]
            if not _recovered.empty:
                _recovery_date = _recovered.index[0]
                _dd_recovery_days = len(closes.loc[_trough_date:_recovery_date]) - 1
        metrics["drawdown_start"] = _dd_start
        metrics["drawdown_trough"] = _dd_trough
        metrics["drawdown_recovery_days"] = _dd_recovery_days
        # --- Valuation percentile ---
        metrics["valuation_pe_percentile"] = compute_valuation_percentile(row.ticker, metric="pe")
        metrics["valuation_pb_percentile"] = compute_valuation_percentile(row.ticker, metric="pb")
        metrics.update(fundamentals_by_ticker.get(row.ticker, {}))
        results.append(metrics)

    df = pd.DataFrame(results)
    _numeric_cols = [
        "last_close", "return_1y", "return_3y_total", "cagr_3y",
        "momentum_6m", "momentum_12m", "annualized_volatility", "max_drawdown",
        "history_years", "valuation_pe_percentile", "valuation_pb_percentile",
    ]
    for _col in _numeric_cols:
        if _col in df.columns:
            df[_col] = pd.to_numeric(df[_col], errors="coerce")
    for _col in ("sharpe", "sortino"):
        if _col in df.columns:
            df[_col] = df[_col].astype(object).where(df[_col].notna(), other=None)
    if "drawdown_recovery_days" in df.columns:
        df["drawdown_recovery_days"] = pd.Series(
            [None if pd.isna(v) else int(v) for v in df["drawdown_recovery_days"]],
            dtype=object,
            index=df.index,
        )
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
