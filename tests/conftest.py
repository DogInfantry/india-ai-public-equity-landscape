"""Shared pytest fixtures for the AI x India test suite."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

TICKERS = ["AAA.NS", "BBB.NS", "CCC.NS"]
SHORT_TICKER = "SHORT.NS"  # only 20 days — too short for Sharpe


def _make_price_series(n_days: int, start_price: float = 100.0, daily_drift: float = 0.0005, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(daily_drift, 0.015, size=n_days)
    prices = start_price * np.cumprod(1 + daily_returns)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    return pd.Series(prices, index=dates)


@pytest.fixture
def price_history_df() -> pd.DataFrame:
    """Long-format price history with 3 full-history tickers and 1 short-history ticker."""
    rows = []
    for i, ticker in enumerate(TICKERS):
        series = _make_price_series(n_days=756, seed=i)  # ~3 years
        for date, price in series.items():
            rows.append({"ticker": ticker, "date": date, "adj_close": price, "source": "test"})
    # Short ticker — only 20 days, should not produce Sharpe
    for date, price in _make_price_series(n_days=20).items():
        rows.append({"ticker": SHORT_TICKER, "date": date, "adj_close": price, "source": "test"})
    return pd.DataFrame(rows)


@pytest.fixture
def universe_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": "AAA.NS", "name": "Alpha Corp", "segment": "IT Services", "notes": "genai platform"},
        {"ticker": "BBB.NS", "name": "Beta Ltd", "segment": "ER&D", "notes": ""},
        {"ticker": "CCC.NS", "name": "Gamma Inc", "segment": "Analytics", "notes": "gpu cluster"},
        {"ticker": SHORT_TICKER, "name": "Short Co", "segment": "IT Services", "notes": ""},
    ])


@pytest.fixture
def fundamentals_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"ticker": "AAA.NS", "trailing_pe": 30.0, "price_to_book": 5.0, "market_cap": 5e11, "dividend_yield": 0.01, "ai_purity_score": 0.9},
        {"ticker": "BBB.NS", "trailing_pe": 20.0, "price_to_book": 3.0, "market_cap": 2e11, "dividend_yield": 0.02, "ai_purity_score": 0.75},
        {"ticker": "CCC.NS", "trailing_pe": 50.0, "price_to_book": 8.0, "market_cap": 1e11, "dividend_yield": 0.005, "ai_purity_score": 0.95},
        {"ticker": SHORT_TICKER, "trailing_pe": 15.0, "price_to_book": 2.0, "market_cap": 5e10, "dividend_yield": 0.03, "ai_purity_score": 0.6},
    ])


@pytest.fixture
def summary_df(universe_df, price_history_df, fundamentals_df) -> pd.DataFrame:
    """Full summary DataFrame built from synthetic data — used by reporting tests."""
    from src.analysis import summarize_price_history
    return summarize_price_history(universe_df, price_history_df, fundamentals_df)
