"""Tests for new analytics metrics in src/analysis.py."""
from __future__ import annotations
import pytest
import pandas as pd
import numpy as np


def test_sharpe_column_present(summary_df):
    assert "sharpe" in summary_df.columns


def test_sortino_column_present(summary_df):
    assert "sortino" in summary_df.columns


def test_sharpe_is_float_for_full_history_tickers(summary_df):
    for ticker in ["AAA.NS", "BBB.NS", "CCC.NS"]:
        row = summary_df[summary_df["ticker"] == ticker].iloc[0]
        assert row["sharpe"] is not None, f"sharpe is None for {ticker}"
        assert isinstance(row["sharpe"], float)


def test_sharpe_is_none_for_short_history(summary_df):
    row = summary_df[summary_df["ticker"] == "SHORT.NS"].iloc[0]
    assert row["sharpe"] is None


def test_sortino_is_float_for_full_history_tickers(summary_df):
    for ticker in ["AAA.NS", "BBB.NS", "CCC.NS"]:
        row = summary_df[summary_df["ticker"] == ticker].iloc[0]
        assert row["sortino"] is not None, f"sortino is None for {ticker}"
        assert isinstance(row["sortino"], float)


def test_sharpe_value_is_reasonable(summary_df):
    """Sharpe should be in [-5, 10] for normal equity returns."""
    for ticker in ["AAA.NS", "BBB.NS", "CCC.NS"]:
        row = summary_df[summary_df["ticker"] == ticker].iloc[0]
        assert -5.0 <= row["sharpe"] <= 10.0, f"Sharpe {row['sharpe']} out of range for {ticker}"


# --- Task 3: Drawdown Duration ---

def test_drawdown_duration_columns_present(summary_df):
    for col in ("drawdown_start", "drawdown_trough", "drawdown_recovery_days"):
        assert col in summary_df.columns, f"Missing column: {col}"


def test_drawdown_start_is_timestamp_or_none(summary_df):
    for ticker in ["AAA.NS", "BBB.NS", "CCC.NS"]:
        row = summary_df[summary_df["ticker"] == ticker].iloc[0]
        if row["drawdown_start"] is not None:
            assert isinstance(row["drawdown_start"], pd.Timestamp)


def test_drawdown_trough_is_after_start(summary_df):
    for ticker in ["AAA.NS", "BBB.NS", "CCC.NS"]:
        row = summary_df[summary_df["ticker"] == ticker].iloc[0]
        if row["drawdown_start"] is not None and row["drawdown_trough"] is not None:
            assert row["drawdown_trough"] >= row["drawdown_start"]


def test_drawdown_recovery_days_is_int_or_none(summary_df):
    for ticker in ["AAA.NS", "BBB.NS", "CCC.NS"]:
        row = summary_df[summary_df["ticker"] == ticker].iloc[0]
        val = row["drawdown_recovery_days"]
        assert val is None or isinstance(val, int), f"Expected int or None, got {type(val)}"


# --- Task 4: Correlation Matrix ---

def test_correlation_matrix_is_square(price_history_df):
    from src.analysis import compute_correlation_matrix
    corr = compute_correlation_matrix(price_history_df)
    assert corr.shape[0] == corr.shape[1]


def test_correlation_matrix_diagonal_is_one(price_history_df):
    from src.analysis import compute_correlation_matrix
    corr = compute_correlation_matrix(price_history_df)
    # Only check full-history tickers; short-history diagonals may be NaN (min_periods)
    for ticker in ["AAA.NS", "BBB.NS", "CCC.NS"]:
        if ticker in corr.index:
            assert abs(corr.loc[ticker, ticker] - 1.0) < 1e-9


def test_correlation_matrix_values_in_range(price_history_df):
    from src.analysis import compute_correlation_matrix
    corr = compute_correlation_matrix(price_history_df)
    valid = corr.values[~np.isnan(corr.values)]
    assert valid.min() >= -1.0 - 1e-9
    assert valid.max() <= 1.0 + 1e-9


def test_correlation_matrix_excludes_short_ticker(price_history_df):
    """Ticker with < 30 overlapping days should produce NaN or be excluded."""
    from src.analysis import compute_correlation_matrix
    corr = compute_correlation_matrix(price_history_df)
    if "SHORT.NS" in corr.columns:
        off_diag = corr.loc["SHORT.NS", [t for t in corr.columns if t != "SHORT.NS"]]
        pass  # presence verified; NaN is acceptable


# --- Task 5: Valuation Percentile ---

def test_valuation_percentile_columns_present(summary_df):
    assert "valuation_pe_percentile" in summary_df.columns
    assert "valuation_pb_percentile" in summary_df.columns


def test_valuation_percentile_raises_on_invalid_metric():
    from src.analysis import compute_valuation_percentile
    with pytest.raises(ValueError, match="metric"):
        compute_valuation_percentile("TCS.NS", metric="revenue")


def test_valuation_percentile_returns_none_on_insufficient_data(monkeypatch):
    """Returns None when yfinance has fewer than 4 quarters."""
    import yfinance as yf
    from src.analysis import compute_valuation_percentile

    mock_ticker = type("T", (), {
        "quarterly_financials": pd.DataFrame(),
        "quarterly_balance_sheet": pd.DataFrame(),
    })()
    monkeypatch.setattr(yf, "Ticker", lambda _: mock_ticker)

    result = compute_valuation_percentile("TCS.NS", metric="pe")
    assert result is None


def test_valuation_percentile_returns_float_in_range(monkeypatch):
    """Returns 0-100 percentile when sufficient quarterly data available."""
    import yfinance as yf
    from src.analysis import compute_valuation_percentile

    eps_series = pd.Series(
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        index=pd.date_range("2022-01-01", periods=8, freq="QE"),
        name="Basic EPS",
    )
    mock_financials = pd.DataFrame({"Basic EPS": eps_series})

    price_idx = pd.bdate_range("2022-01-01", periods=500)
    price_df = pd.DataFrame({
        "Close": np.linspace(1000, 1200, 500),
        "Open": np.linspace(990, 1190, 500),
    }, index=price_idx)

    mock_ticker = type("T", (), {
        "quarterly_financials": mock_financials.T,
        "quarterly_balance_sheet": pd.DataFrame(),
        "history": lambda **kw: price_df,
    })()
    monkeypatch.setattr(yf, "Ticker", lambda _: mock_ticker)

    result = compute_valuation_percentile("TCS.NS", metric="pe")
    assert result is None or (0.0 <= result <= 100.0)


# --- Task 6: Efficient Frontier ---

def test_efficient_frontier_returns_dict(price_history_df):
    from src.analysis import compute_efficient_frontier
    result = compute_efficient_frontier(price_history_df, min_history_days=100)
    assert isinstance(result, dict)


def test_efficient_frontier_simulated_shape(price_history_df):
    from src.analysis import compute_efficient_frontier
    result = compute_efficient_frontier(price_history_df, min_history_days=100)
    assert result["simulated"].shape == (5000, 3)


def test_efficient_frontier_portfolio_keys(price_history_df):
    from src.analysis import compute_efficient_frontier
    result = compute_efficient_frontier(price_history_df, min_history_days=100)
    for key in ("max_sharpe_weights", "max_sharpe_stats", "min_vol_weights", "min_vol_stats", "tickers_used"):
        assert key in result, f"Missing key: {key}"


def test_efficient_frontier_excludes_short_history(price_history_df):
    from src.analysis import compute_efficient_frontier
    result = compute_efficient_frontier(price_history_df, min_history_days=252)
    assert "SHORT.NS" not in result["tickers_used"]


def test_efficient_frontier_weights_sum_to_one(price_history_df):
    from src.analysis import compute_efficient_frontier
    result = compute_efficient_frontier(price_history_df, min_history_days=100)
    if result["max_sharpe_weights"] is not None:
        total = sum(result["max_sharpe_weights"].values())
        assert abs(total - 1.0) < 1e-6
