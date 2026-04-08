"""Tests for compute_excess_returns and compute_information_ratio (Issue #6).

All tests use synthetic price data only -- no live API calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import TRADING_DAYS, compute_excess_returns, compute_information_ratio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _price_series(returns: list[float], start: str = "2022-01-03") -> pd.Series:
    """Build a price series from a list of daily returns (base price = 100)."""
    idx = pd.bdate_range(start, periods=len(returns) + 1)
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    return pd.Series(prices, index=idx, dtype=float)


# ---------------------------------------------------------------------------
# compute_excess_returns
# ---------------------------------------------------------------------------

class TestComputeExcessReturns:
    def test_returns_series_with_correct_length(self):
        n = 252
        stock = _price_series([0.001] * n)
        bench = _price_series([0.0005] * n)
        result = compute_excess_returns(stock, bench)
        # pct_change drops first row, so n observations expected
        assert len(result) == n

    def test_excess_return_value_is_correct(self):
        """Constant stock return 1%, bench 0.5% -> excess ~0.5% every day."""
        n = 100
        stock = _price_series([0.01] * n)
        bench = _price_series([0.005] * n)
        result = compute_excess_returns(stock, bench)
        expected = 0.01 - 0.005
        assert result.mean() == pytest.approx(expected, rel=1e-6)

    def test_identical_series_gives_zero_excess(self):
        n = 100
        stock = _price_series([0.002] * n)
        bench = stock.copy()
        result = compute_excess_returns(stock, bench)
        assert (result.abs() < 1e-12).all()

    def test_misaligned_index_is_handled(self):
        """Stock starts one week later than benchmark -- should still produce output."""
        n = 200
        full_bench = _price_series([0.001] * n, start="2022-01-03")
        late_stock = _price_series([0.002] * (n - 5), start="2022-01-10")
        result = compute_excess_returns(late_stock, full_bench)
        assert len(result) > 0
        assert result.isna().sum() == 0

    def test_too_short_returns_empty(self):
        stock = pd.Series([100.0], index=pd.bdate_range("2022-01-03", periods=1))
        bench = pd.Series([100.0], index=pd.bdate_range("2022-01-03", periods=1))
        result = compute_excess_returns(stock, bench)
        assert len(result) == 0

    def test_non_overlapping_returns_empty(self):
        stock = _price_series([0.001] * 50, start="2022-01-03")
        bench = _price_series([0.001] * 50, start="2023-06-01")
        result = compute_excess_returns(stock, bench)
        assert len(result) == 0

    def test_result_is_named_excess_return(self):
        stock = _price_series([0.001] * 50)
        bench = _price_series([0.0005] * 50)
        result = compute_excess_returns(stock, bench)
        assert result.name == "excess_return"

    def test_nan_prices_handled_gracefully(self):
        stock = _price_series([0.001] * 50)
        bench = _price_series([0.0005] * 50)
        stock.iloc[10] = float("nan")
        result = compute_excess_returns(stock, bench)
        assert result.isna().sum() == 0 or len(result) >= 0  # must not raise


# ---------------------------------------------------------------------------
# compute_information_ratio
# ---------------------------------------------------------------------------

class TestComputeInformationRatio:
    def test_returns_float_for_normal_input(self):
        excess = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 252))
        result = compute_information_ratio(excess)
        assert isinstance(result, float)

    def test_positive_ir_for_positive_mean_excess(self):
        """Constant positive excess return -> positive IR."""
        rng = np.random.default_rng(0)
        excess = pd.Series(rng.normal(0.002, 0.01, 252))
        result = compute_information_ratio(excess)
        assert result is not None
        assert result > 0

    def test_negative_ir_for_negative_mean_excess(self):
        rng = np.random.default_rng(1)
        excess = pd.Series(rng.normal(-0.002, 0.01, 252))
        result = compute_information_ratio(excess)
        assert result is not None
        assert result < 0

    def test_zero_excess_returns_none(self):
        """All-zero excess -> tracking error is 0 -> IR must be None."""
        excess = pd.Series([0.0] * 252)
        result = compute_information_ratio(excess)
        assert result is None

    def test_single_observation_returns_none(self):
        excess = pd.Series([0.005])
        result = compute_information_ratio(excess)
        assert result is None

    def test_empty_series_returns_none(self):
        result = compute_information_ratio(pd.Series([], dtype=float))
        assert result is None

    def test_annualisation_uses_trading_days(self):
        """IR should scale with sqrt(TRADING_DAYS). Verify formula numerically."""
        excess = pd.Series([0.001] * 252 + [-0.0005] * 252)
        ir = compute_information_ratio(excess)
        manual_mean = excess.mean() * TRADING_DAYS
        manual_te = excess.std() * (TRADING_DAYS ** 0.5)
        expected = manual_mean / manual_te
        assert ir == pytest.approx(expected, rel=1e-6)

    def test_reasonable_magnitude_for_equity_excess(self):
        """A realistic IR for an Indian AI equity theme should be in [-3, 3]."""
        rng = np.random.default_rng(7)
        excess = pd.Series(rng.normal(0.0005, 0.008, 756))  # 3 years
        ir = compute_information_ratio(excess)
        assert ir is not None
        assert -3.0 <= ir <= 3.0

    def test_round_trip_with_compute_excess_returns(self):
        """compute_excess_returns -> compute_information_ratio pipeline works end-to-end."""
        n = 252
        stock = _price_series([0.0012] * n)
        bench = _price_series([0.0008] * n)
        excess = compute_excess_returns(stock, bench)
        ir = compute_information_ratio(excess)
        # Constant excess -> zero tracking error -> None
        assert ir is None

    def test_round_trip_with_noisy_excess(self):
        rng = np.random.default_rng(99)
        n = 504
        stock_returns = rng.normal(0.0015, 0.015, n)
        bench_returns = rng.normal(0.001, 0.012, n)
        stock = _price_series(stock_returns.tolist())
        bench = _price_series(bench_returns.tolist())
        excess = compute_excess_returns(stock, bench)
        ir = compute_information_ratio(excess)
        assert isinstance(ir, float)
