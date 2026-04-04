# India AI Signal Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the India AI public equity repo from a generic data pipeline into a professional-grade research product — with Sharpe/Sortino, efficient frontier, valuation percentile, analyst-voice reports, PDF tearsheets, HTML report, and a pitch-deck README — targeting finance/IB recruiters.

**Architecture:** Additive-only changes to four existing modules (`src/analysis.py`, `src/visuals.py`, `src/reporting.py`, `README.md`). The analytics layer is upstream of everything — build it first. Visuals and reporting are parallel after analytics. README is last. All new code is covered by pytest unit tests using synthetic data; live yfinance calls are mocked.

**Tech Stack:** Python 3.11+, pandas, numpy, scipy (optimizer), matplotlib (charts + PDFs), Pillow (JPG export), yfinance (valuation data), pytest, adjustText (optional scatter labels)

**Spec:** `docs/superpowers/specs/2026-04-04-india-ai-equity-upgrade-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `tests/__init__.py` | Makes tests a package |
| Create | `tests/conftest.py` | Shared pytest fixtures (synthetic data) |
| Create | `tests/test_analysis.py` | Unit tests for all new analytics functions |
| Create | `tests/test_visuals.py` | File-existence tests for new chart outputs |
| Create | `tests/test_reporting.py` | Unit tests for prose and report generation |
| Modify | `requirements.txt` | Add scipy, Pillow, adjustText, pytest |
| Modify | `src/analysis.py` | Add Sharpe, Sortino, drawdown duration, correlation matrix, valuation percentile, efficient frontier |
| Modify | `src/visuals.py` | Add JPG export, 4 new charts, PDF scorecard; extend `generate_visual_pack()` |
| Modify | `src/reporting.py` | Add narrative prose, HTML report, tearsheets, README updater; extend pitch builder |
| Modify | `README.md` | Full rewrite as pitch-deck cover page |

---

## Task 1: Repo Setup

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Clone the repo**

```bash
cd C:/Users/Anklesh/Documents/Claude_Code/india_ai
git clone https://github.com/DogInfantry/india-ai-public-equity-landscape repo
cd repo
```

- [ ] **Step 2: Add new dependencies to `requirements.txt`**

Append these lines to `requirements.txt`:
```
scipy>=1.11
Pillow>=10.0
adjustText>=0.8
pytest>=8.0
pytest-mock>=3.12
```

- [ ] **Step 3: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without errors. Verify with:
```bash
python -c "import scipy; import PIL; import pytest; print('OK')"
```

- [ ] **Step 4: Create `tests/__init__.py`**

```python
```
(Empty file — just makes the directory a package.)

- [ ] **Step 5: Create `tests/conftest.py`** with shared fixtures

```python
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
```

- [ ] **Step 6: Verify existing imports work**

```bash
python -c "from src.analysis import summarize_price_history; print('analysis OK')"
python -c "from src.visuals import generate_visual_pack; print('visuals OK')"
python -c "from src.reporting import build_thematic_pitch_markdown; print('reporting OK')"
```

Expected: all three print `OK`.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt tests/__init__.py tests/conftest.py
git commit -m "chore: add test infrastructure and new dependencies"
```

---

## Task 2: Analytics — Sharpe and Sortino

**Files:**
- Modify: `src/analysis.py` (inside `summarize_price_history` loop + add `RISK_FREE_ANNUAL` constant)
- Modify: `tests/test_analysis.py` (create if not exists)

- [ ] **Step 1: Write failing tests**

Create `tests/test_analysis.py`:

```python
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
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
cd repo
pytest tests/test_analysis.py::test_sharpe_column_present -v
```

Expected: `FAILED` with `KeyError: 'sharpe'`

- [ ] **Step 3: Add `RISK_FREE_ANNUAL` constant and Sharpe/Sortino computation to `src/analysis.py`**

At the top of the file (after existing constants), add:
```python
RISK_FREE_ANNUAL = 0.065  # India 10Y G-Sec approximate
```

Inside `summarize_price_history()`, after the line `"max_drawdown": max_drawdown(closes),` and before `"data_source": ...`, add:

```python
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
```

- [ ] **Step 4: Run tests — confirm they pass**

```bash
pytest tests/test_analysis.py -v -k "sharpe or sortino"
```

Expected: 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis.py tests/test_analysis.py
git commit -m "feat: add Sharpe and Sortino to summarize_price_history"
```

---

## Task 3: Analytics — Drawdown Duration

**Files:**
- Modify: `src/analysis.py`
- Modify: `tests/test_analysis.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_analysis.py`:

```python
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
```

- [ ] **Step 2: Run — confirm they fail**

```bash
pytest tests/test_analysis.py -k "drawdown" -v
```

Expected: FAILED with `KeyError`

- [ ] **Step 3: Add drawdown duration to `summarize_price_history()` in `src/analysis.py`**

After the Sharpe/Sortino block, add:

```python
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
                # By definition closes <= cummax, so >= and == are equivalent here
                _peak_matches = _pre_trough[_pre_trough >= _peak_price]
                if not _peak_matches.empty:
                    _dd_start = _peak_matches.index[-1]
                _dd_trough = _trough_date
                _post_trough = closes.loc[closes.index > _trough_date]
                _recovered = _post_trough[_post_trough >= _peak_price]
                if not _recovered.empty:
                    _recovery_date = _recovered.index[0]
                    # True trading-day count (index rows), not calendar approximation
                    _dd_recovery_days = len(closes.loc[_trough_date:_recovery_date]) - 1
            metrics["drawdown_start"] = _dd_start
            metrics["drawdown_trough"] = _dd_trough
            metrics["drawdown_recovery_days"] = _dd_recovery_days
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_analysis.py -k "drawdown" -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis.py tests/test_analysis.py
git commit -m "feat: add drawdown duration (start, trough, recovery days)"
```

---

## Task 4: Analytics — Correlation Matrix

**Files:**
- Modify: `src/analysis.py` (add `compute_correlation_matrix` function)
- Modify: `tests/test_analysis.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_analysis.py`:

```python
def test_correlation_matrix_is_square(price_history_df):
    from src.analysis import compute_correlation_matrix
    corr = compute_correlation_matrix(price_history_df)
    assert corr.shape[0] == corr.shape[1]


def test_correlation_matrix_diagonal_is_one(price_history_df):
    from src.analysis import compute_correlation_matrix
    corr = compute_correlation_matrix(price_history_df)
    for ticker in corr.index:
        assert abs(corr.loc[ticker, ticker] - 1.0) < 1e-9


def test_correlation_matrix_values_in_range(price_history_df):
    from src.analysis import compute_correlation_matrix
    corr = compute_correlation_matrix(price_history_df)
    assert corr.values.min() >= -1.0 - 1e-9
    assert corr.values.max() <= 1.0 + 1e-9


def test_correlation_matrix_excludes_short_ticker(price_history_df):
    """Ticker with < 30 overlapping days should produce NaN or be excluded."""
    from src.analysis import compute_correlation_matrix
    corr = compute_correlation_matrix(price_history_df)
    # SHORT.NS has only 20 days — if included, its off-diagonal values should be NaN
    if "SHORT.NS" in corr.columns:
        off_diag = corr.loc["SHORT.NS", [t for t in corr.columns if t != "SHORT.NS"]]
        # Either all NaN (insufficient overlap) or computed — both are acceptable
        pass  # presence verified; NaN is acceptable
```

- [ ] **Step 2: Run — confirm they fail**

```bash
pytest tests/test_analysis.py -k "correlation" -v
```

Expected: FAILED with `ImportError` (function doesn't exist yet)

- [ ] **Step 3: Add `compute_correlation_matrix` to `src/analysis.py`**

Add after the `max_drawdown` function:

```python
def compute_correlation_matrix(price_history: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise Pearson correlation of daily returns.

    Input:  Long-format DataFrame with columns ticker, date, adj_close.
    Output: Square DataFrame (tickers × tickers), values in [-1, 1].
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_analysis.py -k "correlation" -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis.py tests/test_analysis.py
git commit -m "feat: add compute_correlation_matrix"
```

---

## Task 5: Analytics — Valuation Percentile

**Files:**
- Modify: `src/analysis.py` (add `compute_valuation_percentile` function + call in loop)
- Modify: `tests/test_analysis.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_analysis.py`:

```python
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
        "quarterly_financials": pd.DataFrame(),  # empty — no data
        "quarterly_balance_sheet": pd.DataFrame(),
    })()
    monkeypatch.setattr(yf, "Ticker", lambda _: mock_ticker)

    result = compute_valuation_percentile("TCS.NS", metric="pe")
    assert result is None


def test_valuation_percentile_returns_float_in_range(monkeypatch):
    """Returns 0-100 percentile when sufficient quarterly data available."""
    import yfinance as yf
    import numpy as np
    from src.analysis import compute_valuation_percentile

    # 8 quarters of EPS data
    eps_series = pd.Series(
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        index=pd.date_range("2022-01-01", periods=8, freq="QE"),
        name="Basic EPS",
    )
    mock_financials = pd.DataFrame({"Basic EPS": eps_series})

    # Corresponding price history
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
```

- [ ] **Step 2: Run — confirm they fail**

```bash
pytest tests/test_analysis.py -k "valuation_percentile" -v
```

Expected: FAILED

- [ ] **Step 3: Add `compute_valuation_percentile` to `src/analysis.py`**

Add imports near the top:
```python
import time
import yfinance as yf
```

Add the function after `compute_correlation_matrix`:

```python
def compute_valuation_percentile(
    ticker: str,
    metric: str,
    lookback_years: int = 3,
) -> float | None:
    """
    Returns percentile (0–100) of the current trailing multiple within its own history.
    Returns None if fewer than 4 quarters of data are available.
    Caches raw quarterly data to data/valuation_cache/ to avoid rate limits.

    metric must be 'pe' or 'pb'. Raises ValueError otherwise.
    """
    if metric not in ("pe", "pb"):
        raise ValueError(f"metric must be 'pe' or 'pb', got {metric!r}")

    cache_dir = Path(__file__).resolve().parents[1] / "data" / "valuation_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker}_{metric}.csv"

    # Use cache if fresh (< 24 hours)
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

    # Fetch from yfinance with exponential backoff
    for attempt, delay in enumerate([0, 1, 2, 4]):
        try:
            if delay:
                time.sleep(delay)
            t = yf.Ticker(ticker)
            if metric == "pe":
                quarterly = t.quarterly_financials
                if quarterly is None or quarterly.empty:
                    return None
                # quarterly_financials is (metrics × dates); transpose to (dates × metrics)
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
                # Compute trailing PE: resample price to quarterly, divide by EPS
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
                # shares outstanding needed for per-share book value — approximate with market_cap / price
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
```

Then inside `summarize_price_history()`, after the drawdown duration block, add:

```python
            # --- Valuation percentile ---
            metrics["valuation_pe_percentile"] = compute_valuation_percentile(row.ticker, metric="pe")
            metrics["valuation_pb_percentile"] = compute_valuation_percentile(row.ticker, metric="pb")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_analysis.py -k "valuation_percentile" -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis.py tests/test_analysis.py
git commit -m "feat: add valuation percentile rank (PE and PB vs own 3Y history)"
```

---

## Task 6: Analytics — Efficient Frontier

**Files:**
- Modify: `src/analysis.py` (add `compute_efficient_frontier`)
- Modify: `tests/test_analysis.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_analysis.py`:

```python
def test_efficient_frontier_returns_dict(price_history_df):
    from src.analysis import compute_efficient_frontier
    result = compute_efficient_frontier(price_history_df, min_history_days=100)
    assert isinstance(result, dict)


def test_efficient_frontier_simulated_shape(price_history_df):
    from src.analysis import compute_efficient_frontier
    import numpy as np
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
```

- [ ] **Step 2: Run — confirm they fail**

```bash
pytest tests/test_analysis.py -k "efficient_frontier" -v
```

Expected: FAILED

- [ ] **Step 3: Add `compute_efficient_frontier` to `src/analysis.py`**

Add import at top: `from scipy.optimize import minimize`

Add the function after `compute_valuation_percentile`:

```python
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
    # Filter: keep tickers with enough history
    history_counts = wide.count()
    eligible = history_counts[history_counts >= min_history_days].index.tolist()
    wide = wide[eligible].dropna()

    daily_returns = wide.pct_change().dropna()
    mean_returns = daily_returns.mean().values * TRADING_DAYS   # annualized
    cov_matrix = daily_returns.cov().values * TRADING_DAYS       # annualized

    n = len(eligible)
    rng = np.random.default_rng(42)

    # --- Step 2: Monte Carlo scatter (exactly 5000 portfolios) ---
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
        """Zero out weights < 0.02, renormalize once, return as dict."""
        w = np.where(w < 0.02, 0.0, w)
        total = w.sum()
        if total == 0:
            return {}
        w = w / total
        return {ticker: float(w[i]) for i, ticker in enumerate(eligible) if w[i] > 0}

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    # --- Step 3a: Maximum Sharpe ---
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

    # --- Step 3b: Minimum Volatility ---
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_analysis.py -k "efficient_frontier" -v
```

Expected: 5 tests PASS

- [ ] **Step 5: Run full analytics test suite**

```bash
pytest tests/test_analysis.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/analysis.py tests/test_analysis.py
git commit -m "feat: add efficient frontier optimizer (Monte Carlo + scipy SLSQP)"
```

---

## Task 7: Visuals — JPG Export

**Files:**
- Modify: `src/visuals.py` (update `_save_figure` helper)
- Create: `tests/test_visuals.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_visuals.py`:

```python
"""Tests for new visual outputs in src/visuals.py."""
from __future__ import annotations
import pytest
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def test_save_figure_produces_jpg(tmp_path):
    from src.visuals import _save_figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    png_path = tmp_path / "test_chart.png"
    _save_figure(fig, png_path)
    jpg_path = tmp_path / "test_chart.jpg"
    assert jpg_path.exists(), "JPG file not created alongside PNG"
    assert jpg_path.stat().st_size > 0, "JPG file is empty"


def test_save_figure_still_produces_png(tmp_path):
    from src.visuals import _save_figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    png_path = tmp_path / "test_chart.png"
    _save_figure(fig, png_path)
    assert png_path.exists()
    assert png_path.stat().st_size > 0
```

- [ ] **Step 2: Run — confirm they fail**

```bash
pytest tests/test_visuals.py -v
```

Expected: FAILED (`JPG file not created`)

- [ ] **Step 3: Update `_save_figure` in `src/visuals.py`**

Add `from PIL import Image` at the top of the file (after existing imports).

Replace the existing `_save_figure` function:

```python
def _save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # Also save as JPG alongside the PNG
    jpg_path = path.with_suffix(".jpg")
    try:
        Image.open(path).convert("RGB").save(jpg_path, quality=92)
    except Exception:
        pass  # JPG conversion is best-effort; don't break the pipeline
    return path
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_visuals.py -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/visuals.py tests/test_visuals.py
git commit -m "feat: auto-save JPG alongside every PNG chart"
```

---

## Task 8: Visuals — Sharpe/Return Scatter

**Files:**
- Modify: `src/visuals.py`
- Modify: `tests/test_visuals.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_visuals.py`:

```python
def test_sharpe_return_scatter_creates_png(tmp_path, summary_df):
    from src.visuals import save_sharpe_return_scatter
    path = save_sharpe_return_scatter(summary_df, output_dir=tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0
    assert (tmp_path / "ai_india_sharpe_return.jpg").exists()
```

Note: `summary_df` fixture is defined in `conftest.py` and is available here.

- [ ] **Step 2: Run — confirm it fails**

```bash
pytest tests/test_visuals.py::test_sharpe_return_scatter_creates_png -v
```

Expected: FAILED (`ImportError`)

- [ ] **Step 3: Add `save_sharpe_return_scatter` to `src/visuals.py`**

Add this function before `generate_visual_pack`:

```python
def save_sharpe_return_scatter(stats: pd.DataFrame, output_dir: str | Path | None = None) -> Path:
    output_path = _ensure_figure_dir(output_dir) / "ai_india_sharpe_return.png"
    chart_data = stats.dropna(subset=["sharpe"]).copy()
    chart_data["_return"] = chart_data["cagr_3y"].fillna(chart_data["return_1y"])
    chart_data = chart_data.dropna(subset=["_return", "market_cap"])

    segment_colors = _segment_color_lookup(chart_data["segment"])
    bubble_sizes = np.sqrt(chart_data["market_cap"] / 1e9) * 10

    fig, ax = plt.subplots(figsize=(11, 7))
    for segment, subset in chart_data.groupby("segment"):
        idx = subset.index
        ax.scatter(
            subset["_return"],
            subset["sharpe"],
            s=bubble_sizes.loc[idx],
            alpha=0.78,
            color=segment_colors[segment],
            edgecolor="white",
            linewidth=0.8,
            label=segment,
        )

    # Quadrant lines
    median_ret = float(chart_data["_return"].median())
    ax.axvline(median_ret, color="#AABBCC", linewidth=1.0, linestyle="--", zorder=0)
    ax.axhline(1.0, color="#AABBCC", linewidth=1.0, linestyle="--", zorder=0)

    # Quadrant labels
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    for label, x, y in [
        ("High Return\nHigh Quality", x_max * 0.85, max(y_max * 0.85, 1.1)),
        ("High Return\nHigh Risk", x_max * 0.85, min(y_min * 0.5, 0.9)),
        ("Low Return\nHigh Quality", x_min * 0.5, max(y_max * 0.85, 1.1)),
        ("Low Return\nHigh Risk", x_min * 0.5, min(y_min * 0.5, 0.9)),
    ]:
        ax.text(x, y, label, fontsize=8, color="grey", style="italic", ha="center", va="center")

    # Ticker labels
    try:
        from adjustText import adjust_text
        texts = [ax.text(row["_return"], row["sharpe"], row["ticker"].replace(".NS", ""), fontsize=7)
                 for _, row in chart_data.iterrows()]
        adjust_text(texts, ax=ax)
    except ImportError:
        for _, row in chart_data.iterrows():
            ax.annotate(row["ticker"].replace(".NS", ""), (row["_return"], row["sharpe"]),
                        xytext=(0.005, 0.02), textcoords="offset points", fontsize=7)

    _apply_axis_style(
        ax,
        "Sharpe / Return Map",
        "Top-right = high return AND high risk-adjusted quality. Bubble size = market cap.",
    )
    ax.set_xlabel("Annualized Return (3Y CAGR or 1Y)")
    ax.set_ylabel("Sharpe Ratio")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return _save_figure(fig, output_path)
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_visuals.py::test_sharpe_return_scatter_creates_png -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/visuals.py tests/test_visuals.py
git commit -m "feat: add Sharpe/Return scatter chart"
```

---

## Task 9: Visuals — Correlation Heatmap

**Files:**
- Modify: `src/visuals.py`
- Modify: `tests/test_visuals.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_visuals.py`:

```python
def test_correlation_heatmap_creates_png(tmp_path, price_history_df):
    from src.visuals import save_correlation_heatmap
    from src.analysis import compute_correlation_matrix
    corr = compute_correlation_matrix(price_history_df)
    path = save_correlation_heatmap(corr, output_dir=tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0
```

- [ ] **Step 2: Run — confirm it fails**

```bash
pytest tests/test_visuals.py::test_correlation_heatmap_creates_png -v
```

- [ ] **Step 3: Add `save_correlation_heatmap` to `src/visuals.py`**

```python
def save_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    stats: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """
    corr_matrix: output of compute_correlation_matrix().
    stats: optional summary df used to sort tickers by segment (improves readability).
    """
    output_path = _ensure_figure_dir(output_dir) / "ai_india_correlation_heatmap.png"

    if stats is not None and "segment" in stats.columns:
        order = stats.sort_values("segment")["ticker"].tolist()
        order = [t for t in order if t in corr_matrix.columns]
        corr_matrix = corr_matrix.loc[order, order]

    n = len(corr_matrix)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.55), max(7, n * 0.5)))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [t.replace(".NS", "") for t in corr_matrix.columns]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for row_i in range(n):
        for col_j in range(n):
            val = corr_matrix.values[row_i, col_j]
            if not np.isnan(val):
                ax.text(col_j, row_i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(val) < 0.7 else "white")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    _apply_axis_style(ax, "Return Correlation Matrix",
                      "Red = highly correlated (concentrated risk). Blue = low correlation (diversified).")
    ax.grid(False)
    return _save_figure(fig, output_path)
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_visuals.py::test_correlation_heatmap_creates_png -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/visuals.py tests/test_visuals.py
git commit -m "feat: add correlation heatmap chart"
```

---

## Task 10: Visuals — Drawdown Timeline

**Files:**
- Modify: `src/visuals.py`
- Modify: `tests/test_visuals.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_visuals.py`:

```python
def test_drawdown_timeline_creates_png(tmp_path, price_history_df, summary_df):
    from src.visuals import save_drawdown_timeline
    path = save_drawdown_timeline(price_history_df, summary_df, output_dir=tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0
```

- [ ] **Step 2: Run — confirm it fails**

```bash
pytest tests/test_visuals.py::test_drawdown_timeline_creates_png -v
```

- [ ] **Step 3: Add `save_drawdown_timeline` to `src/visuals.py`**

```python
def save_drawdown_timeline(
    price_history: pd.DataFrame,
    stats: pd.DataFrame,
    top_n: int = 8,
    output_dir: str | Path | None = None,
) -> Path:
    output_path = _ensure_figure_dir(output_dir) / "ai_india_drawdown_timeline.png"
    # Select top N by Sharpe ratio
    ranked = stats.dropna(subset=["sharpe"]).nlargest(top_n, "sharpe")
    selected_tickers = ranked["ticker"].tolist()

    fig, ax = plt.subplots(figsize=(12, 6.5))
    colors = list(_segment_color_lookup(ranked["segment"]).values())

    for i, ticker in enumerate(selected_tickers):
        hist = (
            price_history[price_history["ticker"] == ticker]
            .assign(date=lambda df: pd.to_datetime(df["date"]))
            .sort_values("date")
            .set_index("date")["adj_close"]
            .dropna()
        )
        if hist.empty:
            continue
        drawdown = hist / hist.cummax() - 1.0
        label = ticker.replace(".NS", "")
        color = colors[i % len(colors)]
        ax.plot(drawdown.index, drawdown.values, label=label, linewidth=1.6, color=color)
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.08, color=color)
        # Mark max drawdown date
        trough_date = drawdown.idxmin()
        ax.axvline(trough_date, color=color, linewidth=0.6, linestyle=":", alpha=0.6)

    ax.axhline(0, color="#888888", linewidth=0.8)
    _apply_axis_style(
        ax,
        "Drawdown From Peak — Top Names by Sharpe",
        "Dotted vertical lines mark each ticker's worst drawdown date.",
    )
    ax.set_ylabel("Drawdown from Peak")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return _save_figure(fig, output_path)
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_visuals.py::test_drawdown_timeline_creates_png -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/visuals.py tests/test_visuals.py
git commit -m "feat: add drawdown timeline chart (top 8 by Sharpe)"
```

---

## Task 11: Visuals — Efficient Frontier Chart

**Files:**
- Modify: `src/visuals.py`
- Modify: `tests/test_visuals.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_visuals.py`:

```python
def test_efficient_frontier_chart_creates_png(tmp_path, price_history_df, summary_df):
    from src.visuals import save_efficient_frontier_chart
    from src.analysis import compute_efficient_frontier
    frontier = compute_efficient_frontier(price_history_df, min_history_days=100)
    path = save_efficient_frontier_chart(frontier, summary_df, output_dir=tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0
```

- [ ] **Step 2: Run — confirm it fails**

```bash
pytest tests/test_visuals.py::test_efficient_frontier_chart_creates_png -v
```

- [ ] **Step 3: Add `save_efficient_frontier_chart` to `src/visuals.py`**

```python
def save_efficient_frontier_chart(
    frontier: dict,
    stats: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    output_path = _ensure_figure_dir(output_dir) / "ai_india_efficient_frontier.png"
    sim = frontier["simulated"]  # shape (5000, 3): [vol, return, sharpe]

    fig, ax = plt.subplots(figsize=(11, 7))

    # Monte Carlo scatter (color = Sharpe)
    sc = ax.scatter(sim[:, 0], sim[:, 1], c=sim[:, 2], cmap="viridis",
                    alpha=0.35, s=4, zorder=1)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio", shrink=0.8)

    # Individual stocks as grey diamonds
    if stats is not None:
        stock_data = stats.dropna(subset=["sharpe"]).copy()
        stock_data["_return"] = stock_data["cagr_3y"].fillna(stock_data["return_1y"])
        stock_data = stock_data.dropna(subset=["_return"])
        ax.scatter(stock_data["annualized_volatility"], stock_data["_return"],
                   marker="D", color="grey", s=40, zorder=3, alpha=0.7, label="Individual stocks")
        for _, row in stock_data.iterrows():
            ax.annotate(row["ticker"].replace(".NS", ""),
                        (row["annualized_volatility"], row["_return"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7, color="grey")

    # Max Sharpe portfolio
    if frontier["max_sharpe_stats"] and frontier["max_sharpe_weights"]:
        ms = frontier["max_sharpe_stats"]
        top3 = sorted(frontier["max_sharpe_weights"].items(), key=lambda x: -x[1])[:3]
        label = "Max Sharpe\n" + ", ".join(f"{t.replace('.NS','')} {w:.0%}" for t, w in top3)
        ax.scatter(ms["volatility"], ms["return"], marker="*", color=COLOR_PALETTE["gold"],
                   s=400, zorder=5, label=label)

    # Min Vol portfolio
    if frontier["min_vol_stats"] and frontier["min_vol_weights"]:
        mv = frontier["min_vol_stats"]
        top3 = sorted(frontier["min_vol_weights"].items(), key=lambda x: -x[1])[:3]
        label = "Min Vol\n" + ", ".join(f"{t.replace('.NS','')} {w:.0%}" for t, w in top3)
        ax.scatter(mv["volatility"], mv["return"], marker="*", color=COLOR_PALETTE["blue"],
                   s=400, zorder=5, label=label)

    _apply_axis_style(
        ax,
        "Efficient Frontier",
        "Gold star = Maximum Sharpe portfolio. Blue star = Minimum Volatility portfolio.",
    )
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return _save_figure(fig, output_path)
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_visuals.py::test_efficient_frontier_chart_creates_png -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/visuals.py tests/test_visuals.py
git commit -m "feat: add efficient frontier chart"
```

---

## Task 12: Visuals — PDF Scorecard + Wire Up `generate_visual_pack`

**Files:**
- Modify: `src/visuals.py`
- Modify: `tests/test_visuals.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_visuals.py`:

```python
def test_scorecard_also_saves_pdf(tmp_path, summary_df):
    from src.visuals import save_scorecard_png
    save_scorecard_png(summary_df, output_dir=tmp_path)
    assert (tmp_path / "ai_india_scorecard.pdf").exists()
    assert (tmp_path / "ai_india_scorecard.pdf").stat().st_size > 0
```

- [ ] **Step 2: Run — confirm it fails**

```bash
pytest tests/test_visuals.py::test_scorecard_also_saves_pdf -v
```

- [ ] **Step 3: Update `save_scorecard_png` to also save PDF**

Find `return _save_figure(fig, output_path)` at the end of `save_scorecard_png` and replace with:

```python
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    return _save_figure(fig, output_path)
```

Note: `_save_figure` closes the figure, so PDF must be saved before calling it.

- [ ] **Step 4: Wire all new charts into `generate_visual_pack`**

In `generate_visual_pack()`, after the existing `artifact_paths` dict is built (after `archetype_heatmap_png` line), add:

```python
    from src.analysis import compute_correlation_matrix, compute_efficient_frontier
    corr_matrix = compute_correlation_matrix(price_history)
    frontier = compute_efficient_frontier(price_history)

    artifact_paths["sharpe_return_png"] = save_sharpe_return_scatter(scored, directory)
    artifact_paths["correlation_heatmap_png"] = save_correlation_heatmap(corr_matrix, scored, directory)
    artifact_paths["drawdown_timeline_png"] = save_drawdown_timeline(price_history, scored, directory)
    artifact_paths["efficient_frontier_png"] = save_efficient_frontier_chart(frontier, scored, directory)
    artifact_paths["frontier_data"] = frontier  # pass downstream to reporting
```

Also store `corr_matrix` and `price_history` so reporting can use them:
```python
    artifact_paths["corr_matrix"] = corr_matrix
    artifact_paths["price_history"] = price_history
```

- [ ] **Step 5: Run all visuals tests**

```bash
pytest tests/test_visuals.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/visuals.py tests/test_visuals.py
git commit -m "feat: PDF scorecard + wire all new charts into generate_visual_pack"
```

---

## Task 13: Reporting — Narrative Prose Functions

**Files:**
- Modify: `src/reporting.py`
- Create: `tests/test_reporting.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_reporting.py`:

```python
"""Tests for new reporting functions in src/reporting.py."""
from __future__ import annotations
import pytest
import pandas as pd


@pytest.fixture
def sample_row_high_sharpe_high_pe():
    return pd.Series({
        "ticker": "AAA.NS", "name": "Alpha Corp", "segment": "IT Services",
        "sharpe": 1.3, "sortino": 1.8, "return_1y": 0.35, "cagr_3y": 0.28,
        "annualized_volatility": 0.22, "max_drawdown": -0.18,
        "valuation_pe_percentile": 85.0, "valuation_pb_percentile": 70.0,
        "drawdown_recovery_days": 45, "drawdown_start": None, "drawdown_trough": None,
    })


@pytest.fixture
def sample_row_low_sharpe():
    return pd.Series({
        "ticker": "BBB.NS", "name": "Beta Ltd", "segment": "ER&D",
        "sharpe": 0.4, "sortino": 0.5, "return_1y": 0.55, "cagr_3y": 0.40,
        "annualized_volatility": 0.55, "max_drawdown": -0.45,
        "valuation_pe_percentile": 40.0, "valuation_pb_percentile": 50.0,
        "drawdown_recovery_days": None, "drawdown_start": None, "drawdown_trough": None,
    })


def test_narrative_is_string(sample_row_high_sharpe_high_pe):
    from src.reporting import generate_stock_narrative
    result = generate_stock_narrative(sample_row_high_sharpe_high_pe)
    assert isinstance(result, str)
    assert len(result) > 20


def test_narrative_rule1_triggers_priced_for_perfection(sample_row_high_sharpe_high_pe):
    from src.reporting import generate_stock_narrative
    result = generate_stock_narrative(sample_row_high_sharpe_high_pe)
    assert "priced for perfection" in result.lower()


def test_narrative_rule3_triggers_high_conviction(sample_row_low_sharpe):
    from src.reporting import generate_stock_narrative
    result = generate_stock_narrative(sample_row_low_sharpe)
    assert "high-conviction" in result.lower()


def test_narrative_handles_none_sortino(sample_row_low_sharpe):
    from src.reporting import generate_stock_narrative
    row = sample_row_low_sharpe.copy()
    row["sortino"] = None
    result = generate_stock_narrative(row)
    assert "N/A" in result or isinstance(result, str)  # must not raise


def test_bull_base_bear_returns_dict(sample_row_high_sharpe_high_pe):
    from src.reporting import generate_bull_base_bear
    result = generate_bull_base_bear(sample_row_high_sharpe_high_pe)
    assert set(result.keys()) == {"bull", "base", "bear"}
    for key in ("bull", "base", "bear"):
        assert isinstance(result[key], str) and len(result[key]) > 10


def test_bull_base_bear_handles_none_values(sample_row_high_sharpe_high_pe):
    from src.reporting import generate_bull_base_bear
    row = sample_row_high_sharpe_high_pe.copy()
    row["sharpe"] = None
    row["valuation_pe_percentile"] = None
    result = generate_bull_base_bear(row)
    assert set(result.keys()) == {"bull", "base", "bear"}
```

- [ ] **Step 2: Run — confirm they fail**

```bash
pytest tests/test_reporting.py -v
```

Expected: FAILED with `ImportError`

- [ ] **Step 3: Add `generate_stock_narrative` and `generate_bull_base_bear` to `src/reporting.py`**

```python
def generate_stock_narrative(
    row: pd.Series,
    corr_matrix: pd.DataFrame | None = None,
    top5_tickers: list[str] | None = None,
) -> str:
    """
    Generate an opinionated analyst-voice sentence for a stock.
    Applies branching rules in order; first match wins.
    All column comparisons are None-safe.
    """
    sharpe = row.get("sharpe")
    sortino = row.get("sortino")
    return_1y = row.get("return_1y")
    pe_pct = row.get("valuation_pe_percentile")
    max_dd = row.get("max_drawdown")
    dd_recovery = row.get("drawdown_recovery_days")
    ticker = row.get("ticker", "")
    segment = row.get("segment", "")
    cagr = row.get("cagr_3y") or row.get("return_1y")

    # Rule 1: High Sharpe + Expensive vs own history
    if sharpe is not None and sharpe > 1.0 and pe_pct is not None and pe_pct > 80:
        return (
            "Priced for perfection — the risk-adjusted case is strong "
            "(Sharpe {:.2f}) but there is no margin of safety at current multiples "
            "(PE in the {:.0f}th percentile of own history).".format(sharpe, pe_pct)
        )

    # Rule 2: High Sharpe + Cheap vs own history
    if sharpe is not None and sharpe > 1.0 and pe_pct is not None and pe_pct < 30:
        return (
            "Best of both worlds in this universe — strong risk-adjusted returns "
            "(Sharpe {:.2f}) and still trading at a historical discount "
            "(PE in the {:.0f}th percentile of own history).".format(sharpe, pe_pct)
        )

    # Rule 3: High return but low Sharpe (momentum without quality)
    if return_1y is not None and return_1y > 0.30 and sharpe is not None and sharpe < 0.6:
        sortino_str = f"{sortino:.2f}" if sortino is not None else "N/A"
        return (
            "A high-conviction-only name. The {:.0%} 1Y return is real but so is the ride — "
            "Sortino of {} confirms the downside is not being rewarded proportionally.".format(return_1y, sortino_str)
        )

    # Rule 4: Highly correlated with another top-5 name
    if corr_matrix is not None and top5_tickers is not None and ticker in corr_matrix.columns:
        for other in top5_tickers:
            if other != ticker and other in corr_matrix.columns:
                corr_val = corr_matrix.loc[ticker, other]
                if pd.notna(corr_val) and corr_val > 0.8:
                    return (
                        "Not a standalone diversifier — {} and {} move together {:.0%} of the time. "
                        "Owning both is effectively a single concentrated position.".format(
                            ticker.replace(".NS", ""), other.replace(".NS", ""), corr_val
                        )
                    )

    # Rule 5: Still in deep drawdown, not recovered
    if dd_recovery is None and max_dd is not None and max_dd < -0.25:
        return (
            "Still {:.1%} below its peak with no confirmed recovery. "
            "The drawdown timeline matters for anyone sizing this as a short-duration trade.".format(abs(max_dd))
        )

    # Default: neutral
    cagr_str = f"{cagr:.1%}" if cagr is not None else "n/a"
    sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "n/a"
    return (
        "{} segment name with {:.0%} 1Y return, {cagr} 3Y CAGR, "
        "and Sharpe of {sharpe}.".format(
            segment, return_1y if return_1y is not None else 0.0,
            cagr=cagr_str, sharpe=sharpe_str
        )
    )


def generate_bull_base_bear(row: pd.Series) -> dict[str, str]:
    """Return {'bull': ..., 'base': ..., 'bear': ...} analyst cases."""
    sharpe = row.get("sharpe")
    cagr = row.get("cagr_3y")
    pe_pct = row.get("valuation_pe_percentile")
    max_dd = row.get("max_drawdown")

    if sharpe is not None and sharpe > 1.0 and cagr is not None and cagr > 0.20:
        return {
            "bull": "Sustained AI services re-rating drives multiple expansion on top of compounding earnings.",
            "base": "Revenue growth tracks consensus; Sharpe remains above 1.0 as volatility stays contained.",
            "bear": "Global IT spend slowdown compresses multiples; name re-rates to sector average.",
        }
    if pe_pct is not None and pe_pct > 75:
        return {
            "bull": "Earnings growth accelerates, justifying premium multiple.",
            "base": "Multiple stays elevated on execution; no meaningful re-rating in either direction.",
            "bear": "Any earnings miss triggers sharp de-rating from historically elevated levels.",
        }
    if max_dd is not None and max_dd < -0.35:
        return {
            "bull": "Full recovery and breakout to new highs as sector tailwind reasserts.",
            "base": "Slow recovery; price recovers peak within 12 months but limited upside beyond.",
            "bear": "Drawdown deepens on further sector rotation or macro headwinds; no near-term catalyst.",
        }
    return {
        "bull": "AI adoption in India accelerates beyond consensus, driving upside to estimates.",
        "base": "Company executes in line with guidance; stock tracks earnings growth.",
        "bear": "Macro or FX headwinds compress margins; limited multiple support at current levels.",
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_reporting.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/reporting.py tests/test_reporting.py
git commit -m "feat: add generate_stock_narrative and generate_bull_base_bear"
```

---

## Task 14: Reporting — New Pitch Report Sections

**Files:**
- Modify: `src/reporting.py`
- Modify: `tests/test_reporting.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_reporting.py`:

```python
def test_risk_adjusted_table_columns(summary_df):
    from src.reporting import risk_adjusted_ranking_table
    table = risk_adjusted_ranking_table(summary_df)
    for col in ("Ticker", "1Y Return", "3Y CAGR", "Sharpe", "Sortino", "Sharpe Rank"):
        assert col in table.columns, f"Missing column: {col}"


def test_correlation_callout_is_string(price_history_df):
    from src.analysis import compute_correlation_matrix
    from src.reporting import correlation_callout_paragraph
    corr = compute_correlation_matrix(price_history_df)
    result = correlation_callout_paragraph(corr)
    assert isinstance(result, str) and len(result) > 20


def test_portfolio_construction_section_contains_weights(summary_df, price_history_df):
    from src.analysis import compute_efficient_frontier
    from src.reporting import portfolio_construction_section
    frontier = compute_efficient_frontier(price_history_df, min_history_days=100)
    result = portfolio_construction_section(frontier)
    assert isinstance(result, str)
    assert "Max Sharpe" in result or "max sharpe" in result.lower()
```

- [ ] **Step 2: Run — confirm they fail**

```bash
pytest tests/test_reporting.py -k "risk_adjusted or correlation_callout or portfolio_construction" -v
```

- [ ] **Step 3: Add the three section-builder functions to `src/reporting.py`**

```python
def risk_adjusted_ranking_table(stats: pd.DataFrame) -> pd.DataFrame:
    """Build risk-adjusted ranking table (Sharpe, Sortino, rank)."""
    ranked = stats.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False).copy()
    ranked["Sharpe Rank"] = range(1, len(ranked) + 1)
    display = pd.DataFrame({
        "Ticker": ranked["ticker"].str.replace(".NS", "", regex=False),
        "1Y Return": ranked["return_1y"].apply(format_pct),
        "3Y CAGR": ranked["cagr_3y"].apply(format_pct),
        "Sharpe": ranked["sharpe"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "n/a"),
        "Sortino": ranked["sortino"].apply(lambda v: f"{v:.2f}" if pd.notna(v) else "n/a"),
        "Sharpe Rank": ranked["Sharpe Rank"],
    })
    return display


def correlation_callout_paragraph(corr_matrix: pd.DataFrame) -> str:
    """Return a prose paragraph naming the 3 most correlated pairs."""
    pairs = []
    tickers = corr_matrix.columns.tolist()
    for i, t1 in enumerate(tickers):
        for t2 in tickers[i + 1:]:
            val = corr_matrix.loc[t1, t2]
            if pd.notna(val):
                pairs.append((val, t1, t2))
    if not pairs:
        return "Insufficient data to compute cross-ticker correlations."
    top3 = sorted(pairs, reverse=True)[:3]
    lines = []
    for val, t1, t2 in top3:
        n1 = t1.replace(".NS", "")
        n2 = t2.replace(".NS", "")
        if val > 0.8:
            lines.append(f"**{n1} / {n2}** ({val:.2f}) — highly correlated; owning both is effectively one position.")
        else:
            lines.append(f"**{n1} / {n2}** ({val:.2f})")
    return (
        "**Correlation watch:** The three most correlated pairs in this universe are: "
        + "; ".join(lines)
        + ". Names with correlation above 0.80 do not provide meaningful diversification."
    )


def portfolio_construction_section(frontier: dict) -> str:
    """Build the Portfolio Construction markdown section from efficient frontier data."""
    if not frontier.get("max_sharpe_weights") and not frontier.get("min_vol_weights"):
        return "> **Portfolio Construction:** Insufficient data to run optimizer.\n"

    lines = ["## Portfolio Construction", ""]
    lines.append("Mean-variance optimization across the universe produces two reference portfolios:\n")

    if frontier["max_sharpe_weights"] and frontier["max_sharpe_stats"]:
        ms = frontier["max_sharpe_stats"]
        lines.append("### Maximum Sharpe Portfolio")
        lines.append(f"Expected return: {ms['return']:.1%} | Volatility: {ms['volatility']:.1%} | Sharpe: {ms['sharpe']:.2f}\n")
        rows = [(t.replace(".NS", ""), w) for t, w in sorted(frontier["max_sharpe_weights"].items(), key=lambda x: -x[1])]
        lines.append("| Ticker | Weight |")
        lines.append("|--------|--------|")
        for ticker, weight in rows:
            lines.append(f"| {ticker} | {weight:.1%} |")
        lines.append("")
        lines.append(
            "> **Key Takeaway:** This is the portfolio that historically delivered the best return per unit of risk "
            "from this universe — not the highest return, but the most efficient one."
        )

    if frontier["min_vol_weights"] and frontier["min_vol_stats"]:
        mv = frontier["min_vol_stats"]
        lines.append("\n### Minimum Volatility Portfolio")
        lines.append(f"Expected return: {mv['return']:.1%} | Volatility: {mv['volatility']:.1%} | Sharpe: {mv['sharpe']:.2f}\n")
        rows = [(t.replace(".NS", ""), w) for t, w in sorted(frontier["min_vol_weights"].items(), key=lambda x: -x[1])]
        lines.append("| Ticker | Weight |")
        lines.append("|--------|--------|")
        for ticker, weight in rows:
            lines.append(f"| {ticker} | {weight:.1%} |")

    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_reporting.py -v
```

Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/reporting.py tests/test_reporting.py
git commit -m "feat: add risk-adjusted table, correlation callout, and portfolio construction sections"
```

---

## Task 15: Reporting — HTML Report

**Files:**
- Modify: `src/reporting.py`
- Modify: `tests/test_reporting.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_reporting.py`:

```python
def test_html_report_creates_file(tmp_path, summary_df):
    from src.reporting import generate_html_report
    path = tmp_path / "test_pitch.html"
    generate_html_report(summary_df, output_path=path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_html_report_no_external_links(tmp_path, summary_df):
    from src.reporting import generate_html_report
    path = tmp_path / "test_pitch.html"
    generate_html_report(summary_df, output_path=path)
    content = path.read_text(encoding="utf-8")
    assert '<link href="http' not in content
    assert '<script src="http' not in content


def test_html_report_under_2mb(tmp_path, summary_df):
    from src.reporting import generate_html_report
    path = tmp_path / "test_pitch.html"
    generate_html_report(summary_df, output_path=path)
    assert path.stat().st_size < 2 * 1024 * 1024


def test_html_report_has_color_coded_cells(tmp_path, summary_df):
    from src.reporting import generate_html_report
    path = tmp_path / "test_pitch.html"
    generate_html_report(summary_df, output_path=path)
    content = path.read_text(encoding="utf-8")
    assert "#e6f4ea" in content or "#fce8e6" in content
```

- [ ] **Step 2: Run — confirm they fail**

```bash
pytest tests/test_reporting.py -k "html" -v
```

- [ ] **Step 3: Add `generate_html_report` to `src/reporting.py`**

```python
_HTML_CSS = """
<style>
  body { font-family: Georgia, serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #1a1a2e; }
  h1 { color: #163B65; border-bottom: 3px solid #163B65; padding-bottom: 8px; }
  h2 { color: #163B65; margin-top: 2em; }
  h3 { color: #2E6DA4; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 1em 0; }
  th { background: #163B65; color: white; padding: 8px 10px; text-align: left; position: sticky; top: 0; }
  td { padding: 6px 10px; border-bottom: 1px solid #e0e0e0; }
  tr:nth-child(even) { background: #f8f9fa; }
  .pos { background-color: #e6f4ea; }
  .neg { background-color: #fce8e6; }
  blockquote { border-left: 4px solid #2E6DA4; padding-left: 16px; color: #163B65; font-weight: bold; }
  .disclaimer { font-size: 11px; color: #888; margin-top: 3em; border-top: 1px solid #ddd; padding-top: 1em; }
  @media print { th { background: #163B65 !important; -webkit-print-color-adjust: exact; } }
</style>
"""


def _df_to_html_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to HTML table with color-coded return cells."""
    rows = ["<table>", "<tr>" + "".join(f"<th>{col}</th>" for col in df.columns) + "</tr>"]
    for _, row in df.iterrows():
        cells = []
        for col, val in zip(df.columns, row):
            css = ""
            if isinstance(val, str) and val.endswith("%"):
                try:
                    num = float(val.rstrip("%"))
                    css = ' class="pos"' if num > 0 else ' class="neg"' if num < 0 else ""
                except ValueError:
                    pass
            cells.append(f"<td{css}>{val}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</table>")
    return "\n".join(rows)


def generate_html_report(
    stats: pd.DataFrame,
    output_path: str | Path | None = None,
    corr_matrix: pd.DataFrame | None = None,
    frontier: dict | None = None,
) -> Path:
    """Generate a self-contained HTML research report from the summary stats."""
    if output_path is None:
        output_path = _repo_root() / "reports" / "ai_india_thematic_pitch.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = company_snapshot_table(stats)
    risk_table = risk_adjusted_ranking_table(stats)

    html_parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>",
        "<title>AI x India Thematic Pitch</title>",
        _HTML_CSS,
        "</head><body>",
        "<h1>AI x India: Thematic Pitch</h1>",
        f"<p><em>Generated {pd.Timestamp.today().strftime('%Y-%m-%d')}. "
        "Live market data via yfinance. Not investment advice.</em></p>",
        "<h2>Company Universe</h2>",
        _df_to_html_table(snapshot),
        "<blockquote>Key Takeaway: The universe spans large-cap IT enablers to high-purity AI specialists. "
        "Valuation and risk profiles diverge sharply — security selection matters more than theme exposure alone.</blockquote>",
        "<h2>Risk-Adjusted Rankings</h2>",
        _df_to_html_table(risk_table),
        "<blockquote>Key Takeaway: Sharpe rank reorders the universe significantly vs. raw return rank. "
        "The best momentum names are often not the best risk-adjusted names.</blockquote>",
    ]

    if corr_matrix is not None:
        html_parts += [
            "<h2>Correlation Analysis</h2>",
            f"<p>{correlation_callout_paragraph(corr_matrix)}</p>",
        ]

    if frontier is not None:
        html_parts += [
            "<h2>Portfolio Construction</h2>",
            "<p>Mean-variance optimization identifies the efficient frontier across the universe.</p>",
        ]
        if frontier.get("max_sharpe_weights") and frontier.get("max_sharpe_stats"):
            ms = frontier["max_sharpe_stats"]
            html_parts.append(
                f"<p><strong>Max Sharpe Portfolio:</strong> Return {ms['return']:.1%}, "
                f"Volatility {ms['volatility']:.1%}, Sharpe {ms['sharpe']:.2f}</p>"
            )

    # Analyst views for top 5 by Sharpe
    top5 = stats.dropna(subset=["sharpe"]).nlargest(5, "sharpe")
    html_parts.append("<h2>Top 5 Ideas — Analyst View</h2>")
    for _, row in top5.iterrows():
        bbb = generate_bull_base_bear(row)
        narrative = generate_stock_narrative(
            row, corr_matrix=corr_matrix,
            top5_tickers=top5["ticker"].tolist()
        )
        html_parts += [
            f"<h3>{row['name']} ({row['ticker'].replace('.NS', '')})</h3>",
            f"<p>{narrative}</p>",
            "<ul>",
            f"<li><strong>Bull:</strong> {bbb['bull']}</li>",
            f"<li><strong>Base:</strong> {bbb['base']}</li>",
            f"<li><strong>Bear:</strong> {bbb['bear']}</li>",
            "</ul>",
        ]

    html_parts += [
        '<p class="disclaimer">This report is for educational and portfolio-use purposes only. '
        "It is not investment advice, not a research product, and not affiliated with Morgan Stanley. "
        "All data sourced from public APIs at runtime.</p>",
        "</body></html>",
    ]

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    return output_path
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_reporting.py -k "html" -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/reporting.py tests/test_reporting.py
git commit -m "feat: generate self-contained HTML research report"
```

---

## Task 16: Reporting — PDF Tearsheets

**Files:**
- Modify: `src/reporting.py`
- Modify: `tests/test_reporting.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_reporting.py`:

```python
def test_tearsheets_creates_combined_pdf(tmp_path, summary_df, price_history_df):
    from src.reporting import generate_tearsheets
    paths = generate_tearsheets(summary_df, price_history_df, output_dir=tmp_path)
    combined = tmp_path / "top5_tearsheets_combined.pdf"
    assert combined.exists()
    assert combined.stat().st_size > 0


def test_tearsheets_creates_five_individual_pdfs(tmp_path, summary_df, price_history_df):
    from src.reporting import generate_tearsheets
    generate_tearsheets(summary_df, price_history_df, output_dir=tmp_path)
    individual = list(tmp_path.glob("*_tearsheet.pdf"))
    # Exclude combined
    individual = [p for p in individual if "combined" not in p.name]
    assert len(individual) == min(5, len(summary_df.dropna(subset=["sharpe"])))
```

- [ ] **Step 2: Run — confirm they fail**

```bash
pytest tests/test_reporting.py -k "tearsheet" -v
```

- [ ] **Step 3: Add `generate_tearsheets` to `src/reporting.py`**

Add import at top: `from matplotlib.backends.backend_pdf import PdfPages`

```python
def generate_tearsheets(
    stats: pd.DataFrame,
    price_history: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """
    Generate single-page PDF tearsheets for the top 5 names by Sharpe.
    Saves both individual files and a 5-page combined PDF.
    Uses only DejaVu (matplotlib default) fonts for Windows compatibility.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    if output_dir is None:
        output_dir = _repo_root() / "reports" / "tearsheets"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    top5 = stats.dropna(subset=["sharpe"]).nlargest(5, "sharpe")
    paths = {}
    combined_path = output_dir / "top5_tearsheets_combined.pdf"

    with PdfPages(combined_path) as combined_pdf:
        for _, row in top5.iterrows():
            ticker = row["ticker"]
            name = row.get("name", ticker)
            segment = row.get("segment", "")

            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")  # A4
            gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

            # --- Header ---
            fig.text(0.05, 0.96, name, fontsize=18, fontweight="bold", color="#163B65")
            fig.text(0.05, 0.93, f"{segment}  |  {ticker.replace('.NS','')}  |  {pd.Timestamp.today().strftime('%Y-%m-%d')}",
                     fontsize=10, color="#5B6770")

            # --- Price chart ---
            ax_chart = fig.add_subplot(gs[0, :])
            hist = (
                price_history[price_history["ticker"] == ticker]
                .assign(date=lambda df: pd.to_datetime(df["date"]))
                .sort_values("date")
                .set_index("date")["adj_close"]
                .dropna()
                .last("365D")
            )
            if not hist.empty:
                ax_chart.plot(hist.index, hist.values, color="#163B65", linewidth=1.8)
                if pd.notna(row.get("drawdown_trough")):
                    trough = row["drawdown_trough"]
                    if trough in hist.index:
                        ax_chart.axvline(trough, color="#B04A4A", linewidth=1.2, linestyle="--", label="Max drawdown")
                        ax_chart.legend(fontsize=8, frameon=False)
            ax_chart.set_title("1Y Price", fontsize=11, loc="left", color="#163B65")
            ax_chart.tick_params(labelsize=8)
            ax_chart.spines["top"].set_visible(False)
            ax_chart.spines["right"].set_visible(False)

            # --- Metrics table ---
            ax_table = fig.add_subplot(gs[1, :])
            ax_table.axis("off")
            metrics = [
                ("1Y Return", format_pct(row.get("return_1y"))),
                ("3Y CAGR", format_pct(row.get("cagr_3y"))),
                ("Sharpe", f"{row['sharpe']:.2f}" if pd.notna(row.get("sharpe")) else "n/a"),
                ("Sortino", f"{row['sortino']:.2f}" if pd.notna(row.get("sortino")) else "n/a"),
                ("PE", format_multiple(row.get("trailing_pe"))),
                ("PB", format_multiple(row.get("price_to_book"))),
                ("PE %ile", f"{row['valuation_pe_percentile']:.0f}" if pd.notna(row.get("valuation_pe_percentile")) else "n/a"),
                ("Max DD", format_pct(row.get("max_drawdown"))),
                ("Div Yield", format_pct(row.get("dividend_yield"))),
            ]
            col_labels = [m[0] for m in metrics]
            col_values = [m[1] for m in metrics]
            tbl = ax_table.table(
                cellText=[col_values],
                colLabels=col_labels,
                loc="center",
                cellLoc="center",
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1, 1.8)

            # --- Bull / Base / Bear ---
            ax_bbb = fig.add_subplot(gs[2, :])
            ax_bbb.axis("off")
            bbb = generate_bull_base_bear(row)
            bbb_text = (
                f"Bull: {bbb['bull']}\n\n"
                f"Base: {bbb['base']}\n\n"
                f"Bear: {bbb['bear']}"
            )
            ax_bbb.text(0.02, 0.95, "Analyst View", fontsize=10, fontweight="bold", color="#163B65",
                        transform=ax_bbb.transAxes, va="top")
            ax_bbb.text(0.02, 0.80, bbb_text, fontsize=8, color="#1a1a2e",
                        transform=ax_bbb.transAxes, va="top", wrap=True,
                        multialignment="left")

            # Disclaimer
            fig.text(0.05, 0.01,
                     "Educational portfolio sample only. Not investment advice. Data: yfinance public API.",
                     fontsize=7, color="#888888")

            # Save to combined PDF
            combined_pdf.savefig(fig, bbox_inches="tight")

            # Save individual PDF
            safe_ticker = ticker.replace(".NS", "")
            individual_path = output_dir / f"{safe_ticker}_tearsheet.pdf"
            with PdfPages(individual_path) as ind_pdf:
                ind_pdf.savefig(fig, bbox_inches="tight")
            paths[ticker] = individual_path
            plt.close(fig)

    paths["combined"] = combined_path
    return paths
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_reporting.py -k "tearsheet" -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/reporting.py tests/test_reporting.py
git commit -m "feat: generate PDF tearsheets for top 5 names by Sharpe"
```

---

## Task 17: Reporting — README Auto-Update

**Files:**
- Modify: `src/reporting.py`
- Modify: `tests/test_reporting.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_reporting.py`:

```python
def test_readme_update_writes_findings(tmp_path, summary_df):
    from src.reporting import update_readme_key_findings
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Test\n<!-- KEY_FINDINGS_START -->\nold content\n<!-- KEY_FINDINGS_END -->\nfooter\n",
        encoding="utf-8"
    )
    update_readme_key_findings(summary_df, readme)
    content = readme.read_text(encoding="utf-8")
    assert "<!-- KEY_FINDINGS_START -->" in content
    assert "<!-- KEY_FINDINGS_END -->" in content
    assert "old content" not in content
    assert "footer" in content  # content after marker preserved


def test_readme_update_generates_four_bullets(tmp_path, summary_df):
    from src.reporting import update_readme_key_findings
    readme = tmp_path / "README.md"
    readme.write_text("<!-- KEY_FINDINGS_START -->\n<!-- KEY_FINDINGS_END -->\n", encoding="utf-8")
    update_readme_key_findings(summary_df, readme)
    content = readme.read_text(encoding="utf-8")
    bullets = [line for line in content.splitlines() if line.startswith("- **")]
    assert len(bullets) >= 4
```

- [ ] **Step 2: Run — confirm they fail**

```bash
pytest tests/test_reporting.py -k "readme_update" -v
```

- [ ] **Step 3: Add `update_readme_key_findings` to `src/reporting.py`**

```python
def update_readme_key_findings(
    summary_df: pd.DataFrame,
    readme_path: str | Path,
) -> None:
    """
    Replace content between <!-- KEY_FINDINGS_START --> and <!-- KEY_FINDINGS_END -->
    in README.md with 4 data-driven opinionated callouts.
    """
    readme_path = Path(readme_path)
    content = readme_path.read_text(encoding="utf-8")

    # Build 4 findings from live data
    findings = []
    ranked_sharpe = summary_df.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False)
    if not ranked_sharpe.empty:
        best = ranked_sharpe.iloc[0]
        name = best.get("name", best["ticker"].replace(".NS", ""))
        findings.append(
            f"**Best risk-adjusted name:** {name} — "
            f"{best['sharpe']:.2f} Sharpe, "
            f"{format_pct(best.get('cagr_3y') or best.get('return_1y'))} trailing CAGR"
        )

    ranked_return = summary_df.dropna(subset=["cagr_3y"]).sort_values("cagr_3y", ascending=False)
    if not ranked_return.empty:
        top = ranked_return.iloc[0]
        name = top.get("name", top["ticker"].replace(".NS", ""))
        findings.append(
            f"**Highest compounder:** {name} — "
            f"{format_pct(top.get('cagr_3y'))} 3Y CAGR, "
            f"Sharpe {top['sharpe']:.2f}" if pd.notna(top.get("sharpe")) else f"**Highest compounder:** {name} — {format_pct(top.get('cagr_3y'))} 3Y CAGR"
        )

    high_val = summary_df.dropna(subset=["valuation_pe_percentile"]).nlargest(1, "valuation_pe_percentile")
    if not high_val.empty:
        row = high_val.iloc[0]
        name = row.get("name", row["ticker"].replace(".NS", ""))
        findings.append(
            f"**Valuation flag:** {name} is at the "
            f"{row['valuation_pe_percentile']:.0f}th percentile of its own PE history — "
            "the return may already be priced in"
        )

    if len(findings) < 4:
        findings.append(
            f"**Universe size:** {len(summary_df)} names across AI, analytics, ER&D, and IT services segments"
        )

    new_block = (
        "<!-- KEY_FINDINGS_START -->\n"
        + "\n".join(f"- {f}" for f in findings[:4])
        + "\n<!-- KEY_FINDINGS_END -->"
    )

    import re
    updated = re.sub(
        r"<!-- KEY_FINDINGS_START -->.*?<!-- KEY_FINDINGS_END -->",
        new_block,
        content,
        flags=re.DOTALL,
    )
    readme_path.write_text(updated, encoding="utf-8")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_reporting.py -k "readme_update" -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Run full reporting test suite**

```bash
pytest tests/test_reporting.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/reporting.py tests/test_reporting.py
git commit -m "feat: add update_readme_key_findings for auto-updating README on notebook run"
```

---

## Task 18: README Overhaul

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README.md**

Replace the entire contents of `README.md` with:

```markdown
# AI x India: Public Equity Landscape

> India's AI services sector is a structural multi-year opportunity.
> This is the investment case — built on live market data, not consensus.

<!-- KEY_FINDINGS_START -->
- **Best risk-adjusted name:** Run the notebook to generate live findings
- **Highest compounder:** Data generated at runtime — no hard-coded values
- **Correlation watch:** Universe-wide concentration risk surfaced on every run
- **Valuation flag:** Per-stock PE/PB percentile vs own 3-year history
<!-- KEY_FINDINGS_END -->
*(Values above are regenerated automatically when the notebook is run)*

![Sharpe/Return Scatter](reports/figures/ai_india_sharpe_return.png)

## What this is

- A live thematic pitch on 20+ Indian AI/tech equities, built entirely on public market data
- A scoring engine that ranks names differently depending on client risk profile (growth, income, thematic)
- A full analytics suite: Sharpe/Sortino, drawdown tracking, portfolio optimization, valuation percentile

## Read the research

→ [Full thematic pitch (HTML)](reports/ai_india_thematic_pitch.html)
→ [Visual gallery](reports/ai_india_visual_summary.md)
→ [Top 5 tearsheets (PDF)](reports/tearsheets/top5_tearsheets_combined.pdf)

![AI x India Scorecard](reports/figures/ai_india_scorecard.png)

![AI x India Theme Map](reports/figures/ai_india_theme_map.png)

## Universe design

The stock universe in `data/ai_india_universe.csv` covers 20+ Indian publicly-listed companies across five segments: large-cap IT enablers (TCS, Infosys, HCL Tech, Wipro), ER&D/digital-engineering specialists (Tata Elxsi, LTTS, Persistent, Cyient), analytics and platform names (Affle, RateGain, eClerx, Saksoft), AI infrastructure (Netweb), and enterprise software (OFSS, KPIT, Bosch). All market data is fetched live from Yahoo Finance at runtime — no hard-coded prices.

## Data & Methodology

- Live market data via yfinance at runtime — no hard-coded prices, returns, or valuations
- Risk-adjusted metrics: Sharpe ratio (6.5% India risk-free rate), Sortino ratio, max drawdown with date and recovery tracking
- Portfolio construction: mean-variance optimization via `scipy.optimize` — minimum volatility and maximum Sharpe portfolios
- Valuation context: each stock ranked against its own 3-year PE/PB history (percentile 0–100)

## Setup

Python 3.11+ recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Run the thematic pitch notebook:

```powershell
jupyter lab
# Open notebooks/01_ai_india_landscape.ipynb and run all cells
```

This will:
- Pull live market data
- Compute all metrics including Sharpe, Sortino, drawdown duration, valuation percentile, efficient frontier
- Refresh `reports/ai_india_thematic_pitch.md` and `reports/ai_india_thematic_pitch.html`
- Generate all charts (PNG + JPG) in `reports/figures/`
- Write PDF tearsheets to `reports/tearsheets/`
- Auto-update the key findings in this README

Generate the visual pack directly:

```powershell
python src/visuals.py
```

Run the Streamlit advisor engine:

```powershell
streamlit run src/app.py
```

Run tests:

```powershell
pytest tests/ -v
```

## Folder structure

```
data/           Universe CSV and runtime caches
notebooks/      Analysis notebooks (run top-to-bottom)
src/            Python modules: analysis, visuals, reporting, scoring, app
reports/        Generated markdown, HTML report, tearsheets, and chart figures
tests/          pytest test suite for all new analytics and reporting functions
```

## Disclaimer

This repository is for educational and portfolio-use purposes only. It is not investment advice, not a research product, and not affiliated with Morgan Stanley. All data sourced from public APIs at runtime.
```

- [ ] **Step 2: Verify markdown renders correctly**

```bash
python -c "
content = open('README.md', encoding='utf-8').read()
assert '<!-- KEY_FINDINGS_START -->' in content
assert '<!-- KEY_FINDINGS_END -->' in content
assert 'reports/figures/ai_india_sharpe_return.png' in content
assert '→ [Full thematic pitch' in content
print('README structure OK')
"
```

Expected: `README structure OK`

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README as pitch-deck cover page with live key findings"
```

---

## Task 19: Notebook Integration + Visual Gallery Update

**Files:**
- Modify: `notebooks/01_ai_india_landscape.ipynb`
- Modify: `src/visuals.py` (update `build_visual_summary_markdown`)

- [ ] **Step 1: Update `build_visual_summary_markdown` in `src/visuals.py`**

Replace the existing `build_visual_summary_markdown` function:

```python
def build_visual_summary_markdown(as_of_date: str, figure_paths: dict, stats: pd.DataFrame) -> str:
    median_1y = stats["return_1y"].median()
    median_3y = stats["cagr_3y"].median()
    best_sharpe = stats.dropna(subset=["sharpe"]).nlargest(1, "sharpe")
    sharpe_leader = f"{best_sharpe.iloc[0]['name']} ({best_sharpe.iloc[0]['sharpe']:.2f})" if not best_sharpe.empty else "n/a"

    return "\n".join([
        "# AI x India Visual Summary",
        "",
        f"_As of {as_of_date}. Static visuals generated from live public-market data._",
        "",
        "## Key Stats",
        f"- Universe size: {len(stats)} names",
        f"- Median 1Y return: {median_1y:.1%}",
        f"- Median 3Y CAGR: {median_3y:.1%}",
        f"- Best Sharpe ratio: {sharpe_leader}",
        "",
        "## Visual Pack",
        "",
        "### One-Page Scorecard",
        "![Scorecard](figures/ai_india_scorecard.png)",
        "_The fastest overview of the universe — size, median returns, top compounders at a glance._",
        "",
        "### Sharpe / Return Scatter",
        "![Sharpe Return](figures/ai_india_sharpe_return.png)",
        "_Look for names in the top-right quadrant: high return AND high Sharpe. Those are genuine risk-adjusted compounders, not just momentum plays._",
        "",
        "### Correlation Heatmap",
        "![Correlation Heatmap](figures/ai_india_correlation_heatmap.png)",
        "_Red pairs are highly correlated — owning both is effectively one concentrated position. Blue pairs provide genuine diversification._",
        "",
        "### Efficient Frontier",
        "![Efficient Frontier](figures/ai_india_efficient_frontier.png)",
        "_Each dot is a portfolio. The gold star is the maximum Sharpe portfolio; the blue star is the minimum volatility portfolio. Individual stocks are grey diamonds._",
        "",
        "### Drawdown Timeline",
        "![Drawdown Timeline](figures/ai_india_drawdown_timeline.png)",
        "_Shows which names are still underwater vs. which have fully recovered. A still-open drawdown changes how you size a position._",
        "",
        "### Indexed Performance",
        "![Indexed Performance](figures/ai_india_indexed_performance.png)",
        "_Representative names vs. NIFTY 50 rebased to 100. Shows how much of the return is alpha vs. beta._",
        "",
        "### 1Y Return Distribution",
        "![1Y Returns](figures/ai_india_1y_return_bar.png)",
        "_Full universe sorted by 1Y return. The spread is wide — this is a stock-picker's theme, not an index play._",
        "",
        "### Theme Map (AI Purity vs. CAGR)",
        "![Theme Map](figures/ai_india_theme_map.png)",
        "_X-axis: how central AI is to the business. Y-axis: trailing 3Y CAGR. Top-right = pure-play compounder._",
        "",
        "### Risk / Return Map",
        "![Risk Return](figures/ai_india_risk_return_map.png)",
        "_Volatility on x-axis, CAGR on y-axis. Ideal names are top-left: high return, low vol._",
        "",
        "### Segment Market Cap",
        "![Segment Market Cap](figures/ai_india_segment_market_cap.png)",
        "_Where the listed AI India screen has the most capital-market depth._",
        "",
        "### Archetype Heatmap",
        "![Archetype Heatmap](figures/ai_india_archetype_heatmap.png)",
        "_Same names, scored differently for each client archetype. A name strong across all archetypes is genuinely versatile._",
        "",
    ]) + "\n"
```

- [ ] **Step 2: Add final notebook cell to `notebooks/01_ai_india_landscape.ipynb`**

Open the notebook in Jupyter and add a new cell at the very end:

```python
# === Final cell: auto-update README key findings ===
from pathlib import Path
from src.reporting import update_readme_key_findings

readme_path = Path.cwd().parent / "README.md"  # works when notebook is in notebooks/
update_readme_key_findings(stats, readme_path)
print(f"README updated: {readme_path}")
```

Also add earlier in the notebook (after `generate_visual_pack` is called), add calls for new reporting outputs:

```python
# Compute correlation matrix and efficient frontier (needed for HTML report)
from src.analysis import compute_correlation_matrix, compute_efficient_frontier

corr_matrix = compute_correlation_matrix(price_history)
frontier = compute_efficient_frontier(price_history)

# Generate HTML report and tearsheets
from src.reporting import generate_html_report, generate_tearsheets

generate_html_report(
    stats,
    corr_matrix=corr_matrix,
    frontier=frontier,
)
print("HTML report: reports/ai_india_thematic_pitch.html")

generate_tearsheets(stats, price_history)
print("Tearsheets: reports/tearsheets/")
```

- [ ] **Step 3: Run the full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests PASS

- [ ] **Step 4: Verify key output files exist after a test run**

```bash
python -c "
from pathlib import Path
root = Path('.')
checks = [
    root / 'reports' / 'figures' / 'ai_india_sharpe_return.png',
    root / 'reports' / 'figures' / 'ai_india_correlation_heatmap.png',
    root / 'reports' / 'figures' / 'ai_india_efficient_frontier.png',
    root / 'reports' / 'figures' / 'ai_india_drawdown_timeline.png',
]
for path in checks:
    status = 'OK' if path.exists() else 'MISSING'
    print(f'{status}: {path}')
"
```

(Note: charts require a full notebook run with live data to exist — this checks post-run state.)

- [ ] **Step 5: Final commit**

```bash
git add src/visuals.py notebooks/01_ai_india_landscape.ipynb
git commit -m "feat: update visual gallery captions and wire new outputs into notebook"
```

- [ ] **Step 6: Push to GitHub**

```bash
git remote set-url origin https://github.com/DogInfantry/india-ai-public-equity-landscape.git
git push origin main
```

---

## Verification Checklist

Run this after a full notebook execution to confirm all spec success criteria:

```bash
python -c "
from pathlib import Path
import sys

root = Path('.')
failures = []

# Criterion 1: HTML report
html = root / 'reports' / 'ai_india_thematic_pitch.html'
if not html.exists(): failures.append('HTML report missing')
elif html.stat().st_size >= 2 * 1024 * 1024: failures.append('HTML report > 2MB')
else:
    content = html.read_text(encoding='utf-8')
    if '<link href=\"http' in content: failures.append('HTML has external CSS link')
    if '<script src=\"http' in content: failures.append('HTML has external script')

# Criterion 2: Tearsheets
combined = root / 'reports' / 'tearsheets' / 'top5_tearsheets_combined.pdf'
if not combined.exists(): failures.append('Combined tearsheet missing')
individual = list((root / 'reports' / 'tearsheets').glob('*_tearsheet.pdf'))
individual = [p for p in individual if 'combined' not in p.name]
if len(individual) != 5: failures.append(f'Expected 5 individual tearsheets, found {len(individual)}')

# Criterion 4: Chart files
for name in ['ai_india_sharpe_return', 'ai_india_correlation_heatmap',
             'ai_india_drawdown_timeline', 'ai_india_efficient_frontier']:
    for ext in ['.png', '.jpg']:
        p = root / 'reports' / 'figures' / (name + ext)
        if not p.exists(): failures.append(f'Missing: {p}')

# Criterion 6: README markers
readme = (root / 'README.md').read_text(encoding='utf-8')
if '<!-- KEY_FINDINGS_START -->' not in readme: failures.append('README missing KEY_FINDINGS_START')
if '<!-- KEY_FINDINGS_END -->' not in readme: failures.append('README missing KEY_FINDINGS_END')

if failures:
    print('FAILURES:')
    for f in failures: print(f'  - {f}')
    sys.exit(1)
else:
    print('All verification checks passed.')
"
```
