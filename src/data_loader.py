"""Utilities for loading the AI x India universe and public market data."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf

LOGGER = logging.getLogger(__name__)

DEFAULT_INDIAN_STOCK_API_BASE_URL = os.getenv(
    "INDIAN_STOCK_API_BASE_URL",
    "https://military-jobye-haiqstudios-14f59639.koyeb.app",
)
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


@dataclass(frozen=True)
class FundamentalSnapshot:
    """Container for the small fundamental set used in the pitch materials."""

    ticker: str
    market_cap: float | None
    trailing_pe: float | None
    price_to_book: float | None
    dividend_yield: float | None
    source: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _repo_root() / candidate


def _ensure_yfinance_cache() -> None:
    """Redirect yfinance caches into the repository for sandbox-friendly execution."""

    cache_dir = _repo_root() / ".cache" / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))


def load_universe(path: str = "data/ai_india_universe.csv") -> pd.DataFrame:
    """Load the curated AI x India stock universe.

    If the CSV contains an ``ai_purity_override`` column, rows with a non-null
    value in that column will use that value as the final AI-purity score
    instead of the deterministic segment + keyword logic in
    :func:`src.scoring.derive_ai_purity_score`.  Rows without an override
    (empty / NaN) are unaffected.
    """

    csv_path = _resolve_path(path)
    universe = pd.read_csv(csv_path)
    required_columns = {"ticker", "name", "segment", "notes"}
    missing_columns = required_columns.difference(universe.columns)
    if missing_columns:
        raise ValueError(f"Universe file is missing required columns: {sorted(missing_columns)}")

    # Normalise the optional override column so downstream code can rely on it.
    if "ai_purity_override" not in universe.columns:
        universe["ai_purity_override"] = pd.NA
    else:
        universe["ai_purity_override"] = pd.to_numeric(
            universe["ai_purity_override"], errors="coerce"
        ).clip(lower=0.0, upper=1.0)

    return universe


def load_candidates(
    path: str = "data/ai_india_candidates.csv",
    status_filter: list[str] | None = None,
) -> pd.DataFrame:
    """Load the AI x India candidate funnel.

    Parameters
    ----------
    path:
        Path to the candidates CSV, relative to the repository root or absolute.
    status_filter:
        Optional list of status values to keep (e.g. ``["candidate"]``).  When
        ``None`` all rows are returned regardless of status.

    Returns
    -------
    pd.DataFrame
        Candidates table with at minimum the columns: ``ticker``, ``name``,
        ``source_count``, ``segment_guess``, ``notes``, ``status``.
        Returns an empty DataFrame when the file does not exist.
    """

    csv_path = _resolve_path(path)
    if not csv_path.exists():
        LOGGER.warning("Candidates file not found at %s; returning empty DataFrame.", csv_path)
        return pd.DataFrame(
            columns=["ticker", "name", "source_count", "segment_guess", "notes", "status"]
        )

    candidates = pd.read_csv(csv_path)
    required_columns = {"ticker", "name", "status"}
    missing_columns = required_columns.difference(candidates.columns)
    if missing_columns:
        raise ValueError(f"Candidates file is missing required columns: {sorted(missing_columns)}")

    if status_filter is not None:
        candidates = candidates[candidates["status"].isin(status_filter)].reset_index(drop=True)

    if "source_count" in candidates.columns:
        candidates["source_count"] = pd.to_numeric(candidates["source_count"], errors="coerce")

    return candidates


def _standardize_history_frame(df: pd.DataFrame, ticker: str, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"])

    history = df.copy()
    if isinstance(history.columns, pd.MultiIndex):
        if ticker in history.columns.get_level_values(-1):
            history = history.xs(ticker, axis=1, level=-1)
        else:
            history = history.droplevel(-1, axis=1)

    rename_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    history = history.rename(columns=rename_map).reset_index()
    if "date" not in history.columns:
        history = history.rename(columns={history.columns[0]: "date"})
    if "adj_close" not in history.columns:
        history["adj_close"] = history.get("close")

    history["date"] = pd.to_datetime(history["date"]).dt.tz_localize(None)
    history["ticker"] = ticker
    history["source"] = source
    ordered_columns = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]
    return history[ordered_columns].sort_values("date").reset_index(drop=True)


def _download_yfinance_history(
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    interval: str,
) -> pd.DataFrame:
    _ensure_yfinance_cache()
    history = yf.download(
        ticker,
        start=pd.Timestamp(start).date().isoformat(),
        end=pd.Timestamp(end).date().isoformat(),
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return _standardize_history_frame(history, ticker=ticker, source="yfinance")


def _history_from_api_payload(payload: object, ticker: str, source: str) -> pd.DataFrame:
    """Attempt to coerce a generic JSON payload into OHLCV history."""

    if isinstance(payload, dict):
        for key in ("data", "history", "historical", "candles", "results", "prices"):
            if key in payload:
                payload = payload[key]
                break

    if not isinstance(payload, list):
        return pd.DataFrame()

    history = pd.DataFrame(payload)
    if history.empty:
        return history

    rename_candidates = {
        "datetime": "date",
        "timestamp": "date",
        "time": "date",
        "tradingDate": "date",
        "openPrice": "open",
        "highPrice": "high",
        "lowPrice": "low",
        "closePrice": "close",
        "adjClose": "adj_close",
        "tradedQuantity": "volume",
    }
    history = history.rename(columns=rename_candidates)

    expected = {"date", "open", "high", "low", "close"}
    if not expected.issubset(history.columns):
        return pd.DataFrame()

    if "adj_close" not in history.columns:
        history["adj_close"] = history["close"]
    if "volume" not in history.columns:
        history["volume"] = pd.NA

    history["date"] = pd.to_datetime(history["date"], errors="coerce").dt.tz_localize(None)
    history["ticker"] = ticker
    history["source"] = source
    ordered_columns = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]
    return history[ordered_columns].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _download_from_indian_stock_api(
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    interval: str = "1d",
) -> pd.DataFrame:
    """Try common historical endpoints exposed by community Indian stock API deployments."""

    base_url = DEFAULT_INDIAN_STOCK_API_BASE_URL.rstrip("/")
    session = requests.Session()
    params = {
        "symbol": ticker,
        "ticker": ticker,
        "interval": interval,
        "start": pd.Timestamp(start).date().isoformat(),
        "end": pd.Timestamp(end).date().isoformat(),
        "from": pd.Timestamp(start).date().isoformat(),
        "to": pd.Timestamp(end).date().isoformat(),
    }
    candidate_paths = [
        "/history",
        "/historical",
        "/stock/history",
        "/stock/historical",
        "/candles",
        "/chart",
    ]
    for path in candidate_paths:
        url = f"{base_url}{path}"
        try:
            response = session.get(url, params=params, timeout=20)
            if response.status_code >= 400:
                continue
            history = _history_from_api_payload(response.json(), ticker=ticker, source="indian_stock_api")
            if not history.empty:
                LOGGER.info("Used Indian Stock Market API for %s via %s", ticker, url)
                return history
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning("Indian Stock API request failed for %s via %s: %s", ticker, url, exc)
    return pd.DataFrame()


def _download_yahoo_chart_history(
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fallback REST path that avoids the yfinance package while keeping Yahoo as the source."""

    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp())
    response = requests.get(
        YAHOO_CHART_URL.format(ticker=ticker),
        params={
            "period1": start_ts,
            "period2": end_ts,
            "interval": interval,
            "includeAdjustedClose": "true",
            "events": "div,splits",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("chart", {}).get("result") or []
    if not results:
        return pd.DataFrame()

    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_rows = (result.get("indicators", {}).get("quote") or [{}])[0]
    adj_rows = (result.get("indicators", {}).get("adjclose") or [{}])[0]
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s"),
            "open": quote_rows.get("open"),
            "high": quote_rows.get("high"),
            "low": quote_rows.get("low"),
            "close": quote_rows.get("close"),
            "adj_close": adj_rows.get("adjclose"),
            "volume": quote_rows.get("volume"),
        }
    )
    history["ticker"] = ticker
    history["source"] = "yahoo_chart_api"
    ordered_columns = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"]
    return history[ordered_columns].dropna(subset=["date", "close"]).reset_index(drop=True)


def download_price_history_api(
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """
    Download price history from a non-yfinance REST endpoint.

    The function first attempts the community-hosted Indian Stock Market API.
    If that deployment is unavailable or does not expose historical candles,
    it falls back to the Yahoo Finance chart endpoint via requests.
    """

    indian_api_history = _download_from_indian_stock_api(ticker=ticker, start=start, end=end)
    if not indian_api_history.empty:
        return indian_api_history

    LOGGER.warning(
        "Indian Stock Market API returned no historical data for %s; using direct Yahoo chart REST fallback.",
        ticker,
    )
    try:
        return _download_yahoo_chart_history(ticker=ticker, start=start, end=end)
    except requests.RequestException as exc:
        LOGGER.warning("Yahoo chart REST fallback failed for %s: %s", ticker, exc)
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"])


def download_price_history(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download historical OHLCV data for a list of tickers."""

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        try:
            history = _download_yfinance_history(ticker=ticker, start=start, end=end, interval=interval)
            if history.empty:
                LOGGER.warning("yfinance returned no rows for %s; switching to REST fallback.", ticker)
                history = download_price_history_api(ticker=ticker, start=start, end=end)
        except Exception as exc:  # pragma: no cover - network-driven branch
            LOGGER.warning("yfinance failed for %s: %s. Switching to REST fallback.", ticker, exc)
            history = download_price_history_api(ticker=ticker, start=start, end=end)

        if history.empty:
            LOGGER.warning("No price history could be loaded for %s from any source.", ticker)
            continue
        frames.append(history)

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume", "source"])
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _normalize_dividend_yield(dividend_yield: float | None) -> float | None:
    if dividend_yield is None:
        return None
    if dividend_yield > 1:
        return dividend_yield / 100.0
    if dividend_yield > 0.20:
        return dividend_yield / 100.0
    return dividend_yield


def get_basic_fundamentals(ticker: str) -> dict[str, float | str | None]:
    """Fetch a compact fundamentals snapshot using yfinance, with quote-level fallback support."""

    _ensure_yfinance_cache()
    try:
        instrument = yf.Ticker(ticker)
        info = instrument.info or {}
        fast_info = getattr(instrument, "fast_info", {}) or {}
        market_cap = _safe_float(info.get("marketCap")) or _safe_float(fast_info.get("marketCap"))
        trailing_pe = _safe_float(info.get("trailingPE")) or _safe_float(info.get("forwardPE"))
        price_to_book = _safe_float(info.get("priceToBook"))
        dividend_yield = _normalize_dividend_yield(_safe_float(info.get("dividendYield")))

        if dividend_yield is None:
            dividend_rate = _safe_float(info.get("dividendRate"))
            current_price = _safe_float(info.get("currentPrice")) or _safe_float(fast_info.get("lastPrice"))
            if dividend_rate is not None and current_price:
                dividend_yield = dividend_rate / current_price

        snapshot = FundamentalSnapshot(
            ticker=ticker,
            market_cap=market_cap,
            trailing_pe=trailing_pe,
            price_to_book=price_to_book,
            dividend_yield=dividend_yield,
            source="yfinance",
        )
        return snapshot.__dict__
    except Exception as exc:  # pragma: no cover - network-driven branch
        LOGGER.warning("Could not load yfinance fundamentals for %s: %s", ticker, exc)

    base_url = DEFAULT_INDIAN_STOCK_API_BASE_URL.rstrip("/")
    try:
        response = requests.get(
            f"{base_url}/stock",
            params={"symbol": ticker, "res": "num"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        last_price = _safe_float(payload.get("price") or payload.get("ltp"))
        book_value = _safe_float(payload.get("bookValue"))
        price_to_book = _safe_float(payload.get("priceToBook"))
        if price_to_book is None and last_price and book_value:
            price_to_book = last_price / book_value
        snapshot = FundamentalSnapshot(
            ticker=ticker,
            market_cap=_safe_float(payload.get("marketCap") or payload.get("market_cap")),
            trailing_pe=_safe_float(payload.get("pe") or payload.get("peRatio")),
            price_to_book=price_to_book,
            dividend_yield=_normalize_dividend_yield(
                _safe_float(payload.get("dividendYield") or payload.get("dividend_yield"))
            ),
            source="indian_stock_api",
        )
        return snapshot.__dict__
    except Exception as exc:  # pragma: no cover - network-driven branch
        LOGGER.warning("Could not load fallback fundamentals for %s: %s", ticker, exc)
        return FundamentalSnapshot(
            ticker=ticker,
            market_cap=None,
            trailing_pe=None,
            price_to_book=None,
            dividend_yield=None,
            source="unavailable",
        ).__dict__
