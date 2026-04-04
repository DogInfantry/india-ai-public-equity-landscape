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
