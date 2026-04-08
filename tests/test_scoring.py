"""Edge-case tests for src/scoring.py.

All tests use synthetic data only — no live API calls, no yfinance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.scoring import (
    _winsorized_min_max,
    compute_factor_scores,
    derive_ai_purity_score,
    history_penalty,
)


# ---------------------------------------------------------------------------
# history_penalty
# ---------------------------------------------------------------------------

class TestHistoryPenalty:
    def test_none_returns_minimum(self):
        assert history_penalty(None) == pytest.approx(0.60)

    def test_nan_returns_minimum(self):
        assert history_penalty(float("nan")) == pytest.approx(0.60)

    def test_zero_returns_minimum(self):
        assert history_penalty(0.0) == pytest.approx(0.60)

    def test_half_year_returns_minimum(self):
        assert history_penalty(0.5) == pytest.approx(0.60)

    def test_one_year_returns_75(self):
        assert history_penalty(1.0) == pytest.approx(0.75)

    def test_one_and_half_years_returns_75(self):
        assert history_penalty(1.5) == pytest.approx(0.75)

    def test_two_years_returns_90(self):
        assert history_penalty(2.0) == pytest.approx(0.90)

    def test_two_and_half_years_returns_90(self):
        assert history_penalty(2.5) == pytest.approx(0.90)

    def test_three_years_returns_full(self):
        assert history_penalty(3.0) == pytest.approx(1.00)

    def test_five_years_returns_full(self):
        assert history_penalty(5.0) == pytest.approx(1.00)

    def test_large_value_returns_full(self):
        assert history_penalty(20.0) == pytest.approx(1.00)


# ---------------------------------------------------------------------------
# _winsorized_min_max
# ---------------------------------------------------------------------------

class TestWinsorizedMinMax:
    def _series(self, values):
        return pd.Series(values, dtype=float)

    def test_normal_range_outputs_between_0_and_1(self):
        s = self._series(range(100))
        result = _winsorized_min_max(s)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_all_identical_values_return_neutral(self):
        s = self._series([5.0] * 10)
        result = _winsorized_min_max(s)
        assert (result == pytest.approx(0.50)).all()

    def test_single_row_returns_neutral(self):
        s = self._series([42.0])
        result = _winsorized_min_max(s)
        assert result.iloc[0] == pytest.approx(0.50)

    def test_all_nan_returns_neutral(self):
        s = self._series([float("nan")] * 5)
        result = _winsorized_min_max(s)
        assert (result == pytest.approx(0.50)).all()

    def test_higher_is_better_false_flips_direction(self):
        s = self._series(range(100))
        asc = _winsorized_min_max(s, higher_is_better=True)
        desc = _winsorized_min_max(s, higher_is_better=False)
        # Correlation should be strongly negative
        corr = asc.corr(desc)
        assert corr < -0.95

    def test_mixed_nan_and_valid(self):
        s = self._series([1.0, float("nan"), 3.0, float("nan"), 5.0])
        result = _winsorized_min_max(s)
        assert result.notna().all()
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_custom_neutral_value(self):
        s = self._series([float("nan")] * 3)
        result = _winsorized_min_max(s, neutral=0.25)
        assert (result == pytest.approx(0.25)).all()


# ---------------------------------------------------------------------------
# compute_factor_scores
# ---------------------------------------------------------------------------

def _make_universe(**overrides) -> pd.DataFrame:
    """Build a minimal synthetic universe DataFrame for scoring tests."""
    base = {
        "ticker": ["AAA.NS", "BBB.NS", "CCC.NS"],
        "name": ["Alpha Ltd", "Beta Ltd", "Gamma Ltd"],
        "segment": ["IT services", "analytics platform", "ER&D"],
        "notes": ["machine learning platform", "genai analytics", "gpu compute"],
        "cagr_3y": [0.15, 0.22, 0.10],
        "return_1y": [0.18, 0.25, 0.12],
        "momentum_6m": [0.08, 0.14, 0.05],
        "momentum_12m": [0.12, 0.20, 0.09],
        "annualized_volatility": [0.22, 0.30, 0.18],
        "dividend_yield": [0.015, 0.00, 0.025],
        "history_years": [5.0, 3.0, 4.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class TestComputeFactorScores:
    def test_all_score_columns_present(self):
        df = _make_universe()
        result = compute_factor_scores(df)
        for col in ("growth_score", "momentum_score", "volatility_score", "yield_score", "ai_purity_score", "history_penalty"):
            assert col in result.columns, f"Missing column: {col}"

    def test_scores_in_unit_interval(self):
        df = _make_universe()
        result = compute_factor_scores(df)
        for col in ("growth_score", "momentum_score", "volatility_score", "yield_score"):
            assert result[col].between(0.0, 1.0).all(), f"{col} has values outside [0, 1]"

    def test_cagr_nan_falls_back_to_return_1y(self):
        """When cagr_3y is NaN, growth_score should still compute from return_1y."""
        df = _make_universe(cagr_3y=[float("nan"), float("nan"), float("nan")])
        result = compute_factor_scores(df)
        assert result["growth_score"].notna().all()

    def test_zero_dividend_yield_gives_low_yield_score(self):
        """A row with dividend_yield=0 should score at or near the bottom of yield_score."""
        df = _make_universe(dividend_yield=[0.0, 0.0, 0.0])
        result = compute_factor_scores(df)
        # All identical → neutral (0.5) because _winsorized_min_max returns neutral for constant series
        assert result["yield_score"].between(0.0, 1.0).all()

    def test_all_nan_metrics_do_not_crash(self):
        """A row where all numeric metrics are NaN should not raise and should return neutral scores."""
        df = _make_universe(
            cagr_3y=[float("nan")] * 3,
            return_1y=[float("nan")] * 3,
            momentum_6m=[float("nan")] * 3,
            momentum_12m=[float("nan")] * 3,
            annualized_volatility=[float("nan")] * 3,
            dividend_yield=[float("nan")] * 3,
        )
        result = compute_factor_scores(df)  # must not raise
        for col in ("growth_score", "momentum_score", "volatility_score", "yield_score"):
            assert result[col].notna().all(), f"{col} has NaN after full-NaN input"

    def test_ai_purity_uses_segment_and_notes(self):
        """AI-purity should reflect segment taxonomy and keyword boosts."""
        df = _make_universe(
            segment=["AI infrastructure", "IT services", "IT services"],
            notes=["gpu compute", "no ai here", "machine learning"],
        )
        result = compute_factor_scores(df)
        # AI infrastructure + gpu keyword should score highest
        assert result.loc[0, "ai_purity_score"] > result.loc[1, "ai_purity_score"]

    def test_history_penalty_applied_for_short_history(self):
        df = _make_universe(history_years=[0.5, 5.0, 3.0])
        result = compute_factor_scores(df)
        assert result.loc[0, "history_penalty"] == pytest.approx(0.60)
        assert result.loc[1, "history_penalty"] == pytest.approx(1.00)
        assert result.loc[2, "history_penalty"] == pytest.approx(1.00)


# ---------------------------------------------------------------------------
# derive_ai_purity_score
# ---------------------------------------------------------------------------

class TestDeriveAiPurityScore:
    def test_known_segment_returns_expected_base(self):
        score = derive_ai_purity_score("AI infrastructure", "")
        assert score == pytest.approx(1.00)

    def test_unknown_segment_returns_default_base(self):
        score = derive_ai_purity_score("unknown_segment", "")
        assert score == pytest.approx(0.55)

    def test_keyword_boost_adds_correctly(self):
        base = derive_ai_purity_score("IT services", "")
        boosted = derive_ai_purity_score("IT services", "machine learning genai")
        assert boosted > base
        assert boosted == pytest.approx(min(base + 0.03 + 0.03, 1.0))

    def test_score_clipped_to_1(self):
        score = derive_ai_purity_score("AI infrastructure", "genai machine learning gpu ai compute")
        assert score <= 1.0

    def test_none_notes_does_not_crash(self):
        score = derive_ai_purity_score("IT services", None)
        assert 0.0 <= score <= 1.0
