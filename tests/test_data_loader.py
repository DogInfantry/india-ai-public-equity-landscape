"""Tests for load_candidates() and the ai_purity_override column in load_universe().

All tests use synthetic CSV data written to a tmp_path fixture — no network calls.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_candidates, load_universe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_UNIVERSE_CSV = textwrap.dedent("""\
    ticker,name,segment,notes
    TCS.NS,Tata Consultancy Services,IT services,GenAI platform
    INFY.NS,Infosys Ltd,IT services,machine learning
""").strip()

UNIVERSE_WITH_OVERRIDE_CSV = textwrap.dedent("""\
    ticker,name,segment,notes,ai_purity_override
    TCS.NS,Tata Consultancy Services,IT services,GenAI platform,0.95
    INFY.NS,Infosys Ltd,IT services,machine learning,
    NETWEB.NS,Netweb Technologies,AI infrastructure,GPU compute,1.1
""").strip()

MINIMAL_CANDIDATES_CSV = textwrap.dedent("""\
    ticker,name,source_count,segment_guess,notes,status
    NEWGEN.NS,Newgen Software,3,enterprise software,low-code BPM with AI,candidate
    TATACOMM.NS,Tata Communications,3,AI infrastructure,AI networking edge,candidate
    PROMOTED.NS,Some Promoted Co,2,IT services,was a candidate,promoted
    REJECTED.NS,Some Rejected Co,1,analytics platform,not enough AI,rejected
""").strip()


# ---------------------------------------------------------------------------
# load_universe  —  ai_purity_override behaviour
# ---------------------------------------------------------------------------

class TestLoadUniverseOverride:
    def test_no_override_column_added_as_na(self, tmp_path: Path):
        csv = tmp_path / "universe.csv"
        csv.write_text(MINIMAL_UNIVERSE_CSV)
        df = load_universe(str(csv))
        assert "ai_purity_override" in df.columns
        assert df["ai_purity_override"].isna().all()

    def test_valid_override_parsed_as_float(self, tmp_path: Path):
        csv = tmp_path / "universe.csv"
        csv.write_text(UNIVERSE_WITH_OVERRIDE_CSV)
        df = load_universe(str(csv))
        tcs_row = df[df["ticker"] == "TCS.NS"].iloc[0]
        assert tcs_row["ai_purity_override"] == pytest.approx(0.95)

    def test_empty_override_cell_is_nan(self, tmp_path: Path):
        csv = tmp_path / "universe.csv"
        csv.write_text(UNIVERSE_WITH_OVERRIDE_CSV)
        df = load_universe(str(csv))
        infy_row = df[df["ticker"] == "INFY.NS"].iloc[0]
        assert pd.isna(infy_row["ai_purity_override"])

    def test_override_clipped_to_1(self, tmp_path: Path):
        """Values > 1.0 must be clipped to 1.0."""
        csv = tmp_path / "universe.csv"
        csv.write_text(UNIVERSE_WITH_OVERRIDE_CSV)
        df = load_universe(str(csv))
        netweb_row = df[df["ticker"] == "NETWEB.NS"].iloc[0]
        assert netweb_row["ai_purity_override"] == pytest.approx(1.0)

    def test_missing_required_column_raises(self, tmp_path: Path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("ticker,name\nTCS.NS,TCS")
        with pytest.raises(ValueError, match="missing required columns"):
            load_universe(str(bad_csv))


# ---------------------------------------------------------------------------
# load_candidates
# ---------------------------------------------------------------------------

class TestLoadCandidates:
    def test_loads_all_rows_by_default(self, tmp_path: Path):
        csv = tmp_path / "candidates.csv"
        csv.write_text(MINIMAL_CANDIDATES_CSV)
        df = load_candidates(str(csv))
        assert len(df) == 4

    def test_status_filter_candidate(self, tmp_path: Path):
        csv = tmp_path / "candidates.csv"
        csv.write_text(MINIMAL_CANDIDATES_CSV)
        df = load_candidates(str(csv), status_filter=["candidate"])
        assert list(df["status"].unique()) == ["candidate"]
        assert len(df) == 2

    def test_status_filter_multiple(self, tmp_path: Path):
        csv = tmp_path / "candidates.csv"
        csv.write_text(MINIMAL_CANDIDATES_CSV)
        df = load_candidates(str(csv), status_filter=["candidate", "promoted"])
        assert set(df["status"].unique()) == {"candidate", "promoted"}

    def test_source_count_is_numeric(self, tmp_path: Path):
        csv = tmp_path / "candidates.csv"
        csv.write_text(MINIMAL_CANDIDATES_CSV)
        df = load_candidates(str(csv))
        assert pd.api.types.is_numeric_dtype(df["source_count"])

    def test_missing_file_returns_empty_dataframe(self, tmp_path: Path):
        df = load_candidates(str(tmp_path / "does_not_exist.csv"))
        assert df.empty
        assert "ticker" in df.columns

    def test_missing_required_column_raises(self, tmp_path: Path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("ticker,name\nTCS.NS,TCS")
        with pytest.raises(ValueError, match="missing required columns"):
            load_candidates(str(bad_csv))

    def test_required_columns_present(self, tmp_path: Path):
        csv = tmp_path / "candidates.csv"
        csv.write_text(MINIMAL_CANDIDATES_CSV)
        df = load_candidates(str(csv))
        for col in ("ticker", "name", "source_count", "segment_guess", "notes", "status"):
            assert col in df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# compute_excess_returns + compute_information_ratio edge cases (Issue #6)
# ---------------------------------------------------------------------------

class TestExcessReturnsAndIR:
    def _price_series(self, values, start="2023-01-01"):
        idx = pd.bdate_range(start=start, periods=len(values))
        return pd.Series(values, index=idx, dtype=float)

    def test_excess_returns_normal_case(self):
        from src.analysis import compute_excess_returns
        stock = self._price_series([100, 102, 101, 105, 108])
        bench = self._price_series([100, 101, 101, 103, 105])
        result = compute_excess_returns(stock, bench)
        assert len(result) == 4  # pct_change drops first row
        assert result.name == "excess_return"

    def test_excess_returns_empty_on_no_overlap(self):
        from src.analysis import compute_excess_returns
        stock = self._price_series([100, 102], start="2023-01-02")
        bench = self._price_series([100, 102], start="2024-06-01")
        result = compute_excess_returns(stock, bench)
        assert result.empty

    def test_excess_returns_handles_all_nan_stock(self):
        from src.analysis import compute_excess_returns
        import numpy as np
        stock = self._price_series([float("nan")] * 5)
        bench = self._price_series([100, 101, 102, 103, 104])
        result = compute_excess_returns(stock, bench)
        assert result.empty

    def test_ir_returns_float_for_valid_series(self):
        from src.analysis import compute_information_ratio
        import numpy as np
        rng = pd.bdate_range("2023-01-01", periods=252)
        excess = pd.Series(np.random.default_rng(42).normal(0.0003, 0.008, 252), index=rng)
        ir = compute_information_ratio(excess)
        assert isinstance(ir, float)

    def test_ir_returns_none_for_empty_series(self):
        from src.analysis import compute_information_ratio
        assert compute_information_ratio(pd.Series(dtype=float)) is None

    def test_ir_returns_none_for_zero_tracking_error(self):
        from src.analysis import compute_information_ratio
        flat = pd.Series([0.001] * 100)
        result = compute_information_ratio(flat)
        assert result is None

    def test_ir_returns_none_for_single_observation(self):
        from src.analysis import compute_information_ratio
        assert compute_information_ratio(pd.Series([0.005])) is None
