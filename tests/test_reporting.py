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
    individual = [p for p in individual if "combined" not in p.name]
    assert len(individual) == min(5, len(summary_df.dropna(subset=["sharpe"])))


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
    assert "footer" in content


def test_readme_update_generates_four_bullets(tmp_path, summary_df):
    from src.reporting import update_readme_key_findings
    readme = tmp_path / "README.md"
    readme.write_text("<!-- KEY_FINDINGS_START -->\n<!-- KEY_FINDINGS_END -->\n", encoding="utf-8")
    update_readme_key_findings(summary_df, readme)
    content = readme.read_text(encoding="utf-8")
    bullets = [line for line in content.splitlines() if line.startswith("- **")]
    assert len(bullets) >= 4
