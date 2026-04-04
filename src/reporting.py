"""Helpers for turning analysis tables into slide-ready markdown reports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def write_markdown(path: str | Path, content: str) -> Path:
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = _repo_root() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def format_market_cap(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if value >= 1e12:
        return f"INR {value / 1e12:,.2f}tn"
    return f"INR {value / 1e9:,.0f}bn"


def format_multiple(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.1f}x"


def format_pct(value: float | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{decimals}%}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    return df.to_markdown(index=False)


def company_snapshot_table(stats: pd.DataFrame) -> pd.DataFrame:
    display = stats.copy()
    display["Market Cap"] = display["market_cap"].apply(format_market_cap)
    display["PE"] = display["trailing_pe"].apply(format_multiple)
    display["PB"] = display["price_to_book"].apply(format_multiple)
    display["Dividend Yield"] = display["dividend_yield"].apply(format_pct)
    display["1Y Return"] = display["return_1y"].apply(format_pct)
    display["3Y CAGR"] = display["cagr_3y"].apply(format_pct)
    display["Annualized Vol"] = display["annualized_volatility"].apply(format_pct)
    return display[
        [
            "name",
            "segment",
            "Market Cap",
            "PE",
            "PB",
            "Dividend Yield",
            "1Y Return",
            "3Y CAGR",
            "Annualized Vol",
        ]
    ].rename(columns={"name": "Company", "segment": "Segment"})


def ranking_table(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    subset = df.loc[:, list(columns)].copy()
    for column in subset.columns:
        lower_name = column.lower()
        if "return" in lower_name or "cagr" in lower_name or "yield" in lower_name or "volatility" in lower_name:
            subset[column] = subset[column].apply(format_pct)
        if lower_name in {"market_cap", "market cap"}:
            subset[column] = subset[column].apply(format_market_cap)
        if lower_name in {"trailing_pe", "pe", "price_to_book", "pb"}:
            subset[column] = subset[column].apply(format_multiple)
        if lower_name in {"ai_purity_score", "total_score"}:
            subset[column] = subset[column].apply(lambda value: "n/a" if pd.isna(value) else f"{value:.2f}")
    return subset


def archetype_output_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for pct_column in ("return_1y", "cagr_3y", "annualized_volatility", "dividend_yield"):
        if pct_column in display.columns:
            display[pct_column] = display[pct_column].apply(format_pct)
    for score_column in (
        "total_score",
        "growth_score",
        "momentum_score",
        "volatility_score",
        "yield_score",
        "ai_purity_score",
        "history_penalty",
    ):
        if score_column in display.columns:
            display[score_column] = display[score_column].apply(lambda value: "n/a" if pd.isna(value) else f"{value:.2f}")
    return display


def build_thematic_pitch_markdown(
    as_of_date: str,
    executive_summary: list[str],
    macro_context: list[str],
    company_snapshot: pd.DataFrame,
    ranking_sections: Mapping[str, pd.DataFrame],
    risks: list[str],
    hypothetical_ideas: list[dict[str, str]],
    slide_outline: list[str],
    sources: list[str],
) -> str:
    sections: list[str] = [
        "# AI x India Thematic Pitch",
        "",
        f"_As of {as_of_date}. Market data is fetched at runtime from Yahoo Finance/yfinance, with REST fallbacks where needed._",
        "",
        "## Executive Summary",
        *[f"- {bullet}" for bullet in executive_summary],
        "",
        "## Why AI Matters for India Equities",
        *[f"- {bullet}" for bullet in macro_context],
        "",
        "## Company Snapshot",
        dataframe_to_markdown(company_snapshot),
        "",
        "## Key Rankings",
    ]

    for heading, ranking_df in ranking_sections.items():
        sections.extend(
            [
                "",
                f"### {heading}",
                dataframe_to_markdown(ranking_df),
            ]
        )

    sections.extend(["", "## Risks and What Could Go Wrong", *[f"- {bullet}" for bullet in risks], "", "## Hypothetical M&A / Partnership Ideas"])
    for idea in hypothetical_ideas:
        sections.extend(
            [
                f"1. **{idea['company']} ({idea['ticker']})**",
                f"   {idea['idea']}",
                f"   Rationale: {idea['rationale']}",
            ]
        )

    sections.extend(["", "## Slide-Ready Outline", *[f"1. {item}" for item in slide_outline], "", "## Public Macro Sources"])
    sections.extend([f"- {source}" for source in sources])
    return "\n".join(sections) + "\n"


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


def risk_adjusted_ranking_table(stats: pd.DataFrame) -> pd.DataFrame:
    """Build risk-adjusted ranking table (Sharpe, Sortino, rank)."""
    _stats = stats.copy()
    _stats["sharpe"] = pd.to_numeric(_stats["sharpe"], errors="coerce")
    _stats["sortino"] = pd.to_numeric(_stats["sortino"], errors="coerce")
    ranked = _stats.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False).copy()
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
        lines.append("### Max Sharpe Portfolio")
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
    _stats = stats.copy()
    _stats["sharpe"] = pd.to_numeric(_stats["sharpe"], errors="coerce")
    top5 = _stats.dropna(subset=["sharpe"]).nlargest(5, "sharpe")
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


def generate_tearsheets(
    stats: pd.DataFrame,
    price_history: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """
    Generate single-page PDF tearsheets for the top 5 names by Sharpe.
    Saves both individual files and a 5-page combined PDF.
    """
    if output_dir is None:
        output_dir = _repo_root() / "reports" / "tearsheets"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _stats = stats.copy()
    _stats["sharpe"] = pd.to_numeric(_stats["sharpe"], errors="coerce")
    top5 = _stats.dropna(subset=["sharpe"]).nlargest(5, "sharpe")
    paths: dict[str, Path] = {}
    combined_path = output_dir / "top5_tearsheets_combined.pdf"

    with PdfPages(combined_path) as combined_pdf:
        for _, row in top5.iterrows():
            ticker = row["ticker"]
            name = row.get("name", ticker)
            segment = row.get("segment", "")

            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")  # A4
            gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

            fig.text(0.05, 0.96, name, fontsize=18, fontweight="bold", color="#163B65")
            fig.text(0.05, 0.93,
                     f"{segment}  |  {ticker.replace('.NS', '')}  |  {pd.Timestamp.today().strftime('%Y-%m-%d')}",
                     fontsize=10, color="#5B6770")

            ax_chart = fig.add_subplot(gs[0, :])
            hist = (
                price_history[price_history["ticker"] == ticker]
                .assign(date=lambda df: pd.to_datetime(df["date"]))
                .sort_values("date")
                .set_index("date")["adj_close"]
                .dropna()
            )
            if not hist.empty:
                cutoff = hist.index.max() - pd.Timedelta(days=365)
                hist = hist.loc[hist.index >= cutoff]
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

            fig.text(0.05, 0.01,
                     "Educational portfolio sample only. Not investment advice. Data: yfinance public API.",
                     fontsize=7, color="#888888")

            combined_pdf.savefig(fig, bbox_inches="tight")

            safe_ticker = ticker.replace(".NS", "")
            individual_path = output_dir / f"{safe_ticker}_tearsheet.pdf"
            with PdfPages(individual_path) as ind_pdf:
                ind_pdf.savefig(fig, bbox_inches="tight")
            paths[ticker] = individual_path
            plt.close(fig)

    paths["combined"] = combined_path
    return paths


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

    findings = []
    _df = summary_df.copy()
    _df["sharpe"] = pd.to_numeric(_df["sharpe"], errors="coerce")
    _df["cagr_3y"] = pd.to_numeric(_df["cagr_3y"], errors="coerce")

    ranked_sharpe = _df.dropna(subset=["sharpe"]).sort_values("sharpe", ascending=False)
    if not ranked_sharpe.empty:
        best = ranked_sharpe.iloc[0]
        name = best.get("name", best["ticker"].replace(".NS", ""))
        cagr_or_1y = best.get("cagr_3y") if pd.notna(best.get("cagr_3y")) else best.get("return_1y")
        findings.append(
            f"**Best risk-adjusted name:** {name} — "
            f"{best['sharpe']:.2f} Sharpe, "
            f"{format_pct(cagr_or_1y)} trailing CAGR"
        )

    ranked_return = _df.dropna(subset=["cagr_3y"]).sort_values("cagr_3y", ascending=False)
    if not ranked_return.empty:
        top = ranked_return.iloc[0]
        name = top.get("name", top["ticker"].replace(".NS", ""))
        sharpe_suffix = f", Sharpe {top['sharpe']:.2f}" if pd.notna(top.get("sharpe")) else ""
        findings.append(
            f"**Highest compounder:** {name} — "
            f"{format_pct(top.get('cagr_3y'))} 3Y CAGR{sharpe_suffix}"
        )

    high_val = _df.dropna(subset=["valuation_pe_percentile"]).nlargest(1, "valuation_pe_percentile")
    if not high_val.empty:
        row = high_val.iloc[0]
        name = row.get("name", row["ticker"].replace(".NS", ""))
        findings.append(
            f"**Valuation flag:** {name} is at the "
            f"{row['valuation_pe_percentile']:.0f}th percentile of its own PE history — "
            "the return may already be priced in"
        )

    # Fallback bullet: best 1Y return (when cagr_3y is unavailable)
    if len(findings) < 4:
        ranked_1y = _df.dropna(subset=["return_1y"]).sort_values("return_1y", ascending=False)
        if not ranked_1y.empty:
            top1y = ranked_1y.iloc[0]
            top1y_name = top1y.get("name", top1y["ticker"].replace(".NS", ""))
            findings.append(
                f"**Highest 1Y return:** {top1y_name} — "
                f"{format_pct(top1y.get('return_1y'))} trailing 1Y total return"
            )

    # Fallback bullet: top volatility name
    if len(findings) < 4:
        ranked_vol = _df.dropna(subset=["annualized_volatility"]).sort_values("annualized_volatility", ascending=False)
        if not ranked_vol.empty:
            hv = ranked_vol.iloc[0]
            hv_name = hv.get("name", hv["ticker"].replace(".NS", ""))
            findings.append(
                f"**Highest volatility:** {hv_name} — "
                f"{format_pct(hv.get('annualized_volatility'))} annualized vol; factor into position sizing"
            )

    # Final fallback: universe size
    if len(findings) < 4:
        findings.append(
            f"**Universe size:** {len(summary_df)} names across AI, analytics, ER&D, and IT services segments"
        )

    new_block = (
        "<!-- KEY_FINDINGS_START -->\n"
        + "\n".join(f"- {f}" for f in findings[:4])
        + "\n<!-- KEY_FINDINGS_END -->"
    )

    updated = re.sub(
        r"<!-- KEY_FINDINGS_START -->.*?<!-- KEY_FINDINGS_END -->",
        new_block,
        content,
        flags=re.DOTALL,
    )
    readme_path.write_text(updated, encoding="utf-8")


def build_idea_engine_markdown(
    as_of_date: str,
    model_explanation: list[str],
    limitations: list[str],
    archetype_tables: Mapping[str, pd.DataFrame],
    workflow_stories: list[str],
) -> str:
    sections: list[str] = [
        "# AI-Augmented Idea Engine Notes",
        "",
        f"_As of {as_of_date}. Scores are derived from live market data and deterministic factor rules._",
        "",
        "## Scoring Model",
        *[f"- {bullet}" for bullet in model_explanation],
        "",
        "## Limitations and Guardrails",
        *[f"- {bullet}" for bullet in limitations],
        "",
        "## Archetype Outputs",
    ]

    for archetype_name, table in archetype_tables.items():
        sections.extend(["", f"### {archetype_name}", dataframe_to_markdown(table)])

    sections.extend(["", "## Advisor Workflow Stories"])
    sections.extend([f"1. {story}" for story in workflow_stories])
    return "\n".join(sections) + "\n"
