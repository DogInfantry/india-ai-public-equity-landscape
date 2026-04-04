"""Helpers for turning analysis tables into slide-ready markdown reports."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

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
