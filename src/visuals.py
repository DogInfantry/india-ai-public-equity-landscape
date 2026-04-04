"""Generate saved visual artifacts for the AI x India work samples."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis import (
    DEFAULT_BENCHMARK_TICKER,
    build_ai_india_dataset,
    cache_stats_path,
    indexed_price_frame,
    make_indexed_performance_chart,
    make_valuation_scatter,
)
from src.scoring import ARCHETYPE_CONFIGS, compute_factor_scores, score_for_archetype

FIGURE_DIR = REPO_ROOT / "reports" / "figures"
COLOR_PALETTE = {
    "navy": "#163B65",
    "blue": "#2E6DA4",
    "teal": "#1F8A8A",
    "gold": "#B8860B",
    "slate": "#5B6770",
    "green": "#2E8B57",
    "red": "#B04A4A",
    "light": "#E9EEF4",
}


def _ensure_figure_dir(output_dir: str | Path | None = None) -> Path:
    directory = FIGURE_DIR if output_dir is None else Path(output_dir)
    if not directory.is_absolute():
        directory = REPO_ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    return directory


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


def _apply_axis_style(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=COLOR_PALETTE["navy"], pad=18)
    if subtitle:
        ax.text(0.0, 1.005, subtitle, transform=ax.transAxes, fontsize=10, color=COLOR_PALETTE["slate"], va="bottom")
    ax.grid(axis="y", color="#D8E0EA", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8C5D1")
    ax.spines["bottom"].set_color("#B8C5D1")


def _label_map() -> dict[str, str]:
    return {
        "INFY.NS": "Infosys",
        "PERSISTENT.NS": "Persistent",
        "TATAELXSI.NS": "Tata Elxsi",
        "NETWEB.NS": "Netweb",
        DEFAULT_BENCHMARK_TICKER: "NIFTY 50",
    }


def _segment_color_lookup(segments: Iterable[str]) -> dict[str, str]:
    palette = [
        COLOR_PALETTE["navy"],
        COLOR_PALETTE["blue"],
        COLOR_PALETTE["teal"],
        COLOR_PALETTE["gold"],
        COLOR_PALETTE["slate"],
        COLOR_PALETTE["green"],
        "#8E5EA2",
        "#D2691E",
    ]
    unique_segments = list(dict.fromkeys(segments))
    return {segment: palette[idx % len(palette)] for idx, segment in enumerate(unique_segments)}


def save_indexed_performance_png(
    price_history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> Path:
    output_path = _ensure_figure_dir(output_dir) / "ai_india_indexed_performance.png"
    representative_tickers = ["INFY.NS", "PERSISTENT.NS", "TATAELXSI.NS", "NETWEB.NS"]
    indexed_frame = indexed_price_frame(
        price_history=price_history,
        benchmark_history=benchmark_history,
        tickers=representative_tickers,
        label_map=_label_map(),
    )

    fig, ax = plt.subplots(figsize=(11, 6.2))
    line_colors = {
        "Infosys": COLOR_PALETTE["navy"],
        "Persistent": COLOR_PALETTE["teal"],
        "Tata Elxsi": COLOR_PALETTE["gold"],
        "Netweb": COLOR_PALETTE["green"],
        "NIFTY 50": COLOR_PALETTE["slate"],
    }
    for series_name, subset in indexed_frame.groupby("series"):
        ax.plot(
            subset["date"],
            subset["indexed_price"],
            label=series_name,
            linewidth=2.2 if series_name != "NIFTY 50" else 2.6,
            color=line_colors.get(series_name, COLOR_PALETTE["blue"]),
        )

    _apply_axis_style(
        ax,
        "AI x India Indexed Performance",
        "Representative names versus the NIFTY 50 benchmark (start = 100).",
    )
    ax.set_ylabel("Indexed Price")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    return _save_figure(fig, output_path)


def save_return_bar_chart_png(stats: pd.DataFrame, output_dir: str | Path | None = None) -> Path:
    output_path = _ensure_figure_dir(output_dir) / "ai_india_1y_return_bar.png"
    chart_data = stats.dropna(subset=["return_1y"]).sort_values("return_1y")
    colors = [COLOR_PALETTE["red"] if value < 0 else COLOR_PALETTE["green"] for value in chart_data["return_1y"]]

    fig, ax = plt.subplots(figsize=(10.8, 9.2))
    ax.barh(chart_data["name"], chart_data["return_1y"], color=colors, alpha=0.92)
    _apply_axis_style(
        ax,
        "AI x India 1Y Return Distribution",
        "Full listed universe sorted by trailing one-year total return.",
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.axvline(0, color=COLOR_PALETTE["slate"], linewidth=1.0, linestyle="--")
    ax.tick_params(axis="y", labelsize=8.6)
    ax.set_xlabel("1Y Total Return")
    return _save_figure(fig, output_path)


def save_theme_map_png(stats: pd.DataFrame, output_dir: str | Path | None = None) -> Path:
    output_path = _ensure_figure_dir(output_dir) / "ai_india_theme_map.png"
    chart_data = stats.dropna(subset=["ai_purity_score", "cagr_3y", "market_cap"]).copy()
    segment_colors = _segment_color_lookup(chart_data["segment"])
    bubble_sizes = np.sqrt(chart_data["market_cap"] / 1e9) * 11

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    for segment, subset in chart_data.groupby("segment"):
        idx = subset.index
        ax.scatter(
            subset["ai_purity_score"],
            subset["cagr_3y"],
            s=bubble_sizes.loc[idx],
            alpha=0.78,
            color=segment_colors[segment],
            edgecolor="white",
            linewidth=0.8,
            label=segment,
        )

    annotation_pool = chart_data.nlargest(7, "market_cap").index.union(chart_data.nlargest(5, "cagr_3y").index)
    for _, row in chart_data.loc[annotation_pool].drop_duplicates(subset=["ticker"]).iterrows():
        ax.annotate(
            row["name"],
            (row["ai_purity_score"], row["cagr_3y"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
            color=COLOR_PALETTE["navy"],
        )

    _apply_axis_style(
        ax,
        "AI x India Theme Map",
        "Higher on the x-axis means more AI purity; higher on the y-axis means stronger trailing 3Y CAGR.",
    )
    ax.set_xlabel("AI Purity Score")
    ax.set_ylabel("3Y CAGR")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0.58, 1.03)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return _save_figure(fig, output_path)


def save_risk_return_png(stats: pd.DataFrame, output_dir: str | Path | None = None) -> Path:
    output_path = _ensure_figure_dir(output_dir) / "ai_india_risk_return_map.png"
    chart_data = stats.dropna(subset=["annualized_volatility", "cagr_3y", "market_cap"]).copy()
    segment_colors = _segment_color_lookup(chart_data["segment"])
    bubble_sizes = np.sqrt(chart_data["market_cap"] / 1e9) * 11

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    for segment, subset in chart_data.groupby("segment"):
        idx = subset.index
        ax.scatter(
            subset["annualized_volatility"],
            subset["cagr_3y"],
            s=bubble_sizes.loc[idx],
            alpha=0.78,
            color=segment_colors[segment],
            edgecolor="white",
            linewidth=0.8,
            label=segment,
        )

    for _, row in chart_data.nlargest(6, "cagr_3y").iterrows():
        ax.annotate(
            row["name"],
            (row["annualized_volatility"], row["cagr_3y"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
            color=COLOR_PALETTE["navy"],
        )

    _apply_axis_style(
        ax,
        "Risk / Return Map",
        "Trailing 3Y CAGR plotted against annualized volatility for live-screened names.",
    )
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("3Y CAGR")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    return _save_figure(fig, output_path)


def save_segment_market_cap_png(stats: pd.DataFrame, output_dir: str | Path | None = None) -> Path:
    output_path = _ensure_figure_dir(output_dir) / "ai_india_segment_market_cap.png"
    segment_totals = (
        stats.dropna(subset=["market_cap"])
        .groupby("segment", as_index=False)["market_cap"]
        .sum()
        .sort_values("market_cap", ascending=False)
    )
    colors = list(_segment_color_lookup(segment_totals["segment"]).values())

    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.bar(segment_totals["segment"], segment_totals["market_cap"] / 1e12, color=colors, alpha=0.92)
    _apply_axis_style(
        ax,
        "Aggregate Market Cap by Segment",
        "Shows where the listed AI x India screen has most capital-market depth.",
    )
    ax.set_ylabel("Aggregate Market Cap (INR tn)")
    ax.tick_params(axis="x", rotation=28, labelsize=9)
    return _save_figure(fig, output_path)


def save_scorecard_png(stats: pd.DataFrame, output_dir: str | Path | None = None) -> Path:
    output_path = _ensure_figure_dir(output_dir) / "ai_india_scorecard.png"
    _stats = stats.copy()
    _stats["cagr_3y"] = pd.to_numeric(_stats["cagr_3y"], errors="coerce")
    _stats["return_1y"] = pd.to_numeric(_stats["return_1y"], errors="coerce")
    _stats["ai_purity_score"] = pd.to_numeric(_stats["ai_purity_score"], errors="coerce")
    median_1y = _stats["return_1y"].median()
    median_3y = _stats["cagr_3y"].median()
    positive_3y = int((_stats["cagr_3y"] > 0).sum())
    high_purity = int((_stats["ai_purity_score"] >= 0.85).sum())
    top_cagr = _stats.nlargest(3, "cagr_3y")["name"].tolist()
    largest_name = stats.nlargest(1, "market_cap")["name"].iloc[0]

    fig = plt.figure(figsize=(11.2, 6.8), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    fig.text(0.05, 0.92, "AI x India Scorecard", fontsize=22, fontweight="bold", color=COLOR_PALETTE["navy"])
    fig.text(
        0.05,
        0.875,
        "A recruiter-friendly snapshot of the live-screened thematic universe.",
        fontsize=11,
        color=COLOR_PALETTE["slate"],
    )

    cards = [
        ("Universe Size", f"{len(stats)} names"),
        ("Median 1Y Return", f"{median_1y:.1%}"),
        ("Median 3Y CAGR", f"{median_3y:.1%}"),
        ("Positive 3Y CAGR", f"{positive_3y} / {stats['cagr_3y'].notna().sum()}"),
        ("High AI Purity", f"{high_purity} names"),
        ("Largest Name", largest_name),
    ]
    x_positions = [0.05, 0.36, 0.67, 0.05, 0.36, 0.67]
    y_positions = [0.62, 0.62, 0.62, 0.30, 0.30, 0.30]
    card_width = 0.26
    card_height = 0.22

    for (title, value), x0, y0 in zip(cards, x_positions, y_positions):
        rect = Rectangle((x0, y0), card_width, card_height, facecolor=COLOR_PALETTE["light"], edgecolor="none")
        ax.add_patch(rect)
        fig.text(x0 + 0.02, y0 + 0.145, title, fontsize=11, color=COLOR_PALETTE["slate"], fontweight="bold")
        fig.text(x0 + 0.02, y0 + 0.065, value, fontsize=20, color=COLOR_PALETTE["navy"], fontweight="bold")

    fig.text(
        0.05,
        0.08,
        f"Top trailing 3Y CAGR names: {', '.join(top_cagr)}.",
        fontsize=11,
        color=COLOR_PALETTE["navy"],
    )
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    return _save_figure(fig, output_path)


def save_archetype_heatmap_png(stats: pd.DataFrame, output_dir: str | Path | None = None) -> Path:
    output_path = _ensure_figure_dir(output_dir) / "ai_india_archetype_heatmap.png"
    scored = compute_factor_scores(stats)
    ranked_frames = {
        archetype_name: score_for_archetype(scored, config)
        for archetype_name, config in ARCHETYPE_CONFIGS.items()
    }

    score_matrix = pd.DataFrame(
        {
            archetype_name: frame.set_index("name")["total_score"]
            for archetype_name, frame in ranked_frames.items()
        }
    )
    top_names = score_matrix.max(axis=1).sort_values(ascending=False).head(10).index
    heatmap_data = score_matrix.loc[top_names]

    fig, ax = plt.subplots(figsize=(11, 6.8))
    image = ax.imshow(heatmap_data.values, cmap="Blues", aspect="auto", vmin=0, vmax=max(1.0, float(np.nanmax(heatmap_data.values))))
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns, rotation=18, ha="right", fontsize=9)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index, fontsize=9)
    _apply_axis_style(
        ax,
        "Top Names Across Advisor Archetypes",
        "Heatmap of total idea-engine scores for the strongest recurring names.",
    )
    ax.grid(False)

    for row_idx in range(heatmap_data.shape[0]):
        for col_idx in range(heatmap_data.shape[1]):
            value = heatmap_data.iloc[row_idx, col_idx]
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color="black")

    fig.colorbar(image, ax=ax, shrink=0.78)
    return _save_figure(fig, output_path)


def save_interactive_html(
    price_history: pd.DataFrame,
    benchmark_history: pd.DataFrame,
    stats: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    directory = _ensure_figure_dir(output_dir)
    indexed_frame = indexed_price_frame(
        price_history=price_history,
        benchmark_history=benchmark_history,
        tickers=["INFY.NS", "PERSISTENT.NS", "TATAELXSI.NS", "NETWEB.NS"],
        label_map=_label_map(),
    )
    indexed_html = directory / "ai_india_indexed_performance.html"
    theme_map_html = directory / "ai_india_theme_map.html"

    make_indexed_performance_chart(indexed_frame, "AI x India: Indexed Performance vs. NIFTY 50").write_html(
        indexed_html,
        include_plotlyjs="cdn",
    )
    make_valuation_scatter(
        stats,
        valuation_column="ai_purity_score",
        return_column="cagr_3y",
        title="AI Purity vs. 3Y CAGR",
        x_label="AI Purity Score",
        y_label="3Y CAGR",
    ).write_html(
        theme_map_html,
        include_plotlyjs="cdn",
    )
    return {"indexed_html": indexed_html, "theme_map_html": theme_map_html}


def build_visual_summary_markdown(as_of_date: str, figure_paths: dict[str, Path], stats: pd.DataFrame) -> str:
    median_1y = stats["return_1y"].median()
    median_3y = stats["cagr_3y"].median()
    top_cagr = stats.nlargest(3, "cagr_3y")[["name", "cagr_3y"]]
    top_cagr_text = ", ".join(f"{row.name} ({row.cagr_3y:.1%})" for row in top_cagr.itertuples())
    highest_purity = stats.nlargest(3, "ai_purity_score")[["name", "ai_purity_score"]]
    purity_text = ", ".join(f"{row.name} ({row.ai_purity_score:.2f})" for row in highest_purity.itertuples())

    return "\n".join(
        [
            "# AI x India Visual Summary",
            "",
            f"_As of {as_of_date}. Static visuals are generated from live public-market data and saved for recruiter-friendly review._",
            "",
            "## Key Stats",
            f"- Universe size: {len(stats)} names",
            f"- Median 1Y return: {median_1y:.1%}",
            f"- Median 3Y CAGR: {median_3y:.1%}",
            f"- Top 3 trailing 3Y CAGR names: {top_cagr_text}",
            f"- Highest AI-purity names: {purity_text}",
            "",
            "## Visual Pack",
            "### Scorecard",
            "![AI x India Scorecard](figures/ai_india_scorecard.png)",
            "",
            "### Indexed Performance",
            "![AI x India Indexed Performance](figures/ai_india_indexed_performance.png)",
            "",
            "### 1Y Return Distribution",
            "![AI x India 1Y Return Distribution](figures/ai_india_1y_return_bar.png)",
            "",
            "### Theme Map",
            "![AI x India Theme Map](figures/ai_india_theme_map.png)",
            "",
            "### Risk / Return Map",
            "![AI x India Risk Return Map](figures/ai_india_risk_return_map.png)",
            "",
            "### Segment Market-Cap View",
            "![AI x India Segment Market Cap](figures/ai_india_segment_market_cap.png)",
            "",
            "### Archetype Heatmap",
            "![AI x India Archetype Heatmap](figures/ai_india_archetype_heatmap.png)",
            "",
            "## Interactive Files",
            f"- Indexed performance HTML: `{figure_paths['indexed_html'].name}`",
            f"- Theme map HTML: `{figure_paths['theme_map_html'].name}`",
            "",
        ]
    ) + "\n"


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

    median_ret = float(chart_data["_return"].median())
    ax.axvline(median_ret, color="#AABBCC", linewidth=1.0, linestyle="--", zorder=0)
    ax.axhline(1.0, color="#AABBCC", linewidth=1.0, linestyle="--", zorder=0)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    for label, x, y in [
        ("High Return\nHigh Quality", x_max * 0.85, max(y_max * 0.85, 1.1)),
        ("High Return\nHigh Risk", x_max * 0.85, min(y_min * 0.5, 0.9)),
        ("Low Return\nHigh Quality", x_min * 0.5, max(y_max * 0.85, 1.1)),
        ("Low Return\nHigh Risk", x_min * 0.5, min(y_min * 0.5, 0.9)),
    ]:
        ax.text(x, y, label, fontsize=8, color="grey", style="italic", ha="center", va="center")

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


def save_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    stats: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
) -> Path:
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


def save_drawdown_timeline(
    price_history: pd.DataFrame,
    stats: pd.DataFrame,
    top_n: int = 8,
    output_dir: str | Path | None = None,
) -> Path:
    output_path = _ensure_figure_dir(output_dir) / "ai_india_drawdown_timeline.png"
    _stats = stats.copy()
    _stats["sharpe"] = pd.to_numeric(_stats["sharpe"], errors="coerce")
    ranked = _stats.dropna(subset=["sharpe"]).nlargest(top_n, "sharpe")
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


def save_efficient_frontier_chart(
    frontier: dict,
    stats: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    output_path = _ensure_figure_dir(output_dir) / "ai_india_efficient_frontier.png"
    sim = frontier["simulated"]  # shape (5000, 3): [vol, return, sharpe]

    fig, ax = plt.subplots(figsize=(11, 7))

    sc = ax.scatter(sim[:, 0], sim[:, 1], c=sim[:, 2], cmap="viridis",
                    alpha=0.35, s=4, zorder=1)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio", shrink=0.8)

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

    if frontier["max_sharpe_stats"] and frontier["max_sharpe_weights"]:
        ms = frontier["max_sharpe_stats"]
        top3 = sorted(frontier["max_sharpe_weights"].items(), key=lambda x: -x[1])[:3]
        label = "Max Sharpe\n" + ", ".join(f"{t.replace('.NS','')} {w:.0%}" for t, w in top3)
        ax.scatter(ms["volatility"], ms["return"], marker="*", color=COLOR_PALETTE["gold"],
                   s=400, zorder=5, label=label)

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


def generate_visual_pack(
    start: str = "2021-01-01",
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Generate a persistent visual package for the AI x India repo."""

    directory = _ensure_figure_dir(output_dir)
    _, price_history, benchmark_history, stats = build_ai_india_dataset(start=start)
    scored = compute_factor_scores(stats)
    scored.to_csv(cache_stats_path(), index=False)

    artifact_paths = {
        "scorecard_png": save_scorecard_png(scored, directory),
        "indexed_png": save_indexed_performance_png(price_history, benchmark_history, directory),
        "return_bar_png": save_return_bar_chart_png(scored, directory),
        "theme_map_png": save_theme_map_png(scored, directory),
        "risk_return_png": save_risk_return_png(scored, directory),
        "segment_market_cap_png": save_segment_market_cap_png(scored, directory),
        "archetype_heatmap_png": save_archetype_heatmap_png(scored, directory),
    }
    artifact_paths.update(save_interactive_html(price_history, benchmark_history, scored, directory))

    from src.analysis import compute_correlation_matrix, compute_efficient_frontier
    corr_matrix = compute_correlation_matrix(price_history)
    frontier = compute_efficient_frontier(price_history)

    artifact_paths["sharpe_return_png"] = save_sharpe_return_scatter(scored, directory)
    artifact_paths["correlation_heatmap_png"] = save_correlation_heatmap(corr_matrix, scored, directory)
    artifact_paths["drawdown_timeline_png"] = save_drawdown_timeline(price_history, scored, directory)
    artifact_paths["efficient_frontier_png"] = save_efficient_frontier_chart(frontier, scored, directory)
    artifact_paths["frontier_data"] = frontier
    artifact_paths["corr_matrix"] = corr_matrix
    artifact_paths["price_history"] = price_history

    summary_markdown = build_visual_summary_markdown(
        as_of_date=scored["as_of_date"].iloc[0],
        figure_paths=artifact_paths,
        stats=scored,
    )
    summary_path = directory.parent / "ai_india_visual_summary.md"
    summary_path.write_text(summary_markdown, encoding="utf-8")
    artifact_paths["visual_summary_md"] = summary_path
    return artifact_paths


if __name__ == "__main__":
    outputs = generate_visual_pack()
    for key, path in outputs.items():
        print(f"{key}: {path}")
