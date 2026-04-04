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


def test_sharpe_return_scatter_creates_png(tmp_path, summary_df):
    from src.visuals import save_sharpe_return_scatter
    path = save_sharpe_return_scatter(summary_df, output_dir=tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0
    assert (tmp_path / "ai_india_sharpe_return.jpg").exists()


def test_correlation_heatmap_creates_png(tmp_path, price_history_df):
    from src.visuals import save_correlation_heatmap
    from src.analysis import compute_correlation_matrix
    corr = compute_correlation_matrix(price_history_df)
    path = save_correlation_heatmap(corr, output_dir=tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_drawdown_timeline_creates_png(tmp_path, price_history_df, summary_df):
    from src.visuals import save_drawdown_timeline
    path = save_drawdown_timeline(price_history_df, summary_df, output_dir=tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_efficient_frontier_chart_creates_png(tmp_path, price_history_df, summary_df):
    from src.visuals import save_efficient_frontier_chart
    from src.analysis import compute_efficient_frontier
    frontier = compute_efficient_frontier(price_history_df, min_history_days=100)
    path = save_efficient_frontier_chart(frontier, summary_df, output_dir=tmp_path)
    assert path.exists()
    assert path.stat().st_size > 0


def test_scorecard_also_saves_pdf(tmp_path, summary_df):
    from src.visuals import save_scorecard_png
    save_scorecard_png(summary_df, output_dir=tmp_path)
    assert (tmp_path / "ai_india_scorecard.pdf").exists()
    assert (tmp_path / "ai_india_scorecard.pdf").stat().st_size > 0
