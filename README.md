# AI x India: Public Equity Landscape

[![Status](https://img.shields.io/badge/status-active-brightgreen)]()
[![Domain](https://img.shields.io/badge/domain-public%20equities-blue)]()
[![Theme](https://img.shields.io/badge/theme-AI%20investing-purple)]()
[![Focus](https://img.shields.io/badge/focus-quant%20analysis-orange)]()
[![Focus](https://img.shields.io/badge/focus-portfolio%20optimization-yellow)]()
[![Metrics](https://img.shields.io/badge/metrics-sharpe%20%7C%20sortino%20%7C%20drawdown-red)]()
[![Valuation](https://img.shields.io/badge/valuation-PE%20%7C%20PB%20percentiles-blueviolet)]()
[![Tools](https://img.shields.io/badge/tools-python%20%7C%20streamlit%20%7C%20yfinance-black)]()
[![Output](https://img.shields.io/badge/output-tearsheets%20%7C%20HTML%20report-success)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](CONTRIBUTING.md)

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

## How to contribute

This project is open to contributions from equity researchers, quants, and engineers.

- Browse the **[Issues](../../issues)** tab for items labelled `good first issue` or `help wanted`
- Read **[CONTRIBUTING.md](CONTRIBUTING.md)** for setup, coding style, and PR guidelines
- Open an issue before working on something large — it helps avoid duplicated effort

Contribution tracks: `track:data` · `track:analytics` · `track:app` · `track:infra`

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

This repository is for educational and portfolio-use purposes only. It is not investment advice, not a research product. All data sourced from public APIs at runtime.
