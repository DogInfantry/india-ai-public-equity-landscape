# Design Spec: India AI Public Equity Landscape — Signal Upgrade

**Date:** 2026-04-04
**Project:** `india-ai-public-equity-landscape`
**Source repo:** `https://github.com/DogInfantry/india-ai-public-equity-landscape`
**Local working directory:** `C:/Users/Anklesh/Documents/Claude_Code/india_ai/`
**Audience:** Finance/IB recruiters at Morgan Stanley (2026)
**Goal:** Transform a technically competent but generic portfolio piece into a professional-grade research product that signals investment fluency at every touchpoint — README, reports, visuals, and analytics.

> **Note on source code location:** All existing Python modules (`src/analysis.py`, `src/visuals.py`, `src/reporting.py`, `src/scoring.py`, `src/data_loader.py`, `src/app.py`) and notebooks (`notebooks/01_ai_india_landscape.ipynb`, `notebooks/02_ai_idea_engine.ipynb`) live in the GitHub repo above. Implementation must clone/pull that repo into the local working directory. All changes described in this spec are additive to the existing codebase — no existing function signatures or column names are removed or renamed.

---

## Problem Statement

The current repo reads like a data engineer built it: the pipeline works, the structure is clean, but there is no editorial voice, no judgment surfaced, and no layer that says "here is what this means and why it matters." Key gaps:

- No risk-adjusted metrics (Sharpe, Sortino) — a finance reviewer notices immediately
- Report prose is templated and descriptive, not analytical
- Visuals are functional but not finance-native
- README is a technical setup guide, not a pitch
- No portfolio construction — just rankings
- No valuation context (where is a stock vs. its own history?)
- No tearsheet-style per-stock output

---

## Scope

Four layers, all additive — no breaking changes to existing functionality.

1. **Analytics layer** (`src/analysis.py`) — new metrics
2. **Reporting layer** (`src/reporting.py` + generated reports) — new prose quality and formats
3. **Visuals layer** (`src/visuals.py`) — new charts and export formats
4. **README** — full rewrite as pitch-deck cover page

**Project root** for all relative paths below: the cloned repo root (e.g., `india-ai-public-equity-landscape/`).

---

## Section 1: Analytics Layer (`src/analysis.py`)

### Existing Function Signatures (confirmed from source)

**`summarize_price_history(universe, price_history, fundamentals) -> pd.DataFrame`**
- `universe`: DataFrame from `load_universe()` — columns include `ticker`, `name`, `segment`, `notes`
- `price_history`: **Long-format** DataFrame — columns `ticker`, `date`, `adj_close`, `source`
- `fundamentals`: DataFrame with a `ticker` column plus fundamental metric columns
- Returns: wide DataFrame, one row per ticker

**Existing output columns:** `ticker`, `name`, `segment`, `notes`, `history_years`, `last_close`, `return_1y`, `return_3y_total`, `cagr_3y`, `momentum_6m`, `momentum_12m`, `annualized_volatility`, `max_drawdown`, `data_source` + all columns from `fundamentals` (e.g., `trailing_pe`, `price_to_book`, `market_cap`, `dividend_yield`)

### Annualization Convention
All annualized metrics use `TRADING_DAYS = 252` (constant already defined in the module). Daily return = `pct_change()`. Annualized return = `mean(daily_returns) * 252`. Annualized volatility = `std(daily_returns) * sqrt(252)`. This convention is consistent with the existing `annualized_volatility()` function.

### NSE Ticker Format
All tickers in the universe CSV use the `.NS` suffix (e.g., `TCS.NS`, `INFY.NS`) as required by yfinance. All new functions accept tickers in `.NS` format. Tearsheet filenames strip the `.NS` suffix (e.g., `TCS_tearsheet.pdf`) to avoid double-dot filenames on Windows.

### New Output Columns Added to `summarize_price_history()`

The following columns are computed and appended inside `summarize_price_history()` using the existing `closes` series per ticker:

| Column | Type | Notes |
|--------|------|-------|
| `sharpe` | float \| None | None if < 30 days of data |
| `sortino` | float \| None | None if no negative return days |
| `drawdown_start` | pd.Timestamp \| None | Peak date before max drawdown |
| `drawdown_trough` | pd.Timestamp \| None | Trough date of max drawdown |
| `drawdown_recovery_days` | int \| None | Trading days trough→recovery; None if not recovered |
| `valuation_pe_percentile` | float \| None | 0–100; None if < 4 quarters of financials available |
| `valuation_pb_percentile` | float \| None | 0–100; None if < 4 quarters of financials available |

**NaN/None handling:** `valuation_pe_percentile` and `valuation_pb_percentile` will legitimately be None for smaller-cap names with thin yfinance quarterly coverage. This is expected and documented — downstream code must handle None gracefully. Success criteria do not require these columns to be non-null universally.

### New Metric Formulas

**Sharpe Ratio**
```python
RISK_FREE_ANNUAL = 0.065  # India 10Y G-Sec approximate
risk_free_daily = RISK_FREE_ANNUAL / TRADING_DAYS
daily_returns = closes.pct_change().dropna()
excess_daily = daily_returns - risk_free_daily
# Note: denominator uses std(daily_returns) for consistency with annualized_volatility()
# This is equivalent to std(excess_returns) since risk_free_daily is constant.
sharpe = (excess_daily.mean() * TRADING_DAYS) / (daily_returns.std() * np.sqrt(TRADING_DAYS))
# Returns None if len(daily_returns) < 30
```

**Sortino Ratio**
```python
# annualized_return must be computed before this block (in scope from summarize_price_history):
# annualized_return = daily_returns.mean() * TRADING_DAYS
# Standard semi-deviation: uses all returns with floor at zero (MAR = 0)
downside = np.minimum(daily_returns.values, 0.0)
downside_deviation = np.sqrt(np.mean(downside ** 2)) * np.sqrt(TRADING_DAYS)
sortino = (annualized_return - RISK_FREE_ANNUAL) / downside_deviation
# Returns None if downside_deviation == 0 (no negative days)
```

**Drawdown Duration**
```python
running_peak = closes.cummax()
dd_series = closes / running_peak - 1.0
trough_date = dd_series.idxmin()
# Find the peak date: last date before trough where price == running peak at trough
peak_price = running_peak.loc[trough_date]
pre_trough = closes.loc[closes.index <= trough_date]
peak_date = pre_trough[pre_trough >= peak_price].index[-1]  # last touch of peak before trough
# Note: pre_trough is raw closes. By definition closes <= cummax always holds,
# so `>=` and `==` against peak_price are equivalent here — no off-by-one risk.
# Find recovery: first date after trough where price >= peak_price
post_trough = closes.loc[closes.index > trough_date]
recovered = post_trough[post_trough >= peak_price]
recovery_date = recovered.index[0] if not recovered.empty else None
# Use true trading-day count (number of index rows), not calendar-day approximation:
recovery_days = len(closes.loc[trough_date:recovery_date]) - 1 if recovery_date else None
```
Human-readable quarter string derived in `reporting.py`: `f"Q{(ts.month-1)//3 + 1} {ts.year}"` from the stored `pd.Timestamp`.

### New Functions

**`compute_correlation_matrix(price_history: pd.DataFrame) -> pd.DataFrame`**
```python
def compute_correlation_matrix(price_history: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  Long-format DataFrame — columns: ticker, date, adj_close
            (same format as price_history from build_ai_india_dataset)
    Process: Pivot to wide format (date × ticker), compute pct_change(),
             drop NaN rows, inner-align all tickers, compute Pearson correlation.
    Output: Square pd.DataFrame, index=tickers, columns=tickers, values in [-1, 1].
    """
```

**`compute_valuation_percentile(ticker: str, metric: str, lookback_years: int = 3) -> float | None`**
```python
def compute_valuation_percentile(
    ticker: str,
    metric: str,          # allowed values: 'pe', 'pb' — raises ValueError otherwise
    lookback_years: int = 3
) -> float | None:
    """
    Builds a historical series of the requested multiple by combining:
    - Quarterly EPS or book value per share from yfinance .quarterly_financials
      or .quarterly_balance_sheet
    - Quarterly average price from yfinance price history

    Returns percentile (0–100) of current trailing multiple within lookback window.
    Returns None if fewer than 4 quarters of financial data are available.

    Rate limiting: exponential backoff — 3 retries with delays of 1s, 2s, 4s.
    Caching: raw quarterly data saved to data/valuation_cache/<ticker>_<metric>.csv
             on first successful fetch. Cache reused if file is < 24 hours old.
             Cache directory created via Path.mkdir(parents=True, exist_ok=True)
             at the start of the function — caller does not need to create it.
    """
```

**`compute_efficient_frontier(price_history: pd.DataFrame, min_history_days: int = 252) -> dict`**
```python
def compute_efficient_frontier(
    price_history: pd.DataFrame,   # long-format: ticker, date, adj_close
    min_history_days: int = 252    # tickers with fewer days excluded to avoid singular covariance
) -> dict:
    """
    Two-step computation — Monte Carlo for visualization, scipy.optimize for labeled outputs.

    Step 1 — Filter: exclude tickers with < min_history_days of price data.

    Step 2 — Monte Carlo scatter (for chart):
    Draw exactly 5000 random weight vectors via np.random.dirichlet(alpha=np.ones(n_assets)).
    No portfolios are filtered out — N is always exactly 5000.
    Compute annualized return, volatility, Sharpe for each.

    Step 3 — Exact optimization (for labeled portfolio annotations):
    Use scipy.optimize.minimize with method='SLSQP'.
    Constraints: weights sum to 1, all weights in [0.0, 1.0] (long-only).
    - Minimum volatility: minimize portfolio variance.
    - Maximum Sharpe: minimize negative Sharpe ratio.
    Covariance matrix computed from daily returns of filtered tickers.
    If optimization fails (e.g., singular matrix), return None for the failed portfolio.

    Weight exclusion: weights < 0.02 set to 0 and remaining weights renormalized once.
    This exclusion is applied once — no iteration.
    Stats in `max_sharpe_stats` and `min_vol_stats` are RECOMPUTED from the renormalized
    weights after exclusion — not taken from the raw optimizer output. This ensures the
    stats shown in charts and reports are consistent with the displayed weight labels.

    Returns dict:
      'simulated':        np.ndarray shape (5000, 3) — columns [volatility, return, sharpe]
      'max_sharpe_weights': dict[str, float] | None
      'max_sharpe_stats':   dict{'return': float, 'volatility': float, 'sharpe': float} | None
      'min_vol_weights':    dict[str, float] | None
      'min_vol_stats':      dict{'return': float, 'volatility': float, 'sharpe': float} | None
      'tickers_used':       list[str]
    """
```

---

## Section 2: Reporting Layer (`src/reporting.py` + generated outputs)

### Entry Point
All report generation is triggered from `notebooks/01_ai_india_landscape.ipynb`. The notebook calls `reporting.py` functions after the analytics pipeline runs. The README key-findings auto-update is the final cell in that notebook.

### Analyst-Voice Prose Generation

**`generate_stock_narrative(row: pd.Series, corr_matrix: pd.DataFrame | None = None, top5_tickers: list[str] | None = None) -> str`**

The `row` is one row of the summary DataFrame. `corr_matrix` is the output of `compute_correlation_matrix()`. `top5_tickers` is the list of the 5 highest-Sharpe tickers. Both extra parameters default to None — the function degrades gracefully to no cross-ticker rules if they are not provided.

**Prose branching rules (evaluated in order, first match wins):**
1. `sharpe is not None AND sharpe > 1.0` AND `valuation_pe_percentile is not None AND valuation_pe_percentile > 80`: "Priced for perfection — the risk-adjusted case is strong but there is no margin of safety at current multiples."
2. `sharpe is not None AND sharpe > 1.0` AND `valuation_pe_percentile is not None AND valuation_pe_percentile < 30`: "Best of both worlds in this universe — strong risk-adjusted returns and still trading at a historical discount."
3. `return_1y is not None AND return_1y > 0.30` AND `sharpe is not None AND sharpe < 0.6`: "A high-conviction-only name. The return is real but so is the ride — Sortino of {sortino_str} confirms the downside is not being rewarded proportionally." where `sortino_str = f"{sortino:.2f}" if sortino is not None else "N/A"`
4. `corr_matrix is not None` AND any other top-5 ticker has correlation > 0.8 with this ticker: "Not a diversifier — {ticker} and {other} move together {corr:.0%} of the time. Owning both is effectively a single concentrated position."
5. `drawdown_recovery_days is None` AND `max_drawdown is not None AND max_drawdown < -0.25`: "Still {abs(max_drawdown):.1%} below its peak as of the last data date. The recovery timeline matters for anyone sizing this as a short-duration trade."
6. Default: neutral description using `segment`, `cagr_3y`, `sharpe`.

### Bull / Base / Bear Case Generation

**`generate_bull_base_bear(row: pd.Series) -> dict[str, str]`** — returns dict with keys `'bull'`, `'base'`, `'bear'`.

Branching logic keyed to data:

| Condition | Bull | Base | Bear |
|-----------|------|------|------|
| `sharpe > 1.0` AND `cagr_3y > 0.20` | "Sustained AI services re-rating drives multiple expansion on top of compounding earnings." | "Revenue growth tracks consensus; Sharpe remains above 1.0 as volatility stays contained." | "Global IT spend slowdown compresses multiples; name re-rates to sector average." |
| `valuation_pe_percentile > 75` | "Earnings growth accelerates, justifying premium multiple." | "Multiple stays elevated on execution; no meaningful re-rating in either direction." | "Any earnings miss triggers sharp de-rating from historically elevated levels." |
| `max_drawdown < -0.35` | "Full recovery and breakout to new highs as sector tailwind reasserts." | "Slow recovery; price recovers peak within 12 months but limited upside beyond." | "Drawdown deepens on further sector rotation or macro headwinds; no near-term catalyst." |
| Default | "AI adoption in India accelerates beyond consensus, driving upside to estimates." | "Company executes in line; stock tracks earnings growth." | "Macro or FX headwinds compress margins; limited multiple support at current levels." |

### New Structural Elements in Pitch Report

- `> **Key Takeaway:** [1–2 sentences]` blockquote at end of each major section
- Risk-adjusted ranking table: columns `Ticker`, `1Y Return`, `3Y CAGR`, `Sharpe`, `Sortino`, `Sharpe Rank`
- Correlation callout paragraph: top 3 most correlated pairs (value and concentration warning)
- Drawdown narrative paragraph: worst 3 names by `max_drawdown`, with dates and recovery status using `drawdown_start`, `drawdown_trough`, `drawdown_recovery_days`
- "Portfolio Construction" section: `compute_efficient_frontier` output — max Sharpe weights table, min vol weights table, one-paragraph rationale
- `valuation_pe_percentile` and `valuation_pb_percentile` columns added to universe table
- Bull / Base / Bear for top 5 names by Sharpe (from `generate_bull_base_bear()`)

### New Output Formats

**HTML report** (`reports/ai_india_thematic_pitch.html`)
- Generated by `reporting.py` from the same data as the markdown report
- Inline CSS only — no `<link href="http...">`, no `<script src="http...">`, no CDN references
- File size target: < 2MB
- Color-coded return cells: `#e6f4ea` (positive), `#fce8e6` (negative)
- Sticky table headers: `position: sticky; top: 0`
- Print-to-PDF CSS: `@media print { nav, .sidebar { display: none; } }`

**Auto-updated README key findings**

```python
def update_readme_key_findings(
    summary_df: pd.DataFrame,
    readme_path: str | Path
) -> None:
    """
    Reads README.md, replaces content between
    <!-- KEY_FINDINGS_START --> and <!-- KEY_FINDINGS_END --> markers,
    writes the updated file back to readme_path.
    Generates 4 opinionated data-driven callouts from summary_df.
    The placeholder values in the README skeleton are examples only —
    this function overwrites them on every run.
    """
```

**Per-stock PDF tearsheets**
- Generated for **top 5 names by Sharpe ratio**
- All five pages written in a single pass to one `matplotlib.backends.backend_pdf.PdfPages` object → `reports/tearsheets/top5_tearsheets_combined.pdf`
- Each page also saved individually by opening a separate `PdfPages` object per ticker → `reports/tearsheets/<TICKER>_tearsheet.pdf` (`.NS` stripped from filename)
- No PyPDF2 or pypdf dependency — both outputs produced by matplotlib's PDF backend only
- Layout per page (matplotlib figure, A4 at 150 dpi):
  - Header: company name, segment, generation date
  - 1Y price chart (line plot from `closes`, annotated with `drawdown_trough` marker)
  - Metrics table: 1Y return, 3Y CAGR, Sharpe, Sortino, PE, PB, PE percentile, max drawdown, dividend yield
  - 3-bullet analyst view from `generate_bull_base_bear()`
- **Font constraint:** Use only matplotlib default fonts (DejaVu). No custom or system fonts — avoids glyph-embedding issues on Windows.

---

## Section 3: Visuals Layer (`src/visuals.py`)

**All charts saved to `reports/figures/`.**

### Export Format Changes
- Every chart (existing and new) saves as both PNG and JPG
- JPG conversion after PNG save: `PIL.Image.open(png_path).convert('RGB').save(jpg_path, quality=92)`
- `Pillow` added explicitly to `requirements.txt` — do not assume transitive availability

### Naming — Avoid Collision with Existing Charts
The existing `ai_india_heatmap.png` is a **returns heatmap** (stocks × time periods). The new correlation chart is named `ai_india_correlation_heatmap.png/.jpg` — distinct name, no collision.

### Five New Charts / Outputs

**1. Sharpe/Return Scatter** (`ai_india_sharpe_return.png/.jpg`)
- X: annualized 3Y CAGR (fallback to `return_1y` if `cagr_3y` is None), Y: Sharpe ratio
- Bubble size: `market_cap` (normalized), Color: `segment`
- Ticker labels: use `adjustText` if available (`pip install adjustText`); if not installed, apply fixed offset `(+0.005, +0.02)` in axis units
- Add `adjustText` to `requirements.txt` as an optional install with a comment: `# optional: improves scatter label layout`
- Quadrant lines: vertical at `median(cagr_3y)`, horizontal at `sharpe = 1.0`
- Quadrant labels in `grey`, `fontsize=8`, italic

**2. Correlation Heatmap** (`ai_india_correlation_heatmap.png/.jpg`)
- Source: output of `compute_correlation_matrix(price_history)`
- Colormap: `RdBu_r` (blue = low, red = high correlation)
- Values annotated, rounded to 2 decimal places
- Tickers ordered by segment to cluster the large-cap IT names visually

**3. Drawdown Timeline** (`ai_india_drawdown_timeline.png/.jpg`)
- Rolling drawdown from peak for **top 8 names by Sharpe** (wider cut than tearsheet top 5 — shows more contrast between high- and low-quality names)
- Area shaded below zero via `fill_between`
- Max drawdown date annotated per line with a vertical dashed marker
- Legend placed outside plot area

**4. Efficient Frontier** (`ai_india_efficient_frontier.png/.jpg`)
- 5,000 Monte Carlo portfolios as scatter (x=volatility, y=return, color=Sharpe via colormap `viridis`)
- Individual stocks as grey diamond markers with ticker labels
- Max Sharpe portfolio: gold star (`*`), annotated with top 3 weight labels
- Min Vol portfolio: blue star (`*`), annotated with top 3 weight labels
- Colorbar for Sharpe scale on right side

**5. PDF Scorecard** (`ai_india_scorecard.pdf`)
- Extends existing PNG scorecard — same content, saved additionally as PDF via `matplotlib` PDF backend
- No new layout work; call `plt.savefig('ai_india_scorecard.pdf')` alongside existing PNG save

### Updated Visual Gallery
`reports/ai_india_visual_summary.md` regenerated to include all charts. Each image followed by a plain-English caption stating what to *look for* (not what the chart is), e.g.:
> *"Look for names in the top-right quadrant — high return AND high Sharpe. Those are the genuine risk-adjusted compounders, not just momentum plays."*

---

## Section 4: README Overhaul

Full rewrite. `<!-- KEY_FINDINGS_START -->` and `<!-- KEY_FINDINGS_END -->` markers are required in the skeleton — `reporting.py` overwrites content between them at notebook run time. The skeleton example values are clearly annotated as placeholders.

```markdown
# AI x India: Public Equity Landscape

> India's AI services sector is a structural multi-year opportunity.
> This is the investment case — built on live market data, not consensus.

<!-- KEY_FINDINGS_START -->
- **Best risk-adjusted name:** OFSS — 1.4 Sharpe, 31% 3Y CAGR, trading at PB discount to Infosys
- **Highest momentum:** Persistent Systems — top Sharpe and top 3Y CAGR in the universe
- **Watch for concentration:** TCS, Infosys, Wipro, HCL correlate at >0.85 — owning all four is one position
- **Valuation flag:** Netweb at 84th percentile of own PE history — the 103% 1Y return is priced in
<!-- KEY_FINDINGS_END -->
*(Values above are auto-generated placeholders — regenerated on each notebook run)*

![Sharpe/Return Scatter](reports/figures/ai_india_sharpe_return.png)

## What this is
- A live thematic pitch on 20+ Indian AI/tech equities, built on public market data
- A scoring engine that ranks names differently by client risk profile
- A full analytics suite: risk-adjusted metrics, portfolio optimization, valuation context

## Read the research
→ [Full thematic pitch (HTML)](reports/ai_india_thematic_pitch.html)
→ [Visual gallery](reports/ai_india_visual_summary.md)
→ [Top 5 tearsheets (PDF)](reports/tearsheets/top5_tearsheets_combined.pdf)

![AI x India Scorecard](reports/figures/ai_india_scorecard.png)

## Universe
[3-sentence description of the 20+ names, segments covered, and data source]

## Data & Methodology
- Live market data via yfinance at runtime — no hard-coded prices
- Risk-adjusted metrics: Sharpe and Sortino ratios, max drawdown with recovery tracking
- Portfolio construction via mean-variance optimization (scipy.optimize)
- Valuation context: each stock ranked against its own 3-year PE/PB history

## Setup & Usage
[existing content, unchanged]

## Disclaimer
[existing content, unchanged]
```

---

## Dependencies

Add to `requirements.txt` if not already present:
```
scipy>=1.11          # portfolio optimizer (scipy.optimize.minimize with SLSQP)
Pillow>=10.0         # JPG conversion — do NOT rely on transitive install
adjustText>=0.8      # optional: improves scatter label layout — falls back to fixed offset if absent
```

---

## File Outputs Summary

```
reports/
  ai_india_thematic_pitch.md              (existing, upgraded prose + new sections)
  ai_india_thematic_pitch.html            (new)
  ai_india_idea_engine_notes.md           (existing, minor prose upgrade)
  ai_india_visual_summary.md              (existing, upgraded captions + new charts)
  tearsheets/
    TCS_tearsheet.pdf                     (.NS stripped from filename)
    INFY_tearsheet.pdf
    [top-5-by-sharpe]_tearsheet.pdf
    top5_tearsheets_combined.pdf
  figures/
    ai_india_scorecard.png/.jpg/.pdf      (existing PNG + new JPG + new PDF)
    ai_india_theme_map.png/.jpg           (existing + JPG)
    ai_india_performance.png/.jpg         (existing + JPG)
    ai_india_returns_bar.png/.jpg         (existing + JPG)
    ai_india_valuation_scatter.png/.jpg   (existing + JPG)
    ai_india_heatmap.png/.jpg             (existing returns heatmap + JPG)
    ai_india_sharpe_return.png/.jpg       (new)
    ai_india_correlation_heatmap.png/.jpg (new — distinct from returns heatmap above)
    ai_india_drawdown_timeline.png/.jpg   (new)
    ai_india_efficient_frontier.png/.jpg  (new)
data/
  valuation_cache/
    TCS.NS_pe.csv                         (new — yfinance rate limit cache)
    TCS.NS_pb.csv                         (new)
    [all tickers]_[metric].csv
README.md                                 (full rewrite)
```

---

## Execution Order

1. **Clone repo** to local working directory
2. **Install dependencies:** `pip install -r requirements.txt` (confirms scipy, Pillow, adjustText present)
3. **Run `notebooks/01_ai_india_landscape.ipynb`** top-to-bottom:
   - Cell group 1: `data_loader.py` — pull live price data and fundamentals
   - Cell group 2: `analysis.py` — compute all metrics including new Sharpe, Sortino, drawdown duration, valuation percentiles, efficient frontier
   - Cell group 3: `visuals.py` — generate all charts (PNG + JPG + scorecard PDF)
   - Cell group 4: `reporting.py` — generate markdown and HTML reports, write tearsheets
   - **Final cell:** `reporting.py:update_readme_key_findings(stats, readme_path)` — update README. `readme_path` must be constructed relative to the notebook's `Path(__file__).resolve().parents[1]` or equivalently `Path.cwd().parent` if the notebook is in `notebooks/`. Use `Path(__file__).resolve().parents[1] / "README.md"` — do not hardcode an absolute path.
4. **Run `notebooks/02_ai_idea_engine.ipynb`** for the advisor scoring engine output (minor prose upgrade only; no new analytics dependencies)
5. **Verify outputs** against success criteria below

---

## Success Criteria

### Automated / machine-verifiable

1. `reports/ai_india_thematic_pitch.html` exists, is < 2MB, contains no `<link href="http` or `<script src="http` substrings
2. `reports/tearsheets/top5_tearsheets_combined.pdf` exists and is non-zero bytes. Exactly 5 individual tearsheet PDFs exist in `reports/tearsheets/` matching the pattern `*_tearsheet.pdf` (excluding the combined file). Individual filenames are runtime-determined by Sharpe rank — the combined file is the canonical verification artifact.
3. `sharpe` and `sortino` columns present in summary DataFrame with no NaN values for any ticker with >= 252 days of price history. `valuation_pe_percentile` and `valuation_pb_percentile` may be None/NaN for tickers with < 4 quarters of quarterly financials — this is expected behavior, not a defect.
4. All 10 chart files exist in `reports/figures/` in both PNG and JPG format (6 existing + 4 new = 10 pairs). Scorecard PDF is an additional output and does not count toward the 10-pair target.
5. `reports/figures/ai_india_efficient_frontier.png` is non-zero bytes and `compute_efficient_frontier()` returns a dict with non-None `max_sharpe_weights` for at least one successful optimization run
6. `README.md` contains `<!-- KEY_FINDINGS_START -->` and `<!-- KEY_FINDINGS_END -->` markers with non-empty content between them after a full notebook run

### Qualitative (manual review)

7. A finance/IB recruiter landing on the repo understands the thesis and sees a striking visual within 10 seconds (README hero image + key findings above the fold)
8. The HTML report reads as a professional research document — opinionated prose, structured sections, color-coded tables — without requiring any code execution
9. Each tearsheet reads like sell-side output: company header, price chart, metrics table, and a structured bull/base/bear analyst view
10. The efficient frontier chart clearly distinguishes the max Sharpe and min vol portfolios with annotated weight labels readable at normal screen resolution
