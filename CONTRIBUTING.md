# Contributing to AI x India: Public Equity Landscape

Thanks for your interest in contributing. This is an open, opinionated research stack for tracking NSE/BSE-listed companies building, scaling, and monetizing AI and ML in India.

The goal is to keep the project:
- **Research-grade** — data-driven, documented, and reproducible
- **Modular** — contributors can extend one layer (data / analytics / app) without reworking others
- **Honest** — all values are computed at runtime from public APIs; nothing is hard-coded

---

## How to get started

1. **Fork** the repository and create your feature branch:
   ```
   git checkout -b feature/your-feature-name
   ```

2. **Set up** a local Python environment (Python 3.11+ recommended):

   **Windows (PowerShell)**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   **macOS / Linux**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run tests** to verify everything passes before making any changes:
   ```bash
   pytest tests/ -v
   ```

4. Make your changes, add tests, and open a Pull Request.

---

## Repository tour

```
data/           Universe CSV(s) and runtime caches (ai_india_universe.csv)
notebooks/      Analysis notebooks (run top-to-bottom)
src/            Python modules — analysis, data_loader, scoring, visuals, reporting, app
reports/        Generated markdown/HTML reports, tearsheets, and chart figures
tests/          pytest suite — analysis, reporting, visuals
```

If you are unsure where your change belongs, open an issue first and describe what you have in mind.

---

## Types of contributions

### Universe & data (`track:data`)
- Propose additions or removals from the AI/tech universe with a short AI-exposure justification
- Improve metadata: segment labels, notes, or add new columns (e.g. `ai_purity_override`)
- Add utilities for validating or ingesting new tickers and data sources

### Analytics & factors (`track:analytics`)
- New metrics or factors (benchmark-relative returns, information ratio, sector concentration)
- Enhancements to portfolio optimization (constraints, scenario portfolios, stress tests)
- Robustness checks and edge-case handling for existing metrics

### Reports & visuals
- Improvements to the thematic pitch or visual summaries
- New charts built on top of existing analysis functions

### Streamlit app & UX (`track:app`)
- New views, interactive controls, scenario tools
- Better layout, download options, dark mode

### Engineering & infra (`track:infra`)
- GitHub Actions CI for automated test runs
- Packaging / configuration improvements
- Improved test coverage and tooling

---

## Coding style

- Use **type hints** where practical
- Prefer **pure functions** — keep computation separate from I/O
- Keep **logging** informative but not noisy (`logging.getLogger(__name__)`)
- Follow existing patterns in `analysis.py`, `scoring.py`, `reporting.py`, and `tests/`
- For notebooks: keep them **deterministic** and runnable top-to-bottom with no hard-coded values

---

## Tests

- All non-trivial changes should include tests in `tests/`
- Tests should use synthetic data — **no live API calls in tests**
- Run `pytest tests/ -v` locally and confirm it passes before opening a PR
- If adding a new factor or scoring change, add tests that lock in the expected behaviour

---

## Pull Request process

1. Make sure your branch is up to date with `main`
2. Run `pytest tests/ -v` locally
3. Open a PR with:
   - A clear title
   - A short description of **what** changed and **why**
   - Screenshots for app/visual changes
4. Link the PR to an existing issue where applicable
5. Keep PRs small and focused — they are much easier to review

---

## Proposing universe changes

When proposing new tickers or removing existing ones, include:
- Ticker and exchange (e.g. `XYZTECH.NS`)
- A short justification of the company's AI/ML exposure
- A public reference (company disclosure, product page, or analyst note)

---

## Code of Conduct

By participating, you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
Respectful, evidence-based discussion is strongly encouraged, especially when debating universe composition or factor design.
