"""Factor scoring logic for the AI-augmented wealth idea engine."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Deterministic AI-purity mapping:
# - 0.90 to 1.00: AI infrastructure, analytics, or ad-tech platforms where AI/data is close to the product itself.
# - 0.75 to 0.89: Engineering and vertical software names with material AI exposure, but broader business lines.
# - 0.55 to 0.74: Diversified IT services and auto/industrial names where AI is meaningful but not the whole thesis.
SEGMENT_AI_PURITY = {
    "AI infrastructure": 1.00,
    "analytics platform": 0.95,
    "analytics/BPM": 0.90,
    "adtech/martech": 0.90,
    "digital engineering": 0.84,
    "ER&D": 0.82,
    "auto software": 0.80,
    "enterprise software": 0.78,
    "IT services": 0.66,
    "auto AI": 0.60,
}

AI_KEYWORD_BOOSTS = {
    "genai": 0.03,
    "machine learning": 0.03,
    "ai compute": 0.05,
    "gpu": 0.05,
    "adas": 0.03,
    "autonomous": 0.03,
    "analytics": 0.02,
}

ARCHETYPE_CONFIGS: dict[str, dict[str, Any]] = {
    "Growth-Seeking Tech Entrepreneur": {
        "description": "Leans into high-growth digital and AI exposure, accepts higher volatility, and can live with limited dividend support.",
        "growth_priority": "high",
        "risk_tolerance": "high",
        "dividend_preference": "low",
        "concentration_preference": "medium",
        "ai_purity_preference": "high",
        "weights": {
            "growth_score": 0.34,
            "momentum_score": 0.21,
            "volatility_score": 0.08,
            "yield_score": 0.05,
            "ai_purity_score": 0.32,
        },
    },
    "Balanced Professional (Core + Satellite)": {
        "description": "Needs a core-plus mix of durable large-cap exposure, acceptable volatility, and room for selective thematic names.",
        "growth_priority": "medium",
        "risk_tolerance": "medium",
        "dividend_preference": "medium",
        "concentration_preference": "low",
        "ai_purity_preference": "medium",
        "weights": {
            "growth_score": 0.24,
            "momentum_score": 0.16,
            "volatility_score": 0.24,
            "yield_score": 0.16,
            "ai_purity_score": 0.20,
        },
    },
    "Defensive Income-Focused Investor": {
        "description": "Prefers steadier companies, lower volatility, and visible cash-return characteristics over pure thematic intensity.",
        "growth_priority": "low",
        "risk_tolerance": "low",
        "dividend_preference": "high",
        "concentration_preference": "low",
        "ai_purity_preference": "medium",
        "weights": {
            "growth_score": 0.12,
            "momentum_score": 0.10,
            "volatility_score": 0.33,
            "yield_score": 0.30,
            "ai_purity_score": 0.15,
        },
    },
    "Aggressive Thematic Trader": {
        "description": "Optimizes for momentum and pure-play thematic exposure, with minimal concern for dividend yield and limited concern for volatility.",
        "growth_priority": "high",
        "risk_tolerance": "very high",
        "dividend_preference": "very low",
        "concentration_preference": "high",
        "ai_purity_preference": "very high",
        "weights": {
            "growth_score": 0.25,
            "momentum_score": 0.30,
            "volatility_score": 0.05,
            "yield_score": 0.00,
            "ai_purity_score": 0.40,
        },
    },
}


def derive_ai_purity_score(segment: str, notes: str) -> float:
    """Translate a company segment and business description into a deterministic AI-purity score."""

    base_score = SEGMENT_AI_PURITY.get(segment, 0.55)
    notes_lower = (notes or "").lower()
    keyword_boost = sum(boost for keyword, boost in AI_KEYWORD_BOOSTS.items() if keyword in notes_lower)
    return float(np.clip(base_score + keyword_boost, 0.0, 1.0))


def _winsorized_min_max(series: pd.Series, higher_is_better: bool = True, neutral: float = 0.50) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.full(len(series), neutral), index=series.index, dtype="float64")

    lower = valid.quantile(0.05)
    upper = valid.quantile(0.95)
    clipped = values.clip(lower=lower, upper=upper)
    if np.isclose(lower, upper):
        scaled = pd.Series(np.full(len(series), neutral), index=series.index, dtype="float64")
    else:
        scaled = (clipped - lower) / (upper - lower)

    if not higher_is_better:
        scaled = 1.0 - scaled
    return scaled.fillna(neutral).clip(lower=0.0, upper=1.0)


def history_penalty(history_years: float | None) -> float:
    """Cap the influence of short histories instead of inventing a full track record."""

    if history_years is None or pd.isna(history_years):
        return 0.60
    if history_years >= 3.0:
        return 1.00
    if history_years >= 2.0:
        return 0.90
    if history_years >= 1.0:
        return 0.75
    return 0.60


def compute_factor_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Create normalized factor scores used across all advisor archetypes."""

    scored = df.copy()
    growth_base = scored["cagr_3y"].combine_first(scored["return_1y"])
    momentum_base = 0.4 * pd.to_numeric(scored["momentum_6m"], errors="coerce") + 0.6 * pd.to_numeric(
        scored["momentum_12m"], errors="coerce"
    )

    scored["growth_score"] = _winsorized_min_max(growth_base, higher_is_better=True)
    scored["momentum_score"] = _winsorized_min_max(momentum_base, higher_is_better=True)
    scored["volatility_score"] = _winsorized_min_max(scored["annualized_volatility"], higher_is_better=False)
    scored["yield_score"] = _winsorized_min_max(scored["dividend_yield"], higher_is_better=True)
    scored["ai_purity_score"] = scored.apply(
        lambda row: derive_ai_purity_score(segment=str(row.get("segment", "")), notes=str(row.get("notes", ""))),
        axis=1,
    )
    scored["history_penalty"] = scored["history_years"].apply(history_penalty)
    return scored


def score_for_archetype(df: pd.DataFrame, archetype_config: dict[str, Any]) -> pd.DataFrame:
    """Score each stock for a specific wealth-client archetype."""

    scored = compute_factor_scores(df) if "growth_score" not in df.columns else df.copy()
    factor_weights = archetype_config["weights"]
    factor_scores = pd.DataFrame(
        {
            factor: pd.to_numeric(scored[factor], errors="coerce").fillna(0.50)
            for factor in factor_weights
        }
    )
    weight_total = float(sum(factor_weights.values()))
    normalized_weights = {factor: weight / weight_total for factor, weight in factor_weights.items()}

    scored["raw_score"] = sum(factor_scores[factor] * normalized_weights[factor] for factor in normalized_weights)
    scored["total_score"] = scored["raw_score"] * scored["history_penalty"]

    penalized = scored.loc[scored["history_penalty"] < 1.0, "ticker"].tolist()
    if penalized:
        LOGGER.info("Applied short-history penalty to: %s", ", ".join(penalized))

    ordered_columns = [
        "ticker",
        "name",
        "segment",
        "history_years",
        "return_1y",
        "cagr_3y",
        "momentum_6m",
        "momentum_12m",
        "annualized_volatility",
        "dividend_yield",
        "growth_score",
        "momentum_score",
        "volatility_score",
        "yield_score",
        "ai_purity_score",
        "history_penalty",
        "total_score",
    ]
    available_columns = [column for column in ordered_columns if column in scored.columns]
    return scored.sort_values("total_score", ascending=False)[available_columns].reset_index(drop=True)


def _metric_text(label: str, value: float | None, pct: bool = True) -> str | None:
    if value is None or pd.isna(value):
        return None
    if pct:
        return f"{label} {value:.1%}"
    return f"{label} {value:.2f}"


def build_rationale_text(row: pd.Series, archetype_name: str) -> str:
    """Generate a one-line advisor-friendly rationale using real computed metrics."""

    growth_text = _metric_text("3Y CAGR of", row.get("cagr_3y"))
    if growth_text is None:
        growth_text = _metric_text("1Y return of", row.get("return_1y"))

    momentum = row.get("momentum_12m")
    momentum_text = None
    if momentum is not None and not pd.isna(momentum):
        momentum_direction = "positive" if momentum >= 0 else "negative"
        momentum_text = f"{momentum_direction} 12M momentum ({momentum:.1%})"

    volatility = row.get("annualized_volatility")
    vol_score = row.get("volatility_score")
    volatility_style = "lower-volatility"
    if vol_score is not None and not pd.isna(vol_score):
        if vol_score < 0.33:
            volatility_style = "higher-volatility"
        elif vol_score < 0.67:
            volatility_style = "moderate-volatility"
    volatility_text = _metric_text(f"{volatility_style} profile at", volatility)

    yield_text = None
    dividend_yield = row.get("dividend_yield")
    if dividend_yield is not None and not pd.isna(dividend_yield) and dividend_yield > 0:
        yield_text = f"dividend yield of {dividend_yield:.1%}"

    ai_purity = row.get("ai_purity_score")
    if ai_purity is not None and not pd.isna(ai_purity):
        if ai_purity >= 0.85:
            ai_text = "very high AI-purity exposure"
        elif ai_purity >= 0.70:
            ai_text = "meaningful AI exposure"
        else:
            ai_text = "broader IT exposure with an AI angle"
    else:
        ai_text = "AI exposure that needs additional diligence"

    components = [growth_text, momentum_text, volatility_text, yield_text, ai_text]
    facts = ", ".join(component for component in components if component)
    return (
        f"{row.get('segment', 'Listed technology')} exposure with {facts}; "
        f"suitable for the {archetype_name.lower()} profile."
    )


def attach_rationales(ranked_df: pd.DataFrame, archetype_name: str) -> pd.DataFrame:
    """Append human-readable rationale text to a ranked recommendation table."""

    enriched = ranked_df.copy()
    enriched["rationale_text"] = enriched.apply(build_rationale_text, axis=1, archetype_name=archetype_name)
    return enriched
