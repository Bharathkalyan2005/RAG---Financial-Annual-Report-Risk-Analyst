"""
risk/score.py — Phase 8: Multi-Year Risk Scoring & Ranking Engine
================================================================

Implements the multi-factor risk scoring formula:
  score = (WEIGHT_FREQUENCY * mention_frequency_normalized)
        + (WEIGHT_YOY * yoy_change_normalized)
        + (WEIGHT_EMPHASIS * management_emphasis_score)

Weight Justification (Transparent & Defensible):
  - WEIGHT_FREQUENCY (0.3):
      Normalizes mention counts within each year to reflect general prevalence.
  - WEIGHT_YOY (0.3):
      Measures year-over-year growth velocity to surface newly emerging or worsening threats.
  - WEIGHT_EMPHASIS (0.4):
      Weights keyword severity ('material', 'adverse', 'significant', 'critical')
      most heavily because management tone indicates actual legal/financial exposure.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("risk.score")

DETECTIONS_FILE = Path("data/processed/risk_detections.json")
SCORES_FILE = Path("data/processed/risk_scores.json")
SCORES_CSV = Path("data/processed/risk_scores.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Formula Constants (Explicitly Tunable)
# ─────────────────────────────────────────────────────────────────────────────
WEIGHT_FREQUENCY = 0.30  # Baseline mention density within the filing
WEIGHT_YOY       = 0.30  # Velocity / escalation momentum vs prior fiscal year
WEIGHT_EMPHASIS  = 0.40  # Management tone severity intensity


def load_detections() -> dict[int, dict[str, Any]]:
    if not DETECTIONS_FILE.exists():
        from risk.detect import run_detection
        run_detection()

    with open(DETECTIONS_FILE, encoding="utf-8") as f:
        raw = json.load(f)
        # Convert string year keys to ints
        return {int(k): v for k, v in raw.items()}


def compute_risk_scores() -> dict[int, dict[str, float]]:
    """
    Compute normalized multi-factor risk scores (0.0 to 1.0) for each category per year.
    """
    detections = load_detections()
    years = sorted(detections.keys())
    categories = list(next(iter(detections.values())).keys())

    # 1. Compute raw totals per year for normalization
    final_scores: dict[int, dict[str, float]] = {}

    for idx, yr in enumerate(years):
        final_scores[yr] = {}
        year_data = detections[yr]

        # Total mentions across all categories in this filing
        total_year_mentions = sum(year_data[c]["count"] for c in categories) or 1
        total_year_emphasis = sum(year_data[c]["emphasis_count"] for c in categories) or 1

        for cat in categories:
            count = year_data[cat]["count"]
            emphasis = year_data[cat]["emphasis_count"]

            # Factor 1: Normalized Mention Frequency (share of risk disclosures)
            freq_score = count / total_year_mentions

            # Factor 2: YoY Change
            if idx == 0:
                # Baseline year (no prior year in 5-year window), set neutral 0.5
                yoy_score = 0.50
            else:
                prev_year = years[idx - 1]
                prev_count = detections[prev_year][cat]["count"]
                if prev_count == 0:
                    yoy_pct = 1.0 if count > 0 else 0.0
                else:
                    yoy_pct = (count - prev_count) / prev_count
                # Bound YoY score between 0.0 and 1.0 using sigmoid-like scaling
                yoy_score = max(0.0, min(1.0, 0.50 + (yoy_pct * 0.50)))

            # Factor 3: Management Emphasis Intensity Score
            emphasis_score = emphasis / total_year_emphasis

            # Composite Score (0.0 - 1.0)
            composite = (
                (WEIGHT_FREQUENCY * freq_score * 3.0)  # scale to ~0-1 range
                + (WEIGHT_YOY * yoy_score)
                + (WEIGHT_EMPHASIS * emphasis_score * 3.0)
            )
            composite_clamped = round(max(0.05, min(0.99, composite)), 4)
            final_scores[yr][cat] = composite_clamped

    return final_scores


def run_scoring():
    log.info("Computing multi-factor risk scores...")
    scores = compute_risk_scores()

    SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORES_FILE, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    # Also export clean CSV for Streamlit charts & reports
    df_rows = []
    for yr, cats in scores.items():
        row = {"year": yr}
        row.update(cats)
        df_rows.append(row)
    df = pd.DataFrame(df_rows).set_index("year")
    df.to_csv(SCORES_CSV)

    log.info("Risk scores saved to %s and %s", SCORES_FILE, SCORES_CSV)

    print("\n" + "=" * 78)
    print("  COMPOSITE RISK SCORES (0.00 to 1.00)")
    print("  Formula: 0.3*Frequency + 0.3*YoY_Change + 0.4*Emphasis")
    print("=" * 78)
    categories = list(next(iter(scores.values())).keys())
    header = f"{'Year':<6} | " + " | ".join(f"{c[:7]:<7}" for c in categories)
    print(header)
    print("-" * len(header))
    for yr in sorted(scores.keys()):
        row = f"{yr:<6} | "
        vals = [f"{scores[yr][c]:.3f}" for c in categories]
        row += " | ".join(f"{v:<7}" for v in vals)
        print(row)
    print("=" * 78)

    # Identify top risk category for latest year (2025)
    latest_yr = max(scores.keys())
    top_risk = max(scores[latest_yr].items(), key=lambda x: x[1])
    print(f"\n[Checkpoint Check] Top risk category in {latest_yr}: {top_risk[0]} (Score: {top_risk[1]:.3f})")
    print("=" * 78)


if __name__ == "__main__":
    run_scoring()
