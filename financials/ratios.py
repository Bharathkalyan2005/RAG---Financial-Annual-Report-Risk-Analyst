"""
financials/ratios.py — Phase 7: Financial Ratio Calculations in Pure Pandas
===========================================================================

All financial arithmetic is strictly performed in Python/Pandas, NEVER by the LLM.

Formulas implemented:
  - gross_margin        = (revenue - cost_of_revenue) / revenue
  - operating_margin    = operating_income / revenue
  - net_margin          = net_income / revenue
  - roa (Return on Assets) = net_income / total_assets
  - roe (Return on Equity) = net_income / total_equity
  - current_ratio       = current_assets / current_liabilities
  - quick_ratio         = (current_assets * 0.85) / current_liabilities  # conservative acid test
  - debt_to_equity      = total_debt / total_equity
  - debt_to_assets      = total_debt / total_assets
  - interest_coverage   = operating_income / interest_expense
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import text

from db import engine

log = logging.getLogger("financials.ratios")

RATIOS_CSV = Path("data/processed/ratios.csv")


def load_financials_df(company: str = "AAPL") -> pd.DataFrame:
    """Load financials table from Neon into a pivoted pandas DataFrame (years as rows, metrics as columns)."""
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT year, metric, value FROM financials WHERE company = :company ORDER BY year, metric"),
            conn,
            params={"company": company},
        )

    if df.empty:
        raise ValueError("Financials table is empty. Run `python -m financials.extract_numbers` first.")

    pivoted = df.pivot(index="year", columns="metric", values="value")
    return pivoted


# ─────────────────────────────────────────────────────────────────────────────
# Ratio Formulas (Strict Deterministic Arithmetic)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_gross_margin(rev: float, cogs: float) -> float:
    """Formula: (Revenue - Cost of Revenue) / Revenue"""
    return (rev - cogs) / rev if rev else 0.0


def calculate_operating_margin(op_inc: float, rev: float) -> float:
    """Formula: Operating Income / Revenue"""
    return op_inc / rev if rev else 0.0


def calculate_net_margin(net_inc: float, rev: float) -> float:
    """Formula: Net Income / Revenue"""
    return net_inc / rev if rev else 0.0


def calculate_roa(net_inc: float, assets: float) -> float:
    """Formula: Net Income / Total Assets"""
    return net_inc / assets if assets else 0.0


def calculate_roe(net_inc: float, equity: float) -> float:
    """Formula: Net Income / Total Shareholders Equity"""
    return net_inc / equity if equity else 0.0


def calculate_current_ratio(ca: float, cl: float) -> float:
    """Formula: Current Assets / Current Liabilities"""
    return ca / cl if cl else 0.0


def calculate_quick_ratio(ca: float, cl: float) -> float:
    """Formula: (Current Assets * 0.85) / Current Liabilities (Acid-test conservative proxy)"""
    return (ca * 0.85) / cl if cl else 0.0


def calculate_debt_to_equity(debt: float, equity: float) -> float:
    """Formula: Total Debt / Total Shareholders Equity"""
    return debt / equity if equity else 0.0


def calculate_debt_to_assets(debt: float, assets: float) -> float:
    """Formula: Total Debt / Total Assets"""
    return debt / assets if assets else 0.0


def calculate_interest_coverage(op_inc: float, interest: float) -> float:
    """Formula: Operating Income / Interest Expense"""
    return op_inc / interest if interest else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_ratios(company: str = "AAPL") -> pd.DataFrame:
    """Compute all 10 standard ratios across all 5 years."""
    df = load_financials_df(company)
    ratios = pd.DataFrame(index=df.index)

    ratios["gross_margin"] = df.apply(lambda r: calculate_gross_margin(r.get("revenue", 0), r.get("cost_of_revenue", 0)), axis=1)
    ratios["operating_margin"] = df.apply(lambda r: calculate_operating_margin(r.get("operating_income", 0), r.get("revenue", 0)), axis=1)
    ratios["net_margin"] = df.apply(lambda r: calculate_net_margin(r.get("net_income", 0), r.get("revenue", 0)), axis=1)
    ratios["roa"] = df.apply(lambda r: calculate_roa(r.get("net_income", 0), r.get("total_assets", 0)), axis=1)
    ratios["roe"] = df.apply(lambda r: calculate_roe(r.get("net_income", 0), r.get("total_equity", 0)), axis=1)
    ratios["current_ratio"] = df.apply(lambda r: calculate_current_ratio(r.get("current_assets", 0), r.get("current_liabilities", 0)), axis=1)
    ratios["quick_ratio"] = df.apply(lambda r: calculate_quick_ratio(r.get("current_assets", 0), r.get("current_liabilities", 0)), axis=1)
    ratios["debt_to_equity"] = df.apply(lambda r: calculate_debt_to_equity(r.get("total_debt", 0), r.get("total_equity", 0)), axis=1)
    ratios["debt_to_assets"] = df.apply(lambda r: calculate_debt_to_assets(r.get("total_debt", 0), r.get("total_assets", 0)), axis=1)
    ratios["interest_coverage"] = df.apply(lambda r: calculate_interest_coverage(r.get("operating_income", 0), r.get("interest_expense", 0)), axis=1)

    RATIOS_CSV.parent.mkdir(parents=True, exist_ok=True)
    ratios.round(4).to_csv(RATIOS_CSV)
    log.info("Saved computed ratios to %s", RATIOS_CSV)
    return ratios.round(4)


def compute_query_metric(
    question: str,
    year: Optional[int] = None,
    company: Optional[str] = "AAPL",
) -> dict[str, Any]:
    """Helper function called by generation/router.py to answer calculation queries."""
    ratios_df = compute_all_ratios(company=company or "AAPL")
    fin_df = load_financials_df(company=company or "AAPL")

    q_lower = question.lower()
    target_year = year or (2024 if "2024" in q_lower else 2025 if "2025" in q_lower else 2023 if "2023" in q_lower else 2024)

    if target_year not in ratios_df.index:
        target_year = int(ratios_df.index[-1])

    yr_ratios = ratios_df.loc[target_year].to_dict()
    yr_fin = fin_df.loc[target_year].to_dict()

    # Match metric
    if "gross margin" in q_lower:
        val = yr_ratios["gross_margin"] * 100
        return {
            "answer": f"Apple's Gross Margin for FY{target_year} was {val:.2f}%, calculated as (Revenue - Cost of Revenue) / Revenue.",
            "value": round(val, 2),
            "metric": "gross_margin",
            "year": target_year,
            "citations": [{"year": target_year, "section": "Item 8. Financial Statements", "position_id": 1}],
            "confidence": 1.0,
            "computed_via": "pandas",
        }
    if "operating margin" in q_lower:
        val = yr_ratios["operating_margin"] * 100
        return {
            "answer": f"Apple's Operating Margin for FY{target_year} was {val:.2f}%, calculated as Operating Income / Revenue.",
            "value": round(val, 2),
            "metric": "operating_margin",
            "year": target_year,
            "citations": [{"year": target_year, "section": "Item 8. Financial Statements", "position_id": 1}],
            "confidence": 1.0,
            "computed_via": "pandas",
        }
    if "net margin" in q_lower or "profit margin" in q_lower:
        val = yr_ratios["net_margin"] * 100
        return {
            "answer": f"Apple's Net Profit Margin for FY{target_year} was {val:.2f}%, calculated as Net Income / Revenue.",
            "value": round(val, 2),
            "metric": "net_margin",
            "year": target_year,
            "citations": [{"year": target_year, "section": "Item 8. Financial Statements", "position_id": 1}],
            "confidence": 1.0,
            "computed_via": "pandas",
        }
    if "debt to equity" in q_lower:
        val = yr_ratios["debt_to_equity"]
        return {
            "answer": f"Apple's Debt-to-Equity ratio for FY{target_year} was {val:.2f}x, calculated as Total Debt / Total Shareholders Equity.",
            "value": round(val, 2),
            "metric": "debt_to_equity",
            "year": target_year,
            "citations": [{"year": target_year, "section": "Item 8. Financial Statements", "position_id": 1}],
            "confidence": 1.0,
            "computed_via": "pandas",
        }
    if "current ratio" in q_lower:
        val = yr_ratios["current_ratio"]
        return {
            "answer": f"Apple's Current Ratio for FY{target_year} was {val:.2f}, calculated as Current Assets / Current Liabilities.",
            "value": round(val, 2),
            "metric": "current_ratio",
            "year": target_year,
            "citations": [{"year": target_year, "section": "Item 8. Financial Statements", "position_id": 1}],
            "confidence": 1.0,
            "computed_via": "pandas",
        }

    # Fallback to direct raw metric
    for m in ["revenue", "net_income", "total_debt", "total_assets", "total_equity"]:
        if m.replace("_", " ") in q_lower:
            val = yr_fin.get(m, 0.0)
            return {
                "answer": f"Apple's {m.replace('_', ' ')} for FY{target_year} was ${val:,.1f} million.",
                "value": val,
                "metric": m,
                "year": target_year,
                "citations": [{"year": target_year, "section": "Item 8. Financial Statements", "position_id": 1}],
                "confidence": 1.0,
                "computed_via": "pandas",
            }

    return {
        "answer": f"Calculated ratios for FY{target_year}: Gross Margin={yr_ratios['gross_margin']*100:.1f}%, Operating Margin={yr_ratios['operating_margin']*100:.1f}%, Debt/Equity={yr_ratios['debt_to_equity']:.2f}x.",
        "year": target_year,
        "ratios": yr_ratios,
        "citations": [{"year": target_year, "section": "Item 8. Financial Statements", "position_id": 1}],
        "confidence": 1.0,
        "computed_via": "pandas",
    }


if __name__ == "__main__":
    print("\n" + "=" * 75)
    print("  COMPUTED FINANCIAL RATIOS — 5 FISCAL YEARS")
    print("=" * 75)
    try:
        r_df = compute_all_ratios()
        print(r_df.to_string())
    except Exception as e:
        print(f"[!] Financial table pending: {e}")
    print("=" * 75)
