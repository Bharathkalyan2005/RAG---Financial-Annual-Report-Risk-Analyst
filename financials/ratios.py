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
    company_str = "AAPL"
    if isinstance(company, str):
        company_str = company.split()[0].upper()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT year, metric, value FROM financials WHERE company = :company ORDER BY year, metric"),
            conn,
            params={"company": company_str},
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

def compute_all_ratios(
    company_or_df: str | pd.DataFrame = "AAPL",
    company: Optional[str] = None,
) -> pd.DataFrame:
    """Compute all 10 standard ratios across all 5 years."""
    if isinstance(company_or_df, pd.DataFrame):
        df = company_or_df
    elif company is not None and isinstance(company, str):
        df = load_financials_df(company)
    else:
        df = load_financials_df(str(company_or_df))
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


def get_metric_source_chunks(
    year: int,
    metric_names: list[str],
    company: str = "AAPL",
) -> list[dict[str, Any]]:
    """Fetch source chunk records and real table text from database for calculation citations."""
    company_str = company.split()[0].upper() if isinstance(company, str) else "AAPL"
    with engine.connect() as conn:
        fin_rows = conn.execute(
            text("""
                SELECT metric, value, source_chunk_id
                FROM financials
                WHERE year = :year AND company = :company AND metric = ANY(:metrics)
            """),
            {"year": year, "company": company_str, "metrics": metric_names},
        ).mappings().all()

        chunk_ids = list({r["source_chunk_id"] for r in fin_rows if r["source_chunk_id"] is not None})
        if not chunk_ids:
            return [{
                "year": year,
                "section": "Item 8. Financial Statements",
                "position_id": 1,
                "text": f"Computed from financials table: {', '.join(metric_names)}",
                "source_chunk_id": None,
            }]

        chunk_rows = conn.execute(
            text("""
                SELECT id, year, section, COALESCE(position_id, page, 1) AS position_id,
                       text, display_text, raw_text
                FROM chunks
                WHERE id = ANY(:chunk_ids)
                ORDER BY id
            """),
            {"chunk_ids": chunk_ids},
        ).mappings().all()

    citations = []
    for c in chunk_rows:
        chunk_text = c["display_text"] or c["raw_text"] or c["text"] or ""
        citations.append({
            "source_chunk_id": c["id"],
            "chunk_id": c["id"],
            "year": c["year"],
            "section": c["section"],
            "position_id": c["position_id"],
            "text": chunk_text,
        })
    return citations


def compute_query_metric(
    question: str,
    year: Optional[int] = None,
    company: Optional[str] = "AAPL",
) -> dict[str, Any]:
    """Helper function called by generation/router.py to answer calculation queries with real table chunk citations."""
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
        citations = get_metric_source_chunks(target_year, ["revenue", "cost_of_revenue"], company=company or "AAPL")
        return {
            "answer": f"Apple's Gross Margin for FY{target_year} was {val:.2f}%, calculated as (Revenue - Cost of Revenue) / Revenue.",
            "value": round(val, 2),
            "metric": "gross_margin",
            "year": target_year,
            "citations": citations,
            "confidence": 1.0,
            "computed_via": "pandas",
        }
    if "operating margin" in q_lower:
        val = yr_ratios["operating_margin"] * 100
        citations = get_metric_source_chunks(target_year, ["operating_income", "revenue"], company=company or "AAPL")
        return {
            "answer": f"Apple's Operating Margin for FY{target_year} was {val:.2f}%, calculated as Operating Income / Revenue.",
            "value": round(val, 2),
            "metric": "operating_margin",
            "year": target_year,
            "citations": citations,
            "confidence": 1.0,
            "computed_via": "pandas",
        }
    if "net margin" in q_lower or "profit margin" in q_lower:
        val = yr_ratios["net_margin"] * 100
        citations = get_metric_source_chunks(target_year, ["net_income", "revenue"], company=company or "AAPL")
        return {
            "answer": f"Apple's Net Profit Margin for FY{target_year} was {val:.2f}%, calculated as Net Income / Revenue.",
            "value": round(val, 2),
            "metric": "net_margin",
            "year": target_year,
            "citations": citations,
            "confidence": 1.0,
            "computed_via": "pandas",
        }
    if "roa" in q_lower or "return on assets" in q_lower:
        val = yr_ratios["roa"] * 100
        citations = get_metric_source_chunks(target_year, ["net_income", "total_assets"], company=company or "AAPL")
        return {
            "answer": f"Apple's ROA for FY{target_year} was {val:.2f}%, calculated as Net Income / Total Assets.",
            "value": round(val, 2),
            "metric": "roa",
            "year": target_year,
            "citations": citations,
            "confidence": 1.0,
            "computed_via": "pandas",
        }
    if "roe" in q_lower or "return on equity" in q_lower:
        val = yr_ratios["roe"] * 100
        citations = get_metric_source_chunks(target_year, ["net_income", "total_equity"], company=company or "AAPL")
        return {
            "answer": f"Apple's ROE for FY{target_year} was {val:.2f}%, calculated as Net Income / Total Shareholders Equity.",
            "value": round(val, 2),
            "metric": "roe",
            "year": target_year,
            "citations": citations,
            "confidence": 1.0,
            "computed_via": "pandas",
        }
    if "debt to equity" in q_lower or "debt-to-equity" in q_lower:
        val = yr_ratios["debt_to_equity"]
        citations = get_metric_source_chunks(target_year, ["total_debt", "total_equity"], company=company or "AAPL")
        return {
            "answer": f"Apple's Debt-to-Equity ratio for FY{target_year} was {val:.2f}x, calculated as Total Debt / Total Shareholders Equity.",
            "value": round(val, 2),
            "metric": "debt_to_equity",
            "year": target_year,
            "citations": citations,
            "confidence": 1.0,
            "computed_via": "pandas",
        }
    if "debt to assets" in q_lower or "debt-to-assets" in q_lower:
        val = yr_ratios["debt_to_assets"]
        citations = get_metric_source_chunks(target_year, ["total_debt", "total_assets"], company=company or "AAPL")
        return {
            "answer": f"Apple's Debt-to-Assets ratio for FY{target_year} was {val:.2f}, calculated as Total Debt / Total Assets.",
            "value": round(val, 2),
            "metric": "debt_to_assets",
            "year": target_year,
            "citations": citations,
            "confidence": 1.0,
            "computed_via": "pandas",
        }
    if "interest coverage" in q_lower:
        val = yr_ratios["interest_coverage"]
        citations = get_metric_source_chunks(target_year, ["operating_income", "interest_expense"], company=company or "AAPL")
        return {
            "answer": f"Apple's Interest Coverage for FY{target_year} was {val:.1f}x, calculated as Operating Income / Interest Expense.",
            "value": round(val, 1),
            "metric": "interest_coverage",
            "year": target_year,
            "citations": citations,
            "confidence": 1.0,
            "computed_via": "pandas",
        }
    if "current ratio" in q_lower:
        val = yr_ratios["current_ratio"]
        citations = get_metric_source_chunks(target_year, ["current_assets", "current_liabilities"], company=company or "AAPL")
        return {
            "answer": f"Apple's Current Ratio for FY{target_year} was {val:.2f}, calculated as Current Assets / Current Liabilities.",
            "value": round(val, 2),
            "metric": "current_ratio",
            "year": target_year,
            "citations": citations,
            "confidence": 1.0,
            "computed_via": "pandas",
        }

    if "operating income" in q_lower:
        val = yr_fin.get("operating_income", 0.0)
        citations = get_metric_source_chunks(target_year, ["operating_income"], company=company or "AAPL")
        return {
            "answer": f"Apple's Operating Income for FY{target_year} was ${val:,.1f} million.",
            "value": val,
            "metric": "operating_income",
            "year": target_year,
            "citations": citations,
            "confidence": 1.0,
            "computed_via": "pandas",
        }

    if "total income" in q_lower:
        net_inc = yr_fin.get("net_income", 0.0)
        op_inc = yr_fin.get("operating_income", 0.0)
        rev = yr_fin.get("revenue", 0.0)
        citations = get_metric_source_chunks(target_year, ["net_income", "operating_income", "revenue"], company=company or "AAPL")
        return {
            "answer": (
                f"In US GAAP 10-K filings, Apple does not report a single line item titled 'Total Income'. "
                f"Depending on context, the official earnings metrics for FY{target_year} are:\n"
                f"- **Net Income (Bottom-line profit)**: **${net_inc:,.1f} million**\n"
                f"- **Operating Income (Operating profit)**: **${op_inc:,.1f} million**\n"
                f"- **Total Net Sales / Revenue (Top-line income)**: **${rev:,.1f} million**."
            ),
            "value": net_inc,
            "metric": "net_income",
            "year": target_year,
            "citations": citations,
            "confidence": 1.0,
            "computed_via": "pandas",
        }

    # Fallback to direct raw metric
    for m in ["revenue", "net_income", "total_debt", "total_assets", "total_equity"]:
        if m.replace("_", " ") in q_lower:
            val = yr_fin.get(m, 0.0)
            citations = get_metric_source_chunks(target_year, [m], company=company or "AAPL")
            return {
                "answer": f"Apple's {m.replace('_', ' ')} for FY{target_year} was ${val:,.1f} million.",
                "value": val,
                "metric": m,
                "year": target_year,
                "citations": citations,
                "confidence": 1.0,
                "computed_via": "pandas",
            }

    citations = get_metric_source_chunks(target_year, ["revenue", "operating_income", "total_debt", "total_equity"], company=company or "AAPL")
    return {
        "answer": f"Calculated ratios for FY{target_year}: Gross Margin={yr_ratios['gross_margin']*100:.1f}%, Operating Margin={yr_ratios['operating_margin']*100:.1f}%, Debt/Equity={yr_ratios['debt_to_equity']:.2f}x.",
        "year": target_year,
        "ratios": yr_ratios,
        "citations": citations,
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
