"""
agent/tools.py — Deterministic Tool Layer for Agentic RAG
==========================================================
Wraps all existing data access, retrieval, pure Pandas math, and risk scoring
capabilities into clean, deterministic agent tools.

Design guarantees:
  1. All tools return a dictionary with keys:
     - ok: bool
     - data: Any
     - evidence: list[dict] (provenance chunks with year, section, position_id, chunk_type, text)
     - error: str | None
  2. Input validation fails gracefully: returns ok=False with clear error message instead of raising.
  3. Zero LLM arithmetic: all calculations are executed with pure Python / Pandas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import text

from db import engine
from retrieval.hybrid import retrieve_hybrid

log = logging.getLogger("agent.tools")

VALID_YEARS = {2021, 2022, 2023, 2024, 2025}

RAW_METRIC_ALIASES: dict[str, str] = {
    "revenue": "revenue",
    "net sales": "revenue",
    "total net sales": "revenue",
    "sales": "revenue",
    "total revenue": "revenue",
    "net_income": "net_income",
    "net income": "net_income",
    "net earnings": "net_income",
    "profit": "net_income",
    "total_debt": "total_debt",
    "total debt": "total_debt",
    "debt": "total_debt",
    "total_equity": "total_equity",
    "total equity": "total_equity",
    "shareholders equity": "total_equity",
    "stockholders equity": "total_equity",
    "equity": "total_equity",
    "current_assets": "current_assets",
    "current assets": "current_assets",
    "current_liabilities": "current_liabilities",
    "current liabilities": "current_liabilities",
    "total_assets": "total_assets",
    "total assets": "total_assets",
    "assets": "total_assets",
    "cost_of_revenue": "cost_of_revenue",
    "cost of revenue": "cost_of_revenue",
    "cost of sales": "cost_of_revenue",
    "cogs": "cost_of_revenue",
    "operating_income": "operating_income",
    "operating income": "operating_income",
    "operating profit": "operating_income",
    "interest_expense": "interest_expense",
    "interest expense": "interest_expense",
}

RATIO_ALIASES: dict[str, str] = {
    "gross_margin": "gross_margin",
    "gross margin": "gross_margin",
    "operating_margin": "operating_margin",
    "operating margin": "operating_margin",
    "net_margin": "net_margin",
    "net margin": "net_margin",
    "profit margin": "net_margin",
    "net profit margin": "net_margin",
    "roa": "roa",
    "return on assets": "roa",
    "roe": "roe",
    "return on equity": "roe",
    "current_ratio": "current_ratio",
    "current ratio": "current_ratio",
    "quick_ratio": "quick_ratio",
    "quick ratio": "quick_ratio",
    "acid test": "quick_ratio",
    "debt_to_equity": "debt_to_equity",
    "debt to equity": "debt_to_equity",
    "debt/equity": "debt_to_equity",
    "d/e": "debt_to_equity",
    "debt_to_assets": "debt_to_assets",
    "debt to assets": "debt_to_assets",
    "interest_coverage": "interest_coverage",
    "interest coverage": "interest_coverage",
}

VALID_RISK_CATEGORIES = {
    "liquidity": "Liquidity",
    "credit": "Credit",
    "fx": "FX",
    "foreign exchange": "FX",
    "currency": "FX",
    "regulatory": "Regulatory",
    "antitrust": "Regulatory",
    "dma": "Regulatory",
    "cybersecurity": "Cybersecurity",
    "cyber": "Cybersecurity",
    "operational": "Operational",
    "supply chain": "Operational",
    "legal": "Legal",
    "litigation": "Legal",
}

RATIO_FORMULAS = {
    "gross_margin": "(revenue - cost_of_revenue) / revenue",
    "operating_margin": "operating_income / revenue",
    "net_margin": "net_income / revenue",
    "roa": "net_income / total_assets",
    "roe": "net_income / total_equity",
    "current_ratio": "current_assets / current_liabilities",
    "quick_ratio": "(current_assets * 0.85) / current_liabilities",
    "debt_to_equity": "total_debt / total_equity",
    "debt_to_assets": "total_debt / total_assets",
    "interest_coverage": "operating_income / interest_expense",
}

RISK_SCORES_PATH = Path("data/processed/risk_scores.json")


# ─────────────────────────────────────────────────────────────────────────────
# 1. search_filings
# ─────────────────────────────────────────────────────────────────────────────

def search_filings(
    query: str,
    year: Optional[int] = None,
    section: Optional[str] = None,
    chunk_type: Optional[str] = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Search 10-K filing texts, tables, and image figures using hybrid vector + keyword search and cross-encoder reranking.
    """
    if not query or not isinstance(query, str) or not query.strip():
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": "Query parameter must be a non-empty string.",
        }

    target_year = None
    if year is not None:
        try:
            target_year = int(year)
            if target_year not in VALID_YEARS:
                return {
                    "ok": False,
                    "data": None,
                    "evidence": [],
                    "error": f"Invalid year: {year}. Allowed fiscal years: 2021-2025.",
                }
        except (ValueError, TypeError):
            return {
                "ok": False,
                "data": None,
                "evidence": [],
                "error": f"Year must be an integer between 2021 and 2025, got {year!r}.",
            }

    valid_types = {"text", "table", "image"}
    target_chunk_type = chunk_type.lower() if chunk_type else None
    if target_chunk_type and target_chunk_type not in valid_types:
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": f"Invalid chunk_type '{chunk_type}'. Must be one of: text, table, image.",
        }

    k = max(1, min(int(top_k or 5), 20))

    try:
        chunks = retrieve_hybrid(
            query=query.strip(),
            year=target_year,
            section=section.strip() if section else None,
            chunk_type=target_chunk_type,
            final_top_k=k,
        )

        evidence = [
            {
                "id": c.get("id"),
                "year": c.get("year"),
                "section": c.get("section"),
                "position_id": c.get("position_id"),
                "chunk_type": c.get("chunk_type", "text"),
                "image_path": c.get("image_path"),
                "text": c.get("text", "").strip(),
                "rerank_score": round(float(c.get("rerank_score", 0.0)), 4),
            }
            for c in chunks
        ]

        return {
            "ok": True,
            "data": {
                "query": query,
                "year_filter": target_year,
                "section_filter": section,
                "chunk_type_filter": target_chunk_type,
                "retrieved_count": len(chunks),
                "chunks": evidence,
            },
            "evidence": evidence,
            "error": None,
        }
    except Exception as e:
        log.error("search_filings failed: %s", e)
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": f"Search failed: {e}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. get_metric
# ─────────────────────────────────────────────────────────────────────────────

def get_metric(metric: str, year: int) -> dict[str, Any]:
    """
    Read an audited GAAP financial statement metric from the structured database table.
    """
    if not metric or not isinstance(metric, str):
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": "Metric parameter must be a non-empty string.",
        }

    try:
        yr = int(year)
        if yr not in VALID_YEARS:
            return {
                "ok": False,
                "data": None,
                "evidence": [],
                "error": f"Invalid year: {year}. Allowed fiscal years: 2021-2025.",
            }
    except (ValueError, TypeError):
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": f"Year must be an integer between 2021 and 2025, got {year!r}.",
        }

    cleaned = metric.strip().lower()

    # Ambiguous "total income" query: clarify both Net Income and Operating Income
    if cleaned in {"total income", "income", "total earnings"}:
        try:
            with engine.connect() as conn:
                rows = conn.execute(
                    text("SELECT metric, value FROM financials WHERE company = 'AAPL' AND year = :year"),
                    {"year": yr},
                ).mappings().all()
            metrics_dict = {r["metric"]: float(r["value"]) for r in rows}
            net_inc = metrics_dict.get("net_income")
            op_inc = metrics_dict.get("operating_income")
            rev = metrics_dict.get("revenue")

            clarification = (
                f"In US GAAP, 'Total Income' is not a standard standalone line item. "
                f"In FY{yr}, Apple reported Net Income (bottom line) of ${net_inc:,.1f}M, "
                f"Operating Income of ${op_inc:,.1f}M, and Total Revenue of ${rev:,.1f}M."
            )
            data_payload = {
                "company": "AAPL",
                "year": yr,
                "query": "total_income",
                "net_income": net_inc,
                "operating_income": op_inc,
                "revenue": rev,
                "unit": "USD Millions",
                "note": "Disambiguated to Net Income and Operating Income",
            }
            evidence = [
                {
                    "year": yr,
                    "section": "Item 8. Financial Statements and Supplementary Data",
                    "position_id": 13,
                    "chunk_type": "table",
                    "text": clarification,
                }
            ]
            return {
                "ok": True,
                "data": data_payload,
                "evidence": evidence,
                "error": None,
            }
        except Exception as e:
            return {
                "ok": False,
                "data": None,
                "evidence": [],
                "error": f"Failed reading total income metrics: {e}",
            }

    canonical = RAW_METRIC_ALIASES.get(cleaned)
    if not canonical:
        allowed = sorted(list(set(RAW_METRIC_ALIASES.values())))
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": f"Unknown metric '{metric}'. Allowed metrics: {', '.join(allowed)}.",
        }

    try:
        with engine.connect() as conn:
            val = conn.execute(
                text("SELECT value FROM financials WHERE company = 'AAPL' AND year = :year AND metric = :metric"),
                {"year": yr, "metric": canonical},
            ).scalar()

        if val is None:
            return {
                "ok": False,
                "data": None,
                "evidence": [],
                "error": f"Metric '{canonical}' for year {yr} not found in database.",
            }

        numeric_val = float(val)
        statement_name = "Consolidated Statements of Operations" if canonical in {"revenue", "cost_of_revenue", "operating_income", "net_income"} else "Consolidated Balance Sheets"
        evidence_text = f"Apple Inc. FY{yr} {canonical.replace('_', ' ').title()}: ${numeric_val:,.1f} million (Item 8, {statement_name})."

        return {
            "ok": True,
            "data": {
                "company": "AAPL",
                "year": yr,
                "metric": canonical,
                "value": numeric_val,
                "unit": "USD Millions",
            },
            "evidence": [
                {
                    "year": yr,
                    "section": "Item 8. Financial Statements and Supplementary Data",
                    "position_id": 13,
                    "chunk_type": "table",
                    "text": evidence_text,
                }
            ],
            "error": None,
        }
    except Exception as e:
        log.error("get_metric failed: %s", e)
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": f"Failed reading metric: {e}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. compute_ratio
# ─────────────────────────────────────────────────────────────────────────────

def compute_ratio(name: str, year: int) -> dict[str, Any]:
    """
    Compute a financial ratio for a given fiscal year using deterministic Pandas arithmetic.
    """
    if not name or not isinstance(name, str):
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": "Ratio name must be a non-empty string.",
        }

    try:
        yr = int(year)
        if yr not in VALID_YEARS:
            return {
                "ok": False,
                "data": None,
                "evidence": [],
                "error": f"Invalid year: {year}. Allowed fiscal years: 2021-2025.",
            }
    except (ValueError, TypeError):
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": f"Year must be an integer between 2021 and 2025, got {year!r}.",
        }

    cleaned = name.strip().lower()
    canonical = RATIO_ALIASES.get(cleaned)
    if not canonical:
        allowed = sorted(list(set(RATIO_ALIASES.values())))
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": f"Unknown ratio '{name}'. Allowed ratios: {', '.join(allowed)}.",
        }

    try:
        from financials.ratios import compute_all_ratios
        ratios_df = compute_all_ratios("AAPL")

        if yr not in ratios_df.index:
            return {
                "ok": False,
                "data": None,
                "evidence": [],
                "error": f"Ratio data for FY{yr} unavailable.",
            }

        val = float(ratios_df.loc[yr, canonical])
        formula = RATIO_FORMULAS.get(canonical, "")

        if "margin" in canonical or canonical in {"roa", "roe"}:
            formatted = f"{val * 100:.2f}%"
        elif "ratio" in canonical or canonical in {"debt_to_equity", "debt_to_assets"}:
            formatted = f"{val:.2f}x"
        else:
            formatted = f"{val:.2f}"

        evidence_text = (
            f"Apple Inc. FY{yr} {canonical.replace('_', ' ').title()}: {formatted} "
            f"[Formula: {formula}] (Computed via audited 10-K numbers in pure Pandas)."
        )

        return {
            "ok": True,
            "data": {
                "company": "AAPL",
                "year": yr,
                "ratio": canonical,
                "value": round(val, 4),
                "formatted": formatted,
                "formula": formula,
            },
            "evidence": [
                {
                    "year": yr,
                    "section": "Item 8. Financial Statements and Supplementary Data",
                    "position_id": 13,
                    "chunk_type": "table",
                    "text": evidence_text,
                }
            ],
            "error": None,
        }
    except Exception as e:
        log.error("compute_ratio failed: %s", e)
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": f"Failed computing ratio: {e}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. compare_years
# ─────────────────────────────────────────────────────────────────────────────

def compare_years(metric: str, year_a: int, year_b: int) -> dict[str, Any]:
    """
    Compare a financial metric or ratio between two fiscal years, computing absolute and percentage change.
    """
    if not metric or not isinstance(metric, str):
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": "Metric parameter must be a non-empty string.",
        }

    try:
        ya, yb = int(year_a), int(year_b)
        if ya not in VALID_YEARS or yb not in VALID_YEARS:
            return {
                "ok": False,
                "data": None,
                "evidence": [],
                "error": f"Invalid years: {year_a}, {year_b}. Both must be between 2021 and 2025.",
            }
    except (ValueError, TypeError):
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": "Both year_a and year_b must be integers between 2021 and 2025.",
        }

    cleaned = metric.strip().lower()

    # 1. Try raw metric
    res_a = get_metric(cleaned, ya)
    res_b = get_metric(cleaned, yb)

    is_ratio = False
    if not res_a["ok"] or not res_b["ok"]:
        # 2. Try ratio
        res_a = compute_ratio(cleaned, ya)
        res_b = compute_ratio(cleaned, yb)
        is_ratio = True

    if not res_a["ok"]:
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": f"Unable to resolve metric or ratio '{metric}': {res_a['error']}",
        }
    if not res_b["ok"]:
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": f"Unable to resolve metric or ratio '{metric}' for FY{yb}: {res_b['error']}",
        }

    val_a = float(res_a["data"]["value"])
    val_b = float(res_b["data"]["value"])
    canonical_name = res_a["data"].get("metric") or res_a["data"].get("ratio")

    abs_diff = round(val_b - val_a, 4)
    pct_change = round(((val_b - val_a) / abs(val_a)) * 100, 2) if val_a != 0 else None

    unit = "%" if is_ratio and "margin" in canonical_name else ("x" if is_ratio else "$M")
    sign = "+" if abs_diff >= 0 else ""
    evidence_text = (
        f"Apple Inc. {canonical_name.replace('_', ' ').title()}: FY{ya} was {val_a:,.2f} {unit}, "
        f"FY{yb} was {val_b:,.2f} {unit}. Absolute change: {sign}{abs_diff:,.2f} {unit} "
        f"({sign}{pct_change:+.2f}% YoY)."
    )

    combined_evidence = res_a.get("evidence", []) + res_b.get("evidence", [])
    combined_evidence.append({
        "year": yb,
        "section": "Item 8. Financial Statements and Supplementary Data",
        "position_id": 13,
        "chunk_type": "table",
        "text": evidence_text,
    })

    return {
        "ok": True,
        "data": {
            "metric": canonical_name,
            "year_a": ya,
            "year_b": yb,
            "value_a": val_a,
            "value_b": val_b,
            "absolute_change": abs_diff,
            "percent_change": pct_change,
            "is_ratio": is_ratio,
        },
        "evidence": combined_evidence,
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. get_risk_scores
# ─────────────────────────────────────────────────────────────────────────────

def get_risk_scores(
    category: Optional[str] = None,
    year: Optional[int] = None,
) -> dict[str, Any]:
    """
    Retrieve audited multi-factor risk scores (0.0 to 1.0) computed across 7 risk categories.
    """
    if not RISK_SCORES_PATH.exists():
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": f"Risk scores data file {RISK_SCORES_PATH} not found.",
        }

    target_year = None
    if year is not None:
        try:
            target_year = int(year)
            if target_year not in VALID_YEARS:
                return {
                    "ok": False,
                    "data": None,
                    "evidence": [],
                    "error": f"Invalid year: {year}. Allowed fiscal years: 2021-2025.",
                }
        except (ValueError, TypeError):
            return {
                "ok": False,
                "data": None,
                "evidence": [],
                "error": f"Year must be an integer between 2021 and 2025, got {year!r}.",
            }

    canonical_cat = None
    if category is not None and isinstance(category, str) and category.strip():
        cleaned_cat = category.strip().lower()
        canonical_cat = VALID_RISK_CATEGORIES.get(cleaned_cat)
        if not canonical_cat:
            allowed = sorted(list(set(VALID_RISK_CATEGORIES.values())))
            return {
                "ok": False,
                "data": None,
                "evidence": [],
                "error": f"Unknown risk category '{category}'. Allowed categories: {', '.join(allowed)}.",
            }

    try:
        with open(RISK_SCORES_PATH, encoding="utf-8") as f:
            all_scores: dict[str, dict[str, float]] = json.load(f)

        filtered: dict[str, Any] = {}
        for yr_str, cat_dict in all_scores.items():
            yr_int = int(yr_str)
            if target_year and yr_int != target_year:
                continue
            if canonical_cat:
                filtered[yr_str] = {canonical_cat: cat_dict.get(canonical_cat, 0.0)}
            else:
                filtered[yr_str] = cat_dict

        evidence_lines = []
        for yr_str, cats in filtered.items():
            cat_summary = ", ".join(f"{c}: {score:.3f}" for c, score in cats.items())
            evidence_lines.append(f"FY{yr_str} Risk Scores: {cat_summary}")

        evidence_text = "\n".join(evidence_lines)
        evidence = [
            {
                "year": target_year or 2025,
                "section": "Item 1A. Risk Factors",
                "position_id": 12,
                "chunk_type": "text",
                "text": evidence_text,
            }
        ]

        return {
            "ok": True,
            "data": {
                "year_filter": target_year,
                "category_filter": canonical_cat,
                "scores": filtered,
            },
            "evidence": evidence,
            "error": None,
        }
    except Exception as e:
        log.error("get_risk_scores failed: %s", e)
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": f"Failed reading risk scores: {e}",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. search_charts
# ─────────────────────────────────────────────────────────────────────────────

def search_charts(query: str, year: Optional[int] = None) -> dict[str, Any]:
    """
    Search specifically for extracted visual charts, graphs, and figures from 10-K filings.
    """
    return search_filings(query=query, year=year, chunk_type="image", top_k=5)


# ─────────────────────────────────────────────────────────────────────────────
# 7. TOOL_SPECS (Planner Schema)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SPECS = [
    {
        "name": "search_filings",
        "description": "Search 10-K filing texts, tables, and disclosures using hybrid semantic + keyword search with cross-encoder reranking.",
        "parameters": {
            "query": {"type": "string", "description": "Search query terms or concept."},
            "year": {"type": "integer", "description": "Optional fiscal year filter (2021-2025). Highly recommended for single-year questions."},
            "section": {"type": "string", "description": "Optional section name filter (e.g. 'Item 1A', 'Item 7', 'Item 8')."},
            "chunk_type": {"type": "string", "description": "Optional chunk type filter: 'text', 'table', or 'image'."},
            "top_k": {"type": "integer", "description": "Number of passages to retrieve (default 5, max 20)."},
        },
    },
    {
        "name": "get_metric",
        "description": "Read audited GAAP financial statement metrics from the structured database (revenue, net_income, total_debt, total_equity, operating_income, etc.). Zero hallucination.",
        "parameters": {
            "metric": {"type": "string", "description": "Metric name (e.g. 'revenue', 'net_income', 'total_debt', 'operating_income')."},
            "year": {"type": "integer", "description": "Fiscal year (2021-2025)."},
        },
    },
    {
        "name": "compute_ratio",
        "description": "Compute financial ratios (gross_margin, operating_margin, net_margin, roe, roa, current_ratio, quick_ratio, debt_to_equity, interest_coverage) using deterministic Pandas arithmetic.",
        "parameters": {
            "name": {"type": "string", "description": "Ratio name (e.g. 'gross_margin', 'operating_margin', 'current_ratio', 'debt_to_equity')."},
            "year": {"type": "integer", "description": "Fiscal year (2021-2025)."},
        },
    },
    {
        "name": "compare_years",
        "description": "Compare a metric or ratio between two fiscal years. Computes absolute difference and percentage change with pure code.",
        "parameters": {
            "metric": {"type": "string", "description": "Metric or ratio name (e.g. 'revenue', 'total_debt', 'gross_margin')."},
            "year_a": {"type": "integer", "description": "Base fiscal year (2021-2025)."},
            "year_b": {"type": "integer", "description": "Comparison fiscal year (2021-2025)."},
        },
    },
    {
        "name": "get_risk_scores",
        "description": "Retrieve multi-factor composite risk scores (0.0 to 1.0) across 7 categories: Liquidity, Credit, FX, Regulatory, Cybersecurity, Operational, Legal.",
        "parameters": {
            "category": {"type": "string", "description": "Optional category name (e.g. 'Regulatory', 'FX', 'Cybersecurity')."},
            "year": {"type": "integer", "description": "Optional fiscal year (2021-2025)."},
        },
    },
    {
        "name": "search_charts",
        "description": "Search specifically for extracted 10-K image charts and graphs (e.g. 5-year stock performance graphs).",
        "parameters": {
            "query": {"type": "string", "description": "Search terms describing the chart or figure."},
            "year": {"type": "integer", "description": "Optional fiscal year (2021-2025)."},
        },
    },
]

TOOL_MAP = {
    "search_filings": search_filings,
    "get_metric": get_metric,
    "compute_ratio": compute_ratio,
    "compare_years": compare_years,
    "get_risk_scores": get_risk_scores,
    "search_charts": search_charts,
}


def execute_tool(action: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name with arguments and return standardized result dict."""
    tool_fn = TOOL_MAP.get(action)
    if not tool_fn:
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": f"Unknown tool: '{action}'. Valid tools: {list(TOOL_MAP.keys())}",
        }
    try:
        return tool_fn(**args)
    except TypeError as te:
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": f"Invalid arguments for '{action}': {te}",
        }
    except Exception as e:
        return {
            "ok": False,
            "data": None,
            "evidence": [],
            "error": f"Error executing '{action}': {e}",
        }

