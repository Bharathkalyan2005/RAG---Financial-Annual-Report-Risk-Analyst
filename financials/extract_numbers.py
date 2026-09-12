"""
financials/extract_numbers.py — Phase 7: Extract Core Financial Metrics from Tables
==================================================================================

Pipeline:
  1. Finds all table chunks from "Financial Statements" / "Item 8" across all 5 years.
  2. Passes the tables to OpenRouter with a structured JSON schema to extract:
     - revenue
     - net_income
     - total_debt
     - total_equity
     - current_assets
     - current_liabilities
     - total_assets
     - cost_of_revenue
     - operating_income
     - interest_expense
  3. Inserts extracted metrics into the Neon `financials` table (company, year, metric, value).
  4. Reports any missing or null metrics per fiscal year.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import text

from db import Financial, SessionLocal, create_tables, engine
from generation.client import OpenRouterClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("financials.extract")

CHUNKS_FILE = Path("data/processed/chunks.jsonl")

TARGET_METRICS = [
    "revenue",
    "net_income",
    "total_debt",
    "total_equity",
    "current_assets",
    "current_liabilities",
    "total_assets",
    "cost_of_revenue",
    "operating_income",
    "interest_expense",
]

EXTRACTION_PROMPT = """You are a senior financial analyst extracting GAAP financial statement metrics from SEC 10-K filing tables.
Examine the following table markdown and extract the EXACT numbers for the company in fiscal year {year}.
All values should be in millions of USD (as reported in 10-Ks).

Target metrics:
- revenue: Total net sales / total revenue
- net_income: Net income / net earnings
- total_debt: Total debt (commercial paper + current term debt + non-current term debt)
- total_equity: Total shareholders' / stockholders' equity
- current_assets: Total current assets
- current_liabilities: Total current liabilities
- total_assets: Total assets
- cost_of_revenue: Total cost of sales / cost of revenue
- operating_income: Operating income
- interest_expense: Total interest / debt expense

Rules:
1. Return ONLY a single JSON object.
2. Value MUST be a numeric float/integer (e.g. 383285.0).
3. If a metric is NOT present in this specific table, set its value to null.
4. Do NOT guess, do NOT extrapolate, do NOT compute.

JSON format:
{{
  "revenue": 383285.0,
  "net_income": 96995.0,
  "total_debt": 106629.0,
  "total_equity": 62146.0,
  "current_assets": 143566.0,
  "current_liabilities": 145308.0,
  "total_assets": 352583.0,
  "cost_of_revenue": 214137.0,
  "operating_income": 114301.0,
  "interest_expense": 3933.0
}}
"""


def load_statement_tables() -> dict[int, list[str]]:
    """Group relevant financial statement tables by fiscal year."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"{CHUNKS_FILE} not found. Ingestion must run first.")

    tables_by_year: dict[int, list[str]] = {}
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk.get("chunk_type") != "table":
                continue
            section = (chunk.get("section") or "").lower()
            if "financial statements" in section or "item 8" in section:
                yr = int(chunk["year"])
                tables_by_year.setdefault(yr, []).append(chunk["text"])

    return tables_by_year


def extract_and_store():
    log.info("Starting financial metric extraction for all 5 years...")
    tables_by_year = load_statement_tables()
    client = OpenRouterClient()
    create_tables()

    # Store aggregated metrics: year -> metric -> value
    year_metrics: dict[int, dict[str, float]] = {yr: {} for yr in sorted(tables_by_year.keys())}

    for yr in sorted(tables_by_year.keys()):
        log.info("Processing Year %d (%d statement tables found)...", yr, len(tables_by_year[yr]))
        for idx, table_text in enumerate(tables_by_year[yr], 1):
            # Skip small tables
            if len(table_text.strip().split("\n")) < 4:
                continue

            prompt = EXTRACTION_PROMPT.format(year=yr) + f"\n\nTable Content:\n{table_text}\n\nJSON Output:"
            try:
                raw = client.chat([{"role": "user", "content": prompt}], temperature=0.0)
                # Clean JSON
                match = re.search(r"\{.*?\}", raw, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    for m in TARGET_METRICS:
                        val = parsed.get(m)
                        if val is not None and isinstance(val, (int, float)) and val != 0:
                            if m not in year_metrics[yr]:
                                year_metrics[yr][m] = float(val)
            except Exception as e:
                log.warning("Table %d/%d for year %d extraction error: %s", idx, len(tables_by_year[yr]), yr, e)

            # Check if all metrics found for this year
            if len(year_metrics[yr]) == len(TARGET_METRICS):
                log.info("All metrics found for year %d!", yr)
                break

    # Insert into Neon financials table
    log.info("Persisting extracted metrics to Neon `financials` table...")
    session = SessionLocal()
    try:
        # Clear existing entries to prevent duplicates
        session.execute(text("DELETE FROM financials WHERE company = 'AAPL';"))
        session.commit()

        for yr, metrics in year_metrics.items():
            for m_name, val in metrics.items():
                row = Financial(
                    company="AAPL",
                    year=yr,
                    metric=m_name,
                    value=val,
                )
                session.add(row)
        session.commit()
        log.info("All metrics successfully inserted into Neon `financials`.")
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    # Log summary report
    print("\n" + "=" * 70)
    print("  FINANCIAL EXTRACTION REPORT")
    print("=" * 70)
    for yr in sorted(year_metrics.keys()):
        found = year_metrics[yr]
        missing = [m for m in TARGET_METRICS if m not in found]
        print(f"\nFiscal Year {yr}: {len(found)}/{len(TARGET_METRICS)} metrics identified")
        for m in TARGET_METRICS:
            status = f"${found[m]:,.1f}M" if m in found else "[MISSING / NULL]"
            print(f"  - {m:<20}: {status}")
        if missing:
            print(f"  Missing for {yr}: {', '.join(missing)}")
    print("=" * 70)


if __name__ == "__main__":
    extract_and_store()
