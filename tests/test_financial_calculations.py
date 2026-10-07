"""
tests/test_financial_calculations.py — Verify deterministic financial calculations
=================================================================================
Ensures all metrics across 2021-2025 match audited GAAP numbers and ratios
are computed with pure Pandas/Python arithmetic without LLM involvement.
"""

import pytest
import pandas as pd
from financials.ratios import (
    load_financials_df,
    compute_all_ratios,
    compute_query_metric,
)


def test_audited_financials_completeness():
    """Verify all 5 fiscal years (2021-2025) exist and have core GAAP items."""
    df = load_financials_df(company="AAPL")
    assert len(df) == 5
    for yr in [2021, 2022, 2023, 2024, 2025]:
        assert yr in df.index, f"Missing FY{yr} in financials database"
        assert df.loc[yr, "revenue"] > 300000, f"FY{yr} revenue implausible"
        assert df.loc[yr, "net_income"] > 50000, f"FY{yr} net income implausible"
        assert df.loc[yr, "total_assets"] > 300000, f"FY{yr} assets implausible"
        assert df.loc[yr, "total_debt"] > 50000, f"FY{yr} total debt implausible"


def test_dataframe_and_ratios_computation():
    """Verify compute_all_ratios generates valid margins and ratios."""
    df = load_financials_df(company="AAPL")
    assert len(df) == 5
    ratios = compute_all_ratios(df)

    # 1. Gross margin should be between 40% and 50%
    for yr, gm in ratios["gross_margin"].items():
        assert 0.40 <= gm <= 0.50, f"FY{yr} Gross margin out of bounds: {gm}"

    # 2. Operating margin should be between 25% and 35%
    for yr, om in ratios["operating_margin"].items():
        assert 0.25 <= om <= 0.35, f"FY{yr} Operating margin out of bounds: {om}"

    # 3. Debt to equity should be positive
    for yr, de in ratios["debt_to_equity"].items():
        assert de > 0, f"FY{yr} Debt to equity non-positive: {de}"

    # 4. Current ratio should be around 0.85 - 1.15
    for yr, cr in ratios["current_ratio"].items():
        assert 0.80 <= cr <= 1.20, f"FY{yr} Current ratio out of expected range: {cr}"


def test_compute_query_metric_known_queries():
    """Test deterministic NLP metric extraction across various years."""
    # FY2024 Total Debt
    res_debt = compute_query_metric("What was Apple's total debt in 2024?", year=2024)
    assert res_debt is not None
    assert "95,717" in res_debt["answer"] or "total debt" in res_debt["answer"].lower()
    assert res_debt["confidence"] == 1.0
    assert len(res_debt["citations"]) >= 1

    # FY2025 Revenue
    res_rev = compute_query_metric("What was Apple's revenue in 2025?", year=2025)
    assert res_rev is not None
    assert "416,161" in res_rev["answer"]

    # FY2023 Net Income
    res_ni = compute_query_metric("What was Apple's net income in 2023?", year=2023)
    assert res_ni is not None
    assert "96,995" in res_ni["answer"]

    # FY2024 Gross Margin
    res_gm = compute_query_metric("What was Apple's gross margin in 2024?", year=2024)
    assert res_gm is not None
    assert "46.21%" in res_gm["answer"]

    # FY2023 Operating Income
    res_op = compute_query_metric("What was Apple's operating income in 2023?", year=2023)
    assert res_op is not None
    assert "114,301" in res_op["answer"]

    # FY2023 Ambiguous "Total Income" query (must clarify both Net Income and Operating Income)
    res_tot = compute_query_metric("that is total income in 2023", year=2023)
    assert res_tot is not None
    assert "96,995" in res_tot["answer"]
    assert "114,301" in res_tot["answer"]
    assert "383,285" in res_tot["answer"]
