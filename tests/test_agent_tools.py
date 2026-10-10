"""
tests/test_agent_tools.py — Unit and integration tests for Agentic RAG tools
=============================================================================
Tests:
  1. search_filings: returns ok=True, evidence list, and error handling.
  2. get_metric: returns audited numbers for valid metrics and catches invalid inputs.
  3. compute_ratio: computes deterministic ratios with pure Pandas.
  4. compare_years: calculates absolute change and percent change accurately.
  5. get_risk_scores: filters by year and category without raising.
  6. search_charts: restricts to image chunk types.
  7. TOOL_SPECS: schema validation for planner prompt.
"""

import pytest
from agent.tools import (
    search_filings,
    get_metric,
    compute_ratio,
    compare_years,
    get_risk_scores,
    search_charts,
    TOOL_SPECS,
)


def test_tool_specs_structure():
    """Verify all 6 tools have valid JSON-schema specs for planner prompt."""
    assert len(TOOL_SPECS) == 6
    names = {t["name"] for t in TOOL_SPECS}
    expected = {
        "search_filings",
        "get_metric",
        "compute_ratio",
        "compare_years",
        "get_risk_scores",
        "search_charts",
    }
    assert names == expected
    for spec in TOOL_SPECS:
        assert "description" in spec
        assert "parameters" in spec
        assert isinstance(spec["parameters"], dict)


def test_get_metric_success():
    """Test get_metric for 3 hand-picked inputs."""
    # 1. FY2024 Total Debt ($95,717M)
    res_debt = get_metric("total_debt", 2024)
    assert res_debt["ok"] is True
    assert res_debt["error"] is None
    assert res_debt["data"]["value"] == 95717.0
    assert len(res_debt["evidence"]) >= 1

    # 2. FY2025 Revenue ($416,161M)
    res_rev = get_metric("revenue", 2025)
    assert res_rev["ok"] is True
    assert res_rev["data"]["value"] == 416161.0

    # 3. FY2023 Operating Income ($114,301M)
    res_op = get_metric("operating_income", 2023)
    assert res_op["ok"] is True
    assert res_op["data"]["value"] == 114301.0


def test_get_metric_validation():
    """Test get_metric gracefully handles invalid inputs without raising."""
    # Invalid year
    res_yr = get_metric("revenue", 1999)
    assert res_yr["ok"] is False
    assert "Invalid year" in res_yr["error"]

    # Invalid metric
    res_bad_metric = get_metric("crypto_balance", 2024)
    assert res_bad_metric["ok"] is False
    assert "Unknown metric" in res_bad_metric["error"]

    # Ambiguous "total income" query returns structured disambiguation
    res_tot_inc = get_metric("total income", 2023)
    assert res_tot_inc["ok"] is True
    assert res_tot_inc["data"]["net_income"] == 96995.0
    assert res_tot_inc["data"]["operating_income"] == 114301.0


def test_compute_ratio_success():
    """Test compute_ratio for 3 hand-picked ratios."""
    # 1. FY2024 Gross Margin (46.21%)
    res_gm = compute_ratio("gross_margin", 2024)
    assert res_gm["ok"] is True
    assert res_gm["data"]["formatted"] == "46.21%"
    assert 0.46 <= res_gm["data"]["value"] <= 0.47

    # 2. FY2023 Debt to Equity (1.63x)
    res_de = compute_ratio("debt_to_equity", 2023)
    assert res_de["ok"] is True
    assert "1.63" in res_de["data"]["formatted"]

    # 3. FY2025 Current Ratio (0.89x)
    res_cr = compute_ratio("current_ratio", 2025)
    assert res_cr["ok"] is True
    assert "0.89" in res_cr["data"]["formatted"]


def test_compute_ratio_validation():
    """Test compute_ratio handles invalid ratio names and years."""
    res_bad = compute_ratio("invalid_ratio", 2024)
    assert res_bad["ok"] is False
    assert "Unknown ratio" in res_bad["error"]

    res_bad_yr = compute_ratio("gross_margin", 2035)
    assert res_bad_yr["ok"] is False
    assert "Invalid year" in res_bad_yr["error"]


def test_compare_years_success():
    """Test multi-year comparison with absolute and percent change."""
    # Total Debt deleveraging from 2021 ($115,106M) to 2025 ($86,307M)
    res_comp = compare_years("total_debt", 2021, 2025)
    assert res_comp["ok"] is True
    data = res_comp["data"]
    assert data["value_a"] == 115106.0
    assert data["value_b"] == 86307.0
    assert data["absolute_change"] == -28799.0
    assert -26.0 <= data["percent_change"] <= -24.0
    assert len(res_comp["evidence"]) >= 2


def test_compare_years_validation():
    """Test compare_years handles invalid year ranges."""
    res_bad = compare_years("revenue", 2020, 2024)
    assert res_bad["ok"] is False
    assert "Invalid years" in res_bad["error"]


def test_get_risk_scores_success():
    """Test get_risk_scores filtering by category and year."""
    # 1. FY2024 Regulatory risk (escalated to 0.990)
    res_reg = get_risk_scores(category="Regulatory", year=2024)
    assert res_reg["ok"] is True
    assert res_reg["data"]["scores"]["2024"]["Regulatory"] == 0.99
    assert len(res_reg["evidence"]) >= 1

    # 2. All categories for FY2023
    res_23 = get_risk_scores(year=2023)
    assert res_23["ok"] is True
    assert len(res_23["data"]["scores"]["2023"]) == 7

    # 3. Validation
    res_bad_cat = get_risk_scores(category="MartianRisk", year=2024)
    assert res_bad_cat["ok"] is False
    assert "Unknown risk category" in res_bad_cat["error"]


def test_search_filings_and_charts():
    """Test search_filings and search_charts wrap hybrid retrieval."""
    res_search = search_filings("Digital Markets Act antitrust", year=2024, top_k=3)
    assert res_search["ok"] is True
    assert res_search["data"]["retrieved_count"] > 0
    assert len(res_search["evidence"]) > 0
    first_ev = res_search["evidence"][0]
    assert "text" in first_ev
    assert "section" in first_ev
    assert "year" in first_ev

    # Test search_charts returns image chunks
    res_chart = search_charts("cumulative total return", year=2024)
    assert res_chart["ok"] is True
    for ev in res_chart["evidence"]:
        assert ev["chunk_type"] == "image"
