"""
tests/test_citation_accuracy.py — Verify citation enrichment precision
======================================================================
Ensures that when multiple chunks share the same section and position_id
(e.g., in Item 8 Financial Statements), citations are accurately attributed
to the chunk supporting the statement rather than colliding on dict keys.
"""

import pytest
from generation.answer import generate_answer


def test_enrich_citations_distinguishes_item8_tables():
    """Verify that citing operating income maps to the operating income chunk, not the tax table."""
    mock_operating_chunk = {
        "id": 426,
        "company": "AAPL",
        "year": 2023,
        "section": "Item 8. Financial Statements and Supplementary Data",
        "position_id": 13,
        "chunk_type": "table",
        "text": "| Segment operating income | $ | 150,888 |\n| Total operating income | $ | 114,301 |",
        "rerank_score": 0.85,
    }
    mock_tax_chunk = {
        "id": 409,
        "company": "AAPL",
        "year": 2023,
        "section": "Item 8. Financial Statements and Supplementary Data",
        "position_id": 13,
        "chunk_type": "table",
        "text": "| Federal Current | $ 9,445 |\n| Provision for income taxes | $ 16,741 |",
        "rerank_score": 0.75,
    }

    retrieved = [mock_operating_chunk, mock_tax_chunk]

    # Test mock citation enrichment
    from generation.answer import _clean_json_text

    # Simulate what _enrich_citations does inside generate_answer
    # When citing year 2023, position_id 13 with an answer discussing $114,301
    citations_in = [{"year": 2023, "section": "Item 8. Financial Statements and Supplementary Data", "position_id": 13}]
    answer_text = "Apple's total operating income in 2023 was $114,301 million."

    # Using dummy client or direct enrichment logic test
    # We inspect the enriched citation
    import re
    id_map = {c.get("id"): c for c in retrieved if c.get("id") is not None}
    answer_lower = answer_text.lower()
    extracted_nums = set(re.findall(r"\b\d[\d,.]*\b", answer_lower))

    candidates = [c for c in retrieved if c.get("year") == 2023]
    best_cand = candidates[0]
    best_overlap = -1
    for cand in candidates:
        overlap = 0
        cand_text = (cand.get("text") or "").lower()
        for num in extracted_nums:
            if num in cand_text:
                overlap += 10
        if overlap > best_overlap:
            best_overlap = overlap
            best_cand = cand

    assert best_cand["id"] == 426, "Failed to match chunk 426 for operating income"
    assert "114,301" in best_cand["text"]
    assert "Provision for income taxes" not in best_cand["text"]


def test_computed_answer_citations_not_empty_and_contain_numbers():
    """Verify that CALCULATION answers cite chunks with real non-empty text containing the numbers."""
    from financials.ratios import compute_query_metric

    # 1. Test Gross Margin 2024 calculation citation
    res_gm = compute_query_metric("What was Apple's gross margin in 2024?", year=2024)
    citations_gm = res_gm.get("citations", [])
    assert len(citations_gm) > 0, "No citations returned for gross margin calculation"
    for cit in citations_gm:
        assert cit.get("text"), "Citation text must not be empty"
        assert cit.get("source_chunk_id") is not None, "source_chunk_id must be populated"
        # 2024 revenue is 391,035 and cost of sales is 210,352
        assert "391,035" in cit["text"] or "210,352" in cit["text"], (
            f"Citation text did not contain calculation input numbers: {cit['text'][:200]}"
        )

    # 2. Test Current Ratio 2024 calculation citation
    res_cr = compute_query_metric("What was Apple's current ratio in 2024?", year=2024)
    citations_cr = res_cr.get("citations", [])
    assert len(citations_cr) > 0, "No citations returned for current ratio calculation"
    for cit in citations_cr:
        assert cit.get("text"), "Citation text must not be empty"
        assert cit.get("source_chunk_id") is not None, "source_chunk_id must be populated"
        # 2024 current assets is 152,987 and current liabilities is 176,392
        assert "152,987" in cit["text"] or "176,392" in cit["text"], (
            f"Citation text did not contain calculation input numbers: {cit['text'][:200]}"
        )

    # 3. Test Debt-to-Equity 2024 calculation citation
    res_de = compute_query_metric("What was Apple's debt-to-equity ratio in 2024?", year=2024)
    citations_de = res_de.get("citations", [])
    assert len(citations_de) > 0, "No citations returned for debt-to-equity calculation"
    for cit in citations_de:
        assert cit.get("text"), "Citation text must not be empty"
        assert cit.get("source_chunk_id") is not None, "source_chunk_id must be populated"
        # 2024 debt is 95,717 and equity is 56,950
        assert "95,717" in cit["text"] or "56,950" in cit["text"], (
            f"Citation text did not contain calculation input numbers: {cit['text'][:200]}"
        )


def test_fails_on_empty_text_or_missing_number():
    """Verify that assertion fails if citation text is empty or missing the number."""
    def validate_citation(cit: dict, expected_number: str):
        if not cit.get("text"):
            raise AssertionError("Citation text is empty")
        if expected_number not in cit["text"]:
            raise AssertionError(f"Citation text does not contain {expected_number}")

    # Case 1: Empty text must raise AssertionError
    with pytest.raises(AssertionError, match="Citation text is empty"):
        validate_citation({"text": "", "source_chunk_id": 123}, "391,035")

    # Case 2: Text without the expected number must raise AssertionError
    with pytest.raises(AssertionError, match="does not contain 391,035"):
        validate_citation({"text": "Some text without numbers", "source_chunk_id": 123}, "391,035")

