"""
tests/test_agent_grading_and_verify.py — Test Evidence Grading & Verification Modules
"""

import pytest
from agent.grade import grade_evidence
from agent.verify import (
    extract_numbers_from_text,
    extract_sentences,
    verify_answer,
    ABSTAIN_MESSAGE,
)


def test_grade_evidence_empty():
    res = grade_evidence("Apple revenue 2024", [])
    assert res["grade"] == "weak_evidence"
    assert res["is_weak"] is True
    assert res["best_score"] == 0.0


def test_grade_evidence_with_existing_scores():
    chunks = [
        {"id": "1", "text": "Apple total revenue was $391,035M in 2024.", "rerank_score": 0.88},
        {"id": "2", "text": "Operating expenses were $55,000M.", "rerank_score": 0.25},
    ]
    res = grade_evidence("Apple revenue 2024", chunks, threshold=0.40)
    assert res["grade"] == "relevant"
    assert res["is_weak"] is False
    assert res["best_score"] == 0.88


def test_grade_evidence_weak():
    chunks = [
        {"id": "1", "text": "Some unrelated legal boilerplate text.", "rerank_score": 0.12},
    ]
    res = grade_evidence("Apple semiconductor supply risk", chunks, threshold=0.35)
    assert res["grade"] == "weak_evidence"
    assert res["is_weak"] is True
    assert res["best_score"] == 0.12


def test_extract_sentences():
    text = "Apple's revenue in 2024 was $391,035M. Total debt decreased to $86,300 million. Operating margin reached 31.5%."
    sentences = extract_sentences(text)
    assert len(sentences) == 3
    assert "$391,035M" in sentences[0]
    assert "86,300 million" in sentences[1]


def test_extract_numbers():
    text = "In 2024, revenue was $391,035 million, debt was $86.3 billion, and margin was 31.5% with 2.4x leverage."
    nums = extract_numbers_from_text(text)
    # Check that 391035, 86300, 31.5, 2.4 are parsed
    assert any(abs(n - 391035.0) < 1.0 for n in nums)
    assert any(abs(n - 86300.0) < 100.0 or abs(n - 86.3) < 0.1 for n in nums)
    assert any(abs(n - 31.5) < 0.1 for n in nums)
    assert any(abs(n - 2.4) < 0.1 for n in nums)


def test_verify_numeric_claim_supported():
    draft = "In FY2024, Apple reported total revenue of $391,035 million."
    evidence_chunks = [
        {"text": "Total net sales were $391,035 million in 2024 compared to $383,285 million in 2023."}
    ]
    tool_results = [
        {"data": {"metric": "revenue", "year": 2024, "value": 391035}}
    ]
    res = verify_answer(draft, evidence_chunks, tool_results, client=None)
    assert res["abstain"] is False
    assert res["all_passed"] is True
    assert len(res["supported_claims"]) == 1
    assert len(res["removed_claims"]) == 0
    assert "$391,035" in res["verified_answer"]


def test_verify_numeric_claim_hallucinated():
    draft = "Apple reported total revenue of $999,999 million in 2024."
    evidence_chunks = [
        {"text": "Total net sales were $391,035 million in 2024."}
    ]
    tool_results = []
    res = verify_answer(draft, evidence_chunks, tool_results, client=None)
    assert res["abstain"] is True
    assert res["verified_answer"] == ABSTAIN_MESSAGE
    assert len(res["removed_claims"]) == 1


def test_verify_partial_unsupported_claim_pruning():
    draft = "Apple reported total revenue of $391,035 million in 2024. Apple had total cash of $999,999 million."
    evidence_chunks = [
        {"text": "Total net sales were $391,035 million in 2024."}
    ]
    tool_results = []
    res = verify_answer(draft, evidence_chunks, tool_results, client=None)
    assert res["abstain"] is False
    assert res["all_passed"] is False
    assert len(res["supported_claims"]) == 1
    assert len(res["removed_claims"]) == 1
    assert "$391,035" in res["verified_answer"]
    assert "999,999" not in res["verified_answer"]
