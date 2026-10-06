"""
tests/test_abstention.py — Verify abstention mechanism for out-of-domain queries
==============================================================================
"""

import pytest
from generation.answer import generate_answer


def test_abstention_on_empty_chunks():
    """When no chunks are retrieved, answer generation must abstain."""
    res = generate_answer("What was Apple's revenue in 2024?", retrieved_chunks=[])
    assert res["confidence"] == 0.0
    assert "abstain_reason" in res or "insufficient" in res["answer"].lower()


def test_abstention_on_low_rerank_score():
    """When chunks have very low rerank scores, system must abstain."""
    low_scored_chunks = [
        {
            "id": 1,
            "company": "AAPL",
            "year": 2024,
            "section": "Item 1. Business",
            "position_id": 1,
            "chunk_type": "text",
            "text": "General description unrelated to query.",
            "rerank_score": 0.12,  # well below default threshold of 0.30
        }
    ]
    res = generate_answer(
        "What was Tesla's Cybertruck production in 2024?",
        retrieved_chunks=low_scored_chunks,
        abstain_threshold=0.30,
    )
    assert res["confidence"] == 0.0
    assert "below threshold" in res.get("abstain_reason", "").lower()
