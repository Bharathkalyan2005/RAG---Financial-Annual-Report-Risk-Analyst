"""
generation/router.py — Intent-Based Query Router
=================================================

Routes:
  - INFO, COMPARISON, TREND, RISK: Hybrid Retrieval (retrieval/hybrid.py) + Answer Generation (generation/answer.py)
  - CALCULATION: Math is strictly computed via Pandas on the structured `financials` table.
                 NEVER delegate arithmetic to the LLM.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlalchemy import text

from db import engine
from generation.classify import QueryCategory, classify_question, GREETING_RESPONSE
from generation.answer import generate_answer
from retrieval.hybrid import retrieve_hybrid

log = logging.getLogger("generation.router")


def route_and_execute(
    question: str,
    year: Optional[int] = None,
    company: Optional[str] = "AAPL",
    abstain_threshold: float = 0.30,
) -> dict[str, Any]:
    """
    Classify query, execute appropriate routing pathway, and log classification.
    """
    category: QueryCategory = classify_question(question)
    log.info("Query '%s' routed to category [%s]", question, category)

    if category == "GREETING":
        return {
            "category": "GREETING",
            "question": question,
            "answer": GREETING_RESPONSE,
            "citations": [],
            "confidence": 1.0,
            "retrieved_count": 0,
            "top_rerank_score": 1.0,
        }

    if category == "CALCULATION":
        return _handle_calculation(question, year, company)


    # All qualitative routes: INFO, COMPARISON, TREND, RISK
    retrieved_chunks = retrieve_hybrid(
        query=question,
        company=company,
        year=year,
        final_top_k=5,
    )

    answer_obj = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        abstain_threshold=abstain_threshold,
    )

    return {
        "category": category,
        "question": question,
        "answer": answer_obj.get("answer"),
        "citations": answer_obj.get("citations", []),
        "confidence": answer_obj.get("confidence", 0.0),
        "abstain_reason": answer_obj.get("abstain_reason"),
        "retrieved_count": len(retrieved_chunks),
        "top_rerank_score": retrieved_chunks[0]["rerank_score"] if retrieved_chunks else 0.0,
    }


def _handle_calculation(
    question: str,
    year: Optional[int] = None,
    company: Optional[str] = "AAPL",
) -> dict[str, Any]:
    """
    Handle mathematical and ratio questions using structured database data and Pandas.
    Never relies on the LLM for arithmetic.
    """
    log.info("Handling CALCULATION query via Pandas/Database...")

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM financials;")).scalar()

    if not count or count == 0:
        return {
            "category": "CALCULATION",
            "question": question,
            "answer": "Calculation routing not yet implemented, financials table pending (Phase 7).",
            "citations": [],
            "confidence": 0.0,
            "computed_via": "pandas_stub",
        }

    # Phase 7 full integration:
    # Loads financials DataFrame and runs ratio/growth calculations
    try:
        from financials.ratios import compute_query_metric
        result = compute_query_metric(question, year=year, company=company)
        return {
            "category": "CALCULATION",
            "question": question,
            **result,
        }
    except Exception as e:
        log.error("Calculation execution failed: %s", e)
        return {
            "category": "CALCULATION",
            "question": question,
            "answer": f"Unable to compute requested financial metric: {e}",
            "citations": [],
            "confidence": 0.0,
        }
