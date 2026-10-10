"""
agent/grade.py — Retrieval Evidence Grader & Query Rewrite Tracker
===================================================================

Scores retrieved chunks (0.0 to 1.0) using the neural Cross-Encoder / FlagReranker
(not the LLM) to verify evidence relevance. If best relevance score falls below
GRADE_THRESHOLD (default 0.35), marks the observation as 'weak evidence' so the
planner can rewrite the query, change the section filter, or try another year.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from retrieval.hybrid import get_reranker

log = logging.getLogger("agent.grade")

DEFAULT_GRADE_THRESHOLD = 0.35
MAX_REWRITES_PER_SUBQUESTION = 2


def score_chunks_cross_encoder(query: str, chunks: list[dict[str, Any]]) -> list[float]:
    """Score chunks against query using the neural cross-encoder."""
    if not chunks or not query:
        return []

    texts = [c.get("text", "") for c in chunks]
    pairs = [[query, t] for t in texts]

    reranker = get_reranker()
    if hasattr(reranker, "compute_score"):
        raw_scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(raw_scores, float):
            raw_scores = [raw_scores]
    else:
        logits = reranker.predict(pairs)
        if np.isscalar(logits):
            raw_scores = [float(1 / (1 + np.exp(-logits)))]
        else:
            raw_scores = (1 / (1 + np.exp(-logits))).tolist()

    return [round(float(s), 4) for s in raw_scores]


def grade_evidence(
    query: str,
    chunks: list[dict[str, Any]],
    threshold: float = DEFAULT_GRADE_THRESHOLD,
) -> dict[str, Any]:
    """
    Grade retrieved chunks for relevance to the target query.

    Returns:
      {
        "grade": "relevant" | "weak_evidence",
        "best_score": float,
        "scores": list[float],
        "is_weak": bool,
        "summary": str,
      }
    """
    if not chunks:
        return {
            "grade": "weak_evidence",
            "best_score": 0.0,
            "scores": [],
            "is_weak": True,
            "summary": "No chunks retrieved (empty result set).",
        }

    # If chunks already have rerank_score, we check if we can reuse it
    scores = []
    need_rescore = False
    for c in chunks:
        if "rerank_score" in c and isinstance(c["rerank_score"], (int, float)):
            scores.append(float(c["rerank_score"]))
        else:
            need_rescore = True
            break

    if need_rescore or not scores:
        scores = score_chunks_cross_encoder(query, chunks)

    # Attach score to each chunk for traceability
    for c, s in zip(chunks, scores):
        c["rerank_score"] = s

    best_score = max(scores) if scores else 0.0
    is_weak = best_score < threshold

    grade = "weak_evidence" if is_weak else "relevant"
    summary = (
        f"Weak evidence: top relevance {best_score:.3f} is below threshold {threshold:.2f}."
        if is_weak
        else f"Sufficient evidence: top relevance score {best_score:.3f} (>= {threshold:.2f})."
    )

    return {
        "grade": grade,
        "best_score": round(best_score, 4),
        "scores": scores,
        "is_weak": is_weak,
        "summary": summary,
    }
