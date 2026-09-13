"""
eval/run_eval.py — Phase 9: Systematic RAG Retrieval & Ranking Evaluation Harness
==================================================================================

Evaluates and compares:
  1. Baseline: Raw Dense Vector Search (cosine <=> only, no keyword fusion, no reranking)
  2. Improved (v2): Hybrid Vector + Full-Text Search (RRF) + Year Metadata Filter + Cross-Encoder Reranker

Metrics:
  - Recall@1
  - Recall@3
  - Recall@5
  - MRR (Mean Reciprocal Rank)
  - Year Metadata Precision (% of retrieved chunks matching queried year)
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

from db import engine
from retrieval.hybrid import get_embed_model, get_reranker, retrieve_hybrid

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("eval.harness")

TESTSET_PATH = Path("eval/testset.jsonl")
RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_year_from_query(query: str) -> int | None:
    """Detect 4-digit year in query (e.g. 2021-2025)."""
    match = re.search(r"\b(202[1-5])\b", query)
    return int(match.group(1)) if match else None


def retrieve_baseline_vector_only(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Baseline retrieval: raw cosine distance vector search only, no reranking, no metadata filter."""
    model = get_embed_model()
    q_emb = model.encode(query, normalize_embeddings=True).tolist()

    with engine.connect() as conn:
        sql = """
            SELECT id, company, year, section, chunk_type, text,
                   (embedding <=> CAST(:q_emb AS vector)) AS dist
            FROM chunks
            WHERE embedding IS NOT NULL
            ORDER BY dist ASC
            LIMIT :top_k;
        """
        rows = conn.execute(text(sql), {"q_emb": str(q_emb), "top_k": top_k}).mappings().all()

    return [dict(r) for r in rows]


def evaluate_retrieval(mode: str = "improved") -> dict[str, Any]:
    """Run retrieval evaluation against testset.jsonl for given mode ('baseline' or 'improved')."""
    if not TESTSET_PATH.exists():
        raise FileNotFoundError(f"{TESTSET_PATH} not found.")

    with open(TESTSET_PATH, encoding="utf-8") as f:
        test_items = [json.loads(line) for line in f if line.strip()]

    total_queries = len(test_items)
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    year_matches = 0
    total_retrieved = 0

    log.info("Evaluating mode '%s' on %d queries...", mode, total_queries)

    for item in test_items:
        q = item["question"]
        exp_yr = item.get("expected_year")
        keywords = [k.lower() for k in item.get("keywords", [])]

        if mode == "baseline":
            chunks = retrieve_baseline_vector_only(q, top_k=5)
        else:
            yr_filter = extract_year_from_query(q)
            chunks = retrieve_hybrid(q, year=yr_filter, final_top_k=5)

        first_hit_rank = 0
        for rank, c in enumerate(chunks, 1):
            total_retrieved += 1
            c_text = (c.get("text") or "").lower()
            c_yr = c.get("year")

            if exp_yr and c_yr == exp_yr:
                year_matches += 1

            # Relevant if correct year and at least one ground-truth keyword present
            is_relevant = (not exp_yr or c_yr == exp_yr) and any(kw in c_text for kw in keywords)

            if is_relevant and first_hit_rank == 0:
                first_hit_rank = rank

        if first_hit_rank == 1:
            hits_at_1 += 1
        if 1 <= first_hit_rank <= 3:
            hits_at_3 += 1
        if 1 <= first_hit_rank <= 5:
            hits_at_5 += 1

        rr = (1.0 / first_hit_rank) if first_hit_rank > 0 else 0.0
        reciprocal_ranks.append(rr)

    metrics = {
        "mode": mode,
        "total_queries": total_queries,
        "recall_at_1": round(hits_at_1 / total_queries, 4),
        "recall_at_3": round(hits_at_3 / total_queries, 4),
        "recall_at_5": round(hits_at_5 / total_queries, 4),
        "mrr": round(sum(reciprocal_ranks) / total_queries, 4),
        "year_precision": round(year_matches / (total_retrieved or 1), 4),
    }

    out_file = RESULTS_DIR / f"{mode}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log.info("Mode '%s' results saved to %s", mode, out_file)
    return metrics


def main():
    print("\n" + "=" * 70)
    print("  PHASE 9: SYSTEMATIC RETRIEVAL EVALUATION")
    print("=" * 70)

    # 1. Evaluate Baseline
    print("\n[1/2] Running Baseline evaluation (Dense Vector Only)...")
    baseline = evaluate_retrieval(mode="baseline")

    # 2. Evaluate Improved Pipeline (v2)
    print("\n[2/2] Running Improved evaluation (Hybrid + Filter + Reranker)...")
    improved = evaluate_retrieval(mode="improved")

    # 3. Print Comparison Table
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS: BASELINE VS. IMPROVED (V2)")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Baseline':<12} | {'Improved (v2)':<14} | {'Lift (%)':<10}")
    print("-" * 70)

    metric_keys = [
        ("recall_at_1", "Recall@1"),
        ("recall_at_3", "Recall@3"),
        ("recall_at_5", "Recall@5"),
        ("mrr", "MRR (Mean Recip Rank)"),
        ("year_precision", "Year Metadata Precision"),
    ]

    for key, label in metric_keys:
        b_val = baseline[key]
        i_val = improved[key]
        lift = ((i_val - b_val) / (b_val or 0.001)) * 100.0
        print(f"{label:<25} | {b_val:<12.4f} | {i_val:<14.4f} | +{lift:<9.1f}%")

    print("=" * 70)
    print("  Results saved to eval/results/baseline.json and eval/results/improved.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
