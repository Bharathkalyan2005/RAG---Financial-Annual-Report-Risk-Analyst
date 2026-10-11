"""
retrieval/hybrid.py — Phase 4: Hybrid Retrieval (Vector + BM25 tsvector) + Reranking
=====================================================================================

Pipeline:
  1. Embed query with BAAI/bge-large-en-v1.5 WITH the asymmetric query instruction:
     "Represent this sentence for searching relevant passages: "
  2. Dense Vector Search: top-20 by cosine distance (embedding <=> query_vec)
  3. Sparse Keyword Search: top-20 by Postgres tsvector (search_vector @@ plainto_tsquery)
  4. Merge candidate lists via Reciprocal Rank Fusion (RRF, k=60)
  5. Rerank merged top-20 using BAAI/bge-reranker-v2-m3 via FlagEmbedding
  6. Return top-5 results with rerank scores and full metadata
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from sqlalchemy import text
from sentence_transformers import SentenceTransformer

from db import engine

load_dotenv()

log = logging.getLogger("retrieval")

# ─────────────────────────────────────────────────────────────────────────────
# Config & Singletons
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
RERANKER_MODEL_NAME  = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60

_embed_model: SentenceTransformer | None = None
_reranker: Any = None


def extract_years_from_query(query: str) -> list[int]:
    """Detect all 4-digit fiscal years (2021-2025) mentioned in query in order of appearance."""
    matches = re.findall(r"\b(202[1-5])\b", query)
    years = []
    seen = set()
    for m in matches:
        y = int(m)
        if y not in seen:
            seen.add(y)
            years.append(y)
    return years


def extract_year_from_query(query: str) -> int | None:
    """Detect single 4-digit year in query (first match)."""
    years = extract_years_from_query(query)
    return years[0] if years else None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embed_model


def get_reranker() -> Any:
    global _reranker
    if _reranker is None:
        model_path = RERANKER_MODEL_NAME
        # If relative path exists relative to project root
        if not os.path.exists(model_path) and os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)):
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)

        if "cross-encoder" in model_path.lower() or os.path.exists(model_path) or not "bge" in model_path.lower():
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(model_path)
        else:
            try:
                from FlagEmbedding import FlagReranker
                _reranker = FlagReranker(model_path, use_fp16=False)
            except Exception:
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder(model_path)
    return _reranker


# ─────────────────────────────────────────────────────────────────────────────
# Core Hybrid Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_hybrid(
    query: str,
    company: Optional[str] = None,
    year: Optional[int | list[int]] = None,
    section: Optional[str] = None,
    chunk_type: Optional[str] = None,
    vector_top_k: int = 20,
    keyword_top_k: int = 20,
    final_top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Perform hybrid vector + keyword search followed by neural cross-encoder reranking.

    Returns a list of dicts with:
      - text, company, year, section, position_id, chunk_type, rerank_score
    """
    # Multi-year handling: if 2 or more years are requested or detected, run search once per year and merge
    years_to_search = None
    if isinstance(year, (list, tuple)) and len(year) >= 2:
        years_to_search = list(year)
    elif year is None:
        detected = extract_years_from_query(query)
        if len(detected) >= 2:
            years_to_search = detected
        elif len(detected) == 1:
            year = detected[0]

    if years_to_search:
        results_by_year = [
            retrieve_hybrid(
                query=query,
                company=company,
                year=y,
                section=section,
                chunk_type=chunk_type,
                vector_top_k=vector_top_k,
                keyword_top_k=keyword_top_k,
                final_top_k=final_top_k,
            )
            for y in years_to_search
        ]
        merged: list[dict[str, Any]] = []
        seen_ids: set[Any] = set()
        max_len = max((len(r) for r in results_by_year), default=0)
        for i in range(max_len):
            for yr_res in results_by_year:
                if i < len(yr_res):
                    c = yr_res[i]
                    if c["id"] not in seen_ids:
                        seen_ids.add(c["id"])
                        merged.append(c)
                        if len(merged) == final_top_k:
                            return merged
        return merged

    model = get_embed_model()
    # QUERY-SIDE EMBEDDING: Must include the BGE instruction prefix
    q_emb = model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()

    # Base WHERE clauses for metadata filtering
    where_clauses = []
    # For sparse keyword search, strip redundant company mentions since the 10-K corpus is single-company
    kw_query = re.sub(r"\b(apple('s)?|aapl)\b", "", query, flags=re.IGNORECASE).strip() or query
    params: dict[str, Any] = {"q_emb": str(q_emb), "query_text": kw_query}

    if company:
        where_clauses.append("company = :company")
        params["company"] = company
    if year:
        where_clauses.append("year = :year")
        params["year"] = int(year)
    if section:
        where_clauses.append("section ILIKE :section")
        params["section"] = f"%{section}%"
    if chunk_type:
        where_clauses.append("chunk_type = :chunk_type")
        params["chunk_type"] = chunk_type

    extra_filter = (" AND " + " AND ".join(where_clauses)) if where_clauses else ""

    with engine.connect() as conn:
        # 1. Dense Vector Search (top-20)
        vec_sql = f"""
            SELECT id, company, year, section, COALESCE(position_id, page) AS position_id,
                   chunk_type, image_path, text, (embedding <=> CAST(:q_emb AS vector)) AS dist
            FROM chunks
            WHERE embedding IS NOT NULL {extra_filter}
            ORDER BY embedding <=> CAST(:q_emb AS vector) ASC
            LIMIT {vector_top_k};
        """
        vec_rows = conn.execute(text(vec_sql), params).mappings().all()

        # 2. Sparse Keyword Search (top-20)
        kw_sql = f"""
            SELECT id, company, year, section, COALESCE(position_id, page) AS position_id,
                   chunk_type, image_path, text,
                   ts_rank_cd(search_vector, plainto_tsquery('english', :query_text)) AS rank
            FROM chunks
            WHERE search_vector @@ plainto_tsquery('english', :query_text) {extra_filter}
            ORDER BY rank DESC
            LIMIT {keyword_top_k};
        """
        kw_rows = conn.execute(text(kw_sql), params).mappings().all()

    # 3. Reciprocal Rank Fusion (RRF)
    # RRF score = sum(1 / (k + rank))
    rrf_scores: dict[int, float] = {}
    docs_by_id: dict[int, dict[str, Any]] = {}

    for rank_idx, row in enumerate(vec_rows, start=1):
        doc_id = row["id"]
        docs_by_id[doc_id] = dict(row)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (RRF_K + rank_idx))

    for rank_idx, row in enumerate(kw_rows, start=1):
        doc_id = row["id"]
        if doc_id not in docs_by_id:
            docs_by_id[doc_id] = dict(row)
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (RRF_K + rank_idx))

    if not docs_by_id:
        return []

    # Sort merged candidates by RRF score and take top 35 for reranking
    sorted_candidates = sorted(
        docs_by_id.values(),
        key=lambda d: rrf_scores[d["id"]],
        reverse=True,
    )[:35]

    # 4. Rerank using BAAI/bge-reranker-v2-m3
    reranker = get_reranker()
    pairs = [[query, c["text"]] for c in sorted_candidates]

    if hasattr(reranker, "compute_score"):
        # FlagReranker
        raw_scores = reranker.compute_score(pairs, normalize=True)
        if isinstance(raw_scores, float):
            raw_scores = [raw_scores]
    else:
        # CrossEncoder fallback
        import numpy as np
        logits = reranker.predict(pairs)
        raw_scores = (1 / (1 + np.exp(-logits))).tolist()

    for cand, score in zip(sorted_candidates, raw_scores):
        cand["rerank_score"] = float(score)

    # 5. Sort by rerank score descending and select final top_k
    final_results = sorted(
        sorted_candidates,
        key=lambda d: d["rerank_score"],
        reverse=True,
    )[:final_top_k]

    # Clean returned dicts
    cleaned: list[dict[str, Any]] = []
    for r in final_results:
        cleaned.append({
            "id": r["id"],
            "company": r["company"],
            "year": r["year"],
            "section": r["section"],
            "position_id": r["position_id"],
            "chunk_type": r["chunk_type"],
            "image_path": r.get("image_path"),
            "text": r["text"],
            "rerank_score": round(r["rerank_score"], 4),
        })

    return cleaned
