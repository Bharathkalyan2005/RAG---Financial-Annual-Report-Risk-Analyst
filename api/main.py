"""
api/main.py — Phase 10: FastAPI Backend for Financial Risk Analyst RAG
=====================================================================

Endpoints:
  - GET  /health            — Service & database health check
  - GET  /api/financials    — 5-year financial metrics & computed ratios
  - GET  /api/risks         — Multi-year composite risk scores & top risk categories
  - GET  /api/eval          — Baseline vs. Improved (v2) retrieval evaluation metrics
  - POST /api/query         — End-to-end RAG question answering with citations & source chunks
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import text

from db import engine
from financials.ratios import compute_all_ratios, load_financials_df
from generation.answer import generate_answer
from retrieval.hybrid import retrieve_hybrid

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("api.main")

app = FastAPI(
    title="Financial Annual Report Risk Analyst API",
    description="Multi-Year SEC 10-K RAG System with Hybrid Retrieval, Cross-Encoder Reranking & Quantitative Ratio Analysis",
    version="1.0.0",
)

# Enable CORS for local Streamlit development & deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., example="What was Apple's total debt in 2024?")
    year: Optional[int] = Field(None, example=2024, description="Optional fiscal year filter (2021-2025)")
    top_k: int = Field(5, ge=1, le=10, description="Number of top chunks to return")


class CitationItem(BaseModel):
    year: Optional[int] = None
    section: Optional[str] = None
    position_id: Optional[int] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationItem]
    confidence: float
    abstain_reason: Optional[str] = None
    retrieved_chunks: list[dict[str, Any]]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Verify API status and database connectivity."""
    db_ok = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
            db_ok = True
    except Exception as e:
        log.error("Database connection error: %s", e)

    return {
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected",
        "model_reranker": os.getenv("RERANKER_MODEL", "ms-marco-MiniLM-L-6-v2"),
        "model_llm": os.getenv("OPENROUTER_MODEL", "inclusionai/ling-3.0-flash-fin:free"),
    }


@app.get("/api/financials")
def get_financials(company: str = "AAPL"):
    """Retrieve 5-year structured financials and pure Pandas calculated ratios."""
    try:
        df_fin = load_financials_df(company=company)
        ratios_df = compute_all_ratios(df_fin)

        # Structure as JSON
        financials_dict = {}
        for yr in df_fin.index:
            yr_int = int(yr)
            financials_dict[yr_int] = {
                "metrics": {k: float(v) for k, v in df_fin.loc[yr].dropna().items()},
                "ratios": {k: round(float(v), 4) for k, v in ratios_df.loc[yr].dropna().items()} if yr in ratios_df.index else {},
            }

        return {
            "company": company,
            "years": sorted(list(financials_dict.keys())),
            "data": financials_dict,
        }
    except Exception as e:
        log.error("Error loading financials: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/risks")
def get_risks():
    """Retrieve 5-year multi-factor composite risk scores."""
    scores_file = Path("data/processed/risk_scores.json")
    if not scores_file.exists():
        from risk.score import run_scoring
        run_scoring()

    try:
        with open(scores_file, encoding="utf-8") as f:
            scores = json.load(f)

        # Compute top risk per year
        top_risks = {}
        for yr, cats in scores.items():
            top_cat = max(cats.items(), key=lambda x: x[1])
            top_risks[yr] = {"category": top_cat[0], "score": top_cat[1]}

        return {
            "scores": scores,
            "top_risks": top_risks,
            "formula": "0.30*Frequency + 0.30*YoY_Change + 0.40*Management_Emphasis",
        }
    except Exception as e:
        log.error("Error loading risk scores: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/eval")
def get_eval_results():
    """Retrieve systematic retrieval evaluation metrics (Baseline vs. Improved)."""
    baseline_file = Path("eval/results/baseline.json")
    improved_file = Path("eval/results/improved.json")

    baseline_data = {}
    improved_data = {}

    if baseline_file.exists():
        with open(baseline_file, encoding="utf-8") as f:
            baseline_data = json.load(f)

    if improved_file.exists():
        with open(improved_file, encoding="utf-8") as f:
            improved_data = json.load(f)

    return {
        "baseline": baseline_data,
        "improved": improved_data,
    }


@app.post("/api/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Execute full RAG pipeline:
      1. Extract year metadata if present
      2. Hybrid dense vector + sparse BM25 search with Reciprocal Rank Fusion
      3. Cross-encoder neural reranking
      4. Grounded citation-aware answer generation via OpenRouter
    """
    q = request.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Auto-detect year if not explicitly provided
    year_filter = request.year
    if not year_filter:
        match = re.search(r"\b(202[1-5])\b", q)
        if match:
            year_filter = int(match.group(1))

    # 1. Retrieval
    try:
        chunks = retrieve_hybrid(
            query=q,
            year=year_filter,
            final_top_k=request.top_k,
        )
    except Exception as e:
        log.error("Retrieval error: %s", e)
        raise HTTPException(status_code=500, detail=f"Retrieval failure: {e}")

    # 2. Generation
    try:
        gen_result = generate_answer(
            question=q,
            retrieved_chunks=chunks,
        )
    except Exception as e:
        log.error("Generation error: %s", e)
        raise HTTPException(status_code=500, detail=f"Generation failure: {e}")

    return QueryResponse(
        question=q,
        answer=gen_result.get("answer", "No answer generated."),
        citations=gen_result.get("citations", []),
        confidence=gen_result.get("confidence", 0.0),
        abstain_reason=gen_result.get("abstain_reason"),
        retrieved_chunks=chunks,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
