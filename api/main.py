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
from generation.router import route_and_execute
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

from agent.loop import run_agent_loop

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., json_schema_extra={"example": "What was Apple's total debt in 2024?"})
    year: Optional[int] = Field(None, json_schema_extra={"example": 2024}, description="Optional fiscal year filter (2021-2025)")
    top_k: int = Field(5, ge=1, le=10, description="Number of top chunks to return")


class AgentQueryRequest(BaseModel):
    question: str = Field(..., json_schema_extra={"example": "How did debt and FX risk change from 2022 to 2025?"})
    year: Optional[int] = Field(None, json_schema_extra={"example": 2024}, description="Optional fiscal year filter (2021-2025)")
    mode: str = Field("agent", json_schema_extra={"example": "agent"}, description="'agent' for iterative tool loop, 'fast' for single-pass pipeline")



class CitationItem(BaseModel):
    year: Optional[int] = None
    section: Optional[str] = None
    position_id: Optional[int] = None
    chunk_type: Optional[str] = None
    image_path: Optional[str] = None
    text: Optional[str] = None


class AgentTraceStep(BaseModel):
    step: int
    thought: str
    action: str
    args: dict[str, Any]
    observation: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationItem]
    confidence: float
    abstain_reason: Optional[str] = None
    retrieved_chunks: list[dict[str, Any]] = []


class AgentQueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationItem]
    confidence: float
    mode: str
    steps_used: int
    trace: list[AgentTraceStep] = []
    verification: Optional[dict[str, Any]] = None
    abstain_reason: Optional[str] = None



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
    Execute full RAG pipeline with intent-based routing:
      1. Auto-detect year metadata if present
      2. Intent classification (CALCULATION vs QUALITATIVE vs GREETING)
      3. Hybrid dense vector + sparse BM25 search with Reciprocal Rank Fusion & Cross-Encoder
      4. Grounded citation-aware answer generation or deterministic Pandas calculation
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

    try:
        route_result = route_and_execute(
            question=q,
            year=year_filter,
            company="AAPL",
        )
    except Exception as e:
        log.error("Routing/Execution error: %s", e)
        raise HTTPException(status_code=500, detail=f"Query execution failure: {e}")

    # Convert citations to CitationItem models
    raw_citations = route_result.get("citations", [])
    citation_models = []
    for c in raw_citations:
        citation_models.append(
            CitationItem(
                year=c.get("year"),
                section=c.get("section"),
                position_id=c.get("position_id"),
                chunk_type=c.get("chunk_type"),
                image_path=c.get("image_path"),
                text=c.get("text"),
            )
        )

    return QueryResponse(
        question=q,
        answer=route_result.get("answer", "No answer generated."),
        citations=citation_models,
        confidence=route_result.get("confidence", 0.0),
        abstain_reason=route_result.get("abstain_reason"),
        retrieved_chunks=[],
    )


@app.post("/api/agent_query", response_model=AgentQueryResponse)
def query_agent(request: AgentQueryRequest):
    """
    Execute Agentic RAG loop with tool use, evidence grading, and claim verification.
    Supports mode='agent' (iterative multi-step planner) or mode='fast' (single-pass pipeline).
    """
    q = request.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    year_filter = request.year
    if not year_filter:
        match = re.search(r"\b(202[1-5])\b", q)
        if match:
            year_filter = int(match.group(1))

    mode_selected = request.mode.lower()
    if mode_selected not in ("agent", "fast"):
        mode_selected = "agent"

    try:
        if mode_selected == "fast":
            fast_res = route_and_execute(question=q, year=year_filter, company="AAPL")
            raw_citations = fast_res.get("citations", [])
            citations_list = [
                CitationItem(
                    year=c.get("year"),
                    section=c.get("section"),
                    position_id=c.get("position_id"),
                    chunk_type=c.get("chunk_type"),
                    image_path=c.get("image_path"),
                    text=c.get("text"),
                )
                for c in raw_citations
            ]
            trace_steps = [
                AgentTraceStep(
                    step=1,
                    thought="Direct single-pass execution via fast pipeline.",
                    action="route_and_execute",
                    args={"year": year_filter, "category": fast_res.get("category")},
                    observation=f"Completed {fast_res.get('category')} route. Top rerank score: {fast_res.get('top_rerank_score', 1.0):.3f}",
                )
            ]
            return AgentQueryResponse(
                question=q,
                answer=fast_res.get("answer", "No answer generated."),
                citations=citations_list,
                confidence=fast_res.get("confidence", 0.0),
                mode="fast",
                steps_used=1,
                trace=trace_steps,
                verification={"all_passed": True, "supported_claims": [], "removed_claims": []},
                abstain_reason=fast_res.get("abstain_reason"),
            )
        else:
            agent_res = run_agent_loop(question=q, year=year_filter)
            raw_citations = agent_res.get("citations", [])
            citations_list = [
                CitationItem(
                    year=c.get("year"),
                    section=c.get("section"),
                    position_id=c.get("position_id"),
                    chunk_type=c.get("chunk_type"),
                    image_path=c.get("image_path"),
                    text=c.get("text"),
                )
                for c in raw_citations
            ]
            raw_trace = agent_res.get("trace", [])
            trace_steps = [
                AgentTraceStep(
                    step=t.get("step", idx),
                    thought=t.get("thought", ""),
                    action=t.get("action", ""),
                    args=t.get("args", {}),
                    observation=t.get("observation", ""),
                )
                for idx, t in enumerate(raw_trace, 1)
            ]
            return AgentQueryResponse(
                question=q,
                answer=agent_res.get("answer", "No answer generated."),
                citations=citations_list,
                confidence=agent_res.get("confidence", 0.0),
                mode=agent_res.get("mode", "agent"),
                steps_used=agent_res.get("steps_used", len(trace_steps)),
                trace=trace_steps,
                verification=agent_res.get("verification"),
                abstain_reason=None if agent_res.get("confidence", 0.0) > 0 else "Insufficient evidence",
            )
    except Exception as e:
        log.error("Agent query failure: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent execution failure: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

