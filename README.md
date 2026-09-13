# Financial Annual Report Risk Analyst

A production-grade multi-year RAG (Retrieval-Augmented Generation) system that analyzes 5 years of Apple Inc. SEC 10-K filings (2021–2025) and answers financial and risk questions with retrieval-backed, cited answers and deterministic financial ratio calculations.

---

## Architecture

```
                                 User Question
                                       │
                                       ▼
                             ┌──────────────────┐
                             │  Streamlit UI    │  (Port 8501)
                             └─────────┬────────┘
                                       │ HTTP / REST
                                       ▼
                             ┌──────────────────┐
                             │  FastAPI Backend │  (Port 8000)
                             └─────────┬────────┘
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 ▼                     ▼                     ▼
        [Hybrid Retrieval]     [Financial Ratios]     [Risk Scoring]
         Dense Vector (BGE)      Audited Tables       7 Categories
                 +              Pure Pandas Math     0.3*Freq + 0.3*YoY
         Sparse BM25 (tsvector)  (Zero LLM Math)       + 0.4*Emphasis
                 │
                 ▼
        [RRF Candidate Merge]
         Reciprocal Rank Fusion (k=60)
                 │
                 ▼
        [Cross-Encoder Reranker]
         ms-marco-MiniLM-L-6-v2 (Local, Fast)
                 │
                 ▼
        [LLM Generation Layer]
         OpenRouter Financial API (inclusionai/ling-3.0-flash-fin)
                 │
                 ▼
       Cited Answer (Strict JSON) + Rerank Confidence + Abstention
```

---

## Stack

| Component | Technology | Description |
|---|---|---|
| **LLM Generation** | OpenRouter API (`inclusionai/ling-3.0-flash-fin:free`) | Financial-specialist model with strict JSON citation output |
| **Embeddings** | `BAAI/bge-large-en-v1.5` | 1024-dimensional dense vectors with asymmetric query prefix |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Neural cross-encoder scoring merged candidates |
| **Vector DB** | Neon (Managed Postgres + pgvector) | Cosine distance (`<=>`) vector index + HNSW |
| **Keyword Search** | PostgreSQL `tsvector` & `plainto_tsquery` | Full-text BM25-style inverted index |
| **Backend API** | FastAPI + Uvicorn | Async REST endpoints for query, financials, risks, and health |
| **Frontend UI** | Streamlit | Interactive dashboard with KPI cards, line charts, risk matrix & chat |
| **Financial Math** | Pure Pandas | Deterministic arithmetic (Gross Margin, Net Margin, Current Ratio, etc.) |
| **Containerization** | Docker & Docker Compose | Multi-container setup for local development and cloud deployment |

---

## Empirical Evaluation Benchmark (Phase 9)

Evaluated against `eval/testset.jsonl` (30 ground-truth questions across 5 fiscal years 2021–2025):

| Metric | Baseline (Dense Vector Only) | Improved (Hybrid + Filter + Reranker) | Quality Lift (%) |
|---|---|---|---|
| **Recall@1** | 0.2333 (23.3%) | **0.6667 (66.7%)** | **+185.8%** |
| **Recall@3** | 0.4667 (46.7%) | **0.7667 (76.7%)** | **+64.3%** |
| **Recall@5** | 0.6000 (60.0%) | **0.8333 (83.3%)** | **+38.9%** |
| **MRR (Mean Reciprocal Rank)** | 0.3672 | **0.7278** | **+98.2%** |
| **Year Metadata Precision** | 0.4000 (40.0%) | **0.8467 (84.7%)** | **+111.7%** |

*Detailed metrics stored in `eval/results/baseline.json` and `eval/results/improved.json`.*

---

## Key Financial & Risk Findings (Apple 2021–2025)

1. **Disciplined Deleveraging:** Apple systematically reduced total debt from **$115.1B (2021)** to **$86.3B (2025)** while scaling annual revenue from **$365.8B** to **$416.2B**.
2. **Expanding Margins:** Gross margin expanded steadily from **41.78% (2021)** to **46.91% (2025)** driven by higher-margin Services growth.
3. **Escalating Regulatory Risk:** Regulatory risk composite score escalated to **0.990** (highest among all 7 categories) driven by the EU Digital Markets Act (DMA), App Store alternative billing mandates, and US Department of Justice antitrust litigation.
4. **Resilient Liquidity:** Current ratio remained stable around **0.87–1.07** supported by tens of billions in annual operating cash flow.

---

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/fin-risk-analyst.git
cd fin-risk-analyst

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate         # Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Configure secrets
cp .env.example .env
# Fill in OPENROUTER_API_KEY and DATABASE_URL
```

### 2. Run the Full Pipeline

```bash
# Ingestion & Chunking
python -m ingestion.extract
python -m ingestion.chunk

# Embeddings & Neon Storage
python -m ingestion.embed

# Financial Metrics & Pure Pandas Ratios
python -m financials.extract_numbers
python -m financials.ratios

# Multi-Factor Risk Detection & Scoring
python -m risk.detect
python -m risk.score

# Systematic Evaluation Benchmark
python -m eval.run_eval
```

### 3. Launch Applications

**Start the FastAPI Backend:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# API Docs: http://localhost:8000/docs
```

**Start the Streamlit UI:**
```bash
streamlit run streamlit_app/main.py --server.port 8501
# Web UI: http://localhost:8501
```

### 4. Docker Deployment

```bash
# Build and run both backend and UI
docker-compose up --build
```

---

## Hard Engineering Rules

1. **Zero LLM Arithmetic:** All financial metrics and ratios are calculated strictly in Python/Pandas to eliminate hallucination.
2. **Verified Citations Required:** Every factual statement must cite its exact filing year, section, and position ID, or abstain when confidence is low.
3. **Graceful Abstention:** Queries outside the scope of the filings (e.g. "What is the CEO's favorite movie?") abstain immediately with `confidence: 0.0` rather than hallucinating.

---

## Known Limitations

- **Scanned / Image-only PDFs:** Current pipeline processes digital iXBRL HTML filings; optical character recognition (OCR) for non-digital scanned PDFs is out of scope for v1.
- **Single Company Scope:** Optimized for Apple Inc. (AAPL); expanding to multi-ticker comparative analysis requires automated CIK/ticker mapping.
- **Free-Tier LLM Rate Limits:** Public free OpenRouter endpoints enforce rate limits; production deployment should use paid provisioned throughput.
