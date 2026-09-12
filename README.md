# Financial Annual Report Risk Analyst

A multi-year RAG system that analyzes 5 years of Apple 10-K filings and answers financial and risk questions with retrieval-backed, cited answers.

## Architecture

```
User Question
     │
     ▼
[Streamlit UI]  ──────────────────────────────────────  FastAPI Backend
                                                              │
                                               ┌─────────────┼─────────────┐
                                               ▼             ▼             ▼
                                         [Hybrid         [Financial    [Risk
                                          Retrieval]      Ratios]      Scoring]
                                         pgvector +       Pandas        LLM
                                         tsvector         (no LLM)      tagged
                                               │
                                               ▼
                                         [Reranker]
                                         BAAI/bge-reranker-v2-m3 (local)
                                               │
                                               ▼
                                         [LLM Generation]
                                         OpenRouter API
                                         (with fallback model)
                                               │
                                               ▼
                                    Cited answer + abstention
```

## Stack

| Component | Technology |
|---|---|
| LLM | OpenRouter API (`nvidia/nemotron-3.5-lightning:free`, fallback `llama-3.1-8b`) |
| Embeddings | `BAAI/bge-large-en-v1.5` — local, 1024-dim, no API cost |
| Reranker | `BAAI/bge-reranker-v2-m3` — local, no API cost |
| Vector DB | Neon (managed Postgres + pgvector) |
| Keyword search | Postgres `tsvector` (built-in) |
| Backend | FastAPI |
| Frontend | Streamlit |
| HTML parsing | BeautifulSoup4 + lxml (SEC iXBRL 10-K filings) |
| Evaluation | RAGAS |

## Project Structure

```
fin-risk-analyst/
├── data/
│   ├── raw/                  # Source 10-K HTM files (TICKER_YEAR.htm) — gitignored
│   └── processed/            # Extracted JSON + chunks.jsonl — gitignored
│
├── ingestion/                # Phase 1 — Data pipeline
│   ├── extract.py            # SEC iXBRL HTML → structured JSON per year
│   └── chunk.py              # JSON sections → chunks.jsonl (500-token, overlap)
│
├── retrieval/                # Phase 2 — Hybrid search (Phase 5)
│   └── hybrid.py             # pgvector + tsvector + RRF merge + reranking
│
├── generation/               # Phase 3 — LLM answer layer (Phase 6)
│   ├── client.py             # OpenRouter wrapper + retry/fallback
│   └── answer.py             # Prompt builder + citation extractor + abstention
│
├── financials/               # Phase 4 — Structured financial extraction (Phase 7)
│   ├── extract_numbers.py    # LLM-assisted key metric extraction → DB
│   └── ratios.py             # Pandas ratio computation (never LLM arithmetic)
│
├── risk/                     # Phase 5 — Risk detection & scoring (Phase 8)
│   ├── detect.py             # Category tagging across years
│   └── score.py              # Weighted risk score formula
│
├── api/                      # FastAPI app (Phase 6)
│   └── main.py               # /query, /financials, /risk endpoints
│
├── streamlit_app/            # Streamlit UI (Phase 10)
│   └── main.py               # KPI cards, charts, chat, citations
│
├── eval/                     # Evaluation harness (Phase 9)
│   ├── testset.jsonl         # 100-150 hand-verified Q&A pairs
│   └── run_eval.py           # RAGAS: Recall@5, MRR, faithfulness
│
├── tests/                    # Automated tests
│   └── test_connection.py    # Phase 0 DB smoke test
│
├── scripts/                  # Developer utility scripts (not production)
│   ├── qa_chunks.py          # Chunk quality audit (coherence, table integrity)
│   ├── summary_report.py     # 5-year ingestion summary + outlier detection
│   └── inspect_output.py     # Quick JSON output inspector
│
├── db.py                     # SQLAlchemy engine + ORM models (chunks, financials)
├── requirements.txt          # All dependencies
├── .env.example              # Template — copy to .env and fill in secrets
├── .gitignore                # Excludes .env, venv, raw data, processed data
└── BUILD_GUIDE.md            # Phase-by-phase build instructions with checkpoints
```

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate           # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
copy .env.example .env             # Windows
# cp .env.example .env               # Mac/Linux
# Edit .env — set DATABASE_URL and OPENROUTER_API_KEY

# 4. Verify database connection
$env:PYTHONIOENCODING="utf-8"
python tests/test_connection.py
```

## Running the Pipeline

```bash
# Phase 1a — Extract text + tables from 10-K HTM files in data/raw/
python -m ingestion.extract

# Phase 1b — Chunk extracted JSON into JSONL (500-token chunks with overlap)
python -m ingestion.chunk

# Dev utilities — run from project root
python scripts/summary_report.py   # 5-year stats + outlier detection
python scripts/qa_chunks.py        # Chunk quality audit
```

## Hard Rules

1. **No secrets in code** — Everything goes in `.env`, which is gitignored.
2. **No LLM arithmetic** — All financial math (ratios, sums, %) is computed in Python/Pandas.
3. **Citations required** — Every answer includes `(year, position_id, section)` or abstains.
4. **Phase discipline** — Don't implement Phase N+1 until Phase N checkpoint passes.

## Build Progress

| Phase | Status | Description |
|---|---|---|
| 0 | ✅ Done | Environment, Neon DB, pgvector, ORM models |
| 1 | ✅ Done | HTML extraction + chunking (731 chunks / 5 years) |
| 2 | ⬜ Next | Embeddings + Neon vector storage |
| 3 | ⬜ | Hybrid retrieval + reranking |
| 4 | ⬜ | LLM generation with citations + abstention |
| 5 | ⬜ | Financial ratio extraction (Pandas) |
| 6 | ⬜ | Risk detection & scoring |
| 7 | ⬜ | RAGAS evaluation baseline + improvement |
| 8 | ⬜ | Streamlit UI |
| 9 | ⬜ | Deployment |

## Evaluation Results

*Populated after Phase 9 (RAGAS eval)*

| Version | Recall@5 | MRR | Faithfulness | Answer Relevance |
|---|---|---|---|---|
| Baseline | — | — | — | — |
