# Financial Annual Report Risk Analyst — Build Guide

Stack: OpenRouter (LLM) · Neon (Postgres+pgvector) · local embeddings/reranker · FastAPI · Streamlit · Claude Code

How to use this doc: work top to bottom. Each phase has a **goal**, **commands/prompts**, and a **checkpoint** — do not move to the next phase until the checkpoint passes. Skipping checkpoints is how RAG projects silently rot (they still run, they just answer wrong).

---

## Phase 0 — Environment & Repo Setup

**Goal:** a working skeleton, DB connected, keys safe.

```bash
mkdir fin-risk-analyst && cd fin-risk-analyst
git init
python3.11 -m venv venv
source venv/bin/activate
```

Create folder structure:
```
fin-risk-analyst/
├── data/raw/              # downloaded PDFs
├── data/processed/        # extracted JSON
├── ingestion/             # PDF parsing, chunking
├── retrieval/             # hybrid search, reranking
├── generation/            # LLM calls, prompts
├── financials/            # ratio calc, structured data
├── risk/                  # risk detection & scoring
├── api/                   # FastAPI app
├── streamlit_app/         # UI
├── eval/                  # RAGAS harness, test set
├── tests/                 # unit tests
├── .env.example
├── .gitignore
└── requirements.txt
```

`.gitignore` must include: `.env`, `venv/`, `__pycache__/`, `data/raw/*.pdf`, `*.db`

`.env.example`:
```
OPENROUTER_API_KEY=
OPENROUTER_MODEL=nvidia/nemotron-3.5-lightning:free
OPENROUTER_FALLBACK_MODEL=meta-llama/llama-3.1-8b-instruct:free
DATABASE_URL=postgresql://user:pass@ep-xxx.neon.tech/dbname?sslmode=require
```

**Neon setup:**
1. Create project at console.neon.tech
2. Copy connection string into `.env` (never commit it)
3. Run in Neon's SQL editor: `CREATE EXTENSION IF NOT EXISTS vector;`

**Claude Code prompt:**
> "Set up this repo structure exactly as below [paste tree]. Add SQLAlchemy models for a `chunks` table (id, company, year, section, page, chunk_type, text, embedding vector(1024), search_vector tsvector) and a `financials` table (id, company, year, metric, value). Write a `db.py` with connection pooling to Neon. Add a script `test_connection.py` that inserts and reads one dummy row."

**Checkpoint (do not skip):**
- [ ] `python test_connection.py` succeeds
- [ ] `.env` is in `.gitignore` — run `git status` and confirm it does NOT appear as trackable
- [ ] `SELECT * FROM pg_extension;` in Neon shows `vector`

---

## Phase 1 — Data Collection

**Goal:** 5 years of real filings for one company.

- Go to https://www.sec.gov/cgi-bin/browse-edgar → search company (e.g. ticker) → filter filing type `10-K` → download last 5 years as PDF.
- Save as `data/raw/{TICKER}_{YEAR}.pdf` — consistent naming now saves pain later.

**Senior dev note:** Pick a company with genuinely interesting risk trends over 5 years (rising debt, a lawsuit, a supply chain event) — makes your demo answers non-trivial. Boring flat financials make for a boring interview demo.

**Checkpoint:**
- [ ] 5 PDFs present, named consistently, each opens and is text-based (not a scanned image — if scanned, you'll need OCR, which is extra scope you don't need for v1)

---

## Phase 2 — PDF Extraction

**Goal:** text + tables + page numbers, per document, in JSON.

**Claude Code prompt:**
> "Write `ingestion/extract.py` using PyMuPDF (fitz) to extract text per page, and pdfplumber to extract tables per page. Output one JSON file per PDF into data/processed/, structured as: `{company, year, pages: [{page_num, text, tables: [...]}, ...]}`. Handle pages with no tables gracefully."

Run it:
```bash
python ingestion/extract.py
```

**Checkpoint — manual, non-negotiable:**
- [ ] Open one output JSON, pick a page you know has a balance sheet, confirm the table extracted has correct rows/values (compare against the actual PDF page)
- [ ] Confirm page numbers in JSON match the PDF page numbers (off-by-one errors here break every citation downstream)

This is the phase most people skip verifying and pay for it three phases later.

---

## Phase 3 — Structure-Aware Chunking

**Goal:** chunks that respect document sections, not blind character splits.

**Claude Code prompt:**
> "Write `ingestion/chunk.py`. Detect section boundaries using heading patterns (e.g. 'Item 1A. Risk Factors', 'Item 7. Management's Discussion', 'Item 8. Financial Statements'). Chunk within sections at ~500 tokens with 50-token overlap. Keep tables as single chunks with `chunk_type='table'`, don't split them. Attach metadata: company, year, section, page, chunk_type. Output to data/processed/chunks.jsonl."

**Checkpoint:**
- [ ] Spot check 5 chunks — read them, confirm they're coherent (not cut mid-sentence in a way that loses meaning)
- [ ] Confirm table chunks are intact, not split across multiple chunks
- [ ] Count total chunks — sanity check it's in a reasonable range (a 10-K is usually 100-300 chunks depending on chunk size)

---

## Phase 4 — Embeddings + Storage

**Goal:** every chunk embedded and stored in Neon with metadata.

```bash
pip install sentence-transformers
```

**Claude Code prompt:**
> "Write `ingestion/embed.py`. Load `BAAI/bge-large-en-v1.5` via sentence-transformers. For each chunk in chunks.jsonl, generate an embedding, and insert into the `chunks` Postgres table (chunk text, metadata, embedding, and a `search_vector` tsvector column generated from the text for keyword search). Batch inserts for efficiency."

**Senior dev note:** bge models expect a specific query prefix for asymmetric search (`"Represent this sentence for searching relevant passages: "` on the query side, none on the document side) — check the model card. Getting this backwards silently degrades retrieval quality without erroring.

**Checkpoint:**
- [ ] `SELECT count(*) FROM chunks;` matches your chunk count
- [ ] Run `SELECT embedding FROM chunks LIMIT 1;` — confirm it's a real vector, not null/zeros
- [ ] Run a raw cosine similarity query for one hand-picked question, confirm the top result is topically related

---

## Phase 5 — Hybrid Retrieval + Reranking

**Goal:** vector + keyword search merged, then reranked.

**Claude Code prompt:**
> "Write `retrieval/hybrid.py`. Implement: (1) vector search via pgvector `<=>` operator, top 20. (2) Postgres full-text search on `search_vector`, top 20. (3) Merge both lists using reciprocal rank fusion. (4) Rerank the merged top-20 using local `BAAI/bge-reranker-v2-m3`, return top 5 with scores."

```bash
pip install FlagEmbedding  # for the reranker
```

**Checkpoint — this is the most important checkpoint in the whole project:**
- [ ] Write 10 questions you know the answer to from the PDFs (e.g. "What was total debt in 2023?")
- [ ] Run each through `hybrid.py` directly (no LLM yet), read the actual chunks returned
- [ ] At least 8/10 should return the chunk that actually contains the answer in the top 3

If this fails, do not proceed to generation — fix chunking/retrieval first. A bad retriever wrapped in a good LLM just produces confident wrong answers.

---

## Phase 6 — LLM Generation Layer (OpenRouter)

**Goal:** citation-aware answers, with retries and abstention.

**Claude Code prompt:**
> "Write `generation/client.py` — a wrapper around OpenRouter's chat completions endpoint. Support model + fallback model via env vars. Add exponential backoff retry (max 3 attempts) for 429/5xx errors. Write `generation/answer.py` that takes a question + retrieved chunks, builds a prompt forcing JSON output `{answer, citations: [{year, page, section}], confidence: 0-1}`, calls the client, and parses the JSON (strip markdown fences if present). If reranker's top score is below a configurable threshold, skip the LLM call and return `{answer: 'Insufficient evidence', citations: [], confidence: 0}`."

**Senior dev note on the free model:** `nvidia/nemotron-3.5-lightning:free` and similar free OpenRouter models have low rate limits (often 20 req/min, sometimes daily caps) and can get deprecated without much notice. Build the fallback model switch now — you'll need it during eval (Phase 9) when you're firing 100+ requests.

**Checkpoint:**
- [ ] Ask a question with a clear answer in the docs → confirm correct answer + correct citation
- [ ] Ask a question NOT covered by the docs (e.g. "What's the CEO's favorite color?") → confirm it abstains, doesn't hallucinate
- [ ] Deliberately trigger a rate limit (fire 25 requests fast) → confirm fallback model kicks in instead of crashing

---

## Phase 7 — Structured Financials + Ratio Calculations

**Goal:** numbers computed by code, never by the LLM.

**Claude Code prompt:**
> "Write `financials/extract_numbers.py` — for each year's chunks tagged `chunk_type='table'` in Financial Statements section, use the LLM once per document to extract key line items (revenue, net income, total debt, total equity, current assets, current liabilities) into structured JSON, then insert into the `financials` table. Write `financials/ratios.py` using Pandas to compute: gross margin, net margin, ROA, ROE, current ratio, quick ratio, debt-to-equity, debt-to-assets, interest coverage — from the `financials` table, not from the LLM."

**Checkpoint:**
- [ ] Manually verify 3 extracted numbers against the actual PDF (e.g. 2023 total debt)
- [ ] Manually recompute one ratio by hand, confirm it matches Pandas output exactly

---

## Phase 8 — Risk Detection + Scoring

**Goal:** transparent, defensible risk tracking across years.

**Claude Code prompt:**
> "Write `risk/detect.py`. Define risk categories: Liquidity, Credit, FX, Regulatory, Cybersecurity, Operational, Legal. For each year, retrieve chunks from the Risk Factors section, use the LLM to tag mentions by category and extract a one-sentence justification per mention. Write `risk/score.py` implementing: `score = (0.3 * mention_frequency_normalized) + (0.3 * yoy_change) + (0.4 * management_emphasis_score)`, where management_emphasis_score comes from keyword intensity (e.g. 'significant', 'material', 'substantially increased'). Document the formula inline with comments explaining each weight."

**Senior dev note:** keep the weights as named constants at the top of the file, not magic numbers buried in logic. You will be asked "why 0.3?" in an interview — have an honest answer ("chosen empirically / equally weighted as a starting point, tunable") rather than pretending it's rigorously derived.

**Checkpoint:**
- [ ] For your test company, does the top risk category match what the actual filings emphasize? (read the 2025 Risk Factors section yourself and sanity check)

---

## Phase 9 — Evaluation (this is what makes the project credible)

**Goal:** measurable retrieval quality, with a documented before/after improvement.

```bash
pip install ragas
```

1. **Write the test set yourself.** Go through the PDFs, write 100-150 Q&A pairs with known answer + year + page. Draft with the agent if you want speed, but personally verify every single one — a wrong test set gives you fake numbers you can't defend.

Format (`eval/testset.jsonl`):
```json
{"question": "What was total debt in 2024?", "expected_answer": "$18B", "expected_year": 2024, "expected_page": 102}
```

**Claude Code prompt:**
> "Write `eval/run_eval.py` using RAGAS to measure Recall@5, MRR, faithfulness, and answer relevance against eval/testset.jsonl, running the full retrieval+generation pipeline. Output a results table and save to eval/results/baseline.json."

2. Run baseline, record numbers.
3. Iterate — try: bigger/smaller chunks, metadata filtering (restrict search to the right year when the question specifies one), adjusting hybrid fusion weights, adjusting reranker threshold.
4. Re-run eval after each change, save each result (`eval/results/v2.json`, etc.) so you have a real improvement curve to show.

**Checkpoint:**
- [ ] You have at least a baseline and one improved version with numbers to compare
- [ ] You can explain *why* each change helped or didn't (this is the actual interview material)

---

## Phase 10 — Streamlit UI

**Claude Code prompt:**
> "Build streamlit_app/main.py: KPI cards row (revenue, profit, debt for latest year), a line chart of revenue/debt over 5 years using the financials table, a risk score bar chart, a chat input wired to the FastAPI backend, and citation cards under each answer showing year/page/section — expandable to show the source chunk text."

**Checkpoint:**
- [ ] Full flow works end to end: ask a question in the UI → see answer → expand citation → see the actual source text

---

## Phase 11 — Ship

```bash
# Dockerize backend only — Neon is already hosted
```

- `Dockerfile` for FastAPI backend, `docker-compose.yml` if you want to run backend+Streamlit together locally.
- Push to GitHub — commit per phase (not one giant commit), so your history tells the build story.
- Deploy Streamlit → streamlit.io/cloud (connect GitHub repo, add secrets in their dashboard, not in code).
- Deploy FastAPI backend → Render or Railway free tier.
- Write README with: architecture diagram, setup steps, your eval results table (baseline vs improved), and known limitations (be honest — e.g. "table extraction from scanned PDFs not supported").

---

## Order-of-operations summary (checklist form)

```
[ ] Phase 0  — Repo + Neon connected
[ ] Phase 1  — 5 real 10-Ks downloaded
[ ] Phase 2  — Extraction verified against source PDF
[ ] Phase 3  — Chunks verified coherent + tables intact
[ ] Phase 4  — Embeddings stored, vector search sanity-checked
[ ] Phase 5  — Hybrid+rerank hits 8/10 on hand-checked questions
[ ] Phase 6  — Generation: correct answers + citations + abstention working
[ ] Phase 7  — Financial numbers manually verified, ratios hand-checked
[ ] Phase 8  — Risk scores sanity-checked against actual filings
[ ] Phase 9  — Baseline + improved eval numbers recorded
[ ] Phase 10 — Full UI flow working end to end
[ ] Phase 11 — Deployed, README written, GitHub history clean
```

Do not let Claude Code jump ahead of your current phase — if it offers to "also add X" from a later phase, defer it. Scope creep mid-phase is the main way vibe-coded projects become unmaintainable.
