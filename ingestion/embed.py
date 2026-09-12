"""
ingestion/embed.py — Phase 2: Generate embeddings and store all chunks in Neon
===============================================================================

Pipeline
--------
  1. Load BAAI/bge-large-en-v1.5 via sentence-transformers (local, no API cost)
  2. Read all chunks from data/processed/chunks.jsonl
  3. Embed each chunk's text in batches of EMBED_BATCH_SIZE
  4. Insert into the Neon `chunks` table in batches of DB_BATCH_SIZE rows
  5. search_vector is generated at insert time using Postgres to_tsvector()

CRITICAL NOTE — asymmetric embedding (do NOT get this backwards):
─────────────────────────────────────────────────────────────────
  BGE models use asymmetric search:
    • DOCUMENT side (this file)  → embed raw text, NO prefix
    • QUERY side (retrieval)     → prepend the instruction prefix:
          "Represent this sentence for searching relevant passages: "
  Embedding documents WITH the prefix, or queries WITHOUT it, silently
  degrades cosine similarity scores — the model expects this asymmetry.
  See: https://huggingface.co/BAAI/bge-large-en-v1.5#usage

Runtime estimate (CPU, no GPU):
─────────────────────────────────
  ~700 chunks × avg 412 tokens ≈ 20–35 minutes on a modern CPU.
  With a CUDA GPU it drops to ~2–3 minutes.
  Progress bar shows estimated time remaining.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from tqdm import tqdm

from db import Chunk, SessionLocal, create_tables, engine

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

CHUNKS_FILE     = Path("data/processed/chunks.jsonl")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBED_BATCH     = 32    # chunks per sentence-transformers inference call
DB_BATCH        = 100   # rows per Postgres transaction

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("embed")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_chunks() -> list[dict[str, Any]]:
    """Read all chunks from chunks.jsonl."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"{CHUNKS_FILE} not found. "
            "Run `python -m ingestion.extract` then `python -m ingestion.chunk` first."
        )
    with open(CHUNKS_FILE, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_model() -> SentenceTransformer:
    """Load the BGE embedding model. Downloads on first run (~1.3 GB)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading model: %s  (device=%s)", EMBEDDING_MODEL, device)
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    log.info("Model loaded. Embedding dimension: %d", model.get_sentence_embedding_dimension())
    return model


def embed_in_batches(
    model: SentenceTransformer,
    texts: list[str],
) -> list[list[float]]:
    """
    Generate embeddings for all texts using batched inference.

    DOCUMENT-SIDE: no query prefix is added here.
    The prefix "Represent this sentence for searching relevant passages: "
    is ONLY added at query time (retrieval/hybrid.py).
    """
    all_embeddings: list[list[float]] = []

    for i in tqdm(
        range(0, len(texts), EMBED_BATCH),
        desc="Embedding batches",
        unit="batch",
        total=(len(texts) + EMBED_BATCH - 1) // EMBED_BATCH,
    ):
        batch = texts[i : i + EMBED_BATCH]
        # normalize_embeddings=True gives unit vectors — needed for cosine sim
        # with pgvector's <=> operator which computes 1 - cosine_similarity.
        vecs = model.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=EMBED_BATCH,
        )
        all_embeddings.extend(vecs.tolist())

    return all_embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Database insertion
# ─────────────────────────────────────────────────────────────────────────────

def insert_chunks(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> int:
    """
    Upsert chunks into the Neon `chunks` table in batches.

    search_vector is generated at insert time via Postgres to_tsvector()
    so we don't need to compute it in Python.

    Returns the total number of rows inserted.
    """
    assert len(chunks) == len(embeddings), "Chunk / embedding count mismatch"

    create_tables()   # idempotent — creates table only if it doesn't exist

    inserted = 0

    with tqdm(total=len(chunks), desc="Storing in Neon", unit="chunk") as pbar:
        for batch_start in range(0, len(chunks), DB_BATCH):
            batch_chunks     = chunks[batch_start : batch_start + DB_BATCH]
            batch_embeddings = embeddings[batch_start : batch_start + DB_BATCH]

            session = SessionLocal()
            try:
                rows: list[Chunk] = []
                for chunk, emb in zip(batch_chunks, batch_embeddings):
                    row = Chunk(
                        company     = chunk["company"],
                        year        = chunk["year"],
                        section     = chunk["section"],
                        page        = chunk["position_id"],   # citation page/position reference
                        position_id = chunk["position_id"],
                        chunk_index = chunk.get("chunk_index"),
                        chunk_type  = chunk["chunk_type"],
                        text        = chunk["text"],
                        embedding   = emb,
                        # search_vector populated below via raw SQL after flush
                    )
                    rows.append(row)
                    session.add(row)

                # Flush to get auto-assigned IDs before the tsvector update
                session.flush()

                # Generate tsvector in Postgres (avoids shipping ~50KB of text
                # through a Python tsvector library — let Postgres do it natively)
                ids = [r.id for r in rows]
                session.execute(
                    text(
                        "UPDATE chunks "
                        "SET search_vector = to_tsvector('english', text) "
                        "WHERE id = ANY(:ids)"
                    ),
                    {"ids": ids},
                )

                session.commit()
                inserted += len(rows)
                pbar.update(len(rows))

            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

    return inserted


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def embed_and_store() -> None:
    log.info("=" * 60)
    log.info("Phase 2 — Embedding + Storage")
    log.info("=" * 60)

    # 1. Load chunks
    chunks = load_chunks()
    log.info("Loaded %d chunks from %s", len(chunks), CHUNKS_FILE)
    by_year = {}
    for c in chunks:
        by_year[c["year"]] = by_year.get(c["year"], 0) + 1
    for yr, cnt in sorted(by_year.items()):
        log.info("  Year %s: %d chunks", yr, cnt)

    # 2. Load model
    model = load_model()

    # 3. Generate embeddings
    texts = [c["text"] for c in chunks]
    log.info("Generating embeddings for %d chunks (batch=%d) …", len(texts), EMBED_BATCH)
    embeddings = embed_in_batches(model, texts)
    log.info("Embeddings generated: %d vectors of dim %d", len(embeddings), len(embeddings[0]))

    # 4. Insert into Neon
    log.info("Inserting into Neon (batch=%d rows per transaction) …", DB_BATCH)
    total = insert_chunks(chunks, embeddings)

    log.info("=" * 60)
    log.info("Done. %d chunks stored in Neon `chunks` table.", total)
    log.info("")
    log.info("Sanity-check SQL (run in Neon console):")
    log.info("")
    log.info("  -- Row count")
    log.info("  SELECT COUNT(*) FROM chunks;")
    log.info("")
    log.info("  -- Confirm embeddings are populated (should return 0)")
    log.info("  SELECT COUNT(*) FROM chunks WHERE embedding IS NULL;")
    log.info("")
    log.info("  -- Confirm search_vector is populated (should return 0)")
    log.info("  SELECT COUNT(*) FROM chunks WHERE search_vector IS NULL;")
    log.info("")
    log.info("  -- Per-year breakdown")
    log.info("  SELECT year, chunk_type, COUNT(*) FROM chunks")
    log.info("  GROUP BY year, chunk_type ORDER BY year, chunk_type;")
    log.info("")
    log.info("  -- Peek at one embedding (should be a real vector, not null/zeros)")
    log.info("  SELECT id, company, year, LEFT(text,80),")
    log.info("         ROUND(embedding[1]::numeric,6) AS dim_1,")
    log.info("         ROUND(embedding[2]::numeric,6) AS dim_2")
    log.info("  FROM chunks LIMIT 3;")
    log.info("=" * 60)


if __name__ == "__main__":
    embed_and_store()
