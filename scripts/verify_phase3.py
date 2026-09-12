"""
scripts/verify_phase3.py — Sanity-check Phase 3 outputs in Neon
===============================================================

Checks:
  1. Row count == 731
  2. Zero null embeddings
  3. Zero null search_vectors
  4. Per-year and chunk-type distribution
  5. Live cosine similarity query against Neon to confirm topical correctness
"""

import os
import sys
from dotenv import load_dotenv
from sqlalchemy import text
from sentence_transformers import SentenceTransformer

from db import engine

load_dotenv()

def verify():
    print("=" * 70)
    print("  PHASE 3 CHECKPOINT VERIFICATION — NEON POSTGRES")
    print("=" * 70)

    with engine.connect() as conn:
        # 1. Total row count
        total_rows = conn.execute(text("SELECT COUNT(*) FROM chunks;")).scalar()
        print(f"\n[1] Total rows in `chunks`: {total_rows}  (Expected: 731)")
        assert total_rows == 731, f"Expected 731 rows, found {total_rows}"
        print("    --> PASS")

        # 2. Null embeddings
        null_embs = conn.execute(text("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL;")).scalar()
        print(f"\n[2] Chunks with NULL embedding: {null_embs}  (Expected: 0)")
        assert null_embs == 0, f"Found {null_embs} null embeddings"
        print("    --> PASS")

        # 3. Null search_vectors
        null_sv = conn.execute(text("SELECT COUNT(*) FROM chunks WHERE search_vector IS NULL;")).scalar()
        print(f"\n[3] Chunks with NULL search_vector: {null_sv}  (Expected: 0)")
        assert null_sv == 0, f"Found {null_sv} null search_vectors"
        print("    --> PASS")

        # 4. Per-year breakdown
        print("\n[4] Distribution by Fiscal Year & Chunk Type:")
        rows = conn.execute(
            text("SELECT year, chunk_type, COUNT(*) FROM chunks GROUP BY year, chunk_type ORDER BY year, chunk_type;")
        ).fetchall()
        for r in rows:
            print(f"    Year {r[0]} | Type: {r[1]:<6} | Count: {r[2]}")

        # 5. Live cosine similarity test
        test_query = "What are Apple's principal risks regarding supply chain disruptions and suppliers?"
        print(f"\n[5] Cosine Similarity Test Query:\n    \"{test_query}\"")
        
        # Note: At query time, BGE requires the query instruction prefix:
        prefix = "Represent this sentence for searching relevant passages: "
        model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"))
        q_emb = model.encode(prefix + test_query, normalize_embeddings=True).tolist()

        res = conn.execute(
            text("""
                SELECT id, company, year, section, chunk_type, LEFT(text, 180) AS snippet,
                       1 - (embedding <=> :q_vec::vector) AS similarity
                FROM chunks
                ORDER BY embedding <=> :q_vec::vector
                LIMIT 3;
            """),
            {"q_vec": str(q_emb)}
        ).fetchall()

        print("\n    Top 3 Hits:")
        for idx, hit in enumerate(res, 1):
            print(f"\n    Hit #{idx} [Score: {hit[6]:.4f}] — {hit[1]} {hit[2]} | {hit[3]} ({hit[4]}):")
            print(f"    \"{hit[5]}...\"")

        print("\n" + "=" * 70)
        print("  ALL PHASE 3 CHECKPOINT TESTS PASSED SUCCESSFULLY!")
        print("=" * 70)

if __name__ == "__main__":
    verify()
