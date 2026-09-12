"""
test_connection.py — Phase 0 smoke test
========================================
Verifies the full database bootstrap in one script:

  1. Connects to Neon using DATABASE_URL from .env
  2. Enables the pgvector extension if not already active
  3. Creates the `chunks` and `financials` tables if they don't exist
  4. Inserts one dummy Chunk row
  5. Reads it back and prints it
  6. Confirms pgvector is active by querying pg_extension
  7. Cleans up the dummy row

Expected output (success):
  ✓ Connected to Neon Postgres
  ✓ pgvector extension is ACTIVE (version: 0.x.x)
  ✓ Tables created (chunks, financials)
  ✓ Inserted dummy chunk: <Chunk id=1 ...>
  ✓ Read back:  id=1  company='TEST_CO'  year=2024  section='Risk Factors'  page=42  type='text'
  ✓ Dummy row cleaned up
  ─────────────────────────────────────────────────────
  All Phase 0 checks passed. Database is ready.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `db` module resolves
# whether this script is run from project root or from tests/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from db import Chunk, create_tables, engine, get_session


def check_connection() -> None:
    print("\n─── Phase 0: Database Connection Test ───────────────────────────────────")

    # ------------------------------------------------------------------
    # 1. Basic connectivity
    # ------------------------------------------------------------------
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✓ Connected to Neon Postgres")
    except Exception as exc:
        print(f"✗ Connection FAILED: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Enable pgvector extension
    # ------------------------------------------------------------------
    try:
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        print("✓ pgvector extension enabled (or already present)")
    except Exception as exc:
        print(f"✗ Could not enable pgvector: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Confirm pgvector is active + report version
    # ------------------------------------------------------------------
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
                )
            ).fetchone()
        if row:
            print(f"✓ pgvector extension is ACTIVE (version: {row[0]})")
        else:
            print("✗ pgvector extension not found in pg_extension — install it in Neon console")
            sys.exit(1)
    except Exception as exc:
        print(f"✗ Could not verify pgvector: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Create tables
    # ------------------------------------------------------------------
    try:
        create_tables()
        print("✓ Tables created/verified (chunks, financials)")
    except Exception as exc:
        print(f"✗ Table creation FAILED: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Insert a dummy chunk (embedding left NULL — no model loaded yet)
    # ------------------------------------------------------------------
    dummy_id: int | None = None
    try:
        with get_session() as session:
            dummy = Chunk(
                company="TEST_CO",
                year=2024,
                section="Risk Factors",
                page=42,
                chunk_type="text",
                text=(
                    "This is a dummy chunk inserted by test_connection.py "
                    "to verify the Phase 0 database setup. It will be deleted."
                ),
                embedding=None,      # populated during ingestion (Phase 1)
                search_vector=None,  # populated during ingestion (Phase 1)
            )
            session.add(dummy)
            session.flush()           # assigns the auto-incremented id
            dummy_id = dummy.id
            print(f"✓ Inserted dummy chunk: {dummy!r}")
    except Exception as exc:
        print(f"✗ Insert FAILED: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 6. Read it back
    # ------------------------------------------------------------------
    try:
        with get_session() as session:
            retrieved = session.get(Chunk, dummy_id)
            if retrieved is None:
                print(f"✗ Could not read back chunk id={dummy_id}")
                sys.exit(1)
            print(
                f"✓ Read back:  id={retrieved.id}  company={retrieved.company!r}  "
                f"year={retrieved.year}  section={retrieved.section!r}  "
                f"page={retrieved.page}  type={retrieved.chunk_type!r}"
            )
            print(f"  text preview: \"{retrieved.text[:80]}...\"")
    except Exception as exc:
        print(f"✗ Read-back FAILED: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 7. Clean up the dummy row
    # ------------------------------------------------------------------
    try:
        with get_session() as session:
            row_to_delete = session.get(Chunk, dummy_id)
            if row_to_delete:
                session.delete(row_to_delete)
        print("✓ Dummy row cleaned up")
    except Exception as exc:
        print(f"⚠  Cleanup failed (non-fatal): {exc}")

    # ------------------------------------------------------------------
    print("─" * 60)
    print("All Phase 0 checks passed. Database is ready.\n")


if __name__ == "__main__":
    check_connection()
