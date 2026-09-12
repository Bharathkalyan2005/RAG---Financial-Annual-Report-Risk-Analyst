"""
db.py — Database layer for fin-risk-analyst
============================================
Responsibilities:
  • Load DATABASE_URL from .env (never hardcoded)
  • Create a SQLAlchemy engine with connection pooling tuned for Neon Postgres
  • Declare the ORM models: `chunks` and `financials`
  • Expose `get_session()` as a context-manager for use throughout the app

Tables
------
chunks      — One row per text/table chunk extracted from a 10-K filing.
              Stores the raw text, its 1024-dim pgvector embedding, and a
              tsvector column for fast full-text (keyword) search.

financials  — Structured key-value store for computed financial metrics.
              All arithmetic is done in Python/Pandas, NEVER by the LLM.
"""

import os
from contextlib import contextmanager

from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    Float,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ---------------------------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()  # reads .env in the current working directory

DATABASE_URL: str | None = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise EnvironmentError(
        "DATABASE_URL is not set. "
        "Copy .env.example → .env and fill in your Neon connection string."
    )

# ---------------------------------------------------------------------------
# 2. SQLAlchemy engine — tuned for Neon's managed Postgres
#    Neon sessions are ephemeral; keep the pool lean and recycle often.
# ---------------------------------------------------------------------------
engine = create_engine(
    DATABASE_URL,
    pool_size=5,           # max persistent connections
    max_overflow=5,        # extra burst connections
    pool_timeout=30,       # seconds to wait for a connection from the pool
    pool_recycle=1800,     # recycle connections every 30 min (Neon idle limit)
    pool_pre_ping=True,    # discard stale connections automatically
    connect_args={
        "sslmode": "require",          # Neon always requires TLS
        "connect_timeout": 10,
    },
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# ---------------------------------------------------------------------------
# 3. ORM base
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# 4. Table: chunks
#    Stores one text/table chunk per row, with its vector embedding and the
#    pre-computed tsvector for Postgres full-text search.
# ---------------------------------------------------------------------------
class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # ── Provenance ──────────────────────────────────────────────────────────
    company = Column(String(128), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)       # fiscal year, e.g. 2023
    section = Column(String(256), nullable=True)             # e.g. "Risk Factors", "MD&A"
    page = Column(Integer, nullable=True)                    # PDF page number (1-based) or position_id
    position_id = Column(Integer, nullable=True)             # synthetic HTML position id
    chunk_index = Column(Integer, nullable=True)             # chunk index within file/section
    chunk_type = Column(String(32), nullable=False, default="text")  # "text" | "table"

    # ── Content ─────────────────────────────────────────────────────────────
    text = Column(Text, nullable=False)

    # ── Semantic search — 1024-dim vector (BAAI/bge-large-en-v1.5) ─────────
    # pgvector extension must be enabled before the table is created.
    embedding = Column(Vector(1024), nullable=True)

    # ── Keyword search — Postgres built-in full-text search ─────────────────
    # Populated via a trigger or during ingestion.
    search_vector = Column(TSVECTOR, nullable=True)

    # ── Indexes ──────────────────────────────────────────────────────────────
    __table_args__ = (
        # GIN index for tsvector (fast keyword lookup)
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
        # Composite index for filtering by company + year before vector search
        Index("ix_chunks_company_year", "company", "year"),
        # NOTE: the pgvector HNSW/IVFFlat index on `embedding` is created
        #       separately after the first batch of embeddings is ingested,
        #       because it requires the table to be populated first.
        #       See: ingestion/build_vector_index.py  (Phase 1)
    )

    def __repr__(self) -> str:
        return (
            f"<Chunk id={self.id} company={self.company!r} "
            f"year={self.year} page={self.page} type={self.chunk_type!r}>"
        )


# ---------------------------------------------------------------------------
# 5. Table: financials
#    Structured key-value store for computed financial metrics.
#    All arithmetic is performed in Python/Pandas (see financials/ module).
#    This table stores the RESULTS only — never raw LLM output.
# ---------------------------------------------------------------------------
class Financial(Base):
    __tablename__ = "financials"

    id = Column(Integer, primary_key=True, autoincrement=True)

    company = Column(String(128), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    metric = Column(String(256), nullable=False)   # e.g. "revenue", "gross_margin_pct"
    value = Column(Float, nullable=True)           # numeric result (Python-computed)

    __table_args__ = (
        # Composite index: common query pattern is "give me all metrics for AAPL 2023"
        Index("ix_financials_company_year_metric", "company", "year", "metric"),
    )

    def __repr__(self) -> str:
        return (
            f"<Financial id={self.id} company={self.company!r} "
            f"year={self.year} metric={self.metric!r} value={self.value}>"
        )


# ---------------------------------------------------------------------------
# 6. Convenience helpers
# ---------------------------------------------------------------------------
def create_tables() -> None:
    """Create all tables that don't exist yet. Safe to call repeatedly."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session():
    """
    Yield a SQLAlchemy Session, committing on success and rolling back on error.

    Usage:
        with get_session() as session:
            session.add(chunk)
    """
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
