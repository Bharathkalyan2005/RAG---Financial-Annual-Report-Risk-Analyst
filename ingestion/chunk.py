"""
ingestion/chunk.py — Phase 1 (cont.): Structure-aware chunking
===============================================================
Reads every JSON file in data/processed/ (output of extract.py),
splits sections into chunks, and writes data/processed/chunks.jsonl.

Chunking strategy
-----------------
TEXT chunks  — split within a section at ≈500 tokens with 50-token overlap.
               Boundaries are kept at sentence endings to avoid mid-thought cuts.
               Overlap copies the last sentence(s) of the previous chunk into
               the start of the next, giving the LLM needed context.

TABLE chunks — each table is kept as ONE atomic chunk (never split).
               The table is serialised as a Markdown-style grid string so it
               reads naturally in a prompt and is embeddable as a dense passage.

Chunk schema (one line of JSONL per chunk):
{
  "chunk_id":    "AAPL_2025_0042",   # TICKER_YEAR_NNNN  (zero-padded)
  "company":     "AAPL",
  "year":        2025,
  "section":     "Item 8. Financial Statements and Supplementary Data",
  "position_id": 13,                 # section position_id from extract.py
  "chunk_type":  "text" | "table",
  "chunk_index": 2,                  # 0-based index within this section
  "text":        "...",              # embeddable string
  "token_count": 487                 # approximate (word-based, not BPE)
}

Design decisions
----------------
• Token count uses a simple word-split estimator (÷ 0.75 ratio, common
  approximation for English financial prose). Real BPE tokenizers would be
  more accurate but add a heavy dependency here; accurate-enough for chunking.

• Sentence-boundary splitting: we scan backwards from the target cut point for
  the nearest '.', '!', or '?' followed by whitespace/end-of-string.  This
  prevents mid-sentence cuts without needing NLTK/spaCy (no extra deps).

• Degenerate sections: sections whose text is < MIN_TEXT_CHARS characters after
  stripping are skipped for text chunking (they're usually stub cross-references
  like "See incorporated portions of the Proxy Statement").  Their tables are
  still extracted.

• Empty tables: tables with all-empty cells (layout spacers that slipped
  through) are skipped.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_FILE   = PROCESSED_DIR / "chunks.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chunk")

# ─────────────────────────────────────────────────────────────────────────────
# Tuning parameters
# ─────────────────────────────────────────────────────────────────────────────
TARGET_TOKENS   = 500   # target size per text chunk
OVERLAP_TOKENS  = 50    # overlap with previous chunk
WORDS_PER_TOKEN = 0.75  # English prose approximation (1 token ≈ 0.75 words)
MIN_TEXT_CHARS  = 80    # skip text chunking for very short sections

# iXBRL context data that leaks into the Preamble section of SEC HTM files.
# These chunks contain taxonomy namespace prefixes, not readable prose.
# Skip any text chunk where more than this fraction of "words" are XBRL tokens.
XBRL_NOISE_THRESHOLD = 0.25   # if >25% of words look like XBRL, skip the chunk
_XBRL_TOKEN_RE = re.compile(
    r"(?:us-gaap|dei|aapl|ifrs-full|srt):[A-Za-z]+|"   # namespace:Name
    r"\b\d{10}\b|"                                        # 10-digit CIK numbers
    r"\b\d{4}-\d{2}-\d{2}\b",                            # ISO dates (XBRL context dates)
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Token-count helpers (word-based approximation)
# ─────────────────────────────────────────────────────────────────────────────

def approx_tokens(text: str) -> int:
    """Approximate token count: word count ÷ 0.75."""
    words = len(text.split())
    return max(1, round(words / WORDS_PER_TOKEN))


def tokens_to_chars(n_tokens: int, sample_text: str) -> int:
    """
    Estimate how many characters correspond to n_tokens in sample_text.
    Used to set a character-based cut point.
    """
    total_chars  = len(sample_text)
    total_tokens = approx_tokens(sample_text)
    if total_tokens == 0:
        return total_chars
    chars_per_token = total_chars / total_tokens
    return round(n_tokens * chars_per_token)


# ─────────────────────────────────────────────────────────────────────────────
# Sentence-boundary finder
# ─────────────────────────────────────────────────────────────────────────────

# Sentence terminators: period/bang/question mark followed by whitespace or EOS,
# NOT preceded by a common abbreviation initial (e.g. "Dr.", "U.S.", digits).
_SENT_END_RE = re.compile(
    r"(?<![A-Z])(?<!\d)(?<!\s[A-Za-z])[.!?](?=\s|$)",
    re.MULTILINE,
)


def find_sentence_boundary(text: str, target_char: int) -> int:
    """
    Find the index of the last sentence boundary at or before target_char.
    Returns target_char itself if no boundary is found (hard cut fallback).
    """
    # Search backwards from target_char
    search_in = text[:target_char]
    best = -1
    for m in _SENT_END_RE.finditer(search_in):
        best = m.end()   # position just after the terminator
    if best > 0:
        return best
    return target_char   # fallback: hard cut


# ─────────────────────────────────────────────────────────────────────────────
# Text chunker
# ─────────────────────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks of ≈TARGET_TOKENS tokens.
    Returns a list of chunk strings (may be length 1 for short sections).
    """
    text = text.strip()
    if not text:
        return []

    total_tokens = approx_tokens(text)
    if total_tokens <= TARGET_TOKENS:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        # Estimate character position for the target end of this chunk
        remaining = text[start:]
        target_chars = tokens_to_chars(TARGET_TOKENS, remaining)
        end = start + target_chars

        if end >= len(text):
            # Last chunk — take everything remaining
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Snap to the nearest sentence boundary
        local_boundary = find_sentence_boundary(remaining, target_chars)
        end = start + local_boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Overlap: step back by OVERLAP_TOKENS worth of characters
        overlap_chars = tokens_to_chars(OVERLAP_TOKENS, remaining)
        start = max(start + 1, end - overlap_chars)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Table serialiser
# ─────────────────────────────────────────────────────────────────────────────

def _dedup_row(row: list[str]) -> list[str]:
    """
    Remove consecutive repeated cells that arise from colspan expansion.
    E.g. ['Products', 'Products', 'Products', '$', '307,003', '']
         →  ['Products', '$', '307,003']
    Empty-string padding cells are also collapsed.
    """
    if not row:
        return row
    result = [row[0]]
    for cell in row[1:]:
        if cell != result[-1]:
            result.append(cell)
    # Strip trailing empty strings
    while result and result[-1] == "":
        result.pop()
    return result


def table_to_markdown(table: list[list[str]]) -> str:
    """
    Serialise a 2-D table (list of rows, each a list of cell strings)
    into a Markdown-style pipe-delimited grid.

    Example output:
        | Net sales | 2025 | 2024 | 2023 |
        |---|---|---|---|
        | Products | $307,003 | $294,866 | $298,085 |
        | Services | $109,158 | $96,169 | $85,200 |
    """
    if not table:
        return ""

    # De-duplicate colspan-repeated cells in each row
    clean_rows = [_dedup_row(row) for row in table]

    # Pad all rows to the same width
    max_cols = max((len(r) for r in clean_rows), default=0)
    padded = [r + [""] * (max_cols - len(r)) for r in clean_rows]

    if not padded:
        return ""

    lines: list[str] = []
    for i, row in enumerate(padded):
        line = "| " + " | ".join(str(cell) for cell in row) + " |"
        lines.append(line)
        if i == 0:
            # Insert header separator after the first row
            sep = "| " + " | ".join("---" for _ in row) + " |"
            lines.append(sep)

    return "\n".join(lines)


def is_empty_table(table: list[list[str]]) -> bool:
    """True if the table contains only whitespace/empty cells."""
    return all(cell.strip() == "" for row in table for cell in row)


# ─────────────────────────────────────────────────────────────────────────────
# Main chunking logic for one document
# ─────────────────────────────────────────────────────────────────────────────

def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert one extracted document dict into a flat list of chunk dicts.
    """
    company = doc["company"]
    year    = doc["year"]
    chunks: list[dict[str, Any]] = []
    global_idx = 0   # sequential across ALL sections in this document

    for section in doc.get("sections", []):
        section_name = section["section_name"]
        position_id  = section["position_id"]
        section_text = section.get("text", "").strip()
        section_tables = section.get("tables", [])

        chunk_index_within_section = 0  # reset per section

        # ── TEXT chunks ───────────────────────────────────────────────────
        if len(section_text) >= MIN_TEXT_CHARS:
            text_splits = chunk_text(section_text)
            for split_text in text_splits:
                # Skip XBRL-noise chunks (leaked iXBRL context data)
                words = split_text.split()
                n_xbrl = len(_XBRL_TOKEN_RE.findall(split_text))
                if words and (n_xbrl / len(words)) > XBRL_NOISE_THRESHOLD:
                    log.debug("  Skipping XBRL noise chunk (%d/%d tokens)", n_xbrl, len(words))
                    continue

                tok = approx_tokens(split_text)
                chunk_id = f"{company}_{year}_{global_idx:04d}"
                chunks.append({
                    "chunk_id":    chunk_id,
                    "company":     company,
                    "year":        year,
                    "section":     section_name,
                    "position_id": position_id,
                    "chunk_type":  "text",
                    "chunk_index": chunk_index_within_section,
                    "text":        split_text,
                    "token_count": tok,
                })
                global_idx += 1
                chunk_index_within_section += 1

        # ── TABLE chunks ──────────────────────────────────────────────────
        for tbl in section_tables:
            if is_empty_table(tbl):
                continue
            md = table_to_markdown(tbl)
            if not md.strip():
                continue

            # Prepend section context so the table is self-contained
            table_text = (
                f"[Table from: {section_name}]\n\n"
                f"{md}"
            )
            tok = approx_tokens(table_text)
            chunk_id = f"{company}_{year}_{global_idx:04d}"
            chunks.append({
                "chunk_id":    chunk_id,
                "company":     company,
                "year":        year,
                "section":     section_name,
                "position_id": position_id,
                "chunk_type":  "table",
                "chunk_index": chunk_index_within_section,
                "text":        table_text,
                "token_count": tok,
            })
            global_idx += 1
            chunk_index_within_section += 1

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────

def chunk_all() -> None:
    """
    Read all JSON files in data/processed/ (except chunks.jsonl itself),
    chunk each document, and write all chunks to data/processed/chunks.jsonl.
    """
    json_files = sorted(
        p for p in PROCESSED_DIR.glob("*.json")
        if p.name != "chunks.json"  # safety: skip if someone names it .json
    )

    if not json_files:
        log.warning("No JSON files found in %s — run extract.py first.", PROCESSED_DIR)
        return

    all_chunks: list[dict[str, Any]] = []

    for json_path in json_files:
        log.info("=" * 60)
        log.info("Chunking: %s", json_path.name)
        try:
            with open(json_path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception as exc:
            log.error("  FAILED to read %s: %s", json_path.name, exc)
            continue

        try:
            chunks = chunk_document(doc)
        except Exception as exc:
            log.error("  FAILED to chunk %s: %s", json_path.name, exc, exc_info=True)
            continue

        # Stats
        text_chunks  = [c for c in chunks if c["chunk_type"] == "text"]
        table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
        avg_tok = (
            round(sum(c["token_count"] for c in text_chunks) / len(text_chunks))
            if text_chunks else 0
        )
        log.info(
            "  Total chunks: %d  (text=%d, table=%d)  avg_text_tokens=%d",
            len(chunks), len(text_chunks), len(table_chunks), avg_tok,
        )

        all_chunks.extend(chunks)

    # Write JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    log.info("=" * 60)
    log.info(
        "Done. %d total chunks written to %s",
        len(all_chunks), OUTPUT_FILE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chunk_all()
