"""
qa_chunks.py — Phase 3 audit script
Runs all 4 QA checks in one pass:
  1. 5 random text chunks — coherence check
  2. 3 table chunks — integrity check
  3. Chunk count breakdown (by year, section)
  4. One Item 1A + one Item 8 chunk
"""
import json
import random
import textwrap

random.seed(42)

CHUNKS_FILE = "data/processed/chunks.jsonl"

with open(CHUNKS_FILE, encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]

text_chunks  = [c for c in chunks if c["chunk_type"] == "text"]
table_chunks = [c for c in chunks if c["chunk_type"] == "table"]

DIVIDER  = "=" * 70
SEP      = "-" * 70

def wrap(text, width=78, indent="    "):
    return "\n".join(
        indent + line
        for para in text.split("\n")
        for line in textwrap.wrap(para, width) or [""]
    )

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — 5 random text chunks
# ─────────────────────────────────────────────────────────────────────────────
print(DIVIDER)
print("CHECK 1 — 5 RANDOM TEXT CHUNKS (coherence)")
print(DIVIDER)
sample = random.sample(text_chunks, min(5, len(text_chunks)))
for i, c in enumerate(sample, 1):
    preview = c["text"][:500]
    ends_mid = not preview.rstrip().endswith((".", "?", "!", '"', "'"))
    print(f"\n[TEXT {i}]  chunk_id={c['chunk_id']}")
    print(f"  section   : {c['section']}")
    print(f"  year      : {c['year']}")
    print(f"  tokens    : {c['token_count']}")
    print(f"  ends-OK?  : {'YES' if not ends_mid else 'CHECK — no terminal punct at 500-char mark'}")
    print(f"  text (first 500 chars):")
    print(wrap(preview))

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — 3 table chunks
# ─────────────────────────────────────────────────────────────────────────────
print()
print(DIVIDER)
print("CHECK 2 — TABLE CHUNK INTEGRITY (first 3 table chunks)")
print(DIVIDER)
for i, c in enumerate(table_chunks[:3], 1):
    lines = c["text"].split("\n")
    n_rows = sum(1 for l in lines if l.startswith("|") and "---" not in l)
    n_cols = lines[0].count("|") - 1 if lines else 0
    print(f"\n[TABLE {i}]  chunk_id={c['chunk_id']}")
    print(f"  section   : {c['section']}")
    print(f"  rows      : {n_rows}  cols: {n_cols}")
    print(f"  tokens    : {c['token_count']}")
    print(f"  first 10 lines:")
    for line in lines[:10]:
        print(f"    {line}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — Chunk count breakdown
# ─────────────────────────────────────────────────────────────────────────────
print()
print(DIVIDER)
print("CHECK 3 — CHUNK COUNT BREAKDOWN")
print(DIVIDER)

# By year
from collections import Counter, defaultdict
by_year = Counter(c["year"] for c in chunks)
print("\nBy year:")
for yr, cnt in sorted(by_year.items()):
    bar = "#" * (cnt // 2)
    print(f"  {yr}: {cnt:4d} chunks  {bar}")

# By section (top 10)
by_section = Counter(c["section"] for c in chunks)
print("\nBy section (top 12):")
for sec, cnt in by_section.most_common(12):
    print(f"  {cnt:4d}  {sec[:70]}")

# Text vs table
print(f"\nTotal : {len(chunks)}  |  text={len(text_chunks)}  table={len(table_chunks)}")
tok_counts = [c["token_count"] for c in text_chunks]
if tok_counts:
    print(f"Text token stats: min={min(tok_counts)}  max={max(tok_counts)}  "
          f"avg={round(sum(tok_counts)/len(tok_counts))}")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — One Item 1A chunk + one Item 8 chunk
# ─────────────────────────────────────────────────────────────────────────────
print()
print(DIVIDER)
print("CHECK 4 — SECTION METADATA SPOT-CHECK")
print(DIVIDER)

item1a = next(
    (c for c in text_chunks if "1A" in c["section"] or "risk factor" in c["section"].lower()),
    None
)
item8 = next(
    (c for c in table_chunks if "Item 8" in c["section"] or "financial statement" in c["section"].lower()),
    None
)

for label, c in [("Item 1A (Risk Factors) — text chunk", item1a),
                  ("Item 8  (Financial Stmts) — table chunk", item8)]:
    print(f"\n[{label}]")
    if c is None:
        print("  NOT FOUND")
        continue
    print(f"  chunk_id   : {c['chunk_id']}")
    print(f"  section    : {c['section']}")
    print(f"  position_id: {c['position_id']}")
    print(f"  chunk_type : {c['chunk_type']}")
    print(f"  tokens     : {c['token_count']}")
    print(f"  text (first 700 chars):")
    print(wrap(c["text"][:700]))

print()
print(DIVIDER)
print("AUDIT COMPLETE")
print(DIVIDER)
