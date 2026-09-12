"""summary_report.py — 5-year ingestion quality report"""
import json
from collections import defaultdict
from pathlib import Path

PROCESSED = Path("data/processed")
CHUNKS    = PROCESSED / "chunks.jsonl"

# ── Load chunks ───────────────────────────────────────────────────────────────
with open(CHUNKS, encoding="utf-8") as f:
    chunks = [json.loads(l) for l in f if l.strip()]

# ── Load per-year extraction stats from JSONs ─────────────────────────────────
year_stats = {}
for jp in sorted(PROCESSED.glob("AAPL_*.json")):
    year = int(jp.stem.split("_")[1])
    with open(jp, encoding="utf-8") as f:
        doc = json.load(f)
    sections = doc["sections"]
    year_stats[year] = {
        "sections": len(sections),
        "tables":   sum(len(s["tables"]) for s in sections),
        "section_names": [s["section_name"] for s in sections],
    }

# ── Per-year chunk stats ───────────────────────────────────────────────────────
year_chunks = defaultdict(lambda: {"text": 0, "table": 0})
for c in chunks:
    year_chunks[c["year"]][c["chunk_type"]] += 1

years = sorted(year_stats.keys())
total_chunks = {yr: year_chunks[yr]["text"] + year_chunks[yr]["table"] for yr in years}
avg_chunks = sum(total_chunks.values()) / len(total_chunks)
threshold  = avg_chunks * 0.70   # flag if <70% of average

# ── Print summary table ────────────────────────────────────────────────────────
print()
print("=" * 78)
print("  AAPL 10-K INGESTION SUMMARY — 5 YEARS")
print("=" * 78)
header = f"{'Year':<6} {'Sections':>9} {'Tables':>7} {'TextChk':>8} {'TblChk':>7} {'Total':>6} {'Flag':>5}"
print(header)
print("-" * 78)

for yr in years:
    s  = year_stats[yr]
    tc = year_chunks[yr]["text"]
    tb = year_chunks[yr]["table"]
    tot = tc + tb
    flag = "⚠ LOW" if tot < threshold else "  OK"
    print(f"  {yr}  {s['sections']:>9}  {s['tables']:>7}  {tc:>8}  {tb:>7}  {tot:>6}  {flag}")

print("-" * 78)
grand = sum(total_chunks.values())
t_text  = sum(year_chunks[yr]["text"] for yr in years)
t_table = sum(year_chunks[yr]["table"] for yr in years)
print(f"  {'TOTAL':<5}  {'':>9}  {'':>7}  {t_text:>8}  {t_table:>7}  {grand:>6}")
print(f"\n  Average chunks/year: {avg_chunks:.0f}   Flag threshold (<30% below avg): {threshold:.0f}")
print("=" * 78)

# ── Outlier section detail ────────────────────────────────────────────────────
flagged = [yr for yr in years if total_chunks[yr] < threshold]
if flagged:
    # Also gather the reference set: all sections from NON-flagged years
    ref_sections = set()
    for yr in years:
        if yr not in flagged:
            for name in year_stats[yr]["section_names"]:
                ref_sections.add(name.split(".")[0].strip())   # e.g. "Item 1A"

    print()
    for yr in flagged:
        print(f"  ⚠  YEAR {yr} IS FLAGGED ({total_chunks[yr]} chunks, "
              f"{((total_chunks[yr] - avg_chunks) / avg_chunks * 100):.1f}% vs avg)")
        print(f"     Sections found in {yr}:")
        for name in year_stats[yr]["section_names"]:
            print(f"       • {name}")
        # Check which Item headings are missing vs reference years
        this_items = {n.split(".")[0].strip() for n in year_stats[yr]["section_names"]}
        missing = ref_sections - this_items
        if missing:
            print(f"     ⚠  MISSING vs other years: {', '.join(sorted(missing))}")
        else:
            print(f"     ✓  All standard Item sections present (volume difference may be content-based)")
    print("=" * 78)
else:
    print()
    print("  ✓  No outliers detected — all years within 30% of the average.")
    print("=" * 78)

# ── Section coverage matrix (which Item sections exist each year) ─────────────
print()
print("  SECTION COVERAGE MATRIX (Item sections per year)")
print("-" * 78)
all_items = {}
for yr in years:
    for name in year_stats[yr]["section_names"]:
        key = name.split(".")[0].strip()
        if key.startswith("Item"):
            all_items[key] = all_items.get(key, name)   # canonical name

item_keys = sorted(all_items.keys(), key=lambda x: (
    int(x.replace("Item ", "").rstrip("ABCDEFGHIJ") or 0),
    x
))

yr_header = "  " + " ".join(f"{yr}" for yr in years)
print(yr_header)
for key in item_keys:
    row = []
    for yr in years:
        present = any(
            n.split(".")[0].strip() == key
            for n in year_stats[yr]["section_names"]
        )
        row.append(" ✓  " if present else " —  ")
    canonical = all_items[key][:45]
    print(f"  {''.join(row)}  {canonical}")
print("=" * 78)
