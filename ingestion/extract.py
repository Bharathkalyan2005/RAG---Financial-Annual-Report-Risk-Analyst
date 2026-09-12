"""
ingestion/extract.py — Phase 1: HTML extraction for SEC 10-K filings
=====================================================================
Parses SEC iXBRL / HTML 10-K filings from data/raw/*.htm into structured JSON.

Output per file  →  data/processed/{TICKER}_{YEAR}.json
Schema:
{
  "company":  "AAPL",
  "year":     2024,
  "sections": [
    {
      "section_name": "Item 1A. Risk Factors",
      "position_id":  3,          # sequential ordinal (1-based), used as citation ref
      "text":         "...",      # body text, tables removed
      "tables": [                 # list of tables found in this section
        [                         # one table
          ["col1", "col2", ...],  # header row (or first data row)
          ["val1", "val2", ...],  # data row — colspan/rowspan cells are repeated
          ...
        ]
      ]
    }
  ]
}

Design notes
------------
• SEC 10-K iXBRL files use inline XBRL markup.  Headings are NOT <h2> tags;
  they appear as styled <td> or <span> cells containing text like "Item 1A."
  We detect section boundaries by scanning ALL element text nodes for the
  pattern "Item N" or "Item NA" at the start of the text.

• Two passes over the DOM:
    Pass 1 — collect (element, text) tuples that match the Item heading pattern.
             Deduplicate by normalized heading key so the Table of Contents
             entries don't shadow the real body sections.
    Pass 2 — walk the document top-to-bottom; when a heading element is
             encountered, open a new section bucket.  Everything else
             (non-heading text nodes and tables) is appended to the current
             bucket.

• Tables: colspan/rowspan are handled by expanding into a 2-D grid of strings.
  Empty/whitespace cells are preserved as "".  Tables with <2 rows or <2 cols
  are treated as layout/spacer tables and discarded.

• Synthetic position_id: sections are numbered 1, 2, 3, … in document order.
  This replaces PDF page numbers for citation purposes.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning, Tag

# ── silence the iXBRL "XML parsed as HTML" warning — expected for SEC filings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract")

# ─────────────────────────────────────────────────────────────────────────────
# Constants / patterns
# ─────────────────────────────────────────────────────────────────────────────

# Matches section headings like:
#   "Item 1."  "Item 1A."  "Item 1A. Risk Factors"
#   "Item 12 . Security Ownership …"
# Anchored to start-of-string; heading text may continue after.
ITEM_HEADING_RE = re.compile(
    r"^Item\s+(\d{1,2}[A-Za-z]?)\s*\.?\s*(.*)",
    re.IGNORECASE,
)

# Key used to deduplicate TOC entries vs. body headings.
# Normalises "Item 1A." and "Item 1A. Risk Factors" to the same key "item_1a".
ITEM_KEY_RE = re.compile(r"^Item\s+(\d{1,2}[A-Za-z]?)", re.IGNORECASE)

# Tags we scan for heading candidates (broad — SEC HTML is irregular)
HEADING_SCAN_TAGS = {"td", "th", "span", "p", "div", "h1", "h2", "h3", "h4", "li"}

# Minimum table size to keep (skip layout/spacer tables)
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2

# Maximum characters for a text node to still be a heading (not body text)
MAX_HEADING_TEXT = 160


# ─────────────────────────────────────────────────────────────────────────────
# Table extraction
# ─────────────────────────────────────────────────────────────────────────────

def _cell_text(cell: Tag) -> str:
    """Return stripped plain text from a <td> or <th> cell."""
    return " ".join(cell.get_text(" ", strip=True).split())


def extract_table(table_tag: Tag) -> list[list[str]] | None:
    """
    Convert a <table> BS4 element to a 2-D list of strings.

    Handles:
    • colspan — cell value is repeated across spanned columns
    • rowspan — cell value is repeated down spanned rows
    • Nested tables — ignored (only the outermost rows are processed)

    Returns None if the table is too small to be meaningful (layout/spacer).
    """
    # Collect all <tr> rows that are direct children (or one level deep)
    all_rows = table_tag.find_all("tr", recursive=True)

    # Build a sparse grid to handle rowspan/colspan
    grid: dict[tuple[int, int], str] = {}
    row_idx = 0

    for tr in all_rows:
        cells = tr.find_all(["td", "th"], recursive=False)
        if not cells:
            # Try one level deeper (some SEC tables wrap <td> in extra <tr>)
            cells = tr.find_all(["td", "th"])
        col_idx = 0
        for cell in cells:
            # Skip to the next free column (rowspan from above may block it)
            while (row_idx, col_idx) in grid:
                col_idx += 1

            text = _cell_text(cell)
            try:
                colspan = int(cell.get("colspan", 1) or 1)
            except (ValueError, TypeError):
                colspan = 1
            try:
                rowspan = int(cell.get("rowspan", 1) or 1)
            except (ValueError, TypeError):
                rowspan = 1

            # Fill the grid for all spanned positions
            for r_off in range(rowspan):
                for c_off in range(colspan):
                    grid[(row_idx + r_off, col_idx + c_off)] = text

            col_idx += colspan

        row_idx += 1

    if row_idx == 0:
        return None

    # Determine grid dimensions
    max_col = max(c for (_, c) in grid.keys()) + 1 if grid else 0
    max_row = max(r for (r, _) in grid.keys()) + 1 if grid else 0

    # Materialise into a list of rows
    result = []
    for r in range(max_row):
        row = [grid.get((r, c), "") for c in range(max_col)]
        result.append(row)

    # Drop trivially small tables (likely layout/spacer)
    if len(result) < MIN_TABLE_ROWS or max_col < MIN_TABLE_COLS:
        return None

    # Drop tables that are entirely empty strings (invisible spacers)
    non_empty = sum(1 for row in result for cell in row if cell.strip())
    if non_empty == 0:
        return None

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Section detection
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_key(text: str) -> str:
    """
    Produce a canonical key for deduplicating TOC vs. body headings.
    "Item 1A. Risk Factors 5"  →  "item_1a"
    "Item 1A."                 →  "item_1a"
    """
    m = ITEM_KEY_RE.match(text)
    if m:
        return "item_" + m.group(1).lower()
    return text.lower().strip()


def _is_heading_element(element: Tag) -> tuple[bool, str]:
    """
    Return (True, full_heading_text) if the element looks like an Item heading.
    Criteria:
      • element's direct text starts with "Item N" pattern
      • text length is reasonable for a heading (< MAX_HEADING_TEXT chars)
      • element is not itself inside a <table> that belongs to a financial
        table (we allow it to be inside a layout table)
    """
    if element.name not in HEADING_SCAN_TAGS:
        return False, ""

    text = element.get_text(" ", strip=True)
    text = " ".join(text.split())   # collapse whitespace

    if not text:
        return False, ""
    if len(text) > MAX_HEADING_TEXT:
        return False, ""

    if ITEM_HEADING_RE.match(text):
        return True, text

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# DOM walker — section-aware, table-aware
# ─────────────────────────────────────────────────────────────────────────────

def parse_document(html_path: Path) -> dict[str, Any]:
    """
    Parse one 10-K HTM file and return the structured document dict.
    """
    log.info("  Parsing: %s", html_path.name)

    with open(html_path, encoding="utf-8", errors="replace") as fh:
        soup = BeautifulSoup(fh, "lxml")

    # ── Pass 0: find all <table> elements with their DOM position ────────────
    # We'll pull them out by object identity later.
    all_tables_in_order: list[Tag] = soup.find_all("table")
    table_set: set[int] = {id(t) for t in all_tables_in_order}
    # Track which tables we've already assigned to a section
    consumed_tables: set[int] = set()

    # ── Pass 1: collect heading candidates, deduplicated by key ──────────────
    # We walk every element and record those that match Item N.
    # The SAME heading may appear multiple times (TOC + body); we keep
    # the LAST occurrence because in SEC filings the body comes after the TOC.
    heading_map: dict[str, Tag] = {}   # key → element (last seen wins)

    for element in soup.find_all(HEADING_SCAN_TAGS):
        is_hdg, text = _is_heading_element(element)
        if is_hdg:
            key = _normalise_key(text)
            heading_map[key] = element   # last occurrence wins

    heading_ids: set[int] = {id(el) for el in heading_map.values()}

    # ── Pass 2: walk DOM top-to-bottom, assign content to sections ───────────
    # We traverse all leaf/block elements in document order.
    # When we hit a heading element, we open a new section.
    # When we hit a <table>, we extract it and attach to the current section.
    # Everything else contributes text to the current section.

    sections: list[dict[str, Any]] = []
    current_section: dict[str, Any] | None = None
    position_counter = 0

    def flush_section() -> None:
        """Finalise the current section's text."""
        nonlocal current_section
        if current_section is not None:
            current_section["text"] = " ".join(current_section["_text_parts"]).strip()
            del current_section["_text_parts"]
            sections.append(current_section)
            current_section = None

    def open_section(name: str) -> None:
        nonlocal current_section, position_counter
        flush_section()
        position_counter += 1
        current_section = {
            "section_name": name,
            "position_id": position_counter,
            "_text_parts": [],
            "tables": [],
        }

    # We walk children of <body> (or <html> if no <body>) recursively,
    # but we need to avoid descending *into* elements we've already processed.
    # Strategy: iterate in document order over ALL elements; use a visited set
    # to skip children of already-processed tables.

    body = soup.find("body") or soup

    # Track ancestors of tables we already extracted so we don't re-process
    # their internal <td>/<tr> as text nodes.
    processed_table_ancestors: set[int] = set()

    def walk(node: Tag) -> None:
        """Recursive DOM walker."""
        nonlocal current_section

        for child in node.children:
            if not isinstance(child, Tag):
                # NavigableString — add to current section text
                text = str(child).strip()
                if text and current_section is not None:
                    current_section["_text_parts"].append(text)
                continue

            # ── Is this element a heading? ────────────────────────────────
            if id(child) in heading_ids:
                name = child.get_text(" ", strip=True)
                name = " ".join(name.split())
                # Strip trailing page numbers (e.g. "Item 1A. Risk Factors 5")
                name = re.sub(r'\s+\d+\s*$', '', name).strip()
                open_section(name)
                continue   # don't recurse into the heading element itself

            # ── Is this a <table>? ────────────────────────────────────────
            if child.name == "table" and id(child) not in consumed_tables:
                consumed_tables.add(id(child))
                table_data = extract_table(child)
                if table_data is not None:
                    if current_section is None:
                        # Table before any heading — put in a "preamble" section
                        open_section("Preamble")
                    current_section["tables"].append(table_data)
                # Do NOT recurse into <table> — we've already extracted it
                continue

            # ── Skip children of tables we already consumed ───────────────
            # (their <td>/<tr> nodes would otherwise emit duplicate text)
            if id(child) in consumed_tables:
                continue

            # Check if child is a descendant of an already-consumed table
            # (expensive — use sparingly; only if child.name in row/cell tags)
            if child.name in {"tr", "td", "th", "tbody", "thead", "tfoot"}:
                parent_table = child.find_parent("table")
                if parent_table is not None and id(parent_table) in consumed_tables:
                    continue

            # ── Generic block: gather text and recurse ────────────────────
            walk(child)

    # Before walking, open a "Preamble" section so text before Item 1 isn't lost
    open_section("Preamble")
    walk(body)
    flush_section()

    # ── Remove the preamble if it's empty (no text, no tables) ──────────────
    sections = [
        s for s in sections
        if s["text"].strip() or s["tables"]
    ]

    # ── Re-number position_ids sequentially after filtering ─────────────────
    for idx, sec in enumerate(sections, start=1):
        sec["position_id"] = idx

    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────

def extract_all() -> None:
    """
    Loop over every .htm file in data/raw/, extract, and write JSON to
    data/processed/.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    htm_files = sorted(RAW_DIR.glob("*.htm")) + sorted(RAW_DIR.glob("*.html"))

    if not htm_files:
        log.warning("No .htm / .html files found in %s", RAW_DIR)
        log.warning(
            "Place your 10-K files there named as TICKER_YEAR.htm "
            "(e.g. AAPL_2024.htm) and re-run."
        )
        return

    total_ok = 0
    total_err = 0

    for htm_path in htm_files:
        # ── Parse ticker and year from filename ──────────────────────────────
        stem = htm_path.stem   # e.g. "AAPL_2024"
        parts = stem.split("_")
        if len(parts) < 2:
            log.warning(
                "Skipping %s — filename must be TICKER_YEAR.htm", htm_path.name
            )
            continue

        # Support filenames like AAPL_2024 or AAPL_INC_2024 (ticker may have _)
        try:
            year = int(parts[-1])
        except ValueError:
            log.warning(
                "Skipping %s — last segment before .htm must be a 4-digit year",
                htm_path.name,
            )
            continue
        ticker = "_".join(parts[:-1]).upper()

        log.info("=" * 60)
        log.info("Processing: %s  (company=%s, year=%d)", htm_path.name, ticker, year)

        try:
            sections = parse_document(htm_path)
        except Exception as exc:   # noqa: BLE001  (catch-all per spec)
            log.error("  FAILED to parse %s: %s", htm_path.name, exc, exc_info=True)
            total_err += 1
            continue

        # ── Build output document ────────────────────────────────────────────
        doc = {
            "company": ticker,
            "year": year,
            "sections": sections,
        }

        # ── Stats ────────────────────────────────────────────────────────────
        n_sections = len(sections)
        n_tables = sum(len(s["tables"]) for s in sections)
        item_sections = [s["section_name"] for s in sections
                         if ITEM_HEADING_RE.match(s["section_name"])]

        log.info("  Sections found : %d", n_sections)
        log.info("  Item sections  : %s", ", ".join(item_sections) if item_sections else "(none)")
        log.info("  Tables found   : %d", n_tables)

        # ── Write JSON ───────────────────────────────────────────────────────
        out_path = PROCESSED_DIR / f"{ticker}_{year}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)

        log.info("  Written to     : %s", out_path)
        total_ok += 1

    log.info("=" * 60)
    log.info(
        "Done. %d file(s) processed successfully, %d error(s).",
        total_ok,
        total_err,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    extract_all()
