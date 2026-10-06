"""
ingestion/extract_images.py — Extract and resolve embedded images from SEC 10-K filings
========================================================================================

Responsibilities:
  1. Scan every 10-K HTML file in data/raw/*.htm for <img> tags.
  2. Resolve relative image paths:
     - First, check if image files were downloaded locally alongside the HTML.
     - Second, check if SEC EDGAR submission URL is available in the HTML preamble,
       and fetch the original image file using a compliant SEC User-Agent.
     - Gracefully skip and log any unreachable images (e.g. browser extensions, missing local assets).
  3. Save resolved images to data/processed/images/{company}_{year}_{index}.jpg
  4. Record surrounding context:
     - Section name & position where the image appeared
     - Nearest heading / caption text
  5. Output a structured manifest: data/processed/images/extracted_images.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
MANIFEST_FILE = IMAGES_DIR / "extracted_images.json"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_images")

# ─────────────────────────────────────────────────────────────────────────────
# Constants & Patterns
# ─────────────────────────────────────────────────────────────────────────────
ITEM_HEADING_RE = re.compile(
    r"^Item\s+(\d{1,2}[A-Za-z]?)\s*\.?\s*(.*)",
    re.IGNORECASE,
)

SEC_USER_AGENT = os.getenv(
    "SEC_EDGAR_USER_AGENT",
    "FinRiskAnalystResearch research-team@fin-risk-analyst.org"
)

KNOWN_EDGAR_ACCESSIONS: dict[int, str] = {
    2021: "https://www.sec.gov/Archives/edgar/data/320193/000032019321000105/",
    2022: "https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/",
    2023: "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/",
    2024: "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/",
    2025: "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/",
}


def extract_edgar_base_url(html_content: str, year: int | None = None) -> str | None:
    """Extract original SEC EDGAR base URL from comment header, or use known accession."""
    match = re.search(r"saved from url=\(\d+\)(https?://[^\s>]+)", html_content[:2000])
    if match:
        full_url = match.group(1)
        # Base folder is everything up to the last slash
        return full_url.rsplit("/", 1)[0] + "/"
    if year and year in KNOWN_EDGAR_ACCESSIONS:
        return KNOWN_EDGAR_ACCESSIONS[year]
    return None


def get_surrounding_context(img_tag: Any) -> tuple[str, str]:
    """
    Traverse backwards to find the nearest section heading and caption/paragraph.
    """
    # 1. Section heading
    section_name = "Preamble"
    curr = img_tag
    while curr:
        prev = curr.find_previous(["h1", "h2", "h3", "h4", "p", "td", "span", "div"])
        if prev:
            txt = " ".join(prev.get_text(" ", strip=True).split())
            if ITEM_HEADING_RE.match(txt):
                section_name = txt
                break
            curr = prev
        else:
            break

    # 2. Nearest caption / enclosing paragraph text
    caption = ""
    parent = img_tag.find_parent(["p", "div", "td"])
    if parent:
        caption = " ".join(parent.get_text(" ", strip=True).split())
        if len(caption) > 300:
            caption = caption[:300] + "..."

    return section_name, caption


def extract_images_from_filings() -> list[dict[str, Any]]:
    """
    Find, resolve, and save all 10-K images across filings.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    html_files = sorted(RAW_DIR.glob("*.htm")) + sorted(RAW_DIR.glob("*.html"))

    if not html_files:
        log.warning("No HTML filings found in %s", RAW_DIR)
        return []

    extracted_records: list[dict[str, Any]] = []
    total_imgs_found = 0
    total_imgs_saved = 0
    total_imgs_skipped = 0

    session = requests.Session()
    session.headers.update({"User-Agent": SEC_USER_AGENT})

    for html_path in html_files:
        stem = html_path.stem
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        ticker = parts[0].upper()
        try:
            year = int(parts[-1])
        except ValueError:
            continue

        log.info("=" * 60)
        log.info("Scanning for images in: %s (%s FY%d)", html_path.name, ticker, year)

        with open(html_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        edgar_base = extract_edgar_base_url(content, year=year)
        soup = BeautifulSoup(content, "lxml")
        img_tags = soup.find_all("img")
        log.info("  Found %d <img> tags in document", len(img_tags))
        total_imgs_found += len(img_tags)

        file_idx = 0
        for idx, img in enumerate(img_tags):
            src = img.get("src", "").strip()
            alt = img.get("alt", "").strip()

            if not src:
                log.warning("  [%d] Skipped: empty src attribute", idx)
                total_imgs_skipped += 1
                continue

            # Skip browser extension URLs
            if src.startswith("chrome-extension://") or src.startswith("moz-extension://"):
                log.info("  [%d] Skipped browser extension artifact: %s", idx, src[:60])
                total_imgs_skipped += 1
                continue

            section_name, caption = get_surrounding_context(img)

            # Resolve image data (Local file vs EDGAR URL)
            img_bytes: bytes | None = None
            resolved_source: str = ""

            # Check local file
            local_candidate = (html_path.parent / src).resolve()
            if local_candidate.exists() and local_candidate.is_file():
                try:
                    img_bytes = local_candidate.read_bytes()
                    resolved_source = f"local:{local_candidate}"
                except Exception as e:
                    log.warning("  Could not read local file %s: %s", local_candidate, e)

            # If not local, try EDGAR base URL
            if img_bytes is None and edgar_base:
                # Clean relative path (e.g. ./aapl-20240928_files/aapl-20240928_g2.jpg -> aapl-20240928_g2.jpg)
                filename = os.path.basename(src)
                remote_url = edgar_base + filename
                try:
                    resp = session.get(remote_url, timeout=10)
                    if resp.status_code == 200 and resp.content:
                        img_bytes = resp.content
                        resolved_source = f"edgar:{remote_url}"
                        log.info("  Fetched image from SEC EDGAR: %s (%d bytes)", filename, len(img_bytes))
                    else:
                        log.info("  Remote EDGAR fetch returned status %d for %s", resp.status_code, remote_url)
                except Exception as e:
                    log.info("  Remote EDGAR fetch failed for %s: %s", remote_url, e)

            if img_bytes is None:
                log.warning("  [%d] Unreachable image (not found locally or on EDGAR): src='%s' alt='%s' — SKIPPING GRACEFULLY", idx, src, alt)
                total_imgs_skipped += 1
                continue

            # Save resolved image
            file_idx += 1
            out_filename = f"{ticker}_{year}_{file_idx:02d}.jpg"
            out_path = IMAGES_DIR / out_filename
            with open(out_path, "wb") as img_out:
                img_out.write(img_bytes)

            record = {
                "company": ticker,
                "year": year,
                "image_index": file_idx,
                "filename": out_filename,
                "saved_path": str(out_path.resolve()),
                "relative_path": f"data/processed/images/{out_filename}",
                "original_src": src,
                "alt_text": alt,
                "section": section_name,
                "caption": caption,
                "source": resolved_source,
                "byte_size": len(img_bytes),
            }
            extracted_records.append(record)
            total_imgs_saved += 1
            log.info("  ✓ Saved: %s -> %s (%d bytes, Section: '%s')", src, out_filename, len(img_bytes), section_name[:50])

    # Save manifest
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(extracted_records, f, indent=2)

    log.info("=" * 60)
    log.info("Extraction summary:")
    log.info("  Total <img> tags scanned : %d", total_imgs_found)
    log.info("  Images saved & resolved  : %d", total_imgs_saved)
    log.info("  Images skipped           : %d", total_imgs_skipped)
    log.info("  Manifest saved to        : %s", MANIFEST_FILE)
    log.info("=" * 60)

    return extracted_records


if __name__ == "__main__":
    extract_images_from_filings()
