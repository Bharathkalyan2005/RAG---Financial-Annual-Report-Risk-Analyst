"""
ingestion/describe_images.py — Vision-Language Analysis of 10-K Embedded Images
================================================================================

Responsibilities:
  1. Read extracted images from data/processed/images/extracted_images.json.
  2. For each image, call a vision-capable LLM via OpenRouter.
     - Reads OPENROUTER_VISION_MODEL from .env
     - If unset, reports available models and falls back to a supported vision model
  3. Uses prompt:
     "This is a chart or figure from a company's 10-K annual report. Describe what it shows,
      extract any specific numbers, labels, or data points visible, and state the time period
      covered if shown. If the image is decorative (a logo, a border, not a data chart),
      say so explicitly instead of describing it as data."
  4. Stores returned description as a new chunk with chunk_type='image' and image_path.
  5. Appends new image chunks to data/processed/chunks.jsonl (preserves existing chunks).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Paths & Config
# ─────────────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
MANIFEST_FILE = IMAGES_DIR / "extracted_images.json"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

# Default vision models if OPENROUTER_VISION_MODEL is unset
DEFAULT_VISION_MODELS = [
    "deepseek/deepseek-v4.1-flash",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
]

IMAGE_ANALYSIS_PROMPT = (
    "This is a chart or figure from a company's 10-K annual report. "
    "Describe what it shows, extract any specific numbers, labels, or data points visible, "
    "and state the time period covered if shown. If the image is decorative (a logo, a border, "
    "not a data chart), say so explicitly instead of describing it as data."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("describe_images")


def get_vision_model() -> str:
    """Check OPENROUTER_VISION_MODEL in .env or select first viable default."""
    configured = os.getenv("OPENROUTER_VISION_MODEL")
    if configured:
        return configured.strip()
    return DEFAULT_VISION_MODELS[0]


def get_vision_fallback_model() -> str | None:
    """Check OPENROUTER_VISION_FALLBACK_MODEL in .env or select secondary default."""
    configured = os.getenv("OPENROUTER_VISION_FALLBACK_MODEL")
    if configured:
        return configured.strip()
    return DEFAULT_VISION_MODELS[1] if len(DEFAULT_VISION_MODELS) > 1 else None


def encode_image_base64(image_path: Path) -> str:
    """Read binary image file and encode as base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vision_model(
    image_path: Path,
    model: str,
    api_key: str,
    max_retries: int = 3,
) -> str:
    """
    Send base64 encoded image to OpenRouter vision model.
    """
    b64_image = encode_image_base64(image_path)
    ext = image_path.suffix.lstrip(".").lower()
    mime = "image/png" if ext == "png" else "image/jpeg"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_ANALYSIS_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64_image}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1200,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/fin-risk-analyst",
        "X-Title": "Financial Annual Report Risk Analyst (Multimodal)",
    }

    for attempt in range(1, max_retries + 1):
        try:
            log.info("  Calling vision model %s for %s (attempt %d/%d)...", model, image_path.name, attempt, max_retries)
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)

            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content") or msg.get("reasoning") or ""
                    if content.strip():
                        return content.strip()

            if resp.status_code in (429, 500, 502, 503, 504):
                sleep_time = 2.0 ** attempt
                log.warning("  Vision API status %d (%s). Retrying in %.1fs...", resp.status_code, resp.text[:120], sleep_time)
                time.sleep(sleep_time)
                continue

            # Fail fast on other client errors
            raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}", response=resp)

        except Exception as e:
            if attempt == max_retries:
                raise
            sleep_time = 2.0 ** attempt
            log.warning("  Attempt %d failed (%s). Retrying in %.1fs...", attempt, e, sleep_time)
            time.sleep(sleep_time)

    raise RuntimeError(f"Vision model {model} failed after {max_retries} attempts.")


def is_decorative_image(description: str) -> bool:
    """
    Detect if the vision model explicitly identified the image as decorative
    (e.g., logo, icon, border, not a data chart).
    """
    text_lower = description.lower()
    patterns = [
        "not a data chart",
        "decorative logo",
        "decorative image",
        "decorative figure",
        "is decorative",
        "decorative border",
        "branding element",
        "corporate logo",
        "not a chart",
    ]
    return any(p in text_lower for p in patterns)


def describe_and_append_chunks():
    """
    Main pipeline:
      1. Load extracted images from manifest
      2. Call vision model for each image
      3. Format into chunk dicts with chunk_type='image' and image_path
      4. Filter out decorative images (logos/borders)
      5. Append only data-bearing chunks to data/processed/chunks.jsonl
    """
    if not MANIFEST_FILE.exists():
        log.error("Manifest %s not found. Run `python -m ingestion.extract_images` first.", MANIFEST_FILE)
        return

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        images = json.load(f)

    if not images:
        log.warning("Manifest contains 0 images.")
        return

    if not OPENROUTER_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY is not set in .env.")

    vision_model = get_vision_model()
    log.info("Using Vision Model: %s", vision_model)

    # Load existing chunk IDs to prevent duplicate insertion
    existing_chunk_ids = set()
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        c = json.loads(line)
                        existing_chunk_ids.add(c.get("chunk_id"))
                    except Exception:
                        pass

    new_chunks = []
    described_count = 0
    skipped_count = 0

    for idx, img_info in enumerate(images, 1):
        filename = img_info["filename"]
        img_path = IMAGES_DIR / filename
        company = img_info.get("company", "AAPL")
        year = img_info.get("year", 2024)
        section = img_info.get("section", "Item 5")
        chunk_id = f"{company}_{year}_IMG_{img_info['image_index']:02d}"

        if chunk_id in existing_chunk_ids:
            log.info("[%d/%d] Chunk %s already present in chunks.jsonl. Skipping.", idx, len(images), chunk_id)
            continue

        if not img_path.exists():
            log.warning("[%d/%d] File %s not found on disk. Skipping.", idx, len(images), img_path)
            skipped_count += 1
            continue

        log.info("=" * 60)
        log.info("[%d/%d] Describing: %s (Year %s, %s)", idx, len(images), filename, year, section[:40])

        try:
            try:
                description = call_vision_model(img_path, model=vision_model, api_key=OPENROUTER_KEY)
            except Exception as primary_err:
                fallback_model = get_vision_fallback_model()
                if fallback_model and fallback_model != vision_model:
                    log.warning("  Primary vision model %s failed (%s). Attempting fallback: %s...", vision_model, primary_err, fallback_model)
                    description = call_vision_model(img_path, model=fallback_model, api_key=OPENROUTER_KEY)
                else:
                    raise primary_err

            described_count += 1
            log.info("  ✓ Description received (%d chars)", len(description))

            # Detect if image is decorative (logo, border, etc.)
            if is_decorative_image(description):
                log.info("  ℹ [%s] Identified by vision model as decorative (%s). Skipping chunk storage.", filename, description[:70].replace("\n", " "))
                skipped_count += 1
                continue

            # Approximate tokens (words / 0.75)
            words = len(description.split())
            token_count = max(1, round(words / 0.75))

            # Construct chunk
            chunk_data = {
                "chunk_id": chunk_id,
                "company": company,
                "year": year,
                "section": section,
                "position_id": 9 if "Item 5" in section else 1,  # Item 5 position_id in filings
                "chunk_type": "image",
                "chunk_index": img_info.get("image_index", 0),
                "text": f"[Embedded Figure/Chart from {section}]\n{description}",
                "image_path": img_info.get("relative_path", f"data/processed/images/{filename}"),
                "token_count": token_count,
            }
            new_chunks.append(chunk_data)

        except Exception as e:
            log.error("  ✗ Failed to describe %s: %s", filename, e)
            skipped_count += 1

    if new_chunks:
        log.info("Appending %d new multimodal image chunks to %s...", len(new_chunks), CHUNKS_FILE)
        with open(CHUNKS_FILE, "a", encoding="utf-8") as f:
            for chunk in new_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        log.info("Successfully appended %d image chunks.", len(new_chunks))
    else:
        log.info("No new image chunks to append.")

    log.info("=" * 60)
    log.info("Vision processing summary:")
    log.info("  Total images considered  : %d", len(images))
    log.info("  Successfully described   : %d", described_count)
    log.info("  Skipped / Failed         : %d", skipped_count)
    log.info("=" * 60)


if __name__ == "__main__":
    describe_and_append_chunks()
