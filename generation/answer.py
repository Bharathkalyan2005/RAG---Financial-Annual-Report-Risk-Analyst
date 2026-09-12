"""
generation/answer.py — Citation-Aware LLM Answer Generation with Abstention
============================================================================

Pipeline:
  1. Check top rerank score against ABSTAIN_THRESHOLD (default 0.30).
     If below threshold, abstain immediately without calling the LLM.
  2. Assemble numbered context passages with Year, Section, and Position ID.
  3. Prompt LLM for strict JSON response:
     {"answer": "...", "citations": [{"year": 2024, "section": "...", "position_id": 3}], "confidence": 0.0-1.0}
  4. Parse JSON, stripping markdown code fences.
  5. Retry once if JSON is malformed.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv

from generation.client import OpenRouterClient

load_dotenv()

log = logging.getLogger("generation.answer")

DEFAULT_ABSTAIN_THRESHOLD = float(os.getenv("ABSTAIN_THRESHOLD", "0.30"))


def _clean_json_text(text: str) -> str:
    """Strip markdown code fences and extraneous leading/trailing whitespace."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no fences, attempt to find first '{' and last '}'
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return text[first_brace : last_brace + 1].strip()
    return text


def generate_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
    client: OpenRouterClient | None = None,
) -> dict[str, Any]:
    """
    Generate a cited answer from retrieved chunks or abstain if confidence is insufficient.
    """
    if not retrieved_chunks:
        return {
            "answer": "Insufficient evidence in the available filings to answer this question reliably.",
            "citations": [],
            "confidence": 0.0,
            "abstain_reason": "No chunks retrieved",
        }

    # Abstention check: top rerank score below threshold
    top_score = retrieved_chunks[0].get("rerank_score", 0.0)
    if top_score < abstain_threshold:
        log.info("Top rerank score %.3f < threshold %.3f. Abstaining.", top_score, abstain_threshold)
        return {
            "answer": "Insufficient evidence in the available filings to answer this question reliably.",
            "citations": [],
            "confidence": 0.0,
            "abstain_reason": f"Top rerank score {top_score:.3f} below threshold {abstain_threshold:.3f}",
        }

    # Format context passages
    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        block = (
            f"[Source {idx}]\n"
            f"Company: {chunk.get('company', 'AAPL')}\n"
            f"Year: {chunk.get('year')}\n"
            f"Section: {chunk.get('section', 'General')}\n"
            f"Position ID: {chunk.get('position_id')}\n"
            f"Type: {chunk.get('chunk_type', 'text')}\n"
            f"Text:\n{chunk.get('text', '').strip()}"
        )
        context_blocks.append(block)

    context_str = "\n\n" + ("=" * 50) + "\n\n".join(context_blocks) + "\n\n" + ("=" * 50)

    system_prompt = (
        "You are an expert Financial and Risk Analyst analyzing multi-year 10-K filings.\n"
        "Your task is to answer the user's question accurately using ONLY the provided sources.\n"
        "Rules:\n"
        "1. Every factual statement must cite its exact source filing year, section, and position_id.\n"
        "2. Do NOT invent, extrapolate, or hallucinate numbers or facts.\n"
        "3. You must respond ONLY in strict JSON format with exactly these keys:\n"
        "   {\n"
        "     \"answer\": \"Your comprehensive, cited answer here...\",\n"
        "     \"citations\": [\n"
        "       {\"year\": 2024, \"section\": \"Item 1A. Risk Factors\", \"position_id\": 12}\n"
        "     ],\n"
        "     \"confidence\": 0.95\n"
        "   }\n"
        "Do NOT include any markdown code blocks or conversational text outside the JSON."
    )

    user_prompt = f"Sources:\n{context_str}\n\nQuestion: {question}\n\nJSON Response:"

    client = client or OpenRouterClient()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_response = client.chat(messages, temperature=0.0)

    # Parse JSON with one retry if parsing fails
    cleaned = _clean_json_text(raw_response)
    try:
        parsed = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError:
        log.warning("JSON parsing failed on initial attempt. Retrying with strict formatting prompt...")
        retry_messages = messages + [
            {"role": "assistant", "content": raw_response},
            {"role": "user", "content": "Your previous response was not valid JSON. Please respond with valid JSON only."},
        ]
        second_response = client.chat(retry_messages, temperature=0.0)
        cleaned_second = _clean_json_text(second_response)
        try:
            return json.loads(cleaned_second)
        except json.JSONDecodeError as e:
            log.error("Failed to parse JSON on retry. Raw output: %s", second_response)
            return {
                "answer": second_response.strip(),
                "citations": [],
                "confidence": 0.5,
                "error": f"JSON parse error: {e}",
            }
