"""
agent/verify.py — Claim-Level Verification Before Answering
===========================================================

Validates draft answers against collected evidence:
1. Splits draft into individual sentence-level claims.
2. For numeric claims: checks deterministically (no LLM) whether numbers match
   the evidence chunks or tool results after normalising commas, $, %, B/M units.
3. For non-numeric claims: evaluates factual grounding against evidence passages
   using a targeted LLM verification prompt (or deterministic token overlap fallback).
4. Prunes unsupported claims. If no verified claims remain, returns an abstention.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from generation.client import OpenRouterClient

log = logging.getLogger("agent.verify")

ABSTAIN_MESSAGE = "Insufficient evidence in the available filings to answer this question reliably."


def extract_sentences(text: str) -> list[str]:
    """Split text into distinct sentences / claims."""
    # Split by sentence-ending punctuation or bullet points
    lines = text.strip().split("\n")
    sentences = []
    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue
        # Split line on period, question mark, or exclamation followed by space or end
        raw_parts = re.split(r"(?<=[.!?])\s+", cleaned_line)
        for part in raw_parts:
            p = part.strip()
            # Remove leading list markers like "- ", "* ", "1. "
            p = re.sub(r"^(\*|-|\d+\.)\s+", "", p).strip()
            if len(p) >= 10:  # Ignore trivial fragments
                sentences.append(p)
    return sentences


def extract_numbers_from_text(text: str) -> list[float]:
    """Extract and normalize all numbers (including decimals, millions/billions) from a text string."""
    numbers: list[float] = []

    # Match patterns like $114,301, 114,301 million, 114.3 billion, 24.5%, 3.5x
    pattern = re.compile(
        r"[\$]?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*(billion|million|bn|m|b|%|x)?",
        re.IGNORECASE,
    )

    for match in pattern.finditer(text):
        raw_num = match.group(1).replace(",", "")
        unit = (match.group(2) or "").lower()
        try:
            val = float(raw_num)
            if unit in ("billion", "bn", "b"):
                numbers.append(val * 1000.0)  # In millions
                numbers.append(val * 1_000_000_000.0)
                numbers.append(val)
            elif unit in ("million", "m"):
                numbers.append(val)
                numbers.append(val * 1_000_000.0)
            elif unit == "%":
                numbers.append(val)
                numbers.append(val / 100.0)
            else:
                numbers.append(val)
        except ValueError:
            continue

    return numbers


def _numeric_value_in_evidence(target_val: float, evidence_numbers: list[float], tolerance: float = 0.015) -> bool:
    """Check if target_val is found in evidence_numbers within tolerance."""
    for ev in evidence_numbers:
        if ev == 0.0 and target_val == 0.0:
            return True
        if target_val != 0.0 and abs(ev - target_val) / abs(target_val) <= tolerance:
            return True
        if abs(ev - target_val) <= 0.05:
            return True
    return False


def _verify_numeric_claim(claim: str, evidence_numbers: list[float], evidence_corpus: str) -> bool:
    """Deterministic check: every substantial number in claim must be supported by evidence."""
    claim_nums = extract_numbers_from_text(claim)
    # Filter out calendar years (2020-2026) and single digits that might be bullet numbers
    significant_nums = [
        n for n in claim_nums
        if not (2020 <= n <= 2026 and n.is_integer()) and not (n in (1.0, 2.0, 3.0, 4.0, 5.0) and f"{int(n)}" in claim)
    ]

    if not significant_nums:
        # No significant financial figures to verify
        return True

    # At least all significant numbers must match evidence
    for num in significant_nums:
        # Check raw string occurrence first (e.g. "114,301" or "383,285")
        num_int_str = f"{int(num):,}" if num.is_integer() else f"{num}"
        if num_int_str in evidence_corpus or f"{num}" in evidence_corpus:
            continue
        # Check normalized float values
        if not _numeric_value_in_evidence(num, evidence_numbers):
            return False

    return True


def _verify_non_numeric_claim_llm(
    claim: str,
    evidence_text: str,
    client: Optional[OpenRouterClient] = None,
) -> bool:
    """Verify non-numeric qualitative claim via LLM or keyword overlap fallback."""
    if not client:
        # Deterministic keyword overlap fallback if LLM client is unavailable
        words = [w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", claim)]
        if not words:
            return True
        evidence_lower = evidence_text.lower()
        matches = sum(1 for w in words if w in evidence_lower)
        return (matches / len(words)) >= 0.30

    prompt = (
        "You are a strict financial auditor. Verify if the following claim is supported by the evidence.\n\n"
        f"EVIDENCE PASSAGE:\n\"\"\"{evidence_text[:3000]}\"\"\"\n\n"
        f"CLAIM TO VERIFY:\n\"{claim}\"\n\n"
        "Is this claim supported or entailed by the evidence? Answer with ONLY 'YES' or 'NO'."
    )

    try:
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        return "yes" in response.strip().lower()
    except Exception as e:
        log.warning("LLM claim verification failed (%s), defaulting to grounded token match", e)
        words = [w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", claim)]
        if not words:
            return True
        evidence_lower = evidence_text.lower()
        matches = sum(1 for w in words if w in evidence_lower)
        return (matches / len(words)) >= 0.30


def verify_answer(
    draft_answer: str,
    evidence_chunks: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    client: Optional[OpenRouterClient] = None,
) -> dict[str, Any]:
    """
    Verify draft answer claims against evidence chunks and tool outputs.

    Returns:
      {
        "verified_answer": str,
        "supported_claims": list[str],
        "removed_claims": list[str],
        "all_passed": bool,
        "abstain": bool,
      }
    """
    if not draft_answer or draft_answer.strip() == ABSTAIN_MESSAGE:
        return {
            "verified_answer": ABSTAIN_MESSAGE,
            "supported_claims": [],
            "removed_claims": [],
            "all_passed": False,
            "abstain": True,
        }

    # 1. Build aggregated evidence corpus and numeric pool
    evidence_text_parts = [c.get("text", "") for c in evidence_chunks]
    for tr in tool_results:
        data = tr.get("data")
        if isinstance(data, (dict, list)):
            import json
            evidence_text_parts.append(json.dumps(data))
        elif data is not None:
            evidence_text_parts.append(str(data))

    corpus = "\n\n".join(evidence_text_parts)
    evidence_numbers = extract_numbers_from_text(corpus)

    # 2. Split draft into claims
    claims = extract_sentences(draft_answer)
    if not claims:
        # Fallback to single string check
        claims = [draft_answer.strip()]

    supported_claims: list[str] = []
    removed_claims: list[str] = []

    # 3. Check each claim
    for claim in claims:
        has_numbers = bool(extract_numbers_from_text(claim))
        if has_numbers:
            ok = _verify_numeric_claim(claim, evidence_numbers, corpus)
        else:
            ok = _verify_non_numeric_claim_llm(claim, corpus, client=client)

        if ok:
            supported_claims.append(claim)
        else:
            log.info("Removed unsupported claim: %s", claim)
            removed_claims.append(claim)

    if not supported_claims:
        return {
            "verified_answer": ABSTAIN_MESSAGE,
            "supported_claims": [],
            "removed_claims": removed_claims,
            "all_passed": False,
            "abstain": True,
        }

    # Reconstruct verified answer
    verified_answer = " ".join(supported_claims)

    return {
        "verified_answer": verified_answer,
        "supported_claims": supported_claims,
        "removed_claims": removed_claims,
        "all_passed": len(removed_claims) == 0,
        "abstain": False,
    }
