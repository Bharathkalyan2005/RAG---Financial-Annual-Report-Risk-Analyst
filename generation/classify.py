"""
generation/classify.py — Strict Question Classification into 5 Query Types
===========================================================================

Categories:
  - INFO: Single-filing factual retrieval (e.g. "Who was the auditor in 2023?")
  - COMPARISON: Cross-year qualitative comparison (e.g. "How did risk factors change between 2021 and 2024?")
  - TREND: Multi-year strategic direction / trajectory (e.g. "What is Apple's trend in services revenue?")
  - RISK: Disclosed threats, litigation, cybersecurity, operational risks
  - CALCULATION: Quantitative computations, ratios, growth rates, margins (math strictly in Pandas, never LLM)
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from generation.client import OpenRouterClient

log = logging.getLogger("generation.classify")

QueryCategory = Literal["INFO", "COMPARISON", "TREND", "RISK", "CALCULATION"]
VALID_CATEGORIES: set[QueryCategory] = {"INFO", "COMPARISON", "TREND", "RISK", "CALCULATION"}

CLASSIFY_PROMPT = """You are a financial query intent classifier.
Analyze the user's question and classify it into EXACTLY ONE of the following categories:

- INFO: Single-year or specific fact lookup (e.g. "What was Apple's revenue in 2024?", "Who is the CEO?")
- COMPARISON: Comparing qualitative facts or statements between specific years (e.g. "How did Item 1A differ between 2022 and 2023?")
- TREND: Multi-year qualitative trajectory or strategy evolution (e.g. "How has Apple's supply chain strategy evolved over the 5 years?")
- RISK: Focused specifically on threats, regulatory risks, lawsuits, or cybersecurity (e.g. "What are Apple's principal legal risks in 2024?")
- CALCULATION: Involves computing financial ratios, percentage growth, margins, debt ratios, or arithmetic (e.g. "What was the gross margin in 2024?", "Calculate YoY revenue growth from 2023 to 2024")

Respond with ONLY the single label: INFO, COMPARISON, TREND, RISK, or CALCULATION.
No punctuation, no explanation, no other words."""


def classify_question(
    question: str,
    client: OpenRouterClient | None = None,
) -> QueryCategory:
    """
    Classify a question using the LLM. Defaults to 'INFO' if classification is ambiguous.
    """
    client = client or OpenRouterClient()
    messages = [
        {"role": "system", "content": CLASSIFY_PROMPT},
        {"role": "user", "content": f"Question: {question}\nCategory:"},
    ]

    try:
        response = client.chat(messages, temperature=0.0, max_tokens=10).strip().upper()
        # Clean response (extract first valid word)
        match = re.search(r"\b(INFO|COMPARISON|TREND|RISK|CALCULATION)\b", response)
        if match:
            category = match.group(1)
            log.info("Classified question '%s' -> %s", question, category)
            return category  # type: ignore

        log.warning("Classification returned unexpected output '%s'. Defaulting to INFO.", response)
        return "INFO"
    except Exception as e:
        log.error("Classification failed (%s). Defaulting to INFO.", e)
        return "INFO"
