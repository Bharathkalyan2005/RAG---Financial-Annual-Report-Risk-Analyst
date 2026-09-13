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

QueryCategory = Literal["GREETING", "INFO", "COMPARISON", "TREND", "RISK", "CALCULATION"]
VALID_CATEGORIES: set[QueryCategory] = {"GREETING", "INFO", "COMPARISON", "TREND", "RISK", "CALCULATION"}

GREETING_REGEX = re.compile(
    r"^\s*(hi|hello|hey|greetings|howdy|hola|yo|sup|good\s+(morning|afternoon|evening|day)|"
    r"who\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+this|help|how\s+do\s+you\s+work|"
    r"what\s+do\s+you\s+do|tell\s+me\s+about\s+yourself|capabilities)\b",
    re.IGNORECASE,
)

GREETING_RESPONSE = (
    "Hello! 👋 I am your **Financial Annual Report Risk Analyst**, specialized in auditing and analyzing 5 years (FY2021–FY2025) of **Apple Inc. (AAPL)** SEC Form 10-K filings.\n\n"
    "### What I can help you with:\n"
    "- 💰 **Audited Financials**: Look up debt, revenues, net income, and operating cash flows.\n"
    "- 🧮 **Deterministic Ratios**: Compute gross margins, operating margins, ROE, ROA, and debt-to-equity in pure Pandas.\n"
    "- 🛡️ **Risk Factor Analysis**: In-depth intelligence on EU Digital Markets Act (DMA), App Store antitrust, single-source supplier risks, and litigation.\n"
    "- 📈 **5-Year Trajectories**: Track financial performance and deleveraging trends across 2021–2025.\n\n"
    "**Try asking one of the suggested prompts above or type a question like:**\n"
    "- *\"What was Apple's total debt in 2024?\"*\n"
    "- *\"What are Apple's regulatory risks regarding the App Store?\"*\n"
    "- *\"What supply chain single-source risks does Apple disclose?\"*"
)


def is_greeting_or_help(question: str) -> bool:
    """Check if query is a greeting, capability inquiry, or help request."""
    q = question.strip().lower()
    if not q:
        return True
    if len(q) <= 15 and q in {"hi", "hello", "hey", "help", "yo", "hola", "howdy", "sup", "greetings"}:
        return True
    return bool(GREETING_REGEX.search(q))


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
    if is_greeting_or_help(question):
        return "GREETING"

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
