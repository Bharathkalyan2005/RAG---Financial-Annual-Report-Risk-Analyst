"""
risk/detect.py — Phase 8: Risk Detection across 5-Year 10-K Filings
===================================================================

Defines canonical risk categories:
  1. Liquidity: Cash flow, commercial paper, debt maturities, credit facilities
  2. Credit: Counterparty default, customer credit risk, banking stability
  3. FX (Foreign Exchange): Currency volatility, dollar strength, hedging efficacy
  4. Regulatory: Antitrust, App Store regulations, Digital Markets Act, tariffs, export controls
  5. Cybersecurity: Data breaches, malware, ransomware, unauthorized access, vulnerabilities
  6. Operational: Supply chain concentration, single-source suppliers, manufacturing halts
  7. Legal: Lawsuits, patent disputes, Epic Games antitrust, State Aid tax investigations
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("risk.detect")

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
OUTPUT_FILE = Path("data/processed/risk_detections.json")

RISK_CATEGORIES = {
    "Liquidity": [
        "liquidity", "cash flow", "commercial paper", "credit facility", "debt maturity",
        "refinancing", "borrowing capacity", "debt obligations", "repay", "working capital"
    ],
    "Credit": [
        "credit risk", "counterparty", "default", "customer credit", "banking partner",
        "insolvency", "financial condition of counterparties", "credit losses"
    ],
    "FX": [
        "foreign exchange", "foreign currency", "currency fluctuation", "dollar appreciation",
        "euro", "yen", "renminbi", "exchange rate", "currency exposure", "fx hedge"
    ],
    "Regulatory": [
        "regulatory", "regulation", "antitrust", "digital markets act", "dma", "app store regulation",
        "privacy law", "gdpr", "trade restriction", "tariff", "export control", "sanctions", "compliance"
    ],
    "Cybersecurity": [
        "cybersecurity", "cyberattack", "data breach", "ransomware", "malware",
        "unauthorized access", "system compromise", "confidential data", "security vulnerability",
        "hacker", "network intrusion"
    ],
    "Operational": [
        "supply chain", "single-source", "component shortage", "manufacturing partner",
        "logistics disruption", "production delay", "assembly facility", "supplier concentration",
        "natural disaster", "outsourced manufacturing"
    ],
    "Legal": [
        "litigation", "lawsuit", "patent infringement", "intellectual property dispute",
        "legal proceeding", "antitrust investigation", "court", "legal settlement",
        "state aid", "investigation by government"
    ],
}

EMPHASIS_KEYWORDS = [
    "material", "materially", "significant", "significantly", "adverse", "adversely",
    "substantially", "substantial", "critical", "severe", "unprecedented", "heightened"
]


def detect_risks_in_chunks() -> dict[int, dict[str, Any]]:
    """
    Parse Item 1A chunks and detect category mentions, justifications, and emphasis scores.
    Returns: {year: {category: {"count": N, "emphasis_count": M, "mentions": [...]}}}
    """
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"{CHUNKS_FILE} not found.")

    detections: dict[int, dict[str, Any]] = {}

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            sec = (chunk.get("section") or "").lower()
            if "risk factors" not in sec and "item 1a" not in sec:
                continue

            year = int(chunk["year"])
            if year not in detections:
                detections[year] = {
                    cat: {"count": 0, "emphasis_count": 0, "mentions": []}
                    for cat in RISK_CATEGORIES
                }

            text = chunk.get("text", "")
            # Split into sentences for fine-grained mention & justification extraction
            sentences = re.split(r"(?<=[.!?])\s+", text)

            for sentence in sentences:
                s_lower = sentence.lower()
                if len(s_lower.split()) < 5:
                    continue

                for cat, keywords in RISK_CATEGORIES.items():
                    matched_kw = [kw for kw in keywords if kw in s_lower]
                    if matched_kw:
                        detections[year][cat]["count"] += 1

                        # Check management emphasis intensity
                        matched_emphasis = [ek for ek in EMPHASIS_KEYWORDS if ek in s_lower]
                        if matched_emphasis:
                            detections[year][cat]["emphasis_count"] += len(matched_emphasis)

                        if len(detections[year][cat]["mentions"]) < 10:
                            detections[year][cat]["mentions"].append({
                                "keyword": matched_kw[0],
                                "emphasis": matched_emphasis,
                                "sentence": sentence.strip()[:250],
                                "position_id": chunk.get("position_id"),
                            })

    return detections


def run_detection():
    log.info("Running risk detection across all 5 years of 10-K filings...")
    results = detect_risks_in_chunks()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log.info("Risk detections saved to %s", OUTPUT_FILE)
    print("\n" + "=" * 70)
    print("  RISK DETECTION SUMMARY (Mentions / Management Emphasis Count)")
    print("=" * 70)
    cats = list(RISK_CATEGORIES.keys())
    header = f"{'Year':<6} | " + " | ".join(f"{c[:7]:<7}" for c in cats)
    print(header)
    print("-" * len(header))

    for yr in sorted(results.keys()):
        row = f"{yr:<6} | "
        vals = []
        for c in cats:
            info = results[yr][c]
            vals.append(f"{info['count']}({info['emphasis_count']})")
        row += " | ".join(f"{v:<7}" for v in vals)
        print(row)
    print("=" * 70)


if __name__ == "__main__":
    run_detection()
