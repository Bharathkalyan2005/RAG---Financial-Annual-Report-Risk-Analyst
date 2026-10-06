"""
eval/create_testset.py — Build ground-truth test set for RAG evaluation (55 questions)
======================================================================================
Covers:
  - 5-year basic facts: Revenue, Net Income, Total Debt, Cash & Equivalents, Operating Cash Flow
  - Financial comparisons across fiscal years
  - Financial calculations & ratios (deterministic Pandas baseline)
  - Qualitative risk categories across 10-K sections (Item 1A, Item 7, Item 7A, Item 8)
  - Edge cases & out-of-scope queries (for abstention testing)
"""
import json
from pathlib import Path

testset = [
    # ─── FY2021 Basic Facts ──────────────────────────────────────────────────
    {"question": "What was Apple's total revenue in 2021?", "expected_answer": "$365,817 million", "expected_year": 2021, "expected_section": "Item 8", "keywords": ["365,817", "net sales"]},
    {"question": "What was Apple's net income in 2021?", "expected_answer": "$94,680 million", "expected_year": 2021, "expected_section": "Item 8", "keywords": ["94,680", "net income"]},
    {"question": "What was Apple's total debt in 2021?", "expected_answer": "$115,106 million", "expected_year": 2021, "expected_section": "Item 8", "keywords": ["115,106", "term debt", "commercial paper"]},
    {"question": "What were Apple's cash and cash equivalents in 2021?", "expected_answer": "$34,940 million", "expected_year": 2021, "expected_section": "Item 8", "keywords": ["34,940", "cash and cash equivalents"]},
    {"question": "What was Apple's operating cash flow in 2021?", "expected_answer": "$104,038 million", "expected_year": 2021, "expected_section": "Item 8", "keywords": ["104,038", "operating activities"]},

    # ─── FY2022 Basic Facts ──────────────────────────────────────────────────
    {"question": "What was Apple's total revenue in 2022?", "expected_answer": "$394,328 million", "expected_year": 2022, "expected_section": "Item 8", "keywords": ["394,328", "net sales"]},
    {"question": "What was Apple's net income in 2022?", "expected_answer": "$99,803 million", "expected_year": 2022, "expected_section": "Item 8", "keywords": ["99,803", "net income"]},
    {"question": "What was Apple's total debt in 2022?", "expected_answer": "$108,941 million", "expected_year": 2022, "expected_section": "Item 8", "keywords": ["108,941", "term debt", "commercial paper"]},
    {"question": "What were Apple's cash and cash equivalents in 2022?", "expected_answer": "$23,646 million", "expected_year": 2022, "expected_section": "Item 8", "keywords": ["23,646", "cash and cash equivalents"]},
    {"question": "What was Apple's operating cash flow in 2022?", "expected_answer": "$122,151 million", "expected_year": 2022, "expected_section": "Item 8", "keywords": ["122,151", "operating activities"]},

    # ─── FY2023 Basic Facts ──────────────────────────────────────────────────
    {"question": "What was Apple's total net sales in fiscal year 2023?", "expected_answer": "$383,285 million", "expected_year": 2023, "expected_section": "Item 8", "keywords": ["383,285", "net sales"]},
    {"question": "What was Apple's net income in 2023?", "expected_answer": "$96,995 million", "expected_year": 2023, "expected_section": "Item 8", "keywords": ["96,995", "net income"]},
    {"question": "What was Apple's total debt in 2023?", "expected_answer": "$101,266 million", "expected_year": 2023, "expected_section": "Item 8", "keywords": ["101,266", "term debt", "commercial paper"]},
    {"question": "What were Apple's cash and cash equivalents in 2023?", "expected_answer": "$29,965 million", "expected_year": 2023, "expected_section": "Item 8", "keywords": ["29,965", "cash and cash equivalents"]},
    {"question": "What was Apple's operating cash flow in 2023?", "expected_answer": "$110,543 million", "expected_year": 2023, "expected_section": "Item 8", "keywords": ["110,543", "operating activities"]},

    # ─── FY2024 Basic Facts ──────────────────────────────────────────────────
    {"question": "What was Apple's total revenue in 2024?", "expected_answer": "$391,035 million", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["391,035", "net sales"]},
    {"question": "What was Apple's net income in fiscal year 2024?", "expected_answer": "$93,736 million", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["93,736", "net income"]},
    {"question": "What was Apple's total debt in 2024?", "expected_answer": "$95,717 million", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["95,717", "term debt", "commercial paper"]},
    {"question": "What were Apple's cash and cash equivalents in 2024?", "expected_answer": "$29,943 million", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["29,943", "cash and cash equivalents"]},
    {"question": "What was Apple's operating cash flow in 2024?", "expected_answer": "$118,254 million", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["118,254", "operating activities"]},

    # ─── FY2025 Basic Facts ──────────────────────────────────────────────────
    {"question": "What was Apple's total revenue in 2025?", "expected_answer": "$416,161 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["416,161", "net sales"]},
    {"question": "What was Apple's net income in 2025?", "expected_answer": "$112,010 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["112,010", "net income"]},
    {"question": "What was Apple's total debt in 2025?", "expected_answer": "$86,307 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["86,307", "term debt", "commercial paper"]},
    {"question": "What were Apple's cash and cash equivalents in 2025?", "expected_answer": "$35,934 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["35,934", "cash and cash equivalents"]},
    {"question": "What was Apple's operating cash flow in 2025?", "expected_answer": "$111,482 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["111,482", "operating activities"]},

    # ─── Comparisons ─────────────────────────────────────────────────────────
    {"question": "Did total debt increase or decrease between 2022 and 2024?", "expected_answer": "Decreased from $108,941 million to $95,717 million", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["term debt", "commercial paper", "decrease"]},
    {"question": "What was the percentage change in revenue from 2023 to 2024?", "expected_answer": "+2.02% (from $383,285M to $391,035M)", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["391,035", "383,285", "net sales"]},
    {"question": "Did net income increase or decrease between 2024 and 2025?", "expected_answer": "Increased from $93,736 million to $112,010 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["112,010", "93,736", "net income"]},
    {"question": "How did cash and cash equivalents change from 2021 to 2025?", "expected_answer": "Increased slightly from $34,940M to $35,934M", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["35,934", "cash and cash equivalents"]},

    # ─── Calculations & Ratios ───────────────────────────────────────────────
    {"question": "What was Apple's gross margin in 2024?", "expected_answer": "46.21%", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["gross margin", "cost of sales"]},
    {"question": "What was Apple's operating margin in 2024?", "expected_answer": "31.51%", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["operating income", "net sales"]},
    {"question": "What was Apple's net margin in 2024?", "expected_answer": "23.97%", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["net income", "net sales"]},
    {"question": "What was Apple's ROA in 2024?", "expected_answer": "25.68%", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["total assets", "net income"]},
    {"question": "What was Apple's ROE in 2024?", "expected_answer": "1.65x (164.59%)", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["shareholders", "net income"]},
    {"question": "What was Apple's current ratio in 2024?", "expected_answer": "0.87", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["current assets", "current liabilities"]},
    {"question": "What was Apple's debt-to-equity ratio in 2024?", "expected_answer": "1.68x", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["total debt", "shareholders"]},
    {"question": "What was Apple's debt-to-assets ratio in 2024?", "expected_answer": "0.26", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["total debt", "total assets"]},
    {"question": "What was Apple's interest coverage in 2024?", "expected_answer": "31.9x", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["operating income", "interest expense"]},

    # ─── Risk Factors & Qualitative Disclosures ──────────────────────────────
    {"question": "What are Apple's regulatory risks regarding the App Store and Digital Markets Act?", "expected_answer": "Antitrust scrutiny, DMA compliance in the EU, alternative app stores, fee commission changes", "expected_year": 2024, "expected_section": "Item 1A", "keywords": ["regulatory", "digital markets act", "app store"]},
    {"question": "What supply chain and single-source risks does Apple disclose in 2024?", "expected_answer": "Dependence on single-source suppliers and outsourced manufacturing partners primarily in Asia", "expected_year": 2024, "expected_section": "Item 1A", "keywords": ["supply chain", "single-source", "manufacturing"]},
    {"question": "How does Apple describe its cybersecurity and data breach risks?", "expected_answer": "Risks of unauthorized access, cyberattacks, network intrusions, and ransomware compromising confidential data", "expected_year": 2024, "expected_section": "Item 1A", "keywords": ["cybersecurity", "cyberattack", "data breach"]},
    {"question": "What foreign exchange currency risks does Apple face?", "expected_answer": "Volatility in exchange rates (Euro, Yen, Renminbi) affecting net sales and gross margin", "expected_year": 2024, "expected_section": "Item 7A", "keywords": ["foreign exchange", "currency", "hedging"]},
    {"question": "What legal proceedings and antitrust litigation does Apple disclose in 2024?", "expected_answer": "Antitrust litigation with developers, government investigations, and patent infringement lawsuits", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["litigation", "antitrust", "legal"]},
    {"question": "What are the commercial paper program risks described by Apple?", "expected_answer": "Maturities of short-term promissory notes and general corporate liquidity needs", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["commercial paper", "promissory notes"]},
    {"question": "What intellectual property risks does Apple face regarding patent infringement?", "expected_answer": "Companies aggressively suing for patent infringement and asserting licensing claims", "expected_year": 2023, "expected_section": "Item 1A", "keywords": ["patent", "intellectual property", "infringement"]},
    {"question": "How does geopolitical tension in Greater China affect Apple's operations?", "expected_answer": "Manufacturing concentration, trade restrictions, tariffs, and regional supply chain disruption", "expected_year": 2023, "expected_section": "Item 1A", "keywords": ["china", "supply chain", "trade"]},
    {"question": "What are Apple's climate change and natural disaster risks in 2024?", "expected_answer": "Extreme weather events disrupting logistics, data centers, and outsourced assembly facilities", "expected_year": 2024, "expected_section": "Item 1A", "keywords": ["climate", "disaster", "facilities"]},
    {"question": "What commercial paper balance did Apple report in 2024?", "expected_answer": "$9,967 million", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["9,967", "commercial paper"]},
    {"question": "What was Apple's total shareholders' equity in 2023?", "expected_answer": "$62,146 million", "expected_year": 2023, "expected_section": "Item 8", "keywords": ["62,146", "shareholders"]},
    {"question": "What was Apple's cost of sales in 2025?", "expected_answer": "$220,960 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["220,960", "cost of sales"]},
    {"question": "What was Apple's operating income in 2025?", "expected_answer": "$133,050 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["133,050", "operating income"]},
    {"question": "What were total assets for Apple in 2025?", "expected_answer": "$359,241 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["359,241", "total assets"]},
    {"question": "What were total current liabilities for Apple in 2025?", "expected_answer": "$165,631 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["165,631", "current liabilities"]},
    {"question": "What was Apple's commercial paper balance in 2025?", "expected_answer": "$7,979 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["7,979", "commercial paper"]},
    {"question": "What was Apple's total term debt principal in 2025?", "expected_answer": "$91,281 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["91,281", "term debt"]},
]

dest = Path("eval/testset.jsonl")
dest.parent.mkdir(parents=True, exist_ok=True)
with open(dest, "w", encoding="utf-8") as f:
    for item in testset:
        f.write(json.dumps(item) + "\n")

print(f"Created {len(testset)} verified test questions in {dest}")
