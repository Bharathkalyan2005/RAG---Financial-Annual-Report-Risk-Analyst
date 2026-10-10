"""
eval/update_testset.py — Extend eval/testset.jsonl with 20 New Multi-Step Questions
==================================================================================

Adds 20 multi-step benchmark questions covering:
  - multi_year comparisons (e.g. debt and FX risk 2022 to 2025)
  - ambiguous_line_item queries (e.g. "total income" disambiguation)
  - ratio_and_risk composite questions
  - chart-based questions (Item 5 stock return performance graph)

Tags all questions with a distinct "type" field for granular comparative reporting.
"""

import json
from pathlib import Path

BASE_TESTSET = [
    # 1-25: Single facts (FY2021 - FY2025)
    {"question": "What was Apple's total revenue in 2021?", "expected_answer": "$365,817 million", "expected_year": 2021, "expected_section": "Item 8", "keywords": ["365,817", "net sales"], "type": "single_fact"},
    {"question": "What was Apple's net income in 2021?", "expected_answer": "$94,680 million", "expected_year": 2021, "expected_section": "Item 8", "keywords": ["94,680", "net income"], "type": "single_fact"},
    {"question": "What was Apple's total debt in 2021?", "expected_answer": "$115,106 million", "expected_year": 2021, "expected_section": "Item 8", "keywords": ["115,106", "term debt", "commercial paper"], "type": "single_fact"},
    {"question": "What were Apple's cash and cash equivalents in 2021?", "expected_answer": "$34,940 million", "expected_year": 2021, "expected_section": "Item 8", "keywords": ["34,940", "cash and cash equivalents"], "type": "single_fact"},
    {"question": "What was Apple's operating cash flow in 2021?", "expected_answer": "$104,038 million", "expected_year": 2021, "expected_section": "Item 8", "keywords": ["104,038", "operating activities"], "type": "single_fact"},
    {"question": "What was Apple's total revenue in 2022?", "expected_answer": "$394,328 million", "expected_year": 2022, "expected_section": "Item 8", "keywords": ["394,328", "net sales"], "type": "single_fact"},
    {"question": "What was Apple's net income in 2022?", "expected_answer": "$99,803 million", "expected_year": 2022, "expected_section": "Item 8", "keywords": ["99,803", "net income"], "type": "single_fact"},
    {"question": "What was Apple's total debt in 2022?", "expected_answer": "$108,941 million", "expected_year": 2022, "expected_section": "Item 8", "keywords": ["108,941", "term debt", "commercial paper"], "type": "single_fact"},
    {"question": "What were Apple's cash and cash equivalents in 2022?", "expected_answer": "$23,646 million", "expected_year": 2022, "expected_section": "Item 8", "keywords": ["23,646", "cash and cash equivalents"], "type": "single_fact"},
    {"question": "What was Apple's operating cash flow in 2022?", "expected_answer": "$122,151 million", "expected_year": 2022, "expected_section": "Item 8", "keywords": ["122,151", "operating activities"], "type": "single_fact"},
    {"question": "What was Apple's total net sales in fiscal year 2023?", "expected_answer": "$383,285 million", "expected_year": 2023, "expected_section": "Item 8", "keywords": ["383,285", "net sales"], "type": "single_fact"},
    {"question": "What was Apple's net income in 2023?", "expected_answer": "$96,995 million", "expected_year": 2023, "expected_section": "Item 8", "keywords": ["96,995", "net income"], "type": "single_fact"},
    {"question": "What was Apple's total debt in 2023?", "expected_answer": "$101,266 million", "expected_year": 2023, "expected_section": "Item 8", "keywords": ["101,266", "term debt", "commercial paper"], "type": "single_fact"},
    {"question": "What were Apple's cash and cash equivalents in 2023?", "expected_answer": "$29,965 million", "expected_year": 2023, "expected_section": "Item 8", "keywords": ["29,965", "cash and cash equivalents"], "type": "single_fact"},
    {"question": "What was Apple's operating cash flow in 2023?", "expected_answer": "$110,543 million", "expected_year": 2023, "expected_section": "Item 8", "keywords": ["110,543", "operating activities"], "type": "single_fact"},
    {"question": "What was Apple's total revenue in 2024?", "expected_answer": "$391,035 million", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["391,035", "net sales"], "type": "single_fact"},
    {"question": "What was Apple's net income in fiscal year 2024?", "expected_answer": "$93,736 million", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["93,736", "net income"], "type": "single_fact"},
    {"question": "What was Apple's total debt in 2024?", "expected_answer": "$95,717 million", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["95,717", "term debt", "commercial paper"], "type": "single_fact"},
    {"question": "What were Apple's cash and cash equivalents in 2024?", "expected_answer": "$29,943 million", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["29,943", "cash and cash equivalents"], "type": "single_fact"},
    {"question": "What was Apple's operating cash flow in 2024?", "expected_answer": "$118,254 million", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["118,254", "operating activities"], "type": "single_fact"},
    {"question": "What was Apple's total revenue in 2025?", "expected_answer": "$416,161 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["416,161", "net sales"], "type": "single_fact"},
    {"question": "What was Apple's net income in 2025?", "expected_answer": "$112,010 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["112,010", "net income"], "type": "single_fact"},
    {"question": "What was Apple's total debt in 2025?", "expected_answer": "$86,307 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["86,307", "term debt", "commercial paper"], "type": "single_fact"},
    {"question": "What were Apple's cash and cash equivalents in 2025?", "expected_answer": "$35,934 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["35,934", "cash and cash equivalents"], "type": "single_fact"},
    {"question": "What was Apple's operating cash flow in 2025?", "expected_answer": "$111,482 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["111,482", "operating activities"], "type": "single_fact"},

    # 26-29: Multi-Year comparisons
    {"question": "Did total debt increase or decrease between 2022 and 2024?", "expected_answer": "Decreased from $108,941 million to $95,717 million", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["term debt", "commercial paper", "decrease"], "type": "multi_year"},
    {"question": "What was the percentage change in revenue from 2023 to 2024?", "expected_answer": "+2.02% (from $383,285M to $391,035M)", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["391,035", "383,285", "net sales"], "type": "multi_year"},
    {"question": "Did net income increase or decrease between 2024 and 2025?", "expected_answer": "Increased from $93,736 million to $112,010 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["112,010", "93,736", "net income"], "type": "multi_year"},
    {"question": "How did cash and cash equivalents change from 2021 to 2025?", "expected_answer": "Increased slightly from $34,940M to $35,934M", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["35,934", "cash and cash equivalents"], "type": "multi_year"},

    # 30-38: Ratios
    {"question": "What was Apple's gross margin in 2024?", "expected_answer": "46.21%", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["gross margin", "cost of sales"], "type": "ratio"},
    {"question": "What was Apple's operating margin in 2024?", "expected_answer": "31.51%", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["operating income", "net sales"], "type": "ratio"},
    {"question": "What was Apple's net margin in 2024?", "expected_answer": "23.97%", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["net income", "net sales"], "type": "ratio"},
    {"question": "What was Apple's ROA in 2024?", "expected_answer": "25.68%", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["total assets", "net income"], "type": "ratio"},
    {"question": "What was Apple's ROE in 2024?", "expected_answer": "1.65x (164.59%)", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["shareholders", "net income"], "type": "ratio"},
    {"question": "What was Apple's current ratio in 2024?", "expected_answer": "0.87", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["current assets", "current liabilities"], "type": "ratio"},
    {"question": "What was Apple's debt-to-equity ratio in 2024?", "expected_answer": "1.68x", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["total debt", "shareholders"], "type": "ratio"},
    {"question": "What was Apple's debt-to-assets ratio in 2024?", "expected_answer": "0.26", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["total debt", "total assets"], "type": "ratio"},
    {"question": "What was Apple's interest coverage in 2024?", "expected_answer": "31.9x", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["operating income", "interest expense"], "type": "ratio"},

    # 39-47: Qualitative risk disclosures
    {"question": "What are Apple's regulatory risks regarding the App Store and Digital Markets Act?", "expected_answer": "Antitrust scrutiny, DMA compliance in the EU, alternative app stores, fee commission changes", "expected_year": 2024, "expected_section": "Item 1A", "keywords": ["regulatory", "digital markets act", "app store"], "type": "risk"},
    {"question": "What supply chain and single-source risks does Apple disclose in 2024?", "expected_answer": "Dependence on single-source suppliers and outsourced manufacturing partners primarily in Asia", "expected_year": 2024, "expected_section": "Item 1A", "keywords": ["supply chain", "single-source", "manufacturing"], "type": "risk"},
    {"question": "How does Apple describe its cybersecurity and data breach risks?", "expected_answer": "Risks of unauthorized access, cyberattacks, network intrusions, and ransomware compromising confidential data", "expected_year": 2024, "expected_section": "Item 1A", "keywords": ["cybersecurity", "cyberattack", "data breach"], "type": "risk"},
    {"question": "What foreign exchange currency risks does Apple face?", "expected_answer": "Volatility in exchange rates (Euro, Yen, Renminbi) affecting net sales and gross margin", "expected_year": 2024, "expected_section": "Item 7A", "keywords": ["foreign exchange", "currency", "hedging"], "type": "risk"},
    {"question": "What legal proceedings and antitrust litigation does Apple disclose in 2024?", "expected_answer": "Antitrust litigation with developers, government investigations, and patent infringement lawsuits", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["litigation", "antitrust", "legal"], "type": "risk"},
    {"question": "What are the commercial paper program risks described by Apple?", "expected_answer": "Maturities of short-term promissory notes and general corporate liquidity needs", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["commercial paper", "promissory notes"], "type": "risk"},
    {"question": "What intellectual property risks does Apple face regarding patent infringement?", "expected_answer": "Companies aggressively suing for patent infringement and asserting licensing claims", "expected_year": 2023, "expected_section": "Item 1A", "keywords": ["patent", "intellectual property", "infringement"], "type": "risk"},
    {"question": "How does geopolitical tension in Greater China affect Apple's operations?", "expected_answer": "Manufacturing concentration, trade restrictions, tariffs, and regional supply chain disruption", "expected_year": 2023, "expected_section": "Item 1A", "keywords": ["china", "supply chain", "trade"], "type": "risk"},
    {"question": "What are Apple's climate change and natural disaster risks in 2024?", "expected_answer": "Extreme weather events disrupting logistics, data centers, and outsourced assembly facilities", "expected_year": 2024, "expected_section": "Item 1A", "keywords": ["climate", "disaster", "facilities"], "type": "risk"},

    # 48-55: Additional single facts
    {"question": "What commercial paper balance did Apple report in 2024?", "expected_answer": "$9,967 million", "expected_year": 2024, "expected_section": "Item 8", "keywords": ["9,967", "commercial paper"], "type": "single_fact"},
    {"question": "What was Apple's total shareholders' equity in 2023?", "expected_answer": "$62,146 million", "expected_year": 2023, "expected_section": "Item 8", "keywords": ["62,146", "shareholders"], "type": "single_fact"},
    {"question": "What was Apple's cost of sales in 2025?", "expected_answer": "$220,960 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["220,960", "cost of sales"], "type": "single_fact"},
    {"question": "What was Apple's operating income in 2025?", "expected_answer": "$133,050 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["133,050", "operating income"], "type": "single_fact"},
    {"question": "What were total assets for Apple in 2025?", "expected_answer": "$359,241 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["359,241", "total assets"], "type": "single_fact"},
    {"question": "What were total current liabilities for Apple in 2025?", "expected_answer": "$165,631 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["165,631", "current liabilities"], "type": "single_fact"},
    {"question": "What was Apple's commercial paper balance in 2025?", "expected_answer": "$7,979 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["7,979", "commercial paper"], "type": "single_fact"},
    {"question": "What was Apple's total term debt principal in 2025?", "expected_answer": "$91,281 million", "expected_year": 2025, "expected_section": "Item 8", "keywords": ["91,281", "term debt"], "type": "single_fact"},
]

NEW_20_QUESTIONS = [
    # Multi-Year comparisons (6 questions)
    {
        "question": "How did total debt and foreign exchange risk change from 2022 to 2025?",
        "expected_answer": "Total debt decreased from $108,941M to $86,307M while foreign exchange volatility continued affecting global product pricing and gross margins.",
        "expected_year": 2025,
        "expected_section": "Item 8",
        "keywords": ["108,941", "86,307", "foreign exchange"],
        "type": "multi_year",
    },
    {
        "question": "Compare Apple's net sales growth between 2021 and 2024.",
        "expected_answer": "Revenue grew +6.89% from $365,817 million in FY2021 to $391,035 million in FY2024.",
        "expected_year": 2024,
        "expected_section": "Item 8",
        "keywords": ["365,817", "391,035", "net sales"],
        "type": "multi_year",
    },
    {
        "question": "How did operating cash flow and cash reserves evolve between 2021 and 2025?",
        "expected_answer": "Operating cash flow moved from $104,038M to $111,482M and cash equivalents remained stable from $34,940M to $35,934M.",
        "expected_year": 2025,
        "expected_section": "Item 8",
        "keywords": ["104,038", "111,482", "34,940", "35,934"],
        "type": "multi_year",
    },
    {
        "question": "Did Apple increase or decrease its commercial paper borrowings between 2023 and 2025?",
        "expected_answer": "Commercial paper decreased from $9,967M (or $5,985M in 2023) to $7,979M in 2025.",
        "expected_year": 2025,
        "expected_section": "Item 8",
        "keywords": ["commercial paper", "7,979"],
        "type": "multi_year",
    },
    {
        "question": "Compare Apple's debt-to-equity ratio between 2023 and 2025.",
        "expected_answer": "Debt-to-equity improved (deleveraged) from 1.63x in 2023 to 1.17x in 2025 as total debt fell to $86,307M and equity grew to $73,734M.",
        "expected_year": 2025,
        "expected_section": "Item 8",
        "keywords": ["debt-to-equity", "86,307", "73,734"],
        "type": "multi_year",
    },
    {
        "question": "How did operating income change between 2022 and 2024?",
        "expected_answer": "Operating income rose +3.16% from $119,437 million in 2022 to $123,216 million in 2024.",
        "expected_year": 2024,
        "expected_section": "Item 8",
        "keywords": ["119,437", "123,216", "operating income"],
        "type": "multi_year",
    },

    # Ambiguous line items (5 questions)
    {
        "question": "What was Apple's total income in 2023?",
        "expected_answer": "Total income is disambiguated as Operating Income of $114,301 million or Net Income of $96,995 million.",
        "expected_year": 2023,
        "expected_section": "Item 8",
        "keywords": ["114,301", "96,995", "operating income", "net income"],
        "type": "ambiguous_line_item",
    },
    {
        "question": "What was Apple's total income in 2024?",
        "expected_answer": "Total income is disambiguated as Operating Income of $123,216 million or Net Income of $93,736 million.",
        "expected_year": 2024,
        "expected_section": "Item 8",
        "keywords": ["123,216", "93,736", "operating income", "net income"],
        "type": "ambiguous_line_item",
    },
    {
        "question": "What was Apple's total income in 2022?",
        "expected_answer": "Total income is disambiguated as Operating Income of $119,437 million or Net Income of $99,803 million.",
        "expected_year": 2022,
        "expected_section": "Item 8",
        "keywords": ["119,437", "99,803", "operating income", "net income"],
        "type": "ambiguous_line_item",
    },
    {
        "question": "What was Apple's total provision for taxes in 2023?",
        "expected_answer": "The provision for income taxes was $16,741 million in fiscal 2023.",
        "expected_year": 2023,
        "expected_section": "Item 8",
        "keywords": ["16,741", "taxes", "provision"],
        "type": "ambiguous_line_item",
    },
    {
        "question": "What was Apple's total debt in 2023?",
        "expected_answer": "Apple reported total debt of $101,266 million in 2023, consisting of term debt and commercial paper.",
        "expected_year": 2023,
        "expected_section": "Item 8",
        "keywords": ["101,266", "term debt", "commercial paper"],
        "type": "ambiguous_line_item",
    },

    # Ratio and Risk combinations (5 questions)
    {
        "question": "How does Apple's current ratio in 2024 relate to its disclosed liquidity risks?",
        "expected_answer": "Current ratio was 0.87, which reflects negative working capital common in Apple's business model supported by strong cash flows despite liquidity risks.",
        "expected_year": 2024,
        "expected_section": "Item 8",
        "keywords": ["current ratio", "0.87", "liquidity"],
        "type": "ratio_and_risk",
    },
    {
        "question": "What was Apple's gross margin in 2024 and what supply chain cost risks were highlighted?",
        "expected_answer": "Gross margin was 46.21% while supply chain disclosures highlighted single-source component costs and outsourced manufacturing.",
        "expected_year": 2024,
        "expected_section": "Item 8",
        "keywords": ["46.21%", "gross margin", "supply chain"],
        "type": "ratio_and_risk",
    },
    {
        "question": "What was Apple's debt-to-equity ratio in 2024 and what credit rating risks are noted?",
        "expected_answer": "Debt-to-equity was 1.68x ($95,717M debt on $56,950M equity) with credit risks focusing on interest rates and debt refinancing access.",
        "expected_year": 2024,
        "expected_section": "Item 8",
        "keywords": ["1.68x", "debt-to-equity", "credit"],
        "type": "ratio_and_risk",
    },
    {
        "question": "How did operating margin in 2023 compare with regulatory risks under the Digital Markets Act?",
        "expected_answer": "Operating margin was 29.82% ($114,301M on $383,285M) as European DMA enforcement threatened App Store commission fees.",
        "expected_year": 2023,
        "expected_section": "Item 8",
        "keywords": ["operating margin", "digital markets act", "regulatory"],
        "type": "ratio_and_risk",
    },
    {
        "question": "What was Apple's interest coverage in 2024 and does Apple face interest rate risk on debt?",
        "expected_answer": "Interest coverage was robust at 31.9x ($123,216M operating income vs interest expense), mitigating floating interest rate risks.",
        "expected_year": 2024,
        "expected_section": "Item 8",
        "keywords": ["31.9x", "interest coverage", "interest rate"],
        "type": "ratio_and_risk",
    },

    # Chart and Visual questions (4 questions)
    {
        "question": "According to the stock performance graph in Item 5, how did Apple's 5-year cumulative total return compare to the S&P 500?",
        "expected_answer": "The Item 5 stock performance graph shows Apple substantially outperforming the S&P 500 Index over the 5-year period starting with $100 invested.",
        "expected_year": 2024,
        "expected_section": "Item 5",
        "keywords": ["cumulative total return", "S&P 500", "performance graph"],
        "type": "chart",
    },
    {
        "question": "What cumulative total return does the 5-year stock performance chart show for a $100 investment in Apple in the 2024 10-K?",
        "expected_answer": "The chart illustrates cumulative total returns for a $100 initial investment in Apple common stock compared against standard benchmark indices.",
        "expected_year": 2024,
        "expected_section": "Item 5",
        "keywords": ["100", "cumulative total return", "graph"],
        "type": "chart",
    },
    {
        "question": "What does the 5-year stock performance chart in the 2025 10-K show for Apple vs the S&P 500 Index?",
        "expected_answer": "The graph depicts Apple's 5-year total return trend outperforming the broad S&P 500 Index and tracking closely with tech sector peers.",
        "expected_year": 2025,
        "expected_section": "Item 5",
        "keywords": ["graph", "S&P 500", "return"],
        "type": "chart",
    },
    {
        "question": "What visual chart is presented in Item 5 of the 10-K filing?",
        "expected_answer": "Item 5 includes the Company Stock Performance Graph showing 5-year cumulative total returns of Apple common stock against benchmark indices.",
        "expected_year": 2024,
        "expected_section": "Item 5",
        "keywords": ["graph", "Item 5", "stock performance"],
        "type": "chart",
    },
]


def update_testset():
    all_questions = BASE_TESTSET + NEW_20_QUESTIONS
    out_path = Path(__file__).parent / "testset.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for q in all_questions:
            f.write(json.dumps(q) + "\n")
    print(f"Updated {out_path} with {len(all_questions)} questions (including 20 new multi-step questions).")


if __name__ == "__main__":
    update_testset()
