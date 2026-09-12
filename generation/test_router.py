"""
generation/test_router.py — CLI test script for Classification + Routing
=========================================================================

Usage:
  python generation/test_router.py "What was Apple's total debt in 2024?"
  python generation/test_router.py "Calculate gross margin for fiscal year 2024"
"""

import argparse
import json
import sys

from generation.router import route_and_execute

def main():
    parser = argparse.ArgumentParser(description="Test query classification and routing.")
    parser.add_argument("question", type=str, help="Financial question")
    parser.add_argument("--year", type=int, default=None, help="Optional year filter")

    args = parser.parse_args()

    print("\n" + "=" * 75)
    print(f"  QUESTION: {args.question}")
    if args.year:
        print(f"  FILTER: Year={args.year}")
    print("=" * 75)

    res = route_and_execute(question=args.question, year=args.year)

    print(f"\n[ROUTED CATEGORY]: {res.get('category')}")
    print(f"[ANSWER]: {res.get('answer')}\n")
    if res.get("citations"):
        print("[CITATIONS]:")
        for c in res["citations"]:
            print(f"  - Year {c.get('year')}, {c.get('section')} (Pos: {c.get('position_id')})")
    if res.get("confidence") is not None:
        print(f"\n[CONFIDENCE]: {res.get('confidence'):.2f}")
    if res.get("abstain_reason"):
        print(f"[ABSTAIN REASON]: {res.get('abstain_reason')}")
    print("=" * 75)

if __name__ == "__main__":
    main()
