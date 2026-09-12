"""
retrieval/test_query.py — CLI tool to test hybrid retrieval + reranking
=======================================================================

Usage:
  python retrieval/test_query.py "What was Apple's total debt in 2024?"
  python retrieval/test_query.py "What are the supply chain risks?" --year 2024
"""

import argparse
import sys
from retrieval.hybrid import retrieve_hybrid

def main():
    parser = argparse.ArgumentParser(description="Test hybrid retrieval + reranking.")
    parser.add_argument("query", type=str, help="Question to search")
    parser.add_argument("--year", type=int, default=None, help="Filter by fiscal year (e.g. 2024)")
    parser.add_argument("--section", type=str, default=None, help="Filter by section title")
    parser.add_argument("--company", type=str, default=None, help="Filter by company ticker (e.g. AAPL)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to display")

    args = parser.parse_args()

    print("\n" + "=" * 75)
    print(f"  QUERY: {args.query}")
    filters = []
    if args.year:
        filters.append(f"Year={args.year}")
    if args.section:
        filters.append(f"Section='{args.section}'")
    if args.company:
        filters.append(f"Company='{args.company}'")
    if filters:
        print(f"  FILTERS: {', '.join(filters)}")
    print("=" * 75)

    results = retrieve_hybrid(
        query=args.query,
        company=args.company,
        year=args.year,
        section=args.section,
        final_top_k=args.top_k,
    )

    if not results:
        print("\n  [!] No matching chunks found.")
        return

    print(f"\nFound {len(results)} top-ranked chunks:\n")
    for idx, r in enumerate(results, 1):
        print(f"[{idx}] Rerank Score: {r['rerank_score']:.4f}")
        print(f"    Company: {r['company']} | Year: {r['year']} | Type: {r['chunk_type']}")
        print(f"    Section: {r['section']} (Pos: {r['position_id']})")
        print("    Content snippet:")
        snippet = r['text'].strip().replace("\n", " ")
        if len(snippet) > 350:
            snippet = snippet[:350] + " ... [truncated]"
        print(f"    \"{snippet}\"")
        print("-" * 75)

if __name__ == "__main__":
    main()
