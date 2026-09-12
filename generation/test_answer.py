"""
generation/test_answer.py — CLI test script for end-to-end Retrieval + Generation
==================================================================================

Usage:
  python generation/test_answer.py "What was Apple's total debt in 2024?"
"""

import argparse
import json
import sys

from retrieval.hybrid import retrieve_hybrid
from generation.answer import generate_answer

def main():
    parser = argparse.ArgumentParser(description="Test end-to-end RAG question answering.")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument("--year", type=int, default=None, help="Filter by year")
    parser.add_argument("--threshold", type=float, default=0.30, help="Abstention threshold")

    args = parser.parse_args()

    print("\n" + "=" * 75)
    print(f"  QUESTION: {args.question}")
    if args.year:
        print(f"  FILTER: Year={args.year}")
    print("=" * 75)

    # 1. Retrieval
    print("\n[1] Retrieving and reranking top chunks...")
    chunks = retrieve_hybrid(
        query=args.question,
        year=args.year,
        final_top_k=5,
    )

    if chunks:
        print(f"    Top chunk rerank score: {chunks[0]['rerank_score']:.4f} ({chunks[0]['year']} {chunks[0]['section']})")
    else:
        print("    [!] No chunks retrieved.")

    # 2. Generation
    print("\n[2] Generating cited answer via OpenRouter...")
    result = generate_answer(
        question=args.question,
        retrieved_chunks=chunks,
        abstain_threshold=args.threshold,
    )

    print("\n[3] Final JSON Result:\n")
    print(json.dumps(result, indent=2))
    print("\n" + "=" * 75)

if __name__ == "__main__":
    main()
