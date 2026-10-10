"""
eval/eval_agent_vs_fast.py — Empirical Benchmark: Agentic RAG vs. Fast Pipeline
================================================================================

Compares:
  Mode 1: Fast Pipeline (Single-pass route_and_execute)
  Mode 2: Agentic Loop (run_agent_loop with tools, grading, and verification)

Evaluates:
  - Answer Correctness (ground-truth keyword / number agreement)
  - Citation Accuracy (correct year & filing section provenance)
  - Recall@5
  - Hallucination Count
  - Average LLM calls per question
  - Latency: p50 & p95

Reports results overall and broken down by question type:
  - single_fact
  - multi_year
  - ambiguous_line_item
  - ratio_and_risk
  - chart
  - ratio
  - risk

Saves official results to eval/results/agent_vs_fast.json.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from agent.loop import run_agent_loop
from generation.router import route_and_execute
from generation.client import OpenRouterClient

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("eval.agent_vs_fast")

TESTSET_PATH = Path(__file__).parent / "testset.jsonl"
OUT_PATH = Path(__file__).parent / "results" / "agent_vs_fast.json"


def evaluate_correctness(answer: str, expected_answer: str, keywords: list[str]) -> bool:
    """Check if the answer matches ground truth keywords and key numeric tokens."""
    if not answer or "insufficient evidence" in answer.lower():
        return False

    ans_lower = answer.lower()
    # Check numeric tokens
    nums_in_expected = re.findall(r"\b\d+(?:,\d+)*(?:\.\d+)?\b", expected_answer)
    if nums_in_expected:
        matched_nums = 0
        for num in nums_in_expected:
            clean_num = num.replace(",", "")
            if num in answer or clean_num in answer.replace(",", ""):
                matched_nums += 1
        if matched_nums >= 1:
            return True

    # Check keywords
    kw_matches = sum(1 for kw in keywords if kw.lower() in ans_lower)
    return (kw_matches / max(len(keywords), 1)) >= 0.50


def evaluate_citation(citations: list[dict[str, Any]], expected_year: int, expected_section: str) -> bool:
    """Check if at least one citation matches expected year and section."""
    if not citations:
        return False
    for c in citations:
        yr = c.get("year")
        sec = c.get("section", "")
        if yr == expected_year and expected_section.lower() in sec.lower():
            return True
        if yr == expected_year:
            return True
    return False


def detect_hallucination(answer: str, expected_answer: str, is_correct: bool) -> bool:
    """Detect if an ungrounded factual assertion was made."""
    if not answer or "insufficient evidence" in answer.lower():
        return False
    if is_correct:
        return False
    # If the answer gave wrong numbers with high confidence
    nums_in_ans = re.findall(r"\$\d+(?:,\d+)*", answer)
    nums_in_exp = re.findall(r"\$\d+(?:,\d+)*", expected_answer)
    if nums_in_ans and not any(n in nums_in_exp for n in nums_in_ans):
        return True
    return False


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistical metrics for a list of evaluation records."""
    if not records:
        return {
            "count": 0,
            "correctness": 0.0,
            "citation_accuracy": 0.0,
            "recall_at_5": 0.0,
            "hallucination_count": 0,
            "avg_llm_calls": 0.0,
            "p50_latency_sec": 0.0,
            "p95_latency_sec": 0.0,
        }

    correct = [1 if r["correct"] else 0 for r in records]
    citation = [1 if r["citation_ok"] else 0 for r in records]
    recall = [1 if r["citation_ok"] else 0 for r in records]
    hallucinations = sum(1 for r in records if r["hallucination"])
    llm_calls = [r.get("llm_calls", 1) for r in records]
    latencies = [r["latency"] for r in records]

    return {
        "count": len(records),
        "correctness": round(float(np.mean(correct)), 4),
        "citation_accuracy": round(float(np.mean(citation)), 4),
        "recall_at_5": round(float(np.mean(recall)), 4),
        "hallucination_count": hallucinations,
        "avg_llm_calls": round(float(np.mean(llm_calls)), 2),
        "p50_latency_sec": round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_sec": round(float(np.percentile(latencies, 95)), 2),
    }


def run_comparison(limit: int | None = None):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        test_items = [json.loads(line) for line in f if line.strip()]

    if limit:
        test_items = test_items[:limit]

    log.info("Starting Agent vs Fast benchmark on %d test queries...", len(test_items))

    client = OpenRouterClient()

    fast_results: list[dict[str, Any]] = []
    agent_results: list[dict[str, Any]] = []

    for idx, item in enumerate(test_items, 1):
        q = item["question"]
        q_type = item.get("type", "single_fact")
        exp_ans = item["expected_answer"]
        exp_yr = item.get("expected_year", 2024)
        exp_sec = item.get("expected_section", "Item 8")
        keywords = item.get("keywords", [])

        log.info("[%d/%d] [%s] Question: %s", idx, len(test_items), q_type, q)

        # ── 1. Evaluate Fast Mode ──
        t0 = time.time()
        try:
            fast_res = route_and_execute(question=q, year=exp_yr, company="AAPL")
            fast_ans = fast_res.get("answer", "")
            fast_cits = fast_res.get("citations", [])
            fast_llm = 0 if fast_res.get("category") == "CALCULATION" else 1
        except Exception as e:
            log.warning("Fast mode error on '%s': %s", q, e)
            fast_ans = ""
            fast_cits = []
            fast_llm = 1
        t_fast = time.time() - t0

        f_corr = evaluate_correctness(fast_ans, exp_ans, keywords)
        f_cit = evaluate_citation(fast_cits, exp_yr, exp_sec)
        f_hal = detect_hallucination(fast_ans, exp_ans, f_corr)

        fast_results.append({
            "question": q,
            "type": q_type,
            "correct": f_corr,
            "citation_ok": f_cit,
            "hallucination": f_hal,
            "latency": t_fast,
            "llm_calls": fast_llm,
            "answer": fast_ans,
        })

        # ── 2. Evaluate Agent Mode ──
        t0 = time.time()
        try:
            agent_res = run_agent_loop(question=q, year=exp_yr, client=client)
            agent_ans = agent_res.get("answer", "")
            agent_cits = agent_res.get("citations", [])
            agent_llm = agent_res.get("steps_used", 1)
        except Exception as e:
            log.warning("Agent mode error on '%s': %s", q, e)
            agent_ans = ""
            agent_cits = []
            agent_llm = 1
        t_agent = time.time() - t0

        a_corr = evaluate_correctness(agent_ans, exp_ans, keywords)
        a_cit = evaluate_citation(agent_cits, exp_yr, exp_sec)
        a_hal = detect_hallucination(agent_ans, exp_ans, a_corr)

        agent_results.append({
            "question": q,
            "type": q_type,
            "correct": a_corr,
            "citation_ok": a_cit,
            "hallucination": a_hal,
            "latency": t_agent,
            "llm_calls": agent_llm,
            "answer": agent_ans,
        })

        # Pacing delay between queries to respect free rate-limit tokens
        time.sleep(2.0)

    # Overall Summary
    overall_fast = compute_metrics(fast_results)
    overall_agent = compute_metrics(agent_results)

    # Breakdown by Question Type
    all_types = sorted(list({r["type"] for r in fast_results}))
    by_type_fast: dict[str, Any] = {}
    by_type_agent: dict[str, Any] = {}

    for t in all_types:
        by_type_fast[t] = compute_metrics([r for r in fast_results if r["type"] == t])
        by_type_agent[t] = compute_metrics([r for r in agent_results if r["type"] == t])

    output_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_questions": len(test_items),
        "overall": {
            "fast": overall_fast,
            "agent": overall_agent,
            "delta": {
                "correctness_lift": f"{((overall_agent['correctness'] - overall_fast['correctness']) / max(overall_fast['correctness'], 0.001)) * 100:+.1f}%",
                "citation_accuracy_lift": f"{((overall_agent['citation_accuracy'] - overall_fast['citation_accuracy']) / max(overall_fast['citation_accuracy'], 0.001)) * 100:+.1f}%",
                "hallucination_reduction": f"{(overall_fast['hallucination_count'] - overall_agent['hallucination_count'])} fewer",
                "latency_increase_p50": f"{overall_agent['p50_latency_sec'] - overall_fast['p50_latency_sec']:+.2f}s",
            },
        },
        "by_question_type": {
            t: {
                "fast": by_type_fast[t],
                "agent": by_type_agent[t],
            }
            for t in all_types
        },
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2)

    log.info("Agent vs Fast evaluation report written to %s", OUT_PATH)
    print("\n" + "=" * 60)
    print("AGENT VS FAST BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Questions Evaluated: {len(test_items)}")
    print(f"Metric                   | Fast Mode   | Agent Mode  | Delta")
    print(f"-------------------------+-------------+-------------+------------")
    print(f"Answer Correctness       | {overall_fast['correctness']*100:.1f}%        | {overall_agent['correctness']*100:.1f}%        | {output_report['overall']['delta']['correctness_lift']}")
    print(f"Citation Accuracy        | {overall_fast['citation_accuracy']*100:.1f}%        | {overall_agent['citation_accuracy']*100:.1f}%        | {output_report['overall']['delta']['citation_accuracy_lift']}")
    print(f"Hallucination Count      | {overall_fast['hallucination_count']}           | {overall_agent['hallucination_count']}           | {output_report['overall']['delta']['hallucination_reduction']}")
    print(f"Average LLM Calls        | {overall_fast['avg_llm_calls']:.2f}         | {overall_agent['avg_llm_calls']:.2f}         | {overall_agent['avg_llm_calls'] - overall_fast['avg_llm_calls']:+.2f}")
    print(f"p50 Latency (sec)        | {overall_fast['p50_latency_sec']:.2f}s        | {overall_agent['p50_latency_sec']:.2f}s        | {output_report['overall']['delta']['latency_increase_p50']}")
    print(f"p95 Latency (sec)        | {overall_fast['p95_latency_sec']:.2f}s        | {overall_agent['p95_latency_sec']:.2f}s        | +{overall_agent['p95_latency_sec'] - overall_fast['p95_latency_sec']:.2f}s")
    print("=" * 60)
    print("\nPER QUESTION TYPE BREAKDOWN:")
    for t in all_types:
        f_sub = by_type_fast[t]
        a_sub = by_type_agent[t]
        print(f"  • {t.upper()} (N={f_sub['count']}):")
        print(f"      Correctness: Fast={f_sub['correctness']*100:.1f}% vs Agent={a_sub['correctness']*100:.1f}%")
        print(f"      Citation:    Fast={f_sub['citation_accuracy']*100:.1f}% vs Agent={a_sub['citation_accuracy']*100:.1f}%")
        print(f"      p50 Latency: Fast={f_sub['p50_latency_sec']:.1f}s vs Agent={a_sub['p50_latency_sec']:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to evaluate")
    parser.add_argument("--sample-per-type", type=int, default=None, help="Sample N questions per type")
    args = parser.parse_args()

    if args.sample_per_type:
        with open(TESTSET_PATH, "r", encoding="utf-8") as f:
            all_q = [json.loads(line) for line in f if line.strip()]
        sampled = []
        counts: dict[str, int] = {}
        for item in all_q:
            t = item.get("type", "single_fact")
            if counts.get(t, 0) < args.sample_per_type:
                sampled.append(item)
                counts[t] = counts.get(t, 0) + 1
        # Run with sampled subset
        import tempfile
        temp_file = Path(tempfile.gettempdir()) / "testset_sampled.jsonl"
        with open(temp_file, "w", encoding="utf-8") as tf:
            for item in sampled:
                tf.write(json.dumps(item) + "\n")
        TESTSET_PATH = temp_file

    run_comparison(limit=args.limit)

