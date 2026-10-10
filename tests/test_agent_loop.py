"""
tests/test_agent_loop.py — Unit and Integration Tests for Agent Loop
===================================================================

Tests:
  1. Handled step loop with simulated LLM responses (mock client).
  2. Caching identical tool calls without repeating execution.
  3. JSON malformed recovery and fallback triggering.
  4. Max steps constraint enforcement.
  5. Integration with claim verification on action="final".
"""

import json
import pytest
from unittest.mock import MagicMock

from agent.loop import run_agent_loop, MAX_STEPS
from agent.verify import ABSTAIN_MESSAGE


class MockLLMClient:
    """Mock client returning sequential pre-programmed responses."""
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.call_count = 0
        self.received_messages = []

    def chat(self, messages, temperature=0.0, max_tokens=1000):
        self.call_count += 1
        self.received_messages.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return '{"action": "final", "args": {"draft_answer": "No further steps."}}'


def test_agent_loop_basic_execution():
    # Sequence: step 1 calls get_metric, step 2 calls final
    responses = [
        json.dumps({
            "thought": "I need to check FY2024 total debt using get_metric.",
            "action": "get_metric",
            "args": {"metric": "total_debt", "year": 2024},
        }),
        json.dumps({
            "thought": "Total debt for FY2024 is $95,717 million. I can now provide the final answer.",
            "action": "final",
            "args": {"draft_answer": "In FY2024, Apple's total debt was $95,717 million."},
        }),
    ]
    mock_client = MockLLMClient(responses)
    res = run_agent_loop(
        question="What was Apple's total debt in 2024?",
        year=2024,
        client=mock_client,
    )

    assert res["mode"] == "agent"
    assert res["steps_used"] == 2
    assert "95,717" in res["answer"]
    assert res["confidence"] > 0.0
    assert len(res["trace"]) == 2
    assert res["trace"][0]["action"] == "get_metric"
    assert res["trace"][1]["action"] == "final"
    assert len(res["citations"]) >= 1


def test_agent_loop_caches_identical_tool_call():
    # Model mistakenly calls identical tool twice
    responses = [
        json.dumps({
            "thought": "Fetch revenue 2024.",
            "action": "get_metric",
            "args": {"metric": "revenue", "year": 2024},
        }),
        json.dumps({
            "thought": "Fetch revenue 2024 again.",
            "action": "get_metric",
            "args": {"metric": "revenue", "year": 2024},
        }),
        json.dumps({
            "thought": "Answer finalized.",
            "action": "final",
            "args": {"draft_answer": "Apple's FY2024 revenue was $391,035 million."},
        }),
    ]
    mock_client = MockLLMClient(responses)
    res = run_agent_loop(
        question="What was revenue in 2024?",
        year=2024,
        client=mock_client,
    )

    assert res["steps_used"] == 3
    # Step 2 must be served from cache
    assert "Cached result" in res["trace"][1]["observation"]
    assert "391,035" in res["answer"]


def test_agent_loop_recovers_from_single_malformed_json():
    # First response is malformed, retry response is valid JSON
    responses = [
        "Here is the plan: {\"action\": invalid json",
        json.dumps({
            "thought": "Recovered valid JSON.",
            "action": "get_metric",
            "args": {"metric": "net_income", "year": 2024},
        }),
        json.dumps({
            "thought": "Ready.",
            "action": "final",
            "args": {"draft_answer": "Net income in 2024 was $93,736 million."},
        }),
    ]
    mock_client = MockLLMClient(responses)
    res = run_agent_loop(
        question="Net income 2024?",
        year=2024,
        client=mock_client,
    )

    assert res["mode"] == "agent"
    assert "93,736" in res["answer"]


def test_agent_loop_max_steps_budget():
    # Client keeps calling tools indefinitely without finalizing
    responses = [
        json.dumps({
            "thought": f"Calling step {i}",
            "action": "get_metric",
            "args": {"metric": "revenue", "year": 2021 + (i % 5)},
        })
        for i in range(10)
    ]
    mock_client = MockLLMClient(responses)
    res = run_agent_loop(
        question="Keep searching endlessly",
        max_steps=4,
        client=mock_client,
    )

    # Must terminate at max_steps=4
    assert res["steps_used"] <= 4


def test_agent_loop_verification_prunes_unsupported_claim():
    responses = [
        json.dumps({
            "thought": "Look up revenue.",
            "action": "get_metric",
            "args": {"metric": "revenue", "year": 2025},
        }),
        json.dumps({
            "thought": "Draft answer with one true and one hallucinated claim.",
            "action": "final",
            "args": {
                "draft_answer": "Apple revenue was $416,161 million in 2025. Apple had $999,999 million of gold reserves."
            },
        }),
    ]
    mock_client = MockLLMClient(responses)
    res = run_agent_loop(
        question="Apple 2025 revenue and gold reserves?",
        year=2025,
        client=mock_client,
    )

    assert res["mode"] == "agent"
    assert "$416,161" in res["answer"]
    assert "999,999" not in res["answer"]
    assert len(res["verification"]["removed_claims"]) >= 1
