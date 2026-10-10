"""
tests/test_api_agent_query.py — Test FastAPI /api/agent_query Endpoint
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_agent_query_fast_mode():
    payload = {
        "question": "What was Apple's total debt in 2024?",
        "year": 2024,
        "mode": "fast",
    }
    response = client.post("/api/agent_query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "fast"
    assert "question" in data
    assert "answer" in data
    assert "trace" in data
    assert len(data["trace"]) == 1
    assert data["trace"][0]["action"] == "route_and_execute"


def test_agent_query_validation_error():
    payload = {
        "question": "   ",
        "mode": "agent",
    }
    response = client.post("/api/agent_query", json=payload)
    assert response.status_code == 400


def test_agent_query_agent_mode_structure():
    # Use greeting or simple query to verify response schema
    payload = {
        "question": "hello",
        "mode": "agent",
    }
    response = client.post("/api/agent_query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] in ("agent", "agent_fallback")
    assert "answer" in data
    assert "confidence" in data
    assert "citations" in data
    assert "steps_used" in data
    assert "trace" in data
