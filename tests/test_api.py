"""
tests/test_api.py — Verify FastAPI backend endpoints
====================================================
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["database"] == "connected"


def test_financials_endpoint():
    res = client.get("/api/financials")
    assert res.status_code == 200
    data = res.json()
    assert "data" in data
    assert "years" in data
    assert 2025 in data["years"] or "2025" in data["data"]


def test_risks_endpoint():
    res = client.get("/api/risks")
    assert res.status_code == 200
    data = res.json()
    assert "scores" in data
    assert "top_risks" in data


def test_eval_endpoint():
    res = client.get("/api/eval")
    assert res.status_code == 200


def test_query_calculation_endpoint():
    res = client.post("/api/query", json={"question": "What was Apple's total debt in 2024?", "year": 2024})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert len(data["answer"]) > 10
    assert data["confidence"] > 0.0
