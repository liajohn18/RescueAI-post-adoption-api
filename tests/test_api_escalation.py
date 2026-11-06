"""/api/ask wiring test. Stubs the RAG stack so it runs without models or keys."""

import sys
import types

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient


@pytest.fixture
def api(monkeypatch, tmp_path):
    # Stub heavy RAG imports (sentence-transformers, corpus) and the OpenAI client.
    answer_prompt = types.ModuleType("post_adoption_rag.answer_prompt")
    answer_prompt.build_messages = lambda q, p: [{"content": ""}, {"content": q}]
    answer_prompt.detect_language = lambda text: "English"
    answer_prompt.has_foreign_language_markers = lambda *a: False
    hybrid = types.ModuleType("post_adoption_rag.hybrid_search")
    hybrid.HybridSectionSearch = lambda path: object()
    pkg = types.ModuleType("post_adoption_rag")
    for name, mod in {
        "post_adoption_rag": pkg,
        "post_adoption_rag.answer_prompt": answer_prompt,
        "post_adoption_rag.hybrid_search": hybrid,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    import openai

    monkeypatch.setattr(openai, "OpenAI", lambda *a, **k: object())
    monkeypatch.delitem(sys.modules, "api", raising=False)
    import api as api_module

    return api_module


def test_ask_adds_escalation_field(api, monkeypatch):
    monkeypatch.setattr(api, "answer_question", lambda body: {"answer": "guide answer", "sources": []})
    monkeypatch.setattr(api, "assess_and_escalate", lambda q: {"escalated": True, "severity": "emergency", "ticket_id": "ESC-20261106-ABCD"})
    r = TestClient(api.app).post("/api/ask", json={"question": "My dog was hit by a car"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "guide answer"
    assert body["escalation"]["severity"] == "emergency"


def test_emergency_survives_rag_failure(api, monkeypatch):
    def broken(body):
        raise HTTPException(status_code=503, detail="down")

    monkeypatch.setattr(api, "answer_question", broken)
    monkeypatch.setattr(api, "assess_and_escalate", lambda q: {"escalated": True, "severity": "emergency"})
    r = TestClient(api.app).post("/api/ask", json={"question": "My dog is bleeding"})
    assert r.status_code == 200
    assert r.json()["escalation"]["escalated"] is True


def test_rag_failure_without_escalation_still_errors(api, monkeypatch):
    def broken(body):
        raise HTTPException(status_code=503, detail="down")

    monkeypatch.setattr(api, "answer_question", broken)
    monkeypatch.setattr(api, "assess_and_escalate", lambda q: {"escalated": False})
    r = TestClient(api.app).post("/api/ask", json={"question": "How often to feed?"})
    assert r.status_code == 503


def test_contact_route_is_mounted(api):
    paths = TestClient(api.app).get("/openapi.json").json()["paths"]
    assert "/api/escalations/{ticket_id}/contact" in paths
