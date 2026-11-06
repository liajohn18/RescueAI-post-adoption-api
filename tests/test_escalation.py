import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from escalation import service as service_mod
from escalation.classifier import parse_decision
from escalation.models import EscalationDecision
from escalation.router import router
from escalation.service import EscalationService
from escalation.store import TicketStore


class FakeClassifier:
    def __init__(self, decision=None, error=None):
        self.decision, self.error = decision, error

    def classify(self, question):
        if self.error:
            raise self.error
        return self.decision


class FakeNotifier:
    def __init__(self, ok=True):
        self.ok, self.sent = ok, []

    def send(self, text):
        self.sent.append(text)
        return self.ok


def decision(sev, cat, esc=None):
    return EscalationDecision(
        escalate=(sev != "none") if esc is None else esc,
        severity=sev, category=cat, reason="test", language="en",
    )


@pytest.fixture
def store(tmp_path):
    return TicketStore(str(tmp_path / "t.db"))


def test_routine_question_creates_no_ticket(store):
    n = FakeNotifier()
    svc = EscalationService(FakeClassifier(decision("none", "none")), store, n)
    r = svc.assess("How often should I feed my dog?")
    assert r.escalated is False and r.ticket_id is None
    assert n.sent == [] and store.list() == []


def test_emergency_creates_ticket_and_alerts(store):
    n = FakeNotifier()
    svc = EscalationService(FakeClassifier(decision("emergency", "injury_accident")), store, n)
    r = svc.assess("My dog was hit by a car <b>")
    assert r.escalated and r.ticket_id.startswith("ESC-") and r.volunteer_notified
    assert len(n.sent) == 1 and "EMERGENCY" in n.sent[0]
    assert "&lt;b&gt;" in n.sent[0]  # adopter text is HTML-escaped for Telegram
    assert store.get(r.ticket_id)["notified"] == 1


def test_alert_failure_still_keeps_ticket(store):
    svc = EscalationService(FakeClassifier(decision("urgent", "medical")), store, FakeNotifier(ok=False))
    r = svc.assess("Blood in stool for 3 days")
    assert r.escalated and not r.volunteer_notified
    assert store.get(r.ticket_id)["notified"] == 0


def test_classifier_error_fails_visible(store):
    svc = EscalationService(FakeClassifier(error=TimeoutError()), store, FakeNotifier())
    r = svc.assess("anything")
    assert r.escalated is False and r.check_failed is True


def test_disabled_feature_skips_everything(store):
    svc = EscalationService(FakeClassifier(error=AssertionError("must not run")), store, FakeNotifier(), enabled=False)
    assert svc.assess("My dog is bleeding").escalated is False


def test_parse_decision_fixes_inconsistent_flag():
    d = parse_decision(json.dumps({"escalate": False, "severity": "urgent", "category": "behaviour", "reason": "x"}))
    assert d.escalate is True
    d = parse_decision(json.dumps({"escalate": True, "severity": "none", "category": "medical", "reason": "x"}))
    assert d.escalate is False and d.category.value == "none"


def test_parse_decision_rejects_bad_values():
    with pytest.raises(Exception):
        parse_decision('{"escalate": true, "severity": "very bad", "category": "medical", "reason": "x"}')


def test_contact_endpoint(store, monkeypatch):
    n = FakeNotifier()
    svc = EscalationService(FakeClassifier(decision("urgent", "behaviour")), store, n)
    monkeypatch.setattr(service_mod, "get_service", lambda: svc)
    monkeypatch.setattr("escalation.router.get_service", lambda: svc)
    app = FastAPI(); app.include_router(router)
    c = TestClient(app)

    tid = svc.assess("My dog bit my son").ticket_id
    r = c.post(f"/api/escalations/{tid}/contact", json={"name": "Sara", "phone": "+31 6 1234 5678"})
    assert r.status_code == 200
    assert store.get(tid)["contact_phone"] == "+31 6 1234 5678"
    assert "Contact added" in n.sent[-1]

    assert c.post("/api/escalations/ESC-20260101-ZZZZ/contact", json={"name": "a", "phone": "12345"}).status_code == 404
    assert c.post("/api/escalations/../../x/contact", json={"name": "a", "phone": "12345"}).status_code == 404
    assert c.get("/api/escalations").status_code == 401
