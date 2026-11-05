"""Glue: classify -> ticket -> alert. One call per adopter question."""

from __future__ import annotations

import logging
from functools import lru_cache

from .classifier import Classifier, LLMClassifier
from .config import Settings, get_settings
from .models import ContactDetails, EscalationResult
from .notifier import Notifier, TelegramNotifier, format_contact_update, format_new_ticket
from .store import TicketStore

log = logging.getLogger(__name__)


class EscalationService:
    def __init__(self, classifier: Classifier, store: TicketStore, notifier: Notifier, enabled: bool = True):
        self.classifier = classifier
        self.store = store
        self.notifier = notifier
        self.enabled = enabled

    def assess(self, question: str) -> EscalationResult:
        if not self.enabled or not question.strip():
            return EscalationResult(escalated=False)

        try:
            decision = self.classifier.classify(question)
        except Exception as exc:  # LLM down, bad JSON, timeout ...
            log.exception("Escalation classifier failed: %s", exc)
            # Fail visible, not silent: no ticket, but the UI shows volunteer contacts.
            return EscalationResult(escalated=False, check_failed=True, reason="classifier_unavailable")

        if not decision.escalate:
            return EscalationResult(escalated=False, reason=decision.reason)

        ticket_id = self.store.create(question, decision)
        ticket = self.store.get(ticket_id)
        notified = self.notifier.send(format_new_ticket(ticket))
        self.store.mark_notified(ticket_id, notified)
        log.warning("Escalated %s (%s/%s)", ticket_id, decision.severity.value, decision.category.value)

        return EscalationResult(
            escalated=True,
            severity=decision.severity,
            category=decision.category,
            reason=decision.reason,
            language=decision.language,
            ticket_id=ticket_id,
            volunteer_notified=notified,
        )

    def add_contact(self, ticket_id: str, contact: ContactDetails) -> dict | None:
        ticket = self.store.add_contact(ticket_id, contact)
        if ticket:
            self.notifier.send(format_contact_update(ticket))
        return ticket


@lru_cache(maxsize=1)
def get_service() -> EscalationService:
    s: Settings = get_settings()
    return EscalationService(
        classifier=LLMClassifier(s),
        store=TicketStore(s.db_path),
        notifier=TelegramNotifier(s),
        enabled=s.enabled,
    )


def assess_and_escalate(question: str) -> dict:
    """Call this inside /api/ask and put the result under response['escalation']."""
    return get_service().assess(question).model_dump(mode="json")
