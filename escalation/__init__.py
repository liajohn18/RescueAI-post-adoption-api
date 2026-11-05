"""Escalation agent for the RescueAI post-adoption FAQ chatbot.

The FAQ bot (RAG over the "Richtlijnen eerste weken" guide) answers routine
questions. This package decides, for every question, whether a human
volunteer from Dogs on a Journey must step in (illness, accident, injury,
bite or aggression, missing dog, adopter in crisis). If so it creates a
ticket and alerts the volunteers on Telegram.

Public entry point: ``assess_and_escalate(question)``.
"""

from .models import Category, EscalationDecision, EscalationResult, Severity
from .service import assess_and_escalate

__all__ = [
    "assess_and_escalate",
    "Category",
    "EscalationDecision",
    "EscalationResult",
    "Severity",
]
