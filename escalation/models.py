from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """How fast a volunteer must react."""

    NONE = "none"            # routine question, the guide covers it
    URGENT = "urgent"        # a volunteer should contact the adopter today
    EMERGENCY = "emergency"  # danger now: send to vet / start search immediately


class Category(str, Enum):
    NONE = "none"
    MEDICAL = "medical"                  # illness symptoms that are serious or persist
    INJURY_ACCIDENT = "injury_accident"  # hit by car, wound, fall, poisoning
    BEHAVIOUR = "behaviour"              # bite, serious aggression, dog fights, severe panic
    MISSING_DOG = "missing_dog"          # escaped / lost dog
    ADOPTION_CRISIS = "adoption_crisis"  # adopter wants to return the dog, cannot cope
    OTHER = "other"


class EscalationDecision(BaseModel):
    """What the classifier LLM returns."""

    escalate: bool
    severity: Severity
    category: Category
    reason: str = Field(description="One short sentence: why (not) escalate")
    language: str = Field(default="en", description="'nl' or 'en'")


class EscalationResult(BaseModel):
    """What the API adds to the /api/ask response under the key 'escalation'."""

    escalated: bool
    severity: Severity = Severity.NONE
    category: Category = Category.NONE
    reason: str = ""
    language: Optional[str] = None  # "nl" / "en", detected from the question
    ticket_id: Optional[str] = None
    volunteer_notified: bool = False
    check_failed: bool = False  # classifier unreachable: frontend shows contacts anyway


class ContactDetails(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    phone: str = Field(min_length=5, max_length=40)
    note: Optional[str] = Field(default=None, max_length=1000)
