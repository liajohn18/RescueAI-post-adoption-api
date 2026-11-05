"""Extra endpoints: adopter leaves contact details; volunteers list tickets."""

from __future__ import annotations

import re

from fastapi import APIRouter, Header, HTTPException

from .config import get_settings
from .models import ContactDetails
from .service import get_service

router = APIRouter(prefix="/api/escalations", tags=["escalation"])

_TICKET_RE = re.compile(r"^ESC-\d{8}-[0-9A-F]{4}$")


@router.post("/{ticket_id}/contact")
def add_contact(ticket_id: str, contact: ContactDetails):
    if not _TICKET_RE.match(ticket_id):
        raise HTTPException(status_code=404, detail="Unknown ticket")
    ticket = get_service().add_contact(ticket_id, contact)
    if not ticket:
        raise HTTPException(status_code=404, detail="Unknown ticket")
    return {"ok": True, "ticket_id": ticket_id}


@router.get("")
def list_escalations(x_admin_token: str | None = Header(default=None)):
    expected = get_settings().admin_token
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"escalations": get_service().store.list()}
