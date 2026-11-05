"""Alert the volunteer group on Telegram."""

from __future__ import annotations

import html
import logging
from typing import Any, Protocol

import httpx

from .config import Settings

log = logging.getLogger(__name__)

_ICON = {"emergency": "🚨", "urgent": "⚠️"}
_CATEGORY_LABEL = {
    "medical": "Medical",
    "injury_accident": "Injury / accident",
    "behaviour": "Behaviour / bite",
    "missing_dog": "Missing dog",
    "adoption_crisis": "Adopter in crisis",
    "other": "Other",
    "none": "-",
}


class Notifier(Protocol):
    def send(self, text: str) -> bool: ...


class TelegramNotifier:
    def __init__(self, settings: Settings):
        self.token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> bool:
        if not self.configured:
            log.info("Telegram not configured; alert only logged:\n%s", text)
            return False
        try:
            r = httpx.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            ok = r.status_code == 200 and r.json().get("ok", False)
            if not ok:
                log.error("Telegram send failed: %s %s", r.status_code, r.text[:300])
            return ok
        except httpx.HTTPError as exc:
            log.error("Telegram send error: %s", exc)
            return False


def format_new_ticket(ticket: dict[str, Any]) -> str:
    sev = ticket["severity"]
    return (
        f"{_ICON.get(sev, '⚠️')} <b>{sev.upper()}</b> · {_CATEGORY_LABEL.get(ticket['category'], ticket['category'])}\n"
        f"<b>Ticket:</b> <code>{ticket['id']}</code>\n"
        f"<b>Adopter wrote</b> ({ticket.get('language') or '?'}):\n"
        f"<i>{html.escape(ticket['question'][:800])}</i>\n\n"
        f"<b>Why:</b> {html.escape(ticket.get('reason') or '')}\n"
        f"Contact details follow if the adopter leaves them. Reply here when you pick this up."
    )


def format_contact_update(ticket: dict[str, Any]) -> str:
    note = ticket.get("contact_note")
    return (
        f"📞 <b>Contact added</b> to <code>{ticket['id']}</code> "
        f"({ticket['severity']}, {_CATEGORY_LABEL.get(ticket['category'], ticket['category'])})\n"
        f"<b>Name:</b> {html.escape(ticket.get('contact_name') or '')}\n"
        f"<b>Phone/WhatsApp:</b> {html.escape(ticket.get('contact_phone') or '')}"
        + (f"\n<b>Note:</b> {html.escape(note)}" if note else "")
    )
