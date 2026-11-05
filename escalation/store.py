"""Escalation tickets in SQLite (small, no extra service needed)."""

from __future__ import annotations

import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from .models import ContactDetails, EscalationDecision

_SCHEMA = """
CREATE TABLE IF NOT EXISTS escalations (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    question      TEXT NOT NULL,
    language      TEXT,
    severity      TEXT NOT NULL,
    category      TEXT NOT NULL,
    reason        TEXT,
    status        TEXT NOT NULL DEFAULT 'open',
    notified      INTEGER NOT NULL DEFAULT 0,
    contact_name  TEXT,
    contact_phone TEXT,
    contact_note  TEXT,
    updated_at    TEXT
)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TicketStore:
    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def create(self, question: str, decision: EscalationDecision) -> str:
        ticket_id = f"ESC-{datetime.now(timezone.utc):%Y%m%d}-{secrets.token_hex(2).upper()}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO escalations (id, created_at, question, language, severity, "
                "category, reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ticket_id,
                    _now(),
                    question,
                    decision.language,
                    decision.severity.value,
                    decision.category.value,
                    decision.reason,
                ),
            )
            self._conn.commit()
        return ticket_id

    def mark_notified(self, ticket_id: str, notified: bool) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE escalations SET notified = ?, updated_at = ? WHERE id = ?",
                (int(notified), _now(), ticket_id),
            )
            self._conn.commit()

    def add_contact(self, ticket_id: str, contact: ContactDetails) -> dict[str, Any] | None:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE escalations SET contact_name = ?, contact_phone = ?, contact_note = ?, "
                "updated_at = ? WHERE id = ?",
                (contact.name, contact.phone, contact.note, _now(), ticket_id),
            )
            self._conn.commit()
        if cur.rowcount == 0:
            return None
        return self.get(ticket_id)

    def get(self, ticket_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM escalations WHERE id = ?", (ticket_id,)
            ).fetchone()
        return dict(row) if row else None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM escalations ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
