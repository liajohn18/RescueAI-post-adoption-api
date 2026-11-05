"""Settings come from environment variables (set them on Render)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # LLM used by the classifier (OpenAI Responses API, same as api.py).
    llm_api_key: str | None
    llm_base_url: str | None
    llm_model: str
    llm_timeout_s: float

    # Telegram alert to the volunteer group. If unset, alerts are only logged.
    telegram_bot_token: str | None
    telegram_chat_id: str | None

    # Ticket store (SQLite). On Render's free tier the disk resets on redeploy.
    db_path: str

    # Protects GET /api/escalations (volunteer view).
    admin_token: str | None

    # Turn the whole feature off without code changes.
    enabled: bool


def get_settings() -> Settings:
    return Settings(
        llm_api_key=os.getenv("ESCALATION_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        llm_base_url=os.getenv("ESCALATION_LLM_BASE_URL") or None,
        llm_model=os.getenv("ESCALATION_LLM_MODEL", "gpt-6-luna"),
        llm_timeout_s=float(os.getenv("ESCALATION_LLM_TIMEOUT_S", "15")),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN") or None,
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID") or None,
        db_path=os.getenv("ESCALATION_DB_PATH", "escalations.db"),
        admin_token=os.getenv("ESCALATION_ADMIN_TOKEN") or None,
        enabled=os.getenv("ESCALATION_ENABLED", "true").lower() not in {"0", "false", "no"},
    )
