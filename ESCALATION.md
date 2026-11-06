# Human escalation (post-adoption FAQ)

The FAQ bot answers routine questions from the adoption guide. Some messages
need a person: an injured or sick dog, a bite, a missing dog, or an adopter who
cannot cope. For these, the API now escalates to the Dogs on a Journey
volunteers.

## How it works

```
adopter question ──► /api/ask ─┬─► RAG answer from the guide (unchanged)
                               └─► escalation classifier (LLM, runs in parallel)
                                        │ severity: none / urgent / emergency
                                        ▼
                         urgent or emergency?
                           │ yes
                           ├─► ticket in SQLite (ESC-YYYYMMDD-XXXX)
                           ├─► Telegram alert to the volunteer group
                           └─► response.escalation → FAQ page shows an alert card
                                    adopter leaves name + phone
                                    └─► POST /api/escalations/{id}/contact
                                         └─► second Telegram message with contact
```

* **Classifier** – `escalation/classifier.py`. One OpenAI Responses API call with a
  strict JSON schema (same pattern as `api.py`). It returns `escalate`,
  `severity`, `category` (`medical`, `injury_accident`, `behaviour`,
  `missing_dog`, `adoption_crisis`, `other`), a short `reason` and the
  language. The prompt separates real, current situations from "what if"
  questions and tells the model to prefer `urgent` when unsure about health or
  safety, because a missed emergency costs more than an extra volunteer check.
* **Autonomy** – the agent decides on its own *whether* to alert a human. It
  never gives medical advice and never closes a case; volunteers do that.
* **Fail-safe** – if the classifier is down, `escalation.check_failed` is
  `true` and the page shows the foundation's contact advice. If the RAG answer
  fails but the case is urgent, the escalation is still returned.

## Response shape

`POST /api/ask` keeps `answer` and `sources` and adds:

```json
"escalation": {
  "escalated": true,
  "severity": "emergency",
  "category": "injury_accident",
  "reason": "Dog hit by car and bleeding now.",
  "language": "en",
  "ticket_id": "ESC-20261106-3FA2",
  "volunteer_notified": true,
  "check_failed": false
}
```

Other endpoints:

* `POST /api/escalations/{ticket_id}/contact` – body `{"name", "phone", "note"?}`
* `GET /api/escalations` – volunteer list, header `X-Admin-Token` required

## Configuration (Render → Environment)

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | yes (already set) | classifier uses the same key |
| `TELEGRAM_BOT_TOKEN` | for alerts | from @BotFather |
| `TELEGRAM_CHAT_ID` | for alerts | the volunteer group chat id |
| `ESCALATION_LLM_MODEL` | no | default `gpt-6-luna` |
| `ESCALATION_ADMIN_TOKEN` | no | enables `GET /api/escalations` |
| `ESCALATION_DB_PATH` | no | default `escalations.db` |
| `ESCALATION_ENABLED` | no | `false` turns the feature off |

Without the Telegram variables the alert is only written to the log, and the
page tells the adopter to contact the foundation directly.

Get the chat id: add the bot to the group, send any message, then open
`https://api.telegram.org/bot<TOKEN>/getUpdates` and copy `message.chat.id`
(a negative number for groups).

Note: Render's free tier has no persistent disk, so the SQLite tickets reset on
redeploy. The Telegram messages are the durable record for the demo.

## Tests and evals

```bash
python -m pytest tests                    # 12 unit/wiring tests, no key needed
python -m evals.run_eval                  # classifier on 44 labelled NL/EN questions
python -m evals.run_eval --api https://rescueai-post-adoption-api.onrender.com
                                          # end to end (creates real alerts)
```

`evals/escalation_eval_set.jsonl` covers clear emergencies, urgent cases,
borderline cases, routine guide questions and "what if" questions. The main
metric is **recall on urgent + emergency** (a missed case is the costly
error); precision shows how many alerts were really needed. Results go to
`evals/results/latest.md`.
