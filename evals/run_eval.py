"""Evaluate the escalation classifier on the labelled set.

Usage (from the repo root):
    export OPENAI_API_KEY=sk-...            # or ESCALATION_LLM_BASE_URL for Ollama
    python -m evals.run_eval                 # classifier only, local
    python -m evals.run_eval --api https://rescueai-post-adoption-api.onrender.com
                                             # end to end against the deployed /api/ask
                                             # (this creates real tickets/alerts!)

Writes evals/results/<timestamp>.json and evals/results/latest.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
DATASET = HERE / "escalation_eval_set.jsonl"
RESULTS = HERE / "results"


def load_cases() -> list[dict]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


def predict_local():
    from escalation.classifier import LLMClassifier
    from escalation.config import get_settings

    clf = LLMClassifier(get_settings())

    def run(q: str) -> dict:
        d = clf.classify(q)
        return {"escalate": d.escalate, "severity": d.severity.value, "category": d.category.value, "reason": d.reason}

    return run, f"classifier:{get_settings().llm_model}"


def predict_api(base_url: str):
    import httpx

    def run(q: str) -> dict:
        r = httpx.post(f"{base_url.rstrip('/')}/api/ask", json={"question": q}, timeout=120)
        r.raise_for_status()
        e = r.json().get("escalation") or {}
        return {
            "escalate": bool(e.get("escalated")),
            "severity": e.get("severity", "none"),
            "category": e.get("category", "none"),
            "reason": e.get("reason", ""),
        }

    return run, f"api:{base_url}"


def score(rows: list[dict]) -> dict:
    tp = sum(r["escalate"] and r["pred"]["escalate"] for r in rows)
    fn = sum(r["escalate"] and not r["pred"]["escalate"] for r in rows)
    fp = sum(not r["escalate"] and r["pred"]["escalate"] for r in rows)
    tn = sum(not r["escalate"] and not r["pred"]["escalate"] for r in rows)
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    esc_hits = [r for r in rows if r["escalate"] and r["pred"]["escalate"]]
    emergencies = [r for r in rows if r["severity"] == "emergency"]
    return {
        "n": len(rows),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": recall, "precision": precision, "f1": f1,
        "accuracy": (tp + tn) / len(rows) if rows else None,
        "emergency_recall": (
            sum(r["pred"]["escalate"] for r in emergencies) / len(emergencies) if emergencies else None
        ),
        "severity_accuracy_on_hits": (
            sum(r["pred"]["severity"] == r["severity"] for r in esc_hits) / len(esc_hits) if esc_hits else None
        ),
        "category_accuracy_on_hits": (
            sum(r["pred"]["category"] == r["category"] for r in esc_hits) / len(esc_hits) if esc_hits else None
        ),
    }


def pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", help="Base URL of a deployed backend (end-to-end mode)")
    args = ap.parse_args()

    predict, target = predict_api(args.api) if args.api else predict_local()
    cases = load_cases()
    rows, latencies = [], []
    for c in cases:
        t0 = time.perf_counter()
        try:
            pred = predict(c["question"])
        except Exception as exc:  # count as "not escalated" so failures hurt recall
            pred = {"escalate": False, "severity": "none", "category": "none", "reason": f"ERROR {exc}"}
        latencies.append(time.perf_counter() - t0)
        rows.append({**c, "pred": pred})
        mark = "OK " if pred["escalate"] == c["escalate"] else "XX "
        print(f"{mark}{c['id']:4} exp={c['severity']:9} got={pred['severity']:9} {c['question'][:60]}")

    overall = score(rows)
    by_lang = {k: score([r for r in rows if r["lang"] == k]) for k in sorted({r["lang"] for r in rows})}
    by_kind = {k: score([r for r in rows if r["kind"] == k]) for k in sorted({r["kind"] for r in rows})}
    lat = {"p50_s": statistics.median(latencies), "p95_s": sorted(latencies)[int(0.95 * (len(latencies) - 1))]}
    misses = [r for r in rows if r["pred"]["escalate"] != r["escalate"]]

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"{stamp}.json").write_text(
        json.dumps({"target": target, "overall": overall, "by_lang": by_lang, "by_kind": by_kind,
                    "latency": lat, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = [
        f"# Escalation eval — {stamp}",
        f"Target: `{target}` · {overall['n']} labelled questions (NL + EN)",
        "",
        "| Metric | Value |", "|---|---|",
        f"| **Recall (urgent+emergency caught)** | **{pct(overall['recall'])}** ({overall['tp']}/{overall['tp'] + overall['fn']}) |",
        f"| Emergency recall | {pct(overall['emergency_recall'])} |",
        f"| Precision (alerts that were needed) | {pct(overall['precision'])} |",
        f"| F1 | {pct(overall['f1'])} |",
        f"| Overall accuracy | {pct(overall['accuracy'])} |",
        f"| Severity correct (on caught cases) | {pct(overall['severity_accuracy_on_hits'])} |",
        f"| Category correct (on caught cases) | {pct(overall['category_accuracy_on_hits'])} |",
        f"| Latency p50 / p95 | {lat['p50_s']:.2f}s / {lat['p95_s']:.2f}s |",
        "",
        "| Slice | n | Recall | Precision | Accuracy |", "|---|---|---|---|---|",
        *[f"| lang={k} | {v['n']} | {pct(v['recall'])} | {pct(v['precision'])} | {pct(v['accuracy'])} |" for k, v in by_lang.items()],
        *[f"| kind={k} | {v['n']} | {pct(v['recall'])} | {pct(v['precision'])} | {pct(v['accuracy'])} |" for k, v in by_kind.items()],
        "",
        "## Misclassified",
        *([f"- **{r['id']}** ({'missed escalation' if r['escalate'] else 'false alarm'}): "
           f"\"{r['question']}\" → got `{r['pred']['severity']}` — {r['pred']['reason']}" for r in misses] or ["- none"]),
    ]
    (RESULTS / "latest.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
