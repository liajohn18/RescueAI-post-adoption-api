import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
QUESTIONS_FILE = HERE / "evaluation_questions.json"
RESULTS_FILE = HERE / "api_evaluation_results.json"
API_URL = "http://127.0.0.1:8000/api/ask"


def source_matches(question, sources):
    expected_title = question.get("expected_section_title")
    expected_pages = set(question.get("expected_pages", []))

    # Unsupported questions should return no source.
    if expected_title is None and not expected_pages:
        return not sources

    for source in sources:
        title_matches = (
            source.get("title", "").casefold() == expected_title.casefold()
        )
        pages = set(source.get("pages", []))
        if title_matches and expected_pages.issubset(pages):
            return True

    return False


def ask_api(question):
    body = json.dumps(
        {"question": question["question"]},
        ensure_ascii=False,
    ).encode("utf-8")

    request = Request(
        API_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ids",
        nargs="*",
        help="Optional question IDs to run, for example: --ids q01 q20",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on how many questions to run",
    )
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    if args.ids:
        selected_ids = set(args.ids)
        questions = [q for q in questions if q["id"] in selected_ids]
    if args.limit:
        questions = questions[:args.limit]

    results = []
    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] Asking {question['id']}...")

        try:
            response = ask_api(question)
            sources = response.get("sources", [])
            result = {
                "id": question["id"],
                "question": question["question"],
                "expected_language": question["language"],
                "expected_section_title": question.get("expected_section_title"),
                "expected_pages": question.get("expected_pages", []),
                "actual_sources": sources,
                "source_match": source_matches(question, sources),
                "answer": response.get("answer", ""),
                "must_include": question.get("must_include", []),
                "must_not_include": question.get("must_not_include", []),
                "should_escalate": question.get("should_escalate", False),
                "specialist_review_required": question.get(
                    "specialist_review_required", False
                ),
                "answer_review": "TODO: review answer against the checklist",
            }
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace").strip()
            result = {
                "id": question["id"],
                "question": question["question"],
                "api_error": f"{error}: {detail}" if detail else str(error),
                "source_match": None,
                "answer_review": "API request failed; source was not scored",
            }
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            result = {
                "id": question["id"],
                "question": question["question"],
                "api_error": str(error),
                "source_match": None,
                "answer_review": "API request failed",
            }

        results.append(result)
        RESULTS_FILE.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        status = (
            "API ERROR"
            if result["source_match"] is None
            else "SOURCE PASS"
            if result["source_match"]
            else "SOURCE FAIL"
        )
        print(f"  {status}")

    passed = sum(1 for result in results if result.get("source_match"))
    scored = sum(1 for result in results if result.get("source_match") is not None)
    api_errors = len(results) - scored
    print(f"\nSource matches: {passed}/{scored}")
    if api_errors:
        print(f"API errors (not included in source score): {api_errors}")
    print(f"Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
