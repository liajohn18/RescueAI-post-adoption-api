"""Extract a PDF into section-aware records while preserving source pages.

Usage:
    python -m post_adoption_rag.ingest input.pdf output.json

The output is a preparation artifact, not an approved chatbot knowledge base.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    import pymupdf
except ImportError as exc:  # pragma: no cover - exercised when dependency is absent
    raise SystemExit(
        "Missing dependency 'pymupdf'. Install project dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from exc


SECTION_LINE = re.compile(r"^\s*(\d{1,2})\.\s*(.*?)\s*$")
HEADER_NOISE = {
    "Richtlijnen eerste periode na aankomst",
    "DOGS ON A JOURNEY",
}


def _clean_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.replace("\x00", "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line or line in HEADER_NOISE:
            continue
        # The PDF prints page numbers as e.g. "4 van 11" in the running header.
        if re.fullmatch(r"\d{1,2}\s+van\s+11", line, flags=re.IGNORECASE):
            continue
        lines.append(line)
    return lines


def extract_pdf_sections(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Return section records with text and 1-based source page numbers.

    Section headings in the source are numbered (for example, "7. Voeding").
    Any text before the first numbered heading is retained as an introduction.
    """
    path = Path(pdf_path)
    reader = pymupdf.open(str(path))
    sections: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for page_number, page in enumerate(reader, start=1):
        page_text = page.get_text("text", sort=True) or ""
        lines = _clean_lines(page_text)
        i = 0
        while i < len(lines):
            line = lines[i]
            match = SECTION_LINE.match(line)
            if match:
                number, title = match.groups()
                # Some PDF extractors place the heading number and title on separate lines.
                if not title and i + 1 < len(lines):
                    title = lines[i + 1]
                    i += 1
                current = {
                    "section_number": int(number),
                    "section_title": title.strip(),
                    "text": "",
                    "pages": [],
                }
                sections.append(current)
            elif current is None:
                current = {
                    "section_number": None,
                    "section_title": "Introduction",
                    "text": "",
                    "pages": [],
                }
                sections.append(current)

            if current is not None:
                if page_number not in current["pages"]:
                    current["pages"].append(page_number)
                if not SECTION_LINE.match(line):
                    current["text"] = (current["text"] + " " + line).strip()
            i += 1
    reader.close()
    source_name = path.name
    for index, section in enumerate(sections, start=1):
        section["id"] = f"pdf-section-{index:02d}"
        section["source_file"] = source_name
        section["language"] = "nl"
        section["review_status"] = "unreviewed"
        section["approved_for_adopter_answers"] = False
    return sections


def write_sections_json(pdf_path: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    sections = extract_pdf_sections(pdf_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(sections, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="Path to the source PDF")
    parser.add_argument("output", help="Path for the section JSON output")
    args = parser.parse_args()
    sections = write_sections_json(args.pdf, args.output)
    print(f"Extracted {len(sections)} sections to {args.output}")


if __name__ == "__main__":
    main()
