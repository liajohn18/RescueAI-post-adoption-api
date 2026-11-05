import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
STOP_WORDS = {
    # Frequent grammar words and generic pet terms add little topic signal.
    "de", "het", "een", "en", "of", "maar", "dat", "dit", "wat", "hoe",
    "ik", "je", "jij", "u", "uw", "mijn", "zijn", "haar", "hij", "zij",
    "ze", "er", "te", "in", "op", "aan", "voor", "van", "met", "naar",
    "bij", "als", "om", "is", "zijn", "wordt", "kan", "kunnen", "moet",
    "doe", "doen", "maak", "die", "dan", "ook", "pas", "tot", "meer",
    "the", "a", "an", "and", "or", "but", "that", "this", "what", "how",
    "i", "you", "your", "my", "he", "she", "it", "we", "they", "to",
    "of", "in", "on", "at", "for", "with", "from", "is", "are", "be",
    "do", "does", "can", "should", "would", "could", "new", "newly", "dog",
    "dogs",
}


def tokenize(text: str) -> list[str]:
    """Lowercase and lightly normalize Dutch words for keyword matching."""
    tokens = TOKEN_PATTERN.findall(text.lower())
    # Make common Dutch adjective/noun forms match (e.g. veilig/veiligheid,
    # zindelijk/zindelijkheid) without adding a stemming dependency.
    normalized = [
        token[:-4] if token.endswith("heid") and len(token) > 6 else token
        for token in tokens
    ]
    return [token for token in normalized if token not in STOP_WORDS]


class BM25SectionSearch:
    def __init__(self, corpus_path: str | Path):
        records = json.loads(
            Path(corpus_path).read_text(encoding="utf-8")
        )

        # Exclude the contents-page record and any empty sections.
        self.sections = [
            record
            for record in records
            if record["section_number"] is not None
            and record["text"].strip()
        ]

        # Give heading terms extra weight so topic matches outrank incidental
        # mentions in a broad section (e.g. "wandelen" -> that named section).
        self.tokens = [
            tokenize(record["section_title"]) * 3
            + tokenize(record["text"])
            for record in self.sections
        ]

        self.index = BM25Okapi(self.tokens)

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        query_tokens = tokenize(question)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

        results = []
        for section_index, score in ranked[:top_k]:
            if score <= 0:
                continue

            section = self.sections[section_index]
            results.append({
                "section_number": section["section_number"],
                "section_title": section["section_title"],
                "pages": section["pages"],
                "source_file": section["source_file"],
                "score": float(score),
                "text": section["text"],
            })

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()

    searcher = BM25SectionSearch("data/post_adoption_sections.json")
    for result in searcher.search(args.question):
        print(
            f"\nSection {result['section_number']}: "
            f"{result['section_title']} | pages {result['pages']} "
            f"| score {result['score']:.3f}"
        )
        print(result["text"][:700])
