import json
from pathlib import Path

from post_adoption_rag.bm25_search import BM25SectionSearch
from post_adoption_rag.semantic_search import SemanticSectionSearch
from post_adoption_rag.hybrid_search import HybridSectionSearch

CORPUS = "data/post_adoption_sections.json"
QUESTIONS = "post_adoption_rag/evaluation_questions.json"
TOP_K = 5


def evaluate(name, searcher, questions):
    eligible = 0
    hits = 0
    reciprocal_ranks = []

    print(f"\n{name}")
    print("-" * len(name))

    for item in questions:
        expected = item["expected_section_numbers"]

        # Questions with no supporting section are not retrieval-scored.
        if not expected:
            continue

        eligible += 1
        results = searcher.search(item["question"], top_k=TOP_K)
        ranked_sections = [result["section_number"] for result in results]

        matching_ranks = [
            rank
            for rank, section in enumerate(ranked_sections, start=1)
            if section in expected
        ]

        if matching_ranks:
            rank = min(matching_ranks)
            hits += 1
            reciprocal_ranks.append(1 / rank)
            outcome = f"hit at rank {rank}"
        else:
            reciprocal_ranks.append(0)
            outcome = "miss"

        print(f"{item['id']}: {outcome}; expected {expected}; got {ranked_sections}")

    hit_at_5 = hits / eligible if eligible else 0
    mrr = sum(reciprocal_ranks) / eligible if eligible else 0
    print(f"Hit@{TOP_K}: {hits}/{eligible} = {hit_at_5:.2f}")
    print(f"MRR: {mrr:.2f}")


def main():
    questions = json.loads(
        Path(QUESTIONS).read_text(encoding="utf-8")
    )

    bm25 = BM25SectionSearch(CORPUS)
    semantic = SemanticSectionSearch(CORPUS)
    hybrid = HybridSectionSearch(CORPUS)

    evaluate("BM25", bm25, questions)
    evaluate("Semantic", semantic, questions)
    evaluate("Hybrid RRF", hybrid, questions)


if __name__ == "__main__":
    main()