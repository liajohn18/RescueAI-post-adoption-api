"""Combine BM25 keyword ranking and multilingual semantic ranking."""

from pathlib import Path

from post_adoption_rag.bm25_search import BM25SectionSearch
from post_adoption_rag.semantic_search import SemanticSectionSearch


RRF_CONSTANT = 60

BM25_WEIGHT = 1.0
SEMANTIC_WEIGHT = 3.0

class HybridSectionSearch:
    """Fuse BM25 and semantic ranks with reciprocal rank fusion (RRF)."""

    def __init__(self, corpus_path: str | Path):
        self.bm25 = BM25SectionSearch(corpus_path)
        self.semantic = SemanticSectionSearch(corpus_path)

    def search(
        self,
        question: str,
        top_k: int = 1,
        candidate_k: int = 10,
    ) -> list[dict]:
        bm25_results = self.bm25.search(question, top_k=candidate_k)
        semantic_results = self.semantic.search(question, top_k=candidate_k)

        combined: dict[int, dict] = {}

        for rank, result in enumerate(bm25_results, start=1):
            section_number = result["section_number"]
            item = combined.setdefault(section_number, dict(result))
            item["bm25_score"] = result["score"]
            item["bm25_rank"] = rank
            item["rrf_score"] = item.get("rrf_score", 0.0) + (
    BM25_WEIGHT / (RRF_CONSTANT + rank)
)
        for rank, result in enumerate(semantic_results, start=1):
            section_number = result["section_number"]
            item = combined.setdefault(section_number, dict(result))
            item["semantic_score"] = result["score"]
            item["semantic_rank"] = rank
            item["rrf_score"] = item.get("rrf_score", 0.0) + (
    SEMANTIC_WEIGHT / (RRF_CONSTANT + rank)
)

        for item in combined.values():
            item["score"] = item["rrf_score"]

        ranked = sorted(
            combined.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )
        return ranked[:top_k]
