import argparse
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer, util


MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_CACHE = Path(__file__).resolve().parents[1] / ".model_cache"


class SemanticSectionSearch:
    def __init__(self, corpus_path: str | Path):
        records = json.loads(
            Path(corpus_path).read_text(encoding="utf-8")
        )

        # Skip the contents page; search actual numbered sections.
        self.sections = [
            record
            for record in records
            if record["section_number"] is not None
            and record["text"].strip()
        ]

        self.model = SentenceTransformer(
            MODEL_NAME,
            cache_folder=str(MODEL_CACHE),
            local_files_only=True,
        )

        # E5 expects document passages to start with this prefix.
        passages = [
            "passage: " + section["section_title"] + "\n" + section["text"]
            for section in self.sections
        ]

        self.embeddings = self.model.encode(
            passages,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        # E5 expects questions to use the query prefix.
        query_embedding = self.model.encode(
            "query: " + question,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )

        scores = util.cos_sim(query_embedding, self.embeddings)[0]
        ranked_indices = scores.argsort(descending=True)[:top_k].tolist()

        results = []
        for index in ranked_indices:
            section = self.sections[index]
            results.append({
                "section_number": section["section_number"],
                "section_title": section["section_title"],
                "pages": section["pages"],
                "source_file": section["source_file"],
                "score": scores[index].item(),
                "text": section["text"],
            })

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()

    searcher = SemanticSectionSearch("data/post_adoption_sections.json")
    for result in searcher.search(args.question):
        print(
            f"\nSection {result['section_number']}: "
            f"{result['section_title']} | pages {result['pages']} "
            f"| score {result['score']:.3f}"
        )
        print(result["text"][:700])
