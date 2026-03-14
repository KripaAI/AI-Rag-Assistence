from __future__ import annotations

from src.generation.answer import AnswerGenerator
from src.retrieval.retrieve import RetrievalService


class RagPipeline:
    def __init__(self) -> None:
        self.retriever = RetrievalService()
        self.generator = AnswerGenerator()

    def ask(self, query: str, top_k: int | None = None) -> dict:
        context = self.retriever.build_context(query, top_k=top_k)
        gen = self.generator.answer(query, context["items"])

        return {
            "query": query,
            "answer_text": gen.answer_text,
            "citations": gen.citations,
            "generation": gen.diagnostics,
            "retrieval": context.get("diagnostics", {}),
            "images": [
                {
                    "path": img.image_path,
                    "source_file": img.source_file,
                    "page": img.page,
                    "score": img.score,
                }
                for img in context["images"]
            ],
            "tables": [
                {
                    "markdown": tbl.table_markdown,
                    "source_file": tbl.source_file,
                    "page": tbl.page,
                    "score": tbl.score,
                }
                for tbl in context["tables"]
            ],
            "retrieved": [
                {
                    "id": item.id,
                    "modality": item.modality,
                    "source_file": item.source_file,
                    "page": item.page,
                    "score": item.score,
                    "vector_score": item.vector_score,
                    "overlap_score": item.overlap_score,
                    "source_prior": item.source_prior,
                    "rerank_score": item.rerank_score,
                    "text_snippet": item.text[:500],
                }
                for item in context["items"]
            ],
            "contexts": [item.text[:1500] for item in context["items"]],
        }
