from __future__ import annotations

import re
from dataclasses import dataclass

from openai import OpenAI

from src.config import settings
from src.retrieval.retrieve import RetrievedItem


@dataclass
class GenerationResult:
    answer_text: str
    citations: list[dict]
    diagnostics: dict


class AnswerGenerator:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing in environment.")
        self.client = OpenAI(api_key=settings.openai_api_key)

    def _build_context(self, items: list[RetrievedItem]) -> tuple[str, list[dict]]:
        context_blocks: list[str] = []
        citations: list[dict] = []
        for idx, item in enumerate(items, start=1):
            cite_id = f"C{idx}"
            snippet = item.text.strip()[:1200]
            context_blocks.append(
                f"[{cite_id}] file={item.source_file} page={item.page} modality={item.modality}\n{snippet}"
            )
            citations.append(
                {
                    "id": cite_id,
                    "source_file": item.source_file,
                    "page": item.page,
                    "modality": item.modality,
                    "record_id": item.id,
                }
            )
        return "\n\n".join(context_blocks), citations

    def _validate_citations(self, answer_text: str, citations: list[dict]) -> dict:
        cited_ids = set(re.findall(r"\[(C\d+)\]", answer_text))
        valid_ids = {c["id"] for c in citations}
        invalid_ids = sorted(cited_ids - valid_ids)
        has_any_valid = bool(cited_ids.intersection(valid_ids))
        return {
            "cited_ids": sorted(cited_ids),
            "invalid_ids": invalid_ids,
            "has_any_valid_citation": has_any_valid,
            "citation_count": len(cited_ids),
        }

    def _safe_fallback(self, citations: list[dict]) -> str:
        if not citations:
            return "I could not find enough grounded context to answer reliably."
        top = citations[:2]
        refs = ", ".join(f"[{c['id']}]" for c in top)
        return (
            "I could not produce a reliably grounded answer from the retrieved context. "
            f"Please refine the question or increase retrieval depth. Evidence: {refs}"
        )

    def answer(self, query: str, items: list[RetrievedItem]) -> GenerationResult:
        if not items:
            return GenerationResult(
                answer_text="I could not find relevant context in the indexed documents.",
                citations=[],
                diagnostics={"reason": "no_context_items"},
            )

        context_text, citations = self._build_context(items)
        prompt = (
            "You are a professional, grounded RAG assistant. You must prioritize accuracy using the provided CONTEXT blocks below.\n"
            "Rules:\n"
            "1) Use ONLY the facts present in the CONTEXT. Do NOT use outside knowledge.\n"
            "2) If the CONTEXT does not contain the answer, state: 'I could not find enough grounded context to answer reliably.' "
            "However, if the context contains partial information or multiple fragments that can be synthesized, answer as thoroughly as possible using only those fragments.\n"
            "3) Every factual claim in your answer MUST include an inline citation like [C1].\n"
            "4) Be objective and technical.\n\n"
            f"QUERY:\n{query}\n\n"
            f"CONTEXT:\n{context_text}"
        )

        response = self.client.chat.completions.create(
            model=settings.openai_chat_model,
            temperature=settings.answer_temperature,
            max_tokens=settings.answer_max_tokens,
            messages=[
                {"role": "system", "content": "You are an assistant that prioritizes faithfulness above all else. Never hallucinate. Never guess."},
                {"role": "user", "content": prompt},
            ],
        )

        answer_text = response.choices[0].message.content or "No answer generated."
        citation_check = self._validate_citations(answer_text, citations)
        missing_required = settings.require_citations_per_answer and not citation_check["has_any_valid_citation"]
        fallback_applied = missing_required
        
        if fallback_applied:
            answer_text = self._safe_fallback(citations)
        elif citation_check["invalid_ids"]:
            for inv in citation_check["invalid_ids"]:
                answer_text = answer_text.replace(f"[{inv}]", "")

        return GenerationResult(
            answer_text=answer_text,
            citations=citations,
            diagnostics={
                "used_context_items": len(items),
                "invalid_citations": citation_check["invalid_ids"],
                "has_valid_citation": citation_check["has_any_valid_citation"],
                "citation_count": citation_check["citation_count"],
                "fallback_applied": fallback_applied,
            },
        )
