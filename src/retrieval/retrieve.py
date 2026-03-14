from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from pinecone import Pinecone

from src.config import settings


@dataclass
class RetrievedItem:
    id: str
    score: float
    vector_score: float
    overlap_score: float
    source_prior: float
    rerank_score: float
    modality: str
    text: str
    source_file: str
    page: int
    image_path: str = ""
    table_markdown: str = ""


class RetrievalService:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing in environment.")
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is missing in environment.")

        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self.pc.Index(settings.pinecone_index)

    def _embed_query(self, query: str) -> list[float]:
        emb = self.openai.embeddings.create(model=settings.openai_embed_model, input=[query])
        return emb.data[0].embedding

    def _query_modality(self, vector: list[float], modality: str, top_k: int) -> list[dict]:
        if top_k <= 0:
            return []
        res = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter={"modality": {"$eq": modality}},
        )
        return res.matches or []

    def _tokenize(self, text: str) -> set[str]:
        return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 3}

    def _query_overlap(self, query_tokens: set[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        text_tokens = self._tokenize(text)
        if not text_tokens:
            return 0.0
        return len(query_tokens.intersection(text_tokens)) / len(query_tokens)

    def _infer_preferred_sources(self, query: str) -> list[str]:
        if not settings.retrieve_enable_query_source_routing:
            return []

        qt = self._tokenize(query)
        preferred: list[str] = []

        rag_terms = {
            "rag", "retrieval", "generation", "chunking", "reranking", "faithfulness",
            "precision", "recall", "semantic", "keyword", "hybrid", "vector", "ingestion",
            "architecture", "evaluation", "monitoring", "production",
        }
        agent_terms = {"agent", "agents", "mcp", "memory", "session", "context", "interoperability", "quality"}
        pytorch_terms = {"pytorch", "tensor", "backprop", "optimizer"}

        if qt.intersection(rag_terms):
            preferred.extend(
                [
                    "1introduction to retrieval-augmented generation",
                    "2foundations of information retrieval",
                    "3vector databases and advanced retrieval",
                    "4large language models and text generation",
                    "5rag systems in production",
                ]
            )
        if qt.intersection(agent_terms):
            preferred.extend(
                [
                    "introduction to agents",
                    "context engineering_ sessions & memory",
                    "agent quality",
                    "agent tools & interoperability with model context protocol (mcp)",
                    "prototype to production",
                ]
            )
        if qt.intersection(pytorch_terms):
            preferred.extend(["pytorch_c1_"])
        return preferred

    def _source_prior(self, source_file: str, preferred: list[str]) -> float:
        if not preferred:
            return 0.5
        sf = source_file.lower()
        return 1.0 if any(p in sf for p in preferred) else 0.0

    def _is_low_information_text(self, item: RetrievedItem) -> bool:
        text = (item.text or "").strip()
        if not text:
            return True
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if len(tokens) < 4:
            return True

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) <= 1 and len(tokens) < 10:
            return True
        return False

    def _is_low_information_table(self, item: RetrievedItem) -> bool:
        if not settings.retrieve_drop_table_stubs:
            return False
        text = (item.text or "").lower()
        if "table from page" not in text:
            return False
        tokens = re.findall(r"[a-z0-9]+", text)
        if len(tokens) < settings.retrieve_min_informative_tokens:
            return True
        low_signal_markers = ("| --- |", "table from page")
        marker_hits = sum(1 for m in low_signal_markers if m in text)
        return marker_hits >= 2 and len(tokens) < (settings.retrieve_min_informative_tokens * 2)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedItem]:
        k = top_k or settings.top_k
        vector = self._embed_query(query)
        query_tokens = self._tokenize(query)
        preferred_sources = self._infer_preferred_sources(query)

        text_k = max(settings.retrieve_text_k, k)
        table_k = max(settings.retrieve_table_k, k // 2)
        image_k = max(settings.retrieve_image_k, k // 2)

        text_matches = self._query_modality(vector, "text", text_k)
        table_matches = self._query_modality(vector, "table", table_k)
        image_matches = self._query_modality(vector, "image", image_k)

        weights = {
            "text": settings.retrieve_weight_text,
            "table": settings.retrieve_weight_table,
            "image": settings.retrieve_weight_image,
        }

        fused_scores: dict[str, float] = {}
        best_match: dict[str, dict] = {}

        for modality, matches in (
            ("text", text_matches),
            ("table", table_matches),
            ("image", image_matches),
        ):
            w = weights[modality]
            for rank, match in enumerate(matches, start=1):
                score = w * (1.0 / (settings.retrieve_rrf_k + rank))
                fused_scores[match.id] = fused_scores.get(match.id, 0.0) + score
                if match.id not in best_match:
                    best_match[match.id] = {"match": match, "score": score}

        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        max_fused = max(fused_scores.values()) if fused_scores else 1.0

        dedup: dict[str, RetrievedItem] = {}
        for rec_id in sorted_ids[: max(k, settings.max_context_items) * 2]:
            m = best_match[rec_id]["match"]
            md = m.metadata or {}
            text = str(md.get("text", ""))
            fused = float(fused_scores.get(rec_id, 0.0))
            normalized_fused = fused / max(max_fused, 1e-9)
            overlap = self._query_overlap(query_tokens, text)
            source_prior = self._source_prior(str(md.get("source_file", "")), preferred_sources)
            modality = str(md.get("modality", "text"))
            modality_bonus = (
                settings.retrieve_modality_bonus_text
                if modality == "text"
                else settings.retrieve_modality_bonus_table
                if modality == "table"
                else 0.0
            )
            rerank_score = (
                settings.retrieve_rerank_vector_weight * normalized_fused
                + settings.retrieve_rerank_overlap_weight * overlap
                + settings.retrieve_rerank_source_weight * source_prior
                + modality_bonus
            )

            item = RetrievedItem(
                id=m.id,
                score=fused,
                vector_score=normalized_fused,
                overlap_score=overlap,
                source_prior=source_prior,
                rerank_score=rerank_score,
                modality=modality,
                text=text,
                source_file=str(md.get("source_file", "")),
                page=int(md.get("page", 0) or 0),
                image_path=str(md.get("image_path", "")),
                table_markdown=str(md.get("table_markdown", "")),
            )

            dedup_key = f"{item.modality}|{item.source_file}|{item.page}"
            prev = dedup.get(dedup_key)
            if prev is None or item.rerank_score > prev.rerank_score:
                dedup[dedup_key] = item

        items = sorted(dedup.values(), key=lambda x: x.rerank_score, reverse=True)

        items = [
            i
            for i in items
            if (
                i.modality != "text"
                or i.source_prior > 0.0
                or len((i.text or "").strip()) >= settings.retrieve_min_text_chars
            )
            and not (i.modality == "text" and self._is_low_information_text(i))
            and not (i.modality == "table" and self._is_low_information_table(i))
        ]

        strong_items = [
            i
            for i in items
            if i.score >= settings.retrieve_min_fused_score and i.overlap_score >= settings.retrieve_min_query_overlap
        ]

        pool = strong_items if len(strong_items) >= max(3, min(k, settings.max_context_items // 2)) else items

        domain_items = [i for i in pool if i.source_prior > 0.0]
        if preferred_sources and len(domain_items) >= settings.retrieve_domain_min_items:
            pool = domain_items + [i for i in pool if i.source_prior == 0.0]

        def ensure_text_coverage(base_items: list[RetrievedItem]) -> list[RetrievedItem]:
            text_items = [i for i in base_items if i.modality == "text"]
            req_text = min(settings.min_text_context_items, len(text_items))
            selected_text = text_items[:req_text]
            
            remaining = [i for i in base_items if i not in selected_text]
            return (selected_text + remaining)[:k]

        return ensure_text_coverage(pool)

    def build_context(self, query: str, top_k: int | None = None) -> dict:
        items = self.retrieve(query, top_k=top_k)

        text_or_table = [i for i in items if i.modality in ("text", "table")]
        image_items = [i for i in items if i.modality == "image" and i.image_path]
        table_items = [i for i in items if i.modality == "table" and i.table_markdown]
        text_items = [i for i in text_or_table if i.modality == "text"]
        non_text_items = [i for i in text_or_table if i.modality != "text"]

        filtered_images = []
        for item in image_items:
            p = Path(item.image_path)
            if p.exists():
                filtered_images.append(item)
            if len(filtered_images) >= settings.max_images_in_response:
                break

        context_items: list[RetrievedItem] = []
        used_chars = 0
        for item in text_items:
            snippet_len = len(item.text or "")
            if len(context_items) >= settings.max_context_items:
                break
            if used_chars + snippet_len > settings.max_context_chars and context_items:
                break
            context_items.append(item)
            used_chars += snippet_len
            if len([c for c in context_items if c.modality == "text"]) >= settings.min_text_context_items:
                break

        for item in text_items:
            if item in context_items:
                continue
            snippet_len = len(item.text or "")
            if len(context_items) >= settings.max_context_items:
                break
            if used_chars + snippet_len > settings.max_context_chars and context_items:
                break
            context_items.append(item)
            used_chars += snippet_len

        for item in non_text_items:
            snippet_len = len(item.text or "")
            if len(context_items) >= settings.max_context_items:
                break
            if used_chars + snippet_len > settings.max_context_chars and context_items:
                break
            context_items.append(item)
            used_chars += snippet_len

        for img in filtered_images[: settings.max_images_in_context]:
            if len(context_items) >= settings.max_context_items:
                break
            context_items.append(img)

        selected_tables = table_items[: settings.max_tables_in_response]

        return {
            "query": query,
            "items": context_items,
            "images": filtered_images,
            "tables": selected_tables,
            "diagnostics": {
                "retrieved_total": len(items),
                "selected_context_items": len(context_items),
                "selected_context_chars": used_chars,
                "selected_text_table_items": len([i for i in context_items if i.modality in ("text", "table")]),
                "selected_image_items": len([i for i in context_items if i.modality == "image"]),
                "source_routed_items": len([i for i in items if i.source_prior > 0.0]),
            },
        }
