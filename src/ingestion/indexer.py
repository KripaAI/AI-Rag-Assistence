from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import ensure_local_dirs, settings
from src.ingestion.pdf_parser import extract_pdf_records, write_manifest
from src.models import Record


class IngestionService:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing in environment.")
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is missing in environment.")

        ensure_local_dirs()
        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = self._ensure_index()

    def _ensure_index(self):
        existing = self.pc.list_indexes().names()
        if settings.pinecone_index not in existing:
            self.pc.create_index(
                name=settings.pinecone_index,
                dimension=settings.pinecone_dimension,
                metric=settings.pinecone_metric,
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
        return self.pc.Index(settings.pinecone_index)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=12))
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self.openai.embeddings.create(model=settings.openai_embed_model, input=texts)
        return [row.embedding for row in resp.data]

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=12))
    def _caption_image(self, image_path: Path) -> str:
        mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"
        prompt = (
            "Describe this image for retrieval in a technical RAG system. "
            "Focus on visible entities, chart/table cues, labels, and topic keywords. "
            f"Keep it under {settings.image_caption_max_chars} characters."
        )
        resp = self.openai.chat.completions.create(
            model=settings.openai_vision_model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return "Image with technical content from source document."
        return text[: settings.image_caption_max_chars]

    def _iter_pdfs(self, data_dir: Path) -> Iterable[Path]:
        for path in data_dir.rglob("*.pdf"):
            if path.is_file():
                yield path

    @staticmethod
    def _load_manifest_ids(manifest_path: Path) -> set[str]:
        if not manifest_path.exists():
            return set()

        ids: set[str] = set()
        with manifest_path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec_id = str(raw.get("id", "")).strip()
                if rec_id:
                    ids.add(rec_id)
        return ids

    def _delete_stale_ids(self, stale_ids: set[str]) -> int:
        if not stale_ids:
            return 0

        batch_size = max(1, settings.pinecone_delete_batch_size)
        deleted = 0
        ids_list = list(stale_ids)
        for i in range(0, len(ids_list), batch_size):
            batch = ids_list[i : i + batch_size]
            self.index.delete(ids=batch)
            deleted += len(batch)
        return deleted

    @staticmethod
    def _image_bytes_hash(image_path: Path) -> str:
        return hashlib.sha1(image_path.read_bytes()).hexdigest()

    def build_records(self, data_dir: Path | None = None) -> list[Record]:
        source_dir = data_dir or settings.data_dir
        if not source_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {source_dir}")

        all_records: list[Record] = []
        for pdf_path in self._iter_pdfs(source_dir):
            pdf_records = extract_pdf_records(pdf_path)
            all_records.extend(pdf_records)

        self.enrich_image_captions(all_records)
        return all_records

    def enrich_image_captions(self, records: list[Record]) -> None:
        if not settings.enable_image_captioning:
            return

        image_records = [r for r in records if r.modality == "image" and r.image_path]
        valid_records = []
        for rec in image_records:
            p = Path(rec.image_path)
            if p.exists():
                valid_records.append(rec)

        if not valid_records:
            return

        cache_path = settings.manifests_dir / "image_caption_cache.json"
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                cache = {}
        else:
            cache = {}

        hash_to_path: dict[str, Path] = {}
        rec_hash: dict[str, str] = {}
        for rec in valid_records:
            p = Path(rec.image_path)
            h = self._image_bytes_hash(p)
            rec_hash[rec.id] = h
            if h not in hash_to_path:
                hash_to_path[h] = p

        uncached_hashes = [h for h in hash_to_path.keys() if h not in cache]
        max_images = settings.image_caption_max_images
        if max_images > 0:
            uncached_hashes = uncached_hashes[:max_images]

        workers = max(1, settings.image_caption_workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {
                pool.submit(self._caption_image, hash_to_path[h]): h
                for h in uncached_hashes
            }
            for fut in as_completed(future_map):
                h = future_map[fut]
                try:
                    cache[h] = fut.result()
                except Exception:
                    continue

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")

        for rec in valid_records:
            h = rec_hash.get(rec.id)
            if not h:
                continue
            caption = cache.get(h)
            if caption:
                rec.image_caption = caption
                rec.content = caption

    def upsert_records(self, records: list[Record], batch_size: int = 100) -> int:
        if not records:
            return 0

        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            embeds = self._embed_batch([r.content for r in batch])

            vectors = []
            for rec, emb in zip(batch, embeds):
                vectors.append({"id": rec.id, "values": emb, "metadata": rec.metadata()})

            self.index.upsert(vectors=vectors)
            total += len(vectors)
        return total

    def run(self, data_dir: Path | None = None, sync_index: bool | None = None) -> dict[str, int | str | bool]:
        sync_enabled = settings.sync_index_on_ingest if sync_index is None else bool(sync_index)
        records = self.build_records(data_dir)
        current_ids = {r.id for r in records}
        previous_ids = self._load_manifest_ids(settings.manifest_path)

        upserted = self.upsert_records(records)
        stale_ids = (previous_ids - current_ids) if sync_enabled else set()
        stale_deleted = self._delete_stale_ids(stale_ids)
        write_manifest(records, settings.manifest_path)

        modality_counts: dict[str, int] = {"text": 0, "table": 0, "image": 0}
        for rec in records:
            modality_counts[rec.modality] = modality_counts.get(rec.modality, 0) + 1

        summary = {
            "records_total": len(records),
            "upserted": upserted,
            "sync_index_enabled": sync_enabled,
            "stale_candidates": len(stale_ids),
            "stale_deleted": stale_deleted,
            "text_records": modality_counts.get("text", 0),
            "table_records": modality_counts.get("table", 0),
            "image_records": modality_counts.get("image", 0),
            "manifest": str(settings.manifest_path),
        }

        summary_path = settings.manifests_dir / "ingest_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
