from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz

from src.config import settings
from src.models import Record


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    step = max(chunk_size - chunk_overlap, 1)
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start += step
    return chunks


def rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    max_cols = max(len(r) for r in rows)
    norm = [r + [""] * (max_cols - len(r)) for r in rows]
    header = norm[0]
    sep = ["---"] * max_cols
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for row in norm[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_pdf_records(pdf_path: Path) -> list[Record]:
    doc = fitz.open(pdf_path)
    source_path = str(pdf_path.resolve())
    doc_id = _sha1(source_path)
    records: list[Record] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1

        # Text chunks
        text = page.get_text("text")
        text_chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        for chunk_idx, chunk in enumerate(text_chunks):
            rec_id = f"{doc_id}:p{page_num}:t{chunk_idx}:{_sha1(chunk)}"
            records.append(
                Record(
                    id=rec_id,
                    content=chunk,
                    modality="text",
                    doc_id=doc_id,
                    source_file=pdf_path.name,
                    source_path=source_path,
                    page=page_num,
                    chunk_index=chunk_idx,
                    checksum=_sha1(chunk),
                )
            )

        # Table extraction via PyMuPDF table finder when available.
        try:
            table_finder = page.find_tables()
            tables = table_finder.tables if table_finder else []
        except Exception:
            tables = []

        for table_idx, table in enumerate(tables):
            extracted = table.extract() if table else []
            rows = []
            for row in extracted:
                rows.append([str(cell).strip() if cell is not None else "" for cell in row])
            md = rows_to_markdown(rows)
            if not md.strip():
                continue
            content = f"Table from page {page_num}:\n{md}"
            rec_id = f"{doc_id}:p{page_num}:tbl{table_idx}:{_sha1(md)}"
            records.append(
                Record(
                    id=rec_id,
                    content=content,
                    modality="table",
                    doc_id=doc_id,
                    source_file=pdf_path.name,
                    source_path=source_path,
                    page=page_num,
                    chunk_index=table_idx,
                    checksum=_sha1(md),
                    table_markdown=md,
                )
            )

        # Image extraction from embedded objects.
        image_list = page.get_images(full=True)
        for image_idx, img in enumerate(image_list):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue
            image_bytes = base_image.get("image")
            ext = base_image.get("ext", "png")
            if not image_bytes:
                continue

            image_name = f"{doc_id}_p{page_num}_img{image_idx}.{ext}"
            image_path = settings.images_dir / image_name
            image_path.write_bytes(image_bytes)

            caption = f"Image extracted from {pdf_path.name} page {page_num}."
            content = caption
            rec_id = f"{doc_id}:p{page_num}:img{image_idx}:{_sha1(image_name)}"
            records.append(
                Record(
                    id=rec_id,
                    content=content,
                    modality="image",
                    doc_id=doc_id,
                    source_file=pdf_path.name,
                    source_path=source_path,
                    page=page_num,
                    chunk_index=image_idx,
                    checksum=_sha1(image_name),
                    image_path=str(image_path.resolve()),
                    image_caption=caption,
                )
            )

    doc.close()
    return records


def write_manifest(records: list[Record], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.to_json(), ensure_ascii=True) + "\n")
