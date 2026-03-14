from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Record:
    id: str
    content: str
    modality: str
    doc_id: str
    source_file: str
    source_path: str
    page: int
    chunk_index: int
    checksum: str
    table_markdown: str = ""
    image_path: str = ""
    image_caption: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def metadata(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "source_path": self.source_path,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "text": self.content,
            "table_markdown": self.table_markdown,
            "image_path": self.image_path,
            "image_caption": self.image_caption,
            "checksum": self.checksum,
        }
