from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    openai_vision_model: str = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_chat_model: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-flash-latest")

    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "local-rag-multimodal")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    pinecone_dimension: int = int(os.getenv("PINECONE_DIMENSION", "3072"))
    pinecone_metric: str = os.getenv("PINECONE_METRIC", "cosine")
    sync_index_on_ingest: bool = os.getenv("SYNC_INDEX_ON_INGEST", "true").lower() == "true"
    pinecone_delete_batch_size: int = int(os.getenv("PINECONE_DELETE_BATCH_SIZE", "100"))

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "15"))
    max_context_items: int = int(os.getenv("MAX_CONTEXT_ITEMS", "25"))
    max_images_in_response: int = int(os.getenv("MAX_IMAGES_IN_RESPONSE", "3"))
    max_tables_in_response: int = int(os.getenv("MAX_TABLES_IN_RESPONSE", "1"))
    max_images_in_context: int = int(os.getenv("MAX_IMAGES_IN_CONTEXT", "1"))

    enable_image_captioning: bool = os.getenv("ENABLE_IMAGE_CAPTIONING", "true").lower() == "true"
    image_caption_max_images: int = int(os.getenv("IMAGE_CAPTION_MAX_IMAGES", "300"))
    image_caption_max_chars: int = int(os.getenv("IMAGE_CAPTION_MAX_CHARS", "180"))
    image_caption_workers: int = int(os.getenv("IMAGE_CAPTION_WORKERS", "4"))

    retrieve_text_k: int = int(os.getenv("RETRIEVE_TEXT_K", "30"))
    retrieve_table_k: int = int(os.getenv("RETRIEVE_TABLE_K", "10"))
    retrieve_image_k: int = int(os.getenv("RETRIEVE_IMAGE_K", "5"))
    retrieve_weight_text: float = float(os.getenv("RETRIEVE_WEIGHT_TEXT", "1.0"))
    retrieve_weight_table: float = float(os.getenv("RETRIEVE_WEIGHT_TABLE", "0.4"))
    retrieve_weight_image: float = float(os.getenv("RETRIEVE_WEIGHT_IMAGE", "0.1"))
    retrieve_rrf_k: int = int(os.getenv("RETRIEVE_RRF_K", "50"))
    retrieve_min_fused_score: float = float(os.getenv("RETRIEVE_MIN_FUSED_SCORE", "0.001"))
    retrieve_min_query_overlap: float = float(os.getenv("RETRIEVE_MIN_QUERY_OVERLAP", "0.0"))
    retrieve_rerank_overlap_weight: float = float(os.getenv("RETRIEVE_RERANK_OVERLAP_WEIGHT", "0.20"))
    retrieve_rerank_vector_weight: float = float(os.getenv("RETRIEVE_RERANK_VECTOR_WEIGHT", "0.50"))
    retrieve_rerank_source_weight: float = float(os.getenv("RETRIEVE_RERANK_SOURCE_WEIGHT", "0.50"))
    retrieve_enable_query_source_routing: bool = os.getenv("RETRIEVE_ENABLE_QUERY_SOURCE_ROUTING", "true").lower() == "true"
    retrieve_domain_min_items: int = int(os.getenv("RETRIEVE_DOMAIN_MIN_ITEMS", "3"))
    retrieve_min_text_chars: int = int(os.getenv("RETRIEVE_MIN_TEXT_CHARS", "120"))
    retrieve_min_informative_tokens: int = int(os.getenv("RETRIEVE_MIN_INFORMATIVE_TOKENS", "8"))
    retrieve_max_heading_token_ratio: float = float(os.getenv("RETRIEVE_MAX_HEADING_TOKEN_RATIO", "0.70"))
    retrieve_drop_table_stubs: bool = os.getenv("RETRIEVE_DROP_TABLE_STUBS", "true").lower() == "true"
    retrieve_modality_bonus_text: float = float(os.getenv("RETRIEVE_MODALITY_BONUS_TEXT", "0.08"))
    retrieve_modality_bonus_table: float = float(os.getenv("RETRIEVE_MODALITY_BONUS_TABLE", "0.0"))
    min_text_context_items: int = int(os.getenv("MIN_TEXT_CONTEXT_ITEMS", "2"))
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "25000"))

    answer_max_tokens: int = int(os.getenv("ANSWER_MAX_TOKENS", "700"))
    answer_temperature: float = float(os.getenv("ANSWER_TEMPERATURE", "0.0"))
    require_citations_per_answer: bool = os.getenv("REQUIRE_CITATIONS_PER_ANSWER", "true").lower() == "true"

    data_dir: Path = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
    processed_dir: Path = PROJECT_ROOT / os.getenv("PROCESSED_DIR", "data/processed")

    @property
    def images_dir(self) -> Path:
        return self.processed_dir / "assets" / "images"

    @property
    def manifests_dir(self) -> Path:
        return self.processed_dir / "manifests"

    @property
    def manifest_path(self) -> Path:
        return self.manifests_dir / "records.jsonl"


settings = Settings()


def ensure_local_dirs() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    settings.manifests_dir.mkdir(parents=True, exist_ok=True)
