from __future__ import annotations

import argparse
import json
from pathlib import Path

from pinecone import Pinecone

from src.config import ensure_local_dirs, settings
from src.evaluation.evaluator import RAGEvaluator
from src.ingestion.indexer import IngestionService
from src.pipeline import RagPipeline


def cmd_ingest(args: argparse.Namespace) -> None:
    ensure_local_dirs()
    service = IngestionService()
    data_dir = Path(args.data_dir) if args.data_dir else settings.data_dir
    summary = service.run(data_dir=data_dir, sync_index=args.sync_index)
    print(json.dumps(summary, indent=2))


def cmd_query(args: argparse.Namespace) -> None:
    pipeline = RagPipeline()
    result = pipeline.ask(args.query, top_k=args.top_k)
    print(json.dumps(result, indent=2))


def cmd_status(_: argparse.Namespace) -> None:
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing in environment.")
    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index)
    stats = index.describe_index_stats()
    out = {
        "index": settings.pinecone_index,
        "dimension": settings.pinecone_dimension,
        "metric": settings.pinecone_metric,
        "total_vector_count": stats.get("total_vector_count", 0),
    }
    print(json.dumps(out, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    evaluator = RAGEvaluator()
    summary = evaluator.run(
        dataset_path=Path(args.dataset),
        top_k=args.top_k,
        max_cases=args.max_cases,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Multimodal RAG CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Extract PDFs and upsert into Pinecone")
    ingest.add_argument("--data-dir", type=str, default=None, help="Path to PDF directory")
    ingest.add_argument(
        "--sync-index",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Delete stale vectors not present in the latest manifest (default from SYNC_INDEX_ON_INGEST).",
    )
    ingest.set_defaults(func=cmd_ingest)

    query = sub.add_parser("query", help="Ask a question")
    query.add_argument("query", type=str, help="User question")
    query.add_argument("--top-k", type=int, default=None, help="Retriever top-k")
    query.set_defaults(func=cmd_query)

    status = sub.add_parser("status", help="Show Pinecone index status")
    status.set_defaults(func=cmd_status)

    evaluate = sub.add_parser("evaluate", help="Run dataset-based RAG evaluation")
    evaluate.add_argument("--dataset", type=str, required=True, help="Path to .jsonl/.json evaluation dataset")
    evaluate.add_argument("--top-k", type=int, default=None, help="Retriever top-k")
    evaluate.add_argument("--max-cases", type=int, default=None, help="Limit number of evaluation cases")
    evaluate.add_argument("--output-dir", type=str, default="logs/evaluation", help="Directory for eval reports")
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
