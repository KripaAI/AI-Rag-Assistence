from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from src.config import settings
from src.pipeline import RagPipeline


@dataclass
class EvalCase:
    query: str
    ground_truth_answer: str = ""
    expected_sources: list[str] | None = None


def load_eval_cases(path: Path, max_cases: int | None = None) -> list[EvalCase]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    cases: list[EvalCase] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                cases.append(
                    EvalCase(
                        query=str(raw.get("query", "")).strip(),
                        ground_truth_answer=str(raw.get("ground_truth_answer", "")).strip(),
                        expected_sources=[str(s) for s in (raw.get("expected_sources") or [])],
                    )
                )
    else:
        raw_items = json.loads(path.read_text(encoding="utf-8"))
        for raw in raw_items:
            cases.append(
                EvalCase(
                    query=str(raw.get("query", "")).strip(),
                    ground_truth_answer=str(raw.get("ground_truth_answer", "")).strip(),
                    expected_sources=[str(s) for s in (raw.get("expected_sources") or [])],
                )
            )

    cases = [c for c in cases if c.query]
    if max_cases is not None:
        return cases[:max_cases]
    return cases


def _normalize_source_name(name: str) -> str:
    return Path(name).name.strip().lower()


def compute_retrieval_metrics(result: dict, expected_sources: list[str] | None) -> dict[str, Any]:
    retrieved = result.get("retrieved", [])
    retrieved_sources = [_normalize_source_name(r.get("source_file", "")) for r in retrieved]

    metrics: dict[str, Any] = {
        "retrieved_count": len(retrieved),
        "retrieval_hit": None,
        "retrieval_recall": None,
        "retrieval_mrr": None,
    }

    if not expected_sources:
        return metrics

    expected = {_normalize_source_name(s) for s in expected_sources if str(s).strip()}
    if not expected:
        return metrics

    found_positions: list[int] = []
    found_set = set()
    for idx, src in enumerate(retrieved_sources, start=1):
        if src in expected:
            found_set.add(src)
            if not found_positions:
                found_positions.append(idx)

    metrics["retrieval_hit"] = 1.0 if found_set else 0.0
    metrics["retrieval_recall"] = len(found_set) / max(len(expected), 1)
    metrics["retrieval_mrr"] = (1.0 / found_positions[0]) if found_positions else 0.0
    return metrics


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r.get(key) for r in rows if isinstance(r.get(key), (float, int))]
    if not vals:
        return None
    return float(mean(vals))


class RAGEvaluator:
    def __init__(self) -> None:
        self.pipeline = RagPipeline()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing in environment.")

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
        except Exception as exc:
            raise RuntimeError(
                "RAGAS or Google GenAI dependencies are missing. Install requirements.txt before running evaluation."
            ) from exc

        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_chat_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=settings.gemini_api_key,
        )

        self.ragas_llm = LangchainLLMWrapper(llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    def _run_ragas(self, ragas_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from ragas.run_config import RunConfig

        data = {
            "question": [r["question"] for r in ragas_records],
            "answer": [r["answer"] for r in ragas_records],
            "contexts": [r["contexts"] for r in ragas_records],
            "ground_truth": [r["ground_truth"] for r in ragas_records],
        }
        dataset = Dataset.from_dict(data)

        has_ground_truth = any(str(x).strip() for x in data["ground_truth"])
        metrics = [faithfulness, answer_relevancy, context_precision]
        if has_ground_truth:
            metrics.append(context_recall)

        # Optimization: Enable parallel execution and set batch size
        run_config = RunConfig(max_workers=8, timeout=60)

        result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings,
            run_config=run_config,
        )

        df = result.to_pandas()
        rows = []
        for _, row in df.iterrows():
            rows.append(
                {
                    "faithfulness": _to_float(row.get("faithfulness")),
                    "answer_relevancy": _to_float(row.get("answer_relevancy")),
                    "context_precision": _to_float(row.get("context_precision")),
                    "context_recall": _to_float(row.get("context_recall")),
                }
            )
        return rows

    def run(self, dataset_path: Path, top_k: int | None = None, max_cases: int | None = None, output_dir: Path | None = None) -> dict[str, Any]:
        cases = load_eval_cases(dataset_path, max_cases=max_cases)
        if not cases:
            raise RuntimeError("No valid evaluation cases found.")

        out_dir = output_dir or (Path("logs") / "evaluation")
        out_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        ragas_records: list[dict[str, Any]] = []

        for i, case in enumerate(cases, start=1):
            result = self.pipeline.ask(case.query, top_k=top_k)
            retrieval_metrics = compute_retrieval_metrics(result, case.expected_sources)

            row = {
                "case_id": i,
                "query": case.query,
                "answer_text": result.get("answer_text", ""),
                "expected_sources": " | ".join(case.expected_sources or []),
                "retrieved_count": retrieval_metrics.get("retrieved_count"),
                "retrieval_hit": retrieval_metrics.get("retrieval_hit"),
                "retrieval_recall": retrieval_metrics.get("retrieval_recall"),
                "retrieval_mrr": retrieval_metrics.get("retrieval_mrr"),
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
            }
            rows.append(row)

            ragas_records.append(
                {
                    "question": case.query,
                    "answer": result.get("answer_text", ""),
                    "contexts": result.get("contexts", []),
                    "ground_truth": case.ground_truth_answer,
                }
            )

        ragas_scores = self._run_ragas(ragas_records)
        for row, score in zip(rows, ragas_scores):
            row.update(score)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        csv_path = out_dir / f"eval_{ts}.csv"
        json_path = out_dir / f"eval_{ts}.json"

        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        summary = {
            "method": "official_ragas",
            "cases": len(rows),
            "avg_faithfulness": _avg(rows, "faithfulness"),
            "avg_answer_relevancy": _avg(rows, "answer_relevancy"),
            "avg_context_precision": _avg(rows, "context_precision"),
            "avg_context_recall": _avg(rows, "context_recall"),
            "avg_retrieval_hit": _avg(rows, "retrieval_hit"),
            "avg_retrieval_recall": _avg(rows, "retrieval_recall"),
            "avg_retrieval_mrr": _avg(rows, "retrieval_mrr"),
            "csv_path": str(csv_path.resolve()),
            "json_path": str(json_path.resolve()),
            "dataset": str(dataset_path.resolve()),
        }

        json_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
        return summary


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        n = float(value)
        return max(0.0, min(1.0, n))
    except Exception:
        return None
