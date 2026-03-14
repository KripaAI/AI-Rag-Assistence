import json

from src.evaluation.evaluator import compute_retrieval_metrics, load_eval_cases


def test_load_eval_cases_jsonl(tmp_path):
    p = tmp_path / "cases.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"query": "q1", "ground_truth_answer": "a1", "expected_sources": ["doc.pdf"]}),
                json.dumps({"query": "q2"}),
            ]
        ),
        encoding="utf-8",
    )

    cases = load_eval_cases(p)
    assert len(cases) == 2
    assert cases[0].query == "q1"
    assert cases[0].expected_sources == ["doc.pdf"]


def test_compute_retrieval_metrics_with_expected_sources():
    result = {
        "retrieved": [
            {"source_file": "a.pdf"},
            {"source_file": "target.pdf"},
        ]
    }
    metrics = compute_retrieval_metrics(result, ["target.pdf"])
    assert metrics["retrieval_hit"] == 1.0
    assert metrics["retrieval_recall"] == 1.0
    assert metrics["retrieval_mrr"] == 0.5
