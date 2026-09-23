from __future__ import annotations

import unittest
from pathlib import Path

from common.io import read_jsonl
from common.models import BenchmarkCase
from cot.evaluators import evaluate_all as evaluate_cot
from inner_model_structure.evaluators import evaluate_all as evaluate_structure


ROOT = Path(__file__).resolve().parents[1]


class SchemaTests(unittest.TestCase):
    def test_benchmark_cases_load(self) -> None:
        records = read_jsonl(ROOT / "data" / "sample_benchmark.jsonl")
        cases = [BenchmarkCase.from_dict(record) for record in records]
        self.assertEqual(len(cases), 2)
        self.assertTrue(all(case.id and case.prompt for case in cases))


class EvaluationTests(unittest.TestCase):
    def test_all_cot_metrics_run(self) -> None:
        records = read_jsonl(ROOT / "data" / "sample_cot_artifacts.jsonl")
        results = evaluate_cot(records)
        self.assertEqual(len(results), 12)
        self.assertEqual({result.name for result in results}, {
            "correctness",
            "completeness",
            "consistency",
            "continuity",
            "contrastivity",
            "covariate_complexity",
            "compactness",
            "composition",
            "confidence",
            "context",
            "coherence",
            "controllability",
        })
        self.assertTrue(all(result.status == "ok" for result in results))

    def test_all_structure_metrics_run(self) -> None:
        records = read_jsonl(ROOT / "data" / "sample_structure_artifacts.jsonl")
        results = evaluate_structure(records)
        self.assertEqual(len(results), 12)
        self.assertTrue(all(result.status == "ok" for result in results))

    def test_missing_evidence_is_not_available(self) -> None:
        results = evaluate_structure([{"id": "empty"}])
        self.assertTrue(all(result.status == "not_available" for result in results))


if __name__ == "__main__":
    unittest.main()
