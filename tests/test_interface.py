from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from typing import Any

from interface.assessment import build_assessment
from interface.benchmark import load_benchmark, run_benchmark, save_benchmark
from interface.pipeline_service import AnalysisConfig, AnalysisRequest, analyze_prompt
from common.models import BenchmarkCase


ROOT = Path(__file__).resolve().parents[1]


class FakeGenerator:
    def generate(self, prompt: str, seed: int, temperature: float) -> dict[str, Any]:
        if "No glims" in prompt:
            return {
                "answer": "no",
                "steps": [
                    "No glims are blue.",
                    "Ria is a glim.",
                    "Therefore Ria is not blue.",
                ],
                "step_confidences": [0.95, 0.95, 0.95],
            }
        return {
            "answer": "yes",
            "steps": [
                "All glims are blue.",
                "Ria is a glim.",
                "Therefore Ria is blue.",
            ],
            "step_confidences": [0.95, 0.95, 0.95],
        }


class FakeStructureCollector:
    def collect(self, case: BenchmarkCase, top_k: int) -> dict[str, Any]:
        return {
            "full_score": 2.0,
            "circuit_score": 1.8,
            "total_nodes": 10,
            "nodes": ["layer_1"],
        }


class FakeGeneratorWithStructure(FakeGenerator):
    structure_collector = FakeStructureCollector()


def metric(name: str, score: float | None, status: str = "ok") -> dict[str, Any]:
    return {
        "name": name,
        "score": score,
        "higher_is_better": True,
        "details": {},
        "status": status,
    }


class AssessmentTests(unittest.TestCase):
    def test_low_evidence_stays_yellow(self) -> None:
        assessment = build_assessment([metric("consistency", 0.95)])
        self.assertEqual(assessment.color, "yellow")
        self.assertEqual(assessment.evidence_level, "niedrig")

    def test_critical_failure_is_red(self) -> None:
        metrics = [metric("correctness", 0.1)] + [
            metric(name, 0.9)
            for name in (
                "completeness",
                "consistency",
                "continuity",
                "contrastivity",
                "composition",
                "confidence",
                "context",
                "coherence",
            )
        ]
        assessment = build_assessment(metrics)
        self.assertEqual(assessment.color, "red")

    def test_complete_strong_evidence_can_be_green(self) -> None:
        names = (
            "correctness",
            "completeness",
            "consistency",
            "continuity",
            "contrastivity",
            "composition",
            "confidence",
            "context",
            "coherence",
            "controllability",
            "compactness",
            "covariate_complexity",
        )
        metrics = [metric(name, 0.9) for name in names]
        assessment = build_assessment(metrics)
        self.assertEqual(assessment.color, "green")
        self.assertEqual(assessment.evidence_level, "hoch")

    def test_weak_proxy_does_not_trigger_red(self) -> None:
        proxy = metric("coherence", 0.1)
        proxy["details"]["assessment_eligible"] = False
        assessment = build_assessment([proxy, metric("contrastivity", 0.95)])
        self.assertEqual(assessment.color, "yellow")
        self.assertEqual(assessment.available_metrics, 1)


class LivePipelineTests(unittest.TestCase):
    def test_live_pipeline_without_model_download(self) -> None:
        request = AnalysisRequest(
            prompt="All glims are blue. Ria is a glim. Is Ria blue?",
            expected_answer="yes",
            alternative_answer="no",
            corrupted_prompt="No glims are blue. Ria is a glim. Is Ria blue?",
            gold_steps=[
                "All glims are blue.",
                "Ria is a glim.",
                "Therefore Ria is blue.",
            ],
            gold_concepts=["glims", "blue"],
            max_steps=3,
            required_terms=["therefore"],
        )
        config = AnalysisConfig(mode="schnell", samples=2, run_structure=False)
        report = analyze_prompt(FakeGenerator(), request, config, model_name="fake")

        self.assertEqual(report["answer"], "yes")
        self.assertTrue(report["answer_matches_reference"])
        self.assertEqual(len(report["cot"]["metrics"]), 12)
        self.assertIsNone(report["structure"])
        self.assertEqual(report["overall_assessment"]["color"], "yellow")
        self.assertLess(report["overall_assessment"]["available_metrics"], 8)

    def test_streamlit_source_is_valid_python(self) -> None:
        source = (ROOT / "interface" / "app.py").read_text(encoding="utf-8")
        ast.parse(source)

    def test_benchmark_can_be_saved_and_loaded(self) -> None:
        cases = [
            BenchmarkCase(
                id="logic-test",
                prompt="All glims are blue. Ria is a glim. Is Ria blue?",
                expected_answer="yes",
                alternative_answer="no",
                contrast_prompt="No glims are blue. Ria is a glim. Is Ria blue?",
            )
        ]
        report = run_benchmark(
            FakeGenerator(),
            cases,
            AnalysisConfig(samples=1),
            "testmodell",
        )
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(report["evaluated_cases"], 1)
        self.assertEqual(len(report["evaluation"]["cot"]["metrics"]), 12)
        self.assertEqual(len(report["evaluation"]["structure"]["metrics"]), 12)
        self.assertEqual(report["evaluation"]["structure"]["examples"], 0)
        self.assertTrue(
            all(result["status"] == "not_available" for result in report["evaluation"]["structure"]["metrics"])
        )
        with tempfile.TemporaryDirectory() as directory:
            report_path, artifacts_path = save_benchmark(report, directory)
            self.assertTrue(artifacts_path.exists())
            self.assertEqual(load_benchmark(report_path)["accuracy"], 1.0)

    def test_benchmark_uses_collected_structure_artifacts(self) -> None:
        case = BenchmarkCase(
            id="logic-structure",
            prompt="All glims are blue. Ria is a glim. Is Ria blue?",
            expected_answer="yes",
            alternative_answer="no",
            corrupted_prompt="No glims are blue. Ria is a glim. Is Ria blue?",
        )
        report = run_benchmark(
            FakeGeneratorWithStructure(),
            [case],
            AnalysisConfig(samples=1, run_structure=True),
            "testmodell",
        )
        structure = report["evaluation"]["structure"]
        self.assertEqual(structure["examples"], 1)
        self.assertEqual(len(structure["metrics"]), 12)
        self.assertEqual(structure["metrics"][0]["status"], "ok")
        self.assertEqual(structure["metrics"][1]["status"], "not_available")


if __name__ == "__main__":
    unittest.main()
