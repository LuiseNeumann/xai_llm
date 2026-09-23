from __future__ import annotations

import ast
import unittest
from pathlib import Path
from typing import Any

from interface.assessment import build_assessment
from interface.pipeline_service import AnalysisConfig, AnalysisRequest, analyze_prompt


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
        self.assertGreaterEqual(report["overall_assessment"]["available_metrics"], 8)

    def test_streamlit_source_is_valid_python(self) -> None:
        source = (ROOT / "interface" / "app.py").read_text(encoding="utf-8")
        ast.parse(source)


if __name__ == "__main__":
    unittest.main()
