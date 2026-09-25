"""Stellt sicher, dass jedes C ein branch-spezifisches Testverfahren erklärt."""

from __future__ import annotations

import unittest

from cot.evaluators import evaluate_all as evaluate_cot
from inner_model_structure.evaluators import evaluate_all as evaluate_structure
from interface.method_explanations import COT_METHODS, STRUCTURE_METHODS, explanation_for


class MethodExplanationTests(unittest.TestCase):
    def test_all_metric_names_are_documented_in_both_branches(self) -> None:
        cot_names = {result.name for result in evaluate_cot([{}])}
        structure_names = {result.name for result in evaluate_structure([{}])}
        self.assertEqual(set(COT_METHODS), cot_names)
        self.assertEqual(set(STRUCTURE_METHODS), structure_names)
        for method in (*COT_METHODS.values(), *STRUCTURE_METHODS.values()):
            self.assertTrue(all((method.procedure, method.calculation, method.required, method.limitation)))

    def test_unavailable_structure_explains_cause_and_requirement(self) -> None:
        result = {"name": "correctness", "score": None, "status": "not_available", "details": {
            "reason": "Werte für Gesamtmodell und Circuit werden benötigt"
        }}
        method, status = explanation_for(
            "correctness", "structure", result,
            "LM Studio gibt keine internen Aktivierungen aus.",
        )
        self.assertIn("LM Studio", status)
        self.assertIn("Gesamtmodell und Circuit", status)
        self.assertIn("circuit_score", method.required)

    def test_cot_proxy_explains_why_it_does_not_enter_ampel(self) -> None:
        result = {"name": "coherence", "score": 0.3, "status": "ok", "details": {
            "assessment_eligible": False,
            "assessment_reason": "Token-F1 ist sprachübergreifend unzuverlässig.",
        }}
        method, status = explanation_for("coherence", "cot", result)
        self.assertIn("nicht in die Prüfampel", status)
        self.assertIn("sprachübergreifend", status)
        self.assertIn("Referenzschritten", method.procedure)


if __name__ == "__main__":
    unittest.main()
