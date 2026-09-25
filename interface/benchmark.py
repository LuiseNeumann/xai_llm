"""Benchmarkläufe aus dem UI oder der Kommandozeile starten und auswerten."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.io import write_json, write_jsonl
from common.models import BenchmarkCase
from cot.evaluators import evaluate_all as evaluate_cot
from inner_model_structure.evaluators import evaluate_all as evaluate_structure
from interface.assessment import build_assessment
from interface.pipeline_service import (
    AnalysisConfig,
    AnalysisRequest,
    TextGenerator,
    analyze_artifacts,
    analyze_prompt,
    evaluate_live_cot,
)


BenchmarkProgress = Callable[[float, str], None]


def missing_cot_evaluation() -> dict[str, Any]:
    """Bewahrt das CoT-Profil auch bei fehlgeschlagenen Modellläufen."""
    metrics = [result.to_dict() for result in evaluate_cot([{}])]
    return {
        "examples": 0,
        "metrics": metrics,
        "assessment": build_assessment(metrics).to_dict(),
    }


def missing_structure_evaluation() -> dict[str, Any]:
    """Zeigt alle zwölf Struktur-Cs ohne erfundene Werte an."""
    metrics = [result.to_dict() for result in evaluate_structure([{}])]
    return {
        "examples": 0,
        "metrics": metrics,
        "assessment": build_assessment(metrics).to_dict(),
    }


def run_benchmark(
    model_service: TextGenerator,
    cases: Sequence[BenchmarkCase],
    config: AnalysisConfig,
    model_name: str,
    progress: BenchmarkProgress | None = None,
) -> dict[str, Any]:
    """Erzeugt pro Fall Artefakte und fasst Accuracy und Co-12-Ergebnisse zusammen."""
    if not cases:
        raise ValueError("Der gewählte Benchmark enthält keine Fälle.")
    results: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    structure_artifacts: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if progress:
            progress(index / len(cases), f"Fall {index + 1}/{len(cases)}: {case.id}")
        request = AnalysisRequest(
            prompt=case.prompt,
            expected_answer=case.expected_answer,
            alternative_answer=case.alternative_answer,
            corrupted_prompt=case.contrast_prompt or case.corrupted_prompt,
            gold_steps=case.gold_steps,
            gold_concepts=case.gold_concepts,
            max_steps=case.constraints.get("max_steps"),
            required_terms=case.constraints.get("required_terms", []),
            forbidden_terms=case.constraints.get("forbidden_terms", []),
        )
        try:
            report = analyze_prompt(model_service, request, config, model_name)
        except (RuntimeError, ValueError, KeyError) as exc:
            results.append(
                {
                    "id": case.id,
                    "prompt": case.prompt,
                    "expected_answer": case.expected_answer,
                    "status": "error",
                    "error": str(exc),
                }
            )
            continue
        artifacts.append({**report["cot"]["artifact"], "id": case.id})
        if report.get("structure") and report["structure"].get("artifact"):
            structure_artifacts.append({**report["structure"]["artifact"], "id": case.id})
        results.append(
            {
                "id": case.id,
                "prompt": case.prompt,
                "expected_answer": case.expected_answer,
                "answer": report["answer"],
                "answer_matches_reference": report["answer_matches_reference"],
                "metadata": case.metadata,
                "status": "ok" if report["answer"] else "unparsed",
                "report": report,
            }
        )
    measured = [result for result in results if result["status"] == "ok"]
    correct = sum(result["answer_matches_reference"] is True for result in measured)
    group_breakdown: dict[str, dict[str, dict[str, float | int]]] = {}
    for field in ("context_condition", "question_polarity"):
        groups: dict[str, dict[str, float | int]] = {}
        for result in measured:
            group = result.get("metadata", {}).get(field)
            if not group:
                continue
            stats = groups.setdefault(str(group), {"cases": 0, "correct": 0, "accuracy": 0.0})
            stats["cases"] += 1
            stats["correct"] += int(result["answer_matches_reference"] is True)
        for stats in groups.values():
            stats["accuracy"] = stats["correct"] / stats["cases"]
        if groups:
            group_breakdown[field] = groups
    evaluation = analyze_artifacts(
        cot_records=artifacts or None,
        structure_records=structure_artifacts or None,
    )
    if evaluation:
        if artifacts:
            metrics = evaluate_live_cot(artifacts)
            evaluation["cot"]["metrics"] = metrics
            evaluation["cot"]["assessment"] = build_assessment(metrics).to_dict()
        else:
            evaluation["cot"] = missing_cot_evaluation()
            metrics = evaluation["cot"]["metrics"]
        if not evaluation["structure"]:
            evaluation["structure"] = missing_structure_evaluation()
        evaluation["overall_assessment"] = build_assessment(
            metrics + evaluation["structure"]["metrics"],
            expected_metrics=24,
        ).to_dict()
    if progress:
        progress(1.0, "Benchmark abgeschlossen")
    return {
        "model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "mode": config.mode,
            "samples": config.samples,
            "temperature": config.temperature,
            "run_structure": config.run_structure,
        },
        "benchmark_is_example_data": all(case.id in {"addition-001", "logic-001"} for case in cases),
        "total_cases": len(cases),
        "evaluated_cases": len(measured),
        "correct_answers": correct,
        "accuracy": correct / len(measured) if measured else None,
        "dataset": cases[0].metadata.get("dataset", "Eigene oder synthetische Beispiele"),
        "group_breakdown": group_breakdown,
        "results": results,
        "evaluation": evaluation,
    }


def save_benchmark(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Speichert Gesamtbericht und CoT-Artefakte für die Offline-Auswertung."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    report_path = directory / f"benchmark_{stamp}.json"
    artifacts_path = directory / f"benchmark_{stamp}_cot.jsonl"
    write_json(report_path, report)
    write_jsonl(
        artifacts_path,
        (
            {**result["report"]["cot"]["artifact"], "id": result["id"]}
            for result in report["results"]
            if "report" in result
        ),
    )
    return report_path, artifacts_path


def load_benchmark(path: str | Path) -> dict[str, Any]:
    """Lädt einen früher gespeicherten Bericht für die Ergebnisanzeige."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
