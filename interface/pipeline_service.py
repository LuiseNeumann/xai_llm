"""Anwendungslogik für interaktive Prompt- und Artefaktauswertungen."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from itertools import combinations
from typing import Any, Protocol

from common.metrics import mean, token_f1
from common.models import BenchmarkCase
from cot.collect_hf import HuggingFaceGenerator
from cot.evaluators import evaluate_all as evaluate_cot
from inner_model_structure.collect_hf import LayerPatchingCollector
from inner_model_structure.evaluators import evaluate_all as evaluate_structure
from interface.assessment import build_assessment


ProgressCallback = Callable[[float, str], None]


class TextGenerator(Protocol):
    def generate(self, prompt: str, seed: int, temperature: float) -> dict[str, Any]: ...


@dataclass(slots=True)
class AnalysisRequest:
    prompt: str
    expected_answer: str = ""
    alternative_answer: str = ""
    corrupted_prompt: str = ""
    gold_steps: list[str] = field(default_factory=list)
    gold_concepts: list[str] = field(default_factory=list)
    max_steps: int | None = None
    required_terms: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnalysisConfig:
    mode: str = "schnell"
    samples: int = 3
    temperature: float = 0.7
    seed: int = 42
    run_structure: bool = False
    structure_top_k: int = 5


class LocalModelService:
    """Hält ein lokales Modell für Textgenerierung und Patching gemeinsam im Speicher."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 384,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.generator = HuggingFaceGenerator(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        self.structure_collector = LayerPatchingCollector(
            model_name=model_name,
            model=self.generator.model,
            tokenizer=self.generator.tokenizer,
        )

    def generate(self, prompt: str, seed: int, temperature: float) -> dict[str, Any]:
        return self.generator.generate(prompt, seed, temperature)


def _notify(callback: ProgressCallback | None, progress: float, message: str) -> None:
    if callback:
        callback(progress, message)


def _normal_answer(value: Any) -> str:
    return str(value).strip().casefold()


def _trace_text(trace: dict[str, Any]) -> str:
    return " ".join(str(step) for step in trace.get("steps", []))


def _composition_scores(trace: dict[str, Any]) -> dict[str, float]:
    """Liefert transparente Form-Proxys, aber keine semantische Korrektheitsprüfung."""
    steps = [str(step).strip() for step in trace.get("steps", []) if str(step).strip()]
    if not steps:
        return {"struktur": 0.0, "nicht_redundant": 0.0}
    pairs = list(combinations(steps, 2))
    redundancy = mean(token_f1(left, right) for left, right in pairs) if pairs else 0.0
    complete_sentences = mean(step[-1:] in {".", "!", "?"} for step in steps)
    return {
        "struktur": mean([1.0, complete_sentences]),
        "nicht_redundant": max(0.0, 1.0 - redundancy),
    }


def _constraints(request: AnalysisRequest) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    if request.max_steps is not None:
        constraints["max_steps"] = request.max_steps
    if request.required_terms:
        constraints["required_terms"] = request.required_terms
    if request.forbidden_terms:
        constraints["forbidden_terms"] = request.forbidden_terms
    return constraints


def _wrapper_prompts(prompt: str, full_mode: bool) -> list[str]:
    variants = [f"Bearbeite die folgende Aufgabe inhaltlich unverändert:\n\n{prompt}"]
    if full_mode:
        variants.append(
            "Ignoriere ausschließlich unwichtige Unterschiede in der Formulierung und löse dieselbe Aufgabe:\n\n"
            + prompt
        )
    return variants


def _intervention_prompts(prompt: str, full_mode: bool) -> list[tuple[str, str]]:
    variants = [
        (
            "ausgabeformat",
            prompt + "\n\nGib die Erklärung besonders übersichtlich aus; der Inhalt der Aufgabe bleibt gleich.",
        )
    ]
    if full_mode:
        variants.append(
            (
                "irrelevanter_hinweis",
                "Irrelevanter Hinweis: Die Oberfläche verwendet eine dunkelblaue Akzentfarbe.\n\n"
                + prompt,
            )
        )
    return variants


def collect_live_cot(
    generator: TextGenerator,
    request: AnalysisRequest,
    config: AnalysisConfig,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Erzeugt die für eine direkte CoT-Auswertung verfügbaren Artefakte."""
    full_mode = config.mode == "vollständig"
    _notify(progress, 0.05, "Modellantwort wird erzeugt")
    original = generator.generate(request.prompt, config.seed, temperature=0.0)
    artifact: dict[str, Any] = {
        "id": "live-prompt",
        **original,
        "expected_answer": request.expected_answer,
        "gold_steps": request.gold_steps,
        "gold_concepts": request.gold_concepts,
        "constraints": _constraints(request),
        "sampled_traces": [],
        "paraphrase_traces": [],
        "interventions": [],
        "composition_scores": _composition_scores(original),
        "metadata": {
            "mode": config.mode,
            "composition_is_proxy": True,
            "continuity_variants": "bedeutungserhaltende Instruktionsrahmen",
        },
    }

    trace_text = _trace_text(original).casefold()
    found_concepts = [
        concept for concept in request.gold_concepts if concept.casefold() in trace_text
    ]
    if found_concepts:
        artifact["concepts"] = found_concepts

    sample_count = max(1, config.samples)
    for index in range(sample_count):
        progress_value = 0.10 + 0.30 * (index + 1) / sample_count
        _notify(progress, progress_value, f"Reasoning-Sample {index + 1} von {sample_count}")
        artifact["sampled_traces"].append(
            generator.generate(
                request.prompt,
                config.seed + index + 1,
                temperature=config.temperature,
            )
        )

    wrappers = _wrapper_prompts(request.prompt, full_mode)
    for index, prompt in enumerate(wrappers):
        _notify(progress, 0.45 + 0.10 * (index + 1) / len(wrappers), "Continuity wird geprüft")
        artifact["paraphrase_traces"].append(
            generator.generate(prompt, config.seed + 100 + index, temperature=0.0)
        )

    interventions = _intervention_prompts(request.prompt, full_mode)
    for index, (name, prompt) in enumerate(interventions):
        _notify(progress, 0.58 + 0.12 * (index + 1) / len(interventions), "Correctness-Proxy wird geprüft")
        result = generator.generate(prompt, config.seed + 200 + index, temperature=0.0)
        artifact["interventions"].append(
            {
                "name": name,
                "answer": result.get("answer", ""),
                "steps": result.get("steps", []),
                "expected_answer_change": False,
            }
        )

    if request.corrupted_prompt:
        _notify(progress, 0.74, "Kontrastive Eingabe wird ausgewertet")
        contrast = generator.generate(
            request.corrupted_prompt,
            config.seed + 300,
            temperature=0.0,
        )
        artifact["contrast"] = {
            **contrast,
            "expected_answer": request.alternative_answer,
        }
    return artifact


def _scoring_prompt(prompt: str) -> str:
    return prompt.rstrip() + "\n\nGib jetzt nur die kurze Endantwort aus.\nAntwort:"


def collect_live_structure(
    collector: LayerPatchingCollector,
    request: AnalysisRequest,
    config: AnalysisConfig,
    preferred_answer: str,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Führt eine optionale Layer-Patching-Diagnose mit expliziten Gegenfakten aus."""
    if not request.corrupted_prompt or not request.alternative_answer:
        return None, (
            "Für die Strukturanalyse werden eine Alternativantwort und ein "
            "kontrafaktischer Prompt benötigt."
        )
    if not preferred_answer:
        return None, "Es konnte keine bevorzugte Antwort für die Logit-Differenz bestimmt werden."

    _notify(progress, 0.78, "Aktivierungen des kontrafaktischen Prompts werden gesammelt")
    case = BenchmarkCase(
        id="live-structure",
        prompt=_scoring_prompt(request.prompt),
        expected_answer=preferred_answer,
        alternative_answer=request.alternative_answer,
        corrupted_prompt=_scoring_prompt(request.corrupted_prompt),
    )
    artifact = collector.collect(case, config.structure_top_k)

    if config.mode == "vollständig":
        _notify(progress, 0.90, "Continuity des Layer-Rankings wird geprüft")
        variant = BenchmarkCase(
            id="live-structure-variant",
            prompt=_scoring_prompt(_wrapper_prompts(request.prompt, False)[0]),
            expected_answer=preferred_answer,
            alternative_answer=request.alternative_answer,
            corrupted_prompt=_scoring_prompt(
                _wrapper_prompts(request.corrupted_prompt, False)[0]
            ),
        )
        variant_artifact = collector.collect(variant, config.structure_top_k)
        artifact["perturbed_node_sets"] = [
            artifact.get("nodes", []),
            variant_artifact.get("nodes", []),
        ]
    return artifact, None


def analyze_prompt(
    model_service: TextGenerator,
    request: AnalysisRequest,
    config: AnalysisConfig,
    model_name: str = "",
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Führt die Live-Analyse aus und gibt einen vollständig serialisierbaren Bericht zurück."""
    cot_artifact = collect_live_cot(model_service, request, config, progress)
    cot_metrics = [result.to_dict() for result in evaluate_cot([cot_artifact])]
    answer = str(cot_artifact.get("answer", ""))
    answer_match = (
        _normal_answer(answer) == _normal_answer(request.expected_answer)
        if request.expected_answer
        else None
    )
    cot_assessment = build_assessment(cot_metrics, answer_matches_reference=answer_match)

    structure_artifact: dict[str, Any] | None = None
    structure_note: str | None = None
    structure_metrics: list[dict[str, Any]] = []
    structure_assessment: dict[str, Any] | None = None
    if config.run_structure:
        collector = getattr(model_service, "structure_collector", None)
        if collector is None:
            structure_note = "Der verwendete Modellservice unterstützt kein Aktivierungs-Patching."
        else:
            preferred = request.expected_answer or answer
            structure_artifact, structure_note = collect_live_structure(
                collector,
                request,
                config,
                preferred,
                progress,
            )
        structure_metrics = [
            result.to_dict()
            for result in evaluate_structure([structure_artifact] if structure_artifact else [{}])
        ]
        structure_assessment = build_assessment(structure_metrics).to_dict()

    combined_metrics = cot_metrics + structure_metrics
    overall_expected = 24 if config.run_structure else 12
    overall = build_assessment(
        combined_metrics,
        expected_metrics=overall_expected,
        answer_matches_reference=answer_match,
    )
    _notify(progress, 1.0, "Auswertung abgeschlossen")
    return {
        "model": model_name,
        "request": asdict(request),
        "config": asdict(config),
        "answer": answer,
        "steps": cot_artifact.get("steps", []),
        "answer_matches_reference": answer_match,
        "overall_assessment": overall.to_dict(),
        "cot": {
            "assessment": cot_assessment.to_dict(),
            "metrics": cot_metrics,
            "artifact": cot_artifact,
        },
        "structure": {
            "assessment": structure_assessment,
            "metrics": structure_metrics,
            "artifact": structure_artifact,
            "note": structure_note,
        }
        if config.run_structure
        else None,
    }


def analyze_artifacts(
    cot_records: list[dict[str, Any]] | None = None,
    structure_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Wertet hochgeladene JSONL-Artefakte ohne erneuten Modelllauf aus."""
    report: dict[str, Any] = {"cot": None, "structure": None}
    combined: list[dict[str, Any]] = []
    expected = 0
    if cot_records:
        metrics = [result.to_dict() for result in evaluate_cot(cot_records)]
        report["cot"] = {
            "examples": len(cot_records),
            "metrics": metrics,
            "assessment": build_assessment(metrics).to_dict(),
        }
        combined.extend(metrics)
        expected += 12
    if structure_records:
        metrics = [result.to_dict() for result in evaluate_structure(structure_records)]
        report["structure"] = {
            "examples": len(structure_records),
            "metrics": metrics,
            "assessment": build_assessment(metrics).to_dict(),
        }
        combined.extend(metrics)
        expected += 12
    report["overall_assessment"] = build_assessment(
        combined,
        expected_metrics=expected or 12,
    ).to_dict()
    return report
