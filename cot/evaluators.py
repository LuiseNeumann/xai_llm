"""Abhängigkeitsfreie Basisauswertungen für alle zwölf Co-12-Eigenschaften.

Diese Funktionen werten zuvor gesammelte Artefakte aus. Lexikalische
Ähnlichkeit dient bewusst als transparente Baseline. Vor wissenschaftlichen
Aussagen sollte sie durch Embeddings, NLI, Theorembeweiser oder
aufgabenspezifische Metriken ersetzt werden.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

from common.metrics import (
    brier_score,
    expected_calibration_error,
    mean,
    normalized_entropy,
    pairwise_mean_similarity,
    token_f1,
)
from common.models import MetricResult

Record = dict[str, Any]


def _trace_text(trace: Record) -> str:
    return " ".join(str(step) for step in trace.get("steps", []))


def _normal_answer(value: Any) -> str:
    return str(value).strip().lower()


def correctness(records: list[Record]) -> MetricResult:
    """Prüft, ob Interventionseffekte ihrer erwarteten Richtung entsprechen."""
    outcomes: list[float] = []
    for record in records:
        original = _normal_answer(record.get("answer", ""))
        for intervention in record.get("interventions", []):
            if "expected_answer_change" not in intervention or "answer" not in intervention:
                continue
            changed = _normal_answer(intervention["answer"]) != original
            outcomes.append(float(changed == bool(intervention["expected_answer_change"])))
    if not outcomes:
        return MetricResult.unavailable("correctness", "Keine Interventionsergebnisse gefunden")
    return MetricResult(
        "correctness",
        mean(outcomes),
        True,
        {"interventions": len(outcomes), "definition": "erwartete Interventionseffekte stimmen überein"},
    )


def completeness(records: list[Record]) -> MetricResult:
    """Misst den Recall der erforderlichen Referenzschritte."""
    recalls: list[float] = []
    for record in records:
        gold_steps = record.get("gold_steps", [])
        generated_steps = record.get("steps", [])
        if not gold_steps:
            continue
        recalls.append(
            mean(max((token_f1(gold, generated) for generated in generated_steps), default=0.0) for gold in gold_steps)
        )
    if not recalls:
        return MetricResult.unavailable("completeness", "Keine Referenzschritte gefunden")
    return MetricResult(
        "completeness",
        mean(recalls),
        True,
        {"cases": len(recalls), "metric": "Token-F1-Recall der jeweils ähnlichsten Referenzschritte"},
    )


def consistency(records: list[Record]) -> MetricResult:
    """Misst die Übereinstimmung von Antworten und Begründungen über mehrere Läufe."""
    case_scores: list[float] = []
    answer_entropies: list[float] = []
    trace_similarities: list[float] = []
    for record in records:
        traces = [{"answer": record.get("answer"), "steps": record.get("steps", [])}]
        traces.extend(record.get("sampled_traces", []))
        if len(traces) < 2:
            continue
        answers = [_normal_answer(trace.get("answer", "")) for trace in traces]
        answer_entropy = normalized_entropy(answers)
        trace_similarity = pairwise_mean_similarity([_trace_text(trace) for trace in traces])
        answer_entropies.append(answer_entropy)
        trace_similarities.append(trace_similarity)
        case_scores.append(mean([1.0 - answer_entropy, trace_similarity]))
    if not case_scores:
        return MetricResult.unavailable(
            "consistency", "Pro Fall werden mindestens zwei Reasoning-Verläufe benötigt"
        )
    return MetricResult(
        "consistency",
        mean(case_scores),
        True,
        {
            "answer_consistency": 1.0 - mean(answer_entropies),
            "trace_similarity": mean(trace_similarities),
            "cases": len(case_scores),
        },
    )


def continuity(records: list[Record]) -> MetricResult:
    """Vergleicht ursprüngliche Erklärungen mit bedeutungserhaltenden Paraphrasen."""
    scores: list[float] = []
    answer_stability: list[float] = []
    trace_similarity: list[float] = []
    for record in records:
        original_answer = _normal_answer(record.get("answer", ""))
        original_trace = _trace_text(record)
        for paraphrase in record.get("paraphrase_traces", []):
            stable = float(_normal_answer(paraphrase.get("answer", "")) == original_answer)
            similarity = token_f1(original_trace, _trace_text(paraphrase))
            answer_stability.append(stable)
            trace_similarity.append(similarity)
            scores.append(stable * similarity)
    if not scores:
        return MetricResult.unavailable("continuity", "Keine Paraphrasen-Verläufe gefunden")
    return MetricResult(
        "continuity",
        mean(scores),
        True,
        {
            "answer_stability": mean(answer_stability),
            "trace_similarity": mean(trace_similarity),
            "comparisons": len(scores),
        },
    )


def contrastivity(records: list[Record]) -> MetricResult:
    """Prüft, ob ein minimaler Kontrast die beabsichtigte Antwortänderung auslöst."""
    answer_scores: list[float] = []
    trace_divergences: list[float] = []
    for record in records:
        contrast = record.get("contrast")
        if not contrast:
            continue
        original_answer = _normal_answer(record.get("answer", ""))
        contrast_answer = _normal_answer(contrast.get("answer", ""))
        expected_answer = _normal_answer(contrast.get("expected_answer", ""))
        correct = contrast_answer == expected_answer if expected_answer else True
        changed = contrast_answer != original_answer
        answer_scores.append(float(correct and changed))
        trace_divergences.append(1.0 - token_f1(_trace_text(record), _trace_text(contrast)))
    if not answer_scores:
        return MetricResult.unavailable("contrastivity", "Keine Kontrastläufe gefunden")
    return MetricResult(
        "contrastivity",
        mean(answer_scores),
        True,
        {
            "answer_switch_accuracy": mean(answer_scores),
            "mean_trace_divergence": mean(trace_divergences),
            "note": "Die Divergenz der Verläufe ist diagnostisch; ein größerer Wert ist nicht grundsätzlich besser.",
        },
    )


def covariate_complexity(records: list[Record]) -> MetricResult:
    """Berichtet die Anzahl der Konzepte und expliziten Konzeptinteraktionen."""
    complexities: list[float] = []
    for record in records:
        concepts = record.get("concepts")
        if concepts is None:
            continue
        interactions = record.get("concept_interactions", [])
        complexities.append(float(len(set(concepts)) + len(interactions)))
    if not complexities:
        return MetricResult.unavailable(
            "covariate_complexity", "Keine extrahierten Konzepte gefunden", higher_is_better=False
        )
    return MetricResult(
        "covariate_complexity",
        mean(complexities),
        False,
        {
            "cases": len(complexities),
            "note": "Komplexität nur bei vergleichbarer Correctness und Completeness vergleichen.",
        },
    )


def compactness(records: list[Record]) -> MetricResult:
    """Berichtet die Größe der minimal hinreichenden Begründung, sofern verfügbar."""
    necessary_counts: list[float] = []
    total_counts: list[int] = []
    compression_rates: list[float] = []
    for record in records:
        steps = record.get("steps", [])
        if not steps:
            continue
        indices = record.get("necessary_step_indices")
        if indices is None:
            continue
        necessary = len(set(indices))
        necessary_counts.append(float(necessary))
        total_counts.append(len(steps))
        compression_rates.append(1.0 - necessary / len(steps))
    if not necessary_counts:
        return MetricResult.unavailable(
            "compactness",
            "Keine Löschtests zur minimalen Suffizienz gefunden",
            higher_is_better=False,
        )
    return MetricResult(
        "compactness",
        mean(necessary_counts),
        False,
        {
            "mean_total_steps": mean(total_counts),
            "mean_removable_fraction": mean(compression_rates),
            "unit": "notwendige Schritte",
        },
    )


def composition(records: list[Record]) -> MetricResult:
    """Aggregiert externe Werte für Flüssigkeit, Logik und Organisation."""
    scores: list[float] = []
    dimensions: Counter[str] = Counter()
    for record in records:
        values = record.get("composition_scores", {})
        if not values:
            continue
        scores.append(mean(float(value) for value in values.values()))
        dimensions.update(values.keys())
    if not scores:
        return MetricResult.unavailable("composition", "Keine Kompositionsbewertungen gefunden")
    return MetricResult(
        "composition",
        mean(scores),
        True,
        {"cases": len(scores), "dimensions": sorted(dimensions)},
    )


def confidence(records: list[Record]) -> MetricResult:
    """Bewertet die Kalibrierung verbalisierter Konfidenzen einzelner Schritte."""
    confidences: list[float] = []
    labels: list[int] = []
    for record in records:
        record_confidences = record.get("step_confidences", [])
        record_labels = record.get("step_correctness", [])
        if len(record_confidences) != len(record_labels):
            continue
        confidences.extend(float(value) for value in record_confidences)
        labels.extend(int(value) for value in record_labels)
    if not confidences:
        return MetricResult.unavailable("confidence", "Keine passenden Konfidenzlabels gefunden")
    ece = expected_calibration_error(confidences, labels)
    return MetricResult(
        "confidence",
        1.0 - ece,
        True,
        {
            "expected_calibration_error": ece,
            "brier_score": brier_score(confidences, labels),
            "steps": len(confidences),
        },
    )


def context(records: list[Record]) -> MetricResult:
    """Aggregiert Relevanz- oder Nutzwerturteile der Zielnutzer."""
    scores = [
        float(score)
        for record in records
        for score in record.get("user_relevance_scores", [])
    ]
    if not scores:
        return MetricResult.unavailable(
            "context", "Keine Bewertungen der Nutzerrelevanz oder des Nutzwerts gefunden"
        )
    return MetricResult("context", mean(scores), True, {"ratings": len(scores)})


def coherence(records: list[Record]) -> MetricResult:
    """Verbindet Faktizitätsurteile mit der Ausrichtung an menschlichen Referenzschritten."""
    case_scores: list[float] = []
    factuality_values: list[float] = []
    alignment_values: list[float] = []
    for record in records:
        components: list[float] = []
        factuality = record.get("factuality_scores", [])
        if factuality:
            factuality_score = mean(float(value) for value in factuality)
            factuality_values.append(factuality_score)
            components.append(factuality_score)
        gold_steps = record.get("gold_steps", [])
        generated_steps = record.get("steps", [])
        if gold_steps:
            alignment = mean(
                max((token_f1(gold, generated) for generated in generated_steps), default=0.0)
                for gold in gold_steps
            )
            alignment_values.append(alignment)
            components.append(alignment)
        if components:
            case_scores.append(mean(components))
    if not case_scores:
        return MetricResult.unavailable(
            "coherence", "Keine Faktizitäts- oder Referenzbegründungsdaten gefunden"
        )
    return MetricResult(
        "coherence",
        mean(case_scores),
        True,
        {
            "mean_factuality": mean(factuality_values) if factuality_values else None,
            "mean_reference_alignment": mean(alignment_values) if alignment_values else None,
            "warning": "Referenzübereinstimmung misst Plausibilität, nicht kausale Treue.",
        },
    )


def controllability(records: list[Record]) -> MetricResult:
    """Prüft explizite Längen- und Wortschatzvorgaben für erzeugte Verläufe."""
    checks: list[float] = []
    by_type: Counter[str] = Counter()
    passed_by_type: Counter[str] = Counter()
    for record in records:
        constraints = record.get("constraints", {})
        text = _trace_text(record).lower()
        if "max_steps" in constraints:
            result = len(record.get("steps", [])) <= int(constraints["max_steps"])
            checks.append(float(result))
            by_type["max_steps"] += 1
            passed_by_type["max_steps"] += int(result)
        for term in constraints.get("required_terms", []):
            result = str(term).lower() in text
            checks.append(float(result))
            by_type["required_terms"] += 1
            passed_by_type["required_terms"] += int(result)
        for term in constraints.get("forbidden_terms", []):
            result = str(term).lower() not in text
            checks.append(float(result))
            by_type["forbidden_terms"] += 1
            passed_by_type["forbidden_terms"] += int(result)
    if not checks:
        return MetricResult.unavailable("controllability", "Keine Erklärungsvorgaben gefunden")
    return MetricResult(
        "controllability",
        mean(checks),
        True,
        {
            key: passed_by_type[key] / count
            for key, count in sorted(by_type.items())
        },
    )


EVALUATORS: tuple[Callable[[list[Record]], MetricResult], ...] = (
    correctness,
    completeness,
    consistency,
    continuity,
    contrastivity,
    covariate_complexity,
    compactness,
    composition,
    confidence,
    context,
    coherence,
    controllability,
)


def evaluate_all(records: list[Record]) -> list[MetricResult]:
    return [evaluator(records) for evaluator in EVALUATORS]
