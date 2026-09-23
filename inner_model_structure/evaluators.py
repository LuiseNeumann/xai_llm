"""Auswertungen für kausale Circuits und dünnbesetzte Feature-Graph-Artefakte."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from common.metrics import clamp, mean, pairwise_mean_jaccard, population_std
from common.models import MetricResult

Record = dict[str, Any]


def correctness(records: list[Record]) -> MetricResult:
    """Misst, wie genau der isolierte Circuit das Verhalten des Gesamtmodells reproduziert."""
    scores: list[float] = []
    gaps: list[float] = []
    for record in records:
        if "full_score" not in record or "circuit_score" not in record:
            continue
        full = float(record["full_score"])
        circuit = float(record["circuit_score"])
        gap = abs(full - circuit) / max(abs(full), 1e-12)
        gaps.append(gap)
        scores.append(clamp(1.0 - gap))
    if not scores:
        return MetricResult.unavailable(
            "correctness", "Werte für Gesamtmodell und Circuit werden benötigt"
        )
    return MetricResult(
        "correctness",
        mean(scores),
        True,
        {"normalized_faithfulness_gap": mean(gaps), "cases": len(scores)},
    )


def completeness(records: list[Record]) -> MetricResult:
    """Aggregiert die größte Abweichung zwischen Circuit und Modell bei gleichen Knockouts."""
    worst_scores: list[float] = []
    for record in records:
        values = record.get("incompleteness_scores", [])
        if values:
            worst_scores.append(max(float(value) for value in values))
    if not worst_scores:
        return MetricResult.unavailable(
            "completeness", "Keine gemeinsamen Knockout-Experimente für Circuit und Modell gefunden"
        )
    return MetricResult(
        "completeness",
        mean(clamp(1.0 - value) for value in worst_scores),
        True,
        {"mean_worst_case_incompleteness": mean(worst_scores), "cases": len(worst_scores)},
    )


def consistency(records: list[Record]) -> MetricResult:
    """Vergleicht Circuits aus Bootstrap-Stichproben oder verschiedenen Zufalls-Seeds."""
    scores: list[float] = []
    for record in records:
        node_sets = record.get("replicate_node_sets", [])
        if len(node_sets) >= 2:
            scores.append(pairwise_mean_jaccard(node_sets))
    if not scores:
        return MetricResult.unavailable(
            "consistency", "Mindestens zwei extrahierte Circuits werden benötigt"
        )
    return MetricResult(
        "consistency", mean(scores), True, {"cases": len(scores), "metric": "Knoten-Jaccard"}
    )


def continuity(records: list[Record]) -> MetricResult:
    """Vergleicht Circuits aus bedeutungserhaltenden Eingabeperturbationen."""
    scores: list[float] = []
    for record in records:
        node_sets = record.get("perturbed_node_sets", [])
        if len(node_sets) >= 2:
            scores.append(pairwise_mean_jaccard(node_sets))
    if not scores:
        return MetricResult.unavailable("continuity", "Keine Paare perturbierter Circuits gefunden")
    return MetricResult(
        "continuity", mean(scores), True, {"cases": len(scores), "metric": "Knoten-Jaccard"}
    )


def contrastivity(records: list[Record]) -> MetricResult:
    """Misst, ob Interventionen das Ziel stärker als andere Ausgaben beeinflussen."""
    scores: list[float] = []
    target_effects: list[float] = []
    off_target_effects: list[float] = []
    for record in records:
        if "target_effect" not in record or "off_target_effect" not in record:
            continue
        target = abs(float(record["target_effect"]))
        off_target = abs(float(record["off_target_effect"]))
        scores.append(target / (target + off_target) if target + off_target else 0.0)
        target_effects.append(target)
        off_target_effects.append(off_target)
    if not scores:
        return MetricResult.unavailable(
            "contrastivity", "Target- und Off-Target-Effekte werden benötigt"
        )
    return MetricResult(
        "contrastivity",
        mean(scores),
        True,
        {
            "mean_target_effect": mean(target_effects),
            "mean_off_target_effect": mean(off_target_effects),
        },
    )


def covariate_complexity(records: list[Record]) -> MetricResult:
    """Berichtet semantische Labels pro Feature als Näherung für Polysemantizität."""
    labels_per_feature: list[float] = []
    reconstruction_scores: list[float] = []
    for record in records:
        labels = record.get("feature_labels", {})
        labels_per_feature.extend(float(len(set(values))) for values in labels.values())
        if "reconstruction_score" in record:
            reconstruction_scores.append(float(record["reconstruction_score"]))
    if not labels_per_feature:
        return MetricResult.unavailable(
            "covariate_complexity", "Keine semantischen Feature-Labels gefunden", higher_is_better=False
        )
    return MetricResult(
        "covariate_complexity",
        mean(labels_per_feature),
        False,
        {
            "features": len(labels_per_feature),
            "monosemantic_fraction": mean(value == 1.0 for value in labels_per_feature),
            "mean_reconstruction_score": mean(reconstruction_scores) if reconstruction_scores else None,
        },
    )


def compactness(records: list[Record]) -> MetricResult:
    """Verbindet die Sparsity des Circuits mit seiner Faithfulness."""
    scores: list[float] = []
    size_fractions: list[float] = []
    faithfulness_scores: list[float] = []
    for record in records:
        if not all(key in record for key in ("nodes", "total_nodes", "full_score", "circuit_score")):
            continue
        size_fraction = len(set(record["nodes"])) / max(int(record["total_nodes"]), 1)
        full = float(record["full_score"])
        gap = abs(full - float(record["circuit_score"])) / max(abs(full), 1e-12)
        faithfulness = clamp(1.0 - gap)
        size_fractions.append(size_fraction)
        faithfulness_scores.append(faithfulness)
        scores.append((1.0 - size_fraction) * faithfulness)
    if not scores:
        return MetricResult.unavailable(
            "compactness", "Circuit-Größe und Faithfulness werden benötigt"
        )
    return MetricResult(
        "compactness",
        mean(scores),
        True,
        {
            "mean_node_fraction": mean(size_fractions),
            "mean_faithfulness": mean(faithfulness_scores),
            "warning": "In Forschungsergebnissen sollte die vollständige Sparsity-Faithfulness-Pareto-Kurve berichtet werden.",
        },
    )


def composition(records: list[Record]) -> MetricResult:
    """Aggregiert Werte zur Lesbarkeit und menschlichen Simulierbarkeit des Graphen."""
    scores: list[float] = []
    graph_sizes: list[int] = []
    for record in records:
        values = record.get("composition_scores", {})
        if values:
            scores.append(mean(float(value) for value in values.values()))
            graph_sizes.append(len(record.get("nodes", [])) + len(record.get("edges", [])))
    if not scores:
        return MetricResult.unavailable(
            "composition", "Keine Studienwerte zur Darstellung des Graphen gefunden"
        )
    return MetricResult(
        "composition",
        mean(scores),
        True,
        {"cases": len(scores), "mean_graph_elements": mean(graph_sizes)},
    )


def confidence(records: list[Record]) -> MetricResult:
    """Misst die Stabilität geschätzter Kausaleffekte über wiederholte Läufe."""
    sign_stabilities: list[float] = []
    relative_standard_deviations: list[float] = []
    for record in records:
        for values in record.get("effect_samples", {}).values():
            numeric = [float(value) for value in values]
            if len(numeric) < 2:
                continue
            positives = sum(value > 0 for value in numeric)
            negatives = sum(value < 0 for value in numeric)
            sign_stabilities.append(max(positives, negatives) / len(numeric))
            relative_standard_deviations.append(
                population_std(numeric) / max(abs(mean(numeric)), 1e-12)
            )
    if not sign_stabilities:
        return MetricResult.unavailable(
            "confidence", "Wiederholte Schätzungen der Kausaleffekte werden benötigt"
        )
    return MetricResult(
        "confidence",
        mean(sign_stabilities),
        True,
        {
            "mean_relative_standard_deviation": mean(relative_standard_deviations),
            "features": len(sign_stabilities),
            "note": "Im abschließenden Experiment Bootstrap-Konfidenzintervalle verwenden.",
        },
    )


def context(records: list[Record]) -> MetricResult:
    """Misst die Abdeckung der von Zielnutzern ausgewählten Konzepte."""
    scores: list[float] = []
    coverages: list[float] = []
    for record in records:
        requested = set(record.get("user_concepts", []))
        if not requested:
            continue
        found = set(record.get("relevant_concepts_found", []))
        irrelevant = set(record.get("irrelevant_concepts_found", []))
        coverage = len(requested & found) / len(requested)
        precision = len(requested & found) / len(found | irrelevant) if found or irrelevant else 0.0
        coverages.append(coverage)
        scores.append(mean([coverage, precision]))
    if not scores:
        return MetricResult.unavailable("context", "Keine nutzergewählten Konzeptmengen gefunden")
    return MetricResult(
        "context",
        mean(scores),
        True,
        {"mean_requested_concept_coverage": mean(coverages), "cases": len(scores)},
    )


def coherence(records: list[Record]) -> MetricResult:
    """Aggregiert die semantische Ausrichtung interner Features an Konzepten."""
    values = [
        float(value)
        for record in records
        for value in record.get("feature_alignment_scores", [])
    ]
    if not values:
        return MetricResult.unavailable(
            "coherence", "Keine Werte zur Ausrichtung von Features und Konzepten gefunden"
        )
    return MetricResult(
        "coherence",
        mean(values),
        True,
        {
            "features": len(values),
            "warning": "Semantische Ausrichtung allein belegt keine kausale Correctness.",
        },
    )


def controllability(records: list[Record]) -> MetricResult:
    """Verbindet den Erfolg von Model-Edits mit Generalisierung und Locality."""
    scores: list[float] = []
    dimensions: dict[str, list[float]] = {}
    for record in records:
        values = record.get("edit_results", {})
        required = ("success", "generalization", "locality")
        if not all(key in values for key in required):
            continue
        numeric = [float(values[key]) for key in required]
        scores.append(mean(numeric))
        for key in required:
            dimensions.setdefault(key, []).append(float(values[key]))
    if not scores:
        return MetricResult.unavailable(
            "controllability", "Keine vollständigen Ergebnisse zu Model-Edits gefunden"
        )
    return MetricResult(
        "controllability",
        mean(scores),
        True,
        {key: mean(values) for key, values in dimensions.items()},
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
