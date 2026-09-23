"""Vorsichtige Ampel- und Interpretationlogik für Co-12-Berichte."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


METRIC_INFO: dict[str, dict[str, str]] = {
    "correctness": {
        "title": "Correctness",
        "description": "Prüft, ob die Erklärung auf kontrollierte Eingriffe in erwartbarer Weise reagiert.",
        "action": "Einflussfaktoren und kausale Interventionen manuell kontrollieren.",
    },
    "completeness": {
        "title": "Completeness",
        "description": "Prüft, wie viele notwendige Referenzschritte von der Erklärung abgedeckt werden.",
        "action": "Fehlende Fakten oder Beweisschritte ergänzen und erneut prüfen.",
    },
    "consistency": {
        "title": "Consistency",
        "description": "Vergleicht Antworten und Begründungen über mehrere Modellläufe.",
        "action": "Weitere Samples erzeugen und widersprüchliche Begründungen untersuchen.",
    },
    "continuity": {
        "title": "Continuity",
        "description": "Prüft die Stabilität bei bedeutungserhaltenden Promptvarianten.",
        "action": "Prompt paraphrasieren und starke Änderungen der Begründung untersuchen.",
    },
    "contrastivity": {
        "title": "Contrastivity",
        "description": "Prüft, ob ein relevanter Gegenfakt die Antwort gezielt verändert.",
        "action": "Ein minimales Kontrastpaar mit bekannter Zielantwort bereitstellen.",
    },
    "covariate_complexity": {
        "title": "Covariate Complexity",
        "description": "Beschreibt Anzahl und Interaktionen der verwendeten Konzepte.",
        "action": "Komplexität nur zusammen mit Correctness und Completeness interpretieren.",
    },
    "compactness": {
        "title": "Compactness",
        "description": "Beschreibt die Größe einer minimal hinreichenden Erklärung.",
        "action": "Löschtests durchführen und redundante Schritte identifizieren.",
    },
    "composition": {
        "title": "Composition",
        "description": "Bewertet Struktur, Lesbarkeit und Redundanz der Darstellung.",
        "action": "Unklare, doppelte oder ungeordnete Schritte überarbeiten.",
    },
    "confidence": {
        "title": "Confidence",
        "description": "Prüft, ob angegebene Konfidenzen mit der tatsächlichen Zuverlässigkeit übereinstimmen.",
        "action": "Konfidenzen gegen verifizierte Schrittlabels kalibrieren.",
    },
    "context": {
        "title": "Context",
        "description": "Misst die Relevanz der Erklärung für Zielgruppe und konkrete Aufgabe.",
        "action": "Erklärung mit den vorgesehenen Nutzern in einer Aufgabe testen.",
    },
    "coherence": {
        "title": "Coherence",
        "description": "Vergleicht die Erklärung mit Referenzwissen und menschlichen Begründungen.",
        "action": "Fakten und Schlussfolgerungen gegen vertrauenswürdige Quellen prüfen.",
    },
    "controllability": {
        "title": "Controllability",
        "description": "Prüft, ob gewünschte Form-, Inhalts- und Längenvorgaben eingehalten werden.",
        "action": "Nicht eingehaltene Vorgaben expliziter formulieren und erneut testen.",
    },
}


THRESHOLDS: dict[str, tuple[float, float]] = {
    "correctness": (0.80, 0.50),
    "completeness": (0.75, 0.45),
    "consistency": (0.70, 0.40),
    "continuity": (0.65, 0.35),
    "contrastivity": (0.75, 0.45),
    "composition": (0.70, 0.40),
    "confidence": (0.75, 0.50),
    "context": (0.70, 0.40),
    "coherence": (0.75, 0.45),
    "controllability": (0.75, 0.45),
}

CRITICAL_METRICS = {"correctness", "consistency", "continuity", "coherence"}


@dataclass(slots=True)
class Assessment:
    color: str
    label: str
    message: str
    evidence_level: str
    evidence_coverage: float
    available_metrics: int
    expected_metrics: int
    warnings: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def metric_state(result: dict[str, Any]) -> str:
    """Ordnet einen Metrikwert einer verständlichen Qualitätsstufe zu."""
    if result.get("status") != "ok" or result.get("score") is None:
        return "nicht messbar"
    if not result.get("higher_is_better", True):
        return "diagnostischer Wert"
    name = str(result.get("name", ""))
    score = float(result["score"])
    green, red = THRESHOLDS.get(name, (0.70, 0.40))
    if score >= green:
        return "unauffällig"
    if score < red:
        return "kritisch"
    return "prüfen"


def build_assessment(
    metrics: list[dict[str, Any]],
    expected_metrics: int = 12,
    answer_matches_reference: bool | None = None,
) -> Assessment:
    """Erstellt eine Ampel, ohne fehlende Evidenz als positiven Befund zu werten."""
    available = [
        result
        for result in metrics
        if result.get("status") == "ok" and result.get("score") is not None
    ]
    coverage = len(available) / expected_metrics if expected_metrics else 0.0
    if coverage >= 0.75:
        evidence_level = "hoch"
    elif coverage >= 0.40:
        evidence_level = "mittel"
    else:
        evidence_level = "niedrig"

    warnings: list[str] = []
    recommendations: list[str] = []
    critical_failure = answer_matches_reference is False
    moderate_issue = False

    if answer_matches_reference is False:
        warnings.append("Die Modellantwort stimmt nicht mit der angegebenen Referenzantwort überein.")
        recommendations.append("Die fachliche Richtigkeit der Antwort manuell prüfen.")

    for result in metrics:
        name = str(result.get("name", ""))
        state = metric_state(result)
        info = METRIC_INFO.get(name, {"title": name, "action": "Ergebnis manuell prüfen."})
        if state == "kritisch":
            warnings.append(f"{info['title']} liegt im kritischen Bereich.")
            recommendations.append(info["action"])
            critical_failure = critical_failure or name in CRITICAL_METRICS
            moderate_issue = True
        elif state == "prüfen":
            warnings.append(f"{info['title']} ist nur mittelstark ausgeprägt.")
            recommendations.append(info["action"])
            moderate_issue = True

    if coverage < 0.40:
        warnings.append("Für die meisten Co-12-Eigenschaften fehlen belastbare Messdaten.")
        recommendations.append("Referenzantworten, Gold-Schritte oder kontrafaktische Eingaben ergänzen.")
    elif coverage < 0.75:
        warnings.append("Die Evidenz deckt nur einen Teil der Co-12-Eigenschaften ab.")

    if critical_failure:
        color, label = "red", "Manuelle Prüfung erforderlich"
        message = "Mindestens eine zentrale Prüfung zeigt ein deutliches Warnsignal."
    elif moderate_issue or coverage < 0.75:
        color, label = "yellow", "Weitere Prüfung empfohlen"
        message = "Die gemessenen Eigenschaften sind teilweise unklar oder die Evidenz ist unvollständig."
    else:
        color, label = "green", "In geprüften Dimensionen unauffällig"
        message = "In den tatsächlich gemessenen Dimensionen wurden keine deutlichen Warnsignale gefunden."

    return Assessment(
        color=color,
        label=label,
        message=message,
        evidence_level=evidence_level,
        evidence_coverage=coverage,
        available_metrics=len(available),
        expected_metrics=expected_metrics,
        warnings=list(dict.fromkeys(warnings)),
        recommendations=list(dict.fromkeys(recommendations)),
    )


def interpret_metric(result: dict[str, Any]) -> str:
    """Formuliert eine knappe Interpretation eines einzelnen Ergebnisses."""
    state = metric_state(result)
    if state == "nicht messbar":
        reason = result.get("details", {}).get("reason", "Benötigte Evidenz fehlt.")
        return f"Nicht messbar: {reason}"
    score = result.get("score")
    if state == "diagnostischer Wert":
        return f"Diagnostischer Rohwert: {score:.3f}. Dieser Wert besitzt keine universelle Gut/Schlecht-Schwelle."
    return f"Bewertung: {state}. Gemessener Wert: {score:.3f}."
