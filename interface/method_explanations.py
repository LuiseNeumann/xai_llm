"""Verständliche Testverfahren und Datenvoraussetzungen für beide Co-12-Zweige."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MethodExplanation:
    procedure: str
    calculation: str
    required: str
    limitation: str


COT_METHODS: dict[str, MethodExplanation] = {
    "correctness": MethodExplanation(
        "Die Antwort zum Originalprompt wird mit Antworten nach kontrollierten Eingabeänderungen verglichen. Jede Intervention legt vorher fest, ob die Antwort gleich bleiben oder wechseln soll.",
        "Anteil der Interventionen mit der erwarteten Antwortänderung. Eine bloße Formänderung ohne Gegenfakt prüft nur Antwortstabilität und ist kein Beweis für kausale CoT-Treue.",
        "Originalantwort und `interventions` mit Antwort und `expected_answer_change`; für einen kausalen Gegentest mindestens eine relevante Änderung.",
        "Ein erwartungsgemäßes Antwortverhalten zeigt noch nicht, dass die genannten Begründungsschritte den internen Entscheidungsweg beschreiben.",
    ),
    "completeness": MethodExplanation(
        "Jeder vorgegebene Referenzschritt wird mit dem ähnlichsten generierten CoT-Schritt abgeglichen.",
        "Mittel des jeweils höchsten Token-F1 pro Referenzschritt; das ist eine lexikalische Näherung für die Abdeckung.",
        "Generierte `steps` und annotierte `gold_steps` für die jeweilige Aufgabe.",
        "Andere Formulierungen oder Sprachen können trotz richtiger Begründung einen niedrigen Token-F1 ergeben. Die interne Vollständigkeit wird so nicht geprüft.",
    ),
    "consistency": MethodExplanation(
        "Der Originalprompt wird mehrfach ausgeführt. Antworten und ausgegebene Begründungstexte werden untereinander verglichen.",
        "Mittel aus 1 minus normierter Antwortentropie und paarweiser Textähnlichkeit der Begründungen; beide Teilwerte stehen in den Rohdaten.",
        "Originalantwort und mindestens einen weiteren Lauf in `sampled_traces`.",
        "Gleiche Antworten beweisen nicht gleiche Argumente. Die Textähnlichkeit erkennt sinngleiche Umformulierungen nur eingeschränkt.",
    ),
    "continuity": MethodExplanation(
        "Der Prompt wird mit einem bedeutungserhaltenden Instruktionsrahmen erneut beantwortet; Antwort und CoT-Text werden mit dem Original verglichen.",
        "Anteil stabiler Antworten multipliziert mit dem Token-F1 zwischen den beiden Begründungstexten; bei mehreren Varianten gemittelt.",
        "Originalantwort/-schritte und Ergebnisse in `paraphrase_traces`.",
        "Die Variante ist nicht automatisch eine validierte semantische Paraphrase. Kurze oder anders formulierte Begründungen können trotz gleicher Bedeutung schlecht abschneiden.",
    ),
    "contrastivity": MethodExplanation(
        "Ein relevanter Gegenfakt wird als neuer Prompt beantwortet und mit der ursprünglichen Antwort verglichen.",
        "Anteil der Fälle, in denen die Antwort wechselt und – falls angegeben – genau der erwarteten kontrastiven Antwort entspricht. Die Textdivergenz wird separat angezeigt.",
        "Ein `contrast`-Lauf mit neuer Antwort; idealerweise ein minimales Kontrastpaar mit `expected_answer`.",
        "Ein bloßer Wechsel zeigt noch nicht, ob das Modell aus dem richtigen Grund gewechselt hat. Ohne bekannte Zielantwort ist nur der Wechsel testbar.",
    ),
    "covariate_complexity": MethodExplanation(
        "Die im CoT annotierten Konzepte und gegebenenfalls ihre expliziten Beziehungen werden gezählt.",
        "Anzahl unterschiedlicher `concepts` plus Anzahl `concept_interactions`; kleinere Werte bedeuten nur weniger erfasste Komplexität.",
        "Zuverlässig extrahierte `concepts`, optional `concept_interactions`.",
        "Ohne Konzeptannotation gibt es keinen Wert. Wenige Konzepte sind nicht automatisch besser, wenn wichtige Ursachen fehlen.",
    ),
    "compactness": MethodExplanation(
        "Begründungsschritte werden in separaten Löschtests entfernt, bis nur eine hinreichende Teilmenge übrig bleibt.",
        "Mittlere Anzahl notwendiger Schritte; zusätzlich wird der Anteil entfernbarer Schritte gezeigt.",
        "CoT-`steps` und experimentell bestimmte `necessary_step_indices` aus Löschtests.",
        "Aus der bloßen Länge einer Erklärung lässt sich keine minimale hinreichende Erklärung ableiten.",
    ),
    "composition": MethodExplanation(
        "Die Darstellungsqualität wird anhand der im Artefakt vorhandenen Kompositionswerte beurteilt. Im Live-Modus werden einfache Form- und Redundanzregeln genutzt.",
        "Arithmetisches Mittel der Werte in `composition_scores`; im Live-Modus insbesondere Satzzeichenanteil und 1 minus paarweise Tokenüberlappung.",
        "Mindestens einen Wert in `composition_scores`, zum Beispiel aus externen Textmetriken oder den Live-Formregeln.",
        "Die Live-Regeln prüfen weder logische Gültigkeit noch sprachliche Qualität umfassend. Im Live-Modus ist der Wert deshalb nur diagnostisch.",
    ),
    "confidence": MethodExplanation(
        "Die Konfidenz jedes generierten Schrittes wird gegen ein überprüftes Richtig/Falsch-Label desselben Schrittes abgeglichen.",
        "Score = 1 minus Expected Calibration Error (ECE); der Brier Score steht zusätzlich in den Rohdaten.",
        "Gleich lange Listen `step_confidences` und unabhängig verifizierte `step_correctness`.",
        "Selbst angegebene Konfidenzen ohne externe Korrektheitslabels erlauben keine Kalibrierungsmessung.",
    ),
    "context": MethodExplanation(
        "Zielnutzer bewerten, ob die Erklärung für ihre konkrete Aufgabe verständlich und nützlich ist.",
        "Mittel der vorliegenden Nutzerurteile in `user_relevance_scores`.",
        "Bewertungen der vorgesehenen Nutzergruppe oder eine entsprechende Nutzerstudie.",
        "Automatische Modellantworten enthalten keine Nutzerurteile. Ohne Studie ist dieses C nicht beurteilbar.",
    ),
    "coherence": MethodExplanation(
        "Der CoT-Text wird mit annotierten Referenzschritten und/oder gesonderten Faktizitätsurteilen verglichen.",
        "Mittel verfügbarer Faktizitätswerte und der lexikalischen Referenzübereinstimmung; getrennte Teilwerte stehen in den Details.",
        "Verifizierte `factuality_scores` und/oder annotierte `gold_steps`.",
        "Übereinstimmung mit einer menschlichen Musterbegründung ist keine kausale Treue. Bei unterschiedlichen Sprachen ist Token-F1 besonders unzuverlässig.",
    ),
    "controllability": MethodExplanation(
        "Das Modell erhält zusätzlich konkrete Vorgaben zu Schrittzahl und erlaubten bzw. erforderlichen Begriffen; die Antwort auf diesen Kontrollprompt wird geprüft.",
        "Anteil erfüllter Einzelvorgaben aus `max_steps`, `required_terms` und `forbidden_terms`.",
        "Mindestens eine Vorgabe unter `constraints` und ein passender Lauf; bei Live-Tests steht dieser in `controlled_trace`.",
        "Regelbefolgung bewertet die Steuerbarkeit der Darstellung, nicht die sachliche Richtigkeit der Begründung.",
    ),
}


STRUCTURE_METHODS: dict[str, MethodExplanation] = {
    "correctness": MethodExplanation(
        "Ein angenommener kausaler Circuit wird isoliert und seine Ausgabe mit der des vollständigen Modells verglichen.",
        "1 minus normierte absolute Differenz zwischen `full_score` und `circuit_score`, begrenzt auf 0 bis 1.",
        "Messergebnisse `full_score` und `circuit_score` desselben Falls.",
        "Layer-Patching allein bestimmt noch keinen isolierten Circuit und liefert daher keinen `circuit_score`.",
    ),
    "completeness": MethodExplanation(
        "Dieselben Knoten werden im vorgeschlagenen Circuit und im vollständigen Modell ausgeschaltet; danach wird die verbleibende Leistung verglichen.",
        "1 minus größte normierte Abweichung aus `incompleteness_scores` pro Fall; anschließend über Fälle gemittelt.",
        "Gemeinsame Knockout-Experimente am Circuit und Gesamtmodell mit `incompleteness_scores`.",
        "Ein Circuit kann ohne diese Gegenprüfungen wichtige Ersatzmechanismen außerhalb des Circuits übersehen.",
    ),
    "consistency": MethodExplanation(
        "Der Circuit wird mehrfach über unterschiedliche Stichproben oder Zufalls-Seeds extrahiert.",
        "Mittlere Jaccard-Ähnlichkeit der Knotenlisten in `replicate_node_sets`.",
        "Mindestens zwei unabhängig bestimmte Knotenmengen in `replicate_node_sets`.",
        "Ein einzelner Patching-Lauf liefert keine Aussage darüber, ob ein Circuit reproduzierbar ist.",
    ),
    "continuity": MethodExplanation(
        "Der Circuit wird für eine bedeutungserhaltend variierte Eingabe erneut gesucht.",
        "Mittlere Jaccard-Ähnlichkeit der Knotenlisten in `perturbed_node_sets`.",
        "Mindestens zwei zu vergleichende Knotenmengen aus Original und Perturbation.",
        "Die Änderung der Eingabe muss die Aufgabe tatsächlich erhalten; sonst ist ein veränderter Circuit nicht automatisch ein Fehler.",
    ),
    "contrastivity": MethodExplanation(
        "Dieselbe Intervention wird gegen den Ziel-Output und gegen nicht gewünschte Outputs ausgewertet.",
        "Betrag des `target_effect`, geteilt durch die Summe aus Ziel- und `off_target_effect`.",
        "Gemessene `target_effect`- und `off_target_effect`-Werte.",
        "Ohne Off-Target-Messung ist nicht erkennbar, ob der Mechanismus spezifisch wirkt.",
    ),
    "covariate_complexity": MethodExplanation(
        "Internen Features werden semantische Konzepte zugeordnet; pro Feature wird die Anzahl unterschiedlicher Labels gezählt.",
        "Durchschnittliche Zahl der Konzeptlabels pro Feature in `feature_labels`; Rekonstruktionsqualität wird getrennt berichtet.",
        "Zuordnungen in `feature_labels`, idealerweise zusätzlich `reconstruction_score` eines SAE.",
        "Wenige Labels allein beweisen keine Monosemantizität; eine schlechte Rekonstruktion kann wichtige Information verschweigen.",
    ),
    "compactness": MethodExplanation(
        "Die Zahl benötigter Circuit-Knoten wird relativ zur Modellgröße betrachtet, während die Ausgabe möglichst ähnlich zum vollständigen Modell bleiben soll.",
        "(1 minus Anteil ausgewählter Knoten) mal Faithfulness. Größenanteil und Faithfulness stehen separat in den Details.",
        "`nodes`, `total_nodes`, `full_score` und ein experimentell isolierter `circuit_score`.",
        "Nur die kleinsten Patching-Effekte zu zählen ist keine Minimalitätsprüfung; die Sparsity-Faithfulness-Pareto-Kurve ist aussagekräftiger.",
    ),
    "composition": MethodExplanation(
        "Lesbarkeit und Simulierbarkeit der Circuit-/Feature-Graph-Darstellung werden gesondert bewertet.",
        "Mittel der vorhandenen `composition_scores`; Graphgröße aus `nodes` und `edges` erscheint zusätzlich.",
        "Bewertungen der Darstellung in `composition_scores`, etwa aus einer Nutzerstudie.",
        "Ein automatisch extrahierter Graph enthält noch keine Messung, ob Menschen seine Darstellung verstehen.",
    ),
    "confidence": MethodExplanation(
        "Kausale Effekte derselben Komponenten werden über wiederholte Stichproben oder Modellläufe geschätzt.",
        "Mittlerer Anteil der Wiederholungen mit gleichem Vorzeichen in `effect_samples`; relative Streuung wird separat gezeigt.",
        "Mindestens zwei Effektmessungen pro Feature in `effect_samples`.",
        "Ein Einzelwert aus Aktivierungs-Patching liefert keine statistische Unsicherheit.",
    ),
    "context": MethodExplanation(
        "Vorab festgelegte Nutzerkonzepte werden mit den tatsächlich gefundenen Circuit-/Feature-Labels verglichen.",
        "Mittel aus Abdeckung der `user_concepts` und Präzision der `relevant_concepts_found` gegenüber irrelevanten Funden.",
        "`user_concepts`, `relevant_concepts_found` und gegebenenfalls `irrelevant_concepts_found`.",
        "Ohne festgelegte Zielgruppe und Concept-Mapping ist die Relevanz für einen Nutzungskontext nicht messbar.",
    ),
    "coherence": MethodExplanation(
        "Interne Features werden auf ihre Übereinstimmung mit menschlich verständlichen Konzepten geprüft.",
        "Mittel annotierter Werte in `feature_alignment_scores`.",
        "Validierte Konzept-/Feature-Zuordnungen in `feature_alignment_scores`.",
        "Auch gut benannte, verständliche Features müssen nicht kausal für die Antwort verantwortlich sein.",
    ),
    "controllability": MethodExplanation(
        "Identifizierte Komponenten werden gezielt editiert und auf Zieländerung, Generalisierung und unerwünschte Nebeneffekte getestet.",
        "Mittel aus `success`, `generalization` und `locality` unter `edit_results`.",
        "Vollständige Interventionsergebnisse in `edit_results`.",
        "Eine beschreibende Aktivierung oder ein Probe genügt nicht: Es muss tatsächlich ein Edit oder eine Ablation ausgeführt werden.",
    ),
}


def explanation_for(
    name: str,
    branch: str,
    result: dict[str, Any],
    absence_context: str | None = None,
) -> tuple[MethodExplanation, str]:
    """Liefert Testverfahren und laufbezogene Erklärung der Datenlage."""
    methods = COT_METHODS if branch == "cot" else STRUCTURE_METHODS
    method = methods[name]
    details = result.get("details", {})
    if result.get("status") != "ok" or result.get("score") is None:
        reason = details.get("reason", "Für diesen Lauf liegen keine geeigneten Messdaten vor.")
        if absence_context:
            reason = f"{absence_context} {reason}"
        return method, f"Kein Wert für diesen Lauf: {reason}"
    if details.get("assessment_eligible") is False:
        return method, (
            "Ein diagnostischer Wert wurde berechnet, fließt aber nicht in die Prüfampel ein: "
            + details.get("assessment_reason", "Die vorhandene Messung ist nur eine Näherung.")
        )
    for field, unit in (
        ("cases", "Fälle"),
        ("interventions", "Interventionen"),
        ("comparisons", "Vergleiche"),
        ("features", "Features"),
        ("steps", "Schritte"),
        ("ratings", "Nutzerbewertungen"),
    ):
        if isinstance(details.get(field), (int, float)):
            return method, f"Wert berechnet; zugrunde liegende Messmenge: {details[field]} {unit}."
    return method, "Ein Wert wurde aus den vorhandenen Artefakten berechnet. Die Rohdaten stehen darunter."
