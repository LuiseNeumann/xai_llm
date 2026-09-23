# CoT-Pipeline

`collect_hf.py` erzeugt beobachtbare Reasoning-Verläufe und führt kontrollierte
Eingabevarianten aus. `evaluators.py` ordnet die resultierenden Artefaktfelder
den zwölf Co-12-Eigenschaften zu.

## Zuordnung von Metriken zu Artefakten

| Eigenschaft | Benötigte Evidenz im Artefakt |
|---|---|
| Correctness | `interventions` mit erwarteter Richtung der Antwortänderung |
| Completeness | erzeugte `steps` und `gold_steps` als Referenz |
| Consistency | wiederholte `sampled_traces` |
| Continuity | `paraphrase_traces` aus bedeutungserhaltenden Varianten |
| Contrastivity | ein minimales Kontrastpaar unter `contrast` |
| Covariate Complexity | extrahierte `concepts` und optionale Interaktionen |
| Compactness | `necessary_step_indices` aus Löschtests |
| Composition | externe `composition_scores`, beispielsweise von ROSCOE |
| Confidence | `step_confidences` und verifizierte `step_correctness` |
| Context | `user_relevance_scores` oder nachgelagerter Nutzwert |
| Coherence | `factuality_scores` und/oder Referenzschritte |
| Controllability | angeforderte `constraints` und der erzeugte Verlauf |

Die lexikalischen Baselines in `evaluators.py` sind bewusst einfach. Für eine
echte Studie sollten aufgabenspezifische Prüfer für Arithmetik oder formale
Logik sowie NLI- oder validierte semantische Metriken für freien Text ergänzt
werden.
