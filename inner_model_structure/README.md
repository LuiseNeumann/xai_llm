# Pipeline für die innere Modellstruktur

Das Erklärungsobjekt ist ein kausaler Circuit oder ein dünnbesetzter
Feature-Graph. Ein vollständiges Artefakt sollte sowohl die Erklärung als auch
Ergebnisse von Interventionen enthalten, welche diese Erklärung prüfen.

## Zuordnung von Metriken zu Artefakten

| Eigenschaft | Benötigte Evidenz im Artefakt |
|---|---|
| Correctness | `full_score` und isolierter `circuit_score` |
| Completeness | `incompleteness_scores` aus gemeinsamen Knockouts |
| Consistency | `replicate_node_sets` aus Seeds oder Bootstrap-Läufen |
| Continuity | `perturbed_node_sets` aus äquivalenten Eingaben |
| Contrastivity | `target_effect` und `off_target_effect` |
| Covariate Complexity | `feature_labels` und SAE-Rekonstruktionsqualität |
| Compactness | Circuit-Knoten, Gesamtzahl der Einheiten und Faithfulness |
| Composition | Studienwerte zur Lesbarkeit und Simulierbarkeit des Graphen |
| Confidence | wiederholte kausale `effect_samples` |
| Context | von Nutzern angeforderte und gefundene Konzepte |
| Coherence | semantische `feature_alignment_scores` |
| Controllability | Editiererfolg, Generalisierung und Locality |

`collect_hf.py` lokalisiert lediglich einflussreiche Layer. Das Skript erzeugt
bewusst keine nicht vorhandenen Messwerte für Vollständigkeit, Semantik oder
Nutzerstudien. Abhängig vom gewählten Modell und der Interventionsebene kann es
mit TransformerLens, pyvene, nnsight oder SAELens erweitert werden.
