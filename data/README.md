# Datenschemata

Das Projekt unterscheidet drei Arten von JSONL-Dateien.

## Benchmarkfälle

`sample_benchmark.jsonl` enthält Aufgabeneingaben und verfügbare Referenzdaten:

- `id`: stabiler Bezeichner des Beispiels.
- `prompt`: ursprüngliche Modelleingabe.
- `expected_answer`: korrekte Referenzantwort.
- `alternative_answer`: konkurrierende Antwort für Logit-Differenztests.
- `corrupted_prompt`: passende kontrafaktische Eingabe für Aktivierungs-Patching.
- `gold_steps`: Referenzschritte. Sie messen Plausibilität oder
  Beweisabdeckung, aber nicht automatisch die Treue zum Modell.
- `gold_concepts`: in der Erklärung erwartete Konzepte.
- `paraphrases`: bedeutungserhaltende Varianten.
- `contrast_prompt` und `contrast_answer`: minimales Kontrastpaar.
- `interventions`: Prompts, die die Antwort erhalten oder verändern sollen.
- `constraints`: verlangte Darstellungsvorgaben.

Heruntergeladene echte Benchmarks sollten ebenfalls unter `data/` liegen. Sie
sind in diesen Vorlagen nicht enthalten, da sich Lizenzen und Vorverarbeitung
unterscheiden.

## CoT-Artefakte

`sample_cot_artifacts.jsonl` speichert erzeugte Reasoning-Verläufe und
Interventionsergebnisse. Die Auswertung akzeptiert fehlende optionale Felder
und markiert nicht berechenbare Metriken als `not_available`, statt ihnen
stillschweigend den Wert null zuzuweisen.

## Strukturartefakte

`sample_structure_artifacts.jsonl` repräsentiert einen Circuit oder
Feature-Graphen und die Ergebnisse kausaler Validierungsexperimente. Das
synthetische Beispiel deckt alle zwölf Metriken ab. Ein echtes Experiment
sollte diese Felder aus Aktivierungs-Patching, Ablationen, SAE-Analysen und
Editing-Läufen befüllen.
