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

## BBQ: Gender_identity

Der reale Benchmark `heegyu/bbq` / `Gender_identity` kann in Streamlit unter
**Benchmark starten** gewählt oder mit folgendem Befehl lokal geladen werden:

```bash
uv run python -m interface.download_bbq
```

Die 5.672 Testfälle werden als
`data/downloads/bbq_gender_identity.jsonl` gespeichert. Das Dataset steht auf
Hugging Face unter [heegyu/bbq](https://huggingface.co/datasets/heegyu/bbq)
mit der Lizenz CC BY 4.0. Das ältere Dataset-Skript benötigt `datasets<4`;
der Downloader ruft `load_dataset("heegyu/bbq", "Gender_identity",
trust_remote_code=True)` auf. Der lokale Download wird nicht ins Git-Repository
aufgenommen.

Jede Frage enthält drei Antwortoptionen `(A)` bis `(C)`. Das numerische
Original-Label wird exakt auf den jeweiligen Buchstaben abgebildet. Unter
`metadata` bleiben Original-Label, Optionen, Kontexttyp (`ambig`/`disambig`),
Fragepolarität und weitere BBQ-Informationen erhalten. Der Benchmark enthält
keine Referenz-Begründungsschritte; entsprechende Co-12-Metriken sind daher
nicht belastbar verfügbar.

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
