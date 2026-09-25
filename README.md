# Co-12-Auswertungsvorlagen für LLM-Erklärungen

Dieses Repository enthält zwei kleine Auswertungspipelines:

- `cot/`: wertet erzeugte Chain-of-Thought-Erklärungen aus.
- `inner_model_structure/`: wertet strukturelle Erklärungen aus, die als
  kausale Circuits oder Feature-Graphen dargestellt werden.
- `interface/`: interaktive Streamlit-Oberfläche für freie Prompts und
  hochgeladene Auswertungsartefakte.
- `data/`: gemeinsame Benchmarkfälle und beispielhafte Auswertungsartefakte.
- `common/`: Schemata, Ein-/Ausgabehilfen und Metriken ohne externe
  Abhängigkeiten.
- `CO12_METHODEN.md`: Methodenübersicht, Quellen und Benchmarkideen.

Die enthaltenen Beispielartefakte sind synthetisch. Sie dienen zur technischen
Prüfung der Pipeline und nicht als wissenschaftliche Ergebnisse.

## Schnellstart

Das Basisprojekt benötigt nur Python 3.10 oder neuer.

```bash
python3 -m cot.run --artifacts data/sample_cot_artifacts.jsonl
python3 -m inner_model_structure.run \
  --artifacts data/sample_structure_artifacts.jsonl
python3 -m unittest discover -s tests
```

So werden die Berichte in Dateien geschrieben:

```bash
python3 -m cot.run \
  --artifacts data/sample_cot_artifacts.jsonl \
  --output cot_results.json
python3 -m inner_model_structure.run \
  --artifacts data/sample_structure_artifacts.jsonl \
  --output structure_results.json
```

## Optionale Hugging-Face-Experimente

Optionale Abhängigkeiten installieren:

```bash
python3 -m pip install -e '.[hf]'
```

CoT-Basisartefakte erzeugen:

```bash
python3 -m cot.collect_hf \
  --model google/gemma-2-2b-it \
  --data data/sample_benchmark.jsonl \
  --output cot_hf_artifacts.jsonl
```

Ein grobes Aktivierungs-Patching-Artefakt auf Layer-Ebene erzeugen:

```bash
python3 -m inner_model_structure.collect_hf \
  --model google/gemma-2-2b-it \
  --data data/sample_benchmark.jsonl \
  --output structure_hf_artifacts.jsonl
```

Die Hugging-Face-Skripte sind Ausgangspunkte. Insbesondere ist Patching auf
Layer-Ebene keine vollständige mechanistische Erklärung. Für publizierbare
Ergebnisse sollte es durch Interventionen auf Head-, MLP-, Kanten- oder
SAE-Feature-Ebene ersetzt und der resultierende Circuit mit einer Aufgabe mit
bekannter Referenzstruktur wie Tracr validiert werden.

## Interaktive Oberfläche

Für die lokal in LM Studio vorhandene Gemma-4-26B-A4B-GGUF-Datei die
UI-Abhängigkeiten installieren und das Prüflabor starten:

```bash
uv sync --extra ui
lms load google/gemma-4-26b-a4b --gpu 0.6 -c 4096 -y
lms server start
uv run streamlit run interface/app.py
```

Die Befehle stehen auch in `QUICKSTART.md`. Die Oberfläche verwendet
standardmäßig LM Studio und `google/gemma-4-26b-a4b` auf
`http://127.0.0.1:1234/v1`. Alternativ ist `qwen3.8-27b` auswählbar.
Benchmarks können unter **Benchmark starten** direkt ausgeführt oder über
`uv run python -m interface.run_benchmark` erzeugt werden. Die gespeicherten
Ergebnisse unter `data/results/` sind im UI erneut abrufbar. Details stehen
in `interface/README.md`.

Als echter Datensatz ist `heegyu/bbq` / `Gender_identity` mit 5.672 Fällen
auswählbar. Das UI lädt ihn bei Bedarf herunter; alternativ:

```bash
uv run python -m interface.download_bbq
```

## Empfohlener Ablauf

1. Benchmarkbeispiele nach dem Schema aus `data/README.md` unter `data/`
   ergänzen.
2. Modellausgaben und Interventionsartefakte erzeugen.
3. Beide Co-12-Auswertungen auf diesen Artefakten ausführen.
4. Jede Metrik getrennt untersuchen. Die zwölf Werte nicht ohne ein explizites,
   anwendungsabhängiges Gewichtungsschema mitteln.
5. Konfidenzintervalle über Beispiele und Zufalls-Seeds berichten.

## Wichtige Einschränkung

Ein erzeugtes CoT ist eine beobachtbare Erklärung und kein privilegierter
Zugriff auf das latente Reasoning des Modells. Ebenso belegt ein Probe oder eine
Korrelation in einer Aktivierung noch keine Kausalität. Strukturelle Aussagen
sollten daher durch Interventionen wie Aktivierungs-Patching, Ablation und
Model-Editing gestützt werden.
