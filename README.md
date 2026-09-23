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

Die UI-Abhängigkeiten installieren und das lokale Prüflabor starten:

```bash
python3 -m pip install -e '.[ui]'
streamlit run interface/app.py
```

Die Oberfläche verwendet standardmäßig `Qwen/Qwen2.5-1.5B-Instruct`, bietet
einen Schnell- und einen vollständigen Modus und zeigt alle Co-12-Ergebnisse
einzeln aufklappbar an. Details und Einschränkungen stehen in
`interface/README.md`.

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
