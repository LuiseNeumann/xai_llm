# Interaktive Co-12-Oberfläche

Die Streamlit-Anwendung verbindet ein lokales Open-Weight-Modell mit den
beiden Auswertungspipelines. Sie bietet eine Live-Analyse freier Prompts und
eine Offline-Auswertung bereits gesammelter JSONL-Artefakte.

## Installation

```bash
uv sync --extra ui
```

## Start

```bash
lms load google/gemma-4-26b-a4b --gpu 0.6 -c 4096 -y
lms server start
uv run streamlit run interface/app.py
```

Das in LM Studio bereits heruntergeladene Modell heißt
`google/gemma-4-26b-a4b`. Sein lokaler API-Server läuft standardmäßig auf
`http://127.0.0.1:1234/v1`. Im UI kann auch `qwen3.8-27b` gewählt werden;
das Modell muss dazu ebenfalls in LM Studio geladen sein. Die alternative
Hugging-Face-Anbindung benötigt `uv sync --extra ui-hf`.

## Bedienung

1. Modell und Rechengerät in der Seitenleiste auswählen.
2. Schnell- oder vollständigen Modus auswählen.
3. Prompt eingeben.
4. Optional Referenzantwort, Referenzschritte und kontrollierte Gegenfakten
   ergänzen.
5. `Antwort erzeugen und prüfen` auswählen.
6. Ampel, Evidenzabdeckung und die zwölf aufklappbaren Einzelergebnisse prüfen.

## Benchmark direkt in Streamlit starten

Im Tab **Benchmark starten** zwischen den synthetischen Beispielaufgaben und
**BBQ: Gender_identity (5.672)** wählen oder eine eigene JSONL-Datei im Schema
aus `data/README.md` hochladen. Fehlt BBQ noch lokal, erscheint ein
Downloadknopf. Für BBQ kann zwischen mehrdeutigen und eindeutigen Kontexten
gefiltert und die Anzahl der Testfälle vor dem Start gewählt werden.
Mit **Benchmark jetzt starten** werden die Modellantworten und Co-12-Werte
berechnet. Accuracy, BBQ-Gruppenergebnisse nach Kontexttyp/Fragepolarität und
Einzelfälle werden direkt angezeigt. Die Co-12-Ergebnisse sind in zwei
getrennten Reitern **CoT · 12 Cs** und **Innere Modellstruktur · 12 Cs**
aufklappbar – sowohl insgesamt als auch für jeden einzelnen Fall. Wenn keine
Strukturartefakte vorliegen, werden alle zwölf Eigenschaften ausdrücklich
als nicht messbar ausgewiesen.
Innerhalb jedes C kann **Testverfahren und Datenlage aufklappen** geöffnet
werden: Es beschreibt den konkreten Test, die Score-Berechnung, benötigte
Messdaten, Einschränkungen und bei fehlendem Wert den Grund für diesen Lauf.
Ergebnisse bleiben unter `data/results/` erhalten und können im Tab erneut
ausgewählt werden.

Alternativ denselben Ablauf im Terminal starten:

```bash
uv run python -m interface.run_benchmark --model google/gemma-4-26b-a4b
```

Für BBQ über das Terminal:

```bash
uv run python -m interface.download_bbq
uv run python -m interface.run_benchmark --model google/gemma-4-26b-a4b --data data/downloads/bbq_gender_identity.jsonl --limit 20
```

Mit dem voreingestellten Hugging-Face-Modell statt LM Studio:

```bash
uv sync --extra ui-hf
uv run python -m interface.run_benchmark --backend hf --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --data data/downloads/bbq_gender_identity.jsonl --limit 10 --samples 1
```

Das vollständige BBQ-Subset umfasst 5.672 Fälle und erfordert entsprechend
viele Modellaufrufe. Ein kleiner Testlauf über beispielsweise 4 oder 20 Fälle
ist schneller, aber keine repräsentative Gesamtauswertung.

Die mitgelieferten zwei Aufgaben sind synthetische Funktionsbeispiele, kein
repräsentativer Qualitätsbenchmark. Eigene Fälle benötigen mindestens `id`,
`prompt` und `expected_answer`.

## Schnellmodus

Der Schnellmodus erzeugt eine deterministische Hauptantwort, mehrere
Reasoning-Samples, einen bedeutungserhaltenden Instruktionsrahmen und eine
irrelevante Formänderung. Damit lassen sich vor allem Consistency, Continuity,
Composition und ein eingeschränkter Correctness-Proxy untersuchen.

## Vollständiger Modus

Der vollständige Modus erzeugt zusätzliche Samples, Instruktionsvarianten und
Interventionen. Wenn eine Alternativantwort und ein kontrafaktischer Prompt
vorliegen, kann außerdem eine Layer-Patching-Diagnose ausgeführt werden. Diese
Diagnose ist keine vollständige mechanistische Circuit-Erklärung.

## Ampelinterpretation

- Grün bedeutet nur, dass in den tatsächlich gemessenen Dimensionen kein
  deutliches Warnsignal auftrat.
- Gelb bedeutet, dass Ergebnisse unsicher sind oder wesentliche Evidenz fehlt.
- Rot bedeutet, dass mindestens eine zentrale Prüfung ein deutliches Problem
  zeigt.

Die Evidenzabdeckung wird getrennt von der Ampel angezeigt. Nicht messbare
Eigenschaften erhöhen den Qualitätsstatus nicht.

## Datenschutz und Ressourcen

Die LM-Studio-Modellinferenz erfolgt lokal und nutzt das bereits vorhandene
GGUF-Modell. LM Studio bietet über seine API keine internen Aktivierungen;
Strukturmetriken sind deshalb dort nicht messbar. Bei Hugging Face benötigt
die Strukturanalyse pro Layer zusätzliche Forward-Pässe und kann auf CPU
entsprechend lange dauern.
