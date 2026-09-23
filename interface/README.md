# Interaktive Co-12-Oberfläche

Die Streamlit-Anwendung verbindet ein lokales Open-Weight-Modell mit den
beiden Auswertungspipelines. Sie bietet eine Live-Analyse freier Prompts und
eine Offline-Auswertung bereits gesammelter JSONL-Artefakte.

## Installation

```bash
python3 -m pip install -e '.[ui]'
```

## Start

```bash
streamlit run interface/app.py
```

Beim ersten Modelllauf lädt Hugging Face standardmäßig
`Qwen/Qwen2.5-1.5B-Instruct` herunter. Das Modell wird mit
`st.cache_resource` im Speicher gehalten und für weitere Prompts
wiederverwendet.

## Bedienung

1. Modell und Rechengerät in der Seitenleiste auswählen.
2. Schnell- oder vollständigen Modus auswählen.
3. Prompt eingeben.
4. Optional Referenzantwort, Referenzschritte und kontrollierte Gegenfakten
   ergänzen.
5. `Antwort erzeugen und prüfen` auswählen.
6. Ampel, Evidenzabdeckung und die zwölf aufklappbaren Einzelergebnisse prüfen.

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

Die Modellinferenz erfolgt lokal. Der erste Download kommt von Hugging Face.
Die Strukturanalyse benötigt pro Layer zusätzliche Forward-Pässe und kann auf
CPU entsprechend lange dauern. Für größere Modelle sollte eine CUDA-GPU oder
eine angepasste Quantisierung verwendet werden.
