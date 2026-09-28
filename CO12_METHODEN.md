# Erklärbarkeit von LLMs anhand der Co-12-Eigenschaften

Dieses Dokument verwendet die Co-12-Eigenschaften nach Nauta et al.:
`Correctness`, `Completeness`, `Consistency`, `Continuity`, `Contrastivity`,
`Covariate Complexity`, `Compactness`, `Composition`, `Confidence`, `Context`,
`Coherence` und `Controllability`.

Dabei werden zwei Erklärungsobjekte unterschieden:

- **CoT-Erklärung:** ein vom Modell erzeugter natürlichsprachlicher
  Reasoning-Text.
- **Strukturelle Erklärung:** ein aus Aktivierungen, Attention Heads, MLPs
  oder SAE-Features gewonnener kausaler Circuit beziehungsweise Feature-Graph.

`Correctness` bezeichnet die Treue zum tatsächlichen Modellverhalten, nicht
die Richtigkeit der finalen Antwort. `Coherence` bezeichnet dagegen die
Übereinstimmung mit menschlichem Wissen oder Erwartungen. Eine plausible und
kohärente Erklärung kann trotzdem kausal untreu sein.

## Co-12 für Chain-of-Thought

| C | Vorgeschlagene Methode | Messung | Benchmark | Quelle/Status |
|---|---|---|---|---|
| **Correctness** | Kausale CoT-Interventionen: CoT kürzen, Fehler einfügen oder Bias-Informationen verändern und die Antwort neu erzeugen. | Answer-change rate, AOC der Early-Answering-Kurve und Anteil biasinduzierter Antworten, bei denen der Bias im CoT fehlt. | BBH, BBQ, AQuA, LogiQA | Lanham et al.; Turpin et al.; etabliert |
| **Completeness** | Gold-Proof-Coverage plus Sufficiency/Comprehensiveness. | Recall notwendiger Fakten und Beweiskanten; Leistungsverlust bei Nutzung nur der Erklärung beziehungsweise nach Entfernen erklärter Evidenz. | EntailmentBank, ProofWriter, ERASER | Dalvi et al.; DeYoung et al.; Adaption |
| **Consistency** | Für dieselbe Aufgabe mehrere CoTs mit verschiedenen Seeds und Temperaturen samplen und semantisch clustern. | Antwortentropie, paarweise semantische Ähnlichkeit und Übereinstimmung der verwendeten Fakten. | GSM8K, SVAMP, StrategyQA | Wang et al.; Adaption von Self-Consistency |
| **Continuity** | Semantisch äquivalente Paraphrasen und kleine irrelevante Eingabeänderungen erzeugen. | CoT-Ähnlichkeit oder Graph-Edit-Distance, bedingt darauf, dass die Modellantwort unverändert bleibt. | Paraphrasierte GSM8K- und BBH-Varianten | Lanham et al.; Nauta et al.; Adaption |
| **Contrastivity** | Minimal kontrastive Aufgabenpaare bilden, bei denen genau ein relevanter Fakt oder das Ziel verändert wird. | Korrekter Answer-Switch und Anteil der veränderten CoT-Schritte, die den kontrastiven Fakt betreffen. | Contrast Sets, BBQ, EntailmentBank | Turpin et al.; Adaption |
| **Covariate Complexity** | CoT in einen Reasoning-DAG aus Fakten, Variablen und Schlussfolgerungen umwandeln. | Konzeptanzahl, Fan-in, Interaktionstiefe, Graph-Tiefe und unnötige Abhängigkeiten. | EntailmentBank, ProofWriter | Dalvi et al.; Nauta et al.; Adaption |
| **Compactness** | Minimal-sufficient CoT durch schrittweise Löschung bestimmen. | Anzahl notwendiger Schritte und Tokens, Anteil löschbarer Schritte und Pareto-Kurve aus Länge und Sufficiency. | GSM8K, AQuA, EntailmentBank | Lanham et al.; Nauta et al.; Adaption |
| **Composition** | Sprachliche und logische Organisation mit ROSCOE und gezielten Reihenfolgefehlern prüfen. | Fluency, Grammar, Reasoning Alignment, Repetition, Perplexity und Erkennung vertauschter Schritte. | ROSCOE-Diagnostics, GSM8K, e-SNLI | Golovneva et al.; etabliert |
| **Confidence** | Konfidenz für jeden Schritt und die finale Antwort abfragen. | Brier Score, Expected Calibration Error, AUROC und Reliability Diagrams gegen annotierte Schrittkorrektheit. | CalibratedMath, GSM8K, EntailmentBank | Lin et al.; Adaption auf Schritte |
| **Context** | CoTs für unterschiedliche Nutzergruppen erzeugen und in einer Nutzeraufgabe testen. | Relevanzrating, Zeit, Fehlerquote, Vertrauenskalibrierung und Verbesserung der menschlichen Entscheidung. | Domänenspezifische Nutzerstudie | Nauta et al.; Doshi-Velez und Kim |
| **Coherence** | CoT mit menschlichen Rationales, Goldbeweisen und externen Fakten vergleichen. | Rationale-F1, Fact Precision, NLI-Entailment, Widerspruchsrate und ROSCOE-Factuality. | ERASER, e-SNLI, EntailmentBank | DeYoung et al.; Golovneva et al. |
| **Controllability** | Anforderungen wie maximale Schrittzahl, Detailniveau und ein- oder auszuschließende Fakten variieren. | Constraint-Satisfaction-Rate, gewünschte Konzeptabdeckung und Erhalt der Antwortqualität. | Kontrollierte EntailmentBank- oder GSM8K-Varianten | Nauta et al.; Adaption |

### Adaption-Erklärung

Adaption von Completeness (ERASER / EntailmentBank) | Statt einzelne Bildpixel abzudecken (Pixel Masking) werden bei CoT einzelne Sätze oder Rationales im Text-Prompt maskiert oder entfernt, um zu messen, wie sich die Modellgenauigkeit verändert (Sufficiency / Comprehensiveness). Bei logischen Aufgaben werden CoTs in explizite Beweisbäume (Entailment Trees) zerlegt

Adaption von Consistency (Wang et al. / Self-Consistency) | In der klassischen XAI prüft Konsistenz, ob identische Eingaben zu identischen Feature-Importances führen. Für CoT adaptieren Wang et al. dies, indem sie über stochastisches Decoding (Temperatur > 0) mehrere Denkpdate samplen und prüfen, wie stabil die finale Antwort über unterschiedliche semantische Erklärungswege hinweg bleibt 

Adaption von Continuity (Lanham et al.) | Klassische Kontinuität verlangt, dass geringfügiges Rauschen in Eingabevektoren die Erklärung nicht drastisch verändert. Bei CoT adaptiert Lanham et al. diesen Test durch Paraphrasierung von Denkschritten. Bleibt die Antwort unter verschiedenen Satzformulierungen stabil, ist die logische Substanz entscheidend und nicht ein versteckter Oberflächen-Bias

Adaption von Covariate Complexity (Dalvi et al.) | Statt Monotonie oder Feature-Interaktionen in Tabellendaten zu messen, wird die Textkette als gerichteter azyklischer Graph (DAG) modelliert. Die Komplexität wird nun über graphentheoretische Kennzahlen wie Fan-in, Tiefe und Pfadlängen der logischen Schlüsse berechnet.

Adaption von Confidence (Lin et al. / CalibratedMath) | Anstelle von mathematischen Softmax-Entropiewerten der Modell-Logits zeigt Lin et al., dass Modelle lernen können, ihre Unsicherheit direkt in natürlichen Zahlen oder Wörtern im Textausgabe-Strom auszudrücken (verbalized probability), was mittels Kalibrierungsmetriken (Brier Score, ECE) evaluiert wird

## Co-12 für die innere Modellstruktur

Als einheitliches Erklärungsobjekt wird ein **sparse causal circuit** verwendet:
Knoten sind Heads, MLP-Komponenten oder SAE-Features; Kanten repräsentieren
postulierten kausalen Informationsfluss.

| C | Vorgeschlagene Methode | Messung | Benchmark | Quelle/Status |
|---|---|---|---|---|
| **Correctness** | Activation beziehungsweise Path Patching zwischen Clean- und Counterfactual-Inputs. | Normalisierte Faithfulness und kausaler Logit-Effekt der erklärten Komponenten. | IOI, CounterFact, Greater-Than | Wang et al.; Meng et al.; etabliert |
| **Completeness** | Dieselben Komponentenmengen aus Circuit und Gesamtmodell entfernen. | Maximale normalisierte Differenz `|F(C\\K)-F(M\\K)|`; kleine Werte sind besser. | IOI, Tracr | Wang et al.; etabliert |
| **Consistency** | Circuit über Datenstichproben, Seeds und funktional ähnliche Modelle wiederholt extrahieren. | CKA, Jaccard der Knoten/Kanten und Rangkorrelation kausaler Effekte. | Mehrere Checkpoints, Tracr | Kornblith et al.; Adaption |
| **Continuity** | Circuit-Extraktion auf bedeutungserhaltenden Eingabeperturbationen wiederholen. | Jaccard beziehungsweise Kanten-F1 und Korrelation der kausalen Effekte bei stabilem Modelloutput. | IOI-Templates und Paraphrasen | Nauta et al.; Adaption |
| **Contrastivity** | Für zwei Zielantworten kontrastive Aktivierungsrichtungen oder Circuits bestimmen und intervenieren. | Zieltrennbarkeit und Verhältnis aus gewünschtem Logit-Effekt zu Nebenwirkungen. | IOI-Kontraste, CounterFact | Kim et al.; Meng et al.; Adaption |
| **Covariate Complexity** | Sparse Autoencoder auf interne Aktivierungen trainieren. | L0-Sparsity, Rekonstruktionsfehler, Konzept-Purity, Polysemantizität und Concept Leakage. | IOI, kontrollierte Konzepte | Cunningham et al.; etabliert |
| **Compactness** | Mit ACDC einen möglichst kleinen faithful Circuit suchen. | Anzahl und Anteil der Knoten/Kanten, Faithfulness-Sparsity-Pareto-Kurve und Minimality-Ablationen. | IOI, Greater-Than | Conmy et al.; Wang et al.; etabliert |
| **Composition** | Circuit als hierarchischen, semantisch beschrifteten Sparse-Feature-Graphen darstellen. | Graphgröße, Modularität, Hierarchietiefe sowie menschliche Simulationsgenauigkeit und Bearbeitungszeit. | Feature-Circuit-Aufgaben plus Nutzerstudie | Marks et al.; vorgeschlagene Operationalisierung |
| **Confidence** | Kausale Effekte über Beispiele, Bootstrap-Stichproben und Extraktions-Seeds schätzen. | Konfidenzintervalle, Vorzeichenstabilität, Seed-Varianz und Coverage auf bekannter Ground Truth. | Tracr, IOI | Kim et al.; Lindner et al.; Adaption |
| **Context** | Nutzerdefinierte Konzepte mit TCAV oder SAE-Features lokalisieren. | Abdeckung gewünschter Konzepte, Unterdrückung irrelevanter Konzepte und Nutzen in der konkreten Aufgabe. | Domänenspezifische Konzeptdaten | Kim et al.; etabliertes Prinzip |
| **Coherence** | Interne Features gegen menschlich verständliche Konzeptlabels testen. | Konzept-Purity, Precision/Recall der Aktivierungsbeispiele und Übereinstimmung vorhergesagter mit tatsächlicher Aktivierung. | Kontrollierte Konzeptdaten, IOI-Rollen | Bau et al.; Cunningham et al. |
| **Controllability** | Erklärte Features gezielt aktivieren, ablatieren oder editieren. | Edit Success, Generalisierung, Spezifität, Locality und Nebenwirkungsrate. | CounterFact, zsRE, IOI | Meng et al.; Marks et al.; etabliert |

### Erklärung Adaption 
Adaption von Correctness & Completeness (Causal Patching statt Input-Maskierung): Klassisch: Entfernen/Maskieren von Eingabepixeln oder Wörtern.MI-Adaption: Activation Patching. Aktivierungsvektoren aus einem Run mit sauberem Input (Clean) werden in einen Run mit verfälschtem Input (Counterfactual) injiziert. 

Vollständigkeit wird nicht mehr über Textabdeckung definiert, sondern dadurch, dass der ablatierte Circuit $C \setminus K$ exakt denselben Performanceabfall zeigt wie das Gesamtmodell $M \setminus K$4.

Adaption von Covariate Complexity (Auflösung von Superposition via SAEs):Klassisch: Reduktion der Anzahl der Input-Features oder Vermeidung komplexer Nichtlinearitäten.MI-Adaption: Sparse Autoencoder (SAE). Da einzelne Neuronen in Sprachmodellen oft polysemantisch sind (mehrere unzusammenhängende Konzepte überlagern / Superposition), adaptiert MI das Komplexitätskriterium, indem interne Aktivierungen in ein spärliches ($L_0$), monosemantisches Feature-Format überführt werden

Adaption von Compactness (ACDC & Sparsity-Pareto-Kurven):Klassisch: Zählen von Regeln in Entscheidungsbäumen oder Pfadlängen.MI-Adaption: Automated Circuit Discovery (ACDC). Kompaktheit wird als die minimale Anzahl aktiver Kanten und Knoten im Rechengraphen eines Transformers gemessen, die notwendig ist, um eine hohe Faithfulness aufrechtzuerhalten

Adaption von Consistency (CKA & Tracr Ground Truth):Klassisch: Determinismus von Erklärungsalgorithmen bei identischem Input.MI-Adaption: Vergleiche der extrahierten Schaltkreise über verschiedene Modell-Checkpoints hinweg mittels Centered Kernel Alignment (CKA) oder die Validierung gefundener Circuits gegen mathematisch garantierte Ziel-Circuits in kompilierten Transformern (Tracr)

Adaption von Controllability (Model Editing & Feature Steering):Klassisch: Interaktive Benutzeroberflächen, bei denen Anwender Regelschwellenwerte anpassen können.MI-Adaption: Chirurgisches Modell-Editing und Feature-Pruning. Die Erklärungssteuerung wird zu einer aktiven Intervention im Modell: Mit ROME werden Gewichte im MLP direkt editiert, mit SHIFT werden nicht-relevante SAE-Features ablatiert, um die Modell-Generalisierung gezielt zu verbessern


## Empfohlenes Benchmark-Design

| Stufe | Benchmarks | Zweck |
|---|---|---|
| Ground Truth | Tracr | Bekannte Transformer-Programme und bekannte interne Struktur. |
| Kontrollierte Mechanismen | IOI, Greater-Than | Patching, Circuit Discovery, Minimality und Continuity. |
| Nachvollziehbare Beweise | EntailmentBank, ProofWriter | Gold-Fakten und Gold-Beweisgraphen. |
| Natürliches CoT | GSM8K, SVAMP, AQuA | Überprüfbare Antworten und Rechenschritte, aber keine interne Ground Truth. |
| Faithfulness und Bias | BBH-Bias-Varianten, BBQ | Verdeckte Einflussfaktoren und nachträgliche Rationalisierung. |
| Eingriffe und Kontrolle | CounterFact, zsRE | Controllability, Contrastivity und Locality. |

Für einen ersten Open-Weight-Versuch eignet sich **Gemma 2 2B**, weil mit
Gemma Scope bereits Sparse Autoencoder verfügbar sind. Wenn die Qualität der
expliziten CoTs wichtiger ist, ist **Qwen2.5 7B Instruct** ein stärkerer
Ausgangspunkt, erfordert für SAE-Analysen aber mehr eigene Vorarbeit.

Die zwölf Werte sollten zunächst nicht zu einem Gesamtscore gemittelt werden.
Sie bilden ein zwölfdimensionales Qualitätsprofil und können miteinander in
Konflikt stehen.

## Quellen

1. Nauta et al. (2023), [From Anecdotal Evidence to Quantitative Evaluation Methods](https://doi.org/10.1145/3583558).
2. Lanham et al. (2023), [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702).
3. Turpin et al. (2023), [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388).
4. DeYoung et al. (2020), [ERASER](https://aclanthology.org/2020.acl-main.408/).
5. Golovneva et al. (2023), [ROSCOE](https://arxiv.org/abs/2212.07919).
6. Dalvi et al. (2021), [Explaining Answers with Entailment Trees](https://arxiv.org/abs/2104.08661).
7. Wang et al. (2023), [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171).
8. Lin et al. (2022), [Teaching Models to Express Their Uncertainty in Words](https://arxiv.org/abs/2205.14334).
9. Doshi-Velez and Kim (2017), [Towards a Rigorous Science of Interpretable Machine Learning](https://arxiv.org/abs/1702.08608).
10. Wang et al. (2022), [Interpretability in the Wild](https://arxiv.org/abs/2211.00593).
11. Meng et al. (2022), [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262).
12. Kornblith et al. (2019), [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html).
13. Kim et al. (2018), [Testing with Concept Activation Vectors](https://proceedings.mlr.press/v80/kim18d.html).
14. Cunningham et al. (2023), [Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600).
15. Conmy et al. (2023), [Towards Automated Circuit Discovery](https://arxiv.org/abs/2304.14997).
16. Marks et al. (2025), [Sparse Feature Circuits](https://arxiv.org/abs/2403.19647).
17. Bau et al. (2017), [Network Dissection](https://arxiv.org/abs/1704.05796).
18. Lindner et al. (2023), [Tracr](https://arxiv.org/abs/2301.05062).
