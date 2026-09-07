# Experimentergebnisse

## Ziel und Bewertungsgrundlage

Untersucht wurde eine Retrieval-Pipeline zur Zuordnung inhaltlicher Zusammenfassungen zu passenden Textpassagen in vier literarischen Werken: *Die Verwandlung*, *Das Erdbeben in Chili*, *Die Judenbuche* und *Krambambuli*. Die gemeinsame Evaluationsmenge der späteren Hauptversuche umfasst 936 Anfragen. Ein Treffer liegt vor, wenn der Zeichenbereich eines zurückgegebenen Chunks den annotierten Zielbereich mindestens teilweise ueberlappt.

Die zentrale Kennzahl ist Recall@$k$ (R@$k$): der Anteil der Anfragen, bei denen mindestens ein relevanter Chunk innerhalb der ersten $k$ Ergebnisse erscheint. R@1 beschreibt damit die Qualitaet des obersten Ergebnisses. Der mittlere Rang (AvgRank) wird nur ueber erfolgreiche Anfragen gebildet. Die Laufzeit bezeichnet die mittlere Dauer pro Anfrage.

## Kernergebnis

**Die beste Konfiguration fuer den Einsatz im System ist Pipeline 7 mit grossen Chunks und $k=30$.** Sie kombiniert TF-IDF, dichte Embeddings, Cross-Encoder-Reranking und ein LLM-Reranking. Mit 711 Treffern bei 936 Anfragen erreicht sie **76,0 % R@1** und **95,2 % R@30** bei **3,55 Sekunden pro Anfrage**. Der identische Lauf vom 23.08.2026 reproduziert die Qualitaetswerte des vorherigen Bestlaufs vom 28.07.2026 und dokumentiert dabei das verwendete LLM `vllm/google/gemma-4-31B-it`.

| Konfiguration | R@1 | R@5 | R@10 | R@20 | R@30 | AvgRank | Zeit/Anfrage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pipeline 7, large, $k=30$ | **76,0 %** | 89,4 % | 91,3 % | 93,7 % | 95,2 % | **2,13** | **3,55 s** |

Die Leistung variiert zwischen den Werken deutlich. *Die Verwandlung* ist mit 61,8 % R@1 der schwierigste, *Krambambuli* mit 83,6 % R@1 der leichteste Fall. Auch beim schwierigen Werk wird jedoch bei R@30 noch eine Abdeckung von 86,0 % erreicht.

| Werk | R@1 | R@30 | AvgRank | Zeit/Anfrage |
|---|---:|---:|---:|---:|
| Die Verwandlung | 61,8 % | 86,0 % | 2,35 | 4,52 s |
| Das Erdbeben in Chili | 82,8 % | 98,0 % | 2,15 | 2,98 s |
| Die Judenbuche | 72,5 % | 96,8 % | 2,61 | 4,40 s |
| Krambambuli | 83,6 % | 98,4 % | 1,56 | 2,61 s |

## Wichtigste Verbesserungen

### 1. Mehrstufige Pipeline statt Reranking ohne LLM

Die Erweiterung der hybriden Suche und des Cross-Encoder-Rerankings (Pipeline 4) um das LLM-Reranking (Pipeline 7) ist die wichtigste Verbesserung. Bei mittleren Chunks und $k=30$ steigt R@1 von 48,6 % auf 66,5 %, also um **17,9 Prozentpunkte**. R@30 verbessert sich von 92,9 % auf 95,2 % (bei Pipeline 7 mit grossen Chunks). Die Befunde zeigen vor allem einen Qualitaetsgewinn an der obersten Position, nicht nur eine groessere Kandidatenmenge.

### 2. Groessere Chunks verbessern die Top-1-Qualitaet

Unter Pipeline 7 und $k=30$ steigen die Ergebnisse beim Wechsel von mittleren auf grosse Chunks von 66,5 % auf **76,0 % R@1** (+9,5 Prozentpunkte). Sehr grosse Chunks erreichen 74,6 % R@1 und 96,7 % R@30. Sie maximieren damit die Kandidatenabdeckung, liegen bei der direkt nutzbaren ersten Antwort aber unter grossen Chunks. Fuer eine Anwendung, die primär ein einzelnes Ergebnis zeigt, sind grosse Chunks deshalb vorzuziehen.

### 3. $k=30$ ist der beste Kompromiss aus Qualitaet und Latenz

Bei grossen Chunks liefert $k=15$ 64,3 % R@1. Eine Erhoehung auf $k=30$ steigert R@1 auf 76,0 %. Eine weitere Erhoehung auf $k=40$ verschlechtert R@1 leicht auf 75,5 %, erreicht 95,9 % R@30 und 96,7 % R@40 und kostet rund 0,52 Sekunden mehr pro Anfrage. Daher ist $k=30$ der sinnvolle Standard; $k=40$ ist nur bei einer nachgelagerten manuellen Auswahl mit Prioritaet auf Vollstaendigkeit vertretbar.

### 4. Gemma ist in der vorhandenen Modellvergleichsmessung besser als GPT-OSS

Bei sonst gleicher dokumentierter Konfiguration (Pipeline 7, grosse Chunks, $k=30$) erzielt `gemma-4-31B-it` 76,0 % R@1 bei 3,77 s pro Anfrage. `gpt-oss-120b` erreicht 65,4 % R@1 bei 12,20 s. Das entspricht **+10,6 Prozentpunkten R@1** bei etwa **3,2-fach geringerer Laufzeit** fuer Gemma. Da pro Modell nur ein Lauf vorliegt, ist dies ein klarer Praxisbefund, aber noch kein abgesicherter Modellvergleich.

## Vergleich der Hauptversuche

Die folgende Tabelle fasst alle gespeicherten Laeufe der Reihe `experiment` zusammen. Nur Laeufe mit $n=936$ sind unmittelbar untereinander vergleichbar. Die fruehen Ein-Buch-Laeufe ($n=207$) werden als Vorstudien getrennt eingeordnet.

| Datum | Stichprobe | Pipeline | Chunks | $k$ | R@1 | R@30 | AvgRank | Zeit/Anfrage | Einordnung |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|
| 10.06.2026 00:10 | 207 | 4 | large | 100 | 30,4 % | - | 15,52 | 0,33 s | Vorstudie |
| 10.06.2026 00:14 | 207 | 4 | large | 150 | 30,4 % | - | 20,06 | 0,33 s | Vorstudie |
| 15.06.2026 18:17 | 207 | 7 | medium | 150 | 44,0 % | - | 18,54 | 23,44 s | Vorstudie |
| 16.06.2026 16:26 | 207 | 7 | medium | 30 | 41,5 % | 74,9 % | 4,13 | 2,34 s | Vorstudie |
| 16.06.2026 17:06 | 207 | 4 | medium | 30 | 20,8 % | 74,9 % | 7,43 | 0,29 s | Vorstudie |
| 09.07.2026 13:16 | 936 | 4 | medium | 30 | 36,7 % | 85,0 % | 6,07 | 2,33 s | Hauptreihe |
| 09.07.2026 13:31 | 936 | 4 | medium | 30 | 48,6 % | 92,9 % | 4,72 | 1,46 s | Hauptreihe |
| 13.07.2026 14:08 | 936 | 7 | medium | 30 | 56,3 % | 92,9 % | 3,66 | 3,60 s | Hauptreihe |
| 13.07.2026 15:40 | 936 | 8 | medium | 30 | 56,9 % | 93,2 % | 3,65 | 5,91 s | HyDE |
| 13.07.2026 15:41 | 936 | 9 | medium | 30 | 48,6 % | 93,1 % | 4,73 | 3,93 s | HyDE ohne LLM |
| 13.07.2026 17:40 | 936 | 7 | medium | 30 | 65,6 % | 92,9 % | 2,62 | 2,75 s | Hauptreihe |
| 17.07.2026 13:50 | 936 | 7 | medium | 15 | 64,3 % | - | 1,70 | 2,73 s | $k$-Ablation |
| 17.07.2026 17:12 | 936 | 7 | medium | 30 | 66,5 % | 92,9 % | 2,70 | 2,63 s | Hauptreihe |
| 24.07.2026 16:48 | 936 | 10 | medium | 30 | 65,4 % | 92,9 % | 2,72 | 4,46 s | Multi-Query |
| 25.07.2026 19:13 | 936 | 7 | large | 30 | 70,4 % | 95,2 % | 2,25 | 2,54 s | Chunk-Ablation |
| 25.07.2026 20:03 | 936 | 7 | xlarge | 30 | 74,6 % | **96,7 %** | 2,12 | 2,82 s | Beste Abdeckung |
| 28.07.2026 10:26 | 936 | 7 | large | 30 | **76,0 %** | 95,2 % | **2,13** | 3,75 s | Bester R@1 |
| 13.08.2026 21:38 | 936 | 7 | large | 30 | 65,4 % | 95,2 % | 2,70 | 12,20 s | GPT-OSS-120b |
| 14.08.2026 11:05 | 936 | 7 | large | 30 | **76,0 %** | 95,2 % | **2,13** | 3,77 s | Gemma |
| 14.08.2026 13:40 | 936 | 7 | large | 40 | 75,5 % | 95,9 % | 2,36 | 4,27 s | $k$-Ablation; R@40: **96,7 %** |
| 23.08.2026 20:46 | 936 | 7 | large | 30 | **76,0 %** | 95,2 % | **2,13** | **3,55 s** | Replikation und Empfehlung |

Pipeline 4 umfasst TF-IDF, Embeddings und Cross-Encoder-Reranking. Pipeline 7 erweitert diese um LLM-Reranking. Pipeline 8 verwendet zusaetzlich HyDE, Pipeline 9 HyDE ohne LLM und Pipeline 10 Multi-Query-Retrieval.

## Negative Befunde und verworfene Ansaetze

HyDE erbringt im vorliegenden Vergleich keinen Mehrwert: Pipeline 8 erreicht mit mittleren Chunks 56,9 % R@1 und 5,91 s pro Anfrage, waehrend Pipeline 7 bei vergleichbarer Einstellung 65,6 % R@1 und 2,75 s erreicht. Das sind 8,7 Prozentpunkte weniger R@1 bei mehr als doppelter Laufzeit. Multi-Query (Pipeline 10) liegt mit 65,4 % R@1 unter Pipeline 7 (66,5 %) und ist mit 4,46 s langsamer.

Zwei LLM-zentrierte Baselines fallen ebenfalls klar ab. Beim LLM-Pick waehlt das LLM aus den Kandidaten genau eine Passage aus und erreicht maximal 30,4 % Accuracy (63/207, grosse Chunks). Beim Fulltext-Ansatz erhielt `gpt-4o-mini` den kompletten Text von *Die Verwandlung* und erreichte 28,5 % Hit-Rate (59/207). Beide Versuche sind auf ein Werk beschraenkt und daher nicht direkt gegen die Vier-Buecher-Hauptreihe vergleichbar. Sie belegen dennoch, dass eine reine LLM-Auswahl beziehungsweise Volltextverarbeitung die mehrstufige Retrieval-Pipeline nicht ersetzt.

| Baseline | Stichprobe | Bestes Ergebnis | Zeit/Anfrage | Schlussfolgerung |
|---|---:|---:|---:|---|
| LLM-Pick, Pipeline 7, large | 207 | 30,4 % Accuracy | 1,10 s | Einzelentscheidung des LLM unzureichend |
| LLM-Pick, Pipeline 7, medium | 207 | 20,8 % Accuracy | 0,98 s | Kleinere Chunks nochmals schlechter |
| Fulltext-LLM, `gpt-4o-mini` | 207 | 28,5 % Hit-Rate | 2,33 s | Volltext ohne Retrieval unzureichend |

## Schlussfolgerung fuer die Arbeit

Die Experimente stützen eine mehrstufige hybride Architektur: TF-IDF und Embeddings liefern eine breite Kandidatenmenge, der Cross-Encoder verbessert deren Reihenfolge und das LLM-Reranking maximiert die Qualitaet des ersten Ergebnisses. Als finale Konfiguration sollte daher Pipeline 7 mit dem Embedding-Modell `intfloat/multilingual-e5-large`, grossen Chunks (50--150 Woerter, Ueberlappung 2), $k=30$ und `gemma-4-31B-it` berichtet werden. Sie bietet die beste beobachtete Top-1-Qualitaet bei zugleich nachvollziehbarer Latenz.

Fuer Anwendungsfaelle, in denen mehrere Treffer angezeigt und von Menschen geprueft werden, ist die xlarge-Konfiguration eine Alternative: Sie erreicht 96,7 % R@30, akzeptiert aber eine geringere Top-1-Qualitaet. Auch $k=40$ kann sinnvoll sein, wenn bis zu 40 Ergebnisse betrachtet werden; dieser Lauf erreicht 96,7 % R@40 bei hoeherer Laufzeit.

## Einschraenkungen und naechste Experimente

Die Hauptbefunde sind auf der gemeinsamen Vier-Buecher-Evaluation belastbar. Einige fruehe Laeufe sowie alle LLM-Pick- und Fulltext-Versuche verwenden dagegen nur *Die Verwandlung*; sie sollten in der Arbeit als explorative Vorstudien und nicht als gleichwertige Gesamtvergleiche bezeichnet werden. Zudem fehlen vollstaendige Konfigurationsprotokolle fuer mehrere fruehe Laeufe, weshalb Unterschiede zwischen Wiederholungen nicht ausschliesslich auf die getestete Komponente zurueckgefuehrt werden koennen.

Eine belastbare Erweiterung waere ein wiederholter, kontrollierter Modellvergleich mit mindestens zwei bis drei Durchlaeufen je Modell bei festgehaltenem Datensatz, Chunking und allen Pipeline-Parametern. Sinnvolle offene Ablationen sind ausserdem TF-IDF gegen BM25 sowie Reciprocal-Rank-Fusion gegen alternative Fusionsverfahren.

## Datenquellen

Die aggregierten Hauptdaten stammen aus den Unterordnern von `eval/results/experiment/`; je Lauf liegen `summary.txt` und `results.json` vor. Die Baselines sind in `eval/results/experiment_llm_pick/` und `eval/results/llm_fulltext/` gespeichert. Die Messlogik befindet sich in `eval/experiment.py`, `eval/experiment_llm_pick.py` und `eval/llm_fulltext_experiment.py`; die Pipeline-Presets sind in `src/retrieval/pipeline.py` definiert.