# Bachelor Thesis — Summary-Source Alignment

A multi-stage information retrieval pipeline for literary texts, built as part of a bachelor thesis. The system searches text chunks from literary works using a combination of sparse retrieval, dense embeddings, and neural reranking.

---

## Retrieval Pipeline

The pipeline is configurable via 10 presets defined in `config.py`. The default (preset 7) runs all stages:

```
┌─────────────────────────┐   ┌──────────────────────────────────┐
│  TF-IDF / BM25          │   │  Dense Embeddings (E5-Large)     │
│  Sparse Retrieval       │   │  ± HyDE  ± Multi-Query           │
│  → top K_RETRIEVAL=200  │   │  → top K_RETRIEVAL=200           │
└──────────┬──────────────┘   └──────────────┬───────────────────┘
           │                                 │
           └──────────┐  ┌───────────────────┘
                      ▼  ▼
              Reciprocal Rank Fusion (RRF)
                  → K_RRF=300 candidates
                      │
                      ▼
              CrossEncoder Reranker
              (BAAI/bge-reranker-v2-m3)
                  → K_RERANKER=30 candidates
                      │
             ┌────────┴────────┐
             ▼                 ▼
         NLI Filter       LLM Reranker
         (optional)       (Gemma 31B, optional)
             └────────┬────────┘
                      ▼
                 Final top-k results
```

### Stage details

1. **Chunking** — Regex sentence segmentation with configurable word-budget windows (7 presets: `tiny` 10–30 words → `xlarge` 80–200 words, with sentence overlap).
2. **Sparse Retrieval** — TF-IDF (`sklearn`) or BM25 (`bm25s`) ranked by cosine similarity.
3. **Dense Retrieval** — `intfloat/multilingual-e5-large` (1024-dim). Optionally augmented with:
   - **HyDE**: LLM-generated hypothetical passages used as query proxies.
   - **Multi-Query**: LLM-generated paraphrases fused via RRF.
4. **RRF Fusion** — Combines sparse and dense ranked lists (hyperparameter `RRF_K=60`).
5. **CrossEncoder Reranking** — `BAAI/bge-reranker-v2-m3` scores every (query, chunk) pair jointly.
6. **NLI Verification** *(optional)* — `joeddav/xlm-roberta-large-xnli` filters by entailment score.
7. **LLM Reranking** *(optional)* — OpenAI-compatible API call to Gemma 31B either picks the single best passage or ranks all candidates.

---

## Project Structure

```
bachelor/
├── config.py                   # All model names, pipeline parameters, experiment settings
├── main.py                     # Interactive CLI: search + run experiments
├── requirements.txt
│
├── src/
│   ├── preprocessing/
│   │   ├── book_chunker.py     # Sentence segmentation + word-budget chunking
│   │   └── chunk_presets.py    # 7 named chunk size presets
│   └── retrieval/
│       ├── pipeline.py         # SearchPipeline: orchestrates all stages
│       ├── tfidf.py            # TF-IDF sparse retriever
│       ├── bm25.py             # BM25 sparse retriever
│       ├── embeddings.py       # Dense retriever (SentenceTransformer)
│       ├── reranker.py         # CrossEncoder reranker
│       ├── nli.py              # NLI entailment filter
│       ├── llm_reranker.py     # LLM pick-one / rank-all reranker
│       ├── hyde.py             # HyDE hypothetical passage generator
│       └── multi_query.py      # Multi-query paraphrase generator
│
├── db/
│   ├── database.py             # SQLite schema: books, chunks, embeddings
│   └── migrate.py              # Populates DB from data/ files
│
├── script/
│   ├── build_book_chunks.py    # raw text → chunks JSON
│   ├── build_embeddings.py     # chunks JSON → .npy embeddings
│   └── run_search.py           # standalone colorised search CLI
│
├── eval/
│   ├── parse_data.py           # data.json → eval_pairs_{book}.json
│   ├── experiment.py           # Recall@k evaluation across pipeline presets
│   ├── experiment_llm_pick.py  # LLM single-pick accuracy evaluation
│   ├── experiment_chunk_ablation.py  # Recall@k über verschiedene Chunk-Größen
│   ├── llm_fulltext_experiment.py  # Baseline: LLM given full book text
│   ├── eval_utils.py           # Shared helpers (timestamped result dirs, misses)
│   └── results/                # Timestamped experiment outputs
│       ├── experiment/
│       ├── experiment_llm_pick/
│       ├── chunk_ablation/
│       └── llm_fulltext/
│
├── backend/
│   ├── main.py                 # FastAPI app (CORS: localhost:5173/5174, serves frontend/dist)
│   └── routers/
│       ├── books.py            # Buch-CRUD, Cover-Upload, Chunking + Indexierung (SSE)
│       └── search.py           # Suche über die Pipeline (SSE-Streaming + einfacher Endpoint)
│
├── frontend/                   # React + Vite + Tailwind SPA
│   └── src/
│       ├── App.tsx
│       └── components/         # BookPicker, SearchView, UploadView, PresetBuilder, ...
│
└── data/
    ├── raw/                    # Original .txt files + data.json ground truth
    ├── processed/              # eval_pairs_{book}.json (parsed annotations)
    ├── chunks/                 # {book}/{preset}.json
    └── embeddings/e5-large/    # {book}/{preset}.npy  (float32, 1024-dim)
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Requirements:** `torch`, `sentence-transformers`, `scikit-learn`, `transformers`, `sentencepiece`, `protobuf`, `numpy`, `openai`, `bm25s`, `fastapi`, `pydantic`, `python-multipart`, `uvicorn`

The LLM reranker and HyDE/Multi-Query components require an OpenAI-compatible API endpoint (configured in `config.py`) and an `OPENAI_API_KEY` environment variable.

### Frontend (optional Web-UI)

```bash
cd frontend
npm install
npm run dev
```

---

## Usage

### Interactive CLI

```bash
python main.py
```

Menu options:
- **Suche** — Select a book, load/build chunks and embeddings, run interactive queries.
- **Experimente** — Run one of three evaluation experiments (see below).

### Standalone search script

```bash
python script/run_search.py
```

### Web API

```bash
uvicorn backend.main:app --reload
```

Bücher (`/api/books`):
- `GET /` — Liste aller Bücher
- `GET /{book_id}/presets` — Verfügbare Chunk-Presets für ein Buch
- `POST /` — Neues Buch anlegen (Upload einer .txt-Datei)
- `POST /create-and-index` — Buch anlegen und erstes Preset indexieren (SSE-Fortschritt)
- `POST /{book_id}/index` — Weiteres Chunk-Preset für ein Buch indexieren (SSE-Fortschritt)
- `GET /{book_id}/text` — Rohtext eines Buchs
- `GET`/`POST /{book_id}/cover` — Cover-Bild abrufen/hochladen
- `DELETE /{book_id}` — Buch inkl. Chunks/Embeddings löschen

Suche (`/api/search`):
- `POST /stream` — Suche mit Fortschritts-Updates (SSE)
- `POST /` — Suche ohne Streaming

Das gebaute Frontend (`frontend/dist`) wird automatisch mitausgeliefert, sofern vorhanden.

---

## Evaluation

Ground truth: `data/raw/data.json` — abstractive summaries paired with character-level offsets of the source passages.

Preprocessing: `python eval/parse_data.py` converts `data.json` into per-book `eval_pairs_{book}.json`.

### Experiment 1 — Recall@k (`eval/experiment.py`)

Runs the full pipeline over all eval pairs, checks whether the expected passage overlaps with the top-k results.

**Metrics:** Recall@{1, 5, 10, 20, 30}, average rank, latency  
**Output:** `eval/results/experiment/{timestamp}/summary.txt`, `results.json`, `misses.json`

### Experiment 2 — LLM Pick Accuracy (`eval/experiment_llm_pick.py`)

Pipeline retrieves candidates; LLM selects exactly one. Measures how often the correct passage is chosen.

**Metrics:** Accuracy  
**Output:** `eval/results/experiment_llm_pick/{timestamp}/`

### Experiment 3 — LLM Fulltext Baseline (`eval/llm_fulltext_experiment.py`)

LLM receives the entire book text and must locate the relevant passage without any retrieval pipeline.

**Metrics:** Hit-rate, Recall@k, latency  
**Output:** `eval/results/llm_fulltext/{timestamp}/`

### Experiment 4 — Chunk-Size Ablation (`eval/experiment_chunk_ablation.py`)

Runs Experiment 1's pipeline (preset 7) across all chunk-size presets to compare Recall@k by chunk size.

**Metrics:** Recall@{1, 5, 10, 20, 30}, average rank, latency (per chunk-size preset)  
**Output:** `eval/results/chunk_ablation/{timestamp}/`

---

## Key Configuration (`config.py`)

| Parameter | Value | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Dense retriever |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | CrossEncoder reranker |
| `NLI_MODEL` | `joeddav/xlm-roberta-large-xnli` | NLI entailment filter |
| `LLM_MODEL` | `vllm/google/gemma-4-31B-it` | LLM reranker / HyDE / Multi-Query |
| `K_RETRIEVAL` | 200 | Candidates per retriever |
| `K_RRF` | 300 | Candidates after fusion |
| `K_RERANKER` | 30 | Candidates after CrossEncoder |
| `RRF_K` | 60 | RRF hyperparameter |
| `SEARCH_PIPELINE` | 7 | Default pipeline preset |
| `EXP_CHUNK` | 3 | Chunk preset used in experiments (`medium`, 30–100 words) |
| `EVAL_BOOKS` | verwandlung, erdbeben, judenbuche, krambambuli | Books evaluated |
