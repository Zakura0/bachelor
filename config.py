"""
Zentrale Konfiguration für alle Skripte.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Verzeichnisse
# ---------------------------------------------------------------------------

DIR_PROCESSED  = os.path.join(PROJECT_ROOT, "data/processed")
DIR_EXPERIMENTS = os.path.join(PROJECT_ROOT, "data/experiments/chunk_sizes")

# ---------------------------------------------------------------------------
# Modelle
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
RERANKER_MODEL  = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
NLI_MODEL       = "joeddav/xlm-roberta-large-xnli"   # deaktiviert — schadet bei großen Chunks
LLM_MODEL       = "gpt-4o-mini"

# Prompt-Templates sind fest in den Experiment-Skripten definiert:
# experiment.py          → PROMPT_RANK_ALL
# experiment_llm_pick.py → PROMPT_PICK_ONE
# (definiert in src/retrieval/llm_reranker.py)

# ---------------------------------------------------------------------------
# Pipeline-Parameter  (genutzt von src/retrieval/pipeline.py)
# ---------------------------------------------------------------------------

K_RETRIEVAL = 200   # Kandidaten je Retrieval-Stufe (TF-IDF / Embeddings)
K_RRF       = 300   # Kandidaten nach RRF-Fusion
K_RERANKER  = 50     # Kandidaten nach Cross-Encoder
RRF_K       = 60    # RRF-Hyperparameter (höher = weniger Einfluss der Ränge)

# ---------------------------------------------------------------------------
# Suche  (main.py → script/run_search.py)
# ---------------------------------------------------------------------------

# Verfügbare Presets → src/retrieval/pipeline.py :: PIPELINE_PRESETS
# 1 = TF-IDF only                    5 = TF-IDF + Emb + Reranker + NLI
# 2 = Embeddings only                6 = TF-IDF + Emb + LLM
# 3 = TF-IDF + Embeddings (RRF)      7 = TF-IDF + Emb + Reranker + LLM
# 4 = TF-IDF + Embeddings + Reranker
SEARCH_PIPELINE = 4

# ---------------------------------------------------------------------------
# Evaluation  (eval/experiment.py)
# ---------------------------------------------------------------------------

# EVAL_BOOKS = ["verwandlung", "erdbeben", "judenbuche", "krambambuli"]
EVAL_BOOKS = ["verwandlung"]
RECALL_K   = [1, 30]

# Chunk-Preset → src/preprocessing/chunk_presets.py :: CHUNK_PRESETS
# 1 = tiny (10–30)    3 = medium (30–100)    5 = xlarge (80–200)
# 2 = small (10–50)   4 = large (50–150)     6 = medium_high_overlap   7 = large_high_overlap
EXP_CHUNK    = 4

# Pipeline-Preset (siehe Kommentar bei SEARCH_PIPELINE)
EXP_PIPELINE = 7

# Pipeline-Preset für LLM-Pick-Experiment (eval/experiment_llm_pick.py)
# Muss ein Preset mit use_llm=True sein (6 oder 7)
EXP_LLM_PIPELINE = 7
