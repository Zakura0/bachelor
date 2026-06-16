"""
Zentrale Konfiguration für alle Skripte.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Verzeichnisse

DIR_PROCESSED  = os.path.join(PROJECT_ROOT, "data/processed")
DIR_CHUNKS     = os.path.join(PROJECT_ROOT, "data/chunks")
DIR_EMBEDDINGS = os.path.join(PROJECT_ROOT, "data/embeddings")

# Modelle

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
RERANKER_MODEL  = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
NLI_MODEL       = "joeddav/xlm-roberta-large-xnli"
LLM_MODEL       = "gpt-4o-mini"

# Pipeline-Parameter  (src/retrieval/pipeline.py)

K_RETRIEVAL = 200   # Kandidaten je Retrieval-Stufe (TF-IDF / Embeddings)
K_RRF       = 300   # Kandidaten nach RRF-Fusion
K_RERANKER  = 150   # Kandidaten nach Cross-Encoder
RRF_K       = 60    # RRF-Hyperparameter

# Suche  (script/run_search.py)

# Verfügbare Presets
# 1 = TF-IDF only                    
# 2 = Embeddings only                
# 3 = TF-IDF + Embeddings (RRF)
# 4 = TF-IDF + Embeddings + Reranker
# 5 = TF-IDF + Emb + Reranker + NLI
# 6 = TF-IDF + Emb + LLM
# 7 = TF-IDF + Emb + Reranker + LLM
SEARCH_PIPELINE = 7


# Experimente  (eval/experiment.py)
# EVAL_BOOKS = ["verwandlung", "erdbeben", "judenbuche", "krambambuli"]
EVAL_BOOKS = ["verwandlung"]
RECALL_K   = [1, 5, 10, 150]

# Chunk-Preset 
# 1 = tiny (10–30)    3 = medium (30–100)    5 = xlarge (80–200)
# 2 = small (10–50)   4 = large (50–150)     6 = medium_high_overlap   7 = large_high_overlap
EXP_CHUNK    = 3

# Pipeline-Preset
EXP_PIPELINE = 7

# Pipeline-Preset für LLM-Pick-Experiment
EXP_LLM_PIPELINE = 7
