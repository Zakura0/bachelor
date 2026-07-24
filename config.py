"""
Zentrale Konfiguration für alle Skripte.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Verzeichnisse

DIR_PROCESSED  = os.path.join(PROJECT_ROOT, "data/processed")
DIR_CHUNKS     = os.path.join(PROJECT_ROOT, "data/chunks")
DIR_EMBEDDINGS = os.path.join(PROJECT_ROOT, "data/embeddings/e5-large")

# Modelle

EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"
NLI_MODEL       = "joeddav/xlm-roberta-large-xnli"
LLM_MODEL       = "gpt-4o"

# Pipeline-Parameter

K_RETRIEVAL = 200   # Kandidaten je Retrieval-Stufe (TF-IDF / Embeddings)
K_RRF       = 300   # Kandidaten nach RRF-Fusion
K_RERANKER  = 30    # Kandidaten nach Cross-Encoder

USE_BM25 = True    # True = BM25 statt TF-IDF als erstes Retrieval-Stage

RRF_K       = 60    # RRF-Hyperparameter

# Suche

# Verfügbare Presets
# 1 = TF-IDF only                    
# 2 = Embeddings only                
# 3 = TF-IDF + Embeddings (RRF)
# 4 = TF-IDF + Embeddings + Reranker
# 5 = TF-IDF + Emb + Reranker + NLI
# 6 = TF-IDF + Emb + LLM
# 7 = TF-IDF + Emb + Reranker + LLM
SEARCH_PIPELINE = 4


# Experimente
EVAL_BOOKS = ["verwandlung", "erdbeben", "judenbuche", "krambambuli"]
RECALL_K   = [1, 5, 10, 20, 30]

# Chunk-Preset 
# 1 = tiny (10–30)    3 = medium (30–100)    5 = xlarge (80–200)
# 2 = small (10–50)   4 = large (50–150)     6 = medium_high_overlap   7 = large_high_overlap
EXP_CHUNK    = 3

# Pipeline-Preset
EXP_PIPELINE = 7

# Pipeline-Preset für LLM-Pick-Experiment
EXP_LLM_PIPELINE = 7
