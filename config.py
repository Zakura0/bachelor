"""
Zentrale Konfiguration für alle Skripte.
Alle Modelle, Pipeline-Parameter und Pfade werden hier festgelegt.
"""
import os

# ---------------------------------------------------------------------------
# Verzeichnisse
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DIR_RAW        = os.path.join(PROJECT_ROOT, "data/raw")
DIR_PROCESSED  = os.path.join(PROJECT_ROOT, "data/processed")
DIR_EXPERIMENTS= os.path.join(PROJECT_ROOT, "data/experiments/chunk_sizes")

# ---------------------------------------------------------------------------
# Modelle
# ---------------------------------------------------------------------------

# Embedding-Modell (multilinguales Sentence-Transformer Modell)
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Cross-Encoder für Reranking
RERANKER_MODEL  = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# NLI-Modell (normalerweise deaktiviert — verschlechtert Ergebnisse bei großen Chunks)
NLI_MODEL       = "joeddav/xlm-roberta-large-xnli"

# OpenAI-Modell für LLM-Reranking
LLM_MODEL       = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Pipeline-Parameter
# ---------------------------------------------------------------------------

# Schritt 1 — Erste Retrieval-Stufe: wie viele Kandidaten holt TF-IDF bzw. Embeddings?
K_RETRIEVAL = 200

# Schritt 2 — RRF Fusion: wie viele Kandidaten nach der Fusion weitergegeben werden
K_RRF       = 300

# Schritt 3 — Cross-Encoder Reranker: wie viele Kandidaten reranked und weitergegeben werden
K_RERANKER  = 30

# RRF Hyperparameter (höher = weniger Einfluss der genauen Ränge)
RRF_K       = 60

# ---------------------------------------------------------------------------
# Chunking — beste Konfiguration aus dem Chunk-Experiment (xlarge)
# ---------------------------------------------------------------------------

CHUNK_MIN_WORDS      = 80
CHUNK_MAX_WORDS      = 200
CHUNK_OVERLAP        = 3

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Bücher die evaluiert werden (harrypotter ausgeschlossen)
EVAL_BOOKS = [
    "verwandlung",
    "erdbeben",
    "judenbuche",
    "krambambuli",
]

# Recall@k Werte die gemessen werden
RECALL_K = [1, 3, 5, 10, 20]

# Kandidatenpool für den LLM-Reranker (aus Recall@k Analyse: top-20 optimal)
LLM_TOP_K = 20

# Pipeline für die interaktive Suche (main.py)
# 1 = TF-IDF only
# 2 = Embeddings only
# 3 = TF-IDF + Embeddings (RRF)
# 4 = TF-IDF + Embeddings + Reranker
# 5 = TF-IDF + Embeddings + Reranker + NLI
# 6 = TF-IDF + Embeddings + LLM
# 7 = TF-IDF + Embeddings + Reranker + LLM
SEARCH_PIPELINE = 4

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

# Chunk-Größe für das Experiment
EXP_CHUNK_MIN_WORDS = 80
EXP_CHUNK_MAX_WORDS = 200
EXP_CHUNK_OVERLAP   = 3
EXP_CHUNK_NAME      = "xlarge"   # Nur für Dateinamen / Logs

# Pipeline für das Experiment (siehe SEARCH_PIPELINE für mögliche Werte)
EXP_PIPELINE = 7

# Wie viele Kandidaten bekommt der LLM?
EXP_LLM_TOP_K   = 20

# Kandidaten je Retrieval-Stufe (TF-IDF / Emb)
EXP_K_RETRIEVAL = 200
# Kandidaten nach RRF-Fusion
EXP_K_RRF       = 300
# Kandidaten nach Cross-Encoder
EXP_K_RERANKER  = 30

# Wenn True: vorhandene Chunks/Embeddings wiederverwenden, sonst neu bauen
EXP_REUSE_EXISTING = True
