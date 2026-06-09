import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# SearchPipeline und Hilfsfunktionen leben in src/ — eval-Skripte importieren von dort
from src.retrieval.pipeline import SearchPipeline, SearchResult, rrf_fuse  # noqa: F401

# Alias für Rückwärtskompatibilität mit den Experiment-Skripten
AblationSearchPipeline = SearchPipeline
