"""
Interaktives Such-Skript.
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.retrieval.pipeline import SearchPipeline, PIPELINE_PRESETS
from config import SEARCH_PIPELINE


def run_search_interactive(chunks_path: str, emb_path: str,
                           preset: str = SEARCH_PIPELINE,
                           top_k: int = 10):
    """Interaktive Suche mit konfigurierbarer Pipeline."""
    pipeline = SearchPipeline(chunks_path, emb_path, preset=preset)

    print(f"Pipeline: {pipeline.get_config_name()}")
    print("'exit' zum Beenden.\n")

    COLOR_CYAN  = "\033[96m"
    COLOR_YELLOW = "\033[93m"
    COLOR_RESET = "\033[0m"

    while True:
        query = input("Query: ").strip()
        if not query or query.lower() == "exit":
            break

        results = pipeline.search(query, top_k=top_k, verbose=True)

        print("\nTop-Ergebnisse:\n")
        for rank, r in enumerate(results, start=1):
            c = r.meta
            color = COLOR_CYAN if rank % 2 == 1 else COLOR_YELLOW
            score_str = f"{r.entailment:.3f} ({r.label[0]})" if hasattr(r, "entailment") else f"{r.score:.3f}"
            print(f"{color}{rank}. Chunk-ID={c['id']} | Start={c['start_index']} | {score_str}")
            print(r.text.replace("\n", " ")[:500])
            print("-" * 80 + COLOR_RESET)


def main():
    if len(sys.argv) < 3:
        print("Verwendung: python script/run_search.py <chunks> <embeddings> [Optionen]")
        print(f"Optionen: --preset=NAME  --top-k=N")
        print(f"Presets:  {', '.join(PIPELINE_PRESETS)}")
        sys.exit(1)

    chunks_path = sys.argv[1]
    emb_path    = sys.argv[2]
    args        = sys.argv[3:]

    preset = SEARCH_PIPELINE
    top_k  = 10
    for a in args:
        if a.startswith("--preset="):
            preset = a.split("=", 1)[1]
        elif a.startswith("--top-k="):
            top_k = int(a.split("=")[1])

    run_search_interactive(chunks_path, emb_path, preset=preset, top_k=top_k)


if __name__ == "__main__":
    main()

