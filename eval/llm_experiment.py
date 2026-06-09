"""
LLM Reranker Experiment: Testet LLM-basierte Pipelines auf xlarge-Chunks.
Vergleicht Pipelines mit und ohne LLM-Reranking.
Misst Recall@k für k in RECALL_K um den optimalen Kandidaten-Pool für den LLM-Reranker zu bestimmen."""
import json
import os
import sys
import time
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import (
    EVAL_BOOKS, RECALL_K, LLM_TOP_K,
    CHUNK_MIN_WORDS, CHUNK_MAX_WORDS, CHUNK_OVERLAP,
    DIR_PROCESSED, DIR_EXPERIMENTS,
)
from src.retrieval.pipeline import SearchPipeline as AblationSearchPipeline

CHUNK_CONFIG = {"name": "xlarge", "min_words": CHUNK_MIN_WORDS, "max_words": CHUNK_MAX_WORDS, "overlap": CHUNK_OVERLAP}

PIPELINE_CONFIGS = [
    {"name": "TF-IDF+Emb",           "tfidf": True, "emb": True, "rerank": False, "nli": False, "llm": False},
    {"name": "TF-IDF+Emb+Rerank",    "tfidf": True, "emb": True, "rerank": True,  "nli": False, "llm": False},
    {"name": "TF-IDF+Emb+LLM",       "tfidf": True, "emb": True, "rerank": False, "nli": False, "llm": True},
    {"name": "TF-IDF+Emb+Rerank+LLM","tfidf": True, "emb": True, "rerank": True,  "nli": False, "llm": True},
]


def evaluate_pipeline(pipeline, eval_pairs, use_llm: bool = False):
    """
    Wertet die Pipeline aus und gibt Recall@k für alle k in RECALL_K zurück.
    LLM-Pipelines bekommen top_k=LLM_TOP_K, alle anderen max(RECALL_K).
    """
    top_k = LLM_TOP_K if use_llm else max(RECALL_K)
    hits_at_k = {k: 0 for k in RECALL_K}
    first_hit_ranks = []
    query_times = []
    trials = []

    for pair in eval_pairs:
        query = pair["summary"]
        expected_spans = [tuple(s) for s in pair["spans"]]
        expected_text = pair.get("text", "")

        try:
            t0 = time.perf_counter()
            results = pipeline.search(query, top_k=top_k, verbose=False)
            elapsed = time.perf_counter() - t0
            query_times.append(elapsed)

            first_hit_rank = None
            for rank, r in enumerate(results, start=1):
                result_start = r.meta["start_index"]
                result_end = r.meta["end_index"]
                for expected_start, expected_end in expected_spans:
                    if result_start < expected_end and expected_start < result_end:
                        first_hit_rank = rank
                        break
                if first_hit_rank is not None:
                    break

            if first_hit_rank is not None:
                first_hit_ranks.append(first_hit_rank)
                for k in RECALL_K:
                    if first_hit_rank <= k:
                        hits_at_k[k] += 1

            top1 = results[0] if results else None
            trials.append({
                "query": query,
                "expected_text": expected_text,
                "expected_spans": [list(s) for s in expected_spans],
                "hit": first_hit_rank is not None,
                "first_hit_rank": first_hit_rank,
                "top1_text": top1.text if top1 else None,
                "top1_span": [top1.meta["start_index"], top1.meta["end_index"]] if top1 else None,
                "time_s": round(elapsed, 3),
            })

        except Exception as e:
            print(f"      Fehler bei Query: {e}")
            trials.append({
                "query": query,
                "expected_text": expected_text,
                "expected_spans": [list(s) for s in expected_spans],
                "hit": False,
                "first_hit_rank": None,
                "top1_text": None,
                "top1_span": None,
                "error": str(e),
            })

    total = len(eval_pairs)
    recall_at_k = {k: hits_at_k[k] / total if total > 0 else 0.0 for k in RECALL_K}
    avg_rank = sum(first_hit_ranks) / len(first_hit_ranks) if first_hit_ranks else 0.0
    total_time = sum(query_times)
    avg_time = total_time / len(query_times) if query_times else 0.0
    return hits_at_k, total, avg_rank, recall_at_k, total_time, avg_time, trials


def main():
    eval_dir   = DIR_PROCESSED
    chunks_dir = DIR_EXPERIMENTS
    chunk_name = CHUNK_CONFIG["name"]

    print("=" * 80)
    print("LLM RERANKER EXPERIMENT")
    print("=" * 80)
    print(f"Chunk-Größe: {chunk_name}  (min={CHUNK_CONFIG['min_words']}, max={CHUNK_CONFIG['max_words']})")
    print(f"Pipelines: {[p['name'] for p in PIPELINE_CONFIGS]}\n")

    eval_pairs_per_book = {}
    for book in EVAL_BOOKS:
        ep_path = os.path.join(eval_dir, f"eval_pairs_{book}.json")
        if not os.path.exists(ep_path):
            print(f"  WARNING: {ep_path} nicht gefunden, überspringe {book}")
            continue
        with open(ep_path, encoding="utf-8") as f:
            eval_pairs_per_book[book] = json.load(f)
        print(f"  {book}: {len(eval_pairs_per_book[book])} Pairs geladen")

    available_books = list(eval_pairs_per_book.keys())
    results_all = []

    for pipe_config in PIPELINE_CONFIGS:
        pipe_name = pipe_config["name"]
        print(f"\n{'='*80}")
        print(f"Pipeline: {pipe_name}")
        print("=" * 80)

        pipe_hits_at_k = {k: 0 for k in RECALL_K}
        pipe_total = 0
        pipe_ranks = []
        pipe_total_time = 0.0
        per_book_metrics = {}

        for book in available_books:
            chunks_path = os.path.join(chunks_dir, f"chunks_{book}_{chunk_name}.json")
            emb_path    = os.path.join(chunks_dir, f"embeddings_{book}_{chunk_name}.npy")

            if not os.path.exists(chunks_path) or not os.path.exists(emb_path):
                print(f"  [{book}] Chunks/Embeddings nicht gefunden, überspringe")
                continue

            pairs = eval_pairs_per_book[book]

            try:
                pipeline = AblationSearchPipeline(
                    chunks_path, emb_path,
                    use_tfidf=pipe_config["tfidf"],
                    use_embeddings=pipe_config["emb"],
                    use_reranker=pipe_config["rerank"],
                    use_nli=pipe_config["nli"],
                    use_llm=pipe_config["llm"],
                )

                hits_at_k, total, avg_rank, recall_at_k, total_time, avg_time, trials = evaluate_pipeline(
                    pipeline, pairs, use_llm=pipe_config["llm"]
                )

                per_book_metrics[book] = {
                    "total": total,
                    "avg_rank": avg_rank,
                    "recall_at_k": {str(k): v for k, v in recall_at_k.items()},
                    "hits_at_k": {str(k): hits_at_k[k] for k in RECALL_K},
                    "total_time_s": round(total_time, 2),
                    "avg_time_per_query_s": round(avg_time, 3),
                    "trials": trials,
                }

                for k in RECALL_K:
                    pipe_hits_at_k[k] += hits_at_k[k]
                pipe_total += total
                pipe_total_time += total_time
                if avg_rank > 0:
                    pipe_ranks.extend([avg_rank] * hits_at_k[max(RECALL_K)])

                recall_str = "  ".join(f"R@{k}={recall_at_k[k]:.1%}" for k in RECALL_K)
                print(f"  [{book}]  {recall_str}  avg_rank={avg_rank:.2f}  {avg_time:.2f}s/query")

            except Exception as e:
                print(f"  [{book}] FAILED: {e}")
                import traceback; traceback.print_exc()

        agg_recall_at_k = {k: pipe_hits_at_k[k] / pipe_total if pipe_total > 0 else 0.0 for k in RECALL_K}
        agg_avg_rank = sum(pipe_ranks) / len(pipe_ranks) if pipe_ranks else 0.0
        agg_avg_time = pipe_total_time / pipe_total if pipe_total > 0 else 0.0

        recall_str = "  ".join(f"R@{k}={agg_recall_at_k[k]:.1%}" for k in RECALL_K)
        print(f"  {'GESAMT':<15} {recall_str}  avg_rank={agg_avg_rank:.2f}  {agg_avg_time:.2f}s/query  total={pipe_total_time:.0f}s")

        results_all.append({
            "pipeline_config": pipe_config,
            "aggregate": {
                "total": pipe_total,
                "avg_rank": agg_avg_rank,
                "recall_at_k": {str(k): agg_recall_at_k[k] for k in RECALL_K},
                "hits_at_k": {str(k): pipe_hits_at_k[k] for k in RECALL_K},
                "total_time_s": round(pipe_total_time, 2),
                "avg_time_per_query_s": round(agg_avg_time, 3),
            },
            "per_book": per_book_metrics,
        })

    ts = datetime.now().isoformat()

    # Zusammenfassung (Konsole + TXT)
    summary_lines = [
        "=" * 80,
        "LLM RERANKER EXPERIMENT — ZUSAMMENFASSUNG",
        f"Datum:       {ts[:19]}",
        f"Chunk-Größe: {CHUNK_CONFIG['name']}  "
        f"(min={CHUNK_CONFIG['min_words']}, max={CHUNK_CONFIG['max_words']}, overlap={CHUNK_CONFIG['overlap']})",
        "=" * 80,
    ]
    k_header = "  ".join(f"R@{k:>2}" for k in RECALL_K)
    summary_lines.append(f"{'Pipeline':<28}  {k_header}  {'AvgRank':>7}  {'s/query':>8}  {'Total':>8}")
    summary_lines.append("-" * 80)
    for r in results_all:
        agg = r["aggregate"]
        k_vals = "  ".join(f"{agg['recall_at_k'][str(k)]:>5.1%}" for k in RECALL_K)
        summary_lines.append(
            f"{r['pipeline_config']['name']:<28}  {k_vals}"
            f"  {agg['avg_rank']:>7.2f}  {agg['avg_time_per_query_s']:>7.2f}s  {agg['total_time_s']:>6.0f}s"
        )
    summary_lines.append("=" * 80)

    print("\n" + "\n".join(summary_lines))

    RESULTS_DIR = os.path.join(project_root, "eval", "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 1. Zusammenfassung als TXT ---
    txt_path = os.path.join(RESULTS_DIR, "llm_experiment_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"Zusammenfassung gespeichert: {txt_path}")

    # --- 2. Details-JSON (je Versuch) ---
    details = {
        "timestamp": ts,
        "chunk_config": CHUNK_CONFIG,
        "pipelines": [
            {
                "name": r["pipeline_config"]["name"],
                "config": r["pipeline_config"],
                "aggregate": r["aggregate"],
                "books": [
                    {
                        "book": book,
                        "metrics": {k: v for k, v in bm.items() if k != "trials"},
                        "trials": bm.get("trials", []),
                    }
                    for book, bm in r["per_book"].items()
                ],
            }
            for r in results_all
        ],
    }
    details_path = os.path.join(RESULTS_DIR, "llm_experiment_details.json")
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print(f"Details gespeichert:         {details_path}")

    # --- 3. Aggregate-JSON (wie bisher) ---
    aggregate_results = [
        {k: v for k, v in r.items() if k != "per_book"}
        | {"per_book": {book: {m: val for m, val in bm.items() if m != "trials"}
                        for book, bm in r["per_book"].items()}}
        for r in results_all
    ]
    output_path = os.path.join(DIR_EXPERIMENTS, "llm_experiment_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "chunk_config": CHUNK_CONFIG, "results": aggregate_results},
                  f, ensure_ascii=False, indent=2)
    print(f"Aggregate gespeichert:       {output_path}")


if __name__ == "__main__":
    main()
