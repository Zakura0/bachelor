"""
Chunk-Size-Ablation

Vergleicht alle Chunk-Größen-Presets unter Pipeline 7
(TF-IDF + Embeddings + Cross-Encoder-Reranker + LLM).
Misst Recall@k über alle Eval-Bücher, je Chunk-Größe.
"""
import json
import os
import sys
import time
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import (
    EVAL_BOOKS, RECALL_K,
    DIR_PROCESSED, DIR_CHUNKS, DIR_EMBEDDINGS,
    K_RETRIEVAL, K_RRF, K_RERANKER, RRF_K,
    EMBEDDING_MODEL, LLM_MODEL, USE_BM25,
)
from src.preprocessing.chunk_presets import CHUNK_PRESETS
from src.retrieval.pipeline import SearchPipeline
from eval.eval_utils import make_run_dir, build_misses, save_misses

PIPELINE_PRESET = 7  # TF-IDF + Embeddings + Reranker + LLM (beste Konfiguration, siehe eval/results/)

BASE_RESULTS_DIR = os.path.join(project_root, "eval", "results")


def prepare_book(book: str, chunk_name: str):
    """Pfade zu Chunks und Embeddings zurückgeben. Bricht ab wenn Dateien fehlen."""
    chunks_path = os.path.join(DIR_CHUNKS,     book, f"{chunk_name}.json")
    emb_path    = os.path.join(DIR_EMBEDDINGS, book, f"{chunk_name}.npy")

    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks nicht gefunden: {chunks_path}")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings nicht gefunden: {emb_path}")

    return chunks_path, emb_path


def run_trial(pipeline: SearchPipeline, pair: dict):
    """Einen Eval-Pair auswerten. Gibt Trial-Dict zurück."""
    query          = pair["summary"]
    expected_spans = [tuple(s) for s in pair["spans"]]
    expected_text  = pair.get("text", "")

    t0 = time.perf_counter()
    results = pipeline.search(query, top_k=K_RERANKER, verbose=False)
    elapsed = time.perf_counter() - t0

    first_hit_rank = None
    for rank, r in enumerate(results, start=1):
        for exp_start, exp_end in expected_spans:
            if r.meta["start_index"] < exp_end and exp_start < r.meta["end_index"]:
                first_hit_rank = rank
                break
        if first_hit_rank is not None:
            break

    top1 = results[0] if results else None
    return {
        "query":          query,
        "expected_text":  expected_text,
        "expected_spans": [list(s) for s in expected_spans],
        "hit":            first_hit_rank is not None,
        "first_hit_rank": first_hit_rank,
        "top1_text":      top1.text if top1 else None,
        "top1_span":      [top1.meta["start_index"], top1.meta["end_index"]] if top1 else None,
        "time_s":         round(elapsed, 3),
    }


def compute_recall(trials: list):
    total = len(trials)
    hits_at_k = {k: 0 for k in RECALL_K}
    ranks = []
    for t in trials:
        if t["first_hit_rank"] is not None:
            ranks.append(t["first_hit_rank"])
            for k in RECALL_K:
                if t["first_hit_rank"] <= k:
                    hits_at_k[k] += 1
    recall = {k: hits_at_k[k] / total if total else 0.0 for k in RECALL_K}
    avg_rank = sum(ranks) / len(ranks) if ranks else 0.0
    return recall, avg_rank, hits_at_k


def run_chunk_size(chunk_preset: int, eval_pairs_per_book: dict):
    """Führt das Recall-Experiment für eine Chunk-Größe über alle Bücher aus."""
    cfg  = CHUNK_PRESETS[chunk_preset]
    name = cfg["name"]
    print(f"\n{'='*80}")
    print(f"Chunk-Größe: {name}  (min={cfg['min_words']}, max={cfg['max_words']}, overlap={cfg['overlap']})")
    print(f"{'='*80}")

    per_book_results = {}
    for book, pairs in eval_pairs_per_book.items():
        print(f"  Buch: {book}")
        chunks_path, emb_path = prepare_book(book, name)
        pipeline = SearchPipeline(chunks_path, emb_path, preset=PIPELINE_PRESET)

        trials = []
        for i, pair in enumerate(pairs, 1):
            trial = run_trial(pipeline, pair)
            trials.append(trial)
            status = "✓" if trial["hit"] else "✗"
            rank   = trial["first_hit_rank"] or "-"
            print(f"    {i:>3}/{len(pairs)}  {status}  Rank={rank:<4}  {trial['query'][:55]}")

        recall, avg_rank, hits_at_k = compute_recall(trials)
        recall_str = "  ".join(f"R@{k}={recall[k]:.1%}" for k in RECALL_K)
        avg_t = sum(t["time_s"] for t in trials) / len(trials) if trials else 0
        print(f"    -> {recall_str}  avg_rank={avg_rank:.2f}  {avg_t:.2f}s/query")

        per_book_results[book] = {
            "metrics": {
                "total":    len(trials),
                "avg_rank": round(avg_rank, 3),
                "recall_at_k": {str(k): round(recall[k], 4) for k in RECALL_K},
                "hits_at_k":   {str(k): hits_at_k[k] for k in RECALL_K},
                "avg_time_per_query_s": round(avg_t, 3),
            },
            "trials": trials,
        }

    all_trials = [t for bm in per_book_results.values() for t in bm["trials"]]
    agg_recall, agg_avg_rank, agg_hits = compute_recall(all_trials)
    agg_total = len(all_trials)
    agg_avg_t = sum(t["time_s"] for t in all_trials) / agg_total if agg_total else 0

    return {
        "chunk_preset": chunk_preset,
        "chunk_name":   name,
        "chunk_config": cfg,
        "per_book":     per_book_results,
        "aggregate": {
            "total":    agg_total,
            "avg_rank": round(agg_avg_rank, 3),
            "recall_at_k": {str(k): round(agg_recall[k], 4) for k in RECALL_K},
            "hits_at_k":   {str(k): agg_hits[k] for k in RECALL_K},
            "avg_time_per_query_s": round(agg_avg_t, 3),
        },
    }


def main():
    ts      = datetime.now().isoformat()
    run_dir = make_run_dir(BASE_RESULTS_DIR, "chunk_ablation")

    print("=" * 80)
    print("CHUNK-SIZE-ABLATION")
    print("=" * 80)
    print(f"Pipeline:     {PIPELINE_PRESET}  [TF-IDF + Embeddings + Reranker + LLM]")
    print(f"Top-K:        {K_RERANKER}")
    print(f"Bücher:       {', '.join(EVAL_BOOKS)}")
    print(f"Chunk-Größen: {', '.join(cfg['name'] for cfg in CHUNK_PRESETS.values())}")
    print()

    # --- Eval-Pairs laden (einmalig, gilt für alle Chunk-Größen) ---
    eval_pairs_per_book = {}
    for book in EVAL_BOOKS:
        ep_path = os.path.join(DIR_PROCESSED, f"eval_pairs_{book}.json")
        if not os.path.exists(ep_path):
            print(f"  WARNUNG: {ep_path} nicht gefunden — {book} wird übersprungen")
            continue
        with open(ep_path, encoding="utf-8") as f:
            eval_pairs_per_book[book] = json.load(f)
        print(f"  {book}: {len(eval_pairs_per_book[book])} Pairs geladen")
    print()

    # --- Je Chunk-Größe auswerten ---
    chunk_results = [run_chunk_size(preset, eval_pairs_per_book) for preset in sorted(CHUNK_PRESETS)]

    # --- Vergleichstabelle (Konsole + TXT) ---
    summary_lines = [
        "=" * 80,
        "CHUNK-SIZE-ABLATION — ZUSAMMENFASSUNG",
        f"Datum:      {ts[:19]}",
        f"Pipeline:   {PIPELINE_PRESET}  [TF-IDF + Embeddings + Reranker + LLM]",
        f"Top-K:      {K_RERANKER}",
        f"Embedding:  {EMBEDDING_MODEL}",
        f"LLM:        {LLM_MODEL}",
        f"BM25:       {'ja' if USE_BM25 else 'nein'}",
        "=" * 80,
        f"{'Chunk-Größe':<22}  " + "  ".join(f"R@{k:>2}" for k in RECALL_K) + f"  {'AvgRank':>7}  {'s/query':>7}",
        "-" * 80,
    ]

    for r in chunk_results:
        m = r["aggregate"]
        k_vals = "  ".join(f"{m['recall_at_k'][str(k)]:>5.1%}" for k in RECALL_K)
        summary_lines.append(
            f"{r['chunk_name']:<22}  {k_vals}  {m['avg_rank']:>7.2f}  {m['avg_time_per_query_s']:>6.2f}s"
        )

    summary_lines.append("=" * 80)

    print("\n" + "\n".join(summary_lines))

    os.makedirs(run_dir, exist_ok=True)
    txt_path = os.path.join(run_dir, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nZusammenfassung: {txt_path}")

    # --- Detail-JSON ---
    details = {
        "timestamp": ts,
        "pipeline":  PIPELINE_PRESET,
        "pipeline_params": {
            "K_RETRIEVAL": K_RETRIEVAL,
            "K_RRF":       K_RRF,
            "K_RERANKER":  K_RERANKER,
            "RRF_K":       RRF_K,
        },
        "chunk_sizes": [
            {
                "chunk_preset": r["chunk_preset"],
                "chunk_name":   r["chunk_name"],
                "chunk_config": r["chunk_config"],
                "aggregate":    r["aggregate"],
                "books": [
                    {"book": book, "metrics": bm["metrics"], "trials": bm["trials"]}
                    for book, bm in r["per_book"].items()
                ],
            }
            for r in chunk_results
        ],
    }
    details_path = os.path.join(run_dir, "results.json")
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print(f"Ergebnisse:      {details_path}")

    # --- Misses je Chunk-Größe ---
    for r in chunk_results:
        misses = build_misses(r["per_book"], got_key="top1_text")
        chunk_dir = os.path.join(run_dir, r["chunk_name"])
        misses_path = save_misses(misses, chunk_dir)
        print(f"Misses ({r['chunk_name']}): {misses_path}  ({len(misses)} Einträge)")


if __name__ == "__main__":
    main()
