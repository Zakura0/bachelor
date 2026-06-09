"""
Evaluation-Experiment
=====================
Konfigurierbares Experiment zur Auswertung der Retrieval-Pipeline.
Misst Recall@k über alle Eval-Bücher.

Konfiguration: config.py, Abschnitt "Experiment".
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
    DIR_PROCESSED, DIR_EXPERIMENTS,
    EXP_CHUNK        as CHUNK_PRESET,
    EXP_PIPELINE     as PIPELINE_PRESET,
    EXP_LLM_TOP_K    as LLM_TOP_K,
)
from src.preprocessing.chunk_presets import CHUNK_PRESETS
from src.retrieval.pipeline import SearchPipeline

CHUNK_CFG  = CHUNK_PRESETS[CHUNK_PRESET]
CHUNK_NAME = CHUNK_CFG["name"]


RESULTS_DIR = os.path.join(project_root, "eval", "results")


def prepare_book(book: str, output_dir: str):
    """Pfade zu Chunks und Embeddings zurückgeben. Bricht ab wenn Dateien fehlen."""
    chunks_path = os.path.join(output_dir, f"chunks_{book}_{CHUNK_NAME}.json")
    emb_path    = os.path.join(output_dir, f"embeddings_{book}_{CHUNK_NAME}.npy")

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
    results = pipeline.search(query, top_k=LLM_TOP_K, verbose=False)
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


def main():
    ts = datetime.now().isoformat()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DIR_EXPERIMENTS, exist_ok=True)

    print("=" * 80)
    print("TOP-1 LLM EXPERIMENT")
    print("=" * 80)
    print(f"Pipeline:    {PIPELINE_PRESET}")
    print(f"Chunks:      {CHUNK_NAME}  (min={CHUNK_MIN_WORDS}, max={CHUNK_MAX_WORDS}, overlap={CHUNK_OVERLAP})")
    print(f"LLM Top-K:   {LLM_TOP_K}")
    print(f"Bücher:      {', '.join(EVAL_BOOKS)}")
    print()

    # --- Eval-Pairs laden ---
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

    # --- Je Buch auswerten ---
    per_book_results = {}

    for book in eval_pairs_per_book:
        print(f"{'='*80}")
        print(f"Buch: {book}")
        chunks_path, emb_path = prepare_book(book, DIR_EXPERIMENTS)

        pipeline = SearchPipeline(
            chunks_path, emb_path,
            preset=PIPELINE_PRESET,
        )

        pairs  = eval_pairs_per_book[book]
        trials = []

        for i, pair in enumerate(pairs, 1):
            trial = run_trial(pipeline, pair)
            trials.append(trial)
            status = "✓" if trial["hit"] else "✗"
            rank   = trial["first_hit_rank"] or "-"
            print(f"  {i:>3}/{len(pairs)}  {status}  Rank={rank:<4}  {trial['query'][:60]}")

        recall, avg_rank, hits_at_k = compute_recall(trials)
        recall_str = "  ".join(f"R@{k}={recall[k]:.1%}" for k in RECALL_K)
        avg_t = sum(t["time_s"] for t in trials) / len(trials) if trials else 0
        print(f"\n  {recall_str}  avg_rank={avg_rank:.2f}  {avg_t:.2f}s/query\n")

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

    # --- Aggregation ---
    all_trials = [t for bm in per_book_results.values() for t in bm["trials"]]
    agg_recall, agg_avg_rank, agg_hits = compute_recall(all_trials)
    agg_total = len(all_trials)
    agg_avg_t = sum(t["time_s"] for t in all_trials) / agg_total if agg_total else 0

    # --- Summary-Tabelle (Konsole + TXT) ---
    summary_lines = [
        "=" * 80,
        "TOP-1 LLM EXPERIMENT — ZUSAMMENFASSUNG",
        f"Datum:     {ts[:19]}",
        f"Pipeline:  {PIPELINE_PRESET}",
        f"Chunks:    {CHUNK_NAME}  (min={CHUNK_MIN_WORDS}, max={CHUNK_MAX_WORDS}, overlap={CHUNK_OVERLAP})",
        f"LLM Top-K: {LLM_TOP_K}",
        "=" * 80,
        f"{'Buch':<20}  " + "  ".join(f"R@{k:>2}" for k in RECALL_K) + f"  {'AvgRank':>7}  {'s/query':>7}",
        "-" * 80,
    ]

    for book, bm in per_book_results.items():
        m = bm["metrics"]
        k_vals = "  ".join(f"{m['recall_at_k'][str(k)]:>5.1%}" for k in RECALL_K)
        summary_lines.append(
            f"{book:<20}  {k_vals}  {m['avg_rank']:>7.2f}  {m['avg_time_per_query_s']:>6.2f}s"
        )

    summary_lines.append("-" * 80)
    k_vals = "  ".join(f"{agg_recall[k]:>5.1%}" for k in RECALL_K)
    summary_lines.append(
        f"{'GESAMT':<20}  {k_vals}  {agg_avg_rank:>7.2f}  {agg_avg_t:>6.2f}s"
        f"  ({agg_hits[1]}/{agg_total} @1)"
    )
    summary_lines.append("=" * 80)

    print("\n" + "\n".join(summary_lines))

    txt_path = os.path.join(RESULTS_DIR, "experiment_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nZusammenfassung: {txt_path}")

    # --- Detail-JSON ---
    details = {
        "timestamp":    ts,
        "pipeline":     PIPELINE_PRESET,
        "chunk_config": {
            "name": CHUNK_NAME,
            "min_words": CHUNK_MIN_WORDS,
            "max_words": CHUNK_MAX_WORDS,
            "overlap":   CHUNK_OVERLAP,
        },
        "llm_top_k": LLM_TOP_K,
        "aggregate": {
            "total":    agg_total,
            "avg_rank": round(agg_avg_rank, 3),
            "recall_at_k": {str(k): round(agg_recall[k], 4) for k in RECALL_K},
            "hits_at_k":   {str(k): agg_hits[k] for k in RECALL_K},
            "avg_time_per_query_s": round(agg_avg_t, 3),
        },
        "books": [
            {
                "book":    book,
                "metrics": bm["metrics"],
                "trials":  bm["trials"],
            }
            for book, bm in per_book_results.items()
        ],
    }
    details_path = os.path.join(RESULTS_DIR, "experiment_details.json")
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print(f"Details:         {details_path}")


if __name__ == "__main__":
    main()
