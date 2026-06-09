"""
LLM-Pick-Experiment
===================
Das LLM wählt aus den Reranker-Kandidaten genau EINE Textstelle.
Gemessen wird nur Accuracy (trifft die gewählte Stelle oder nicht).
"""
import json
import os
import sys
import time
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import (
    EVAL_BOOKS,
    DIR_PROCESSED, DIR_EXPERIMENTS,
    K_RERANKER,
    EXP_CHUNK          as CHUNK_PRESET,
    EXP_LLM_PIPELINE   as PIPELINE_PRESET,
)
from src.preprocessing.chunk_presets import CHUNK_PRESETS
from src.retrieval.pipeline import SearchPipeline
from src.retrieval.llm_reranker import PROMPT_PICK_ONE

CHUNK_CFG       = CHUNK_PRESETS[CHUNK_PRESET]
CHUNK_NAME      = CHUNK_CFG["name"]
CHUNK_MIN_WORDS = CHUNK_CFG["min_words"]
CHUNK_MAX_WORDS = CHUNK_CFG["max_words"]
CHUNK_OVERLAP   = CHUNK_CFG["overlap"]

RESULTS_DIR = os.path.join(project_root, "eval", "results")


def prepare_book(book: str, output_dir: str):
    chunks_path = os.path.join(output_dir, f"chunks_{book}_{CHUNK_NAME}.json")
    emb_path    = os.path.join(output_dir, f"embeddings_{book}_{CHUNK_NAME}.npy")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"Chunks nicht gefunden: {chunks_path}")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings nicht gefunden: {emb_path}")
    return chunks_path, emb_path


def run_trial(pipeline: SearchPipeline, pair: dict):
    """LLM-Pick-Trial: Pipeline gibt genau 1 Ergebnis zurück."""
    query          = pair["summary"]
    expected_spans = [tuple(s) for s in pair["spans"]]
    expected_text  = pair.get("text", "")

    t0 = time.perf_counter()
    results = pipeline.search(query, top_k=1, verbose=False)
    elapsed = time.perf_counter() - t0

    pick = results[0] if results else None
    hit = False
    if pick:
        for exp_start, exp_end in expected_spans:
            if pick.meta["start_index"] < exp_end and exp_start < pick.meta["end_index"]:
                hit = True
                break

    return {
        "query":          query,
        "expected_text":  expected_text,
        "expected_spans": [list(s) for s in expected_spans],
        "hit":            hit,
        "pick_text":      pick.text if pick else None,
        "pick_span":      [pick.meta["start_index"], pick.meta["end_index"]] if pick else None,
        "time_s":         round(elapsed, 3),
    }


def main():
    now = datetime.now()
    ts = now.isoformat()
    ts_file = now.strftime("%Y%m%d_%H%M%S")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DIR_EXPERIMENTS, exist_ok=True)

    print("=" * 80)
    print("LLM-PICK-EXPERIMENT")
    print("=" * 80)
    print(f"Pipeline:    {PIPELINE_PRESET}  (LLM wählt genau 1 Treffer)")
    print(f"Chunks:      {CHUNK_NAME}  (min={CHUNK_MIN_WORDS}, max={CHUNK_MAX_WORDS}, overlap={CHUNK_OVERLAP})")
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

        pipeline = SearchPipeline(chunks_path, emb_path, preset=PIPELINE_PRESET, llm_prompt=PROMPT_PICK_ONE)
        pairs  = eval_pairs_per_book[book]
        trials = []

        for i, pair in enumerate(pairs, 1):
            trial = run_trial(pipeline, pair)
            trials.append(trial)
            status = "✓" if trial["hit"] else "✗"
            print(f"  {i:>3}/{len(pairs)}  {status}  {trial['query'][:60]}")

        hits  = sum(1 for t in trials if t["hit"])
        total = len(trials)
        acc   = hits / total if total else 0.0
        avg_t = sum(t["time_s"] for t in trials) / total if trials else 0
        print(f"\n  Accuracy={acc:.1%}  ({hits}/{total})  {avg_t:.2f}s/query\n")

        per_book_results[book] = {
            "metrics": {
                "total":    total,
                "hits":     hits,
                "accuracy": round(acc, 4),
                "avg_time_per_query_s": round(avg_t, 3),
            },
            "trials": trials,
        }

    # --- Aggregation ---
    all_trials = [t for bm in per_book_results.values() for t in bm["trials"]]
    agg_total  = len(all_trials)
    agg_hits   = sum(1 for t in all_trials if t["hit"])
    agg_acc    = agg_hits / agg_total if agg_total else 0.0
    agg_avg_t  = sum(t["time_s"] for t in all_trials) / agg_total if agg_total else 0

    # --- Summary ---
    summary_lines = [
        "=" * 80,
        "LLM-PICK-EXPERIMENT — ZUSAMMENFASSUNG",
        f"Datum:     {ts[:19]}",
        f"Pipeline:  {PIPELINE_PRESET}  (LLM wählt genau 1 Treffer)",
        f"Chunks:    {CHUNK_NAME}  (min={CHUNK_MIN_WORDS}, max={CHUNK_MAX_WORDS}, overlap={CHUNK_OVERLAP})",
        "=" * 80,
        f"{'Buch':<20}  {'Accuracy':>8}  {'Hits':>6}  {'Total':>6}  {'s/query':>7}",
        "-" * 80,
    ]

    for book, bm in per_book_results.items():
        m = bm["metrics"]
        summary_lines.append(
            f"{book:<20}  {m['accuracy']:>8.1%}  {m['hits']:>6}  {m['total']:>6}  {m['avg_time_per_query_s']:>6.2f}s"
        )

    summary_lines.append("-" * 80)
    summary_lines.append(
        f"{'GESAMT':<20}  {agg_acc:>8.1%}  {agg_hits:>6}  {agg_total:>6}  {agg_avg_t:>6.2f}s"
    )
    summary_lines.append("=" * 80)

    print("\n" + "\n".join(summary_lines))

    txt_path = os.path.join(RESULTS_DIR, f"llm_pick_summary_{ts_file}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nZusammenfassung: {txt_path}")

    details = {
        "timestamp":    ts,
        "experiment":   "llm_pick",
        "pipeline":     PIPELINE_PRESET,
        "chunk_config": {
            "name":      CHUNK_NAME,
            "min_words": CHUNK_MIN_WORDS,
            "max_words": CHUNK_MAX_WORDS,
            "overlap":   CHUNK_OVERLAP,
        },
        "aggregate": {
            "total":    agg_total,
            "hits":     agg_hits,
            "accuracy": round(agg_acc, 4),
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
    details_path = os.path.join(RESULTS_DIR, f"llm_pick_details_{ts_file}.json")
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print(f"Details:         {details_path}")


if __name__ == "__main__":
    main()
