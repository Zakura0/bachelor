"""
LLM Fulltext Experiment
=======================
Das LLM erhält den vollständigen Buchtext und eine Zusammenfassung und soll
die passende Textstelle wortwörtlich zurückgeben.

Konfiguration: config.py, Abschnitt "Experimente".
"""
import json
import os
import sys
import time
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from openai import OpenAI
from eval.eval_utils import make_run_dir, build_misses, save_misses

from config import (
    EVAL_BOOKS, RECALL_K,
    DIR_PROCESSED, LLM_MODEL,
)

BASE_RESULTS_DIR = os.path.join(project_root, "eval", "results")

PROMPT_TEMPLATE = """\
Hier ist ein literarischer Text:

{book_text}

Finde die Textstelle, die inhaltlich am besten zu dieser Zusammenfassung passt:
\"{query}\"

Gib nur die Textstelle wortwörtlich zurück, keine Erklärungen."""


def run_trial(client: OpenAI, book_text: str, pair: dict) -> dict:
    query          = pair["summary"]
    expected_spans = [tuple(s) for s in pair["spans"]]
    expected_text  = pair.get("text", "")

    prompt = PROMPT_TEMPLATE.format(book_text=book_text, query=query)

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.perf_counter() - t0

    answer = response.choices[0].message.content or ""
    answer_clean = answer.strip().strip("\"'„\u201c\u201d\u201f\u00bb\u00ab")

    # Textstelle im Buch suchen (erste 50 Zeichen als Anker)
    found_at = book_text.find(answer_clean[:50]) if answer_clean else -1
    answer_end = found_at + len(answer_clean) if found_at >= 0 else -1

    hit = False
    if found_at >= 0:
        for exp_start, exp_end in expected_spans:
            if found_at < exp_end and exp_start < answer_end:
                hit = True
                break

    return {
        "query":          query,
        "expected_text":  expected_text,
        "expected_spans": [list(s) for s in expected_spans],
        "hit":            hit,
        "answer":         answer_clean,
        "found_at":       found_at,
        "answer_end":     answer_end,
        "time_s":         round(elapsed, 3),
    }


def compute_metrics(trials: list) -> dict:
    total  = len(trials)
    hits   = sum(1 for t in trials if t["hit"])
    avg_t  = sum(t["time_s"] for t in trials) / total if total else 0.0
    # Recall@k: da es kein Ranking gibt ist Recall@k = Hit-Rate für alle k
    recall = {k: hits / total if total else 0.0 for k in RECALL_K}
    return {
        "total":       total,
        "hits":        hits,
        "hit_rate":    round(hits / total, 4) if total else 0.0,
        "recall_at_k": {str(k): round(recall[k], 4) for k in RECALL_K},
        "avg_time_per_query_s": round(avg_t, 3),
    }


def main():
    ts      = datetime.now().isoformat()
    run_dir = make_run_dir(BASE_RESULTS_DIR, "llm_fulltext")
    os.makedirs(run_dir, exist_ok=True)

    client = OpenAI()

    print("=" * 80)
    print("LLM FULLTEXT EXPERIMENT")
    print("=" * 80)
    print(f"Modell:  {LLM_MODEL}")
    print(f"Bücher:  {', '.join(EVAL_BOOKS)}")
    print()

    per_book_results = {}

    for book in EVAL_BOOKS:
        print(f"{'='*80}")
        print(f"Buch: {book}")

        ep_path   = os.path.join(DIR_PROCESSED, f"eval_pairs_{book}.json")
        text_path = os.path.join(project_root, "data/raw", f"{book}.txt")

        if not os.path.exists(ep_path):
            print(f"  WARNUNG: {ep_path} nicht gefunden — übersprungen")
            continue
        if not os.path.exists(text_path):
            print(f"  WARNUNG: {text_path} nicht gefunden — übersprungen")
            continue

        pairs     = json.load(open(ep_path, encoding="utf-8"))
        book_text = open(text_path, encoding="utf-8").read()

        trials = []
        for i, pair in enumerate(pairs, 1):
            trial = run_trial(client, book_text, pair)
            trials.append(trial)
            status = "✓" if trial["hit"] else "✗"
            loc    = f"@{trial['found_at']}" if trial["found_at"] >= 0 else "nicht gefunden"
            print(f"  {i:>3}/{len(pairs)}  {status}  {loc:<14}  {trial['query'][:60]}")

        m = compute_metrics(trials)
        recall_str = "  ".join(f"R@{k}={m['recall_at_k'][str(k)]:.1%}" for k in RECALL_K)
        print(f"\n  {recall_str}  hit_rate={m['hit_rate']:.1%}  {m['avg_time_per_query_s']:.2f}s/query\n")

        per_book_results[book] = {"metrics": m, "trials": trials}

    # --- Aggregation ---
    all_trials = [t for bm in per_book_results.values() for t in bm["trials"]]
    agg = compute_metrics(all_trials)

    # --- Summary ---
    summary_lines = [
        "=" * 80,
        "LLM FULLTEXT EXPERIMENT — ZUSAMMENFASSUNG",
        f"Datum:   {ts[:19]}",
        f"Modell:  {LLM_MODEL}",
        "=" * 80,
        f"{'Buch':<20}  {'Hit-Rate':>10}  {'Hits':>8}  {'s/query':>8}",
        "-" * 80,
    ]
    for book, bm in per_book_results.items():
        m = bm["metrics"]
        summary_lines.append(
            f"{book:<20}  {m['hit_rate']:>9.1%}  {m['hits']:>4}/{m['total']:<3}  {m['avg_time_per_query_s']:>7.2f}s"
        )
    summary_lines.append("-" * 80)
    summary_lines.append(
        f"{'GESAMT':<20}  {agg['hit_rate']:>9.1%}  {agg['hits']:>4}/{agg['total']:<3}  {agg['avg_time_per_query_s']:>7.2f}s"
    )
    summary_lines.append("=" * 80)

    print("\n" + "\n".join(summary_lines))

    txt_path = os.path.join(run_dir, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nZusammenfassung: {txt_path}")

    details = {
        "timestamp": ts,
        "model":     LLM_MODEL,
        "aggregate": agg,
        "books": [
            {"book": book, "metrics": bm["metrics"], "trials": bm["trials"]}
            for book, bm in per_book_results.items()
        ],
    }
    details_path = os.path.join(run_dir, "results.json")
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    print(f"Ergebnisse:      {details_path}")

    misses = build_misses(per_book_results, got_key="answer")
    misses_path = save_misses(misses, run_dir)
    print(f"Misses:          {misses_path}  ({len(misses)} Einträge)")


if __name__ == "__main__":
    main()
