"""
Chunk size experiment: Test different chunking strategies across multiple books.
Evaluates all combinations of chunk sizes × pipeline configs × books and reports
aggregated metrics.
"""
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from script.build_book_chunks import build_chunks
from script.build_embeddings import build_embeddings

# Books to evaluate (excluding harrypotter)
BOOKS = [
    "verwandlung",
    "erdbeben",
    "judenbuche",
    "krambambuli",
]

CHUNK_CONFIGS = [
    {"name": "tiny",               "min_words": 10,  "max_words": 30,  "overlap": 1},
    {"name": "small",              "min_words": 10,  "max_words": 50,  "overlap": 1},
    {"name": "medium",             "min_words": 30,  "max_words": 100, "overlap": 2},
    {"name": "large",              "min_words": 50,  "max_words": 150, "overlap": 2},
    {"name": "xlarge",             "min_words": 80,  "max_words": 200, "overlap": 3},
    {"name": "medium_high_overlap","min_words": 30,  "max_words": 100, "overlap": 3},
    {"name": "large_high_overlap", "min_words": 50,  "max_words": 150, "overlap": 4},
]

PIPELINE_CONFIGS = [
    {"name": "TF-IDF+Emb",       "tfidf": True,  "emb": True,  "rerank": False, "nli": False},
    {"name": "Emb Only",          "tfidf": False, "emb": True,  "rerank": False, "nli": False},
    {"name": "Full Pipeline",     "tfidf": True,  "emb": True,  "rerank": True,  "nli": True},
    {"name": "TF-IDF+Emb+Rerank", "tfidf": True,  "emb": True,  "rerank": True,  "nli": False},
]


def evaluate_pipeline(pipeline, eval_pairs):
    """
    Run a pipeline over eval_pairs, return (hits, total, avg_rank).
    """
    hits = 0
    ranks = []

    for pair in eval_pairs:
        query = pair["summary"]
        expected_spans = [tuple(s) for s in pair["spans"]]

        try:
            results = pipeline.search(query, top_k=10, verbose=False)
            hit = False
            for rank, r in enumerate(results, start=1):
                result_start = r.meta["start_index"]
                result_end = r.meta["end_index"]
                for expected_start, expected_end in expected_spans:
                    if result_start < expected_end and expected_start < result_end:
                        hit = True
                        ranks.append(rank)
                        break
                if hit:
                    break
            if hit:
                hits += 1
        except Exception:
            pass  # counts as miss

    total = len(eval_pairs)
    avg_rank = sum(ranks) / len(ranks) if ranks else 0.0
    return hits, total, avg_rank


def run_chunk_experiment(output_dir: str = None):
    eval_dir = os.path.join(project_root, "eval")

    if output_dir is None:
        output_dir = os.path.join(project_root, "data/experiments/chunk_sizes")

    os.makedirs(output_dir, exist_ok=True)

    n_combos = len(CHUNK_CONFIGS) * len(PIPELINE_CONFIGS) * len(BOOKS)
    print("=" * 80)
    print("CHUNK SIZE EXPERIMENT (multi-book)")
    print("=" * 80)
    print(f"Books:      {', '.join(BOOKS)}")
    print(f"Output dir: {output_dir}")
    print(
        f"Testing {len(CHUNK_CONFIGS)} chunk sizes × {len(PIPELINE_CONFIGS)} pipelines"
        f" × {len(BOOKS)} books = {n_combos} combinations\n"
    )

    # Load eval pairs per book
    eval_pairs_per_book = {}
    for book in BOOKS:
        ep_path = os.path.join(eval_dir, f"eval_pairs_{book}.json")
        if not os.path.exists(ep_path):
            print(f"  WARNING: eval_pairs file not found for {book}: {ep_path}")
            continue
        with open(ep_path, "r", encoding="utf-8") as f:
            eval_pairs_per_book[book] = json.load(f)
        print(f"  Loaded {len(eval_pairs_per_book[book])} eval pairs for {book}")

    available_books = list(eval_pairs_per_book.keys())
    print()

    # results_all: list of dicts with keys chunk_config, pipeline_config, book, metrics
    results_all = []

    from eval.ablation_search import AblationSearchPipeline

    for i, chunk_config in enumerate(CHUNK_CONFIGS, 1):
        chunk_name = chunk_config["name"]
        print(f"\n{'='*80}")
        print(
            f"[{i}/{len(CHUNK_CONFIGS)}] CHUNK SIZE: {chunk_name}"
            f"  (min={chunk_config['min_words']}, max={chunk_config['max_words']},"
            f" overlap={chunk_config['overlap']})"
        )
        print("=" * 80)

        # --- Prepare chunks + embeddings for every book ---
        book_data = {}  # book -> {"chunks_path", "emb_path", "total_chunks"}
        all_ready = True

        for book in available_books:
            book_path = os.path.join(project_root, "data/raw", f"{book}.txt")
            chunks_path = os.path.join(output_dir, f"chunks_{book}_{chunk_name}.json")
            emb_path    = os.path.join(output_dir, f"embeddings_{book}_{chunk_name}.npy")

            try:
                if not os.path.exists(chunks_path):
                    print(f"  [{book}] Building chunks → {os.path.basename(chunks_path)}")
                    build_chunks(
                        book_path=book_path,
                        output_path=chunks_path,
                        min_words=chunk_config["min_words"],
                        max_words=chunk_config["max_words"],
                        sentence_overlap=chunk_config["overlap"],
                    )
                if not os.path.exists(emb_path):
                    print(f"  [{book}] Building embeddings → {os.path.basename(emb_path)}")
                    build_embeddings(chunks_path=chunks_path, output_path=emb_path)

                with open(chunks_path, "r", encoding="utf-8") as f:
                    cd = json.load(f)

                book_data[book] = {
                    "chunks_path": chunks_path,
                    "emb_path": emb_path,
                    "total_chunks": cd["total_chunks"],
                }
                print(f"  [{book}] {cd['total_chunks']} chunks  ✓")

            except Exception as e:
                print(f"  [{book}] FAILED to prepare: {e}")
                import traceback; traceback.print_exc()
                all_ready = False

        if not all_ready:
            print("  Skipping pipeline tests for this chunk size due to preparation errors.")
            continue

        print()

        # --- Test every pipeline config across all books ---
        for j, pipe_config in enumerate(PIPELINE_CONFIGS, 1):
            pipe_name = pipe_config["name"]
            print(f"  [{j}/{len(PIPELINE_CONFIGS)}] {pipe_name}")

            pipe_hits  = 0
            pipe_total = 0
            pipe_ranks = []
            per_book_metrics = {}

            for book in available_books:
                bd = book_data[book]
                ep = eval_pairs_per_book[book]

                try:
                    pipeline = AblationSearchPipeline(
                        bd["chunks_path"], bd["emb_path"],
                        use_tfidf=pipe_config["tfidf"],
                        use_embeddings=pipe_config["emb"],
                        use_reranker=pipe_config["rerank"],
                        use_nli=pipe_config["nli"],
                    )

                    hits, total, avg_rank = evaluate_pipeline(pipeline, ep)
                    hit_rate = hits / total if total > 0 else 0.0

                    per_book_metrics[book] = {
                        "hits": hits, "total": total,
                        "hit_rate": hit_rate, "avg_rank": avg_rank,
                    }

                    # collect for aggregation
                    pipe_hits  += hits
                    pipe_total += total
                    # extend ranks by reconstructing (we track aggregate rank separately)
                    if avg_rank > 0:
                        pipe_ranks.extend([avg_rank] * hits)

                    print(
                        f"    {book:<15} {hits}/{total} ({hit_rate:.1%})"
                        f"  avg_rank={avg_rank:.2f}"
                    )

                except Exception as e:
                    print(f"    {book:<15} FAILED: {e}")

            # Aggregate across books
            agg_hit_rate = pipe_hits / pipe_total if pipe_total > 0 else 0.0
            agg_avg_rank = sum(pipe_ranks) / len(pipe_ranks) if pipe_ranks else 0.0

            # Average chunks across books
            avg_chunks = int(
                sum(bd["total_chunks"] for bd in book_data.values()) / len(book_data)
            )

            print(
                f"    {'TOTAL':<15} {pipe_hits}/{pipe_total} ({agg_hit_rate:.1%})"
                f"  avg_rank={agg_avg_rank:.2f}\n"
            )

            results_all.append({
                "chunk_config": chunk_config,
                "pipeline_config": pipe_config,
                "aggregate": {
                    "hits": pipe_hits,
                    "total": pipe_total,
                    "hit_rate": agg_hit_rate,
                    "avg_rank": agg_avg_rank,
                    "avg_chunks_per_book": avg_chunks,
                },
                "per_book": per_book_metrics,
            })

    # -------------------------------------------------------------------------
    # Print aggregated summary table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("AGGREGATED SUMMARY (all books combined)")
    print("=" * 80)
    header = (
        f"{'Chunk Size':<25} {'Pipeline':<22} {'Avg Chunks':>10}"
        f" {'Hit Rate':>10} {'Avg Rank':>9} {'Hits':>10}"
    )
    print(header)
    print("-" * 80)

    valid = [r for r in results_all if "aggregate" in r]
    results_sorted = sorted(
        valid,
        key=lambda x: (x["aggregate"]["hit_rate"], -x["aggregate"]["avg_rank"]),
        reverse=True,
    )

    for r in results_sorted:
        cname = r["chunk_config"]["name"]
        pname = r["pipeline_config"]["name"]
        agg   = r["aggregate"]
        print(
            f"{cname:<25} {pname:<22} {agg['avg_chunks_per_book']:>10}"
            f" {agg['hit_rate']:>9.1%} {agg['avg_rank']:>9.2f}"
            f" {agg['hits']:>5}/{agg['total']:<4}"
        )

    print("=" * 80)

    # Best pipeline per chunk size
    print("\n" + "=" * 80)
    print("BEST PIPELINE PER CHUNK SIZE (aggregated)")
    print("=" * 80)
    best_per_chunk = {}
    for r in valid:
        cname = r["chunk_config"]["name"]
        if cname not in best_per_chunk or r["aggregate"]["hit_rate"] > best_per_chunk[cname]["aggregate"]["hit_rate"]:
            best_per_chunk[cname] = r

    print(f"{'Chunk Size':<25} {'Best Pipeline':<22} {'Hit Rate':>10} {'Avg Rank':>9}")
    print("-" * 80)
    for cc in CHUNK_CONFIGS:
        cname = cc["name"]
        if cname in best_per_chunk:
            r = best_per_chunk[cname]
            agg = r["aggregate"]
            print(
                f"{cname:<25} {r['pipeline_config']['name']:<22}"
                f" {agg['hit_rate']:>9.1%} {agg['avg_rank']:>9.2f}"
            )

    print("=" * 80)

    # Per-book breakdown for top-5 configurations
    print("\n" + "=" * 80)
    print("PER-BOOK BREAKDOWN (top 5 configurations by aggregated hit rate)")
    print("=" * 80)
    books_col = available_books
    header2 = f"{'Config':<50}" + "".join(f" {b.capitalize():>14}" for b in books_col)
    print(header2)
    print("-" * 80)
    for r in results_sorted[:5]:
        label = f"{r['chunk_config']['name']} / {r['pipeline_config']['name']}"
        row = f"{label:<50}"
        for book in books_col:
            bm = r["per_book"].get(book)
            if bm:
                row += f" {bm['hits']:>5}/{bm['total']:<3} ({bm['hit_rate']:.0%})"
            else:
                row += f" {'N/A':>14}"
        print(row)
    print("=" * 80)

    # Save full results
    output_path = os.path.join(output_dir, "chunk_experiment_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": datetime.now().isoformat(), "books": BOOKS, "results": results_all},
            f, ensure_ascii=False, indent=2,
        )
    print(f"\nDetailed results saved to: {output_path}")


def main():
    output_dir = None
    if len(sys.argv) >= 2:
        output_dir = sys.argv[1]
    run_chunk_experiment(output_dir)


if __name__ == "__main__":
    main()
