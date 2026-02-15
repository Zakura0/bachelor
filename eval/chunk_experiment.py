"""
Chunk size experiment: Test different chunking strategies and evaluate their impact.
"""
import json
import os
import sys
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from script.build_book_chunks import build_chunks
from script.build_embeddings import build_embeddings
from data.eval.evaluation import evaluate


def run_chunk_experiment(book_path: str, eval_pairs_path: str, output_dir: str = None):
    """
    Test different chunk configurations and evaluate their performance.
    """
    if output_dir is None:
        output_dir = os.path.join(project_root, "data/experiments/chunk_sizes")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define chunk configurations to test
    chunk_configs = [
        {"name": "tiny", "min_words": 10, "max_words": 30, "overlap": 1},
        {"name": "small", "min_words": 10, "max_words": 50, "overlap": 1},  # Current
        {"name": "medium", "min_words": 30, "max_words": 100, "overlap": 2},
        {"name": "large", "min_words": 50, "max_words": 150, "overlap": 2},
        {"name": "xlarge", "min_words": 80, "max_words": 200, "overlap": 3},
        {"name": "medium_high_overlap", "min_words": 30, "max_words": 100, "overlap": 3},
        {"name": "large_high_overlap", "min_words": 50, "max_words": 150, "overlap": 4},
    ]
    
    # Define pipeline configurations to test for each chunk size
    pipeline_configs = [
        {"name": "TF-IDF+Emb", "tfidf": True, "emb": True, "rerank": False, "nli": False},
        {"name": "Emb Only", "tfidf": False, "emb": True, "rerank": False, "nli": False},
        {"name": "Full Pipeline", "tfidf": True, "emb": True, "rerank": True, "nli": True},
        {"name": "TF-IDF+Emb+Rerank", "tfidf": True, "emb": True, "rerank": True, "nli": False},
    ]
    
    results_all = []
    
    print("="*80)
    print("CHUNK SIZE EXPERIMENT")
    print("="*80)
    print(f"Book: {book_path}")
    print(f"Output directory: {output_dir}")
    print(f"Testing {len(chunk_configs)} chunk sizes × {len(pipeline_configs)} pipeline configs = {len(chunk_configs) * len(pipeline_configs)} combinations\n")
    
    # Load eval pairs once
    with open(eval_pairs_path, "r", encoding="utf-8") as f:
        eval_pairs = json.load(f)
    
    for i, chunk_config in enumerate(chunk_configs, 1):
        chunk_name = chunk_config["name"]
        print(f"\n{'='*80}")
        print(f"[{i}/{len(chunk_configs)}] CHUNK SIZE: {chunk_name}")
        print(f"  min_words={chunk_config['min_words']}, max_words={chunk_config['max_words']}, overlap={chunk_config['overlap']}")
        print('='*80)
        
        # Define output paths for this configuration
        chunks_path = os.path.join(output_dir, f"chunks_{chunk_name}.json")
        emb_path = os.path.join(output_dir, f"embeddings_{chunk_name}.npy")
        
        try:
            # Step 1: Build chunks (if not exists)
            if not os.path.exists(chunks_path):
                print(f"  [1/2] Building chunks...")
                build_chunks(
                    book_path=book_path,
                    output_path=chunks_path,
                    min_words=chunk_config["min_words"],
                    max_words=chunk_config["max_words"],
                    sentence_overlap=chunk_config["overlap"]
                )
            else:
                print(f"  [1/2] Chunks already exist: {chunks_path}")
            
            # Step 2: Build embeddings (if not exists)
            if not os.path.exists(emb_path):
                print(f"  [2/2] Building embeddings...")
                build_embeddings(
                    chunks_path=chunks_path,
                    output_path=emb_path
                )
            else:
                print(f"  [2/2] Embeddings already exist: {emb_path}")
            
            # Load chunk stats
            with open(chunks_path, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)
            
            print(f"\n  Total chunks: {chunk_data['total_chunks']}")
            print(f"  Now testing {len(pipeline_configs)} pipeline configurations...\n")
            
            # Test all pipeline configurations with this chunk size
            for j, pipe_config in enumerate(pipeline_configs, 1):
                pipe_name = pipe_config["name"]
                print(f"    [{j}/{len(pipeline_configs)}] Testing: {pipe_name}... ", end="", flush=True)
                
                try:
                    # Import here to avoid loading models too early
                    from data.eval.ablation_search import AblationSearchPipeline
                    
                    pipeline = AblationSearchPipeline(
                        chunks_path, emb_path,
                        use_tfidf=pipe_config["tfidf"],
                        use_embeddings=pipe_config["emb"],
                        use_reranker=pipe_config["rerank"],
                        use_nli=pipe_config["nli"]
                    )
                    
                    # Run evaluation
                    hits = 0
                    misses = 0
                    ranks = []
                    
                    for pair in eval_pairs:
                        query = pair["summary"]
                        expected_spans = [tuple(s) for s in pair["spans"]]
                        
                        try:
                            results = pipeline.search(query, top_k=10, verbose=False)
                            
                            # Check for hit
                            hit = False
                            for rank, r in enumerate(results, start=1):
                                result_start = r.meta["start_index"]
                                result_end = r.meta["end_index"]
                                
                                for expected_start, expected_end in expected_spans:
                                    # Check overlap
                                    if result_start < expected_end and expected_start < result_end:
                                        hit = True
                                        ranks.append(rank)
                                        break
                                if hit:
                                    break
                            
                            if hit:
                                hits += 1
                            else:
                                misses += 1
                                
                        except Exception as e:
                            misses += 1
                    
                    # Calculate metrics
                    total = hits + misses
                    hit_rate = hits / total if total > 0 else 0
                    avg_rank = sum(ranks) / len(ranks) if ranks else 0
                    
                    result = {
                        "chunk_config": chunk_config,
                        "pipeline_config": pipe_config,
                        "metrics": {
                            "total": total,
                            "hits": hits,
                            "misses": misses,
                            "hit_rate": hit_rate,
                            "avg_rank": avg_rank
                        },
                        "chunk_stats": {
                            "total_chunks": chunk_data["total_chunks"],
                            "text_length": chunk_data["original_text_length"]
                        }
                    }
                    
                    results_all.append(result)
                    
                    print(f"✓ {hits}/{total} ({hit_rate:.1%}), Rank {avg_rank:.2f}")
                    
                except Exception as e:
                    print(f"✗ FAILED: {e}")
                    results_all.append({
                        "chunk_config": chunk_config,
                        "pipeline_config": pipe_config,
                        "error": str(e)
                    })
            
        except Exception as e:
            print(f"  ✗ FAILED to prepare chunks/embeddings: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary comparison
    print("\n" + "="*80)
    print("CHUNK SIZE EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Chunk Size':<25} {'Pipeline':<22} {'Chunks':>7} {'Hit Rate':>10} {'Rank':>6} {'Hits':>8}")
    print("-"*80)
    
    # Sort by hit rate descending
    results_sorted = sorted(
        [r for r in results_all if "error" not in r],
        key=lambda x: (x["metrics"]["hit_rate"], -x["metrics"]["avg_rank"]),
        reverse=True
    )
    
    for result in results_sorted:
        chunk_name = result["chunk_config"]["name"]
        pipe_name = result["pipeline_config"]["name"]
        metrics = result["metrics"]
        chunks = result["chunk_stats"]["total_chunks"]
        print(f"{chunk_name:<25} {pipe_name:<22} {chunks:>7} {metrics['hit_rate']:>9.1%} {metrics['avg_rank']:>6.2f} {metrics['hits']:>4}/{metrics['total']:<3}")
    
    print("="*80)
    
    # Group by chunk size for comparison
    print("\n" + "="*80)
    print("BEST PIPELINE PER CHUNK SIZE")
    print("="*80)
    
    chunk_best = {}
    for result in results_all:
        if "error" in result:
            continue
        chunk_name = result["chunk_config"]["name"]
        if chunk_name not in chunk_best or result["metrics"]["hit_rate"] > chunk_best[chunk_name]["metrics"]["hit_rate"]:
            chunk_best[chunk_name] = result
    
    print(f"{'Chunk Size':<25} {'Best Pipeline':<22} {'Hit Rate':>10} {'Rank':>6}")
    print("-"*80)
    for chunk_name in [c["name"] for c in chunk_configs]:
        if chunk_name in chunk_best:
            result = chunk_best[chunk_name]
            pipe_name = result["pipeline_config"]["name"]
            metrics = result["metrics"]
            print(f"{chunk_name:<25} {pipe_name:<22} {metrics['hit_rate']:>9.1%} {metrics['avg_rank']:>6.2f}")
    
    print("="*80)
    
    # Save detailed results
    output_path = os.path.join(output_dir, "chunk_experiment_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "book_path": book_path,
            "results": results_all
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")


def main():
    book_path = os.path.join(project_root, "data/raw/verwandlung.txt")
    eval_pairs_path = os.path.join(project_root, "data/eval/eval_pairs.json")
    
    if len(sys.argv) >= 2:
        book_path = sys.argv[1]
    if len(sys.argv) >= 3:
        eval_pairs_path = sys.argv[2]
    
    run_chunk_experiment(book_path, eval_pairs_path)


if __name__ == "__main__":
    main()
