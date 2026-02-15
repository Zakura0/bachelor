"""
Evaluation script for the search pipeline.
Uses summaries from eval_pairs.json as queries and checks if the corresponding 
text chunks are found in the top results.
"""

import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from script.run_search import SearchPipeline


def spans_overlap(span1: tuple, span2: tuple) -> bool:
    """Check if two (start, end) spans overlap."""
    start1, end1 = span1
    start2, end2 = span2
    return start1 < end2 and start2 < end1


def is_hit(expected_spans: list, results: list) -> tuple:
    """
    Check if any expected span overlaps with any result chunk.
    expected_spans: list of (start, end) tuples from eval_pairs
    results: list of search results with meta containing start_index/end_index
    Returns (is_hit: bool, rank: int or None)
    """
    for rank, r in enumerate(results, start=1):
        result_start = r.meta["start_index"]
        result_end = r.meta["end_index"]
        
        for expected_start, expected_end in expected_spans:
            if spans_overlap((expected_start, expected_end), (result_start, result_end)):
                return True, rank
    
    return False, None


def evaluate(chunks_path: str, emb_path: str, eval_pairs_path: str, top_k: int = 10):
    """Run evaluation on all eval pairs."""
    
    print("Loading eval pairs...")
    with open(eval_pairs_path, "r", encoding="utf-8") as f:
        eval_pairs = json.load(f)
    
    print("Loading search pipeline...")
    pipeline = SearchPipeline(chunks_path, emb_path)
    
    print(f"\nStarting evaluation with {len(eval_pairs)} query-text pairs...")
    print(f"Top-K = {top_k}\n")
    print("="*80)
    
    hits = 0
    misses = 0
    ranks = []
    results_log = []
    
    for i, pair in enumerate(eval_pairs):
        query = pair["summary"]
        expected_text = pair["text"]
        expected_spans = [tuple(s) for s in pair["spans"]]
        
        print(f"\n[{i+1}/{len(eval_pairs)}] Query: {query}")
        
        try:
            results = pipeline.search(query, top_k=top_k, verbose=False)
            hit, rank = is_hit(expected_spans, results)
            
            if hit:
                hits += 1
                ranks.append(rank)
                status = f"HIT at rank {rank}"
            else:
                misses += 1
                status = "MISS"
            
            print(f"  -> {status}")
            
            results_log.append({
                "query": query,
                "expected_text": expected_text,
                "expected_spans": expected_spans,
                "hit": hit,
                "rank": rank,
                "top_result": results[0].text if results else None,
                "top_result_span": (results[0].meta["start_index"], results[0].meta["end_index"]) if results else None
            })
            
        except Exception as e:
            print(f"  -> ERROR: {e}")
            misses += 1
            results_log.append({
                "query": query,
                "expected_text": expected_text,
                "hit": False,
                "rank": None,
                "error": str(e)
            })
    
    # Calculate metrics
    total = hits + misses
    hit_rate = hits / total if total > 0 else 0
    avg_rank = sum(ranks) / len(ranks) if ranks else 0
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Total queries:    {total}")
    print(f"Hits:             {hits}")
    print(f"Misses:           {misses}")
    print(f"Hit Rate:         {hit_rate:.2%}")
    print(f"Avg Rank (hits):  {avg_rank:.2f}")
    print("="*80)
    
    # Save detailed results
    output_path = os.path.join(os.path.dirname(eval_pairs_path), "evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "total": total,
                "hits": hits,
                "misses": misses,
                "hit_rate": hit_rate,
                "avg_rank": avg_rank
            },
            "results": results_log
        }, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    chunks_path = os.path.join(project_root, "data/processed/verwandlung_chunks.json")
    emb_path = os.path.join(project_root, "data/processed/verwandlung.embeddings.npy")
    eval_pairs_path = os.path.join(script_dir, "eval_pairs.json")
    
    top_k = 10
    
    if len(sys.argv) >= 4:
        chunks_path = sys.argv[1]
        emb_path = sys.argv[2]
        eval_pairs_path = sys.argv[3]
    if len(sys.argv) >= 5:
        top_k = int(sys.argv[4])
    
    evaluate(chunks_path, emb_path, eval_pairs_path, top_k)


if __name__ == "__main__":
    main()
