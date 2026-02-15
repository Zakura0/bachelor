"""
Ablation study evaluation script.
Tests different pipeline configurations and compares their performance.
"""

import json
import os
import sys
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from data.eval.ablation_search import AblationSearchPipeline


def spans_overlap(span1: tuple, span2: tuple) -> bool:
    """Check if two spans overlap."""
    start1, end1 = span1
    start2, end2 = span2
    return start1 < end2 and start2 < end1


def is_hit(expected_spans: list, results: list) -> tuple:
    """
    Check if any expected span overlaps with any result chunk.
    Returns (is_hit: bool, rank: int or None)
    """
    for rank, r in enumerate(results, start=1):
        result_start = r.meta["start_index"]
        result_end = r.meta["end_index"]
        
        for expected_start, expected_end in expected_spans:
            if spans_overlap((expected_start, expected_end), (result_start, result_end)):
                return True, rank
    
    return False, None


def evaluate_config(pipeline: AblationSearchPipeline, eval_pairs: list, top_k: int = 10) -> dict:
    """Evaluate a single pipeline configuration."""
    hits = 0
    misses = 0
    ranks = []
    
    for pair in eval_pairs:
        query = pair["summary"]
        expected_spans = [tuple(s) for s in pair["spans"]]
        
        try:
            results = pipeline.search(query, top_k=top_k, verbose=False)
            hit, rank = is_hit(expected_spans, results)
            
            if hit:
                hits += 1
                ranks.append(rank)
            else:
                misses += 1
                
        except Exception as e:
            print(f"  ERROR on query '{query[:50]}...': {e}")
            misses += 1
    
    total = hits + misses
    hit_rate = hits / total if total > 0 else 0
    avg_rank = sum(ranks) / len(ranks) if ranks else 0
    
    return {
        "total": total,
        "hits": hits,
        "misses": misses,
        "hit_rate": hit_rate,
        "avg_rank": avg_rank
    }


def run_ablation_study(chunks_path: str, emb_path: str, eval_pairs_path: str, top_k: int = 10):
    """Run ablation study with all configurations."""
    
    print("Loading eval pairs...")
    with open(eval_pairs_path, "r", encoding="utf-8") as f:
        eval_pairs = json.load(f)
    
    print(f"Running ablation study with {len(eval_pairs)} queries, Top-K={top_k}\n")
    print("="*80)
    
    configs = [
        {"name": "Full Pipeline", "tfidf": True, "emb": True, "rerank": True, "nli": True},
        {"name": "No NLI", "tfidf": True, "emb": True, "rerank": True, "nli": False},
        {"name": "No Reranker", "tfidf": True, "emb": True, "rerank": False, "nli": True},
        {"name": "No Reranker, No NLI", "tfidf": True, "emb": True, "rerank": False, "nli": False},
        {"name": "TF-IDF Only", "tfidf": True, "emb": False, "rerank": False, "nli": False},
        {"name": "Embeddings Only", "tfidf": False, "emb": True, "rerank": False, "nli": False},
        {"name": "TF-IDF + Reranker", "tfidf": True, "emb": False, "rerank": True, "nli": False},
        {"name": "Embeddings + Reranker", "tfidf": False, "emb": True, "rerank": True, "nli": False},
    ]
    
    results_all = []
    
    for config in configs:
        print(f"\n[Testing: {config['name']}]")
        
        pipeline = AblationSearchPipeline(
            chunks_path, emb_path,
            use_tfidf=config["tfidf"],
            use_embeddings=config["emb"],
            use_reranker=config["rerank"],
            use_nli=config["nli"]
        )
        
        metrics = evaluate_config(pipeline, eval_pairs, top_k)
        
        print(f"  Hits: {metrics['hits']}/{metrics['total']} ({metrics['hit_rate']:.1%})")
        print(f"  Avg Rank: {metrics['avg_rank']:.2f}")
        
        results_all.append({
            "config": config,
            "metrics": metrics
        })
    
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"{'Configuration':<30} {'Hit Rate':>10} {'Avg Rank':>10} {'Hits':>8}")
    print("-"*80)
    
    results_sorted = sorted(results_all, key=lambda x: x["metrics"]["hit_rate"], reverse=True)
    
    for result in results_sorted:
        config_name = result["config"]["name"]
        metrics = result["metrics"]
        print(f"{config_name:<30} {metrics['hit_rate']:>9.1%} {metrics['avg_rank']:>10.2f} {metrics['hits']:>4}/{metrics['total']:<3}")
    
    print("="*80)
    
    output_path = os.path.join(os.path.dirname(eval_pairs_path), "ablation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "top_k": top_k,
            "num_queries": len(eval_pairs),
            "results": results_all
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    chunks_path = os.path.join(project_root, "data/processed/verwandlung_chunks.json")
    emb_path = os.path.join(project_root, "data/processed/verwandlung.embeddings.npy")
    eval_pairs_path = os.path.join(project_root, "data/eval/eval_pairs.json")
    
    top_k = 10
    
    if len(sys.argv) >= 4:
        chunks_path = sys.argv[1]
        emb_path = sys.argv[2]
        eval_pairs_path = sys.argv[3]
    if len(sys.argv) >= 5:
        top_k = int(sys.argv[4])
    
    run_ablation_study(chunks_path, emb_path, eval_pairs_path, top_k)


if __name__ == "__main__":
    main()
