import pandas as pd
import numpy as np
from typing import List, Dict, Set
import time
from collections import defaultdict

def load_data():
    """Load all necessary data for evaluation."""
    processed_data = pd.read_parquet('processed_data/processed_data.parquet')
    lsh_results = pd.read_parquet('processed_data/lsh_results.parquet')
    return processed_data, lsh_results

def compute_jaccard_similarity(ngrams1: List[str], ngrams2: List[str]) -> float:
    """
    Compute Jaccard similarity between two sets of n-grams.
    
    Args:
        ngrams1: List of n-grams from first document
        ngrams2: List of n-grams from second document
        
    Returns:
        Jaccard similarity score
    """
    set1 = set(ngrams1)
    set2 = set(ngrams2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

def evaluate_deduplication(processed_data: pd.DataFrame,
                         lsh_results: pd.DataFrame,
                         similarity_threshold: float = 0.8) -> Dict:
    """
    Evaluate the deduplication results.
    
    Args:
        processed_data: DataFrame containing processed documents
        lsh_results: DataFrame containing LSH results
        similarity_threshold: Threshold for considering documents as duplicates
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Create document lookup
    doc_lookup = {row['id']: row for _, row in processed_data.iterrows()}
    
    # Initialize metrics
    metrics = {
        'total_documents': len(processed_data),
        'candidate_pairs': len(lsh_results),
        'true_duplicates': 0,
        'false_positives': 0,
        'processing_time': 0
    }
    
    # Evaluate each candidate pair
    start_time = time.time()
    for _, row in lsh_results.iterrows():
        doc1_id = row['doc1_id']
        doc2_id = row['doc2_id']
        
        # Get n-grams
        ngrams1 = doc_lookup[doc1_id]['ngrams']
        ngrams2 = doc_lookup[doc2_id]['ngrams']
        
        # Compute similarity
        similarity = compute_jaccard_similarity(ngrams1, ngrams2)
        
        # Update metrics
        if similarity >= similarity_threshold:
            metrics['true_duplicates'] += 1
        else:
            metrics['false_positives'] += 1
    
    metrics['processing_time'] = time.time() - start_time
    
    # Compute additional metrics
    metrics['duplicate_rate'] = metrics['true_duplicates'] / metrics['total_documents']
    metrics['precision'] = metrics['true_duplicates'] / metrics['candidate_pairs'] if metrics['candidate_pairs'] > 0 else 0
    
    return metrics

def analyze_by_method(lsh_results: pd.DataFrame) -> Dict[str, Dict]:
    """
    Analyze results by LSH method (MinHash vs SimHash).
    
    Args:
        lsh_results: DataFrame containing LSH results
        
    Returns:
        Dictionary containing metrics for each method
    """
    method_metrics = {}
    
    for method in ['MinHash', 'SimHash']:
        method_results = lsh_results[lsh_results['method'] == method]
        method_metrics[method] = {
            'candidate_pairs': len(method_results),
            'unique_documents': len(set(method_results['doc1_id'].unique()) | 
                                 set(method_results['doc2_id'].unique()))
        }
    
    return method_metrics

def generate_report(metrics: Dict, method_metrics: Dict[str, Dict]):
    """
    Generate a detailed evaluation report.
    
    Args:
        metrics: Overall evaluation metrics
        method_metrics: Metrics by LSH method
    """
    print("\n=== Deduplication Evaluation Report ===\n")
    
    print("Overall Metrics:")
    print(f"Total Documents: {metrics['total_documents']}")
    print(f"Candidate Pairs: {metrics['candidate_pairs']}")
    print(f"True Duplicates: {metrics['true_duplicates']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"Duplicate Rate: {metrics['duplicate_rate']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Processing Time: {metrics['processing_time']:.2f} seconds")
    
    print("\nMethod-specific Metrics:")
    for method, method_metrics in method_metrics.items():
        print(f"\n{method}:")
        print(f"Candidate Pairs: {method_metrics['candidate_pairs']}")
        print(f"Unique Documents: {method_metrics['unique_documents']}")

def main():
    """Main function to run the evaluation."""
    print("Loading data...")
    processed_data, lsh_results = load_data()
    
    print("Evaluating deduplication results...")
    metrics = evaluate_deduplication(processed_data, lsh_results)
    
    print("Analyzing results by method...")
    method_metrics = analyze_by_method(lsh_results)
    
    print("Generating report...")
    generate_report(metrics, method_metrics)
    
    # Save results
    results = {
        'overall_metrics': metrics,
        'method_metrics': method_metrics
    }
    pd.DataFrame([results]).to_json('processed_data/evaluation_results.json')
    print("\nSaved evaluation results to processed_data/evaluation_results.json")

if __name__ == "__main__":
    main() 