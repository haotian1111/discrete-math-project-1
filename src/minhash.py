import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import mmh3
from joblib import Parallel, delayed

class MinHash:
    def __init__(self, num_permutations: int = 100):
        self.num_permutations = num_permutations
        self.prime = 2147483647  # 2^31 - 1
        self.a = np.random.randint(1, self.prime, num_permutations, dtype=np.int64)
        self.b = np.random.randint(0, self.prime, num_permutations, dtype=np.int64)
    
    def _hash_function(self, x: str, a: int, b: int) -> int:
        """使用MurmurHash的增强哈希函数"""
        x_hash = mmh3.hash64(x)[0] % self.prime
        return (a * x_hash + b) % self.prime
    
    def compute_signature(self, ngrams: List[str]) -> np.ndarray:
        """向量化优化的签名计算"""
        if not ngrams:
            return np.full(self.num_permutations, self.prime - 1)
        
        hash_matrix = np.array([
            [self._hash_function(ngram, a, b) for a, b in zip(self.a, self.b)]
            for ngram in ngrams
        ])
        return np.min(hash_matrix, axis=0)
    
    def compute_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        return np.mean(sig1 == sig2)

def find_candidate_pairs_lsh(
    signatures: Dict[str, np.ndarray],
    bands: int = 20,
    rows: int = 5,
    similarity_threshold: float = 0.8
) -> List[Tuple[str, str]]:
    """基于LSH的候选对查找"""
    minhash = MinHash()
    buckets = defaultdict(list)
    candidate_pairs = set()
    
    # 分桶
    for doc_id, sig in signatures.items():
        for band in range(bands):
            start = band * rows
            end = min(start + rows, len(sig))
            bucket_key = tuple(sig[start:end])
            buckets[bucket_key].append(doc_id)
    
    # 精细比对
    for bucket in buckets.values():
        if len(bucket) > 1:
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    doc1, doc2 = bucket[i], bucket[j]
                    similarity = minhash.compute_similarity(
                        signatures[doc1], signatures[doc2]
                    )
                    if similarity >= similarity_threshold:
                        candidate_pairs.add((doc1, doc2))
    
    return list(candidate_pairs)

def main():
    # 加载数据
    processed_data = pd.read_parquet('processed_data/processed_data.parquet')
    
    # 计算签名（并行优化）
    minhash = MinHash()
    signatures = dict(Parallel(n_jobs=4)(
        delayed(lambda row: (row['id'], minhash.compute_signature(row['ngrams'])))(row)
        for _, row in processed_data.iterrows()
    ))
    
    # 查找候选对（LSH优化）
    candidate_pairs = find_candidate_pairs_lsh(signatures)
    
    # 保存结果
    results_df = pd.DataFrame(candidate_pairs, columns=['doc1_id', 'doc2_id'])
    results_df.to_parquet('processed_data/minhash_results.parquet')

if __name__ == "__main__":
    main()