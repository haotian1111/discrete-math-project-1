import numpy as np
import pandas as pd
from typing import List, Dict
from collections import Counter
import hashlib

class SimHash:
    def __init__(self, num_bits: int = 64):
        """
        Initialize SimHash with specified number of bits.
        
        Args:
            num_bits: Number of bits in the hash (default: 64)
        """
        self.num_bits = num_bits
    
    def _hash(self, word: str) -> np.ndarray:
        """
        Generate a binary hash vector for a word.
        
        Args:
            word: Input word to hash
            
        Returns:
            Binary hash vector
        """
        # Use MD5 hash and convert to binary
        hash_value = int(hashlib.md5(word.encode()).hexdigest(), 16)
        binary = bin(hash_value)[2:].zfill(128)[:self.num_bits]
        return np.array([1 if b == '1' else -1 for b in binary])
    
    def compute_signature(self, ngrams: List[str]) -> str:
        """
        Compute SimHash signature for a set of n-grams.
        
        Args:
            ngrams: List of n-grams from a document
            
        Returns:
            Binary string representing the SimHash signature
        """
        # Count n-gram frequencies
        ngram_counts = Counter(ngrams)
        
        # Initialize hash vector
        hash_vector = np.zeros(self.num_bits)
        
        # Sum weighted hash vectors
        for ngram, count in ngram_counts.items():
            hash_vector += self._hash(ngram) * count
        
        # Convert to binary string
        signature = ''.join(['1' if x > 0 else '0' for x in hash_vector])
        return signature
    
    def compute_similarity(self, sig1: str, sig2: str) -> float:
        """
        Compute Hamming distance-based similarity between two signatures.
        
        Args:
            sig1: SimHash signature of first document
            sig2: SimHash signature of second document
            
        Returns:
            Similarity score (1 - normalized Hamming distance)
        """
        # Convert binary strings to arrays
        arr1 = np.array([int(b) for b in sig1])
        arr2 = np.array([int(b) for b in sig2])
        
        # Compute Hamming distance
        hamming_distance = np.sum(arr1 != arr2)
        
        # Convert to similarity score
        return 1 - (hamming_distance / self.num_bits)

def process_documents(documents: pd.DataFrame, num_bits: int = 64) -> Dict[str, str]:
    """
    Process all documents and compute their SimHash signatures.
    
    Args:
        documents: DataFrame containing document information
        num_bits: Number of bits in the hash
        
    Returns:
        Dictionary mapping document IDs to their SimHash signatures
    """
    simhash = SimHash(num_bits)
    signatures = {}
    
    for _, row in documents.iterrows():
        doc_id = row['id']
        ngrams = row['ngrams']
        signatures[doc_id] = simhash.compute_signature(ngrams)
    
    return signatures

def find_candidate_pairs(signatures: Dict[str, str], 
                        threshold: float = 0.8) -> List[tuple]:
    """
    Find candidate pairs of similar documents using SimHash signatures.
    
    Args:
        signatures: Dictionary mapping document IDs to their signatures
        threshold: Similarity threshold for considering documents as duplicates
        
    Returns:
        List of tuples containing pairs of similar document IDs
    """
    candidate_pairs = []
    doc_ids = list(signatures.keys())
    simhash = SimHash()
    
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            doc1_id = doc_ids[i]
            doc2_id = doc_ids[j]
            
            similarity = simhash.compute_similarity(
                signatures[doc1_id],
                signatures[doc2_id]
            )
            
            if similarity >= threshold:
                candidate_pairs.append((doc1_id, doc2_id))
    
    return candidate_pairs

def main():
    """Example usage of SimHash for document deduplication."""
    # Load processed documents
    processed_data = pd.read_parquet('processed_data/processed_data.parquet')
    
    # Compute SimHash signatures
    print("Computing SimHash signatures...")
    signatures = process_documents(processed_data)
    
    # Find candidate pairs
    print("Finding candidate pairs...")
    candidate_pairs = find_candidate_pairs(signatures)
    
    print(f"Found {len(candidate_pairs)} candidate pairs")
    
    # Save results
    results_df = pd.DataFrame(candidate_pairs, columns=['doc1_id', 'doc2_id'])
    results_df.to_parquet('processed_data/simhash_results.parquet')
    print("Saved results to processed_data/simhash_results.parquet")

if __name__ == "__main__":
    main() 