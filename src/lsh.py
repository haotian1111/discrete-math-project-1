import numpy as np
import pandas as pd
from typing import List, Dict, Set
from collections import defaultdict
import hashlib

class LSH:
    def __init__(self, num_bands: int = 20, num_rows: int = 5):
        """
        Initialize LSH with specified number of bands and rows.
        
        Args:
            num_bands: Number of bands for MinHash LSH
            num_rows: Number of rows per band
        """
        self.num_bands = num_bands
        self.num_rows = num_rows
        self.bands = defaultdict(list)
    
    def _hash_band(self, band: np.ndarray) -> str:
        """
        Hash a band of MinHash signatures.
        
        Args:
            band: Array of MinHash signatures for a band
            
        Returns:
            Hash string for the band
        """
        return hashlib.md5(band.tobytes()).hexdigest()
    
    def add_signatures(self, doc_id: str, signatures: np.ndarray):
        """
        Add document signatures to LSH bands.
        
        Args:
            doc_id: Document ID
            signatures: MinHash signatures for the document
        """
        for band_idx in range(self.num_bands):
            start = band_idx * self.num_rows
            end = start + self.num_rows
            band = signatures[start:end]
            band_hash = self._hash_band(band)
            self.bands[band_hash].append(doc_id)
    
    def get_candidate_pairs(self) -> Set[tuple]:
        """
        Get candidate pairs of similar documents.
        
        Returns:
            Set of tuples containing pairs of document IDs
        """
        candidate_pairs = set()
        
        for band_docs in self.bands.values():
            if len(band_docs) > 1:
                for i in range(len(band_docs)):
                    for j in range(i + 1, len(band_docs)):
                        candidate_pairs.add((band_docs[i], band_docs[j]))
        
        return candidate_pairs

class SimHashLSH:
    def __init__(self, num_bits: int = 64, num_bands: int = 8):
        """
        Initialize SimHash LSH with specified parameters.
        
        Args:
            num_bits: Number of bits in SimHash
            num_bits_per_band: Number of bits per band
        """
        self.num_bits = num_bits
        self.num_bits_per_band = num_bits // num_bands
        self.bands = defaultdict(list)
    
    def add_signature(self, doc_id: str, signature: str):
        """
        Add document signature to LSH bands.
        
        Args:
            doc_id: Document ID
            signature: SimHash signature for the document
        """
        for band_idx in range(self.num_bits // self.num_bits_per_band):
            start = band_idx * self.num_bits_per_band
            end = start + self.num_bits_per_band
            band = signature[start:end]
            self.bands[band].append(doc_id)
    
    def get_candidate_pairs(self) -> Set[tuple]:
        """
        Get candidate pairs of similar documents.
        
        Returns:
            Set of tuples containing pairs of document IDs
        """
        candidate_pairs = set()
        
        for band_docs in self.bands.values():
            if len(band_docs) > 1:
                for i in range(len(band_docs)):
                    for j in range(i + 1, len(band_docs)):
                        candidate_pairs.add((band_docs[i], band_docs[j]))
        
        return candidate_pairs

def find_duplicates_minhash(documents: pd.DataFrame, 
                          signatures: Dict[str, np.ndarray],
                          num_bands: int = 20,
                          num_rows: int = 5) -> List[tuple]:
    """
    Find duplicate documents using MinHash LSH.
    
    Args:
        documents: DataFrame containing document information
        signatures: Dictionary mapping document IDs to their MinHash signatures
        num_bands: Number of bands for LSH
        num_rows: Number of rows per band
        
    Returns:
        List of tuples containing pairs of duplicate document IDs
    """
    lsh = LSH(num_bands, num_rows)
    
    # Add signatures to LSH
    for doc_id, sig in signatures.items():
        lsh.add_signatures(doc_id, sig)
    
    # Get candidate pairs
    candidate_pairs = lsh.get_candidate_pairs()
    
    return list(candidate_pairs)

def find_duplicates_simhash(documents: pd.DataFrame,
                          signatures: Dict[str, str],
                          num_bits: int = 64,
                          num_bands: int = 8) -> List[tuple]:
    """
    Find duplicate documents using SimHash LSH.
    
    Args:
        documents: DataFrame containing document information
        signatures: Dictionary mapping document IDs to their SimHash signatures
        num_bits: Number of bits in SimHash
        num_bands: Number of bands for LSH
        
    Returns:
        List of tuples containing pairs of duplicate document IDs
    """
    lsh = SimHashLSH(num_bits, num_bands)
    
    # Add signatures to LSH
    for doc_id, sig in signatures.items():
        lsh.add_signature(doc_id, sig)
    
    # Get candidate pairs
    candidate_pairs = lsh.get_candidate_pairs()
    
    return list(candidate_pairs)

def main():
    """Example usage of LSH for document deduplication."""
    # Load processed documents
    processed_data = pd.read_parquet('processed_data/processed_data.parquet')
    
    # Load MinHash signatures
    minhash_signatures = pd.read_parquet('processed_data/minhash_results.parquet')
    
    # Load SimHash signatures
    simhash_signatures = pd.read_parquet('processed_data/simhash_results.parquet')
    
    # Find duplicates using MinHash LSH
    print("Finding duplicates using MinHash LSH...")
    minhash_duplicates = find_duplicates_minhash(
        processed_data,
        minhash_signatures
    )
    
    # Find duplicates using SimHash LSH
    print("Finding duplicates using SimHash LSH...")
    simhash_duplicates = find_duplicates_simhash(
        processed_data,
        simhash_signatures
    )
    
    # Save results
    results_df = pd.DataFrame({
        'method': ['MinHash'] * len(minhash_duplicates) + ['SimHash'] * len(simhash_duplicates),
        'doc1_id': [p[0] for p in minhash_duplicates] + [p[0] for p in simhash_duplicates],
        'doc2_id': [p[1] for p in minhash_duplicates] + [p[1] for p in simhash_duplicates]
    })
    results_df.to_parquet('processed_data/lsh_results.parquet')
    print("Saved results to processed_data/lsh_results.parquet")

if __name__ == "__main__":
    main() 