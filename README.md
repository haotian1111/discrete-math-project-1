# Text Deduplication using LSH

This project implements a text deduplication system using Locality-Sensitive Hashing (LSH) techniques. It processes the Wiki40B dataset to identify and remove duplicate documents using both MinHash and SimHash approaches.

## Project Structure

```
project/
├── data/                  # Raw data directory
├── src/                   # Source code directory
│   ├── preprocess.py     # Text preprocessing and n-gram generation
│   ├── minhash.py        # MinHash implementation
│   ├── simhash.py        # SimHash implementation
│   ├── lsh.py            # LSH implementation
│   └── evaluate.py       # Evaluation metrics and reporting
└── processed_data/       # Output directory (created after running)
    ├── processed_data.parquet    # Preprocessed documents
    ├── minhash_results.parquet   # MinHash deduplication results
    ├── simhash_results.parquet   # SimHash deduplication results
    ├── lsh_results.parquet       # Combined LSH results
    └── evaluation_results.json    # Evaluation metrics
```

## Features

- **Text Preprocessing**:
  - HTML tag removal
  - Special character handling
  - N-gram generation (3-gram by default)
  
- **MinHash Implementation**:
  - Universal hash functions
  - MinHash signature computation
  - Jaccard similarity estimation
  
- **SimHash Implementation**:
  - Binary hash vector generation
  - Hamming distance-based similarity
  - Weighted feature vectors
  
- **LSH Implementation**:
  - Band-based LSH for MinHash
  - Bit-based LSH for SimHash
  - Efficient candidate pair generation
  
- **Evaluation Metrics**:
  - Duplicate rate calculation
  - Precision measurement
  - Processing time analysis
  - Method comparison

## Prerequisites

- Python 3.7 or higher
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - datasets (HuggingFace)
  - tqdm (for progress bars)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn datasets tqdm
```

## Usage

1. Run the preprocessing pipeline:
```bash
python src/preprocess.py
```

2. Run MinHash deduplication:
```bash
python src/minhash.py
```

3. Run SimHash deduplication:
```bash
python src/simhash.py
```

4. Run LSH-based deduplication:
```bash
python src/lsh.py
```

5. Evaluate the results:
```bash
python src/evaluate.py
```

## Implementation Details

### MinHash
- Uses 100 permutations by default
- Implements universal hash functions
- Computes Jaccard similarity estimates

### SimHash
- Uses 64-bit signatures by default
- Implements weighted feature vectors
- Computes Hamming distance-based similarity

### LSH
- MinHash LSH: 20 bands, 5 rows per band
- SimHash LSH: 8 bands, 8 bits per band
- Efficient candidate pair generation

## Evaluation Metrics

The system evaluates:
- Total number of documents
- Number of candidate pairs
- True duplicates vs. false positives
- Duplicate rate
- Precision
- Processing time
- Method-specific metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or issues, please open an issue in the repository. 
