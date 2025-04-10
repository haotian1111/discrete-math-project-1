import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
import os

def load_local_data():
    """Load local parquet files."""
    print("Loading local parquet files...")
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    validation_data = pd.read_parquet(os.path.join(data_dir, '0000.parquet'))
    test_data = pd.read_parquet(os.path.join(data_dir, '0001.parquet'))
    return validation_data, test_data

def clean_text(text):
    """Clean text by removing HTML tags, special characters, and extra whitespace."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text.strip()

def create_ngrams(text, n=3):
    """Create n-grams from text."""
    vectorizer = CountVectorizer(ngram_range=(n, n), analyzer='char')
    try:
        X = vectorizer.fit_transform([text])
        return vectorizer.get_feature_names_out()
    except:
        return []

def preprocess_dataset(dataset, output_dir):
    """Preprocess the dataset and save to parquet files."""
    print(f"Preprocessing {len(dataset)} documents...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process documents
    processed_data = []
    for i, (_, row) in enumerate(dataset.iterrows()):
        if i % 1000 == 0:
            print(f"Processing document {i}/{len(dataset)}")
        
        # Clean text
        cleaned_text = clean_text(row['text'])
        
        # Create n-grams
        ngrams = create_ngrams(cleaned_text)
        
        processed_data.append({
            'id': i,  # Using index as ID since we don't have IDs in the local files
            'text': cleaned_text,
            'ngrams': ngrams
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(processed_data)
    output_file = os.path.join(output_dir, 'processed_data.parquet')
    df.to_parquet(output_file)
    print(f"Saved processed data to {output_file}")
    
    return df

def main():
    """Main function to run the preprocessing pipeline."""
    # Load data
    validation_data, test_data = load_local_data()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'processed_data')
    
    # Process validation set
    print("\nProcessing validation set...")
    validation_df = preprocess_dataset(validation_data, output_dir)
    
    # Process test set
    print("\nProcessing test set...")
    test_df = preprocess_dataset(test_data, output_dir)
    
    print("\nPreprocessing completed successfully!")

if __name__ == "__main__":
    main()


