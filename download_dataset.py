#!/usr/bin/env python3
"""
Download Salesforce/wikitext dataset from HuggingFace
and prepare it for HNSW experiments.
"""
import os
import numpy as np
from datasets import load_dataset

def download_wikitext(subset="wikitext-2-raw-v1", cache_dir="./data"):
    """
    Download WikiText dataset from HuggingFace.
    
    Args:
        subset: which wikitext version to download
                - "wikitext-2-raw-v1" (small, ~4MB)
                - "wikitext-103-raw-v1" (large, ~500MB)
        cache_dir: where to cache the dataset
    """
    print(f"Downloading {subset} dataset...")
    print(f"Cache directory: {cache_dir}")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Download dataset
    dataset = load_dataset("wikitext", subset, cache_dir=cache_dir)
    
    print(f"\nDataset loaded successfully!")
    print(f"Available splits: {list(dataset.keys())}")
    
    for split in dataset.keys():
        print(f"\n{split} split:")
        print(f"  - Number of examples: {len(dataset[split])}")
        print(f"  - Features: {dataset[split].features}")
        print(f"  - First example length: {len(dataset[split][0]['text'])} chars")
    
    return dataset

def save_text_samples(dataset, output_dir="./data", num_samples=10000):
    """
    Save text samples to files for processing.
    
    Args:
        dataset: HuggingFace dataset object
        output_dir: directory to save samples
        num_samples: number of samples to extract from train split
    """
    print(f"\nSaving text samples to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract train samples
    train_texts = []
    for i, example in enumerate(dataset['train']):
        text = example['text'].strip()
        if len(text) > 50:  # Filter out very short texts
            train_texts.append(text)
        if len(train_texts) >= num_samples:
            break
    
    # Save to file
    train_file = os.path.join(output_dir, "wikitext_train.txt")
    with open(train_file, 'w', encoding='utf-8') as f:
        for text in train_texts:
            f.write(text + "\n")
    
    print(f"Saved {len(train_texts)} training samples to {train_file}")
    
    # Extract validation samples
    val_texts = []
    for example in dataset['validation']:
        text = example['text'].strip()
        if len(text) > 50:
            val_texts.append(text)
    
    val_file = os.path.join(output_dir, "wikitext_validation.txt")
    with open(val_file, 'w', encoding='utf-8') as f:
        for text in val_texts:
            f.write(text + "\n")
    
    print(f"Saved {len(val_texts)} validation samples to {val_file}")
    
    # Extract test samples
    test_texts = []
    for example in dataset['test']:
        text = example['text'].strip()
        if len(text) > 50:
            test_texts.append(text)
    
    test_file = os.path.join(output_dir, "wikitext_test.txt")
    with open(test_file, 'w', encoding='utf-8') as f:
        for text in test_texts:
            f.write(text + "\n")
    
    print(f"Saved {len(test_texts)} test samples to {test_file}")
    
    return train_texts, val_texts, test_texts

def main():
    """Main function to download and prepare dataset."""
    # Download dataset (start with smaller wikitext-2)
    dataset = download_wikitext(subset="wikitext-2-raw-v1", cache_dir="./data/cache")
    
    # Save text samples
    train_texts, val_texts, test_texts = save_text_samples(dataset, output_dir="./data")
    
    print("\n" + "="*50)
    print("Dataset download and preparation complete!")
    print("="*50)
    print(f"Train samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    print("\nNext steps:")
    print("1. Convert texts to embeddings using a model (e.g., sentence-transformers)")
    print("2. Save embeddings in format compatible with HNSW (e.g., .npy, .h5)")
    print("3. Run HNSW benchmarks on both C++ and CUDA implementations")

if __name__ == "__main__":
    main()
