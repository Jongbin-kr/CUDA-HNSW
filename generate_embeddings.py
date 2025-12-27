#!/usr/bin/env python3
"""
Generate embeddings from WikiText dataset for HNSW experiments.

This script provides two modes:
1. Random embeddings (for quick testing)
2. Real embeddings using sentence-transformers (requires additional installation)
"""
import os
import numpy as np
import h5py
import argparse

def generate_random_embeddings(texts, dim=128):
    """
    Generate random embeddings for testing purposes.
    
    Args:
        texts: list of text strings
        dim: embedding dimension
    
    Returns:
        numpy array of shape (len(texts), dim)
    """
    print(f"Generating random {dim}-dimensional embeddings for {len(texts)} texts...")
    
    # Generate random normalized embeddings
    embeddings = np.random.randn(len(texts), dim).astype(np.float32)
    
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    return embeddings

def generate_simple_embeddings(texts, dim=128):
    """
    Generate simple character-based embeddings (more realistic than random).
    Uses character frequency and basic text statistics.
    
    Args:
        texts: list of text strings
        dim: embedding dimension
    
    Returns:
        numpy array of shape (len(texts), dim)
    """
    print(f"Generating simple character-based {dim}-dimensional embeddings...")
    
    embeddings = []
    for i, text in enumerate(texts):
        if i % 1000 == 0:
            print(f"Processing {i}/{len(texts)}...")
        
        # Create a simple feature vector based on text characteristics
        vec = np.zeros(dim, dtype=np.float32)
        
        # Character frequency features (first 128 positions)
        for j, char in enumerate(text[:dim//2]):
            vec[ord(char) % (dim//2)] += 1.0
        
        # Add some text statistics
        if dim > 64:
            vec[dim//2] = len(text)  # length
            vec[dim//2 + 1] = text.count(' ')  # word count approximation
            vec[dim//2 + 2] = text.count('.')  # sentence count approximation
            vec[dim//2 + 3] = sum(1 for c in text if c.isupper())  # uppercase count
            vec[dim//2 + 4] = sum(1 for c in text if c.isdigit())  # digit count
        
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        embeddings.append(vec)
    
    return np.array(embeddings, dtype=np.float32)

def save_embeddings_numpy(embeddings, labels, output_file):
    """Save embeddings in .npy format."""
    print(f"Saving embeddings to {output_file}...")
    np.save(output_file, embeddings)
    print(f"Saved {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    
    # Also save labels if provided
    if labels is not None:
        label_file = output_file.replace('.npy', '_labels.npy')
        np.save(label_file, labels)
        print(f"Saved labels to {label_file}")

def save_embeddings_h5(embeddings, labels, output_file):
    """Save embeddings in HDF5 format (compatible with many HNSW benchmarks)."""
    print(f"Saving embeddings to {output_file}...")
    
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('embeddings', data=embeddings, compression='gzip')
        if labels is not None:
            # Save labels as strings
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('labels', data=labels, dtype=dt, compression='gzip')
    
    print(f"Saved {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]} to HDF5")

def save_embeddings_fbin(embeddings, output_file):
    """
    Save embeddings in .fbin format (common format for HNSW benchmarks).
    Format: [n_vectors (int32)][dim (int32)][vector_data (float32)]
    """
    print(f"Saving embeddings to {output_file} (fbin format)...")
    
    n, dim = embeddings.shape
    with open(output_file, 'wb') as f:
        # Write header
        f.write(np.array([n], dtype=np.int32).tobytes())
        f.write(np.array([dim], dtype=np.int32).tobytes())
        # Write data
        f.write(embeddings.tobytes())
    
    print(f"Saved {n} embeddings of dimension {dim}")

def load_texts(text_file):
    """Load texts from file."""
    print(f"Loading texts from {text_file}...")
    with open(text_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(texts)} texts")
    return texts

def main():
    parser = argparse.ArgumentParser(description='Generate embeddings from WikiText dataset')
    parser.add_argument('--input', type=str, default='./data/wikitext_train.txt',
                        help='Input text file')
    parser.add_argument('--output', type=str, default='./data/embeddings',
                        help='Output directory for embeddings')
    parser.add_argument('--dim', type=int, default=128,
                        help='Embedding dimension (default: 128)')
    parser.add_argument('--mode', type=str, choices=['random', 'simple'], default='simple',
                        help='Embedding generation mode: random or simple (default: simple)')
    parser.add_argument('--format', type=str, choices=['npy', 'h5', 'fbin'], default='npy',
                        help='Output format: npy, h5, or fbin (default: npy)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load texts
    texts = load_texts(args.input)
    
    if args.max_samples:
        texts = texts[:args.max_samples]
        print(f"Limited to {args.max_samples} samples")
    
    # Generate embeddings
    if args.mode == 'random':
        embeddings = generate_random_embeddings(texts, dim=args.dim)
    else:  # args.mode == 'simple'
        embeddings = generate_simple_embeddings(texts, dim=args.dim)
    
    # Create labels (just indices for now)
    labels = np.arange(len(texts))
    
    # Determine output filename
    base_name = os.path.basename(args.input).replace('.txt', '')
    
    # Save embeddings
    if args.format == 'npy':
        output_file = os.path.join(args.output, f'{base_name}_{args.mode}_dim{args.dim}.npy')
        save_embeddings_numpy(embeddings, labels, output_file)
    elif args.format == 'h5':
        output_file = os.path.join(args.output, f'{base_name}_{args.mode}_dim{args.dim}.h5')
        save_embeddings_h5(embeddings, labels, output_file)
    elif args.format == 'fbin':
        output_file = os.path.join(args.output, f'{base_name}_{args.mode}_dim{args.dim}.fbin')
        save_embeddings_fbin(embeddings, output_file)
    
    # Print statistics
    print("\n" + "="*60)
    print("Embedding Statistics:")
    print("="*60)
    print(f"Number of embeddings: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Data type: {embeddings.dtype}")
    print(f"Memory size: {embeddings.nbytes / (1024*1024):.2f} MB")
    print(f"Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    print(f"Min value: {embeddings.min():.4f}")
    print(f"Max value: {embeddings.max():.4f}")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Generate embeddings for validation and test sets")
    print("2. Build HNSW indices using both C++ and CUDA implementations")
    print("3. Run search queries and measure performance")
    print("\nFor train set:")
    print(f"  python generate_embeddings.py --input ./data/wikitext_train.txt --output ./data/embeddings")
    print("\nFor validation set:")
    print(f"  python generate_embeddings.py --input ./data/wikitext_validation.txt --output ./data/embeddings")
    print("\nFor test set (queries):")
    print(f"  python generate_embeddings.py --input ./data/wikitext_test.txt --output ./data/embeddings")

if __name__ == "__main__":
    main()
