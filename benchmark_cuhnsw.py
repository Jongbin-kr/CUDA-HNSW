#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import psutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'hnswlib-cuda'))
import hnswlib


class HNSWLibCudaBenchmark:
    def __init__(self, config):
        self.config = config
        self.index = None
        self.build_stats = {}
        self.search_stats = {}
        
    def load_data(self):
        print("Loading data...")
        train_file = self.config['train_embeddings']
        test_file = self.config['test_embeddings']
        
        self.train_data = np.load(train_file).astype(np.float32)
        self.test_data = np.load(test_file).astype(np.float32)
        self.num_elements = len(self.train_data)
        self.dim = self.train_data.shape[1]
        self.num_queries = len(self.test_data)
        
        print(f"Train: {self.train_data.shape}, Test: {self.test_data.shape}")
        
    def build_index(self):
        print("\nBuilding index (GPU)...")
        space = self.config['space']
        M = self.config['M']
        ef_construction = self.config['ef_construction']
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        self.index = hnswlib.Index(space=space, dim=self.dim)
        self.index.init_index(
            max_elements=self.num_elements,
            M=M,
            ef_construction=ef_construction,
            random_seed=100
        )
        
        build_start = time.time()
        self.index.add_items(self.train_data, np.arange(self.num_elements))
        build_time = time.time() - build_start
        
        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_used = mem_after - mem_before
        
        self.build_stats = {
            'total_time_sec': build_time,
            'throughput_items_per_sec': self.num_elements / build_time,
            'memory_mb': mem_used,
            'num_elements': self.num_elements,
            'dimension': self.dim,
            'M': M,
            'ef_construction': ef_construction,
        }
        
        print(f"Build time: {build_time:.3f}s, Memory: {mem_used:.2f}MB")
        return build_time
    
    def save_index(self, filepath):
        self.index.save_index(filepath)
        
    def run_search(self):
        print("\nRunning search (GPU)...")
        ef_search = self.config['ef_search']
        k = self.config['k']
        
        self.index.set_ef(ef_search)
        
        search_start = time.time()
        all_labels = []
        all_distances = []
        
        for i in range(self.num_queries):
            labels, distances = self.index.knn_query(self.test_data[i], k=k)
            all_labels.append(labels)
            all_distances.append(distances)
        
        search_time = time.time() - search_start
        avg_query_time = search_time / self.num_queries
        qps = self.num_queries / search_time
        
        all_labels = np.array(all_labels)
        all_distances = np.array(all_distances)
        
        self.search_stats = {
            'total_time_sec': search_time,
            'avg_query_time_sec': avg_query_time,
            'avg_query_time_us': avg_query_time * 1e6,
            'queries_per_second': qps,
            'num_queries': self.num_queries,
            'k': k,
            'ef_search': ef_search,
            'avg_distance': float(np.mean(all_distances)),
        }
        
        print(f"Search time: {search_time:.3f}s, QPS: {qps:.2f}")
        return all_labels, all_distances
    
    def save_results(self, output_file):
        results = {
            'implementation': 'hnswlib-cuda',
            'version': 'GPU-accelerated (CUDA)',
            'dataset': {
                'train_size': self.num_elements,
                'test_size': self.num_queries,
                'dimension': self.dim,
            },
            'parameters': {
                'space': self.config['space'],
                'M': self.config['M'],
                'ef_construction': self.config['ef_construction'],
                'ef_search': self.config['ef_search'],
                'k': self.config['k'],
            },
            'build_stats': self.build_stats,
            'search_stats': self.search_stats,
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def main():
    config = {
        'train_embeddings': 'data/embeddings/wikitext_train_simple_dim128.npy',
        'test_embeddings': 'data/embeddings/wikitext_test_simple_dim128.npy',
        'space': 'l2',
        'M': 16,
        'ef_construction': 200,
        'ef_search': 50,
        'k': 10,
    }
    
    print("GPU Benchmark (hnswlib-cuda)")
    
    benchmark = HNSWLibCudaBenchmark(config)
    benchmark.load_data()
    benchmark.build_index()
    benchmark.save_index('data/cuhnsw_index.bin')
    benchmark.run_search()
    benchmark.save_results('results_cuhnsw.json')
    
    print("\nDone.")


if __name__ == '__main__':
    main()
