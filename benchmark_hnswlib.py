#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import psutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'hnswlib'))
import hnswlib


class HNSWBenchmark:
    def __init__(self, config):
        self.config = config
        self.index = None
        self.build_stats = {}
        self.search_stats = {}
        
    def load_data(self):
        print("Loading data...")
        train_file = self.config['train_embeddings']
        test_file = self.config['test_embeddings']
        
        self.train_data = np.load(train_file)
        self.test_data = np.load(test_file)
        self.num_elements = len(self.train_data)
        self.dim = self.train_data.shape[1]
        self.num_queries = len(self.test_data)
        
        print(f"Train: {self.train_data.shape}, Test: {self.test_data.shape}")
        
    def build_index(self):
        print("\nBuilding index...")
        space = self.config['space']
        M = self.config['M']
        ef_construction = self.config['ef_construction']
        num_threads = self.config['num_threads']
        
        self.index = hnswlib.Index(space=space, dim=self.dim)
        self.index.init_index(
            max_elements=self.num_elements,
            M=M,
            ef_construction=ef_construction,
            random_seed=100
        )
        self.index.set_num_threads(num_threads)
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        build_start = time.time()
        self.index.add_items(self.train_data, np.arange(self.num_elements))
        build_time = time.time() - build_start
        
        mem_after = process.memory_info().rss / (1024 * 1024)
        mem_used = mem_after - mem_before
        
        self.build_stats = {
            'build_time': build_time,
            'throughput': self.num_elements / build_time,
            'memory_used_mb': mem_used,
            'num_elements': self.num_elements,
            'dimension': self.dim,
            'M': M,
            'ef_construction': ef_construction,
            'num_threads': num_threads
        }
        
        print(f"Build time: {build_time:.3f}s, Memory: {mem_used:.2f}MB")
        return self.build_stats
    
    def run_search(self):
        print("\nRunning search...")
        ef_search = self.config['ef_search']
        k = self.config['k']
        
        self.index.set_ef(ef_search)
        
        query_start = time.time()
        all_labels = []
        all_distances = []
        
        for i in range(self.num_queries):
            labels, distances = self.index.knn_query(self.test_data[i], k=k)
            all_labels.append(labels)
            all_distances.append(distances)
        
        query_time = time.time() - query_start
        avg_query_time = query_time / self.num_queries
        qps = self.num_queries / query_time
        
        all_labels = np.array(all_labels)
        all_distances = np.array(all_distances)
        
        self.search_stats = {
            'total_query_time': query_time,
            'avg_query_time_ms': avg_query_time * 1000,
            'queries_per_second': qps,
            'num_queries': self.num_queries,
            'k': k,
            'ef_search': ef_search,
            'avg_distance': float(np.mean(all_distances)),
        }
        
        print(f"Search time: {query_time:.3f}s, QPS: {qps:.2f}")
        return self.search_stats, all_labels, all_distances
    
    def save_index(self, filepath):
        self.index.save_index(filepath)
        
    def save_results(self, output_file):
        results = {
            'implementation': 'hnswlib (C++)',
            'config': self.config,
            'build_stats': self.build_stats,
            'search_stats': self.search_stats,
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def main():
    config = {
        'train_embeddings': './data/embeddings/wikitext_train_simple_dim128.npy',
        'test_embeddings': './data/embeddings/wikitext_test_simple_dim128.npy',
        'space': 'l2',
        'M': 16,
        'ef_construction': 200,
        'ef_search': 50,
        'k': 10,
        'num_threads': 4,
    }
    
    print("CPU Benchmark (hnswlib)")
    
    benchmark = HNSWBenchmark(config)
    benchmark.load_data()
    benchmark.build_index()
    benchmark.save_index('./data/hnswlib_index.bin')
    benchmark.run_search()
    benchmark.save_results('./results_hnswlib.json')
    
    print("\nDone.")


if __name__ == "__main__":
    main()
