#!/usr/bin/env python3
import numpy as np
import time
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "hnswlib"))
sys.path.insert(0, str(project_root / "hnswlib-cuda"))

def load_data():
    train_file = project_root / "data/embeddings/wikitext_train_simple_dim128.npy"
    test_file = project_root / "data/embeddings/wikitext_test_simple_dim128.npy"
    
    if not train_file.exists() or not test_file.exists():
        print("Error: Run generate_embeddings.py first")
        sys.exit(1)
    
    train_data = np.load(train_file).astype('float32')
    test_data = np.load(test_file).astype('float32')
    
    print(f"Train: {train_data.shape}, Test: {test_data.shape}")
    return train_data, test_data


def benchmark_cpu(train_data, test_data):
    import hnswlib
    
    num_elements = len(train_data)
    num_queries = len(test_data)
    dim = train_data.shape[1]
    
    print(f"\n[CPU] Benchmarking...")
    
    for _ in range(3):
        temp_index = hnswlib.Index(space='l2', dim=dim)
        temp_index.init_index(max_elements=100, ef_construction=50, M=16)
        temp_index.add_items(train_data[:100])
    
    build_times = []
    for run in range(5):
        index = hnswlib.Index(space='l2', dim=dim)
        index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        
        start = time.perf_counter()
        index.add_items(train_data)
        build_times.append(time.perf_counter() - start)
    
    build_time = np.median(build_times) * 1000
    
    index.set_ef(50)
    search_times = []
    for run in range(5):
        start = time.perf_counter()
        labels, distances = index.knn_query(test_data, k=10)
        search_times.append(time.perf_counter() - start)
    
    search_time = np.median(search_times) * 1000
    
    build_ops = num_elements * 16 * 200
    search_ops = num_queries * 50
    
    results = {
        'build_time_ms': float(build_time),
        'search_time_ms': float(search_time),
        'build_throughput': float(build_ops / (build_time / 1000)),
        'search_throughput': float(search_ops / (search_time / 1000)),
        'build_latency_ns': float((build_time * 1e6) / build_ops),
        'search_latency_ns': float((search_time * 1e6) / search_ops),
    }
    
    print(f"Build: {build_time:.2f}ms, Search: {search_time:.2f}ms")
    return results


def benchmark_gpu(train_data, test_data):
    import hnswlib
    
    num_elements = len(train_data)
    num_queries = len(test_data)
    dim = train_data.shape[1]
    
    print(f"\n[GPU] Benchmarking...")
    
    for _ in range(3):
        temp_index = hnswlib.Index(space='l2', dim=dim)
        temp_index.init_index(max_elements=100, ef_construction=50, M=16)
        temp_index.add_items(train_data[:100])
    
    build_times = []
    final_index = None
    for run in range(5):
        index = hnswlib.Index(space='l2', dim=dim)
        index.init_index(max_elements=num_elements, ef_construction=200, M=16)
        
        start = time.perf_counter()
        index.add_items(train_data)
        build_times.append(time.perf_counter() - start)
        
        if run == 4:
            final_index = index
    
    build_time = np.median(build_times) * 1000
    
    search_times = []
    if final_index:
        final_index.set_ef(50)
        for run in range(5):
            start = time.perf_counter()
            labels, distances = final_index.knn_query(test_data, k=10)
            search_times.append(time.perf_counter() - start)
    
    search_time = np.median(search_times) * 1000
    
    build_ops = num_elements * 16 * 200
    search_ops = num_queries * 50
    
    results = {
        'build_time_ms': float(build_time),
        'search_time_ms': float(search_time),
        'build_throughput': float(build_ops / (build_time / 1000)),
        'search_throughput': float(search_ops / (search_time / 1000)),
        'build_latency_ns': float((build_time * 1e6) / build_ops),
        'search_latency_ns': float((search_time * 1e6) / search_ops),
    }
    
    print(f"Build: {build_time:.2f}ms, Search: {search_time:.2f}ms")
    return results


def compare(cpu, gpu):
    if not cpu or not gpu:
        return
    
    print("\n" + "="*60)
    print("Distance Computation Performance")
    print("="*60)
    
    print(f"\nBuild:")
    print(f"  CPU: {cpu['build_time_ms']:.2f}ms ({cpu['build_latency_ns']:.2f}ns/op)")
    print(f"  GPU: {gpu['build_time_ms']:.2f}ms ({gpu['build_latency_ns']:.2f}ns/op)")
    print(f"  Speedup: {cpu['build_time_ms']/gpu['build_time_ms']:.2f}x")
    
    print(f"\nSearch:")
    print(f"  CPU: {cpu['search_time_ms']:.2f}ms ({cpu['search_latency_ns']:.2f}ns/op)")
    print(f"  GPU: {gpu['search_time_ms']:.2f}ms ({gpu['search_latency_ns']:.2f}ns/op)")
    print(f"  Speedup: {cpu['search_time_ms']/gpu['search_time_ms']:.2f}x")


def main():
    print("Distance Computation Benchmark")
    print("="*60)
    
    train_data, test_data = load_data()
    
    cpu_results = benchmark_cpu(train_data, test_data)
    gpu_results = benchmark_gpu(train_data, test_data)
    
    compare(cpu_results, gpu_results)
    
    output = {'cpu': cpu_results, 'gpu': gpu_results}
    with open(project_root / "results_distance_benchmark.json", 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nDone.")


if __name__ == "__main__":
    main()
