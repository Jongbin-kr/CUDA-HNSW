import os
import sys
import json
import subprocess
import time
from pathlib import Path

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_data_files():
    print_header("Checking Data")
    
    required_files = [
        "data/embeddings/wikitext_train_simple_dim128.npy",
        "data/embeddings/wikitext_test_simple_dim128.npy"
    ]
    
    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
            print(f"Missing: {f}")
        else:
            print(f"Found: {f}")
    
    if missing:
        print("\nGenerating embeddings...")
        subprocess.run([sys.executable, "generate_embeddings.py"], check=True)
    
    return True

def run_cpu_benchmark():
    print_header("CPU Benchmark")
    start = time.time()
    subprocess.run([sys.executable, "benchmark_hnswlib.py"], check=True)
    elapsed = time.time() - start
    
    print(f"\nCompleted in {elapsed:.2f}s")
    
    if os.path.exists("results_hnswlib.json"):
        with open("results_hnswlib.json") as f:
            results = json.load(f)
        print(f"Build: {results['build_stats']['build_time']*1000:.2f}ms")
        print(f"Search: {results['search_stats']['total_query_time']*1000:.2f}ms")

def run_gpu_benchmark():
    print_header("GPU Benchmark")
    
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, timeout=5, check=True)
    except:
        print("Warning: GPU may not be available")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return False
    
    start = time.time()
    subprocess.run([sys.executable, "benchmark_cuhnsw.py"], check=True)
    elapsed = time.time() - start
    
    print(f"\nCompleted in {elapsed:.2f}s")
    
    if os.path.exists("results_cuhnsw.json"):
        with open("results_cuhnsw.json") as f:
            results = json.load(f)
        print(f"Build: {results['build_stats']['total_time_sec']*1000:.2f}ms")
        print(f"Search: {results['search_stats']['total_time_sec']*1000:.2f}ms")
    
    return True

def run_visualizations():
    print_header("Generating Plots")
    
    os.makedirs("report", exist_ok=True)
    
    if os.path.exists("visualize_distance_metrics.py"):
        print("Distance computation plot...")
        subprocess.run([sys.executable, "visualize_distance_metrics.py"], check=True)
    
    if os.path.exists("benchmark_parallel_search.py"):
        print("Search performance plot...")
        subprocess.run([sys.executable, "benchmark_parallel_search.py"], check=True)
    
    print("\nPlots saved to report/")

def compare_results():
    print_header("Results")
    
    if not os.path.exists("results_hnswlib.json") or not os.path.exists("results_cuhnsw.json"):
        print("Missing result files")
        return
    
    with open("results_hnswlib.json") as f:
        cpu = json.load(f)
    with open("results_cuhnsw.json") as f:
        gpu = json.load(f)
    
    build_speedup = cpu['build_stats']['build_time'] / gpu['build_stats']['total_time_sec']
    search_speedup = cpu['search_stats']['total_query_time'] / gpu['search_stats']['total_time_sec']
    
    print(f"\nBuild speedup: {build_speedup:.2f}x")
    print(f"Search speedup: {search_speedup:.2f}x")
    
    print(f"\nCPU build: {cpu['build_stats']['build_time']*1000:.2f}ms")
    print(f"GPU build: {gpu['build_stats']['total_time_sec']*1000:.2f}ms")
    
    print(f"\nCPU search: {cpu['search_stats']['total_query_time']*1000:.2f}ms")
    print(f"GPU search: {gpu['search_stats']['total_time_sec']*1000:.2f}ms")

def main():
    print("\nGPU-Accelerated HNSW Benchmark")
    print("Author: Jongbin Won (2025-23160)\n")
    
    if not os.path.exists("README.md"):
        print("Error: Run from project root directory")
        sys.exit(1)
    
    try:
        check_data_files()
        run_cpu_benchmark()
        gpu_success = run_gpu_benchmark()
        
        if gpu_success:
            run_visualizations()
            compare_results()
        
        print_header("Done")
        print("\nGenerated files:")
        print("  results_hnswlib.json")
        print("  results_cuhnsw.json")
        print("  report/*.png")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(1)

if __name__ == "__main__":
    main()
