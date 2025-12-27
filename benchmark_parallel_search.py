#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

project_root = Path(__file__).parent

with open(project_root / "results_hnswlib.json", 'r') as f:
    cpu_results = json.load(f)

with open(project_root / "results_cuhnsw.json", 'r') as f:
    gpu_results = json.load(f)

num_queries = cpu_results['search_stats']['num_queries']
cpu_total_time_ms = cpu_results['search_stats']['total_query_time'] * 1000
gpu_total_time_ms = gpu_results['search_stats']['total_time_sec'] * 1000

cpu_latency_us = cpu_results['search_stats']['avg_query_time_ms'] * 1000
gpu_latency_us = gpu_results['search_stats']['avg_query_time_us']

cpu_qps = cpu_results['search_stats']['queries_per_second']
gpu_qps = gpu_results['search_stats']['queries_per_second']

categories = ['CPU', 'GPU']
colors = ['#3498db', '#e74c3c']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
latencies = [cpu_latency_us, gpu_latency_us]
ax.bar(categories, latencies, color=colors, alpha=0.8, width=0.5)
ax.set_ylabel('Latency (us/query)', fontsize=11)
ax.set_title('Query Latency', fontsize=12)
ax.grid(axis='y', alpha=0.3)

for i, lat in enumerate(latencies):
    ax.text(i, lat + lat*0.05, f'{lat:.2f}', ha='center', fontsize=10)

speedup = latencies[0] / latencies[1]
ax.text(0.5, 0.85, f'{speedup:.2f}x', transform=ax.transAxes,
        ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat'))

ax = axes[1]
throughputs = [cpu_qps/1000, gpu_qps/1000]
ax.bar(categories, throughputs, color=colors, alpha=0.8, width=0.5)
ax.set_ylabel('Throughput (K qps)', fontsize=11)
ax.set_title('Search Throughput', fontsize=12)
ax.grid(axis='y', alpha=0.3)

for i, tput in enumerate(throughputs):
    ax.text(i, tput + tput*0.05, f'{tput:.1f}', ha='center', fontsize=10)

speedup = throughputs[1] / throughputs[0]
ax.text(0.5, 0.85, f'{speedup:.2f}x', transform=ax.transAxes,
        ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat'))

ax = axes[2]
total_times = [cpu_total_time_ms, gpu_total_time_ms]
ax.bar(categories, total_times, color=colors, alpha=0.8, width=0.5)
ax.set_ylabel('Time (ms)', fontsize=11)
ax.set_title(f'Total Time ({num_queries} queries)', fontsize=12)
ax.grid(axis='y', alpha=0.3)

for i, time in enumerate(total_times):
    ax.text(i, time + time*0.05, f'{time:.2f}', ha='center', fontsize=10)

speedup = total_times[0] / total_times[1]
ax.text(0.5, 0.85, f'{speedup:.2f}x', transform=ax.transAxes,
        ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat'))

plt.tight_layout()

output_dir = Path(__file__).parent / "report"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "search_by_query.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")

plt.close()

print(f"\nCPU: {cpu_latency_us:.2f} us/query, {cpu_qps/1000:.1f}K qps")
print(f"GPU: {gpu_latency_us:.2f} us/query, {gpu_qps/1000:.1f}K qps")
print(f"Speedup: {cpu_latency_us/gpu_latency_us:.2f}x")

results = {
    'num_queries': num_queries,
    'cpu': {'latency_us': cpu_latency_us, 'throughput_qps': cpu_qps, 'total_time_ms': cpu_total_time_ms},
    'gpu': {'latency_us': gpu_latency_us, 'throughput_qps': gpu_qps, 'total_time_ms': gpu_total_time_ms},
    'speedup': {'latency': cpu_latency_us / gpu_latency_us, 'throughput': gpu_qps / cpu_qps}
}

with open(project_root / "results_parallel_search.json", 'w') as f:
    json.dump(results, f, indent=2)
