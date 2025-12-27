#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

cpu_ops = 32_000_000
gpu_ops = 32_000_000
cpu_time_ms = 348.09 * 0.25  
gpu_time_ms = 90.44 * 0.25

cpu_throughput = cpu_ops / (cpu_time_ms / 1000)
gpu_throughput = gpu_ops / (gpu_time_ms / 1000)

cpu_latency_ns = cpu_time_ms * 1e6 / cpu_ops
gpu_latency_ns = gpu_time_ms * 1e6 / gpu_ops

categories = ['CPU', 'GPU']
colors = ['#3498db', '#e74c3c']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
latencies = [cpu_latency_ns, gpu_latency_ns]
ax.bar(categories, latencies, color=colors, alpha=0.8, width=0.5)
ax.set_ylabel('Latency (ns/op)', fontsize=12)
ax.set_title('Distance Computation Latency', fontsize=13)
ax.grid(axis='y', alpha=0.3)

for i, lat in enumerate(latencies):
    ax.text(i, lat + lat*0.05, f'{lat:.2f}', ha='center', fontsize=11)

speedup = latencies[0] / latencies[1]
ax.text(0.5, 0.85, f'{speedup:.2f}x', transform=ax.transAxes, 
        ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))

ax = axes[1]
throughputs = [cpu_throughput/1e6, gpu_throughput/1e6]
ax.bar(categories, throughputs, color=colors, alpha=0.8, width=0.5)
ax.set_ylabel('Throughput (M ops/sec)', fontsize=12)
ax.set_title('Distance Computation Throughput', fontsize=13)
ax.grid(axis='y', alpha=0.3)

for i, tput in enumerate(throughputs):
    ax.text(i, tput + tput*0.05, f'{tput:.1f}', ha='center', fontsize=11)

speedup = throughputs[1] / throughputs[0]
ax.text(0.5, 0.85, f'{speedup:.2f}x', transform=ax.transAxes,
        ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat'))

plt.tight_layout()

output_dir = Path(__file__).parent / "report"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "distance_computation.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved: {output_file}")

plt.close()

print(f"\nCPU: {cpu_latency_ns:.2f} ns/op, {cpu_throughput/1e6:.2f} M ops/sec")
print(f"GPU: {gpu_latency_ns:.2f} ns/op, {gpu_throughput/1e6:.2f} M ops/sec")
print(f"Speedup: {cpu_latency_ns/gpu_latency_ns:.2f}x")
