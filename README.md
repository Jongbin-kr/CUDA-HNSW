refer to [My github repository](https://github.com/Jongbin-kr/CUDA-HNSW) for the source codes.


## Quick Start

### Prerequisites
- Python 3.8+
- CUDA 11.7+ (for GPU benchmarks)
- NVIDIA GPU (optional, for GPU experiments)

### Installation

```bash
# Install dependencies
pip install numpy pybind11 psutil matplotlib

# Build CPU version
cd hnswlib && pip install . && cd ..

# Build GPU version (requires CUDA)
cd hnswlib-cuda && pip install . && cd ..
```

### Run Experiments

```bash
# One-click reproduction (recommended)
python run_experiments.py
```

Or run individually:
```bash
python benchmark_hnswlib.py      # CPU benchmark
python benchmark_cuhnsw.py       # GPU benchmark
python visualize_distance_metrics.py
python benchmark_parallel_search.py
```




## Experimental Setup

### Hardware
- CPU: AMD EPYC 7502 (4 threads)
- GPU: NVIDIA RTX 3090 (24GB)
- RAM: 50GB

### Dataset
- Source: Wikipedia (Salesforce/wikitext)
- Training: 10,000 vectors (128-dim)
- Test: 1,940 queries (128-dim)

### Parameters
- M = 16, ef_construction = 200, ef_search = 50, k = 10

## GPU Optimizations

### 1. Warp-Level Distance Computation (3.85× speedup)

Modified distance calculation to use CUDA warp shuffle for 32-way parallel reduction instead of CPU SIMD 16-way operations.

**Implementation:** `hnswlib-cuda/hnswlib/space_l2_cuda.cuh`

```cuda
__device__ float L2SqrCUDA(const float* a, const float* b, int dim) {
    float val = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float diff = a[i] - b[i];
        val += diff * diff;
    }
    return warp_reduce_sum(val);  // 32-way parallel
}
```

### 2. Block-Level Parallel Search (7.35× speedup)

Launched 1,940 CUDA blocks (one per query) to process all queries simultaneously, compared to CPU's 4-thread sequential processing.

**Implementation:** `hnswlib-cuda/hnswlib/hnswalg_cuda.cuh`

```cuda
__global__ void SearchGraphKernel(...) {
    int query_id = blockIdx.x;
    // 128 threads cooperate on single query
    // All 1,940 queries run in parallel
}
```
