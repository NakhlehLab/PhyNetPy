# CUDA Kernel Design for MCMC BiMarkers

This document explains the GPU acceleration strategy used in `MCMC_BiMarkers_CUDA.py`, which replaces C# DLL calls with CUDA kernels for phylogenetic network inference using biallelic markers (SNPs).

## Table of Contents
1. [Overview](#overview)
2. [Hardware Target](#hardware-target)
3. [Data Structures](#data-structures)
4. [Device Helper Functions](#device-helper-functions)
5. [Rule 0 Kernel](#rule-0-kernel)
6. [Rule 1 Kernel](#rule-1-kernel)
7. [Rule 2 Kernel](#rule-2-kernel)
8. [Rule 3 Kernel](#rule-3-kernel)
9. [Rule 4 Kernel](#rule-4-kernel)
10. [Memory Management](#memory-management)
11. [Performance Considerations](#performance-considerations)

---

## Overview

The SNP likelihood calculation (from [Rabier et al. 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005932)) traverses a phylogenetic network and computes partial likelihoods at each node using five rules (0-4). Each rule involves:

- **Site-level independence**: Computations for different sites are completely independent
- **Combinatorial loops**: Iterating over (n, r) pairs where n = lineage count, r = "red" allele count
- **Matrix operations**: Transition probability matrices and their exponentials

These properties make the algorithm highly parallelizable on GPUs.

### Parallelization Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    PARALLELIZATION HIERARCHY                    │
├─────────────────────────────────────────────────────────────────┤
│  Level 1: Sites (thousands)        → Thread blocks              │
│  Level 2: (n,r) pairs (hundreds)   → Threads within blocks      │
│  Level 3: Summations               → Sequential per thread      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Hardware Target

**Target GPU**: NVIDIA RTX 5070ti (Ada Lovelace / Blackwell architecture)

| Specification | Value | Implication |
|--------------|-------|-------------|
| CUDA Cores | ~8,000+ | High parallelism capacity |
| SM Count | ~60+ | Many concurrent thread blocks |
| L2 Cache | 48+ MB | Good for repeated matrix access |
| Memory Bandwidth | 500+ GB/s | Fast data transfer |
| Compute Capability | 8.9+ | Latest CUDA features available |

### Configuration Constants

```python
THREADS_PER_BLOCK = 256   # Sweet spot for occupancy vs. register pressure
MAX_BLOCKS = 65535        # CUDA grid dimension limit
```

**Why 256 threads per block?**
- Divisible by warp size (32)
- Provides good occupancy on modern GPUs
- Leaves room for registers and shared memory
- Works well for vector_len values typically seen (tens to hundreds)

---

## Data Structures

### The (n, r) Index Mapping

The algorithm works with pairs (n, r) where:
- `n` = number of lineages (1 to max_samples)
- `r` = number of "red" alleles (0 to n)

These are linearized into a 1D index for efficient GPU memory access:

```
(n, r) pairs laid out as:
Index:  0     1     2     3     4     5     6     7    ...
(n,r): (1,0) (1,1) (2,0) (2,1) (2,2) (3,0) (3,1) (3,2) ...
```

**Conversion formulas:**
```python
# (n, r) → index
index = n*(n+1)/2 - 1 + r  # Simplified: triangular number + offset

# index → (n, r)  
n = floor((-1 + sqrt(1 + 8*index)) / 2)  # Inverse triangular
r = index - n*(n-1)/2 - n + 1
```

### Array Layouts

| Array | Shape | Description |
|-------|-------|-------------|
| `F_b`, `F_t` | `[site_count, vector_len]` | Partial likelihoods per site and (n,r) |
| `Qt` | `[vector_len, vector_len]` | Transition matrix exponential |
| `reds` | `[site_count]` | Red allele counts at leaves |

---

## Device Helper Functions

These are `@cuda.jit(device=True)` functions that run on the GPU and are called by kernels:

### `d_n_to_index(n)`
Computes the starting index for a given n value.
```python
@cuda.jit(device=True)
def d_n_to_index(n: int64) -> int64:
    return int64(0.5 * (n - 1) * (n + 2))
```

### `d_nr_to_index(n, r)`
Maps (n, r) pair to linear index.
```python
@cuda.jit(device=True)
def d_nr_to_index(n: int64, r: int64) -> int64:
    return d_n_to_index(n) + r
```

### `d_index_to_n(index)` and `d_index_to_r(index, n)`
Inverse mappings from linear index back to (n, r).

### `d_comb(n, k)`
Binomial coefficient computation on GPU. Uses multiplicative formula to avoid factorial overflow:
```python
@cuda.jit(device=True)
def d_comb(n: int64, k: int64) -> float64:
    # C(n,k) = n!/(k!(n-k)!) computed as product
    result = 1.0
    for i in range(k):
        result = result * (n - i) / (i + 1)
    return result
```

---

## Rule 0 Kernel

### Mathematical Definition
Rule 0 initializes partial likelihoods at leaf nodes:

```
F(x_bot)[site, (n,r)] = 1  if n == samples AND r == reds[site]
                       = 0  otherwise
```

### Kernel Design

```python
@cuda.jit
def rule0_kernel(reds, site_count, vector_len, samples, output):
    # 2D grid: x-dim = sites, y-dim = (n,r) indices
    site = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    index = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    if site < site_count and index < vector_len:
        n = d_index_to_n(index)
        r = d_index_to_r(index, n)
        
        if int64(reds[site]) == r and n == samples:
            output[site, index] = 1.0
        else:
            output[site, index] = 0.0
```

### Grid/Block Configuration

```python
threads_per_block = (16, 16)  # 256 total threads
blocks_x = ceil(site_count / 16)
blocks_y = ceil(vector_len / 16)
blocks = (blocks_x, blocks_y)
```

**Why 2D grid (16×16)?**
- Natural mapping to 2D output array
- Each thread writes exactly one output element
- No thread divergence (all threads do same work)
- Coalesced memory writes (threads in same warp write adjacent sites)

### Memory Access Pattern
```
Thread (0,0) → output[0, 0]
Thread (0,1) → output[0, 1]
Thread (1,0) → output[1, 0]
...
```
✅ Coalesced writes along the site dimension

---

## Rule 1 Kernel

### Mathematical Definition
Rule 1 transitions from branch bottom to top using the matrix exponential:

```
F(x_top)[n_top, r_top] = Σ F(x_bot)[n_b, r_b] × Qt[n_b,r_b → n_top,r_top]
                         n_b≥n_top
```

This is essentially a sparse matrix-vector product per site.

### Kernel Design

```python
@cuda.jit
def rule1_kernel(F_b, Qt, site_count, vector_len, mx, output):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    total_work = site_count * vector_len
    if tid >= total_work:
        return
    
    # Decode thread ID to (site, top_index)
    site = tid // vector_len
    top_index = tid % vector_len
    
    n_top = d_index_to_n(top_index)
    r_top = d_index_to_r(top_index, n_top)
    
    if n_top > mx:
        output[site, top_index] = 0.0
        return
    
    # Summation loop (sequential per thread)
    evaluation = 0.0
    max_index = d_n_to_index(mx + 1)
    
    for bot_index in range(max_index):
        n_b = d_index_to_n(bot_index)
        r_b = d_index_to_r(bot_index, n_b)
        
        if n_b >= n_top:
            qt_val = Qt[bot_index, top_index]
            fb_val = F_b[site, bot_index]
            evaluation += fb_val * qt_val
    
    output[site, top_index] = evaluation
```

### Grid/Block Configuration

```python
total_work = site_count * vector_len
threads = 256
blocks = min(ceil(total_work / 256), 65535)
```

**Why 1D grid?**
- Flattens the 2D work into linear thread assignment
- Simple work distribution
- Each thread computes one output element independently

### Work Distribution
```
Thread 0 → site=0, index=0
Thread 1 → site=0, index=1
...
Thread vector_len → site=1, index=0
```

### Computational Intensity
Each thread performs:
- ~O(vector_len) iterations in the summation loop
- 2 memory reads per iteration (Qt, F_b)
- 1 memory write at the end

This is **compute-bound** with good arithmetic intensity.

---

## Rule 2 Kernel

### Mathematical Definition
Rule 2 combines two disjoint branches at a speciation node:

```
F(z_bot)[n_z, r_z] = Σ C(n_x,r_x) × C(n_z-n_x, r_z-r_x) / C(n_z,r_z)
                      × F(x_top)[n_x, r_x] × F(y_top)[n_z-n_x, r_z-r_x]
```

where the sum is over valid (n_x, r_x) pairs.

### Kernel Design

```python
@cuda.jit
def rule2_kernel(F_t_x, F_t_y, site_count, vector_len, output):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    total_work = site_count * vector_len
    if tid >= total_work:
        return
    
    site = tid // vector_len
    z_index = tid % vector_len
    
    n_zbot = d_index_to_n(z_index)
    r_zbot = d_index_to_r(z_index, n_zbot)
    
    evaluation = 0.0
    
    # Double loop over valid (n_x, r_x) combinations
    for n_xtop in range(n_zbot + 1):
        for r_xtop in range(r_zbot + 1):
            # Validity check
            if r_xtop <= n_xtop and (r_zbot - r_xtop) <= (n_zbot - n_xtop):
                n_ytop = n_zbot - n_xtop
                r_ytop = r_zbot - r_xtop
                
                const = d_comb(n_xtop, r_xtop) * d_comb(n_ytop, r_ytop) / d_comb(n_zbot, r_zbot)
                
                x_index = d_nr_to_index(n_xtop, r_xtop) if n_xtop > 0 else 0
                y_index = d_nr_to_index(n_ytop, r_ytop) if n_ytop > 0 else 0
                
                if n_xtop > 0 and n_ytop > 0:
                    term1 = F_t_x[site, x_index]
                    term2 = F_t_y[site, y_index]
                    evaluation += term1 * term2 * const
    
    output[site, z_index] = evaluation
```

### Grid/Block Configuration

```python
total_work = site_count * vector_len
threads = 256
blocks = min(ceil(total_work / 256), 65535)
```

**Same as Rule 1** - the output shape is identical.

### Computational Characteristics

**Nested loops**: O(n_z² × r_z) iterations per thread
- More work per thread than Rule 1
- Higher arithmetic intensity
- Good for hiding memory latency

**Thread divergence**: Some threads skip iterations due to validity checks
- Warp efficiency may vary with (n, r) values
- Not a major concern as the check is simple

---

## Rule 3 Kernel

### Mathematical Definition
Rule 3 handles reticulation nodes (one branch splits into two):

```
F(y_bot, z_bot)[n_y, n_z, r_y, r_z] = 
    F(x_top)[n_y+n_z, r_y+r_z] × C(n_y+n_z, n_y) × γ_y^n_y × γ_z^n_z
```

### Kernel Design

This is the most complex kernel because it has 4D output structure (n_y, n_z, r_y, r_z per site):

```python
@cuda.jit
def rule3_kernel(F_t_x, site_count, vector_len, mx, gamma_y, gamma_z, output_y, output_z):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    # 3D work: site × y_index × z_index
    total_work = site_count * vector_len * vector_len
    if tid >= total_work:
        return
    
    # Decode 3D index
    site = tid // (vector_len * vector_len)
    remainder = tid % (vector_len * vector_len)
    y_index = remainder // vector_len
    z_index = remainder % vector_len
    
    n_y = d_index_to_n(y_index)
    r_y = d_index_to_r(y_index, n_y)
    n_z = d_index_to_n(z_index)
    r_z = d_index_to_r(z_index, n_z)
    
    # Skip invalid combinations
    if n_y + n_z > mx or n_y + n_z < 1:
        return
    
    n_xtop = n_y + n_z
    r_xtop = r_y + r_z
    
    if r_xtop > n_xtop:
        return
    
    x_index = d_nr_to_index(n_xtop, r_xtop)
    if x_index >= vector_len:
        return
    
    top_value = F_t_x[site, x_index]
    
    # Compute γ^n powers
    gamma_y_pow = 1.0
    gamma_z_pow = 1.0
    for _ in range(n_y):
        gamma_y_pow *= gamma_y
    for _ in range(n_z):
        gamma_z_pow *= gamma_z
    
    evaluation = top_value * d_comb(n_xtop, n_y) * gamma_y_pow * gamma_z_pow
    
    # Atomic add for race condition safety
    cuda.atomic.add(output_y, (site, y_index), evaluation)
    cuda.atomic.add(output_z, (site, z_index), evaluation)
```

### Grid/Block Configuration

```python
total_work = site_count * vector_len * vector_len
threads = 256
blocks = min(ceil(total_work / 256), 65535)
```

**Why O(vector_len²) work?**
- Rule 3 produces outputs for *two* branches (y and z)
- Each (n_y, r_y, n_z, r_z) combination maps to an input state
- Must iterate all valid combinations

### Atomic Operations

```python
cuda.atomic.add(output_y, (site, y_index), evaluation)
cuda.atomic.add(output_z, (site, z_index), evaluation)
```

**Why atomics?**
- Multiple threads may write to the same output cell
- Different (n_y, n_z) pairs can produce the same (n_y, r_y) or (n_z, r_z)
- Atomic add ensures correct accumulation

**Performance impact**: Atomics are slower than regular writes, but:
- Modern GPUs have fast atomic hardware
- Contention is relatively low (spread across sites)
- Alternative (reduction) would be more complex

---

## Rule 4 Kernel

### Mathematical Definition
Rule 4 combines branches with common leaf descendants:

```
F(z_bot)[n_z, r_z] = Σ C(n_x,r_x) × C(n_z-n_x, r_z-r_x) / C(n_z,r_z)
                      × F(x_top, y_top)[n_x, n_z-n_x, r_x, r_z-r_x]
```

### Kernel Design

```python
@cuda.jit
def rule4_kernel(F_t, site_count, vector_len, output):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    total_work = site_count * vector_len
    if tid >= total_work:
        return
    
    site = tid // vector_len
    z_index = tid % vector_len
    
    n_zbot = d_index_to_n(z_index)
    r_zbot = d_index_to_r(z_index, n_zbot)
    
    evaluation = 0.0
    
    for n_xtop in range(1, n_zbot + 1):
        for r_xtop in range(r_zbot + 1):
            n_ytop = n_zbot - n_xtop
            r_ytop = r_zbot - r_xtop
            
            if r_xtop <= n_xtop and r_ytop <= n_ytop and n_ytop >= 0:
                const = d_comb(n_xtop, r_xtop) * d_comb(n_ytop, r_ytop) / d_comb(n_zbot, r_zbot)
                
                x_index = d_nr_to_index(n_xtop, r_xtop)
                y_index = d_nr_to_index(n_ytop, r_ytop) if n_ytop > 0 else 0
                
                if x_index < vector_len and y_index < vector_len:
                    ft_val = F_t[site, x_index, y_index]
                    evaluation += ft_val * const
    
    output[site, z_index] = evaluation
```

### Grid/Block Configuration
Same as Rule 1 and Rule 2.

### Input Array Shape
Rule 4 takes 3D input: `F_t[site, x_index, y_index]`
- Represents the joint distribution over x_top and y_top states
- Each thread reads from multiple (x, y) pairs

---

## Memory Management

### CUDAEvaluator Class

The `CUDAEvaluator` class handles all GPU memory operations:

```python
class CUDAEvaluator:
    def Rule0(self, reds, site_count, vector_len, samples):
        # 1. Transfer input to GPU
        d_reds = cuda.to_device(reds.astype(np.float64))
        
        # 2. Allocate output on GPU
        d_output = cuda.device_array((site_count, vector_len), dtype=np.float64)
        
        # 3. Launch kernel
        rule0_kernel[blocks, threads](d_reds, ..., d_output)
        
        # 4. Copy result back to CPU
        output = d_output.copy_to_host()
        
        # 5. Convert to dict format for compatibility
        return self._array_to_dict(output, ...)
```

### Memory Transfer Overhead

| Operation | Typical Size | Transfer Time (PCIe 4.0) |
|-----------|-------------|--------------------------|
| `reds` | 10,000 sites × 8 bytes | ~0.01 ms |
| `Qt` matrix | 1000×1000 × 8 bytes | ~0.1 ms |
| `F_b` array | 10,000 × 1000 × 8 bytes | ~1 ms |

**Mitigation strategies:**
1. Batch multiple sites together
2. Keep data on GPU between rule applications
3. Use pinned memory for async transfers

### Dict ↔ Array Conversion

The original code uses nested dictionaries for flexibility. The CUDA version converts to/from contiguous arrays:

```python
def _dict_to_array(self, F, site_count, vector_len):
    arr = np.zeros((site_count, vector_len), dtype=np.float64)
    for site in range(site_count):
        for (nx, rx), prob in F[site].items():
            idx = nr_to_index(nx[-1], rx[-1])
            arr[site, idx] = prob
    return arr

def _array_to_dict(self, arr, site_count, vector_len):
    F = {}
    for site in range(site_count):
        F[site] = {}
        for idx in range(vector_len):
            n, r = index_to_nr(idx)
            if arr[site, idx] != 0.0:  # Sparse storage
                F[site][(tuple([n]), tuple([r]))] = arr[site, idx]
    return F
```

---

## Performance Considerations

### Occupancy Analysis

For RTX 5070ti with 256 threads/block:

| Resource | Per Thread | Per Block | SM Limit | Occupancy |
|----------|-----------|-----------|----------|-----------|
| Registers | ~32 | ~8,192 | 65,536 | ~12.5% per block |
| Shared Mem | 0 | 0 | 100 KB | N/A |
| Threads | 1 | 256 | 2,048 | ~12.5% per block |

Multiple blocks can run concurrently on each SM, achieving ~50-100% occupancy.

### Bottleneck Analysis

| Rule | Primary Bottleneck |   Arithmetic Intensity   |
|------|--------------------|--------------------------|
|   0  | Memory bandwidth   | Very low (simple writes) |
|   1  | Compute            | High (summation loop)    |
|   2  | Compute            | High (loops + binomials) |
|   3  | Memory + Atomics   | Medium                   |
|   4  | Compute            | High (similar to Rule 2) |

### Future Optimizations

1. **Shared Memory**: Cache Qt matrix tiles for Rule 1
2. **Warp-level Primitives**: Use `__shfl` for reductions
3. **Stream Parallelism**: Overlap CPU↔GPU transfers
4. **Persistent Kernels**: Keep data on GPU across multiple rule applications
5. **Mixed Precision**: Use FP32 for intermediate, FP64 for accumulation

### Expected Speedup

Based on similar algorithms, expect:
- **Rule 0**: 10-50× (memory bound, limited by PCIe)
- **Rule 1**: 50-200× (compute bound, good GPU utilization)
- **Rule 2**: 50-200× (similar to Rule 1)
- **Rule 3**: 20-100× (atomic contention limits scaling)
- **Rule 4**: 50-200× (similar to Rule 1)

**Overall MCMC speedup**: 10-100× depending on dataset size and network complexity.

---

## Debugging Tips

### Check CUDA Availability
```python
from MCMC_BiMarkers_CUDA import get_cuda_device_info
get_cuda_device_info()
```

### Force CPU Mode
```python
from MCMC_BiMarkers_CUDA import SNP_LIKELIHOOD
result = SNP_LIKELIHOOD("data.nex", use_gpu=False)
```

### Benchmark
```python
from MCMC_BiMarkers_CUDA import benchmark_cuda_vs_cpu
results = benchmark_cuda_vs_cpu("data.nex", iterations=10)
```

### CUDA Error Debugging
```python
import numba.cuda
numba.cuda.detect()  # List all CUDA devices

# Enable CUDA debug mode
import os
os.environ['NUMBA_CUDA_DEBUGINFO'] = '1'
```

---

## References

1. Rabier CE, Berry V, Stoltz M, et al. (2021) On the inference of complex phylogenetic networks by Markov Chain Monte-Carlo. PLOS Computational Biology 17(9): e1008380.

2. Bryant D, Bouckaert R, Felsenstein J, et al. (2012) Inferring Species Trees Directly from Biallelic Genetic Markers. Molecular Biology and Evolution 29(8): 1917–1932.

3. NVIDIA CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

4. Numba CUDA Documentation: https://numba.readthedocs.io/en/stable/cuda/

