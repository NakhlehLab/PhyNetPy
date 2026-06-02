#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author : Mark Kessler
Last Edit : 5/12/26
First Included in Version : 1.0.0

BiMarkers -- biallelic SNP (single-nucleotide polymorphism) likelihood
under the multispecies network coalescent for phylogenetic networks.

This module implements the SNAPP-style two-state continuous-time Markov
chain likelihood (Bryant et al. 2012) generalised to reticulate
topologies.  It provides:

* :class:`SNP_LIKELIHOOD` and :class:`SNP_LIKELIHOOD_DATA` -- the public
  scoring functions for biallelic site patterns on a species network.
* :class:`MCMC_BIMARKERS` -- a Metropolis-Hastings search driver wired to
  the SNP likelihood, mirroring the API of the other ``MCMC_*`` entry
  points.
* A NumPy reference implementation suitable for small/medium networks.
  An optional CUDA-accelerated implementation lives in
  ``MCMC_BiMarkers_CUDA`` and is auto-wired by ``phynetpy/__init__.py``
  when ``cupy``/``numba`` are importable.

Docs   - [ ]
Tests  - [ ]
Design - [ ]
"""

from __future__ import annotations

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##
##  See "LICENSE.txt" for terms and conditions of usage.
##
##  If you use this work or any portion thereof in published work,
##  please cite it as:
##
##     Mark Kessler, Luay Nakhleh. 2025.
##
##############################################################################

from math import sqrt, comb, pow
import time
from typing import Callable
import numpy as np
from scipy.linalg import expm
from dataclasses import dataclass

# CUDA imports - with graceful fallback
# Numba CUDA requires CUDA Toolkit 12.x (tested with CUDA 12.2)
# Both Numba and CuPy are used for GPU acceleration

CUPY_AVAILABLE = False
CUPY_RUNTIME_OK = False
NUMBA_CUDA_AVAILABLE = False

try:
    import cupy as cp
    # Check if CuPy can actually access a GPU
    CUPY_AVAILABLE = cp.cuda.is_available()
    if CUPY_AVAILABLE:
        # Verify CuPy can actually run operations (CUDA version compatibility check)
        # Use cp.random which requires kernel compilation - this catches CUDA DLL mismatches
        try:
            _test = cp.random.rand(10, dtype=cp.float64)
            _ = cp.sum(_test)
            del _test
            CUPY_RUNTIME_OK = True
        except (RuntimeError, Exception) as e:
            # CuPy may detect GPU but fail at runtime due to CUDA version mismatch
            # Common error: missing nvrtc64_XXX_0.dll for wrong CUDA toolkit version
            CUPY_RUNTIME_OK = False
except ImportError:
    cp = None

try:
    from numba import cuda, float64, int32, int64
    # Check if numba CUDA is actually available (requires compatible toolkit)
    NUMBA_CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    # Create mock decorators for when numba CUDA is not available
    class MockCuda:
        @staticmethod
        def is_available():
            return False
        @staticmethod
        def jit(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        @staticmethod
        def to_device(arr):
            return arr
        @staticmethod
        def device_array(*args, **kwargs):
            return np.zeros(args[0])
        @staticmethod
        def select_device(n):
            pass
        @staticmethod
        def grid(n):
            return 0
        class atomic:
            @staticmethod
            def add(*args):
                pass
    cuda = MockCuda()
    float64 = float
    int32 = int
    int64 = int

# Combined availability: CuPy for array ops, numba for custom kernels
# Use CUPY_RUNTIME_OK for actual operations, CUPY_AVAILABLE just means import worked
CUDA_IMPORTS_AVAILABLE = CUPY_RUNTIME_OK or NUMBA_CUDA_AVAILABLE


# ═══════════════════════════════════════════════════════════════════════════════
# GPU HARDWARE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GPUSpecs:
    """Hardware specs for the user's GPU (detected at import time)."""
    available: bool = False
    name: str = "No GPU"
    vram_bytes: int = 0
    vram_mb: int = 0
    compute_capability: tuple[int, int] = (0, 0)
    
    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / (1024 ** 3)
    
    def __repr__(self) -> str:
        if not self.available:
            return "GPUSpecs(available=False)"
        return (f"GPUSpecs({self.name}, "
                f"VRAM={self.vram_gb:.1f} GB, "
                f"CC={self.compute_capability[0]}.{self.compute_capability[1]})")


def _detect_gpu() -> GPUSpecs:
    """
    Detect the GPU hardware specs for the current machine.
    
    Uses CuPy to query device properties. Falls back gracefully
    if no GPU or no CuPy is available.
    
    Returns:
        GPUSpecs: Detected GPU hardware specifications.
    """
    if not CUPY_RUNTIME_OK:
        return GPUSpecs()
    
    try:
        device = cp.cuda.Device(0)
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        total_mem = device.mem_info[1]  # (free, total) → total
        
        return GPUSpecs(
            available=True,
            name=props['name'].decode() if isinstance(props['name'], bytes) else str(props['name']),
            vram_bytes=total_mem,
            vram_mb=total_mem // (1024 * 1024),
            compute_capability=(props['major'], props['minor']),
        )
    except Exception:
        return GPUSpecs()


# Detect GPU specs once at module import
GPU_SPECS = _detect_gpu()

# Relative imports
from .MSA import MSA, DataSequence
from .BirthDeath import CBDP
from .IO import read_nexus
from .Alphabet import Alphabet
from .Matrix import Matrix
# from .ModelGraph import (
#     Model, ModelNode, CalculationNode, Parameter, Accumulator, ExtantSpecies
# )
from .ModelFactory import *
from .Network import *
from .MetropolisHastings import MetropolisHastings, ProposalKernel
from .Visitor import *
from .Strategy import *
from .Executor import *
from .Traversal import *
from .ModelGraph import *
from .GraphUtils import level as network_level


"""
SOURCES:

(1): 
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005932

CUDA Implementation Notes:
- Rules 0-4 are implemented as CUDA kernels for parallel execution
- Site-level parallelism is exploited (each site computed independently)
- Combinatorial loops are parallelized across GPU threads
- Matrix operations use CuPy for GPU-accelerated linear algebra
"""

# Global flag to toggle CUDA acceleration
global use_cuda
use_cuda = True

# CUDA device configuration
THREADS_PER_BLOCK = 256
MAX_BLOCKS = 65535

def n_to_index(n : int) -> int:
    """
    Computes the starting index in computing a linear index for an (n,r) pair.
    Returns the index, if r is 0.
    
    i.e n=1 returns 0, since (1,0) is index 0
    i.e n=3 returns 5 since (3,0) is preceded by 
        (1,0), (1,1), (2,0), (2,1), and (2,2)

    Args:
        n (int): an n value (number of lineages) from an (n,r) pair
    Returns:
        int: starting index for that block of n values
    """
    return int(.5 * (n - 1) * (n + 2))

def index_to_nr(index : int) -> list[int]:
    """
    Takes an index from the linear vector and turns it into an (n,r) pair
    
    i.e 7 -> [3,2]

    Args:
        index (int): the index

    Returns:
        list[int]: a 2-tuple (n,r)
    """
    a = 1
    b = 1
    c = -2 - 2 * index
    d = (b ** 2) - (4 * a * c)
    sol = (-b + sqrt(d)) / (2 * a)
    n = int(sol)
    r = index - n_to_index(n)

    return [n, r]

def nr_to_index(n : int, r : int) -> int:
    """
    Takes an (n,r) pair and maps it to a 1d vector index

    (1,0) -> 0
    (1,1) -> 1
    (2,0) -> 2
    ...
    
    Args:
        n (int): the number of lineages
        r (int): the number of red lineages (<= n)

    Returns:
        int: the index into the linear vector, that represents by (n, r)
    """
    
    return n_to_index(n) + r

@dataclass
class NodeVPI:
    """Partial-likelihood vector bundle carried by a model node.

    Attributes:
        tensor: Partial-likelihood array of shape ``(S, d1, d2, ...)``,
            where *S* is the number of sites and each subsequent axis
            corresponds to an interface.
        interfaces: Labels for each tensor axis beyond the site axis
            (length equals ``tensor.ndim - 1``).
        max_lineages: Maximum lineage count per interface.
        log_scale: Per-site log rescaling factors, shape ``(S,)``.
    """
    tensor: np.ndarray
    interfaces: list[str]
    max_lineages: list[int]
    log_scale: np.ndarray = None

def state_dim(m: int) -> int:
    """Return the state-space dimension for *m* lineages.

    The dimension equals the number of valid ``(n, r)`` pairs where
    ``1 <= n <= m`` and ``0 <= r <= n``.
    """
    return nr_to_index(m, m) + 1

def build_split_tensor(m: int, gamma: float) -> np.ndarray:
    """
    Build the split coefficient tensor S for a reticulation node
    with max lineages m and inheritance probability gamma.

    S[i, j, k] = C(n, n_b) * C(r, r_b) * gamma^{n_b} * (1-gamma)^{n_d}

    where i ↔ (n, r), j ↔ (n_b, r_b), k ↔ (n_d, r_d)
    and n = n_b + n_d, r = r_b + r_d (zero otherwise).

    Shape: [dim(m), dim(m), dim(m)]
    """
    d = state_dim(m)
    S = np.zeros((d, d, d))

    for n in range(1, m + 1):
        for r in range(n + 1):
            i = nr_to_index(n, r)
            for nb in range(1, n + 1):
                nd = n - nb
                if nd < 1:
                    continue
                for rb in range(min(r, nb) + 1):
                    rd = r - rb
                    if rd < 0 or rd > nd:
                        continue

                    j = nr_to_index(nb, rb)
                    k = nr_to_index(nd, rd)

                    S[i, j, k] = (comb(n, nb) * comb(r, rb)
                                  * (gamma ** nb) * ((1 - gamma) ** nd))

    return S

def build_merge_tensor(mx: int, my: int) -> np.ndarray:
    """
    Build the merge coefficient tensor M for combining two interfaces
    with max lineages mx and my.

    M[i, j, k] = C(rz, rx) * C(nz-rz, nx-rx) / C(nz, nx)
    
    where i ↔ (nx, rx), j ↔ (ny, ry), k ↔ (nz, rz)
    and nz = nx + ny, rz = rx + ry (zero otherwise).

    Used by both Rule 2 (disjoint merge) and Rule 4 (overlapping merge).

    Shape: [dim(mx), dim(my), dim(mx + my)]
    """
    mz = mx + my
    M = np.zeros((state_dim(mx), state_dim(my), state_dim(mz)))

    for nx in range(1, mx + 1):
        for rx in range(nx + 1):
            i = nr_to_index(nx, rx)
            for ny in range(1, my + 1):
                for ry in range(ny + 1):
                    j = nr_to_index(ny, ry)
                    
                    nz = nx + ny
                    rz = rx + ry
                    k = nr_to_index(nz, rz)

                    M[i, j, k] = (comb(rz, rx) * comb(nz - rz, nx - rx) 
                                  / comb(nz, nx))

    return M


# ═══════════════════════════════════════════════════════════════════════════════
# SPARSE COEFFICIENT REPRESENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# The dense merge/split 3D tensors are 95–99.7% zeros. Only entries where
# nz = nx + ny, rz = rx + ry (merge) or n = nb + nd, r = rb + rd (split)
# are non-zero. Storing as COO (coordinate) arrays and using vectorized
# scatter-add replaces the O(dim_x · dim_y · dim_z) einsum with
# O(dim_x · dim_y) work — a speedup factor of dim_z (hundreds to thousands).
#

@dataclass
class SparseMerge:
    """
    Sparse COO representation of the merge tensor M[i, j, k].
    
    For each non-zero entry: stores (i_arr[n], j_arr[n], k_arr[n], coeff_arr[n]).
    There is exactly one k for each (i, j) pair (since nz = nx + ny is unique).
    
    nnz = state_dim(mx) * state_dim(my)   (one per valid (i,j) pair)
    """
    i_arr: np.ndarray       # int32 indices into F_x's last axis
    j_arr: np.ndarray       # int32 indices into F_y's last axis
    k_arr: np.ndarray       # int32 indices into result's last axis
    coeff_arr: np.ndarray   # float64 coefficients
    dim_x: int              # state_dim(mx)
    dim_y: int              # state_dim(my)
    dim_z: int              # state_dim(mx + my)
    mx: int
    my: int


@dataclass
class SparseSplit:
    """
    Sparse COO representation of the split tensor S[i, j, k].
    
    For each non-zero entry: stores (i_arr[n], j_arr[n], k_arr[n], coeff_arr[n]).
    Multiple (j, k) pairs map to each i (since n = nb + nd has many valid splits).
    """
    i_arr: np.ndarray
    j_arr: np.ndarray
    k_arr: np.ndarray
    coeff_arr: np.ndarray
    dim: int               # state_dim(m)
    m: int


def build_sparse_merge(mx: int, my: int) -> SparseMerge:
    """
    Build sparse COO merge tensor. 
    
    ~50× less memory and dim_z × faster contraction than the dense version.
    
    Args:
        mx: Max lineages for the left interface.
        my: Max lineages for the right interface.
    Returns:
        SparseMerge with coordinate arrays.
    """
    i_list, j_list, k_list, c_list = [], [], [], []
    
    for nx in range(1, mx + 1):
        for rx in range(nx + 1):
            i = nr_to_index(nx, rx)
            for ny in range(1, my + 1):
                for ry in range(ny + 1):
                    j = nr_to_index(ny, ry)
                    nz = nx + ny
                    rz = rx + ry
                    k = nr_to_index(nz, rz)
                    coeff = comb(rz, rx) * comb(nz - rz, nx - rx) / comb(nz, nx)
                    
                    i_list.append(i)
                    j_list.append(j)
                    k_list.append(k)
                    c_list.append(coeff)
    
    return SparseMerge(
        i_arr=np.array(i_list, dtype=np.int32),
        j_arr=np.array(j_list, dtype=np.int32),
        k_arr=np.array(k_list, dtype=np.int32),
        coeff_arr=np.array(c_list, dtype=np.float64),
        dim_x=state_dim(mx),
        dim_y=state_dim(my),
        dim_z=state_dim(mx + my),
        mx=mx,
        my=my,
    )


def build_sparse_split(m: int, gamma: float) -> SparseSplit:
    """
    Build sparse COO split tensor.
    
    Args:
        m: Max lineages at the node being split.
        gamma: Inheritance probability for the left parent branch.
    Returns:
        SparseSplit with coordinate arrays.
    """
    i_list, j_list, k_list, c_list = [], [], [], []
    
    for n in range(1, m + 1):
        for r in range(n + 1):
            i = nr_to_index(n, r)
            for nb in range(1, n + 1):
                nd = n - nb
                if nd < 1:
                    continue
                for rb in range(min(r, nb) + 1):
                    rd = r - rb
                    if rd < 0 or rd > nd:
                        continue
                    
                    j = nr_to_index(nb, rb)
                    k = nr_to_index(nd, rd)
                    coeff = (comb(n, nb) * comb(r, rb)
                             * (gamma ** nb) * ((1 - gamma) ** nd))
                    
                    i_list.append(i)
                    j_list.append(j)
                    k_list.append(k)
                    c_list.append(coeff)
    
    return SparseSplit(
        i_arr=np.array(i_list, dtype=np.int32),
        j_arr=np.array(j_list, dtype=np.int32),
        k_arr=np.array(k_list, dtype=np.int32),
        coeff_arr=np.array(c_list, dtype=np.float64),
        dim=state_dim(m),
        m=m,
    )


# Module-level caches to avoid rebuilding sparse tensors
_SPARSE_MERGE_CACHE: dict[tuple[int, int], SparseMerge] = {}
_SPARSE_SPLIT_CACHE: dict[tuple[int, float], SparseSplit] = {}


def get_sparse_merge(mx: int, my: int) -> SparseMerge:
    """Get or build a cached sparse merge tensor."""
    key = (mx, my)
    if key not in _SPARSE_MERGE_CACHE:
        _SPARSE_MERGE_CACHE[key] = build_sparse_merge(mx, my)
    return _SPARSE_MERGE_CACHE[key]


def get_sparse_split(m: int, gamma: float) -> SparseSplit:
    """Get or build a cached sparse split tensor."""
    key = (m, gamma)
    if key not in _SPARSE_SPLIT_CACHE:
        _SPARSE_SPLIT_CACHE[key] = build_sparse_split(m, gamma)
    return _SPARSE_SPLIT_CACHE[key]


# ── GPU sparse caches ────────────────────────────────────────────────────
# CuPy arrays cannot be indexed with NumPy arrays, so we keep separate
# caches where the coordinate/coefficient arrays live on GPU memory.

_SPARSE_MERGE_CACHE_GPU: dict[tuple[int, int], SparseMerge] = {}
_SPARSE_SPLIT_CACHE_GPU: dict[tuple[int, float], SparseSplit] = {}


def get_sparse_merge_gpu(mx: int, my: int) -> SparseMerge:
    """Get or build a GPU-resident cached sparse merge tensor."""
    key = (mx, my)
    if key not in _SPARSE_MERGE_CACHE_GPU:
        sm = get_sparse_merge(mx, my)  # build on CPU first
        _SPARSE_MERGE_CACHE_GPU[key] = SparseMerge(
            i_arr=cp.asarray(sm.i_arr),
            j_arr=cp.asarray(sm.j_arr),
            k_arr=cp.asarray(sm.k_arr),
            coeff_arr=cp.asarray(sm.coeff_arr),
            dim_x=sm.dim_x, dim_y=sm.dim_y, dim_z=sm.dim_z,
            mx=sm.mx, my=sm.my,
        )
    return _SPARSE_MERGE_CACHE_GPU[key]


def get_sparse_split_gpu(m: int, gamma: float) -> SparseSplit:
    """Get or build a GPU-resident cached sparse split tensor."""
    key = (m, gamma)
    if key not in _SPARSE_SPLIT_CACHE_GPU:
        ss = get_sparse_split(m, gamma)  # build on CPU first
        _SPARSE_SPLIT_CACHE_GPU[key] = SparseSplit(
            i_arr=cp.asarray(ss.i_arr),
            j_arr=cp.asarray(ss.j_arr),
            k_arr=cp.asarray(ss.k_arr),
            coeff_arr=cp.asarray(ss.coeff_arr),
            dim=ss.dim, m=ss.m,
        )
    return _SPARSE_SPLIT_CACHE_GPU[key]

def _disjoint_subnets(n: InternalNode) -> bool:
    
    """
    Determine if the left and right subnets of an internal node are disjoint.
    """
    lr = n.get_model_children()
    assert len(lr) == 2, "Internal node must have exactly two children"
    subnets = (set(), set())
    
    for i, child in enumerate(lr):
        q = deque([child])
        while q:
            cur = q.popleft()
            subnets[i].add(cur)
            children = cur.get_model_children()
            if children:  # Guard against None
                q.extend(children)
    
    return subnets[0].isdisjoint(subnets[1])

def deduplicate_vpis(vpis: list[NodeVPI]) -> list[NodeVPI]:
    """
    Remove duplicates by identity (``is``), preserving order.

    Uses an ``id()``-based set for O(n) performance instead of O(n^2)
    linear scans.

    Args:
        vpis (list[NodeVPI]): VPI objects, possibly with duplicate references.

    Returns:
        list[NodeVPI]: De-duplicated list in original order.
    """
    seen_ids: set[int] = set()
    unique: list[NodeVPI] = []
    for vpi in vpis:
        vid = id(vpi)
        if vid not in seen_ids:
            seen_ids.add(vid)
            unique.append(vpi)
    return unique


def _compute_max_lineages(model: Model, samples: dict[str, int]) -> int:
    """
    Pre-compute the maximum possible lineage count at any node in the model,
    accounting for lineage duplication at reticulation nodes.
    
    In a network with reticulations, both parent branches of a reticulation 
    node inherit ALL the child's lineages. When these branches merge later 
    at a common ancestor, the lineage count can exceed sum(samples).
    
    The Q matrix must be large enough to accommodate this maximum.
    
    Args:
        model: The built SNP model.
        samples: Dict mapping leaf names to their sample counts.
    Returns:
        int: The maximum number of lineages at any merge point.
    """
    max_lin_at : dict = {}
    
    for node in Traversal(model.get_root(), TraversalOrder.POST_ORDER):
        ntype = node.get_node_type()
        if ntype == "leaf":
            max_lin_at[node] = samples.get(node.get_name(), 1)
        elif ntype == "reticulation":
            children = node.get_model_children()
            # Both parent branches inherit ALL the child's lineages
            max_lin_at[node] = max_lin_at[children[0]]
        elif ntype in ("internal", "root"):
            children = node.get_model_children()
            max_lin_at[node] = sum(max_lin_at[c] for c in children)
        elif ntype == "root_aggregator":
            children = node.get_model_children()
            max_lin_at[node] = max_lin_at[children[0]]
    
    return max(max_lin_at.values())


def _compute_network_level(model: Model) -> int:
    """
    Compute the level of the phylogenetic network from the model graph.
    
    The level is the maximum number of reticulation nodes in any single
    biconnected component (blob) of the underlying undirected graph.
    This directly determines the maximum tensor dimensionality:
      level-0 (tree): 2D tensors  (sites × d)
      level-1:        3D tensors  (sites × d × d)
      level-2:        4D tensors  (sites × d × d × d)
      ...
    
    Args:
        model: The built SNP model.
    Returns:
        int: The network level (max reticulations per blob).
    """
    if model.network is not None:
        return network_level(model.network)
    return len(model.nodetypes.get("reticulation", []))


def _estimate_peak_vpi_memory(model: Model, samples: dict[str, int], 
                               n_sites: int) -> tuple[int, list[int]]:
    """
    Estimate the peak VPI tensor memory (in bytes) before running the algorithm.
    
    Simulates the VPI flow through the model graph to track the maximum tensor
    shape that will be created during computation. This allows us to reject 
    infeasible computations before allocating any large arrays.
    
    The VPI tensor at each node has shape:
      (n_sites, state_dim(m1), state_dim(m2), ..., state_dim(mk))
    where m1..mk are the max lineages for each open interface.
    
    Args:
        model: The built SNP model.
        samples: Dict mapping leaf names to their sample counts.
        n_sites: Number of alignment sites.
    Returns:
        Tuple of (peak_bytes, peak_shape) where peak_shape is the tensor
        shape that would cause the peak memory allocation.
    """
    from collections import deque
    
    # Track open interface dimensions per VPI, using the same logic
    # as the visitor: track (interfaces, max_lineages) per node
    vpi_dims : dict = {}   # ModelNode -> list[int] (max_lineages per interface)
    vpi_id   : dict = {}   # ModelNode -> id tracking which VPI object this belongs to
    
    peak_bytes = 0
    peak_shape = [n_sites]
    
    id_counter = 0
    
    for node in Traversal(model.get_root(), TraversalOrder.POST_ORDER):
        ntype = node.get_node_type()
        
        if ntype == "leaf":
            s = samples.get(node.get_name(), 1)
            vpi_dims[node] = [s]
            vpi_id[node] = id_counter
            id_counter += 1
            
        elif ntype == "reticulation":
            children = node.get_model_children()
            child = children[0]
            child_dims = vpi_dims[child]
            # Split: child's last dimension becomes two interfaces (same size)
            m = child_dims[-1]
            vpi_dims[node] = child_dims[:-1] + [m, m]
            vpi_id[node] = vpi_id[child]
            
        elif ntype in ("internal", "root"):
            children = node.get_model_children()
            child_dims_list = [vpi_dims[c] for c in children]
            child_ids = [vpi_id[c] for c in children]
            
            if child_ids[0] == child_ids[1]:
                # Rule 4 (same VPI): merge last two dims into one
                dims = child_dims_list[0]
                merged = dims[-2] + dims[-1]
                new_dims = dims[:-2] + [merged]
            else:
                # Rule 2 (disjoint): outer product of the two VPIs
                # Remove last dim from each (being merged), combine extras
                d0 = child_dims_list[0]
                d1 = child_dims_list[1]
                merged = d0[-1] + d1[-1]
                new_dims = d0[:-1] + d1[:-1] + [merged]
            
            vpi_dims[node] = new_dims
            vpi_id[node] = vpi_id[children[0]]
            
        elif ntype == "root_aggregator":
            children = node.get_model_children()
            vpi_dims[node] = vpi_dims[children[0]]
            vpi_id[node] = vpi_id[children[0]]
        
        # Compute memory for this node's VPI tensor
        shape = [n_sites] + [state_dim(m) for m in vpi_dims.get(node, [1])]
        tensor_bytes = 8  # float64
        for s in shape:
            tensor_bytes *= s
        
        if tensor_bytes > peak_bytes:
            peak_bytes = tensor_bytes
            peak_shape = shape
    
    return peak_bytes, peak_shape


class SNPResourceError(RuntimeError):
    """
    Raised when the SNP likelihood computation would exceed available 
    hardware resources (GPU VRAM or system RAM).
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO CPU/GPU ROUTING THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

# Taxa count thresholds above which GPU is required for reasonable performance.
# These are based on the O(S × D^{k+1}) complexity of the VPI algorithm,
# where k = network level and D = state_dim(max_lineages).
GPU_THRESHOLD = {
    0: float('inf'),  # Pure trees: CPU is always fine
    1: 20,            # Level-1: GPU after 20 taxa
    2: 12,            # Level-2: GPU after 12 taxa
    3: 8,             # Level-3: GPU after 8 taxa (if ever supported)
}

# Safety margin: require this fraction of VRAM to be available
# (i.e., peak tensor < 80% of VRAM to leave room for merge/split tensors, 
# Q matrix, etc.)
GPU_VRAM_SAFETY_FACTOR = 0.80


def _compute_batch_size(model: Model, samples: dict[str, int],
                        n_sites: int, use_gpu: bool) -> int:
    """
    Determine the optimal site batch size that fits in available memory.
    
    Sites are independent in the SNP likelihood, so we can compute
    the likelihood in chunks and sum. This function finds the largest
    batch size where the peak VPI tensor fits comfortably in memory.
    
    Args:
        model: The built SNP model.
        samples: Dict mapping leaf names to their sample counts.
        n_sites: Total number of alignment sites.
        use_gpu: Whether GPU will be used.
    Returns:
        Optimal batch size (≤ n_sites). If the full alignment fits, 
        returns n_sites (no batching needed).
    
    Raises:
        SNPResourceError: If even a single site exceeds available memory.
    """
    # Estimate memory for 1 site to get per-site cost
    _, peak_shape_1 = _estimate_peak_vpi_memory(model, samples, 1)
    per_site_elements = 1
    for s in peak_shape_1[1:]:      # skip the sites dimension (=1)
        per_site_elements *= s
    per_site_bytes = per_site_elements * 8  # float64
    
    # Determine available memory
    if use_gpu and GPU_SPECS.available:
        available = int(GPU_SPECS.vram_bytes * GPU_VRAM_SAFETY_FACTOR)
    else:
        try:
            import psutil
            available = int(psutil.virtual_memory().available * 0.5)
        except ImportError:
            available = 8 * (1024 ** 3)  # conservative 8 GB
    
    # We need memory for at least 2 tensors at once (old + new during a rule)
    # plus the Q/P matrices, so use a factor of 3x
    memory_per_site = per_site_bytes * 3
    
    if memory_per_site == 0:
        return n_sites
    
    max_batch = max(1, available // memory_per_site)
    
    if max_batch < 1:
        level = _compute_network_level(model)
        n_taxa = len(model.nodetypes.get("leaf", []))
        max_n = _compute_max_lineages(model, samples)
        shape_str = " × ".join(str(s) for s in peak_shape_1)
        raise SNPResourceError(
            f"Even a single site exceeds available memory.\n"
            f"  Network: {n_taxa} taxa, level-{level}, max_lineages={max_n}\n"
            f"  Per-site VPI tensor: ({shape_str}) = "
            f"{per_site_bytes / (1024**2):.1f} MB\n"
            f"  This network topology is too complex for available hardware."
        )
    
    return min(max_batch, n_sites)


def _check_feasibility(model: Model, samples: dict[str, int],
                        n_sites: int, use_gpu: bool) -> None:
    """
    Pre-flight check: verify that even a single-site VPI tensor fits in memory.
    
    With site batching, we no longer need the full n_sites to fit at once.
    This check only ensures that the per-site overhead is feasible. The 
    actual batch size is determined by _compute_batch_size().
    
    Args:
        model: The built SNP model.
        samples: Dict mapping leaf names to their sample counts.
        n_sites: Number of alignment sites (used only for diagnostics).
        use_gpu: Whether GPU will be used for this computation.
    
    Raises:
        SNPResourceError: If even a single site exceeds available resources.
    """
    # Check with 1 site — if that doesn't fit, nothing will
    peak_bytes_1, peak_shape_1 = _estimate_peak_vpi_memory(model, samples, 1)
    level = _compute_network_level(model)
    n_taxa = len(model.nodetypes.get("leaf", []))
    max_n = _compute_max_lineages(model, samples)
    
    shape_str = " × ".join(str(s) for s in peak_shape_1)
    per_site_mb = peak_bytes_1 / (1024 * 1024)
    
    if use_gpu:
        gpu = GPU_SPECS
        if not gpu.available:
            raise SNPResourceError(
                f"GPU computation required for {n_taxa} taxa at level-{level}, "
                f"but no GPU was detected. Install CuPy with a compatible "
                f"CUDA toolkit, or reduce the problem size."
            )
        
        usable_vram = gpu.vram_bytes * GPU_VRAM_SAFETY_FACTOR
        # Need at least ~3x per-site for working memory
        if peak_bytes_1 * 3 > usable_vram:
            raise SNPResourceError(
                f"Even a single site exceeds GPU memory.\n"
                f"  Network: {n_taxa} taxa, level-{level}, "
                f"max_lineages={max_n}\n"
                f"  Per-site VPI tensor: ({shape_str}) = "
                f"{per_site_mb:.1f} MB\n"
                f"  GPU: {gpu.name} with {gpu.vram_gb:.1f} GB VRAM\n"
                f"  This network topology is too complex for this GPU."
            )
    else:
        try:
            import psutil
            available_ram = psutil.virtual_memory().available
        except ImportError:
            available_ram = 16 * (1024 ** 3)
        
        # Need at least ~3x per-site for working memory
        if peak_bytes_1 * 3 > available_ram * 0.5:
            ram_gb = available_ram / (1024 ** 3)
            raise SNPResourceError(
                f"Even a single site exceeds available system RAM.\n"
                f"  Network: {n_taxa} taxa, level-{level}, "
                f"max_lineages={max_n}\n"
                f"  Per-site VPI tensor: ({shape_str}) = "
                f"{per_site_mb:.1f} MB\n"
                f"  Available RAM: ~{ram_gb:.1f} GB\n"
                f"  This network topology is too complex for CPU computation."
            )


#####################
# Method Signatures #
#####################

def MCMC_BIMARKERS(filename: str, 
                   u : float = .5 ,
                   v : float = .5, 
                   coal : float = 1) -> dict[Network, float]:
    
    """
    Given a set of taxa with SNP data, perform a Markov Chain Monte Carlo
    chain to infer the most likely phylogenetic network that describes the
    taxa and data.

    Args:
        filename (str): string path destination of a nexus file that contains 
                        SNP data
        u (float, optional): Parameter for the probability of an
                             allele changing from red to green. Defaults to .5.
        v (float, optional): Parameter for the probability of an
                             allele changing from green to red. Defaults to .5.
        coal (float, optional): Parameter for the rate of coalescence. 
                                Defaults to 1.
        
    Returns:
        dict[Network, float]: The log likelihood (a negative number) of the most 
                              probable network, along with the network itself 
                              that achieved that score.
    """

    # Parse the data file into a sequence alignment
    aln = MSA(filename)
    
    # Generate starting network and place into model component
    start_net = CBDP(1, .5, aln.num_groups()).generate_network()
    
    snp_model = build_model(filename, 
                            start_net,
                            u, 
                            v, 
                            coal)
    
    mh = MetropolisHastings(ProposalKernel(),
                            data = Matrix(aln, Alphabet("SNP")), 
                            num_iter = 600,
                            model = snp_model) 
     
    result_state = mh.run()
    result_model = result_state.current_model

    return {result_model.network : result_model.likelihood()}

def SNP_LIKELIHOOD(filename : str,
                   u : float,
                   v : float, 
                   coal : float,
                   samples : dict[str, int],
                   max_workers: int = 8,
                   sequential: bool = True,
                   executor: Executor = None) -> float:
    """
    Given a set of taxa with SNP data and a phylogenetic network, calculate the 
    likelihood of the network given the data using the SNP likelihood algorithm.
    
    Automatically selects CPU or GPU execution based on network complexity:
      - Level-0 (tree):    always CPU
      - Level-1 (1 retic): GPU when taxa > 20
      - Level-2 (2 retics): GPU when taxa > 12
    
    Before starting computation, a pre-flight feasibility check predicts
    peak memory usage and raises SNPResourceError if the computation would 
    exceed GPU VRAM or system RAM.

    Args:
        filename (str): string path destination of a nexus file that 
                        contains SNP data and a network
        u (float): Parameter for the probability of an allele changing 
                   from red to green.
        v (float): Parameter for the probability of an allele changing 
                   from green to red.
        coal (float): Parameter for the rate of coalescence.
        samples (dict[str, int]): Mapping from taxon names to sample counts.
        max_workers (int, optional): The number of workers to use for parallel 
                                     computation. Only used if sequential is False.
                                     Defaults to 8.
        sequential (bool, optional): Whether to use sequential computation. 
                                     Defaults to True.
        executor (Executor, optional): The executor to use. If None, one is 
                                    auto-selected based on network complexity.
                                    Defaults to None.
    Returns:
        float: The log likelihood (a negative number) of the network.
    
    Raises:
        SNPResourceError: If the computation would exceed available hardware
                          resources (GPU VRAM or system RAM).
    """
    
    net = read_nexus(filename)[0]
    
    aln = MSA(filename)
    
    snp_model = build_model(filename, net)
    
    # ── Analyze network complexity ──────────────────────────────────────
    level = _compute_network_level(snp_model)
    n_taxa = len(snp_model.nodetypes.get("leaf", []))
    n_sites = aln.dim()[1]
    
    # Compute the true max lineages: with reticulations, lineage duplication
    # at split points means the effective lineage count can exceed sum(samples).
    max_n = _compute_max_lineages(snp_model, samples)
    
    # ── Auto CPU/GPU routing ────────────────────────────────────────────
    gpu_taxa_threshold = GPU_THRESHOLD.get(level, 8)
    needs_gpu = n_taxa > gpu_taxa_threshold
    use_gpu = needs_gpu and GPU_SPECS.available
    
    if needs_gpu and not GPU_SPECS.available:
        import warnings
        warnings.warn(
            f"Network has {n_taxa} taxa at level-{level} "
            f"(GPU recommended above {gpu_taxa_threshold} taxa), "
            f"but no GPU detected. Falling back to CPU — "
            f"computation may be slow.",
            RuntimeWarning,
            stacklevel=2,
        )
    
    # ── Determine site batch size ──────────────────────────────────────
    batch_size = _compute_batch_size(snp_model, samples, n_sites, use_gpu)
    n_batches = (n_sites + batch_size - 1) // batch_size
    
    peak_bytes, peak_shape = _estimate_peak_vpi_memory(
        snp_model, samples, min(batch_size, n_sites)
    )
    peak_mb = peak_bytes / (1024 * 1024)
    
    device_str = (f"GPU ({GPU_SPECS.name})" if use_gpu 
                  else "CPU")
    print(f"SNP_LIKELIHOOD: {n_taxa} taxa, level-{level}, "
          f"max_lineages={max_n}, {n_sites} sites")
    batch_info = f" ({n_batches} batches of {batch_size})" if n_batches > 1 else ""
    print(f"  Device: {device_str} | "
          f"Peak tensor: {' × '.join(str(s) for s in peak_shape)} "
          f"({peak_mb:.1f} MB){batch_info}")
    
    # ── Build shared objects ────────────────────────────────────────────
    q = BiMarkersTransition(max_n, u, v, coal)
    
    for leaf in snp_model.nodetypes["leaf"]:
        assert(type(leaf) is LeafNode)
        leaf.samples = samples[leaf.get_name()]
    
    def _run_batch(site_slice: tuple[int, int] | None) -> float:
        """Run one site batch and return its log-likelihood contribution."""
        batch_sites = (site_slice[1] - site_slice[0]) if site_slice else n_sites
        strategy = SNPStrategy(q, u, v, coal, n_sites, max_n, 
                               site_slice=site_slice,
                               use_gpu=use_gpu)
        visitor = SNPModelVisitor(strategy)
        for node in Traversal(snp_model.get_root(), TraversalOrder.POST_ORDER):
            visitor.visit(node)
        return strategy.L
    
    # ── Execute ─────────────────────────────────────────────────────────
    start_t = time.perf_counter()
    
    if n_batches == 1:
        # No batching needed — process all sites at once
        total_log_lik = _run_batch(None)
    else:
        # Site batching: sum log-likelihoods across independent batches
        total_log_lik = 0.0
        for b in range(n_batches):
            s_start = b * batch_size
            s_end = min(s_start + batch_size, n_sites)
            batch_lik = _run_batch((s_start, s_end))
            total_log_lik += batch_lik
            if n_batches <= 20 or b % max(1, n_batches // 10) == 0:
                print(f"    Batch {b+1}/{n_batches}: sites [{s_start}:{s_end}] "
                      f"log-lik={batch_lik:.4f}")
    
    end_t = time.perf_counter()
    print(f"  Total time: {end_t - start_t:.3f}s | log-lik = {total_log_lik:.6f}")
    
    return total_log_lik


##################
# Model Building #
##################

def build_model(filename : str,
                net : Network) -> Model:
    """
    Build a SNP model from a data file and network.
    """
    #Parse data 
    aln = MSA(filename)
    
    #Build components
    network = NetworkComponent(net = net)
    msa = MSAComponent({NetworkComponent}, aln)
    
    #Auto Build Model
    model = ModelFactory(network, msa).build()
    
    #Attach the root likelihood aggregator
    snp_root = RootAggregatorNode()
    model.root = snp_root
    net_root : RootNode = model.nodetypes["root"][0]
    net_root.join(snp_root)
    
    return model

#########################
### Transition Matrix ###
#########################

class BiMarkersTransition:
    """
    Class that encodes the probabilities of transitioning from one (n,r) pair 
    to another under a Biallelic model.

    Includes method for efficiently computing e^Qt

    Inputs:
    1) n-- the total number of samples in the species tree
    2) u-- the probability of going from the red allele to the green one
    3) v-- the probability of going from the green allele to the red one
    4) coal-- the coalescent rate constant, theta

    Assumption: Matrix indexes start with n=1, r=0, so Q[0][0] is Q(1,0);(1,0)

    Q Matrix is given by Equation 15 from:

    David Bryant, Remco Bouckaert, Joseph Felsenstein, Noah A. Rosenberg, 
    Arindam RoyChoudhury, Inferring Species Trees Directly from Biallelic 
    Genetic Markers: Bypassing Gene Trees in a Full Coalescent Analysis, 
    Molecular Biology and Evolution, Volume 29, Issue 8, August 2012, 
    Pages 1917–1932, https://doi.org/10.1093/molbev/mss086
    """

    def __init__(self, n : int, u : float, v : float, coal : float) -> None:
        """
        Initialize the Q matrix

        Args:
            n (int): sample count
            u (float): probability of a lineage going from red to green
            v (float): probability of a lineage going from green to red
            coal (float): coal rate, theta.
        Returns:
            N/A
        """

        # Build Q matrix
        self.n = n 
        self.u = u
        self.v = v
        self.coal = coal

        rows = int(.5 * self.n * (self.n + 3))
        self.Q : np.ndarray = np.zeros((rows, rows))
        
        # n ranges from 1 to individuals sampled (both inclusive)
        for n_prime in range(1, self.n + 1):  
            # r ranges from 0 to n (both inclusive)
            for r_prime in range(n_prime + 1):  
                
                # get indeces from n,r pair 
                n_r = nr_to_index(n_prime, r_prime)
                nm_rm = nr_to_index(n_prime - 1, r_prime - 1)
                n_rm = nr_to_index(n_prime, r_prime - 1)
                nm_r = nr_to_index(n_prime - 1, r_prime)
                n_rp = nr_to_index(n_prime, r_prime + 1)
                

                #### EQ 15 ####
                
                # THE DIAGONAL. always calculated
                self.Q[n_r][n_r] = - (n_prime * (n_prime - 1) / coal) \
                                       - (v * (n_prime - r_prime)) \
                                       - (r_prime * u)

                # These equations only make sense if r isn't 0 
                # (and the second, if n isn't 1).
                if 0 < r_prime <= n_prime:
                    if n_prime > 1:
                        self.Q[n_r][nm_rm] = (r_prime - 1) * n_prime / coal
                    self.Q[n_r][n_rm] = (n_prime - r_prime + 1) * v

                # These equations only make sense if r is strictly less than n 
                # (and the second, if n is not 1).
                if 0 <= r_prime < n_prime:
                    if n_prime > 1:
                        self.Q[n_r][nm_r] = (n_prime - 1 - r_prime) \
                                            * n_prime / coal
                    self.Q[n_r][n_rp] = (r_prime + 1) * u

    def expt(self, t : float = 1) -> np.ndarray:
        """
        Compute e^(Q*t) efficiently.
        
        Args:
            t (float): time, generally in coalescent units. Optional, defaults 
                       to 1, in which case e^Q is computed.
        
        Returns:
            np.ndarray: e^(Q*t).
        """
        return expm(self.Q * t)

    def cols(self) -> int:
        """
        return the dimension of the Q matrix
        
        Args:
            N/A
        Returns:
            int: the number of columns in the Q matrix
        """
        return self.Q.shape[1]

    def getQ(self) -> np.ndarray:
        """
        Retrieve the Q matrix.

        Args:
            N/A
        Returns:
            np.ndarray: The Q matrix.
        """
        return self.Q

class SNPStrategy(Strategy):
    """
    Visitor for the SNP model.
    
    Supports both CPU (NumPy) and GPU (CuPy) execution. When use_gpu=True,
    all tensor operations run on the GPU via CuPy's drop-in NumPy API.
    """
    def __init__(self, q : BiMarkersTransition, u : float, v : float, 
                 coal : float, sites : int, max_samples : int,
                 site_slice: tuple[int, int] | None = None,
                 use_gpu: bool = False) -> None:
        self.q : BiMarkersTransition = q
        self.u : float = u
        self.v : float = v
        self.coal : float = coal
        self.total_sites : int = sites
        self.site_slice : tuple[int, int] | None = site_slice
        self.sites : int = (site_slice[1] - site_slice[0]) if site_slice else sites
        self.vector_len : int = state_dim(max_samples)
        self.L : float = 0.0
        
        # GPU support: xp is the array module (numpy or cupy)
        self.use_gpu : bool = use_gpu and CUPY_RUNTIME_OK
        if self.use_gpu:
            self.xp = cp
        else:
            self.xp = np
        
        #Cache is from nodes to [is_dirty, vpi]. If is_dirty is true, needs recalculation
        self.cache : dict[ModelNode, tuple[bool, NodeVPI]] = dict()

    def _get_cached_vpi(self, n: ModelNode) -> NodeVPI | None:
        """Return the cached VPI if *n* is clean, otherwise ``None``."""
        entry = self.cache.get(n)
        if entry is not None and entry[0] is False:
            return entry[1]
        return None
    
    # ───────────────────────────────────────────────────────────────────────
    # GPU helpers
    # ───────────────────────────────────────────────────────────────────────
    
    def _scatter_add(self, target, slices, values) -> None:
        """
        Scatter-add: target[slices] += values, handling duplicate indices.
        
        On CPU: uses np.add.at (unbuffered).
        On GPU: uses cupyx.scatter_add (CUDA-accelerated).
        """
        if self.use_gpu:
            import cupyx
            cupyx.scatter_add(target, slices, values)
        else:
            np.add.at(target, slices, values)
    
    def _get_sm(self, mx: int, my: int) -> SparseMerge:
        """Get sparse merge tensor on the correct device."""
        if self.use_gpu:
            return get_sparse_merge_gpu(mx, my)
        return get_sparse_merge(mx, my)
    
    def _get_ss(self, m: int, gamma: float) -> SparseSplit:
        """Get sparse split tensor on the correct device."""
        if self.use_gpu:
            return get_sparse_split_gpu(m, gamma)
        return get_sparse_split(m, gamma)
    
    def _rescale(self, F, log_scale):
        """
        Per-vector rescaling to prevent numerical underflow.
        
        Normalizes each vector (along the last axis) independently so that 
        its max absolute value is 1.0. The per-vector log scaling factors 
        are accumulated into log_scale, which grows to shape F.shape[:-1].
        
        Works on both CPU (NumPy) and GPU (CuPy) via self.xp.
        """
        xp = self.xp
        # Per-vector max: max over the LAST axis only
        max_val = xp.max(xp.abs(F), axis=-1, keepdims=True)  # shape (..., 1)
        max_val = xp.where(max_val > 0, max_val, 1.0)
        
        F = F / max_val
        
        log_max = xp.log(max_val.squeeze(-1))  # shape = F.shape[:-1]
        
        # Expand log_scale with trailing singletons if it has fewer dims
        if log_scale.ndim < log_max.ndim:
            extra_dims = log_max.ndim - log_scale.ndim
            log_scale = log_scale.reshape(log_scale.shape + (1,) * extra_dims)
        
        log_scale = log_scale + log_max
        
        return F, log_scale
    
    def _rescale_global(self, F, log_scale):
        """
        Global per-site rescaling (axis-order agnostic).
        
        Divides all elements by the global per-site maximum. This is 
        compatible with any axis arrangement and is used within the 
        reticulation computation where axes are being swapped.
        
        The log_scale must be per-site (1-D) for this method.
        Works on both CPU (NumPy) and GPU (CuPy) via self.xp.
        """
        xp = self.xp
        reduce_axes = tuple(range(1, F.ndim))
        max_val = xp.max(xp.abs(F), axis=reduce_axes)  # shape (S,)
        max_val = xp.where(max_val > 0, max_val, 1.0)
        
        scale_shape = (F.shape[0],) + (1,) * (F.ndim - 1)
        F = F / max_val.reshape(scale_shape)
        
        if log_scale.ndim > 1:
            raise ValueError(
                f"_rescale_global requires per-site log_scale (1-D), "
                f"got shape {log_scale.shape}"
            )
        log_scale = log_scale + xp.log(max_val)
        
        return F, log_scale
    
    def _equalize_scales(self, F, log_scale):
        """
        Equalize per-vector scales to a common per-site maximum.
        
        Before merge/split operations that combine values across 
        different vector positions, we need all vectors within the same 
        site to be on a comparable scale.
        
        Works on both CPU (NumPy) and GPU (CuPy) via self.xp.
        """
        xp = self.xp
        if log_scale.ndim <= 1:
            # Already per-site scalar — nothing to equalize
            return F, log_scale
        
        # Per-site max of the log scales
        reduce_axes = tuple(range(1, log_scale.ndim))
        common_log = xp.max(log_scale, axis=reduce_axes)  # shape (S,)
        
        # Compute adjustment: exp(log_scale - common_log)
        broadcast_shape = (log_scale.shape[0],) + (1,) * (log_scale.ndim - 1)
        diff = log_scale - common_log.reshape(broadcast_shape)
        
        # Clamp diff to prevent exp() underflow for hugely negative diffs
        diff = xp.maximum(diff, -500.0)
        
        # Apply adjustment to F: add trailing dim for broadcasting with last axis
        adjustment = xp.exp(diff)[..., xp.newaxis]  # shape (S, d1, ..., dk-1, 1)
        F = F * adjustment
        
        return F, common_log
    
    def _rule1(self, F, branch_len: float, d: int):
        """
        Given vpi tensor F, with interface α, and branch length t, and max lineages m_α, 
        compute the vpi tensor F_top at the top of the branch.

        P(t) is computed on CPU (small matrix, scipy.linalg.expm) and transferred
        to GPU if needed. The matmul F @ P_sub runs on the tensor's device.
        """
        P_full = self.q.expt(branch_len)   # always NumPy (CPU)
        P_sub = P_full[:d, :d]
        if self.use_gpu:
            P_sub = self.xp.asarray(P_sub)  # transfer to GPU
        return F @ P_sub
    
    def _rule2(self, F_x, F_y, m_x: int, m_y: int):
        """
        Merge two VPI tensors from disjoint sub-networks (SPARSE).
        
        F_x has shape (S, *x_extra, state_dim(m_x))
        F_y has shape (S, *y_extra, state_dim(m_y))
        Result has shape (S, *x_extra, *y_extra, state_dim(m_x + m_y))
        
        Uses sparse COO merge coefficients. GPU-compatible via self.xp.
        The nnz dimension is processed in chunks to prevent GPU/CPU OOM
        from the broadcast intermediate (S, *x_extra, *y_extra, nnz).
        """
        xp = self.xp
        sm = self._get_sm(m_x, m_y)
        nnz = len(sm.i_arr)
        
        n_x_extra = F_x.ndim - 2
        n_y_extra = F_y.ndim - 2
        
        # Pre-compute result shape: (S, *x_extra, *y_extra, dim_z)
        result_shape = (list(F_x.shape[:-1]) 
                        + list(F_y.shape[1:-1]) 
                        + [sm.dim_z])
        result = xp.zeros(result_shape, dtype=F_x.dtype)
        
        # ---------- determine chunk size for the nnz dimension ----------
        # The broadcast intermediate has shape (S, *x_extra, *y_extra, chunk)
        # and we want that to stay within a memory budget.
        S = F_x.shape[0]
        prefix_elems = S
        for d in F_x.shape[1:-1]:   # x_extra dims
            prefix_elems *= d
        for d in F_y.shape[1:-1]:   # y_extra dims
            prefix_elems *= d
        
        # Budget: ~1 GB on CPU, 25% of free VRAM on GPU
        if self.use_gpu:
            free = cp.cuda.Device().mem_info[0]
            budget_bytes = max(int(free * 0.25), 256 * 1024**2)
        else:
            budget_bytes = 1 * 1024**3   # 1 GB
        
        # Each nnz element in the chunk costs 8 bytes × prefix_elems
        # (the contribution array is the dominant allocation)
        bytes_per_nz = prefix_elems * 8
        chunk_size = max(1, budget_bytes // bytes_per_nz) if bytes_per_nz > 0 else nnz
        
        # ---------- chunked gather → broadcast → scatter-add ----------
        for c0 in range(0, nnz, chunk_size):
            c1 = min(c0 + chunk_size, nnz)
            
            fx_g = F_x[..., sm.i_arr[c0:c1]]   # (S, *x_extra, chunk)
            fy_g = F_y[..., sm.j_arr[c0:c1]]   # (S, *y_extra, chunk)
            
            # Reshape for outer-product broadcast over extra dims
            if n_y_extra > 0:
                fx_shape = list(fx_g.shape)
                for _ in range(n_y_extra):
                    fx_shape.insert(-1, 1)
                fx_g = fx_g.reshape(fx_shape)
            
            if n_x_extra > 0:
                fy_shape = list(fy_g.shape)
                for _ in range(n_x_extra):
                    fy_shape.insert(1, 1)
                fy_g = fy_g.reshape(fy_shape)
            
            contributions = fx_g * fy_g * sm.coeff_arr[c0:c1]
            self._scatter_add(result, (..., sm.k_arr[c0:c1]), contributions)
            
            # Free intermediates eagerly on GPU
            if self.use_gpu:
                del fx_g, fy_g, contributions
        
        return result
    
    def _rule3(self, F, mx, gammax):
        """
        Split a VPI tensor at a reticulation node (SPARSE).
        
        F has shape (..., state_dim(mx))
        Result has shape (..., state_dim(mx), state_dim(mx))
        
        GPU-compatible via self.xp and self._scatter_add.
        """
        xp = self.xp
        ss = self._get_ss(mx, gammax)
        
        # Gather: F[..., i_arr] → shape (..., nnz)
        f_g = F[..., ss.i_arr]
        
        # Weighted contributions: (..., nnz)
        contributions = f_g * ss.coeff_arr
        
        # Scatter into 2D output (j, k)
        prefix_shape = F.shape[:-1]
        result = xp.zeros((*prefix_shape, ss.dim, ss.dim), dtype=F.dtype)
        
        # Use linear index for scatter: j * dim + k
        linear_idx = ss.j_arr * ss.dim + ss.k_arr
        result_flat = result.reshape(*prefix_shape, ss.dim * ss.dim)
        self._scatter_add(result_flat, (..., linear_idx), contributions)
        
        return result_flat.reshape(result.shape)
    
    def _rule4(self, F, mx, my):
        """
        Merge two interfaces from the same VPI tensor (SPARSE).
        
        F has shape (..., state_dim(mx), state_dim(my))
        Result has shape (..., state_dim(mx + my))
        
        GPU-compatible via self.xp and self._scatter_add.
        """
        xp = self.xp
        sm = self._get_sm(mx, my)
        
        # F[..., i_arr, j_arr] gathers the (i, j) pairs → shape (..., nnz)
        f_g = F[..., sm.i_arr, sm.j_arr]
        
        # Weighted contributions: (..., nnz)
        contributions = f_g * sm.coeff_arr
        
        # Scatter into result
        prefix_shape = F.shape[:-2]
        result = xp.zeros((*prefix_shape, sm.dim_z), dtype=F.dtype)
        self._scatter_add(result, (..., sm.k_arr), contributions)
        
        return result  
        
    def compute_at_leaf(self, n: LeafNode) -> NodeVPI:
        """
        Compute the partial likelihoods at a leaf node.

        The format for the partial likelihoods is a two dimensional array where 
        the first dimension (rows) is the site index and the second dimension 
        (columns) is the number of samples for this leaf. 
        
        Uses vectorized indexing for GPU efficiency (no Python per-site loop).
        """
        xp = self.xp
        
        if n in self.cache:
            if self.cache[n][0] is False:
                return self.cache[n][1]
            
        assert len(n.data) == 1, "Leaf node must have exactly one data sequence"
        reds : list[int] = n.data[0].get_numerical_seq()
        
        # Slice to current batch if site batching is active
        if self.site_slice is not None:
            reds = reds[self.site_slice[0]:self.site_slice[1]]
        
        # Vectorized leaf initialization (GPU-friendly: single kernel launch)
        # Pre-compute state indices on CPU, then transfer
        state_indices_np = np.array([nr_to_index(n.samples, r) for r in reds], dtype=np.int64)
        
        F = xp.zeros((self.sites, state_dim(n.samples)), dtype=xp.float64)
        site_indices = xp.arange(self.sites)
        state_indices = xp.asarray(state_indices_np) if self.use_gpu else state_indices_np
        F[site_indices, state_indices] = 1.0
        
        # log_scale shape matches F.shape[:-1] = (sites,)
        log_scale = xp.zeros(self.sites, dtype=xp.float64)
        F = self._rule1(F, n.branch().length, state_dim(n.samples))
        F, log_scale = self._rescale(F, log_scale)
        
        n.vpi = NodeVPI(F, [f"{n.get_name()}_top"], [n.samples], log_scale)
        self.cache[n] = (False, n.vpi)
        return n.vpi

    def compute_at_internal(self, n: InternalNode, x : NodeVPI, y: NodeVPI) -> NodeVPI:
        """
        Compute the partial likelihoods at an internal node.
        """
        cached = self._get_cached_vpi(n)
        if cached is not None:
            return cached
        
        rule2 = _disjoint_subnets(n)
        
        if rule2:
            mx = x.max_lineages[-1]
            my = y.max_lineages[-1]
            
            # Equalize both VPIs to per-site scales before merge
            x_F, x_ls = self._equalize_scales(x.tensor, x.log_scale)
            y_F, y_ls = self._equalize_scales(y.tensor, y.log_scale)
            
            F = self._rule2(x_F, y_F, mx, my)
            interfaces = x.interfaces[:-1] + y.interfaces[:-1] + [f"{n.get_name()}_top"] 
            max_lin = x.max_lineages[:-1] + y.max_lineages[:-1] + [mx + my]
            # After equalization, both log_scales are per-site (S,)
            log_scale = x_ls + y_ls
        else:
            mx = x.max_lineages[-2]
            my = x.max_lineages[-1]
            
            # Equalize multi-dim scales to per-site before rule4 merge
            x_F, x_ls = self._equalize_scales(x.tensor, x.log_scale)
            
            F = self._rule4(x_F, mx, my)
            interfaces = x.interfaces[:-2] + [f"{n.get_name()}_top"]
            max_lin = x.max_lineages[:-2] + [mx + my]
            log_scale = x_ls
        
        # Rescale result (log_scale starts per-site, grows with F's dims)
        F, log_scale = self._rescale(F, log_scale)
        F = self._rule1(F, n.branch().length, state_dim(max_lin[-1]))
        F, log_scale = self._rescale(F, log_scale)
           
        n.vpi = NodeVPI(F, interfaces, max_lin, log_scale)
        self.cache[n] = (False, n.vpi)
        return n.vpi
    
    def compute_at_reticulation(self, n : ReticulationNode, x : NodeVPI) -> NodeVPI:
        """
        Compute the partial likelihoods at a reticulation node.
        
        The split creates two new interfaces (one per parent branch).
        During the two transition steps, we use global per-site rescaling
        (axis-order agnostic) to prevent intermediate underflow.
        After both transitions, we do a final per-vector rescale.
        
        GPU-compatible via self.xp.
        """
        xp = self.xp
        
        cached = self._get_cached_vpi(n)
        if cached is not None:
            return cached
        
        branches : tuple[Branch, Branch]= n.branch_info
        
        gamma = branches[0].inheritance_probability
        m = x.max_lineages[-1]
        
        # Equalize to per-site log_scale before the split 
        x_F, log_scale = self._equalize_scales(x.tensor, x.log_scale)
        
        # rule3 (split): F shape (..., d) → (..., d_branch0, d_branch1)
        F = self._rule3(x_F, m, gamma)
        
        # Global per-site rescale (safe across any axis arrangement)
        F, log_scale = self._rescale_global(F, log_scale)
        
        # Apply transition on branch 0's axis (axis -2)
        F = xp.moveaxis(F, -2, -1)
        F = self._rule1(F, branches[0].length, state_dim(m))
        F, log_scale = self._rescale_global(F, log_scale)
        
        # Apply transition on branch 1's axis (now at -2, move to -1)
        F = xp.moveaxis(F, -1, -2)
        F = self._rule1(F, branches[1].length, state_dim(m))
        F, log_scale = self._rescale_global(F, log_scale)
        
        # Final per-vector rescale for downstream merge/split compatibility
        F, log_scale = self._rescale(F, log_scale)
        
        #Book keep the interfaces and lineages
        interfaces = x.interfaces[:-1] + [f"{n.get_name()}_{branches[0].parent_id}_top", f"{n.get_name()}_{branches[1].parent_id}_top"]  
        max_lin = x.max_lineages[:-1] + [m, m] 
        
        n.vpi = NodeVPI(F, interfaces, max_lin, log_scale)
        self.cache[n] = (False, n.vpi)
        return n.vpi
        
    def compute_at_root(self, n: RootNode, x : NodeVPI, y : NodeVPI) -> NodeVPI:
        """
        Compute the partial likelihoods at the root node.
        """
        cached = self._get_cached_vpi(n)
        if cached is not None:
            return cached
        
        rule2 = _disjoint_subnets(n)
        
        if rule2:
            # Equalize both VPIs to per-site scales before merge
            x_F, x_ls = self._equalize_scales(x.tensor, x.log_scale)
            y_F, y_ls = self._equalize_scales(y.tensor, y.log_scale)
            
            F = self._rule2(x_F, y_F, x.max_lineages[-1], y.max_lineages[-1])
            final_lin = x.max_lineages[-1] + y.max_lineages[-1]
            log_scale = x_ls + y_ls
        else:
            x_F, x_ls = self._equalize_scales(x.tensor, x.log_scale)
            
            F = self._rule4(x_F, x.max_lineages[-2], x.max_lineages[-1])
            final_lin = x.max_lineages[-2] + x.max_lineages[-1]
            log_scale = x_ls
        
        F, log_scale = self._rescale(F, log_scale)
        
        n.vpi = NodeVPI(F, [f"{n.get_name()}_bottom"], [final_lin], log_scale)
        self.cache[n] = (False, n.vpi)
        return n.vpi
        
    def compute_at_aggregator(self, n: RootAggregatorNode, root : NodeVPI) -> None:
        """
        Compute the partial likelihoods at a root aggregator node.
        
        Uses the accumulated per-vector log_scale factors to recover the
        true log-likelihood without numerical underflow. The root VPI
        should have shape (S, d) with log_scale shape (S,) after 
        equalization at the root node.
        
        GPU-compatible: pi is built on CPU and transferred; final
        log-likelihood is brought back to CPU as a Python float.
        """
        xp = self.xp
        
        if self._get_cached_vpi(n) is not None:
            return 
        
        m = root.max_lineages[-1]
        
        # Compute stationary distribution on CPU (small vector, scalar math)
        theta_r = self.v / (self.u + self.v)
        theta_g = self.u / (self.u + self.v)
        
        pi_np = np.zeros(state_dim(m))
        for n_lin in range(1, m + 1):
            for r in range(n_lin + 1):
                idx = nr_to_index(n_lin, r)
                pi_np[idx] = comb(n_lin, r) * (theta_r ** r) * (theta_g ** (n_lin - r))
        
        # Normalize
        pi_np = pi_np / np.sum(pi_np)
        
        # Transfer to GPU if needed
        pi = xp.asarray(pi_np) if self.use_gpu else pi_np
        
        # Equalize to per-site scale if log_scale is still multi-dimensional
        root_F, root_ls = self._equalize_scales(root.tensor, root.log_scale)
        
        # Compute log likelihood per site:
        #   true_lik[s] = scaled_lik[s] * exp(log_scale[s])
        #   log(true_lik[s]) = log(scaled_lik[s]) + log_scale[s]
        scaled_site_likelihoods = root_F @ pi  # [S] array
        
        # Clamp to avoid log(0) — sites where lik=0 get -inf contribution
        scaled_site_likelihoods = xp.maximum(
            scaled_site_likelihoods, xp.finfo(xp.float64).tiny
        )
        
        log_site_likelihoods = xp.log(scaled_site_likelihoods) + root_ls
        
        # Bring result back to CPU as a Python float
        self.L = float(xp.sum(log_site_likelihoods))
        n.vpi = NodeVPI(self.L, [], [], np.zeros(0))
        self.cache[n] = (False, n.vpi)
        
class SNPModelVisitor(Visitor):
    """
    Visitor for the SNP model.
    """
    def __init__(self, strategy: SNPStrategy) -> None:
        self.strategy : SNPStrategy = strategy
        self.vpis : list[NodeVPI] = []
            
    def visit_leaf(self, n: LeafNode) -> None:
        self.vpis.append(self.strategy.compute_at_leaf(n))
        
    def visit_internal(self, n: InternalNode) -> None:
        child_vpis : list[NodeVPI]= [
            self._get_vpi_for(n, child.get_name(), retic=(child.get_node_type() == "reticulation"))
            for child in n.get_model_children()
        ]
        unique_vpis = deduplicate_vpis(child_vpis)
        self._remove(unique_vpis[0])
        if len(unique_vpis) == 1:
            self.vpis.append(self.strategy.compute_at_internal(n, unique_vpis[0], unique_vpis[0]))
        else:
            self._remove(unique_vpis[1])
            self.vpis.append(self.strategy.compute_at_internal(n, unique_vpis[0], unique_vpis[1]))
        
    def visit_reticulation(self, n: ReticulationNode) -> None:
        child_vpi : NodeVPI = [self._get_vpi_for(n, child.get_name()) for child in n.get_model_children()][0]
        self._remove(child_vpi)
        self.vpis.append(self.strategy.compute_at_reticulation(n, child_vpi))
        
    def visit_root(self, n: RootNode) -> None:
        child_vpis : list[NodeVPI]= [
            self._get_vpi_for(n, child.get_name(), retic=(child.get_node_type() == "reticulation"))
            for child in n.get_model_children()
        ]
        unique_vpis = deduplicate_vpis(child_vpis)
        self._remove(unique_vpis[0])
        if len(unique_vpis) == 1:
            self.vpis.append(self.strategy.compute_at_root(n, unique_vpis[0], unique_vpis[0]))
        else:
            self._remove(unique_vpis[1])
            self.vpis.append(self.strategy.compute_at_root(n, unique_vpis[0], unique_vpis[1]))
        
    def visit_aggregator(self, n: RootAggregatorNode) -> None:
        child_vpi : NodeVPI = [self._get_vpi_for(n, child.get_name()) for child in n.get_model_children()][0]
        self.strategy.compute_at_aggregator(n, child_vpi)
        
    
    def visit(self, n: ModelNode) -> None:
        """
        Visit a node.
        """
        """Dispatch to the correct visit method based on node type."""
        dispatch = {
            "leaf": self.visit_leaf,
            "internal": self.visit_internal,
            "root": self.visit_root,
            "reticulation": self.visit_reticulation,
            "root_aggregator": self.visit_aggregator,
        }
        return dispatch[n.get_node_type()](n)

    def _get_vpi_for(self, n : ModelNode, node_label : str, retic : bool = False) -> NodeVPI:
        """
        Find the VPI containing the interface for a given child node.
        
        Args:
            n: The parent model node requesting the VPI.
            node_label: The name of the child model node to look up.
            retic: If True, the child is a reticulation node, and we must
                   match the exact interface '{node_label}_{n.get_name()}_top'
                   to select the correct parent-specific interface.
        """
        
        def _iface_matches_label(iface: str, label: str) -> bool:
            """
            Delimiter-aware match: checks startswith('{label}_').
            Prevents 'T1' from matching 'T10_top'.
            """
            return iface.startswith(label + "_") or iface == label
        
        def get() -> tuple[int, NodeVPI]:
            if retic:
                # Reticulation interfaces have the form: {retic_name}_{parent_name}_top
                # We need to find the exact interface for THIS parent
                expected_iface = f"{node_label}_{n.get_name()}_top"
                for vpi in self.vpis:
                    for index, iface in enumerate(vpi.interfaces):
                        if iface == expected_iface:
                            return index, vpi
            else:
                for vpi in self.vpis:
                    for index, iface in enumerate(vpi.interfaces):
                        if _iface_matches_label(iface, node_label):
                            return index, vpi 
            return None
        
        def move_to_end(v : NodeVPI, index : int) -> None:
            if index == len(v.interfaces) - 1:
                return  # Already at the end, nothing to do
            
            # Use the strategy's xp module for GPU compatibility
            xp = self.strategy.xp
            
            # If log_scale is multi-dimensional, equalize to per-site BEFORE
            # rearranging axes (log_scale must match tensor.shape[:-1], which
            # changes after moveaxis)
            if v.log_scale is not None and v.log_scale.ndim > 1:
                v.tensor, v.log_scale = self.strategy._equalize_scales(
                    v.tensor, v.log_scale
                )
            
            #Move interface to end (use pop to handle duplicate values safely)
            interface_to_move = v.interfaces.pop(index)
            v.interfaces.append(interface_to_move)
            #Move max_lineages to end (must stay in sync with interfaces)
            lineage_to_move = v.max_lineages.pop(index)
            v.max_lineages.append(lineage_to_move)
            #Move tensor axis to end (xp.moveaxis returns a new array!)
            # Tensor axes: axis 0 = sites, axes 1..N = interfaces
            v.tensor = xp.moveaxis(v.tensor, index + 1, -1)
            # log_scale is now per-site (1-D) — compatible with any axis order
            
        
        result = get()
        if result is None:
            raise ValueError(
                f"Could not find VPI for node '{node_label}' "
                f"(parent: '{n.get_name()}', retic={retic}). "
                f"Available interfaces: {[iface for vpi in self.vpis for iface in vpi.interfaces]}"
            )
        i, n_vpi = result
        move_to_end(n_vpi, i)
        return n_vpi

    def _remove(self, nv : NodeVPI) -> None:
        self.vpis = [v for v in self.vpis if v is not nv]
    
    