#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author : Mark Kessler
Last Edit : 5/12/26
First Included in Version : 1.0.0

BiMarkers -- biallelic SNP (single-nucleotide polymorphism) likelihood
under the multispecies network coalescent for phylogenetic networks.

This module implements the SNAPP-style two-state continuous-time Markov
chain likelihood (Bryant et al. 2012) generalised to reticulate topologies
(Zhu et al. 2018, `PLOS Comput Biol e1005932
<https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005932>`_).

This is an implementation module.  Users reach it through the two verbs
with :class:`~phynetpy.data.BiallelicMarkers` data::

    score(net, markers, criterion=Likelihood())   # the SNP likelihood
    infer(markers, criterion=Bayesian())          # MH over network space

It provides:

* :func:`_snp_log_likelihood` -- the log likelihood of a species network
  given an in-memory marker matrix; the numerical core of the
  ``(BiallelicMarkers, MSC, Likelihood)`` cell.
* :func:`_snp_mcmc` -- a Metropolis-Hastings search driver wired to that
  likelihood; the core of the ``(BiallelicMarkers, MSC, Bayesian)`` cell.
* :class:`SNPScorer` -- the ``Callable[[Model], float]`` adapter both use.
* A NumPy implementation that offloads the site loop to the GPU via
  ``cupy`` when both the hardware and that package are present, and runs
  on the CPU otherwise.

Docs   - [x]
Tests  - [x]
Design - [x]
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

import copy
import math
from collections import deque
from functools import lru_cache
from math import comb
import time
from typing import TYPE_CHECKING, Optional
import numpy as np
from scipy.linalg import expm, null_space
from dataclasses import dataclass

# CuPy import - with graceful fallback to the pure NumPy CPU path.

CUPY_RUNTIME_OK = False

try:
    import cupy as cp
    if cp.cuda.is_available():
        # Seeing a device is not enough: a CUDA version mismatch only
        # surfaces once a kernel is compiled, so run one real operation
        # (``cp.random`` needs compilation) before trusting the GPU path.
        try:
            _test = cp.random.rand(10, dtype=cp.float64)
            _ = cp.sum(_test)
            del _test
            CUPY_RUNTIME_OK = True
        except Exception:
            pass
except ImportError:
    cp = None


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
        """VRAM capacity in gigabytes (converted from :attr:`vram_bytes`)."""
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
from .MSA import MSA
from .Network import Network, Node, Edge, Branch
from ._units import (
    BranchLengthUnit,
    require_branch_length_unit,
    resolve_branch_theta,
)
from .ModelGraph import Model
from ._snp_model import (
    SNPModel,
    ModelNode,
    InternalNode,
    LeafNode,
    ReticulationNode,
    RootNode,
    RootAggregatorNode,
    build_snp_model,
    postorder,
)
from .GraphUtils import level as network_level, network_clusters

if TYPE_CHECKING:
    # Imported lazily at call sites to keep the MCMC stack out of the
    # import path of the likelihood-only entry points.
    from ._mcmc_gt import MCMC_GTPriors

#: Public surface of this module.  Everything else here (Q-matrix builders,
#: sparse split/merge tensors, VPI bookkeeping, GPU shims) is internal to the
#: biallelic-marker likelihood and is not part of the PhyNetPy API.
#:
#: The user-facing entry points are ``infer`` and ``score`` with
#: :class:`~phynetpy.data.BiallelicMarkers` data; the engines in
#: :mod:`phynetpy._engines` call ``_snp_mcmc`` and ``_snp_log_likelihood``
#: here directly.
__all__ = [
    "SNPScorer",
    "SNPResourceError",
]


def n_to_index(n : int) -> int:
    """
    Computes the starting index in computing a linear index for an (n,r) pair.
    Returns the index, if r is 0.

    The state space **includes the empty state** ``(0, 0)`` at index 0.  This
    is required for the phylogenetic-network coalescent: at a reticulation the
    lineages entering the hybrid split binomially between the two parent
    branches, and the boundary cases where *all* lineages inherit from one
    parent leave the other branch with **zero** lineages.  Omitting ``n = 0``
    silently drops those terms, so every reticulation loses probability mass
    (the total site-pattern probability falls below 1 and becomes branch-length
    dependent).  Including ``(0, 0)`` makes the split conserve probability.

    i.e n=0 returns 0, since (0,0) is index 0
    i.e n=1 returns 1, since (1,0) is index 1 (preceded by (0,0))
    i.e n=3 returns 6 since (3,0) is preceded by
        (0,0), (1,0), (1,1), (2,0), (2,1), and (2,2)

    Args:
        n (int): an n value (number of lineages) from an (n,r) pair
    Returns:
        int: starting index for that block of n values
    """
    return int(n * (n + 1) / 2)

def nr_to_index(n : int, r : int) -> int:
    """
    Takes an (n,r) pair and maps it to a 1d vector index

    (0,0) -> 0
    (1,0) -> 1
    (1,1) -> 2
    (2,0) -> 3
    ...

    Args:
        n (int): the number of lineages (``0`` is the empty state)
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
    ``0 <= n <= m`` and ``0 <= r <= n`` -- i.e. it includes the empty state
    ``(0, 0)``.  For ``m`` this is ``(m + 1)(m + 2) / 2``.
    """
    return nr_to_index(m, m) + 1


def _root_stationary(Q: np.ndarray, m: int) -> np.ndarray:
    """Root allele-frequency prior over the ``(n, r)`` states, ``0 <= n <= m``.

    Implements the root treatment of the biallelic likelihood (Bryant et al.
    2012, eq. 20; Zhu et al. 2018): the equilibrium distribution of the
    coalescent-with-mutation rate matrix ``Q``.  Because ``Q`` couples mutation
    (``r -> r +/- 1``) with coalescence (``n -> n - 1``), the correct prior is
    the null space of ``Q`` restricted to the ``n >= 1`` states -- not a plain
    binomial with the mutational base frequencies.  It is normalised so the
    single-lineage block ``{(1, 0), (1, 1)}`` sums to 1 (one ancestral lineage
    is red with probability ``theta_r``), matching the validated reference.

    Args:
        Q: The full biallelic rate matrix (state ordering includes the empty
            ``(0, 0)`` state at index 0; sized for the network's global maximum
            lineage count).
        m: Maximum lineage count at the root interface.

    Returns:
        A length ``state_dim(m)`` vector ``pi`` with ``pi[(0, 0)] = 0`` and
        ``pi[(n, r)]`` giving the root-population probability of ``(n, r)``.
    """
    d = state_dim(m)
    # Restrict to the n >= 1 block: index 0 is the absorbing empty state, which
    # contributes an extra (spurious) null vector and never occurs at the root.
    q_sub = Q[1:d, 1:d]
    ns = null_space(q_sub)
    if ns.shape[1] == 0:
        raise ValueError("Biallelic rate matrix has no stationary distribution")
    x = ns[:, 0]
    # Normalise so the single-lineage block (indices for (1,0) and (1,1),
    # i.e. the first two entries of the n >= 1 block) sums to 1.
    x = x / (x[0] + x[1])
    pi = np.zeros(d)
    pi[1:d] = x
    return pi

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
    
    for nx in range(0, mx + 1):
        for rx in range(nx + 1):
            i = nr_to_index(nx, rx)
            for ny in range(0, my + 1):
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
    Build sparse COO split tensor: ``S[i, j, k]`` where ``i`` is ``(n, r)``,
    ``j`` is ``(n_b, r_b)``, ``k`` is ``(n_d, r_d)``, and the coefficient is
    ``C(n, n_b) * gamma^{n_b} * (1-gamma)^{n_d}``.

    This is Rule 3 of the biallelic-network recursion (Zhu et al. 2018, PLOS
    Comp Biol e1005932; Rabier et al. 2021): the ``n`` lineages entering the
    hybrid partition into ``n_b`` inheriting from the branch-b parent and
    ``n_d = n - n_b`` from branch-d, with the ``C(n, n_b)`` ways of choosing
    which lineages go where and probability ``gamma^{n_b} (1-gamma)^{n_d}``.
    The coefficient does **not** depend on the red split ``(r_b, r_d)`` beyond
    the constraint ``r_b + r_d = r`` -- the hypergeometric merge (Rules 2/4)
    supplies the red-allele combinatorics, so applying it here too would double
    count.  The boundary cases ``n_b = 0`` / ``n_b = n`` (all lineages inherit
    from a single parent, leaving the other branch empty) MUST be included;
    dropping them is what makes a tree-only engine lose probability on
    networks.  ``n = 0`` is the empty state and splits to two empty states.

    Args:
        m: Max lineages at the node being split.
        gamma: Inheritance probability for the left parent branch.
    Returns:
        SparseSplit with coordinate arrays.
    """
    i_list, j_list, k_list, c_list = [], [], [], []
    
    for n in range(0, m + 1):
        for r in range(n + 1):
            i = nr_to_index(n, r)
            for nb in range(0, n + 1):
                nd = n - nb
                for rb in range(max(0, r - nd), min(r, nb) + 1):
                    rd = r - rb
                    
                    j = nr_to_index(nb, rb)
                    k = nr_to_index(nd, rd)
                    coeff = (comb(n, nb)
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


def _compute_max_lineages(model: SNPModel, samples: dict[str, int]) -> int:
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
    
    for node in postorder(model.root):
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


def _compute_network_level(model: SNPModel) -> int:
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


def _estimate_peak_vpi_memory(model: SNPModel, samples: dict[str, int], 
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
    # Track open interface dimensions per VPI, using the same logic
    # as the visitor: track (interfaces, max_lineages) per node
    vpi_dims : dict = {}   # ModelNode -> list[int] (max_lineages per interface)
    vpi_id   : dict = {}   # ModelNode -> id tracking which VPI object this belongs to
    
    peak_bytes = 0
    peak_shape = [n_sites]
    
    id_counter = 0
    
    for node in postorder(model.root):
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

# Only use this fraction of currently free VRAM. The VPI peak estimator tracks
# the largest persistent tensor; a 3x allowance covers its input/output
# workspaces. GPU batches are additionally capped because small estimated VPIs
# can still create large scatter temporaries at high site counts.
GPU_VRAM_SAFETY_FACTOR = 0.80
VPI_WORKSPACE_MULTIPLIER = 3
GPU_MAX_SITE_BATCH = 1_000


def _compute_batch_size(model: SNPModel, samples: dict[str, int],
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
        free_bytes = cp.cuda.Device(0).mem_info[0]
        available = int(free_bytes * GPU_VRAM_SAFETY_FACTOR)
    else:
        try:
            import psutil
            available = int(psutil.virtual_memory().available * 0.5)
        except ImportError:
            available = 8 * (1024 ** 3)  # conservative 8 GB
    
    # Sparse contractions retain inputs, an output, and transition workspaces.
    memory_per_site = per_site_bytes * VPI_WORKSPACE_MULTIPLIER
    
    if memory_per_site == 0:
        return n_sites
    
    max_batch = available // memory_per_site
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
    
    if use_gpu:
        max_batch = min(max_batch, GPU_MAX_SITE_BATCH)
    return min(max_batch, n_sites)


#####################
# Method Signatures #
#####################

_SNP_LOG_FLOOR: float = math.log(1e-200)


def _snp_starting_tree(taxa: list[str], delta: float = 0.02) -> Network:
    """Build an ultrametric caterpillar starting tree over ``taxa``.

    The Metropolis-Hastings search only needs a *valid* labelled starting
    point; the topology / branch-length / gamma moves explore from there.
    A caterpillar with strictly increasing internal-node heights guarantees
    a well-formed rooted binary tree whose leaf labels match the alignment
    (so :func:`~phynetpy._snp_model.build_snp_model` can bind each sequence
    to a leaf).

    Args:
        taxa: Leaf labels (must match the alignment taxon names).
        delta: Height increment between successive internal nodes.  Chosen
            on the coalescent-unit scale so the initial likelihood is finite.

    Returns:
        A rooted binary :class:`~phynetpy.Network.Network` on ``taxa``.
    """
    taxa = list(taxa)
    n = len(taxa)
    if n < 2:
        raise ValueError("need at least 2 taxa to build a starting tree")

    net = Network(
        branch_length_unit=BranchLengthUnit.SUBSTITUTIONS_PER_SITE
    )
    leaves = {t: Node(t) for t in taxa}
    internals = [Node(f"I{k}") for k in range(n - 1)]
    net.add_nodes(*leaves.values())
    net.add_nodes(*internals)

    # Internal node k sits at height (k + 1) * delta; leaves at height 0.
    height = [(k + 1) * delta for k in range(n - 1)]
    edges: list[Edge] = [
        Edge(internals[0], leaves[taxa[0]], length=height[0]),
        Edge(internals[0], leaves[taxa[1]], length=height[0]),
    ]
    for k in range(1, n - 1):
        edges.append(
            Edge(internals[k], internals[k - 1],
                 length=height[k] - height[k - 1])
        )
        edges.append(Edge(internals[k], leaves[taxa[k + 1]], length=height[k]))
    net.add_edges(edges)
    return net


class SNPScorer:
    """Callable biallelic-marker log-posterior scorer for MCMC.

    Mirrors the interface of :class:`phynetpy._mcmc_gt.MCMCGTScorer`
    (``__call__(model) -> float`` plus a ``last_log_likelihood`` attribute)
    so it drops directly into the shared Metropolis-Hastings loop.  Every
    call rebuilds the SNP model graph from ``model.network`` and the shared
    in-memory alignment, evaluates the Bryant biallelic likelihood via
    :func:`_snp_log_likelihood`, and (in posterior mode) adds the network
    prior.

    The alignment is parsed once and reused for the whole chain, so per
    iteration the only re-parsing cost is the (cheap) model-graph rebuild.
    """

    def __init__(self,
                 aln: MSA,
                 u: float,
                 v: float,
                 theta: float,
                 samples: dict[str, int],
                 priors: "MCMC_GTPriors",
                 *,
                 posterior: bool = True) -> None:
        """Create a new ``SNPScorer``.

        Args:
            aln: Biallelic-marker alignment shared across every call.
            u: Mutation rate red-to-green (1->0).
            v: Mutation rate green-to-red (0->1).
            theta: Population mutation rate ``4*N*mu``.
            samples: Per-taxon sample counts (number of alleles observed).
            priors: Network-prior hyperparameters used when ``posterior``
                is ``True``.
            posterior: When ``True``, add the network prior to the
                likelihood so calls return a log-posterior; when
                ``False``, return the log-likelihood alone.
        """
        self.aln = aln
        self.u = u
        self.v = v
        self.theta = theta
        self.samples = samples
        self.priors = priors
        self.posterior = posterior
        self.last_log_likelihood: Optional[float] = None
        self.last_log_posterior: Optional[float] = None

    def __call__(self, model: Model) -> float:
        net = model.network
        try:
            ll = _snp_log_likelihood(
                net, self.aln, self.u, self.v, self.theta, self.samples,
                verbose=False,
            )
        except Exception:
            ll = float("-inf")
        if not math.isfinite(ll):
            ll = _SNP_LOG_FLOOR
        self.last_log_likelihood = ll

        if self.posterior:
            from ._mcmc_gt import log_prior_network
            lp = ll + log_prior_network(net, self.priors)
            self.last_log_posterior = lp
            return lp
        self.last_log_posterior = ll
        return ll


def _snp_mcmc(aln: MSA,
              u: float = 1.0,
              v: float = 1.0,
              theta: float = 0.02,
              *,
              num_iter: int = 50000,
              burn_in: int = 10000,
              sample_freq: int = 100,
              seed: Optional[int] = None,
              samples: Optional[dict[str, int]] = None,
              max_reticulations: int = 4,
              max_level: Optional[int] = None,
              priors: "MCMC_GTPriors | None" = None,
              start_net: Optional[Network] = None,
              backbone: Optional[Network] = None,
              ) -> dict[Network, float]:
    """Sample networks from biallelic SNP data via Metropolis-Hastings.

    The numerical core behind ``infer(BiallelicMarkers(...),
    criterion=Bayesian())``.  Runs a rigorous MH chain over network space
    with the Bryant et al. (2012) biallelic-marker likelihood as the data
    term and a :class:`~phynetpy._mcmc_gt.MCMC_GTPriors` network prior.
    Proposals come from the shared
    :class:`~phynetpy._mcmc_gt.MCMCGTKernel` (SPR, ChangeNodeHeight, gamma
    tuning, and the full add/remove/relocate/flip reticulation suite), so
    acceptance uses the correct ``log_posterior_delta +
    log_hastings_ratio`` test and the adaptive kernel is frozen at the end
    of burn-in to preserve detailed balance.

    Args:
        aln: In-memory biallelic marker matrix (per-site red-allele counts).
        u: Red->green mutation rate.  Defaults to 1.0.
        v: Green->red mutation rate.  Defaults to 1.0.
        theta: Population mutation rate ``4*N*mu``. Defaults to 0.02.
        num_iter: Total proposed moves.  Defaults to 50000.
        burn_in: Iterations discarded before sampling (and before the
            adaptive kernel freezes).  Defaults to 10000.
        sample_freq: Thinning interval for post-burn-in samples.
            Defaults to 100.
        seed: Master RNG seed for reproducibility.  ``None`` draws from OS
            entropy.
        samples: Map taxon label -> number of sampled gene copies.  When
            ``None`` every taxon is assumed to have one sampled copy.
        max_reticulations: Cap on the number of reticulation nodes.
        max_level: Optional cap on network level; proposals exceeding it are
            rejected as part of the MH step.  ``None`` disables the cap.
        priors: Network prior hyperparameters.  Defaults to
            :class:`MCMC_GTPriors` defaults.
        start_net: Optional starting network (must be labelled with the
            alignment taxa).  When ``None`` an ultrametric caterpillar tree
            is built automatically.
        backbone: Network every accepted state must contain as a subgraph
            (``StartMode.AUGMENT``).  ``None`` leaves the chain
            unconstrained.

    Returns:
        dict[Network, float]: A single-entry mapping from the maximum a
        posteriori network to its log-posterior score.
    """
    from ._mcmc_gt import MCMC_GTPriors, MCMCGTKernel, _is_valid_network
    from ._search_flags import resolve_move_types

    # ── Taxa + sampling effort ───────────────────────────────────────────
    taxa = [rec.get_name() for rec in aln.get_records()]
    if samples is None:
        samples = {name: 1 for name in taxa}

    if priors is None:
        priors = MCMC_GTPriors()

    # ── Seeded RNGs (independent streams for kernel + accept/reject) ─────
    root_ss = np.random.SeedSequence(seed)
    kernel_ss, driver_ss = root_ss.spawn(2)
    kernel_rng = np.random.default_rng(kernel_ss)
    driver_rng = np.random.default_rng(driver_ss)

    # ── Starting model ───────────────────────────────────────────────────
    if start_net is None:
        start_net = _snp_starting_tree(taxa)
    require_branch_length_unit(
        start_net,
        BranchLengthUnit.SUBSTITUTIONS_PER_SITE,
        context="biallelic-marker MCMC",
    )

    model = Model(rng=np.random.default_rng(root_ss.spawn(1)[0]))
    model.network = start_net

    scorer = SNPScorer(aln, u, v, theta, samples, priors, posterior=True)
    model.set_likelihood_calculator(scorer)

    kernel = MCMCGTKernel(
        max_reticulations=max_reticulations,
        max_level=max_level,
        move_types=resolve_move_types(augment_only=backbone is not None),
        rng=kernel_rng,
    )
    required_clusters = (
        network_clusters(backbone) if backbone is not None else None
    )

    # ── Metropolis-Hastings loop (mirrors MCMC_GT._run_mh semantics) ─────
    cur_score = float(scorer(model))
    best_score = cur_score
    best_network = copy.deepcopy(model.network)

    freeze_fn = getattr(kernel, "freeze_adaptation", None)
    adaptation_frozen = False

    for iter_no in range(num_iter):
        if freeze_fn is not None and not adaptation_frozen \
                and iter_no >= burn_in:
            freeze_fn()
            adaptation_frozen = True

        move = kernel.generate(model)
        try:
            move.execute(model)
            # Reject structurally invalid / over-level proposals up front so
            # they count as rejections (preserving detailed balance) instead
            # of being silently floored by the scorer.
            if (
                not _is_valid_network(model.network)
                or (
                    max_level is not None
                    and network_level(model.network) > max_level
                )
                or (
                    required_clusters is not None
                    and not required_clusters
                    <= network_clusters(model.network)
                )
            ):
                move.undo(model)
                kernel.report_outcome(False, delta=0.0)
                continue
            prop_score = float(scorer(model))
        except Exception:
            try:
                move.undo(model)
            except Exception:
                pass
            kernel.report_outcome(False, delta=0.0)
            continue

        delta = prop_score - cur_score
        log_alpha = delta + move.log_hastings_ratio()
        accept = (log_alpha >= 0.0) or (math.log(driver_rng.random()) < log_alpha)
        if accept:
            kernel.report_outcome(True, delta=delta)
            cur_score = prop_score
            if prop_score > best_score:
                best_score = prop_score
                best_network = copy.deepcopy(model.network)
        else:
            try:
                move.undo(model)
            except Exception:
                pass
            kernel.report_outcome(False, delta=delta)

    return {best_network: best_score}

def _snp_log_likelihood(net: Network,
                        aln: MSA,
                        u: float,
                        v: float,
                        theta: float,
                        samples: dict[str, int],
                        *,
                        branch_thetas: Optional[dict] = None,
                        max_workers: int = 8,
                        sequential: bool = True,
                        verbose: bool = False) -> float:
    """Biallelic-marker log-likelihood of ``net`` given an in-memory alignment.

    The shared numerical core behind ``score(net, BiallelicMarkers(...),
    criterion=Likelihood())`` and the :class:`SNPScorer` that drives
    :func:`_snp_mcmc`.  It builds the SNP model graph from ``net`` and
    ``aln``, sizes the Bryant Q matrix for the true maximum lineage count
    (accounting for reticulation lineage duplication), auto-routes to the GPU
    when the network is large enough, batches over sites when the peak VPI
    tensor would not fit in memory, and returns ``sum_sites log P(site | net)``.

    Args:
        net: The species network to score (with branch lengths + gammas).
        aln: The biallelic alignment (leaf red-allele counts per site).
        u: Red->green mutation rate.
        v: Green->red mutation rate.
        theta: Population mutation rate ``4*N*mu``.
        branch_thetas: Fixed per-population theta overrides.
        samples: Map taxon label -> number of sampled gene copies.
        max_workers: Worker count for the (optional) parallel site loop.
        sequential: Reserved for the parallel site loop.
        verbose: When ``True`` print the per-call diagnostics (device, peak
            tensor, timing).  MCMC leaves this ``False`` to avoid per-iteration
            spam.

    Returns:
        The total log-likelihood (a negative float; ``-inf`` if a site has
        zero probability under ``net``).
    """
    require_branch_length_unit(
        net,
        BranchLengthUnit.SUBSTITUTIONS_PER_SITE,
        context="biallelic-marker likelihood",
    )
    snp_model = build_snp_model(net, aln)

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

    if needs_gpu and not GPU_SPECS.available and verbose:
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

    if verbose:
        peak_bytes, peak_shape = _estimate_peak_vpi_memory(
            snp_model, samples, min(batch_size, n_sites)
        )
        peak_mb = peak_bytes / (1024 * 1024)
        device_str = f"GPU ({GPU_SPECS.name})" if use_gpu else "CPU"
        print(f"SNP likelihood: {n_taxa} taxa, level-{level}, "
              f"max_lineages={max_n}, {n_sites} sites")
        batch_info = (f" ({n_batches} batches of {batch_size})"
                      if n_batches > 1 else "")
        print(f"  Device: {device_str} | "
              f"Peak tensor: {' × '.join(str(s) for s in peak_shape)} "
              f"({peak_mb:.1f} MB){batch_info}")

    # ── Build shared objects ────────────────────────────────────────────
    root = net.root()
    root_theta = resolve_branch_theta(
        theta,
        branch_thetas,
        root,
    )
    q = _get_transition(max_n, u, v, root_theta)

    for leaf in snp_model.nodetypes["leaf"]:
        assert(type(leaf) is LeafNode)
        leaf.samples = samples[leaf.get_name()]

    def _run_batch(site_slice: tuple[int, int] | None) -> float:
        """Run one site batch and return its log-likelihood contribution."""
        strategy = SNPStrategy(q, u, v, theta, n_sites, max_n,
                               site_slice=site_slice,
                               use_gpu=use_gpu,
                               branch_thetas=branch_thetas)
        visitor = SNPModelVisitor(strategy)
        try:
            for node in postorder(snp_model.root):
                visitor.visit(node)
            return strategy.L
        except Exception as exc:
            if (
                use_gpu
                and cp is not None
                and isinstance(exc, cp.cuda.memory.OutOfMemoryError)
            ):
                raise SNPResourceError(
                    "GPU memory was exhausted while evaluating a marker "
                    f"batch of {n_sites if site_slice is None else site_slice[1] - site_slice[0]} "
                    "sites. Reduce the site batch or use hardware with more "
                    "free VRAM."
                ) from exc
            raise
        finally:
            # Model nodes and strategy caches otherwise retain every batch's
            # device tensors until the full likelihood call returns. Release
            # those references before asking CuPy to return unused blocks.
            visitor.vpis.clear()
            strategy.cache.clear()
            for nodes in snp_model.nodetypes.values():
                for node in nodes:
                    node.vpi = None
            if use_gpu and cp is not None:
                try:
                    cp.cuda.Stream.null.synchronize()
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    # A CUDA OOM can leave cleanup calls failing too; preserve
                    # the actionable SNPResourceError raised above.
                    pass

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
            if verbose and (n_batches <= 20 or b % max(1, n_batches // 10) == 0):
                print(f"    Batch {b+1}/{n_batches}: sites [{s_start}:{s_end}] "
                      f"log-lik={batch_lik:.4f}")

    if verbose:
        end_t = time.perf_counter()
        print(f"  Total time: {end_t - start_t:.3f}s | "
              f"log-lik = {total_log_lik:.6f}")

    return total_log_lik


#########################
### Transition Matrix ###
#########################

@lru_cache(maxsize=128)
def _get_transition(n: int, u: float, v: float, theta: float) -> "BiMarkersTransition":
    """Return a cached :class:`BiMarkersTransition` for ``(n, u, v, theta)``.

    The Q matrix and its (lazily-built) eigendecomposition depend only on the
    rate parameters and the state-space size, all of which are constant for a
    given MCMC chain except ``n`` (which takes only a handful of distinct
    values as the network's lineage count changes).  Caching the transition
    object -- and therefore its one-time spectral decomposition -- across
    scorer calls turns every subsequent ``e^{Qt}`` on the chain into a pair of
    matmuls, instead of rebuilding Q and re-running Padé from scratch each
    iteration.
    """
    return BiMarkersTransition(n, u, v, theta)


class BiMarkersTransition:
    """
    Class that encodes the probabilities of transitioning from one (n,r) pair 
    to another under a Biallelic model.

    Includes method for efficiently computing e^Qt

    Inputs:
    1) n-- the total number of samples in the species tree
    2) u-- the probability of going from the red allele to the green one
    3) v-- the probability of going from the green allele to the red one
    4) theta-- population mutation rate ``4*N*mu``

    Assumption: Matrix indexes start with n=1, r=0, so Q[0][0] is Q(1,0);(1,0)

    Q Matrix is given by Equation 15 from:

    David Bryant, Remco Bouckaert, Joseph Felsenstein, Noah A. Rosenberg, 
    Arindam RoyChoudhury, Inferring Species Trees Directly from Biallelic 
    Genetic Markers: Bypassing Gene Trees in a Full Coalescent Analysis, 
    Molecular Biology and Evolution, Volume 29, Issue 8, August 2012, 
    Pages 1917–1932, https://doi.org/10.1093/molbev/mss086
    """

    def __init__(self, n : int, u : float, v : float, theta : float) -> None:
        """
        Initialize the Q matrix

        Args:
            n (int): sample count
            u (float): Mutation rate from red to green.
            v (float): Mutation rate from green to red.
            theta (float): Population mutation rate ``4*N*mu``.
        Returns:
            N/A
        """

        # Build Q matrix
        self.n = n 
        self.u = u
        self.v = v
        self.theta = theta

        # Lazily-built spectral decomposition Q = V diag(w) V^{-1}, used to
        # evaluate e^{Qt} for arbitrary t as V diag(e^{wt}) V^{-1} in two
        # matmuls instead of a fresh scaling-and-squaring Padé approximation
        # per branch.  ``None`` until the first :meth:`expt` call; set to
        # ``False`` if the decomposition is rejected as numerically unsafe
        # (then :meth:`expt` falls back to :func:`scipy.linalg.expm`).
        self._eig = None

        # State space includes the empty state (0, 0) at index 0, which is
        # absorbing (no lineages -> no mutation, no coalescence).  We never
        # iterate it below, so its Q row/column stay zero and ``expm`` maps it
        # to itself with probability 1 -- an empty branch stays empty.
        rows = state_dim(self.n)
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
                self.Q[n_r][n_r] = - (n_prime * (n_prime - 1) / theta) \
                                       - (v * (n_prime - r_prime)) \
                                       - (r_prime * u)

                # These equations only make sense if r isn't 0 
                # (and the second, if n isn't 1).
                if 0 < r_prime <= n_prime:
                    if n_prime > 1:
                        self.Q[n_r][nm_rm] = (r_prime - 1) * n_prime / theta
                    self.Q[n_r][n_rm] = (n_prime - r_prime + 1) * v

                # These equations only make sense if r is strictly less than n 
                # (and the second, if n is not 1).
                if 0 <= r_prime < n_prime:
                    if n_prime > 1:
                        self.Q[n_r][nm_r] = (n_prime - 1 - r_prime) \
                                            * n_prime / theta
                    self.Q[n_r][n_rp] = (r_prime + 1) * u

    def _build_eig(self) -> None:
        """Diagonalise ``Q`` once and validate against a Padé reference.

        The biallelic rate matrix is diagonalisable with a well-conditioned
        eigenbasis (condition number grows only polynomially with the state
        dimension), so ``e^{Qt}`` can be reconstructed to ~1e-12 accuracy.
        We verify this on a spread of representative ``t`` values against
        :func:`scipy.linalg.expm`; if the round-trip error is ever too large
        (a degenerate / defective ``Q``), we mark the decomposition unusable
        and permanently fall back to the exact Padé path.
        """
        try:
            w, V = np.linalg.eig(self.Q)
            Vinv = np.linalg.inv(V)
        except np.linalg.LinAlgError:
            self._eig = False
            return
        # Validate: max abs error vs expm across representative branch times.
        for t in (0.001, 0.01, 0.1, 1.0, 5.0):
            approx = np.real((V * np.exp(w * t)) @ Vinv)
            if not np.allclose(approx, expm(self.Q * t), atol=1e-9, rtol=1e-7):
                self._eig = False
                return
        self._eig = (w, V, Vinv)

    def expt(self, t : float = 1) -> np.ndarray:
        """
        Compute e^(Q*t) efficiently.
        
        Args:
            t (float): time, generally in coalescent units. Optional, defaults 
                       to 1, in which case e^Q is computed.
        
        Returns:
            np.ndarray: e^(Q*t).
        """
        if self._eig is None:
            self._build_eig()
        if self._eig is False:
            return expm(self.Q * t)
        w, V, Vinv = self._eig
        return np.real((V * np.exp(w * t)) @ Vinv)

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

class SNPStrategy:
    """
    Per-node partial-likelihood computation for the SNP model.
    
    Supports both CPU (NumPy) and GPU (CuPy) execution. When use_gpu=True,
    all tensor operations run on the GPU via CuPy's drop-in NumPy API.
    """
    def __init__(self, q : BiMarkersTransition, u : float, v : float,
                 theta : float, sites : int, max_samples : int,
                 site_slice: tuple[int, int] | None = None,
                 use_gpu: bool = False,
                 branch_thetas: Optional[dict] = None) -> None:
        """Create a new ``SNPStrategy``.

        Args:
            q: Transition-rate model for the 0/1 substitution process.
            u: Mutation rate red-to-green (1->0).
            v: Mutation rate green-to-red (0->1).
            theta: Default population mutation rate ``4*N*mu``.
            sites: Total number of sites in the alignment.
            max_samples: Maximum per-taxon sample count, used to size the
                per-node partial-likelihood vectors.
            site_slice: Optional ``(start, stop)`` restricting computation
                to a contiguous subrange of sites (for chunked/parallel
                evaluation). ``None`` uses all ``sites``.
            use_gpu: When ``True``, run tensor operations on the GPU via
                CuPy instead of NumPy.
            branch_thetas: Fixed branch-specific population mutation rates.
        """
        self.q : BiMarkersTransition = q
        self.u : float = u
        self.v : float = v
        self.theta : float = theta
        self.branch_thetas = branch_thetas
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
    
    def _rule1(
        self,
        F,
        branch_len: float,
        d: int,
        *,
        child_name: str,
        parent_name: str,
    ):
        """
        Given vpi tensor F, with interface α, and branch length t, and max lineages m_α, 
        compute the vpi tensor F_top at the top of the branch.

        P(t) is computed on CPU (small matrix, scipy.linalg.expm) and transferred
        to GPU if needed. The matmul F @ P_sub runs on the tensor's device.
        """
        branch_theta = resolve_branch_theta(
            self.theta,
            self.branch_thetas,
            child_name,
            parent=parent_name,
        )
        q = _get_transition(self.q.n, self.u, self.v, branch_theta)
        P_full = q.expt(branch_len)   # always NumPy (CPU)
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
        F = self._rule1(
            F,
            n.branch().length,
            state_dim(n.samples),
            child_name=n.get_name(),
            parent_name=n.branch().parent_id,
        )
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
        F = self._rule1(
            F,
            n.branch().length,
            state_dim(max_lin[-1]),
            child_name=n.get_name(),
            parent_name=n.branch().parent_id,
        )
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
        F = self._rule1(
            F,
            branches[0].length,
            state_dim(m),
            child_name=n.get_name(),
            parent_name=branches[0].parent_id,
        )
        F, log_scale = self._rescale_global(F, log_scale)
        
        # Apply transition on branch 1's axis (now at -2, move to -1)
        F = xp.moveaxis(F, -1, -2)
        F = self._rule1(
            F,
            branches[1].length,
            state_dim(m),
            child_name=n.get_name(),
            parent_name=branches[1].parent_id,
        )
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
        
        # Root allele-frequency prior (Bryant et al. 2012, eq. 20): the site
        # likelihood is  L = sum_{n,r} x_root[n, r] * pi[n, r]  where ``pi`` is
        # the *stationary distribution of the coalescent-with-mutation rate
        # matrix Q itself* -- NOT a plain binomial with base frequencies.  Q
        # couples mutation and coalescence, so its equilibrium over the (n, r)
        # states is obtained from the (right) null space of Q, normalised so
        # the single-lineage block ``{(1,0), (1,1)}`` sums to 1 (i.e. one
        # ancestral lineage is red with probability ``theta_r``).  This matches
        # the validated reference implementation and Rabier's published tables.
        #
        # We work on the ``n >= 1`` sub-block: the empty state ``(0, 0)`` is
        # absorbing, so it contributes a spurious extra null vector and is
        # excluded here (the root population always has >= 1 lineage).
        pi_np = _root_stationary(self.q.getQ(), m)
        
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
        
class SNPModelVisitor:
    """
    Drives a :class:`SNPStrategy` over the SNP model graph, threading each
    node's partial-likelihood result up to its parent.
    """
    def __init__(self, strategy: SNPStrategy) -> None:
        """Create a new visitor driving ``strategy`` over the model graph.

        Args:
            strategy: Per-node partial-likelihood strategy to invoke at
                each visited node.
        """
        self.strategy : SNPStrategy = strategy
        self.vpis : list[NodeVPI] = []
            
    def visit_leaf(self, n: LeafNode) -> None:
        """Compute and record the leaf partial likelihood for ``n``."""
        self.vpis.append(self.strategy.compute_at_leaf(n))
        
    def visit_internal(self, n: InternalNode) -> None:
        """Combine both children's partials at internal node ``n`` and record the result."""
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
        """Split the child partial across both parent interfaces of reticulation ``n``."""
        child_vpi : NodeVPI = [self._get_vpi_for(n, child.get_name()) for child in n.get_model_children()][0]
        self._remove(child_vpi)
        self.vpis.append(self.strategy.compute_at_reticulation(n, child_vpi))
        
    def visit_root(self, n: RootNode) -> None:
        """Combine both children's partials at the root ``n`` and record the result."""
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
        """Fold the child partial into the final log-likelihood at aggregator ``n``."""
        child_vpi : NodeVPI = [self._get_vpi_for(n, child.get_name()) for child in n.get_model_children()][0]
        self.strategy.compute_at_aggregator(n, child_vpi)
        
    
    def visit(self, n: ModelNode) -> None:
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
            """Find the ``(interface_index, vpi)`` pair matching ``node_label``."""
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
            """Move the interface at ``index`` to the last axis of ``v``'s tensor, in place."""
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
    
    