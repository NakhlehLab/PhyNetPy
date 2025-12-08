#! /usr/bin/env python
# -*- coding: utf-8 -*-

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

""" 
Author : Mark Kessler
Last Stable Edit : 3/11/25
First Included in Version : 1.0.0
Approved to Release Date : No
"""

from PhyNetPy.GTR import *
import numpy as np
try:
    import cupy as cp  # GPU backend
    HAS_CUPY = True
except Exception:
    cp = None
    HAS_CUPY = False

from PhyNetPy.GraphUtils import sample_displayed_tree, topological_order

# Optional PyTorch backend (MPS on Apple Silicon)
try:
    import torch as th
    HAS_TORCH = True
except Exception:
    th = None
    HAS_TORCH = False


class SeqSim:
        """
        Class that simulates the evolution of DNA sequences
        """
        
        def __init__(self, submodel : GTR = JC()) -> None:
                """
                Initialize the simulator with a substitution model

                Args:
                        submodel (GTR, optional): A substitution model. 
                                                  Defaults to JC().
                """
                self.sub = submodel

        def modify_seq(self, seq : list) -> list:
                """
                Modify a sequence of DNA letters according to the substitution 
                model

                Args:
                    seq (list): _description_

                Returns:
                    list: _description_
                """
                func = np.vectorize(self.dna_evolve)
                return func(seq)

        def dna_evolve(self, letter : str) -> str:
                """
                Simulate the evolution of a DNA sequence 

                Args:
                    letter (str): A DNA letter

                Returns:
                    str: A new DNA letter
                """
                alphabet = ['A', 'C', 'G', 'T']
                probs = self.transition[alphabet.index(letter)]
                new_letter = np.random.choice(alphabet, 1, p = probs)
                return new_letter[0]

        def change_transition(self, t : float) -> None:
                """
                Change the transition matrix to simulate evolution at a different
                rate

                Args:
                    t (float): the time at which to simulate evolution
                Returns:
                    N/A
                """
                self.transition = self.sub.expt(t)


# ------------------------------
# GPU-accelerated simulation API
# ------------------------------

def _cdf_from_probs(arr):
        """
        Build cumulative distributions along last axis for numpy/cupy arrays.
        """
        if HAS_CUPY and isinstance(arr, cp.ndarray):
                return cp.cumsum(arr, axis=-1)
        return np.cumsum(arr, axis=-1)


def _random_uniform(n, like):
        """
        Generate n uniforms on same device as 'like'.
        """
        if HAS_CUPY and isinstance(like, cp.ndarray):
                return cp.random.random(n, dtype=like.dtype)
        return np.random.random(n).astype(like.dtype, copy=False)


def _to_device(a, prefer_gpu: bool):
        if prefer_gpu and HAS_CUPY:
                return cp.asarray(a)
        return np.asarray(a)


def _to_cpu(a):
        if HAS_CUPY and isinstance(a, cp.ndarray):
                return cp.asnumpy(a)
        return a


def _evolve_block(parent_states, P_cdf):
        """
        Evolve a block of iid sites given parent state codes and a CDF matrix.

        parent_states: (S,) uint8/int array (np or cp) with values in {0,1,2,3}
        P_cdf: (4,4) cumulative probabilities (np or cp)
        returns child_states: (S,) uint8 array on same device
        """
        # Gather per-site row CDFs
        rows = P_cdf[parent_states]
        u = _random_uniform(rows.shape[0], rows)
        # Compare u against row CDFs and take first True
        # Ensure last column is 1.0 to avoid edge cases
        if HAS_CUPY and isinstance(rows, cp.ndarray):
                rows = cp.concatenate([rows[:, :3], cp.ones((rows.shape[0], 1), dtype=rows.dtype)], axis=1)
                mask = (u[:, None] <= rows)
                idx = mask.argmax(axis=1)
                return idx.astype(cp.uint8, copy=False)
        else:
                rows = np.concatenate([rows[:, :3], np.ones((rows.shape[0], 1), dtype=rows.dtype)], axis=1)
                mask = (u[:, None] <= rows)
                idx = mask.argmax(axis=1)
                return idx.astype(np.uint8, copy=False)


def _sample_root_states(num_sites, base_freqs, prefer_gpu: bool):
        """
        Sample root states according to base frequencies.
        Returns device array (np/cp) of shape (num_sites,) dtype uint8
        """
        probs = _to_device(np.asarray(base_freqs, dtype=np.float64), prefer_gpu)
        cdf = _cdf_from_probs(probs)
        u = _random_uniform(num_sites, cdf)
        if HAS_CUPY and isinstance(cdf, cp.ndarray):
                cdf = cp.concatenate([cdf[:3], cp.ones((1,), dtype=cdf.dtype)])
                # broadcast search: build (num_sites, 4) CDF rows identical
                rows = cp.broadcast_to(cdf[None, :], (num_sites, 4))
                mask = (u[:, None] <= rows)
                idx = mask.argmax(axis=1)
                return idx.astype(cp.uint8)
        else:
                cdf = np.concatenate([cdf[:3], np.ones((1,), dtype=cdf.dtype)])
                rows = np.broadcast_to(cdf[None, :], (num_sites, 4))
                mask = (u[:, None] <= rows)
                idx = mask.argmax(axis=1)
                return idx.astype(np.uint8)


def _edge_P_cdf(edge, model, cache, rate_scale=1.0, prefer_gpu: bool = True):
        """
        Get per-edge cumulative transition matrix (np/cp) with caching.
        Cache by (edge_id, rate_scale) on CPU, convert to device lazily.
        """
        key = (id(edge), float(rate_scale))
        if key not in cache:
                t = edge.get_length() * rate_scale
                P = model.expt(float(t))  # CPU numpy
                cache[key] = P.astype(np.float64, copy=False)
        P_cpu = cache[key]
        P_dev = _to_device(P_cpu, prefer_gpu)
        return _cdf_from_probs(P_dev)


def simulate_alignment_on_tree(tree, length, model: GTR, device: str = "gpu", rates: list[float] | None = None, leaves_only: bool = True) -> dict:
        """
        Simulate an alignment of given length on a directed tree (Network without retics).

        - device: "gpu" uses CuPy if available, otherwise falls back to CPU.
        - rates: optional per-site rate multipliers; if provided, expects shape (length,) on CPU.
        - leaves_only: if True, return leaf sequences only.

        Returns a dict: name -> string sequence
        """
        use_gpu = device == "gpu" and HAS_CUPY
        use_mps = device == "mps" and HAS_TORCH and getattr(th.backends, "mps", None) is not None and th.backends.mps.is_available()
        if rates is not None and use_gpu:
                # Current GPU path does not implement per-site rate scaling
                raise NotImplementedError("GPU path with per-site rates not yet supported")
        if rates is not None and use_mps:
                # Not implemented for MPS path yet
                raise NotImplementedError("MPS path with per-site rates not yet supported")

        # Prepare base frequencies from model (assumes GTR-like with freqs)
        base_freqs = getattr(model, "freqs", [0.25, 0.25, 0.25, 0.25])

        # Topological order
        order = topological_order(tree)
        root = order[0]

        # Sample root states
        if use_mps:
                # Torch/MPS path
                device_t = th.device("mps")
                probs = th.tensor(base_freqs, dtype=th.float64, device=device_t)
                cdf = th.cumsum(probs, dim=0)
                cdf[-1] = 1.0
                u = th.rand((length,), dtype=th.float64, device=device_t)
                # Broadcast compare and argmax
                rows = cdf.unsqueeze(0).expand(length, -1)
                root_states_t = (u.unsqueeze(1) <= rows).to(th.int64).argmax(dim=1).to(th.uint8)
                node_states = {root: root_states_t}
                P_cache_cpu: dict = {}
                for u_node in order:
                        for e in tree.out_edges(u_node):
                                v = e.dest
                                parent = node_states[u_node]
                                key = (id(e), 1.0)
                                if key not in P_cache_cpu:
                                        P = model.expt(float(e.get_length()))
                                        P_cache_cpu[key] = P.astype(np.float64, copy=False)
                                P_t = th.tensor(P_cache_cpu[key], dtype=th.float64, device=device_t)
                                Pcdf_t = th.cumsum(P_t, dim=1)
                                Pcdf_t[:, -1] = 1.0
                                rows = Pcdf_t[parent]
                                u2 = th.rand((rows.shape[0],), dtype=th.float64, device=device_t)
                                child = (u2.unsqueeze(1) <= rows).to(th.int64).argmax(dim=1).to(th.uint8)
                                node_states[v] = child
                leaves = tree.get_leaves()
                code_to_char = np.array(["A", "C", "G", "T"], dtype=object)
                result = {}
                for leaf in leaves:
                        codes = node_states[leaf].cpu().numpy()
                        seq = "".join(code_to_char[codes])
                        result[leaf.label] = seq
                if not leaves_only:
                        for node, arr in node_states.items():
                                if node in leaves:
                                        continue
                                codes = arr.cpu().numpy()
                                seq = "".join(code_to_char[codes])
                                result[node.label] = seq
                return result
        else:
                root_states = _sample_root_states(length, base_freqs, use_gpu)

        node_states = {root: root_states}
        P_cache_cpu: dict = {}

        for u in order:
                for e in tree.out_edges(u):
                        v = e.dest
                        parent = node_states[u]
                        if rates is None:
                                Pcdf = _edge_P_cdf(e, model, P_cache_cpu, 1.0, use_gpu)
                                child = _evolve_block(parent, Pcdf)
                        else:
                                # Per-site rates: evolve in blocks of equal rates to reuse P
                                # Fallback simple implementation: per unique rate
                                unique_rates = np.unique(rates)
                                like = parent
                                if HAS_CUPY and use_gpu and isinstance(parent, cp.ndarray):
                                        like = parent
                                child = None
                                for r in unique_rates:
                                        idx = np.nonzero(np.asarray(rates) == r)[0]
                                        if idx.size == 0:
                                                continue
                                        Pcdf = _edge_P_cdf(e, model, P_cache_cpu, float(r))
                                        sub_parent = parent[idx]
                                        sub_child = _evolve_block(sub_parent, Pcdf)
                                        if child is None:
                                                if HAS_CUPY and use_gpu and isinstance(sub_child, cp.ndarray):
                                                        child = cp.empty_like(parent)
                                                else:
                                                        child = np.empty_like(parent)
                                        if HAS_CUPY and use_gpu and isinstance(child, cp.ndarray):
                                                child[idx] = sub_child
                                        else:
                                                child[idx] = _to_cpu(sub_child)
                        node_states[v] = child

        # Collect leaves
        leaves = tree.get_leaves()
        code_to_char = np.array(["A", "C", "G", "T"], dtype=object)
        result = {}
        for leaf in leaves:
                codes = _to_cpu(node_states[leaf])
                seq = "".join(code_to_char[codes])
                result[leaf.label] = seq
        if not leaves_only:
                # include internal nodes too
                for node, arr in node_states.items():
                        if node in leaves:
                                continue
                        codes = _to_cpu(arr)
                        seq = "".join(code_to_char[codes])
                        result[node.label] = seq
        return result


def simulate_alignment_on_network(net, length, model: GTR, device: str = "gpu", mode: str = "per_locus", rng: np.random.Generator | None = None) -> dict:
        """
        Simulate on a network by sampling a displayed tree, then simulating on it.

        - mode: currently only "per_locus" supported (single displayed tree for all sites).
                A "per_site" mode would resample parent edges at retics per-site.
        """
        if mode != "per_locus":
            raise ValueError("Only per_locus mode is implemented in this version")
        tree = sample_displayed_tree(net, rng=rng)
        return simulate_alignment_on_tree(tree, length, model, device=device)
