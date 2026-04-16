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
Last Stable Edit : 3/26/26
First Included in Version : 0.3.2
 
Docs   - [ ]
Tests  - [ ]
Design - [x]
 
Maximum pseudo-likelihood (MPL) for phylogenetic network inference.
 
Implements the scoring function from:
    Yu, Y. & Nakhleh, L. (2015). "A maximum pseudo-likelihood approach
    for phylogenetic networks." BMC Genomics, 16(S10), S10.
 
    log L(Psi, gamma | G)
        = sum over {X,Y,Z} of
          [ rho(XY|Z) * log P(XY|Z | Psi, gamma)
          + rho(XZ|Y) * log P(XZ|Y | Psi, gamma)
          + rho(YZ|X) * log P(YZ|X | Psi, gamma) ]
"""
 
from __future__ import annotations

import copy
import math
import os
import random
from collections import deque
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np

from .Network import Network, Node
from .GeneTrees import GeneTrees
from . import IO as io
from .ModelGraph import Model
from .MetropolisHastings import ProposalKernel, HillClimbing, SimulatedAnnealing
from .ModelMove import *
 
_LOG_FLOOR = math.log(1e-200)

try:
    from .cython.mpl_engine_cy import score_all_triplets as _cy_score
    _HAS_CYTHON_MPL = True
except ImportError:
    _HAS_CYTHON_MPL = False
 
 
class GeneTreeTripletResult:
    """
    Container for gene-tree triplet frequencies (rho values).

    This object is returned by compute_gene_tree_triplets and then consumed
    by score_species_network_triplets.
    """
    def __init__(self, 
                 triplets: list[tuple[str, str, str]],
                 rho_by_triplet: dict[tuple[str, str, str], tuple[float, float, float]]) -> None:
        """
        Initialize container values.

        Args:
            triplets (list[tuple[str, str, str]]): Canonical species triplets.
            rho_by_triplet (dict[tuple[str, str, str], tuple[float, float, float]]): 
                            Mapping from triplet -> (rho_xy_z, rho_xz_y, rho_yz_x).
        Returns:
            N/A
        """
        self.triplets : list[tuple[str, str, str]] = triplets
        self.rho_by_triplet : dict[tuple[str, str, str], tuple[float, float, float]] = rho_by_triplet


class SpeciesNetworkTripletResult:
    """
    Container for species-network triplet probabilities and final MPL score.
    """
    def __init__(self, 
                 triplets: list[tuple[str, str, str]],
                 probs_by_triplet: dict[tuple[str, str, str], tuple[float, float, float]],
                 log_pseudo_likelihood: float) -> None:
        """
        Initialize container values.

        Args:
            triplets (list[tuple[str, str, str]]): Canonical species triplets.
            probs_by_triplet (dict[tuple[str, str, str], tuple[float, float, float]]): 
                            Mapping from triplet -> (P(XY|Z), P(XZ|Y), P(YZ|X)).
            log_pseudo_likelihood (float): Final pseudo-likelihood score.
        Returns:
            N/A
        """
        self.triplets : list[tuple[str, str, str]] = triplets
        self.probs_by_triplet : dict[tuple[str, str, str], tuple[float, float, float]] = probs_by_triplet
        self.log_pseudo_likelihood : float = log_pseudo_likelihood


def _subtree_leaf_labels(net: Network, node: Node, visited: set[Node] | None = None) -> set[str]:
    """Leaves reachable from *node* following child edges (for comparison reports)."""
    if visited is None:
        visited = set()
    if node in visited:
        return set()
    visited.add(node)
    if net.out_degree(node) == 0:
        return {node.label}
    out: set[str] = set()
    for c in net.get_children(node):
        out |= _subtree_leaf_labels(net, c, visited)
    return out


def _nontrivial_clades(net: Network) -> set[frozenset[str]]:
    """All multisets of leaf labels induced by non-leaf nodes (for clade comparison)."""
    clades: set[frozenset[str]] = set()
    for node in net.V():
        if net.out_degree(node) == 0:
            continue
        leaves = _subtree_leaf_labels(net, node)
        if len(leaves) > 1:
            clades.add(frozenset(leaves))
    return clades


def format_mpl_reference_comparison(
    found: Network,
    reference: Network,
    rho: dict[tuple[str, str, str], tuple[float, float, float]],
    triplets: list[tuple[str, str, str]],
    *,
    top_k: int = 25,
) -> str:
    """Build a text report: scores, retic summaries, clade diff, top triplet LL gaps.

    Uses the same rho and triplet list as :class:`MPL` (gene-tree statistics).
    """
    scorer = MPLScorer(rho, triplets)
    m_found = Model()
    m_found.network = found
    m_ref = Model()
    m_ref.network = reference
    s_found = scorer(m_found)
    s_ref = scorer(m_ref)
    gap = s_ref - s_found

    lines: list[str] = []
    lines.append("=== MPL reference comparison ===")
    lines.append(f"Found score:     {s_found:.4f}")
    lines.append(f"Reference score: {s_ref:.4f}")
    lines.append(f"Gap (ref - found): {gap:.4f}  (positive => reference is better)")
    lines.append("")

    def _retic_lines(net: Network, label: str) -> None:
        retics = [n for n in net.V() if net.in_degree(n) > 1]
        lines.append(f"{label} reticulations: {len(retics)}")
        for r in retics:
            sub = sorted(_subtree_leaf_labels(net, r))
            pars = [p.label for p in net.get_parents(r)]
            lines.append(f"  {r.label}: parents={pars}, subtree_leaves={sub}")

    _retic_lines(found, "Found")
    _retic_lines(reference, "Reference")
    lines.append("")

    cf = _nontrivial_clades(found)
    cr = _nontrivial_clades(reference)
    shared = cf & cr
    only_f = cf - cr
    only_r = cr - cf
    lines.append(f"Clades: shared={len(shared)}, only_found={len(only_f)}, only_reference={len(only_r)}")
    if only_r:
        lines.append("Clades only in reference (sample up to 12):")
        for c in sorted(only_r, key=len)[:12]:
            lines.append(f"  {sorted(c)}")
    if only_f:
        lines.append("Clades only in found (sample up to 12):")
        for c in sorted(only_f, key=len)[:12]:
            lines.append(f"  {sorted(c)}")
    lines.append("")

    eng_f = _TripleDPEngine(found)
    eng_r = _TripleDPEngine(reference)
    trip_list = [t for t in triplets if any(rho[t][i] > 0.0 for i in range(3))]
    diffs: list[tuple[tuple[str, str, str], float]] = []
    for trip in trip_list:
        x, y, z = trip
        r_xy, r_xz, r_yz = rho[trip]
        pf_xy = eng_f.calculate_triple_probability((x, y, z))
        pf_xz = eng_f.calculate_triple_probability((x, z, y))
        pf_yz = max(0.0, 1.0 - pf_xy - pf_xz)
        pr_xy = eng_r.calculate_triple_probability((x, y, z))
        pr_xz = eng_r.calculate_triple_probability((x, z, y))
        pr_yz = max(0.0, 1.0 - pr_xy - pr_xz)
        ll_f = ll_r = 0.0
        for rr, a, b in [(r_xy, pf_xy, pr_xy), (r_xz, pf_xz, pr_xz), (r_yz, pf_yz, pr_yz)]:
            if rr > 0:
                ll_f += rr * math.log(max(a, 1e-300))
                ll_r += rr * math.log(max(b, 1e-300))
        diffs.append((trip, ll_r - ll_f))
    diffs.sort(key=lambda x: x[1], reverse=True)
    lines.append(f"Top {top_k} triplets where reference beats found (largest LL gap):")
    lines.append(f"{'triplet':>28} {'gap':>10}")
    for trip, d in diffs[:top_k]:
        lines.append(f"{','.join(trip):>28} {d:>10.2f}")
    lines.append("=== end comparison ===")
    return "\n".join(lines)


def save_mpl_network_newick(net: Network, path: str | os.PathLike[str]) -> None:
    """Write *net* as a Newick string to *path* (UTF-8)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(net.newick() + "\n", encoding="utf-8")


###################################
#### RHO PRECOMPUTATION ###########
###################################

class _GeneTreeLCAIndex:
    """Precomputed LCA index for O(1) triplet topology queries on a gene tree.

    One BFS from the root builds depth and parent maps. Pairwise LCA depths
    for all relevant leaf pairs (allele labels) are then computed via
    parent-climbing and cached in a flat dict for instant lookup.
    """

    __slots__ = ('_pair_depth', '_leaf_labels')

    def __init__(self, tree: Network, relevant_labels: frozenset[str]) -> None:
        root = tree.root()
        depth: dict[Node, int] = {}
        parent: dict[Node, Node | None] = {}
        label_to_node: dict[str, Node] = {}

        # Single BFS to build depth + parent maps.
        q: deque[tuple[Node, Node | None, int]] = deque()
        q.append((root, None, 0))
        while q:
            node, par, d = q.popleft()
            depth[node] = d
            parent[node] = par
            for child in tree.get_children(node):
                q.append((child, node, d + 1))

        for leaf in tree.get_leaves():
            if leaf.label in relevant_labels:
                label_to_node[leaf.label] = leaf

        self._leaf_labels: frozenset[str] = frozenset(label_to_node)

        # Precompute pairwise LCA depths for all relevant leaves.
        sorted_labels = sorted(label_to_node)
        pair_depth: dict[tuple[str, str], int] = {}
        for i in range(len(sorted_labels)):
            a_label = sorted_labels[i]
            na_orig = label_to_node[a_label]
            da_orig = depth[na_orig]
            for j in range(i + 1, len(sorted_labels)):
                b_label = sorted_labels[j]
                na, nb = na_orig, label_to_node[b_label]
                da, db = da_orig, depth[nb]
                while da > db:
                    na = parent[na]
                    da -= 1
                while db > da:
                    nb = parent[nb]
                    db -= 1
                while na is not nb:
                    na = parent[na]
                    nb = parent[nb]
                    da -= 1
                pair_depth[(a_label, b_label)] = da
                pair_depth[(b_label, a_label)] = da

        self._pair_depth = pair_depth

    def induced_triple_fast(self, x: str, y: str, z: str) -> str:
        """Determine induced triple topology via 3 cached lookups."""
        pd = self._pair_depth
        d_xy = pd[(x, y)]
        d_xz = pd[(x, z)]
        d_yz = pd[(y, z)]

        if d_xy == d_xz == d_yz:
            return "star"
        max_d = d_xy
        if d_xz > max_d:
            max_d = d_xz
        if d_yz > max_d:
            max_d = d_yz
        if d_xy == max_d:
            return "xy|z"
        if d_xz == max_d:
            return "xz|y"
        return "yz|x"


def _process_tree_batch(batch: list[Network],
                        relevant: frozenset[str],
                        mapping: dict[str, list[str]],
                        triplets: list[tuple[str, str, str]],
                        identity_mapping: bool) -> dict[tuple[str, str, str], list[float]]:
    """Process a batch of gene trees, returning partial rho accumulators.

    Separated from the main function so it can be dispatched to a
    ``concurrent.futures`` worker.
    """
    partial: dict[tuple[str, str, str], list[float]] = {
        t: [0.0, 0.0, 0.0] for t in triplets
    }

    if identity_mapping:
        _ONE_THIRD = 1.0 / 3.0
        for tree in batch:
            idx = _GeneTreeLCAIndex(tree, relevant)
            present = idx._leaf_labels
            pd = idx._pair_depth

            for triplet in triplets:
                X, Y, Z = triplet
                if X not in present or Y not in present or Z not in present:
                    continue

                d_xy = pd[(X, Y)]
                d_xz = pd[(X, Z)]
                d_yz = pd[(Y, Z)]

                acc = partial[triplet]
                if d_xy == d_xz == d_yz:
                    acc[0] += _ONE_THIRD
                    acc[1] += _ONE_THIRD
                    acc[2] += _ONE_THIRD
                elif d_xy >= d_xz and d_xy >= d_yz:
                    acc[0] += 1.0
                elif d_xz >= d_yz:
                    acc[1] += 1.0
                else:
                    acc[2] += 1.0
    else:
        for tree in batch:
            idx = _GeneTreeLCAIndex(tree, relevant)
            present = idx._leaf_labels

            for triplet in triplets:
                X, Y, Z = triplet
                ax = [a for a in mapping[X] if a in present]
                ay = [a for a in mapping[Y] if a in present]
                az = [a for a in mapping[Z] if a in present]

                denom = len(ax) * len(ay) * len(az)
                if denom == 0:
                    continue

                cnt_xy, cnt_xz, cnt_yz = 0.0, 0.0, 0.0
                for xi in ax:
                    for yj in ay:
                        for zk in az:
                            topo = idx.induced_triple_fast(xi, yj, zk)
                            if topo == "xy|z":
                                cnt_xy += 1.0
                            elif topo == "xz|y":
                                cnt_xz += 1.0
                            elif topo == "yz|x":
                                cnt_yz += 1.0
                            else:
                                cnt_xy += 1.0 / 3.0
                                cnt_xz += 1.0 / 3.0
                                cnt_yz += 1.0 / 3.0

                inv = 1.0 / denom
                acc = partial[triplet]
                acc[0] += cnt_xy * inv
                acc[1] += cnt_xz * inv
                acc[2] += cnt_yz * inv

    return partial


def _compute_all_rhos_fast(gene_trees: GeneTrees,
                           mapping: dict[str, list[str]],
                           triplets: list[tuple[str, str, str]],
                           n_workers: int = 1) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    """Compute rho for every triplet in one pass over gene trees.

    Flips the loop order to tree-first so each gene tree is indexed
    once and reused across all 816+ triplets.  Uses _GeneTreeLCAIndex
    for O(1) topology queries instead of 4 × BFS per query.

    When *n_workers* > 1 the gene-tree list is split into batches and
    processed in parallel with ``concurrent.futures.ProcessPoolExecutor``.
    """
    relevant = frozenset(a for alleles in mapping.values() for a in alleles)

    identity_mapping = all(len(v) == 1 for v in mapping.values())

    trees = gene_trees.trees
    if n_workers <= 1 or len(trees) < 4:
        merged = _process_tree_batch(
            trees, relevant, mapping, triplets, identity_mapping
        )
        return {t: tuple(merged[t]) for t in triplets}

    # Split trees into batches for parallel processing.
    from concurrent.futures import ProcessPoolExecutor
    batch_size = max(1, (len(trees) + n_workers - 1) // n_workers)
    batches = [
        trees[i : i + batch_size]
        for i in range(0, len(trees), batch_size)
    ]

    futures = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for batch in batches:
            futures.append(
                executor.submit(
                    _process_tree_batch,
                    batch, relevant, mapping, triplets, identity_mapping,
                )
            )

    # Merge partial results.
    merged: dict[tuple[str, str, str], list[float]] = {
        t: [0.0, 0.0, 0.0] for t in triplets
    }
    for fut in futures:
        partial = fut.result()
        for t in triplets:
            p = partial[t]
            m = merged[t]
            m[0] += p[0]
            m[1] += p[1]
            m[2] += p[2]

    return {t: tuple(merged[t]) for t in triplets}


#####################################
#### SUBNETWORK PROBABILITY LOGIC ###
#####################################
 
class _TripleDPEngine:
    """
    Dynamic-programming engine for 3-taxon triplet probabilities.

    The implementation follows the same state-space approach used in
    PhyloNet's pseudo-likelihood code path. Each DP state encodes which
    lineages are currently present, and each configuration tracks reticulation
    choices made so far while moving from leaves up to the root.

    This class is intentionally low-level because it mirrors the mathematics.
    The surrounding helper methods in this module expose a simpler public API.
    """

    _MERGING_MAP = {
        (1, 2): 4,
        (1, 3): 5,
        (1, 6): 7,
        (2, 3): 6,
        (2, 5): 7,
        (3, 4): 7,
        (3, 8): 9,
    }

    _SPLITTING_MAP = {
        0: [(0, 0)],
        1: [(0, 1)],
        2: [(0, 2)],
        3: [(0, 3)],
        4: [(0, 4), (1, 2)],
        5: [(0, 5), (1, 3)],
        6: [(0, 6), (2, 3)],
        7: [(0, 7), (1, 6), (2, 5), (3, 4)],
        8: [(0, 8)],
        9: [(0, 9), (3, 8)],
        10: [(0, 10)],
    }

    # target_state -> (lineages_in, lineages_out) for gij branch transition
    _COALESCING_MAP = {
        0: [(0, (0, 0))],
        1: [(1, (1, 1))],
        2: [(2, (1, 1))],
        3: [(3, (1, 1))],
        4: [(4, (2, 2)), (8, (2, 1))],
        5: [(5, (2, 2))],
        6: [(6, (2, 2))],
        7: [(7, (3, 3)), (9, (3, 2)), (10, (3, 1))],
        8: [(8, (1, 1))],
        9: [(9, (2, 2)), (10, (2, 1))],
        10: [(10, (1, 1))],
    }

    class _Configuration:
        """One DP configuration using a tuple for the reticulation-choice vector.

        Tuples are hashable natively, cheaper to copy (no list alloc), and
        compare faster than lists.
        """

        __slots__ = ('total_prob', '_idx')

        def __init__(self,
                     net_node_num: int,
                     total_prob: float = 1.0,
                     net_node_index: tuple[int, ...] | None = None) -> None:
            self.total_prob = total_prob
            self._idx: tuple[int, ...] = (
                net_node_index if net_node_index is not None
                else (0,) * net_node_num
            )

        @property
        def net_node_index(self) -> tuple[int, ...]:
            return self._idx

        def copy(self) -> "_TripleDPEngine._Configuration":
            c = _TripleDPEngine._Configuration.__new__(_TripleDPEngine._Configuration)
            c.total_prob = self.total_prob
            c._idx = self._idx
            return c

        @classmethod
        def merge(cls,
                  c1: "_TripleDPEngine._Configuration",
                  c2: "_TripleDPEngine._Configuration") -> "_TripleDPEngine._Configuration":
            merged = tuple(
                max(a, b) for a, b in zip(c1._idx, c2._idx)
            )
            c = cls.__new__(cls)
            c.total_prob = max(0.0, c1.total_prob * c2.total_prob)
            c._idx = merged
            return c

        def is_compatible(self, other: "_TripleDPEngine._Configuration") -> bool:
            for mine, theirs in zip(self._idx, other._idx):
                if mine != theirs and mine != 0 and theirs != 0:
                    return False
            return True

        def add_choice(self, net_id: int, choice: int) -> None:
            lst = list(self._idx)
            lst[net_id] = choice
            self._idx = tuple(lst)

        def clear_choices(self) -> None:
            self._idx = (0,) * len(self._idx)

        def __hash__(self) -> int:
            return hash(self._idx)

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, _TripleDPEngine._Configuration):
                return False
            return self._idx == other._idx

    def __init__(self, network: Network) -> None:
        """
        Initialize the DP engine for one fixed species network.

        Pre-caches all topology info so the per-triplet DP never touches
        Network methods.

        Args:
            network (Network): Species network that will be queried for
                                triplet probabilities.
        Returns:
            N/A
        """
        self.network = network
        self.net_node_num = sum(
            1 for n in self.network.V() if self.network.in_degree(n) > 1
        )
        self.articulation_nodes: set[Node] = set()
        self.lowest_articulation_nodes: set[Node] = set()
        self._compute_articulation_nodes()
        self._topo_leaf_to_root: list[Node] = list(
            reversed(self.network.topological_order())
        )

        # --- Pre-cache per-node topology for hot DP loop ---
        self._node_is_leaf: dict[Node, bool] = {}
        self._node_is_retic: dict[Node, bool] = {}
        self._node_children: dict[Node, list[Node]] = {}
        self._node_parents: dict[Node, list[Node]] = {}
        self._node_label: dict[Node, str] = {}
        # edge key -> (branch_length, gamma)
        self._edge_info: dict[tuple[Node, Node], tuple[float, float]] = {}

        for node in self._topo_leaf_to_root:
            out_d = network.out_degree(node)
            in_d = network.in_degree(node)
            self._node_is_leaf[node] = (out_d == 0)
            self._node_is_retic[node] = (in_d > 1)
            children = network.get_children(node)
            self._node_children[node] = children
            parents = network.get_parents(node)
            self._node_parents[node] = parents
            self._node_label[node] = node.label

            for parent in parents:
                edge = network.get_edge(parent, node)
                bl = edge.get_length()
                gamma = edge.get_gamma()
                self._edge_info[(parent, node)] = (bl, gamma)

        # --- Memoize _gij for the distinct branch lengths in this network ---
        self._gij_cache: dict[tuple[float, int, int], float] = {}

    @staticmethod
    def _fact(start: int, end: int) -> float:
        """
        Compute multiplicative range start * ... * end.

        Args:
            start (int): Lower bound (inclusive).
            end (int): Upper bound (inclusive).
        Returns:
            float: Product over range.
        """
        result = 1.0
        for i in range(start, end + 1):
            result *= i
        return result

    @staticmethod
    def _gij_raw(length: float, i: int, j: int) -> float:
        """Compute coalescent transition probability (uncached)."""
        if length is None or length == -1:
            return 1.0 if j == 1 else 0.0
        if length == 0:
            return 1.0 if i == j else 0.0
        if i == 0:
            return 1.0

        _fact = _TripleDPEngine._fact
        result = 0.0
        for k in range(j, i + 1):
            temp = (
                math.exp(0.5 * k * (1.0 - k) * length)
                * (2.0 * k - 1.0)
                * ((-1.0) ** (k - j))
                * _fact(j, j + k - 2)
                * _fact(i - k + 1, i)
            )
            denom = _fact(1, j) * _fact(1, k - j) * _fact(i, i + k - 1)
            result += temp / denom
        return result

    def _gij(self, length: float, i: int, j: int) -> float:
        """Memoized coalescent transition probability."""
        key = (length, i, j)
        cached = self._gij_cache.get(key)
        if cached is not None:
            return cached
        val = self._gij_raw(length, i, j)
        self._gij_cache[key] = val
        return val

    def _is_valid_network(self, ignore_node: Node) -> bool:
        """
        Check structural validity after temporarily removing an edge.

        Args:
            ignore_node (Node): Node exempted from degree-2 check.
        Returns:
            bool: True if network remains structurally valid.
        """
        roots = self.network.roots()
        if not roots:
            return False

        visited: set[Node] = set()
        queue: list[Node] = [roots[0]]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            queue.extend(self.network.get_children(cur))

        referenced: set[Node] = set()
        for node in visited:
            if (
                self.network.in_degree(node) == 1
                and self.network.out_degree(node) == 1
                and node != ignore_node
            ):
                return False
            for parent in self.network.get_parents(node):
                referenced.add(parent)
            for child in self.network.get_children(node):
                referenced.add(child)
        return len(visited) == len(referenced)

    def _compute_articulation_nodes(self) -> None:
        """
        Identify articulation and lowest-articulation nodes in the network.

        These sets are used to compress compatible configurations during DP,
        which keeps state growth manageable and matches PhyloNet semantics.

        Returns:
            N/A
        """
        self.articulation_nodes.clear()
        self.lowest_articulation_nodes.clear()

        for node in reversed(self.network.topological_order()):
            is_leaf = self.network.out_degree(node) == 0
            is_root = node == self.network.root()
            is_tree_node = self.network.in_degree(node) <= 1 and not is_leaf

            if is_leaf:
                self.articulation_nodes.add(node)
                continue

            all_children_articulate = all(
                child in self.articulation_nodes
                for child in self.network.get_children(node)
            )

            if is_root:
                if not all_children_articulate:
                    self.lowest_articulation_nodes.add(node)
                self.articulation_nodes.add(node)
                continue

            if is_tree_node:
                if all_children_articulate:
                    self.articulation_nodes.add(node)
                else:
                    parent = self.network.get_parents(node)[0]
                    edge = self.network.get_edge(parent, node)
                    self.network.remove_edge(edge)
                    disconnect = self._is_valid_network(parent)
                    self.network.add_edges(edge)
                    if disconnect:
                        self.lowest_articulation_nodes.add(node)
                        self.articulation_nodes.add(node)

    def _compute_ac_minus(self,
                          cacs: dict[int, list["_TripleDPEngine._Configuration"]],
                          branch_length: float,
                          inheritance_prob: float) -> dict[int, list["_TripleDPEngine._Configuration"]]:
        """
        Propagate DP configurations through one branch upward.

        This is the AC+ -> AC- transition in PhyloNet notation.

        Args:
            cacs (dict[int, list[_Configuration]]): State/config map before
                        branch transition.
            branch_length (float): Length of the current branch.
            inheritance_prob (float): Inheritance gamma for retic edges
                        (1.0 for tree edges).
        Returns:
            dict[int, list[_Configuration]]: New state/config map after branch.
        """
        ac_minus_map: dict[
            int,
            dict[
                _TripleDPEngine._Configuration,
                _TripleDPEngine._Configuration,
            ],
        ] = {}

        for state_id, configs in cacs.items():
            for target_state, (lineages_in, lineages_out) in self._COALESCING_MAP[state_id]:
                # Start with inheritance contribution; for tree edges this is 1.
                prob = 1.0
                if inheritance_prob != 1.0:
                    prob = inheritance_prob ** lineages_in

                # Apply coalescent reduction probability on this branch.
                if lineages_in > 1:
                    prob *= self._gij(branch_length, lineages_in, lineages_out)
                    if lineages_in == 3 and lineages_out != 3:
                        prob = prob / 3.0

                if prob == 0:
                    continue

                # Aggregate equal configurations by summing their probabilities.
                state_map = ac_minus_map.setdefault(target_state, {})
                for config in configs:
                    copy = config.copy()
                    new_prob = max(0.0, copy.total_prob * prob)
                    existing = state_map.get(copy)
                    if existing is None:
                        copy.total_prob = new_prob
                        state_map[copy] = copy
                    else:
                        existing.total_prob += new_prob

        ac_minus: dict[int, list[_TripleDPEngine._Configuration]] = {}
        for state_id, cfg_map in ac_minus_map.items():
            ac_minus[state_id] = list(cfg_map.keys())
        return ac_minus

    def _split_at_network_node(self,
                               cacs: dict[int, list["_TripleDPEngine._Configuration"]],
                               net_node_id: int) -> tuple[dict[int, list["_TripleDPEngine._Configuration"]],
                                                          dict[int, list["_TripleDPEngine._Configuration"]]]:
        """
        Split configurations at a reticulation into two parent directions.

        Args:
            cacs (dict[int, list[_Configuration]]): Incoming state/config map.
            net_node_id (int): Index of reticulation currently being processed.
        Returns:
            tuple[dict[int, list[_Configuration]], dict[int, list[_Configuration]]]:
                Configuration maps for parent-1 and parent-2 branches.
        """
        ac_plus_1: dict[int, list[_TripleDPEngine._Configuration]] = {}
        ac_plus_2: dict[int, list[_TripleDPEngine._Configuration]] = {}
        net_index = 1

        for state_id, configs in cacs.items():
            for split1, split2 in self._SPLITTING_MAP[state_id]:
                list1 = ac_plus_1.setdefault(split1, [])
                list2 = ac_plus_2.setdefault(split2, [])

                op_list1: Optional[list[_TripleDPEngine._Configuration]] = None
                op_list2: Optional[list[_TripleDPEngine._Configuration]] = None
                if state_id != 0:
                    op_list1 = ac_plus_1.setdefault(split2, [])
                    op_list2 = ac_plus_2.setdefault(split1, [])

                for config in configs:
                    # Probabilities are split symmetrically before applying
                    # edge-specific inheritance/coalescent transitions.
                    prob = math.sqrt(config.total_prob)
                    new_cfg_1 = config.copy()
                    new_cfg_1.add_choice(net_node_id, net_index)
                    new_cfg_1.total_prob = prob
                    new_cfg_2 = new_cfg_1.copy()
                    new_cfg_2.total_prob = prob
                    list1.append(new_cfg_1)
                    list2.append(new_cfg_2)
                    net_index += 1

                    if state_id != 0 and op_list1 is not None and op_list2 is not None:
                        new_cfg_1 = config.copy()
                        new_cfg_1.add_choice(net_node_id, net_index)
                        new_cfg_1.total_prob = prob
                        new_cfg_2 = new_cfg_1.copy()
                        new_cfg_2.total_prob = prob
                        op_list1.append(new_cfg_1)
                        op_list2.append(new_cfg_2)
                        net_index += 1

        return ac_plus_1, ac_plus_2

    def calculate_triple_probability(self, triple: tuple[str, str, str]) -> float:
        """
        Compute P(AB|C)-style probability for one ordered triplet.

        Uses pre-cached topology lookups so the inner DP loop never touches
        Network methods.

        Args:
            triple (tuple[str, str, str]): Ordered species labels.
        Returns:
            float: Probability for the corresponding triplet resolution.
        """
        triple_list = list(triple)
        edge_to_ac_minus: dict[
            tuple[Node, Node],
            dict[int, list[_TripleDPEngine._Configuration]],
        ] = {}
        net_node_id = 0
        total_prob = 0.0

        _Cfg = self._Configuration
        nnn = self.net_node_num
        node_is_leaf = self._node_is_leaf
        node_is_retic = self._node_is_retic
        node_children = self._node_children
        node_parents = self._node_parents
        node_label = self._node_label
        edge_info = self._edge_info
        lowest_art = self.lowest_articulation_nodes
        art_nodes = self.articulation_nodes
        MERGING = self._MERGING_MAP

        for node in self._topo_leaf_to_root:
            cacs = None
            is_leaf = node_is_leaf[node]
            is_retic = node_is_retic[node]

            if is_leaf:
                lbl = node_label[node]
                if lbl in triple_list:
                    idx = triple_list.index(lbl)
                    cacs = {idx + 1: [_Cfg(nnn)]}

            elif is_retic:
                children = node_children[node]
                if children:
                    cacs = edge_to_ac_minus.get((node, children[0]))

            else:
                children = node_children[node]
                if len(children) >= 2:
                    ac1 = edge_to_ac_minus.get((node, children[0]))
                    ac2 = edge_to_ac_minus.get((node, children[1]))

                    if (ac1 is None) ^ (ac2 is None):
                        cacs = ac2 if ac1 is None else ac1
                        if node in lowest_art:
                            for state, config_list in list(cacs.items()):
                                if not config_list:
                                    continue
                                merged = config_list[0]
                                for config in config_list[1:]:
                                    merged.total_prob += config.total_prob
                                merged.clear_choices()
                                cacs[state] = [merged]
                    elif ac1 is not None and ac2 is not None:
                        cacs = {}
                        is_articulation = node in lowest_art
                        for state1, cfgs1 in ac1.items():
                            for state2, cfgs2 in ac2.items():
                                can_merge = state1 == 0 or state2 == 0
                                target_state = state1 if state1 != 0 else state2
                                if not can_merge:
                                    key = (state1, state2) if state1 < state2 else (state2, state1)
                                    ts = MERGING.get(key)
                                    if ts is not None:
                                        target_state = ts
                                        can_merge = True
                                if not can_merge:
                                    continue

                                merged_list = cacs.setdefault(target_state, [])
                                merged_articulation_cfg = None
                                for cfg1 in cfgs1:
                                    for cfg2 in cfgs2:
                                        if not cfg1.is_compatible(cfg2):
                                            continue
                                        if is_articulation:
                                            if not merged_list:
                                                merged_articulation_cfg = _Cfg.merge(
                                                    cfg1, cfg2
                                                )
                                                merged_articulation_cfg.clear_choices()
                                                merged_list.append(merged_articulation_cfg)
                                            else:
                                                if merged_articulation_cfg is None:
                                                    merged_articulation_cfg = merged_list[0]
                                                merged_articulation_cfg.total_prob += max(
                                                    0.0, cfg1.total_prob * cfg2.total_prob
                                                )
                                        else:
                                            merged_list.append(_Cfg.merge(cfg1, cfg2))

                                if not merged_list:
                                    cacs.pop(target_state, None)

            if cacs is None:
                continue

            if 7 in cacs and node in art_nodes:
                total_prob = cacs[7][0].total_prob / 3.0
                if 9 in cacs:
                    total_prob += cacs[9][0].total_prob
                if 10 in cacs:
                    total_prob += cacs[10][0].total_prob
                break

            if is_retic:
                ac_plus_1, ac_plus_2 = self._split_at_network_node(cacs, net_node_id)
                net_node_id += 1
                parents = node_parents[node]
                for idx, parent in enumerate(parents):
                    bl, gamma = edge_info[(parent, node)]
                    inheritance_prob = gamma if gamma is not None else 1.0 / len(parents)

                    ac_plus = ac_plus_1 if idx == 0 else ac_plus_2
                    ac_minus = self._compute_ac_minus(
                        ac_plus,
                        branch_length=bl,
                        inheritance_prob=inheritance_prob,
                    )
                    edge_to_ac_minus[(parent, node)] = ac_minus
            else:
                parents = node_parents[node]
                if parents:
                    parent = parents[0]
                    bl, _gamma = edge_info[(parent, node)]
                    ac_minus = self._compute_ac_minus(
                        cacs,
                        branch_length=bl,
                        inheritance_prob=1.0,
                    )
                    edge_to_ac_minus[(parent, node)] = ac_minus

        return total_prob


_CY_MAX_CHILDREN = 4
_CY_MAX_PARENTS = 2


def _extract_topology_for_cython(engine: _TripleDPEngine) -> dict:
    """Convert a _TripleDPEngine's cached topology into flat numpy arrays.

    Returns a dict consumed by mpl_engine_cy.score_all_triplets.
    """
    nodes = engine._topo_leaf_to_root
    n_nodes = len(nodes)
    node_to_idx: dict[Node, int] = {node: i for i, node in enumerate(nodes)}

    is_leaf = np.zeros(n_nodes, dtype=np.intc)
    is_retic = np.zeros(n_nodes, dtype=np.intc)
    in_art = np.zeros(n_nodes, dtype=np.intc)
    in_low_art = np.zeros(n_nodes, dtype=np.intc)
    n_ch = np.zeros(n_nodes, dtype=np.intc)
    ch = np.full((n_nodes, _CY_MAX_CHILDREN), -1, dtype=np.intc)
    n_pa = np.zeros(n_nodes, dtype=np.intc)
    pa = np.full((n_nodes, _CY_MAX_PARENTS), -1, dtype=np.intc)
    pa_bl = np.ones((n_nodes, _CY_MAX_PARENTS), dtype=np.float64)
    pa_gamma = np.ones((n_nodes, _CY_MAX_PARENTS), dtype=np.float64)
    ch_pa_slot = np.full((n_nodes, _CY_MAX_CHILDREN), 0, dtype=np.intc)

    label_to_idx: dict[str, int] = {}

    for i, node in enumerate(nodes):
        is_leaf[i] = 1 if engine._node_is_leaf[node] else 0
        is_retic[i] = 1 if engine._node_is_retic[node] else 0
        in_art[i] = 1 if node in engine.articulation_nodes else 0
        in_low_art[i] = 1 if node in engine.lowest_articulation_nodes else 0

        children = engine._node_children[node]
        n_ch[i] = len(children)
        for j, child in enumerate(children[:_CY_MAX_CHILDREN]):
            ch[i, j] = node_to_idx[child]

        parents = engine._node_parents[node]
        n_pa[i] = len(parents)
        for j, parent in enumerate(parents[:_CY_MAX_PARENTS]):
            pa[i, j] = node_to_idx[parent]
            bl, gamma = engine._edge_info.get((parent, node), (1.0, None))
            pa_bl[i, j] = bl if bl is not None else -1.0
            if engine._node_is_retic[node]:
                if gamma is not None and gamma > 0:
                    pa_gamma[i, j] = gamma
                else:
                    pa_gamma[i, j] = 1.0 / len(parents)
            else:
                pa_gamma[i, j] = 1.0

        if is_leaf[i]:
            label_to_idx[engine._node_label[node]] = i

    for i, node in enumerate(nodes):
        children = engine._node_children[node]
        for j, child in enumerate(children[:_CY_MAX_CHILDREN]):
            child_idx = node_to_idx[child]
            child_parents = engine._node_parents[child]
            for k, p in enumerate(child_parents[:_CY_MAX_PARENTS]):
                if node_to_idx[p] == i:
                    ch_pa_slot[i, j] = k
                    break

    return {
        "n_nodes": n_nodes,
        "net_node_num": engine.net_node_num,
        "is_leaf": is_leaf,
        "is_retic": is_retic,
        "in_art": in_art,
        "in_low_art": in_low_art,
        "n_children": n_ch,
        "children": ch,
        "n_parents": n_pa,
        "parents": pa,
        "pa_bl": pa_bl,
        "pa_gamma": pa_gamma,
        "ch_pa_slot": ch_pa_slot,
        "label_to_idx": label_to_idx,
    }


def _score_with_cython(
    engine: _TripleDPEngine,
    triplets: list[tuple[str, str, str]],
    rho: dict[tuple[str, str, str], tuple[float, float, float]],
) -> float:
    """Fast scoring path using the Cython DP engine."""
    topo = _extract_topology_for_cython(engine)
    lbl_idx = topo["label_to_idx"]

    trip_idx: list[tuple[int, int, int]] = []
    rho_vals: list[tuple[float, float, float]] = []
    for t in triplets:
        x, y, z = t
        if x in lbl_idx and y in lbl_idx and z in lbl_idx:
            trip_idx.append((lbl_idx[x], lbl_idx[y], lbl_idx[z]))
            rho_vals.append(rho[t])

    return _cy_score(
        topo["n_nodes"],
        topo["net_node_num"],
        topo["is_leaf"],
        topo["is_retic"],
        topo["in_art"],
        topo["in_low_art"],
        topo["n_children"],
        topo["children"],
        topo["n_parents"],
        topo["parents"],
        topo["pa_bl"],
        topo["pa_gamma"],
        topo["ch_pa_slot"],
        trip_idx,
        rho_vals,
    )


def compute_gene_tree_triplets(gene_trees: GeneTrees,
                               mapping: dict[str, list[str]],
                               species_labels: Optional[list[str]] = None) -> GeneTreeTripletResult:
    """
    Compute rho values for every species triplet represented in the input.

    Args:
        gene_trees (GeneTrees): A GeneTrees object with one or many trees.
        mapping (dict[str, list[str]]): Species to allele labels map.
        species_labels (Optional[list[str]], optional): Explicit list of species
                    labels to use instead of mapping keys. Defaults to None.
    Returns:
        GeneTreeTripletResult: triplet list + rho values for each triplet.
    """
    if species_labels is None:
        species_labels = sorted(mapping.keys())
    else:
        species_labels = sorted(species_labels)

    if len(species_labels) < 3:
        raise ValueError("Need at least 3 species labels to form triplets")

    triplets = list(combinations(species_labels, 3))
    rho_by_triplet = _compute_all_rhos_fast(gene_trees, mapping, triplets)

    return GeneTreeTripletResult(triplets=triplets, rho_by_triplet=rho_by_triplet)


def score_species_network_triplets(species_net: Network,
                                   gene_triplet_result: GeneTreeTripletResult) -> SpeciesNetworkTripletResult:
    """
    Compute all species-network triplet probabilities and final MPL score.

    Args:
        species_net (Network): Species network to evaluate.
        gene_triplet_result (GeneTreeTripletResult): Output from
                    compute_gene_tree_triplets().
    Returns:
        SpeciesNetworkTripletResult: triplets, per-triplet probabilities,
                    and total log pseudo-likelihood.
    """
    # Use network leaves as the scoring universe.
    triplets = list(combinations(sorted(n.label for n in species_net.get_leaves()), 3))
    triple_engine = _TripleDPEngine(species_net)
    probs_by_triplet: dict[tuple[str, str, str], tuple[float, float, float]] = {}

    total = 0.0
    for triplet in triplets:
        # For triplet (x, y, z), engine computes P(xy|z) and P(xz|y).
        # P(yz|x) is derived by normalization.
        x, y, z = triplet
        p_xy = triple_engine.calculate_triple_probability((x, y, z))
        p_xz = triple_engine.calculate_triple_probability((x, z, y))
        probs = (p_xy, p_xz, max(1.0 - p_xy - p_xz, 0.0))
        probs_by_triplet[triplet] = probs

        rho = gene_triplet_result.rho_by_triplet.get(triplet)
        if rho is None:
            raise ValueError(
                f"Missing gene-tree triplet frequencies for species triplet {triplet}"
            )

        # MPL contribution for one species triplet.
        for rho_i, p_i in zip(rho, probs):
            if rho_i > 0.0:
                total += rho_i * (math.log(p_i) if p_i > 0.0 else _LOG_FLOOR)

    return SpeciesNetworkTripletResult(
        triplets=triplets,
        probs_by_triplet=probs_by_triplet,
        log_pseudo_likelihood=total,
    )


##############################
#### SCORER AND KERNEL #######
##############################


class MPLScorer:
    """Callable scorer for use with ``Model.set_likelihood_calculator()``.

    Holds the precomputed gene-tree rho values (constant across the search)
    and rebuilds the species-network DP engine on every call, since topology
    moves change the articulation-node structure.
    """

    def __init__(self,
                 rho: dict[tuple[str, str, str], tuple[float, float, float]],
                 triplets: list[tuple[str, str, str]]) -> None:
        """
        Args:
            rho: Mapping triplet -> (rho_xy_z, rho_xz_y, rho_yz_x).
            triplets: Canonical species triplets to iterate over.
        """
        self._rho = rho
        # Skip triplets never observed in gene trees (zero contribution).
        self._triplets = [
            t for t in triplets if any(rho[t][i] > 0.0 for i in range(3))
        ]

    def __call__(self, model: Model) -> float:
        """Compute log pseudo-likelihood for the current network in *model*.

        Args:
            model: A Model whose ``network`` attribute is the species network.
        Returns:
            Log pseudo-likelihood score (negative; higher = better).
        """
        engine = _TripleDPEngine(model.network)

        if _HAS_CYTHON_MPL:
            return _score_with_cython(engine, self._triplets, self._rho)

        total = 0.0
        for triplet in self._triplets:
            x, y, z = triplet
            p_xy = engine.calculate_triple_probability((x, y, z))
            p_xz = engine.calculate_triple_probability((x, z, y))
            probs = (p_xy, p_xz, max(1.0 - p_xy - p_xz, 0.0))
            rho = self._rho[triplet]
            for rho_i, p_i in zip(rho, probs):
                if rho_i > 0.0:
                    total += rho_i * (math.log(p_i) if p_i > 0.0 else _LOG_FLOOR)
        return total


class MPLKernel(ProposalKernel):
    """Phase-aware proposal kernel for MPL network search.

    Cycles between two complementary search phases:

      **TOPOLOGY** -- Explores tree space via SPR and branch-length
      adjustments (ChangeNodeHeight), with ChangeInheritanceProb
      mixed in for gamma tuning.

      **RETICULATION** -- Searches for hybridization events via
      Add/Remove/Flip reticulation and source/dest moves, with
      ChangeInheritanceProb and ChangeNodeHeight for local
      refinement after reticulation changes.

    Phase transitions fire after ``phase_patience`` consecutive
    proposals without accepted improvement.  Within each phase,
    selection weights adapt from phase-specific base distributions
    using a sliding window of recent acceptance rates.

    When explicit *weights* are supplied, the kernel falls back to
    flat (non-phased) mode for backward compatibility.
    """

    TOPOLOGY = "topology"
    RETICULATION = "reticulation"

    _TOPOLOGY_BASE: dict[type, float] = {
        SPR: 5.0,
        ChangeNodeHeight: 3.0,
        ChangeInheritanceProb: 1.0,
    }

    _RETICULATION_BASE: dict[type, float] = {
        AddReticulation: 3.0,
        RemoveReticulation: 2.0,
        FlipReticulation: 1.5,
        ChangeReticSource: 1.5,
        ChangeReticDest: 1.5,
        RelocateReticulation: 2.5,
        ChangeInheritanceProb: 1.5,
        ChangeNodeHeight: 1.0,
    }

    _PHASE_ORDER = [TOPOLOGY, RETICULATION]

    def __init__(self,
                 move_types: list[type[Move]] | None = None,
                 weights: list[float] | None = None,
                 max_reticulations: int | None = None,
                 adaptive: bool = True,
                 window_size: int = 30,
                 min_weight: float = 0.05,
                 phase_patience: int = 25,
                 warmup: int = 8) -> None:
        """
        Args:
            move_types: Move classes available to the kernel.  When
                ``None`` the full default set is used.
            weights: Fixed per-move selection weights.  Supplying this
                disables phased cycling and adaptive tuning.
            max_reticulations: Cap on reticulation nodes.
            adaptive: Enable within-phase adaptive weight scaling.
            window_size: Sliding-window length for acceptance stats.
            min_weight: Minimum scale factor (fraction of base weight)
                to prevent any move from being fully starved.
            phase_patience: Consecutive non-improving proposals before
                the kernel switches to the next phase.
            warmup: Minimum observations per move class before the
                adaptive scaling activates for that class.
        """
        super().__init__()
        self._max_retics: int | None = max_reticulations
        self._all_moves: list[type[Move]] = move_types or [
            SPR,
            ChangeNodeHeight,
            ChangeInheritanceProb,
            AddReticulation,
            RemoveReticulation,
            FlipReticulation,
            ChangeReticSource,
            ChangeReticDest,
            RelocateReticulation,
        ]

        self._fixed_weights: list[float] | None = weights
        self._phased: bool = (weights is None)
        self._adaptive: bool = adaptive and (weights is None)

        self._phase: str = self.TOPOLOGY
        self._phase_patience: int = phase_patience
        self._stagnation: int = 0
        self._phase_switches: int = 0

        self._window_size: int = window_size
        self._min_weight: float = min_weight
        self._warmup: int = warmup

        from collections import deque
        self._history: dict[type[Move], deque] = {
            cls: deque(maxlen=window_size) for cls in self._all_moves
        }
        self._last_cls: type[Move] | None = None

    # ── phase helpers ───────────────────────────────────────────

    def _phase_base_map(self) -> dict[type, float]:
        """Base weight map for the current phase."""
        if self._phase == self.TOPOLOGY:
            return self._TOPOLOGY_BASE
        return self._RETICULATION_BASE

    def _active_moves_and_base(self) -> tuple[list[type[Move]], list[float]]:
        """Active move classes and base weights for the current phase."""
        base = self._phase_base_map()
        moves: list[type[Move]] = []
        weights: list[float] = []
        for cls in self._all_moves:
            if cls in base:
                moves.append(cls)
                weights.append(base[cls])
        return moves, weights

    def _maybe_switch_phase(self) -> None:
        """Advance to the next phase when stagnation exceeds patience."""
        if self._stagnation < self._phase_patience:
            return
        idx = self._PHASE_ORDER.index(self._phase)
        for offset in range(1, len(self._PHASE_ORDER) + 1):
            candidate = self._PHASE_ORDER[
                (idx + offset) % len(self._PHASE_ORDER)
            ]
            base = (self._TOPOLOGY_BASE if candidate == self.TOPOLOGY
                    else self._RETICULATION_BASE)
            if any(cls in base for cls in self._all_moves):
                self._phase = candidate
                self._phase_switches += 1
                break
        self._stagnation = 0

    # ── adaptive tuning ─────────────────────────────────────────

    def _adapt_weights(self,
                       moves: list[type[Move]],
                       base_weights: list[float]) -> list[float]:
        """Scale base weights by recent acceptance rate.

        Scaling is linear from ``min_weight`` of the base (at 0%
        acceptance) to 2x the base (at 100% acceptance).  Moves with
        fewer than ``warmup`` observations keep their full base weight.
        """
        floor = self._min_weight
        weights: list[float] = []
        for cls, base_w in zip(moves, base_weights):
            hist = self._history[cls]
            if len(hist) < self._warmup:
                weights.append(base_w)
                continue
            n_acc = sum(1 for accepted, _ in hist if accepted)
            acc_rate = n_acc / len(hist)
            scale = floor + (2.0 - floor) * acc_rate
            weights.append(base_w * scale)
        return weights

    # ── public interface ────────────────────────────────────────

    def generate(self) -> Move:
        """Sample a move from the current phase's weighted distribution."""
        if self._phased:
            self._maybe_switch_phase()
            moves, base_weights = self._active_moves_and_base()
            weights = (self._adapt_weights(moves, base_weights)
                       if self._adaptive else base_weights)
        else:
            moves = self._all_moves
            weights = self._fixed_weights

        cls = random.choices(moves, weights=weights, k=1)[0]
        self._last_cls = cls

        if cls is AddReticulation and self._max_retics is not None:
            return AddReticulation(max_reticulations=self._max_retics)
        return cls()

    def report_outcome(self, accepted: bool, delta: float = 0.0) -> None:
        """Record move outcome and update phase-stagnation counter."""
        if self._last_cls is not None and self._last_cls in self._history:
            self._history[self._last_cls].append((accepted, delta))
        if accepted and delta > 0:
            self._stagnation = 0
        else:
            self._stagnation += 1

    def get_weights(self) -> dict[str, float]:
        """Return current effective selection weights (normalized)."""
        if self._phased:
            moves, base = self._active_moves_and_base()
            weights = (self._adapt_weights(moves, base)
                       if self._adaptive else base)
        else:
            moves = self._all_moves
            weights = self._fixed_weights or [1.0] * len(moves)
        total = sum(weights)
        return {cls.__name__: w / total for cls, w in
                zip(moves, weights)}

    @property
    def phase(self) -> str:
        """Current search phase name."""
        return self._phase

    @property
    def phase_switches(self) -> int:
        """Total number of phase transitions so far."""
        return self._phase_switches


###################
#### MPL CLASS ####
###################
 
class MPL:
    """Maximum pseudo-likelihood scorer for a species network.
    
    Precomputes rho (triple frequencies from gene trees) once at init.
    Call :meth:`score` to evaluate the current species network; only the
    network-side probabilities are recomputed.
    
    Example::
    
        >>> mpl = MPL(species_net, gene_trees, mapping)
        >>> log_pl = mpl.score()
    """
    
    def __init__(self,
                 species_net: Network,
                 gene_trees: GeneTrees,
                 mapping: dict[str, list[str]]) -> None:
        """
        Initialize MPL scorer state for one species network.

        Initialization performs only the gene-tree side precomputation
        (rho values). Network-side probabilities are computed when score()
        is called.

        Args:
            species_net (Network): Species network to evaluate.
            gene_trees (GeneTrees): Input gene trees.
            mapping (dict[str, list[str]]): Species -> allele labels map.
        Returns:
            N/A
        """
        self.net = species_net
        self.gene_trees = gene_trees
        self.mapping = mapping

        # Enumerate triplets and precompute rho (constant for fixed gene trees)
        precomputed = compute_gene_tree_triplets(
            gene_trees=self.gene_trees,
            mapping=self.mapping,
            species_labels=[n.label for n in self.net.get_leaves()],
        )
        self._triplets = precomputed.triplets
        self._rho = precomputed.rho_by_triplet
        self._active_triplets = [
            t for t in self._triplets
            if any(self._rho[t][i] > 0.0 for i in range(3))
        ]

        # Match PhyloNet semantics: compute each triplet on the full network.
        self._triple_engine = _TripleDPEngine(self.net)
    
    @classmethod
    def from_nexus(cls, gt_file: str, st_file: str, mapping: dict[str, list[str]]):
        """
        Instantiate instead from nexus file paths.

        Args:
            gt_file (str): Path to the gene tree file.
            st_file (str): Path to the species tree file.
            mapping (dict[str, list[str]]): Mapping of genes to allele labels.
        Returns:
            MPL: A MPL object.
        """
        st : Network = io.read_nexus(st_file, return_type="networks")
        gts : GeneTrees = io.read_nexus(gt_file, return_type="genetrees")
        gts.species_gene_mapping(mapping)
        return cls(st, gts, mapping)

    @staticmethod
    def compute_gene_tree_triplets(gene_trees: GeneTrees,
                                   mapping: dict[str, list[str]],
                                   species_labels: Optional[list[str]] = None) -> GeneTreeTripletResult:
        """
        Wrapper for the module-level compute_gene_tree_triplets function.

        Args:
            gene_trees (GeneTrees): Gene tree collection.
            mapping (dict[str, list[str]]): Species to allele map.
            species_labels (Optional[list[str]], optional): Explicit species list.
        Returns:
            GeneTreeTripletResult: precomputed triplet frequencies.
        """
        return compute_gene_tree_triplets(
            gene_trees=gene_trees,
            mapping=mapping,
            species_labels=species_labels,
        )

    @staticmethod
    def score_species_network_triplets(species_net: Network,
                                       gene_triplet_result: GeneTreeTripletResult) -> SpeciesNetworkTripletResult:
        """
        Wrapper for module-level score_species_network_triplets.

        Args:
            species_net (Network): Species network to score.
            gene_triplet_result (GeneTreeTripletResult): Triplet frequencies from
                            gene trees.
        Returns:
            SpeciesNetworkTripletResult: per-triplet probabilities and final score.
        """
        return score_species_network_triplets(
            species_net=species_net,
            gene_triplet_result=gene_triplet_result,
        )

    # ── Scoring ───────────────────────────────────────────────────
    
    def score(self) -> float:
        """
        Compute total log pseudo-likelihood for the current species network.

        This loops over all species triplets and sums their individual
        contributions.

        Returns:
            float: Total log pseudo-likelihood.
        """
        if _HAS_CYTHON_MPL:
            return _score_with_cython(
                self._triple_engine, self._active_triplets, self._rho,
            )

        total = 0.0
        for triplet in self._active_triplets:
            total += self._score_triplet(triplet)
        return total
    
    def search(self,
               method: str = "hc",
               num_iter: int = 700,
               kernel: MPLKernel | None = None,
               max_reticulations: int | None = None,
               *,
               save_best_path: str | os.PathLike[str] | None = None,
               reference_network: Network | None = None,
               comparison_report_path: str | os.PathLike[str] | None = None,
               comparison_top_k: int = 25,
               print_comparison: bool = True,
               **kwargs) -> float:
        """Search the network space for the species network that maximises
        the pseudo-likelihood of the observed gene trees.

        Args:
            method: ``"hc"`` for Hill Climbing (default) or ``"sa"`` for
                    Simulated Annealing.
            num_iter: Number of topology moves to evaluate.
            kernel: Custom :class:`MPLKernel` instance. When ``None`` a
                    default kernel is used.
            max_reticulations: Upper bound on reticulation nodes allowed.
                    ``None`` means unlimited.  Ignored if a custom *kernel*
                    is supplied.
            save_best_path: If set, write the best-found network as Newick
                to this path after the search completes.
            reference_network: If set (e.g. a simulator ground truth),
                compare found vs reference MPL score, reticulations, clades,
                and top triplet log-likelihood gaps.
            comparison_report_path: If set with *reference_network*, also
                write the comparison report to this file (UTF-8).
            comparison_top_k: Number of triplets listed in the report.
            print_comparison: When *reference_network* is set, print the
                report to stdout unless this is ``False``.
            **kwargs: Forwarded to the search constructor.  For SA these
                      include ``t_start``, ``t_end``, ``n_restarts``, ``seed``,
                      ``plateau_frac``.

        Returns:
            Log pseudo-likelihood of the best network found.  The MPL
            instance's network (``self.net``) is updated in-place with the
            result.
        """
        scorer = MPLScorer(self._rho, self._active_triplets)

        model = Model()
        model.network = copy.deepcopy(self.net)
        model.set_likelihood_calculator(scorer)

        if kernel is None:
            kernel = MPLKernel(max_reticulations=max_reticulations)

        if method == "hc":
            searcher = HillClimbing(
                pkernel=kernel,
                model=model,
                num_iter=num_iter,
                **kwargs,
            )
        elif method == "sa":
            searcher = SimulatedAnnealing(
                pkernel=kernel,
                model=model,
                num_iter=num_iter,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown search method {method!r}; use 'hc' or 'sa'")

        end_state = searcher.run()

        self.net = end_state.current_model.network
        self._triple_engine = _TripleDPEngine(self.net)

        if save_best_path is not None:
            save_mpl_network_newick(self.net, save_best_path)

        if reference_network is not None:
            report = format_mpl_reference_comparison(
                self.net,
                reference_network,
                self._rho,
                self._active_triplets,
                top_k=comparison_top_k,
            )
            if print_comparison:
                print(report, flush=True)
            if comparison_report_path is not None:
                Path(comparison_report_path).write_text(report + "\n", encoding="utf-8")

        return end_state.likelihood()
    
    def _score_triplet(self, triplet: tuple[str, str, str]) -> float:
        """
        Compute the MPL contribution of one species triplet.

        Args:
            triplet (tuple[str, str, str]): Canonical species triplet.
        Returns:
            float: Log pseudo-likelihood contribution from this triplet.
        """
        x, y, z = triplet
        # Evaluate two ordered resolutions directly.
        p_xy = self._triple_engine.calculate_triple_probability((x, y, z))
        p_xz = self._triple_engine.calculate_triple_probability((x, z, y))
        probs = (p_xy, p_xz, max(1.0 - p_xy - p_xz, 0.0))
        rho = self._rho[triplet]
        
        contribution = 0.0
        # Skip zero rho terms and guard log(0) with a floor.
        for rho_i, p_i in zip(rho, probs):
            if rho_i > 0.0:
                contribution += rho_i * (math.log(p_i) if p_i > 0.0 else _LOG_FLOOR)
        return contribution
    
