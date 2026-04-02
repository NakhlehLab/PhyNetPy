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
 
import math
from collections import deque
from itertools import combinations
from typing import TYPE_CHECKING, Optional

from .Network import Network, Node, Edge
from .GeneTrees import GeneTrees
from .GraphUtils import subnet_given_leaves
from . import IO as io
 
if TYPE_CHECKING:
    pass
 
_LOG_FLOOR = math.log(1e-200)
 
 
class GeneTreeTripletResult:
    """
    Container for gene-tree triplet frequencies (rho values).

    This object is returned by compute_gene_tree_triplets and then consumed
    by score_species_network_triplets.
    """
    def __init__(self, 
                 triplets : list[tuple[str, str, str]],
                 rho_by_triplet : dict[tuple[str, str, str], tuple[float, float, float]]) -> None:
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
                 triplets : list[tuple[str, str, str]],
                 probs_by_triplet : dict[tuple[str, str, str], tuple[float, float, float]],
                 log_pseudo_likelihood : float) -> None:
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


##########################
#### HELPER FUNCTIONS ####
##########################
 
def _induced_triple(tree: Network, x: str, y: str, z: str) -> str:
    """
    Determine the induced rooted topology for three leaves in a gene tree.

    This helper asks which pair coalesces first (if any) by comparing the
    MRCA of each pair against the MRCA of all three leaves.

    Args:
        tree (Network): One gene tree.
        x (str): First allele label.
        y (str): Second allele label.
        z (str): Third allele label.
    Returns:
        str: One of "xy|z", "xz|y", "yz|x", or "star".
    """
    mrca_xy = tree.mrca({x, y})
    mrca_xz = tree.mrca({x, z})
    mrca_yz = tree.mrca({y, z})
    mrca_xyz = tree.mrca({x, y, z})
 
    if mrca_xy != mrca_xyz:
        return "xy|z"
    if mrca_xz != mrca_xyz:
        return "xz|y"
    if mrca_yz != mrca_xyz:
        return "yz|x"
    return "star"
 
 
def _coalescent_triple_probs(tree: Network, X: str, Y: str, Z: str) -> tuple[float, float, float]:
    """
    Compute closed-form triplet probabilities on one 3-taxon tree.

    For a resolved rooted tree with internal branch length tau:
        P(match) = 1 - (2/3) * exp(-tau)
        P(mismatch) = (1/3) * exp(-tau)

    Args:
        tree (Network): A 3-leaf tree (no reticulation nodes).
        X (str): First leaf label in canonical order.
        Y (str): Second leaf label in canonical order.
        Z (str): Third leaf label in canonical order.
    Returns:
        tuple[float, float, float]: (P(XY|Z), P(XZ|Y), P(YZ|X)).
    """
    root = tree.root()
    children = tree.get_children(root)
    
    # Find which pair of taxa are sisters (share an internal MRCA != root)
    sister_pair: Optional[set[str]] = None
    tau = 0.0
    
    for child in children:
        descs = {n.label for n in tree.leaf_descendants(child)}
        if len(descs) == 2:
            sister_pair = descs
            # Branch length of the edge into this cherry node
            in_edges = list(tree.in_edges(child))
            if in_edges:
                length = in_edges[0].get_length()
                if length is not None and length > 0:
                    tau = length
            break
    
    # Star topology - no resolved cherry
    if sister_pair is None:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    
    exp_neg_tau = math.exp(-tau)
    p_match = 1.0 - (2.0 / 3.0) * exp_neg_tau
    p_mismatch = (1.0 / 3.0) * exp_neg_tau
 
    if sister_pair == {X, Y}:
        return (p_match, p_mismatch, p_mismatch)
    if sister_pair == {X, Z}:
        return (p_mismatch, p_match, p_mismatch)
    if sister_pair == {Y, Z}:
        return (p_mismatch, p_mismatch, p_match)
    
    return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
 
 
##########################
#### RHO COMPUTATION #####
##########################
 
def _compute_rho(
    X: str, Y: str, Z: str,
    gene_trees: GeneTrees,
    mapping: dict[str, list[str]],
) -> tuple[float, float, float]:
    """
    Compute rho counts for one species triplet across all gene trees.

    A rho value is the sum, over gene trees, of normalized counts for each
    triplet resolution. If a gene-tree induced triplet is unresolved (star),
    that observation is split evenly across all three resolutions.

    Args:
        X (str): First species label.
        Y (str): Second species label.
        Z (str): Third species label.
        gene_trees (GeneTrees): Gene tree collection.
        mapping (dict[str, list[str]]): Species -> allele labels map.
    Returns:
        tuple[float, float, float]: (rho_xy_z, rho_xz_y, rho_yz_x).
    """
    ax_all = mapping.get(X, [])
    ay_all = mapping.get(Y, [])
    az_all = mapping.get(Z, [])
 
    rho_xy, rho_xz, rho_yz = 0.0, 0.0, 0.0
 
    for tree in gene_trees.trees:
        leaves = {leaf.label for leaf in tree.get_leaves()}
        ax = [a for a in ax_all if a in leaves]
        ay = [a for a in ay_all if a in leaves]
        az = [a for a in az_all if a in leaves]
 
        denom = len(ax) * len(ay) * len(az)
        if denom == 0:
            continue
 
        cnt_xy, cnt_xz, cnt_yz = 0.0, 0.0, 0.0
        for xi in ax:
            for yj in ay:
                for zk in az:
                    topo = _induced_triple(tree, xi, yj, zk)
                    if topo == "xy|z":
                        cnt_xy += 1.0
                    elif topo == "xz|y":
                        cnt_xz += 1.0
                    elif topo == "yz|x":
                        cnt_yz += 1.0
                    else:  # star
                        cnt_xy += 1.0 / 3.0
                        cnt_xz += 1.0 / 3.0
                        cnt_yz += 1.0 / 3.0
 
        rho_xy += cnt_xy / denom
        rho_xz += cnt_xz / denom
        rho_yz += cnt_yz / denom
 
    return (rho_xy, rho_xz, rho_yz)
 
 
###################################
#### FAST RHO PRECOMPUTATION ######
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


def _process_tree_batch(
    batch: list[Network],
    relevant: frozenset[str],
    mapping: dict[str, list[str]],
    triplets: list[tuple[str, str, str]],
    identity_mapping: bool,
) -> dict[tuple[str, str, str], list[float]]:
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


def _compute_all_rhos_fast(
    gene_trees: GeneTrees,
    mapping: dict[str, list[str]],
    triplets: list[tuple[str, str, str]],
    n_workers: int = 1,
) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    """Compute rho for every triplet in one pass over gene trees.

    Flips the loop order to tree-first so each gene tree is indexed
    once and reused across all 816+ triplets.  Uses _GeneTreeLCAIndex
    for O(1) topology queries instead of 4 × BFS per query.

    When *n_workers* > 1 the gene-tree list is split into batches and
    processed in parallel with ``concurrent.futures.ProcessPoolExecutor``.
    """
    all_allele_labels: set[str] = set()
    for alleles in mapping.values():
        all_allele_labels.update(alleles)
    relevant = frozenset(all_allele_labels)

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
        """
        One DP configuration for a single ancestral state.

        A configuration stores:
        1) total_prob: Probability mass for this configuration.
        2) net_node_index: Which parent edge was chosen at each reticulation.
        """

        def __init__(
            self,
            net_node_num: int,
            total_prob: float = 1.0,
            net_node_index: Optional[list[int]] = None,
        ) -> None:
            """
            Build a configuration object.

            Args:
                net_node_num (int): Number of reticulation nodes in the network.
                total_prob (float, optional): Probability mass for this state.
                                            Defaults to 1.0.
                net_node_index (Optional[list[int]], optional): Existing list of
                                            reticulation choices. If None, a zero
                                            initialized choice vector is created.
            Returns:
                N/A
            """
            self.total_prob = total_prob
            self.net_node_index = (
                net_node_index.copy()
                if net_node_index is not None
                else [0] * net_node_num
            )

        def copy(self) -> "_TripleDPEngine._Configuration":
            """
            Create a deep-enough copy for DP branching.

            Returns:
                _TripleDPEngine._Configuration: Copied configuration.
            """
            return _TripleDPEngine._Configuration(
                len(self.net_node_index),
                total_prob=self.total_prob,
                net_node_index=self.net_node_index,
            )

        @classmethod
        def merge(
            cls,
            c1: "_TripleDPEngine._Configuration",
            c2: "_TripleDPEngine._Configuration",
        ) -> "_TripleDPEngine._Configuration":
            """
            Merge two compatible configurations from sibling branches.

            Args:
                c1 (_TripleDPEngine._Configuration): First configuration.
                c2 (_TripleDPEngine._Configuration): Second configuration.
            Returns:
                _TripleDPEngine._Configuration: Merged configuration whose
                    probability is the product of both input probabilities.
            """
            merged_idx = [0] * len(c1.net_node_index)
            for i in range(len(merged_idx)):
                if c1.net_node_index[i] == c2.net_node_index[i]:
                    merged_idx[i] = c1.net_node_index[i]
                else:
                    merged_idx[i] = max(c1.net_node_index[i], c2.net_node_index[i])
            return cls(
                len(merged_idx),
                total_prob=max(0.0, c1.total_prob * c2.total_prob),
                net_node_index=merged_idx,
            )

        def is_compatible(self, other: "_TripleDPEngine._Configuration") -> bool:
            """
            Check whether two configurations can be merged.

            Two configurations conflict when both assigned different non-zero
            choices for the same reticulation.

            Args:
                other (_TripleDPEngine._Configuration): Candidate configuration.
            Returns:
                bool: True if configurations are compatible.
            """
            for mine, theirs in zip(self.net_node_index, other.net_node_index):
                if mine != theirs and mine != 0 and theirs != 0:
                    return False
            return True

        def add_choice(self, net_id: int, choice: int) -> None:
            """
            Record one reticulation parent choice.

            Args:
                net_id (int): Reticulation index.
                choice (int): Chosen split identifier.
            Returns:
                N/A
            """
            self.net_node_index[net_id] = choice

        def clear_choices(self) -> None:
            """
            Clear reticulation choices after articulation-point compression.

            Returns:
                N/A
            """
            for i in range(len(self.net_node_index)):
                self.net_node_index[i] = 0

        def __hash__(self) -> int:
            """
            Hash by reticulation-choice vector.

            Returns:
                int: Hash value.
            """
            return hash(tuple(self.net_node_index))

        def __eq__(self, other: object) -> bool:
            """
            Equality by reticulation-choice vector.

            Args:
                other (object): Candidate object.
            Returns:
                bool: True when other is a configuration with same choices.
            """
            if not isinstance(other, _TripleDPEngine._Configuration):
                return False
            return self.net_node_index == other.net_node_index

    def __init__(self, network: Network) -> None:
        """
        Initialize the DP engine for one fixed species network.

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

    @classmethod
    def _gij(cls, length: float, i: int, j: int) -> float:
        """
        Probability of i lineages reducing to j over branch length.

        This is the coalescent transition term used in the DP recurrence.
        Edge-case behavior matches PhyloNet for missing branch lengths.

        Args:
            length (float): Branch length.
            i (int): Lineages entering branch.
            j (int): Lineages leaving branch.
        Returns:
            float: Transition probability.
        """
        # Match PhyloNet special handling of missing branch lengths.
        if length is None or length == -1:
            return 1.0 if j == 1 else 0.0
        if length == 0:
            return 1.0 if i == j else 0.0
        if i == 0:
            return 1.0

        result = 0.0
        for k in range(j, i + 1):
            temp = (
                math.exp(0.5 * k * (1.0 - k) * length)
                * (2.0 * k - 1.0)
                * ((-1.0) ** (k - j))
                * cls._fact(j, j + k - 2)
                * cls._fact(i - k + 1, i)
            )
            denom = cls._fact(1, j) * cls._fact(1, k - j) * cls._fact(i, i + k - 1)
            result += temp / denom
        return result

    def _is_valid_network(self, ignore_node: Node) -> bool:
        """
        Check structural validity after temporarily removing an edge.

        Args:
            ignore_node (Node): Node exempted from degree-2 check.
        Returns:
            bool: True if network remains structurally valid.
        """
        visited: set[Node] = set()
        seen: set[Node] = set()
        roots = self.network.roots()
        if not roots:
            return False
        queue: list[Node] = [roots[0]]
        reachable_nodes: list[Node] = []
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            reachable_nodes.append(cur)
            queue.extend(self.network.get_children(cur))

        visited.clear()
        for node in reachable_nodes:
            if (
                self.network.in_degree(node) == 1
                and self.network.out_degree(node) == 1
                and node != ignore_node
            ):
                return False
            visited.add(node)
            for parent in self.network.get_parents(node):
                seen.add(parent)
            for child in self.network.get_children(node):
                seen.add(child)
        return len(visited) == len(seen)

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

            if is_root:
                all_children_articulate = all(
                    child in self.articulation_nodes
                    for child in self.network.get_children(node)
                )
                if not all_children_articulate:
                    self.lowest_articulation_nodes.add(node)
                self.articulation_nodes.add(node)
                continue

            if is_tree_node:
                all_children_articulate = all(
                    child in self.articulation_nodes
                    for child in self.network.get_children(node)
                )
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

    def _compute_ac_minus(
        self,
        cacs: dict[int, list["_TripleDPEngine._Configuration"]],
        branch_length: float,
        inheritance_prob: float,
    ) -> dict[int, list["_TripleDPEngine._Configuration"]]:
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
                        copy.total_prob = max(0.0, new_prob)
                        state_map[copy] = copy
                    else:
                        existing.total_prob += new_prob

        ac_minus: dict[int, list[_TripleDPEngine._Configuration]] = {}
        for state_id, cfg_map in ac_minus_map.items():
            ac_minus[state_id] = list(cfg_map.keys())
        return ac_minus

    def _split_at_network_node(
        self,
        cacs: dict[int, list["_TripleDPEngine._Configuration"]],
        net_node_id: int,
    ) -> tuple[
        dict[int, list["_TripleDPEngine._Configuration"]],
        dict[int, list["_TripleDPEngine._Configuration"]],
    ]:
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

        The input order matters:
            (X, Y, Z) computes P(XY|Z)

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

        # Bottom-up dynamic program over the network DAG.
        for node in reversed(self.network.topological_order()):
            cacs: Optional[dict[int, list[_TripleDPEngine._Configuration]]] = None
            is_leaf = self.network.out_degree(node) == 0
            is_network_node = self.network.in_degree(node) > 1

            if is_leaf:
                # Initialize one active lineage if this leaf belongs to the triplet.
                idx = -1
                for i, name in enumerate(triple_list):
                    if node.label == name:
                        idx = i
                        break
                if idx < 3 and idx >= 0:
                    cacs = {idx + 1: [self._Configuration(self.net_node_num)]}

            elif is_network_node:
                # Reticulation has one child in this bottom-up traversal state.
                children = self.network.get_children(node)
                if children:
                    cacs = edge_to_ac_minus.get((node, children[0]))

            else:
                # Tree node merges ancestry from up to two child branches.
                children = self.network.get_children(node)
                if len(children) >= 2:
                    ac1 = edge_to_ac_minus.get((node, children[0]))
                    ac2 = edge_to_ac_minus.get((node, children[1]))

                    if (ac1 is None) ^ (ac2 is None):
                        cacs = ac2 if ac1 is None else ac1
                        if cacs is not None and node in self.lowest_articulation_nodes:
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
                        is_articulation = node in self.lowest_articulation_nodes
                        for state1, cfgs1 in ac1.items():
                            for state2, cfgs2 in ac2.items():
                                can_merge = state1 == 0 or state2 == 0
                                target_state = state1 if state1 != 0 else state2
                                if not can_merge:
                                    key = (state1, state2) if state1 < state2 else (state2, state1)
                                    if key in self._MERGING_MAP:
                                        target_state = self._MERGING_MAP[key]
                                        can_merge = True
                                if not can_merge:
                                    continue

                                merged_list = cacs.setdefault(target_state, [])
                                merged_articulation_cfg: Optional[
                                    _TripleDPEngine._Configuration
                                ] = None
                                for cfg1 in cfgs1:
                                    for cfg2 in cfgs2:
                                        if not cfg1.is_compatible(cfg2):
                                            continue
                                        if is_articulation:
                                            # At lowest articulation points we can
                                            # sum compatible masses and forget the
                                            # specific reticulation choices.
                                            if len(merged_list) == 0:
                                                merged_articulation_cfg = self._Configuration.merge(
                                                    cfg1, cfg2
                                                )
                                                merged_articulation_cfg.clear_choices()
                                                merged_list.append(merged_articulation_cfg)
                                            else:
                                                if merged_articulation_cfg is None:
                                                    merged_articulation_cfg = merged_list[0]
                                                new_prob = max(
                                                    0.0, cfg1.total_prob * cfg2.total_prob
                                                )
                                                merged_articulation_cfg.total_prob += new_prob
                                        else:
                                            merged_list.append(self._Configuration.merge(cfg1, cfg2))

                                if len(merged_list) == 0:
                                    cacs.pop(target_state, None)

            if cacs is None:
                continue

            # If all three lineages are present at an articulation, extract total.
            if 7 in cacs and node in self.articulation_nodes:
                total_prob = cacs[7][0].total_prob / 3.0
                if 9 in cacs:
                    total_prob += cacs[9][0].total_prob
                if 10 in cacs:
                    total_prob += cacs[10][0].total_prob
                break

            if is_network_node:
                # Split at reticulation and propagate to each parent edge.
                ac_plus_1, ac_plus_2 = self._split_at_network_node(cacs, net_node_id)
                net_node_id += 1
                parents = self.network.get_parents(node)
                for idx, parent in enumerate(parents):
                    edge = self.network.get_edge(parent, node)
                    branch_length = edge.get_length()
                    inheritance_prob = edge.get_gamma()
                    if inheritance_prob is None:
                        inheritance_prob = 1.0 / len(parents) if len(parents) > 0 else 1.0

                    ac_plus = ac_plus_1 if idx == 0 else ac_plus_2
                    ac_minus = self._compute_ac_minus(
                        ac_plus,
                        branch_length=branch_length,
                        inheritance_prob=inheritance_prob,
                    )
                    edge_to_ac_minus[(parent, node)] = ac_minus
            else:
                # Tree nodes have one parent; no inheritance split.
                parents = self.network.get_parents(node)
                if parents:
                    parent = parents[0]
                    edge = self.network.get_edge(parent, node)
                    ac_minus = self._compute_ac_minus(
                        cacs,
                        branch_length=edge.get_length(),
                        inheritance_prob=1.0,
                    )
                    edge_to_ac_minus[(parent, node)] = ac_minus

        return total_prob


def _subnet_triple_probs(subnet: Network) -> tuple[float, float, float]:
    """
    Compute (P(XY|Z), P(XZ|Y), P(YZ|X)) for a 3-leaf subnetwork.

    This helper is the bridge between general network logic and the
    3-taxon dynamic program.

    Args:
        subnet (Network): Restricted subnetwork containing exactly three leaves.
    Returns:
        tuple[float, float, float]: Probabilities for all three rooted
                    triplet resolutions.
    """
    leaves = sorted(subnet.get_leaves(), key=lambda n: n.label)
    assert len(leaves) == 3, f"Expected 3 leaves, got {len(leaves)}"
    X, Y, Z = (leaf.label for leaf in leaves)

    engine = _TripleDPEngine(subnet)
    p_xy = engine.calculate_triple_probability((X, Y, Z))
    p_xz = engine.calculate_triple_probability((X, Z, Y))
    p_yz = max(1.0 - p_xy - p_xz, 0.0)

    return (p_xy, p_xz, p_yz)
 
 
def compute_gene_tree_triplets(
    gene_trees: GeneTrees,
    mapping: dict[str, list[str]],
    species_labels: Optional[list[str]] = None,
) -> GeneTreeTripletResult:
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


def score_species_network_triplets(
    species_net: Network,
    gene_triplet_result: GeneTreeTripletResult,
) -> SpeciesNetworkTripletResult:
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
    
    def __init__(
        self,
        species_net: Network,
        gene_trees: GeneTrees,
        mapping: dict[str, list[str]],
    ) -> None:
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

        # Match PhyloNet semantics: compute each triplet on the full network.
        self._triple_engine = _TripleDPEngine(self.net)
    
    @classmethod
    def from_nexus(cls, gt_file : str, st_file : str, mapping : dict[str, list[str]]):
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
    def compute_gene_tree_triplets(
        gene_trees: GeneTrees,
        mapping: dict[str, list[str]],
        species_labels: Optional[list[str]] = None,
    ) -> GeneTreeTripletResult:
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
    def score_species_network_triplets(
        species_net: Network,
        gene_triplet_result: GeneTreeTripletResult,
    ) -> SpeciesNetworkTripletResult:
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
        total = 0.0
        for triplet in self._triplets:
            total += self._score_triplet(triplet)
        return total
    
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
    
    def _restricted_subnet(self, triplet: tuple[str, str, str]) -> Network:
        """
        Build the induced subnetwork on exactly the provided species triplet.

        Args:
            triplet (tuple[str, str, str]): Species labels.
        Returns:
            Network: Restricted subnetwork containing only those species.
        Raises:
            ValueError: If any species label is not present in the network.
        """
        leaf_nodes = []
        for label in triplet:
            node = self.net.has_node_named(label)
            if node is None:
                raise ValueError(f"Species leaf '{label}' not in network")
            leaf_nodes.append(node)
        return subnet_given_leaves(self.net, leaf_nodes)
