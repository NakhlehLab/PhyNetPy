#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author : Mark Kessler
First Included in Version : 0.5.0

ReticulationComparison -- reticulation-aware dissimilarity for phylogenetic
networks.

This module implements the tripartition-matching dissimilarity of Nakhleh,
*Reticulation-Aware Dissimilarity for Phylogenetic Networks: A
Tripartition-Matching Approach* (2026).  Unlike the global measures in
:mod:`~.GraphUtils` (mu-distance, hardwired / softwired clusters,
Robinson--Foulds, ...), which score a network in bulk and are therefore
dominated by its overwhelmingly tree-like structure, the measures here
isolate and score the *reticulation content* of a network -- the hybridization
/ introgression / horizontal-transfer events a biologist runs an inference
method to detect in the first place.

Pipeline (following Section 9 of the note):

1. **Leaf-descendant sets** ``L(x)`` -- the hardwired cluster of every node,
   computed in one reverse-topological pass (:func:`_leaf_descendant_sets`).
2. **Reticulation tripartitions** ``(A_h, B_h, C_h)`` -- one per reticulation
   node (:func:`reticulation_tripartitions`, Definition 1).
3. **Tripartition dissimilarity** ``delta`` -- a parent-symmetry-invariant
   comparison of two tripartitions (:func:`tripartition_dissimilarity`,
   Equation 3).
4. **Reticulation dissimilarity** ``D_ret`` -- a minimum-weight bipartite
   matching of the two tripartition multisets, padded with a deletion penalty
   ``rho`` for unmatched reticulations (Equation 4), solved with the Hungarian
   algorithm.
5. **Precision / recall / F1** localized to reticulation events (Definition 4).
6. **Global metric** ``D`` -- Nakhleh's metric on reduced networks (Equation 5)
   via bottom-up canonical labeling (:func:`nakhleh_metric`), and the convex
   combination ``D_lambda = lambda * D_ret + (1 - lambda) * D`` (Equation 6).
7. **Topology-aware refinement** ``D*_ret`` -- each block is compared through
   the induced sub-network via ``D`` rather than its leaf set alone
   (Section 8), controlled by the block weight ``alpha``.

The single public entry point is :func:`compare_networks`, which returns a
:class:`NetworkComparison` bundling ``(D, D_ret_hat, Prec, Rec)`` and the full
matching.  The result object is iterable, yielding exactly the
``(D, D_ret_hat, Prec, Rec)`` tuple of Section 9.

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

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .Network import Network, Node, NetworkError

__all__ = [
    "JACCARD",
    "SYMMETRIC",
    "ReticulationTripartition",
    "NetworkComparison",
    "reticulation_tripartitions",
    "set_distance",
    "tripartition_dissimilarity",
    "nakhleh_metric",
    "reticulation_dissimilarity",
    "reticulation_precision_recall",
    "combined_dissimilarity",
    "compare_networks",
]

# ── Ground set distances (Section 4) ───────────────────────────────────

#: Normalized Jaccard (Marczewski--Steinhaus) set distance in ``[0, 1]``.
JACCARD = "jaccard"

#: Symmetric-difference cardinality set distance (unbounded).
SYMMETRIC = "symmetric"

# The maximum value a single ``delta`` (Eq. 3) can take under the Jaccard
# ground distance: three block comparisons, each in ``[0, 1]``.
_JACCARD_DELTA_MAX = 3.0


def set_distance(P: FrozenSet[str],
                 Q: FrozenSet[str],
                 kind: str = JACCARD) -> float:
    """
    Distance between two finite sets of leaf labels (Equations 1--2).

    Two choices are supported, both of which are metrics on finite sets:

    * :data:`SYMMETRIC` -- the symmetric-difference cardinality
      ``|P triangle Q|``;
    * :data:`JACCARD` -- the normalized Jaccard distance
      ``|P triangle Q| / |P union Q|`` (0 when both sets are empty), which
      takes values in ``[0, 1]``.

    Args:
        P (FrozenSet[str]): First set of leaf labels.
        Q (FrozenSet[str]): Second set of leaf labels.
        kind (str): Either :data:`JACCARD` (default) or :data:`SYMMETRIC`.

    Raises:
        NetworkError: If ``kind`` is not a recognized distance.

    Returns:
        float: The set distance ``d(P, Q)``.
    """
    if kind == SYMMETRIC:
        return float(len(P ^ Q))
    if kind == JACCARD:
        union_size = len(P | Q)
        if union_size == 0:
            return 0.0
        return len(P ^ Q) / union_size
    raise NetworkError(
        f"Unknown set distance '{kind}'. Use '{JACCARD}' or '{SYMMETRIC}'."
    )


# ── Reticulation tripartitions (Section 3) ──────────────────────────────

@dataclass(frozen=True)
class ReticulationTripartition:
    """
    The reticulation tripartition ``theta_h = (A_h, B_h, C_h)`` of a
    reticulation node ``h`` with parents ``u`` and ``v`` (Definition 1).

    * ``B = L(h)`` is the cluster of the hybrid (its leaf descendants);
    * ``A = L(u) \\ L(h)`` are the descendants contributed by parent ``u``
      other than through ``h``;
    * ``C = L(v) \\ L(h)`` likewise for parent ``v``.

    Because ``L(h)`` is a subset of both ``L(u)`` and ``L(v)``, the middle
    block is always disjoint from the outer blocks; the two outer blocks are
    disjoint iff the parents share no descendant except through ``h``
    (Proposition 1).

    There is no canonical order on the two parents, so the triple is only
    defined up to the swap ``(A, B, C) <-> (C, B, A)`` (Remark 2); every
    comparison in this module is invariant under that swap.

    Attributes:
        A (FrozenSet[str]): Outer block from the first parent.
        B (FrozenSet[str]): Hybrid cluster ``L(h)``.
        C (FrozenSet[str]): Outer block from the second parent.
        hybrid (str): Label of the reticulation node ``h``.
    """

    A: FrozenSet[str]
    B: FrozenSet[str]
    C: FrozenSet[str]
    hybrid: str = ""

    def is_proper(self) -> bool:
        """
        Whether the three blocks are pairwise disjoint (Proposition 1).

        The tripartition is proper iff the two parents share no descendant
        outside the hybrid, i.e. ``A`` and ``C`` are disjoint (the middle
        block is disjoint from the outer blocks unconditionally).

        Returns:
            bool: True if ``A``, ``B``, ``C`` are pairwise disjoint.
        """
        return len(self.A & self.C) == 0

    def swapped(self) -> "ReticulationTripartition":
        """
        Return the parent-swapped tripartition ``(C, B, A)`` (Remark 2).

        Returns:
            ReticulationTripartition: The swapped triple.
        """
        return ReticulationTripartition(self.C, self.B, self.A, self.hybrid)


def _leaf_descendant_sets(net: Network) -> Dict[Node, FrozenSet[str]]:
    """
    Compute the leaf-descendant (hardwired-cluster) set ``L(x)`` for every
    node of the network in one reverse-topological pass (Stage 1).

    Processing nodes from leaves to root, ``L(x)`` is the union of the
    ``L(c)`` over the children ``c`` of ``x`` (with ``L(leaf) = {leaf}``);
    because every child precedes its parent in the reversed topological order,
    a single pass suffices even on a DAG, and reticulations are handled
    automatically by the union.  Unlike
    :func:`phynetpy.GraphUtils._leaf_labels_below`, which descends from a
    single root, this covers every node the topological order reaches.

    Args:
        net (Network): A phylogenetic network (must be acyclic).

    Returns:
        Dict[Node, FrozenSet[str]]: Map from node to its set of descendant
                                    leaf labels.
    """
    leaves = set(net.get_leaves())
    descendants: Dict[Node, FrozenSet[str]] = {}

    for node in reversed(net.topological_order()):
        acc: set = set()
        if node in leaves:
            acc.add(node.label)
        for child in net.get_children(node):
            acc |= descendants[child]
        descendants[node] = frozenset(acc)

    return descendants


def reticulation_tripartitions(net: Network) -> List[ReticulationTripartition]:
    """
    Compute the reticulation tripartitions of a network (Definition 1).

    One tripartition ``(A_h, B_h, C_h)`` is produced per reticulation node
    ``h``.  A tree (no reticulations) yields the empty list.

    Args:
        net (Network): A phylogenetic network.  Every reticulation node is
                       assumed binary (in-degree exactly 2).

    Raises:
        NetworkError: If a reticulation node does not have exactly two
                      parents, i.e. the binary-network assumption is violated.

    Returns:
        List[ReticulationTripartition]: One tripartition per reticulation,
                                        as a (multiset-representing) list.
    """
    descendants = _leaf_descendant_sets(net)
    tripartitions: List[ReticulationTripartition] = []

    for node in net.V():
        if not node.is_reticulation():
            continue

        parents = net.get_parents(node)
        if len(parents) != 2:
            raise NetworkError(
                f"Reticulation node '{node.label}' has {len(parents)} "
                "parent(s); the tripartition measure requires binary "
                "reticulations with exactly 2 parents."
            )

        u, v = parents
        b_block = descendants[node]
        a_block = descendants[u] - b_block
        c_block = descendants[v] - b_block
        tripartitions.append(
            ReticulationTripartition(a_block, b_block, c_block, node.label)
        )

    return tripartitions


# ── Tripartition dissimilarity delta (Section 4) ────────────────────────

def _block_distance_factory(
    net_ref: Network,
    net_inf: Network,
    kind: str,
    alpha: float,
) -> Callable[[FrozenSet[str], FrozenSet[str]], float]:
    """
    Build the block distance ``d`` (or its topology-aware refinement ``d*``)
    used inside :func:`tripartition_dissimilarity`.

    For ``alpha == 1`` this is the plain leaf-set distance
    :func:`set_distance` (Sections 4--5).  For ``alpha < 1`` it is the
    tri-structure block distance of Definition 5,

        ``d*(S, S') = alpha * d(S, S') + (1 - alpha) * d_top(S, S')``,

    where ``d_top(S, S') = D(N_ref | S∩S', N_inf | S∩S')`` compares the induced
    sub-networks on the shared leaves with the global metric ``D`` (Eq. 5).
    The left argument always originates from ``net_ref`` and the right from
    ``net_inf``.

    Args:
        net_ref (Network): Reference network (source of "left" blocks).
        net_inf (Network): Inferred network (source of "right" blocks).
        kind (str): Ground set distance, :data:`JACCARD` or :data:`SYMMETRIC`.
        alpha (float): Block weight in ``[0, 1]``; 1 recovers the leaf-set
                       measure, values below 1 mix in substructure topology.

    Returns:
        Callable[[FrozenSet[str], FrozenSet[str]], float]: The block distance.
    """
    if alpha >= 1.0:
        return lambda P, Q: set_distance(P, Q, kind)

    # Imported lazily to avoid a circular import at module load time
    # (GraphUtils re-exports this module).
    from .GraphUtils import induced_subnetwork_by_taxa

    def block_distance(P: FrozenSet[str], Q: FrozenSet[str]) -> float:
        base = set_distance(P, Q, kind)
        shared = P & Q
        if len(shared) < 2:
            # A single shared leaf (or none) induces a trivial, identical
            # substructure, so the topological term contributes nothing.
            d_top = 0.0
        else:
            taxa = sorted(shared)
            sub_ref = induced_subnetwork_by_taxa(net_ref, taxa)
            sub_inf = induced_subnetwork_by_taxa(net_inf, taxa)
            d_top = nakhleh_metric(sub_ref, sub_inf)
        return alpha * base + (1.0 - alpha) * d_top

    return block_distance


def _delta(
    theta: ReticulationTripartition,
    theta_prime: ReticulationTripartition,
    block_distance: Callable[[FrozenSet[str], FrozenSet[str]], float],
) -> float:
    """
    Evaluate the tripartition dissimilarity ``delta`` (Equation 3) given a
    precomputed block distance.

    The two hybrid clusters (middle blocks) are always compared to each other;
    the outer blocks are compared under whichever of the two parent-to-parent
    correspondences is cheaper, discharging the parent symmetry of Remark 2.

    Args:
        theta (ReticulationTripartition): Reference tripartition
                                          ``(A, B, C)``.
        theta_prime (ReticulationTripartition): Inferred tripartition
                                                ``(D, E, F)``.
        block_distance (Callable): Distance on individual blocks (left from the
                                   reference network, right from the inferred).

    Returns:
        float: ``delta(theta, theta_prime)``.
    """
    middle = block_distance(theta.B, theta_prime.B)
    direct = (block_distance(theta.A, theta_prime.A)
              + block_distance(theta.C, theta_prime.C))
    swapped = (block_distance(theta.A, theta_prime.C)
               + block_distance(theta.C, theta_prime.A))
    return middle + min(direct, swapped)


def tripartition_dissimilarity(
    theta: ReticulationTripartition,
    theta_prime: ReticulationTripartition,
    kind: str = JACCARD,
    normalize: bool = False,
) -> float:
    """
    Tripartition dissimilarity ``delta`` between two reticulations
    (Equation 3, the leaf-set variant of Sections 4--5).

    ``delta`` is a metric on tripartitions (up to parent swap) whenever the
    ground set distance is a metric (Proposition 2).  Under the Jaccard ground
    distance ``delta`` lies in ``[0, 3]``; ``normalize=True`` divides by 3 to
    give ``delta_bar`` in ``[0, 1]`` (Remark 3).

    Args:
        theta (ReticulationTripartition): First tripartition.
        theta_prime (ReticulationTripartition): Second tripartition.
        kind (str): Ground set distance, :data:`JACCARD` or :data:`SYMMETRIC`.
        normalize (bool): If True (and ``kind`` is Jaccard), divide by 3 to
                          return ``delta_bar`` in ``[0, 1]``.

    Returns:
        float: ``delta`` (or ``delta_bar`` when normalized).
    """
    block_distance = lambda P, Q: set_distance(P, Q, kind)
    value = _delta(theta, theta_prime, block_distance)
    if normalize:
        return value / _JACCARD_DELTA_MAX
    return value


# ── Global metric D (Section 7, Equation 5) ─────────────────────────────

def _canonical_key_multiset(
    net: Network,
    interner: Dict[Tuple, int],
) -> Counter:
    """
    Canonically label every node bottom-up and return the multiset of keys.

    Two nodes receive the same key iff they are leaves with the same label or
    have pairwise-equivalent children (a bisimulation on the rooted sub-DAGs,
    in the style of the Aho--Hopcroft--Ullman tree-isomorphism procedure).
    The multiplicity of a key is Nakhleh's ``kappa`` -- the number of nodes
    equivalent to a given one.

    The ``interner`` is shared across the two networks being compared so that
    identical substructures map to the same integer key in both, making the
    two returned multisets directly comparable.

    Args:
        net (Network): A phylogenetic network.
        interner (Dict[Tuple, int]): Shared signature -> integer-key table,
                                     mutated in place.

    Returns:
        Counter: Multiset mapping each integer key to its multiplicity in
                 ``net``.
    """
    key_of: Dict[Node, int] = {}
    counts: Counter = Counter()

    for node in reversed(net.topological_order()):
        children = net.get_children(node)
        if not children:
            signature: Tuple = ("leaf", node.label)
        else:
            signature = ("node", tuple(sorted(key_of[c] for c in children)))
        key = interner.setdefault(signature, len(interner))
        key_of[node] = key
        counts[key] += 1

    return counts


def nakhleh_metric(net1: Network,
                   net2: Network,
                   normalize: bool = False) -> float:
    """
    Nakhleh's global metric ``D`` on reduced phylogenetic networks
    (Equation 5).

    ``D`` is half the cardinality of the symmetric difference of the two
    multisets of rooted subnetworks (each subnetwork counted with its
    multiplicity ``kappa``):

        ``D(N1, N2) = 0.5 * sum_key |kappa_N1(key) - kappa_N2(key)|``.

    It is a metric on reduced networks, tree-child networks, semibinary
    tree-sibling time-consistent networks, and multilabeled trees, computable
    in polynomial time, and reduces on trees to half the symmetric difference
    of their rooted-subtree multisets.  Being global, it separates networks
    but is dominated by tree-like structure -- which is precisely why the note
    pairs it with the reticulation-aware :func:`reticulation_dissimilarity`.

    Args:
        net1 (Network): First network.
        net2 (Network): Second network.
        normalize (bool): If True, return the bounded score
                          ``D_hat = 2 D / (|V1| + |V2|)`` in ``[0, 1]``.

    Returns:
        float: ``D`` (or ``D_hat`` when normalized).
    """
    interner: Dict[Tuple, int] = {}
    counts1 = _canonical_key_multiset(net1, interner)
    counts2 = _canonical_key_multiset(net2, interner)

    keys = set(counts1) | set(counts2)
    sym_diff = sum(abs(counts1.get(k, 0) - counts2.get(k, 0)) for k in keys)
    distance = 0.5 * sym_diff

    if normalize:
        denom = len(net1.V()) + len(net2.V())
        return (2.0 * distance / denom) if denom > 0 else 0.0

    return distance


# ── Reticulation dissimilarity D_ret (Section 5, Equation 4) ────────────

def _leaf_universe_size(net_ref: Network, net_inf: Network) -> int:
    """
    Size of the combined taxon set of the two networks.

    Args:
        net_ref (Network): Reference network.
        net_inf (Network): Inferred network.

    Returns:
        int: ``|X_ref union X_inf|``.
    """
    labels = {leaf.label for leaf in net_ref.get_leaves()}
    labels |= {leaf.label for leaf in net_inf.get_leaves()}
    return len(labels)


def _delta_max(kind: str, rho: float, universe_size: int) -> float:
    """
    The maximum per-pair cost used to normalize ``D_ret`` (Definition 3).

    Under the Jaccard ground distance a real--real ``delta`` is at most 3;
    under the symmetric-difference distance it is at most ``3 |X|``.  The
    deletion penalty ``rho`` is folded in so that an unmatched reticulation is
    never "more than maximally" wrong.

    Args:
        kind (str): Ground set distance.
        rho (float): Deletion penalty.
        universe_size (int): ``|X|`` for the pair of networks.

    Returns:
        float: ``delta_max``.
    """
    base = _JACCARD_DELTA_MAX if kind == JACCARD else 3.0 * universe_size
    return max(base, rho)


def _default_rho(kind: str, universe_size: int) -> float:
    """
    The default deletion penalty ``rho`` ("an unmatched reticulation is
    maximally wrong", Proposition 3): the maximum real--real ``delta``.

    Args:
        kind (str): Ground set distance.
        universe_size (int): ``|X|`` for the pair of networks.

    Returns:
        float: Default ``rho`` (3 under Jaccard, ``3 |X|`` under symmetric).
    """
    return _JACCARD_DELTA_MAX if kind == JACCARD else 3.0 * universe_size


def _assign(
    tripartitions_ref: List[ReticulationTripartition],
    tripartitions_inf: List[ReticulationTripartition],
    block_distance: Callable[[FrozenSet[str], FrozenSet[str]], float],
    rho: float,
) -> Tuple[float, List[Tuple[Optional[int], Optional[int], float]]]:
    """
    Solve the padded minimum-weight bipartite matching of Equation 4.

    The two tripartition multisets are padded with the formal "no counterpart"
    symbol to a common size ``m = max(r1, r2)``; a real reticulation paired
    with padding costs ``rho`` and two paddings cost 0.  The Hungarian
    algorithm returns the optimal assignment.

    Args:
        tripartitions_ref (List[ReticulationTripartition]): Reference
            tripartitions (``r1`` of them).
        tripartitions_inf (List[ReticulationTripartition]): Inferred
            tripartitions (``r2`` of them).
        block_distance (Callable): Block distance for ``delta``.
        rho (float): Deletion penalty.

    Returns:
        Tuple[float, List[Tuple[Optional[int], Optional[int], float]]]:
            ``(D_ret, matching)`` where ``matching`` lists
            ``(ref_index, inf_index, cost)`` triples.  A padding slot is
            represented by ``None``; pure padding--padding pairs are omitted.
    """
    r1 = len(tripartitions_ref)
    r2 = len(tripartitions_inf)
    m = max(r1, r2)

    if m == 0:
        return 0.0, []

    cost = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(m):
            if i < r1 and j < r2:
                cost[i, j] = _delta(
                    tripartitions_ref[i],
                    tripartitions_inf[j],
                    block_distance,
                )
            elif i < r1 or j < r2:
                # Exactly one side is a real reticulation -> deletion penalty.
                cost[i, j] = rho
            else:
                # Both sides are padding -> free.
                cost[i, j] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)
    total = float(cost[row_ind, col_ind].sum())

    matching: List[Tuple[Optional[int], Optional[int], float]] = []
    for i, j in zip(row_ind.tolist(), col_ind.tolist()):
        ref_idx = i if i < r1 else None
        inf_idx = j if j < r2 else None
        if ref_idx is None and inf_idx is None:
            continue
        matching.append((ref_idx, inf_idx, float(cost[i, j])))

    return total, matching


# ── Result bundle and public entry point (Section 9) ────────────────────

@dataclass
class NetworkComparison:
    """
    The result of :func:`compare_networks`: a reticulation-aware comparison of
    a reference network against an inferred one.

    Iterating the object yields exactly the ``(D, D_ret_hat, Prec, Rec)``
    tuple that Section 9 specifies as the library's score entry point, so::

        D, D_ret_hat, prec, rec = compare_networks(reference, inferred)

    while the remaining fields expose the unnormalized metrics (for callers
    that rely on the metric axioms; Remark 4), the matching, and the recovery
    counts.

    Attributes:
        D (float): Global metric ``D`` (unnormalized; Equation 5).
        D_hat (float): Bounded global score ``2 D / (|V1| + |V2|)`` in
                       ``[0, 1]``.
        D_ret (float): Reticulation dissimilarity ``D_ret`` (unnormalized
                       metric on tripartition multisets; Equation 4).
        D_ret_hat (float): Normalized reticulation dissimilarity
                           ``D_ret / (m * delta_max)`` in ``[0, 1]``.
        precision (float): Reticulation precision ``g / r(N_inf)``.
        recall (float): Reticulation recall ``g / r(N_ref)``.
        f1 (float): Harmonic mean of precision and recall.
        r_reference (int): Reticulation number of the reference network.
        r_inferred (int): Reticulation number of the inferred network.
        recovered (int): Number ``g`` of recovered reticulation pairs
                         (matched real--real at cost ``<= tolerance``).
        matching (List[Tuple[Optional[int], Optional[int], float]]): The
            optimal matching as ``(ref_index, inf_index, cost)`` triples;
            ``None`` marks a deletion against the padding symbol.
        distance (str): Ground set distance used.
        rho (float): Deletion penalty used.
        tolerance (float): Recovery tolerance ``tau`` used.
        alpha (float): Block weight used (1 = leaf-set; < 1 = topology-aware).
        delta_max (float): Per-pair normalizer used for ``D_ret_hat``.
    """

    D: float
    D_hat: float
    D_ret: float
    D_ret_hat: float
    precision: float
    recall: float
    f1: float
    r_reference: int
    r_inferred: int
    recovered: int
    matching: List[Tuple[Optional[int], Optional[int], float]] = field(
        default_factory=list
    )
    distance: str = JACCARD
    rho: float = _JACCARD_DELTA_MAX
    tolerance: float = 0.0
    alpha: float = 1.0
    delta_max: float = _JACCARD_DELTA_MAX

    def __iter__(self):
        """Yield the ``(D, D_ret_hat, Prec, Rec)`` tuple of Section 9."""
        yield self.D
        yield self.D_ret_hat
        yield self.precision
        yield self.recall

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """
        Return the ``(D, D_ret_hat, Prec, Rec)`` score tuple.

        Returns:
            Tuple[float, float, float, float]: The four headline scores.
        """
        return (self.D, self.D_ret_hat, self.precision, self.recall)

    def combined(self, lam: float) -> float:
        """
        The unnormalized convex combination ``D_lambda`` (Equation 6).

        ``D_lambda = lambda * D_ret + (1 - lambda) * D`` is a metric for every
        ``lambda < 1`` on any class where ``D`` is a metric, and a
        pseudometric at ``lambda = 1`` (Proposition 5).

        Args:
            lam (float): Weight ``lambda`` in ``[0, 1]`` on the reticulation
                         term.

        Returns:
            float: ``D_lambda(N1, N2)``.
        """
        return lam * self.D_ret + (1.0 - lam) * self.D

    def combined_normalized(self, lam: float) -> float:
        """
        The bounded reporting combination
        ``lambda * D_ret_hat + (1 - lambda) * D_hat`` (Remark 4).

        This trades the triangle inequality for a value in ``[0, 1]``.

        Args:
            lam (float): Weight ``lambda`` in ``[0, 1]`` on the reticulation
                         term.

        Returns:
            float: The bounded combined score.
        """
        return lam * self.D_ret_hat + (1.0 - lam) * self.D_hat


def compare_networks(
    reference: Network,
    inferred: Network,
    distance: str = JACCARD,
    rho: Optional[float] = None,
    tolerance: float = 0.0,
    alpha: float = 1.0,
) -> NetworkComparison:
    """
    Reticulation-aware comparison of a reference network against an inferred
    one -- the single score entry point of Section 9.

    Computes, in one pass over the reticulations, the global metric ``D``
    (Eq. 5), the reticulation dissimilarity ``D_ret`` and its normalized form
    ``D_ret_hat`` (Eq. 4), and reticulation-level precision / recall / F1
    (Definition 4).  The returned :class:`NetworkComparison` is iterable,
    yielding the headline ``(D, D_ret_hat, Prec, Rec)`` tuple.

    Reference / inferred orientation matters for precision and recall: recall
    is ``g / r(reference)`` (fraction of true events recovered) and precision
    is ``g / r(inferred)`` (fraction of inferred events that are correct),
    with the conventions ``Prec = 1`` when ``r(inferred) = 0`` and
    ``Rec = 1`` when ``r(reference) = 0`` (Definition 4).  ``D`` and ``D_ret``
    themselves are symmetric.

    Boundary behaviour (Section 6.1): two trees give ``D_ret_hat = 0``; a
    network compared against a tree gives ``D_ret_hat = 1`` (every
    reticulation is unmatched); self-comparison gives all zeros with
    ``Prec = Rec = 1``.

    Args:
        reference (Network): The reference (true) network ``N1``.
        inferred (Network): The inferred network ``N2``.
        distance (str): Ground set distance, :data:`JACCARD` (default) or
                        :data:`SYMMETRIC`.
        rho (Optional[float]): Deletion penalty for an unmatched reticulation.
                               Defaults to the maximum real--real ``delta``
                               (3 under Jaccard), i.e. "maximally wrong".
        tolerance (float): Recovery tolerance ``tau >= 0``; a matched pair
                           counts as recovered iff its cost is ``<= tau``.
                           ``tau = 0`` demands exact recovery of the hybrid and
                           both parents.
        alpha (float): Block weight in ``[0, 1]`` for the topology-aware
                       refinement (Section 8).  ``alpha = 1`` (default) is the
                       leaf-set measure; ``alpha < 1`` mixes in the induced
                       sub-network distance ``D``.

    Raises:
        NetworkError: If a distance name is unrecognized, if ``alpha`` /
                      ``tolerance`` are out of range, or if a reticulation is
                      not binary.

    Returns:
        NetworkComparison: The bundled scores and optimal matching.
    """
    if distance not in (JACCARD, SYMMETRIC):
        raise NetworkError(
            f"Unknown set distance '{distance}'. "
            f"Use '{JACCARD}' or '{SYMMETRIC}'."
        )
    if not 0.0 <= alpha <= 1.0:
        raise NetworkError(f"alpha must lie in [0, 1]; got {alpha}.")
    if tolerance < 0.0:
        raise NetworkError(f"tolerance (tau) must be >= 0; got {tolerance}.")

    universe_size = _leaf_universe_size(reference, inferred)
    if rho is None:
        rho = _default_rho(distance, universe_size)
    if rho < 0.0:
        raise NetworkError(f"rho must be >= 0; got {rho}.")
    delta_max = _delta_max(distance, rho, universe_size)

    tripartitions_ref = reticulation_tripartitions(reference)
    tripartitions_inf = reticulation_tripartitions(inferred)
    r1 = len(tripartitions_ref)
    r2 = len(tripartitions_inf)
    m = max(r1, r2)

    block_distance = _block_distance_factory(
        reference, inferred, distance, alpha
    )
    d_ret, matching = _assign(
        tripartitions_ref, tripartitions_inf, block_distance, rho
    )
    d_ret_hat = (d_ret / (m * delta_max)) if m > 0 else 0.0

    # A pair is "recovered" when both sides are real and matched at cost <= tau.
    recovered = sum(
        1
        for ref_idx, inf_idx, cost in matching
        if ref_idx is not None and inf_idx is not None and cost <= tolerance
    )

    precision = 1.0 if r2 == 0 else recovered / r2
    recall = 1.0 if r1 == 0 else recovered / r1
    f1 = (0.0 if precision + recall == 0.0
          else 2.0 * precision * recall / (precision + recall))

    d_global = nakhleh_metric(reference, inferred)
    denom = len(reference.V()) + len(inferred.V())
    d_global_hat = (2.0 * d_global / denom) if denom > 0 else 0.0

    return NetworkComparison(
        D=d_global,
        D_hat=d_global_hat,
        D_ret=d_ret,
        D_ret_hat=d_ret_hat,
        precision=precision,
        recall=recall,
        f1=f1,
        r_reference=r1,
        r_inferred=r2,
        recovered=recovered,
        matching=matching,
        distance=distance,
        rho=rho,
        tolerance=tolerance,
        alpha=alpha,
        delta_max=delta_max,
    )


# ── Thin convenience wrappers ───────────────────────────────────────────

def reticulation_dissimilarity(
    net1: Network,
    net2: Network,
    distance: str = JACCARD,
    rho: Optional[float] = None,
    alpha: float = 1.0,
    normalize: bool = True,
) -> float:
    """
    Reticulation dissimilarity ``D_ret`` between two networks (Equation 4).

    A convenience wrapper over :func:`compare_networks` that returns just the
    scalar.  ``D_ret`` is a pseudometric on networks (a metric on
    reticulation-tripartition multisets) under the penalty condition of
    Proposition 3, and is deliberately blind to tree-only differences.

    Args:
        net1 (Network): First network.
        net2 (Network): Second network.
        distance (str): Ground set distance, :data:`JACCARD` or
                        :data:`SYMMETRIC`.
        rho (Optional[float]): Deletion penalty; see :func:`compare_networks`.
        alpha (float): Block weight for the topology-aware refinement.
        normalize (bool): If True (default), return the bounded ``D_ret_hat``
                          in ``[0, 1]``; otherwise the unnormalized metric
                          ``D_ret``.

    Returns:
        float: ``D_ret_hat`` (normalized) or ``D_ret`` (unnormalized).
    """
    result = compare_networks(
        net1, net2, distance=distance, rho=rho, alpha=alpha
    )
    return result.D_ret_hat if normalize else result.D_ret


def reticulation_precision_recall(
    reference: Network,
    inferred: Network,
    distance: str = JACCARD,
    rho: Optional[float] = None,
    tolerance: float = 0.0,
    alpha: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Reticulation precision, recall, and F1 (Definition 4).

    A convenience wrapper over :func:`compare_networks`.  With ``reference``
    the true network and ``inferred`` the estimate, recall is the fraction of
    true reticulations recovered and precision the fraction of inferred
    reticulations that are correct, where "recovered" means matched at cost
    ``<= tolerance``.

    Args:
        reference (Network): The reference (true) network.
        inferred (Network): The inferred network.
        distance (str): Ground set distance.
        rho (Optional[float]): Deletion penalty; see :func:`compare_networks`.
        tolerance (float): Recovery tolerance ``tau >= 0``.
        alpha (float): Block weight for the topology-aware refinement.

    Returns:
        Tuple[float, float, float]: ``(precision, recall, f1)``.
    """
    result = compare_networks(
        reference,
        inferred,
        distance=distance,
        rho=rho,
        tolerance=tolerance,
        alpha=alpha,
    )
    return (result.precision, result.recall, result.f1)


def combined_dissimilarity(
    net1: Network,
    net2: Network,
    lam: float = 0.5,
    distance: str = JACCARD,
    rho: Optional[float] = None,
    alpha: float = 1.0,
    normalize: bool = False,
) -> float:
    """
    The convex combination ``D_lambda`` of the reticulation and global terms
    (Equation 6).

    ``D_lambda = lambda * D_ret + (1 - lambda) * D`` interpolates between the
    global comparison (``lambda -> 0``) and the reticulation-aware one
    (``lambda -> 1``).  On any class where ``D`` is a metric it is a metric for
    every ``lambda < 1`` and a pseudometric at ``lambda = 1`` (Proposition 5).

    Args:
        net1 (Network): First network.
        net2 (Network): Second network.
        lam (float): Weight ``lambda`` in ``[0, 1]`` on the reticulation term.
        distance (str): Ground set distance.
        rho (Optional[float]): Deletion penalty; see :func:`compare_networks`.
        alpha (float): Block weight for the topology-aware refinement.
        normalize (bool): If True, combine the bounded scores
                          ``lambda * D_ret_hat + (1 - lambda) * D_hat`` in
                          ``[0, 1]`` (Remark 4); otherwise the unnormalized
                          metrics.

    Raises:
        NetworkError: If ``lam`` is outside ``[0, 1]``.

    Returns:
        float: ``D_lambda`` (unnormalized) or its bounded reporting form.
    """
    if not 0.0 <= lam <= 1.0:
        raise NetworkError(f"lambda must lie in [0, 1]; got {lam}.")
    result = compare_networks(
        net1, net2, distance=distance, rho=rho, alpha=alpha
    )
    return result.combined_normalized(lam) if normalize else result.combined(lam)
