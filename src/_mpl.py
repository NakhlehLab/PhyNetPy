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
Last Stable Edit : 4/24/26
First Included in Version : 0.3.2

Docs   - [x]
Tests  - [ ]
Design - [x]

Maximum pseudo-likelihood (MPL) for phylogenetic network inference.

Implements the scoring function of

    Yu, Y. & Nakhleh, L. (2015). "A maximum pseudo-likelihood approach
    for phylogenetic networks." BMC Genomics, 16(S10), S10.

whose objective (to maximise) is

    log L(Psi, gamma | G)
        = sum_{X,Y,Z} [
              rho(XY|Z) * log P(XY|Z | Psi, gamma)
            + rho(XZ|Y) * log P(XZ|Y | Psi, gamma)
            + rho(YZ|X) * log P(YZ|X | Psi, gamma)
          ]

where Psi is the candidate species network, gamma is the vector of
reticulation inheritance probabilities, G is the input gene-tree set,
rho(·|·) are empirical triplet frequencies computed from G, and
P(·|·) are expected triplet probabilities under the coalescent model
on Psi.

Pipeline overview
-----------------
End-to-end control flow when a user runs ``MPL.search(...)``:

    1. Enumerate all 3-taxon species triplets from the species labels.

    2. Precompute rho (triplet frequencies from gene trees) once.
       Driver:  ``compute_gene_tree_triplets`` -> ``_compute_all_rhos_fast``
                -> ``_process_tree_batch`` (optionally parallel via
                ``ProcessPoolExecutor``).  Uses ``_GeneTreeLCAIndex``
                for O(1) topology queries per triplet per tree.

    3. Build an initial ``Model`` wrapping the starting species network.
       Attach an ``MPLScorer`` so the search driver can call
       ``score(model)`` to get log-pseudo-likelihood.

    4. Run the search driver (``HillClimbing`` or ``SimulatedAnnealing``)
       using ``MPLKernel`` to propose moves.  Each accepted proposal
       mutates the model's network in place; the kernel adapts move
       weights and continuous proposal sigmas from rolling per-move
       statistics.

    5. After the search, report final score, move-kernel stats, and
       (optionally) a reference-network comparison.

Scoring path (per call to ``MPLScorer.__call__``):

    Model.network  ->  _TripleDPEngine(network)  ->  either
      - _score_with_cython(engine, triplets, rho)  (fast path), or
      - Python fallback: engine.calculate_triple_probability(triplet)
        summed over all active triplets.

The Cython path and the Python path compute the same quantity; the
Python path is the reference implementation kept for portability and
debugging.  ``_HAS_CYTHON_MPL`` gates which one is used.

Module layout
-------------
Sections are marked with banner comments.  Top-to-bottom:

    (A) Result containers                  (GeneTreeTripletResult, ...)
    (B) Reference-comparison helpers       (format_mpl_reference_comparison)
    (C) Rho precomputation                 (_GeneTreeLCAIndex, _compute_all_rhos_fast)
    (D) Subnetwork probability DP engine   (_TripleDPEngine)
    (E) Cython bridge                      (_extract_topology_for_cython)
    (F) Public triplet API                 (compute_gene_tree_triplets, score_species_network_triplets)
    (G) Scorer + adaptive proposal kernel  (MPLScorer, _AdaptiveConfig, MPLKernel)
    (H) Orchestration class                (MPL)
"""
 
from __future__ import annotations

import copy
import math
import os
import warnings
from collections import deque
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np

from .Network import Network, Node
from .GeneTrees import GeneTrees
from . import IO as io
from .ModelGraph import Model
from .GraphUtils import count_reticulations, _hardwired_clusters_by_label
from .MetropolisHastings import ProposalKernel, HillClimbing, SimulatedAnnealing
from .ModelMove import (
    Move,
    SPR,
    ChangeNodeHeight,
    ChangeInheritanceProb,
    AddReticulation,
    RemoveReticulation,
    FlipReticulation,
    ChangeReticSource,
    ChangeReticDest,
    RelocateReticulation,
)
from ._optimize import optimize_network_parameters
from ._search_flags import (
    resolve_move_types, make_level_validator, make_containment_validator,
    CONTINUOUS_MOVES, resolve_search_preset,
)
 
# log(p) for zero-probability triplet outcomes is replaced by this
# floor to keep the score finite.  math.log(1e-200) ≈ -460; any triplet
# the DP engine assigns probability 0 is effectively penalised by this
# amount per unit of observed rho mass.  Chosen large enough that the
# search is strongly deterred from such topologies but not so large
# that it single-handedly dominates the score.
_LOG_FLOOR = math.log(1e-200)

# Cython backend detection.  The ``.so`` must be built for the
# interpreter's ABI; see README for the one-liner
# ``python setup.py build_ext --inplace``.  When the import fails,
# the pure-Python DP path in ``_TripleDPEngine`` is used instead -- both
# paths compute the same score.
try:
    from .cython.mpl_engine_cy import score_all_triplets as _cy_score
    _HAS_CYTHON_MPL = True
except ImportError:
    _HAS_CYTHON_MPL = False


# ======================================================================
# (A) Result containers
# ======================================================================

class GeneTreeTripletResult:
    """Container for precomputed gene-tree triplet frequencies (rho values).

    Returned by :func:`compute_gene_tree_triplets` and consumed by
    :func:`score_species_network_triplets` (or stored on :class:`MPL`
    for repeated scoring against varying species networks).  Encodes
    the "gene-tree side" of the MPL inputs: all rho values are
    functions of the observed gene trees only and never change during
    a network search.

    Attributes:
        triplets: Canonical ordered species triplets, e.g.
            ``[('A', 'B', 'C'), ('A', 'B', 'D'), ...]``.
        rho_by_triplet: For each triplet ``(X, Y, Z)``, the empirical
            frequencies of the three rooted resolutions as a tuple
            ``(rho_XY|Z, rho_XZ|Y, rho_YZ|X)``.  Values sum to
            approximately the number of gene trees that covered all
            three taxa (ties contribute 1/3 to each resolution).
    """

    def __init__(
        self,
        triplets: list[tuple[str, str, str]],
        rho_by_triplet: dict[tuple[str, str, str], tuple[float, float, float]],
    ) -> None:
        """Initialise the container.

        Args:
            triplets: Canonical species triplets.
            rho_by_triplet: Mapping ``triplet -> (rho_XY|Z, rho_XZ|Y,
                rho_YZ|X)``.
        """
        self.triplets: list[tuple[str, str, str]] = triplets
        self.rho_by_triplet: dict[
            tuple[str, str, str], tuple[float, float, float]
        ] = rho_by_triplet


class SpeciesNetworkTripletResult:
    """Container for species-network triplet probabilities and final score.

    Returned by :func:`score_species_network_triplets`.  Holds the
    "species-network side" of the scoring call: per-triplet predicted
    probabilities plus the summed log-pseudo-likelihood.

    Attributes:
        triplets: Canonical ordered species triplets (matching the
            order used during scoring).
        probs_by_triplet: For each triplet, predicted probabilities
            ``(P(XY|Z), P(XZ|Y), P(YZ|X))`` under the DP model.
        log_pseudo_likelihood: Total MPL score (higher is better;
            always <= 0 since it is a sum of ``rho * log(p)`` terms).
    """

    def __init__(
        self,
        triplets: list[tuple[str, str, str]],
        probs_by_triplet: dict[tuple[str, str, str], tuple[float, float, float]],
        log_pseudo_likelihood: float,
    ) -> None:
        """Initialise the container.

        Args:
            triplets: Canonical species triplets.
            probs_by_triplet: Mapping ``triplet -> (P(XY|Z), P(XZ|Y),
                P(YZ|X))`` under the DP model.
            log_pseudo_likelihood: Final pseudo-likelihood score.
        """
        self.triplets: list[tuple[str, str, str]] = triplets
        self.probs_by_triplet: dict[
            tuple[str, str, str], tuple[float, float, float]
        ] = probs_by_triplet
        self.log_pseudo_likelihood: float = log_pseudo_likelihood


# ======================================================================
# (B) Reference-comparison helpers
#
# These are only used when ``MPL.search(reference_network=...)`` is set
# (e.g. when benchmarking against a simulator ground truth).  They
# don't participate in the scoring path.
# ======================================================================

def format_mpl_reference_comparison(
    found: Network,
    reference: Network,
    rho: dict[tuple[str, str, str], tuple[float, float, float]],
    triplets: list[tuple[str, str, str]],
    *,
    top_k: int = 25,
) -> str:
    """Build a text report comparing a found network against a reference.

    The report covers four sections:

      1. Final MPL scores for both networks and the signed gap.
      2. Per-network reticulation summary (parent labels, subtree
         leaves) to eyeball where each reticulation sits.
      3. Clade-level Venn counts: shared, only-in-found, only-in-reference.
      4. Top ``top_k`` triplets where the reference outperforms the
         found network, ranked by log-likelihood gap -- useful for
         diagnosing *why* the found topology under-fits.

    Uses the same ``rho`` and ``triplets`` passed to :class:`MPL` so
    the reference is scored against the same observed gene-tree
    statistics.

    Args:
        found: Best network returned by the search.
        reference: Ground-truth or baseline network.
        rho: Triplet frequency table (gene-tree side).
        triplets: Canonical triplets to iterate over.
        top_k: How many worst-performing triplets to list.

    Returns:
        Multi-line formatted report string (no trailing newline).
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
        """Append one line per reticulation describing parents + subtree."""
        retics = [n for n in net.V() if net.in_degree(n) > 1]
        lines.append(f"{label} reticulations: {len(retics)}")
        for r in retics:
            sub = sorted(n.label for n in net.leaf_descendants(r))
            pars = [p.label for p in net.get_parents(r)]
            lines.append(f"  {r.label}: parents={pars}, subtree_leaves={sub}")

    _retic_lines(found, "Found")
    _retic_lines(reference, "Reference")
    lines.append("")

    cf = _hardwired_clusters_by_label(found)
    cr = _hardwired_clusters_by_label(reference)
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
    """Serialize ``net`` as Newick to ``path`` (UTF-8, newline-terminated).

    Parent directories are created if they don't exist.

    Args:
        net: Network to serialise.
        path: Output path (str or ``os.PathLike``).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(net.newick() + "\n", encoding="utf-8")


# ======================================================================
# (C) Rho precomputation
#
# "rho" = triplet frequency statistic derived from the gene-tree set.
# For each species triplet (X, Y, Z), we walk every gene tree that
# covers all three taxa, determine the induced rooted topology via
# LCA depths, and accumulate a weight into one of the three
# resolution buckets (XY|Z, XZ|Y, YZ|X).  Each tree contributes a
# unit of mass (split 1/3 across ties if the triplet is a soft
# polytomy).  The resulting rho table is constant for the duration
# of a search, so it's computed once up front.
# ======================================================================

class _GeneTreeLCAIndex:
    """Precomputed LCA index for O(1) triplet topology queries on a gene tree.

    Construction is O(L^2) where L is the number of relevant leaves
    (BFS to build depth + parent maps, then pairwise LCA via parent
    climbing).  After construction, each triplet topology query is
    three dict lookups plus a few comparisons -- no BFS per query.

    For a dataset with T trees and K triplets, naive re-walking costs
    O(T * K * L).  Indexing trades a one-time O(T * L^2) construction
    for O(T * K) queries, which is a big win when K >> L.

    Attributes:
        _pair_depth: ``dict[(label_a, label_b)] -> lca_depth``, stored
            symmetrically in both orderings.
        _leaf_labels: Relevant leaves actually present in this tree
            (the tree may be missing taxa compared to the full set).
    """

    __slots__ = ('_pair_depth', '_leaf_labels')

    def __init__(self, tree: Network, relevant_labels: frozenset[str]) -> None:
        """Index ``tree`` for fast triplet queries over ``relevant_labels``.

        Args:
            tree: A gene tree.
            relevant_labels: Frozen set of allele labels that will
                actually be queried (typically the union of alleles
                across all species in the mapping).  Leaves whose
                labels fall outside this set are skipped, so a tree
                missing some taxa still indexes cleanly.
        """
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
        """Classify the induced topology of ``{x, y, z}`` via cached LCA depths.

        The deepest pairwise LCA names the two taxa that are siblings
        in the induced triplet; the third is the outgroup.  Ties in
        all three pair depths indicate a soft polytomy.

        Args:
            x: Taxon label.
            y: Taxon label.
            z: Taxon label.

        Returns:
            One of ``"xy|z"``, ``"xz|y"``, ``"yz|x"`` (rooted resolution
            strings) or ``"star"`` (polytomy).
        """
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
    """Accumulate partial rho counts over a batch of gene trees.

    Separated from :func:`_compute_all_rhos_fast` so it can be
    dispatched to a ``concurrent.futures`` worker without closing
    over non-picklable state.

    Two code paths are kept for performance:

      * ``identity_mapping=True``: each species maps to exactly one
        allele (no within-species ambiguity), and triplet lookup is a
        straight dict read.  This is the common case on empirical
        datasets that haven't been resampled.

      * ``identity_mapping=False``: species can map to multiple
        alleles, so each species triplet expands to the Cartesian
        product of allele triplets.  Counts are weighted by
        ``1 / (|ax| * |ay| * |az|)`` to give each species triplet
        unit mass regardless of allele count.

    Args:
        batch: Gene trees to process.
        relevant: Flattened set of allele labels used anywhere in the
            mapping (for :class:`_GeneTreeLCAIndex` filtering).
        mapping: Species -> list of allele labels.
        triplets: Canonical species triplets to accumulate over.
        identity_mapping: True iff all species map to a single allele.

    Returns:
        ``dict`` mapping each triplet to a 3-element list
        ``[count_XY|Z, count_XZ|Y, count_YZ|X]``.  The caller merges
        partials across batches.
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
    """Compute the full rho table in one pass over ``gene_trees``.

    Flips the naive "triplet outer, tree inner" loop to "tree outer,
    triplet inner".  Each gene tree is indexed exactly once and the
    index is reused for all triplets -- a large win when the triplet
    count (cubic in species count) dominates the tree count.

    When ``n_workers > 1`` the tree list is split into batches and
    processed with ``ProcessPoolExecutor``; partial rho counts are
    summed across workers.

    Args:
        gene_trees: Collection of gene trees.
        mapping: Species -> list of allele labels.
        triplets: Canonical species triplets.
        n_workers: Worker-process count for the parallel path.  Set
            to 1 (default) for the single-process path, which is
            preferred for small datasets where process-spawn overhead
            dwarfs the compute.

    Returns:
        ``dict`` mapping each triplet to a tuple
        ``(rho_XY|Z, rho_XZ|Y, rho_YZ|X)``.
    """
    # Universe of allele labels that will ever be queried.  Leaves
    # outside this set are skipped by the LCA index constructor.
    relevant = frozenset(a for alleles in mapping.values() for a in alleles)

    # Detect the common "one allele per species" case so the inner
    # loop can skip the allele-product expansion entirely.
    identity_mapping = all(len(v) == 1 for v in mapping.values())

    trees = gene_trees.trees

    # Single-process fast path.  Preferred for small datasets where
    # process-spawn overhead dwarfs the compute.
    if n_workers <= 1 or len(trees) < 4:
        merged = _process_tree_batch(
            trees, relevant, mapping, triplets, identity_mapping
        )
        return {t: tuple(merged[t]) for t in triplets}

    # Parallel path: split trees into roughly equal batches, fan out.
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

    # Merge partial rho counts from each worker.
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


# ======================================================================
# (D) Subnetwork probability DP engine
#
# Computes P(resolution | Psi, gamma) for a single rooted triplet
# against a fixed species network Psi.  The algorithm mirrors
# PhyloNet's pseudo-likelihood code path, which itself follows Yu &
# Nakhleh (2015).
#
# State-space encoding:
#   Each DP state is a small integer 0..10 representing which of the
#   three triplet lineages are present at a node and how they have
#   coalesced so far.  The three transition maps below
#   (_MERGING_MAP / _SPLITTING_MAP / _COALESCING_MAP) encode the
#   combinatorics of lineage merges (tree-node children join up),
#   lineage splits (reticulation node sends each lineage up one of
#   two parent edges), and within-branch coalescent transitions.
#
# State legend (triplet lineages labelled 1, 2, 3):
#    0  nothing                             1  lineage 1 only
#    2  lineage 2 only                      3  lineage 3 only
#    4  lineages 1+2                        5  lineages 1+3
#    6  lineages 2+3                        7  lineages 1+2+3
#    8, 9, 10  intermediate states used when 3 lineages have partially
#              coalesced; differ in how many common ancestors are still
#              distinct.  State 10 (single lineage from triplet) is a
#              terminal articulation state.
#
# Each _Configuration additionally tracks the sequence of
# reticulation-parent choices made so far (indexed by
# reticulation-node id).  Two configurations are "compatible" iff
# they never disagree on any retic-node choice, which is how
# independent lineages avoid double-counting reticulation decisions.
# ======================================================================

class _TripleDPEngine:
    """Dynamic-programming engine for 3-taxon triplet probabilities.

    Each instance is bound to one fixed species network ``Psi``.
    Construction caches topology info (adjacency, articulation nodes,
    branch lengths, inheritance probabilities) so the per-triplet DP
    never has to touch :class:`Network` methods again.

    The engine is called thousands of times during a search (once per
    triplet resolution per score evaluation), so all hot-path data
    structures are kept as plain ``dict`` s keyed by :class:`Node`.
    """

    # Tree-node merge rules.  When a tree node has two children whose
    # DP subtrees currently carry states s1 and s2 (with s1 < s2),
    # (s1, s2) -> target_state gives the merged state.  The state
    # names above describe which lineages each bit pattern encodes.
    _MERGING_MAP = {
        (1, 2): 4,
        (1, 3): 5,
        (1, 6): 7,
        (2, 3): 6,
        (2, 5): 7,
        (3, 4): 7,
        (3, 8): 9,
    }

    # Reticulation-node split rules.  When a reticulation is processed
    # (moving from child toward parents), the current state splits
    # into two sub-states, one for each parent edge.  The list of
    # (split1, split2) pairs enumerates all ways the lineages can be
    # partitioned between the two parents.
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

    # Within-branch coalescent rules.  ``target_state ->
    # [(source_state, (lineages_in, lineages_out))]`` lists every way
    # ``source_state`` can transition to ``target_state`` on a single
    # branch, paired with the number of ancestral lineages entering
    # the branch (``lineages_in``) and the number leaving it
    # (``lineages_out``).  The coalescent probability per branch is
    # computed by ``_gij`` using (length, lineages_in, lineages_out).
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
        """One DP configuration: total probability + retic-choice vector.

        The retic-choice vector has one slot per reticulation in the
        network; each slot is 0 (unvisited) or a positive index
        identifying which parent edge the lineage ascending through
        this reticulation took.  Two configurations can only be
        combined (by :meth:`merge`) if they agree on every shared
        reticulation choice -- that's what :meth:`is_compatible`
        verifies.

        Using a tuple (vs a list) for the index vector matters here:
        tuples are hashable natively, cheaper to copy, and compare
        faster -- all of which show up in the DP hot loop.
        """

        __slots__ = ('total_prob', '_idx')

        def __init__(
            self,
            net_node_num: int,
            total_prob: float = 1.0,
            net_node_index: tuple[int, ...] | None = None,
        ) -> None:
            """Create a new configuration.

            Args:
                net_node_num: Total number of reticulation nodes in
                    the engine's network (i.e. the length of the
                    retic-choice vector).
                total_prob: Accumulated probability mass.  Starts at
                    1.0 at the leaves and is multiplied by branch
                    transition probabilities on the way up.
                net_node_index: Explicit retic-choice vector; when
                    ``None`` a zero vector is allocated.
            """
            self.total_prob = total_prob
            self._idx: tuple[int, ...] = (
                net_node_index if net_node_index is not None
                else (0,) * net_node_num
            )

        @property
        def net_node_index(self) -> tuple[int, ...]:
            """Retic-choice vector (read-only view)."""
            return self._idx

        def copy(self) -> "_TripleDPEngine._Configuration":
            """Return a shallow copy sharing the retic-choice tuple."""
            c = _TripleDPEngine._Configuration.__new__(_TripleDPEngine._Configuration)
            c.total_prob = self.total_prob
            c._idx = self._idx
            return c

        @classmethod
        def merge(
            cls,
            c1: "_TripleDPEngine._Configuration",
            c2: "_TripleDPEngine._Configuration",
        ) -> "_TripleDPEngine._Configuration":
            """Merge two compatible configurations at a tree node.

            The merged retic-choice vector is the element-wise max
            (since one of each pair is guaranteed to be 0 when the
            pair is compatible).  The merged probability is the
            product -- two independent lineages joining.
            """
            merged = tuple(
                max(a, b) for a, b in zip(c1._idx, c2._idx)
            )
            c = cls.__new__(cls)
            c.total_prob = max(0.0, c1.total_prob * c2.total_prob)
            c._idx = merged
            return c

        def is_compatible(self, other: "_TripleDPEngine._Configuration") -> bool:
            """True iff ``self`` and ``other`` never disagree on a retic choice.

            Two configurations are compatible when, for every
            reticulation-index slot, at least one of them is 0
            (unvisited) or both agree on the parent-edge choice.
            """
            for mine, theirs in zip(self._idx, other._idx):
                if mine != theirs and mine != 0 and theirs != 0:
                    return False
            return True

        def add_choice(self, net_id: int, choice: int) -> None:
            """Record a parent-edge choice at reticulation ``net_id``."""
            lst = list(self._idx)
            lst[net_id] = choice
            self._idx = tuple(lst)

        def clear_choices(self) -> None:
            """Zero out every slot in the retic-choice vector.

            Used at articulation nodes where the downstream choices
            no longer matter for the remaining DP (the articulation
            collapses the subtree's retic history).
            """
            self._idx = (0,) * len(self._idx)

        def __hash__(self) -> int:
            # Hash only by retic-choice vector so dedup/merge uses
            # the choice vector as the identity key; probabilities
            # of equal configurations are summed by the caller.
            return hash(self._idx)

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, _TripleDPEngine._Configuration):
                return False
            return self._idx == other._idx

    def __init__(self, network: Network) -> None:
        """Bind the engine to a fixed species network.

        Performs all one-time topology caching up front so the
        per-triplet DP in :meth:`calculate_triple_probability` can
        run without touching :class:`Network` methods.  Includes:

          * a reversed topological order (leaves to root),
          * per-node flags (is-leaf, is-retic),
          * adjacency lists (children, parents),
          * articulation / lowest-articulation node sets,
          * per-edge ``(branch_length, gamma)`` tuples,
          * an empty memoization cache for :meth:`_gij`.

        Args:
            network: Species network to be scored.  Must already
                have valid branch lengths and gamma values on its
                edges -- the engine does not touch :class:`Network`
                for any of that data after ``__init__`` returns.
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

    def refresh_params(self) -> None:
        """Re-read branch lengths / gammas from the bound network in place.

        Lever 3 fast path for the pure-Python scorer: when only continuous
        parameters changed (topology unchanged), the cached adjacency /
        articulation structure is still valid -- only ``_edge_info`` needs
        refreshing.  ``_gij_cache`` is keyed by the branch-length *value*,
        so changed lengths simply miss the old keys; no clearing is needed
        for correctness.
        """
        net = self.network
        for node in self._topo_leaf_to_root:
            for parent in self._node_parents[node]:
                edge = net.get_edge(parent, node)
                self._edge_info[(parent, node)] = (
                    edge.get_length(), edge.get_gamma(),
                )

    @staticmethod
    def _fact(start: int, end: int) -> float:
        """Return the product ``start * (start+1) * ... * end``.

        Used by :meth:`_gij_raw` to evaluate the closed-form
        coalescent transition probability.  Empty ranges
        (``start > end``) return 1.0.

        Args:
            start: Lower bound (inclusive).
            end: Upper bound (inclusive).

        Returns:
            Product of integers in the closed range.
        """
        result = 1.0
        for i in range(start, end + 1):
            result *= i
        return result

    @staticmethod
    def _gij_raw(length: float, i: int, j: int) -> float:
        """Closed-form coalescent transition probability g_{ij}(t).

        Computes the probability that ``i`` lineages entering a
        branch of length ``t = length`` coalesce down to ``j`` by the
        top of the branch, under the standard Kingman coalescent.

        Special cases:
          * ``length is None`` or ``length == -1`` are treated as an
            infinite branch: every ancestral lineage count eventually
            coalesces down to one, so return 1.0 iff ``j == 1``.
          * ``length == 0`` means no coalescence can happen; return
            1.0 iff ``i == j``.
          * ``i == 0`` is a no-op (no lineages to coalesce).

        Args:
            length: Branch length (in coalescent units).
            i: Lineages entering the branch.
            j: Lineages exiting the branch (``1 <= j <= i``).

        Returns:
            Transition probability in [0, 1].
        """
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
        """Memoized wrapper around :meth:`_gij_raw`.

        Real phylogenetic networks typically have only a handful of
        distinct (length, i, j) triples, so memoisation cuts the
        per-triplet work significantly.
        """
        key = (length, i, j)
        cached = self._gij_cache.get(key)
        if cached is not None:
            return cached
        val = self._gij_raw(length, i, j)
        self._gij_cache[key] = val
        return val

    def _is_valid_network(self, ignore_node: Node) -> bool:
        """Structural-validity probe used during articulation discovery.

        A network is considered valid here when every reachable node
        has in-degree + out-degree compatible with "either leaf, or
        internal non-degree-2 (unless it's the exempted node)", and
        no reachable node refers to a parent/child that isn't itself
        reachable.

        This is called after temporarily removing a tree-edge in
        :meth:`_compute_articulation_nodes` to see whether its
        removal disconnects the subtree below; a disconnection means
        the edge's endpoint is an articulation node.

        Args:
            ignore_node: Node exempted from the degree-2 filter.

        Returns:
            True iff the probed network remains structurally valid.
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
        """Populate ``self.articulation_nodes`` / ``self.lowest_articulation_nodes``.

        An articulation node is a tree node whose removal would
        disconnect one or more leaves from the root.  A "lowest"
        articulation node is an articulation whose children are not
        all themselves articulations -- these are exactly the nodes
        at which the DP can collapse compatible configurations
        (matching PhyloNet's semantics).

        Called once from ``__init__``; results are cached on the
        instance and read during :meth:`calculate_triple_probability`.
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

    def _compute_ac_minus(
        self,
        cacs: dict[int, list["_TripleDPEngine._Configuration"]],
        branch_length: float,
        inheritance_prob: float,
    ) -> dict[int, list["_TripleDPEngine._Configuration"]]:
        """Propagate DP configurations through one branch, bottom to top.

        Implements the ``AC+ -> AC-`` transition in PhyloNet
        notation: given the configurations at the child end of a
        branch (``cacs``), compute the configurations at the parent
        end after applying both the inheritance-probability factor
        (on reticulation edges) and the coalescent reduction factor
        (on any branch where multiple lineages entered).

        For each source state, :data:`_COALESCING_MAP` lists the
        ``(target_state, (lineages_in, lineages_out))`` transitions
        available on this branch.  We loop over them, multiply each
        configuration's probability by
        ``gamma**lineages_in * g_{in,out}(length)``, and aggregate
        identical retic-choice vectors by summing probabilities.

        Args:
            cacs: State/config map at the bottom of the branch.
            branch_length: Branch length (coalescent units).
            inheritance_prob: Reticulation gamma (1.0 on tree edges).

        Returns:
            State/config map at the top of the branch.
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

    def _split_at_network_node(
        self,
        cacs: dict[int, list["_TripleDPEngine._Configuration"]],
        net_node_id: int,
    ) -> tuple[
        dict[int, list["_TripleDPEngine._Configuration"]],
        dict[int, list["_TripleDPEngine._Configuration"]],
    ]:
        """Split configurations at a reticulation into its two parent edges.

        At a reticulation node with lineage state ``s``,
        :data:`_SPLITTING_MAP` lists each way ``s`` can be partitioned
        into ``(s1, s2)`` across the two parent edges.  For each
        (config, partition) pair we emit two new configurations,
        annotate each with the parent-edge choice (via
        :meth:`_Configuration.add_choice`), and multiply the
        probability by ``sqrt(total)`` -- this is the symmetric split
        of the joint mass, which is later completed by the
        inheritance-probability multiplier applied in
        :meth:`_compute_ac_minus`.

        Args:
            cacs: Incoming state/config map (at the reticulation).
            net_node_id: Index of the reticulation currently being
                processed; used to slot the parent-edge choice into
                the retic-choice vector.

        Returns:
            ``(ac_plus_1, ac_plus_2)``: configuration maps for the
            first and second parent-edge branches, respectively.
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
        """Compute ``P(xy|z)`` for one ordered triplet against this network.

        The triplet is interpreted as "x and y are siblings, z is the
        outgroup".  To get ``P(xz|y)`` or ``P(yz|x)`` call this method
        again with the taxa reordered.

        Algorithm (leaves to root pass):

          1. Initialise DP at each leaf whose label is in the triple
             (assigning lineage-state 1, 2, or 3 depending on position
             in ``triple``).
          2. Walk :attr:`_topo_leaf_to_root`.  For each internal tree
             node, merge child configurations using
             :data:`_MERGING_MAP` (and collapse at articulation
             nodes).  For each reticulation node, split
             configurations across both parent edges via
             :meth:`_split_at_network_node`.
          3. Propagate through each parent branch via
             :meth:`_compute_ac_minus`, which applies the
             inheritance-probability factor and the coalescent
             reduction.
          4. Terminate early at the root (or at any articulation that
             has fully collapsed the triplet): sum probabilities in
             the "fully coalesced" states (7 / 9 / 10) to get the
             final triplet probability.

        Args:
            triple: Ordered species labels ``(x, y, z)``; see above for
                how ordering maps to the returned resolution.

        Returns:
            ``P(xy|z)`` under the species network.
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

            # --- Seed DP at each relevant leaf ---------------------
            if is_leaf:
                lbl = node_label[node]
                if lbl in triple_list:
                    idx = triple_list.index(lbl)
                    # Lineage state = 1, 2, or 3 depending on the
                    # leaf's position in the input ``triple``.
                    cacs = {idx + 1: [_Cfg(nnn)]}

            # --- Reticulations: inherit one child's config ----------
            elif is_retic:
                # Reticulations always have a single child edge; the
                # AC- from that child becomes the incoming state.
                children = node_children[node]
                if children:
                    cacs = edge_to_ac_minus.get((node, children[0]))

            # --- Internal tree nodes: merge children -----------------
            else:
                children = node_children[node]
                if len(children) >= 2:
                    ac1 = edge_to_ac_minus.get((node, children[0]))
                    ac2 = edge_to_ac_minus.get((node, children[1]))

                    # Only one child carries any DP state: adopt its
                    # configurations directly, collapsing at lowest
                    # articulation nodes where retic-choice histories
                    # no longer need to be distinguished.
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
                    # Both children carry DP state: merge every
                    # compatible (cfg1, cfg2) pair via _MERGING_MAP.
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

            # --- Early termination at articulation w/ full coalescence ---
            # State 7 means all three lineages are present but not yet
            # fully coalesced (divide by 3 to select the target
            # resolution); states 9 and 10 represent progressively more
            # coalesced configurations that already pin the resolution.
            if 7 in cacs and node in art_nodes:
                total_prob = cacs[7][0].total_prob / 3.0
                if 9 in cacs:
                    total_prob += cacs[9][0].total_prob
                if 10 in cacs:
                    total_prob += cacs[10][0].total_prob
                break

            # --- Propagate through parent branch(es) ---------------
            if is_retic:
                # Reticulation: split state across the two parent
                # edges, then independently propagate each half.
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
                # Tree node: single parent, no inheritance factor.
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


# ======================================================================
# (E) Cython bridge
#
# The Cython implementation mirrors :meth:`_TripleDPEngine.calculate_triple_probability`
# but operates on flat numpy arrays instead of Python dicts, which is
# where most of the 10-20x speedup comes from.  When the ``.so`` isn't
# available the Python DP is used instead; both compute the same
# quantity to full floating-point accuracy.
# ======================================================================

# These caps bound the node arity assumed by the Cython extension.
# Phylogenetic networks in the wild almost always have binary internal
# nodes with at most two parents per reticulation, so these are
# generous upper bounds.
_CY_MAX_CHILDREN = 4
_CY_MAX_PARENTS = 2


def _extract_topology_for_cython(engine: _TripleDPEngine) -> dict:
    """Flatten a :class:`_TripleDPEngine`'s topology into numpy arrays.

    The Cython engine can't accept Python :class:`Node` objects
    directly, so this helper walks the engine's cached topology and
    produces a ``dict`` of numpy arrays keyed by flat node indices
    (0..n_nodes-1, matching the engine's leaves-to-root traversal
    order).  The resulting dict is shaped exactly the way
    ``mpl_engine_cy.score_all_triplets`` expects.

    Args:
        engine: Python DP engine already initialised on the target
            network (and thus holding all cached topology data).

    Returns:
        ``dict`` with numpy arrays for adjacency, branch lengths,
        gammas, articulation flags, and a ``label_to_idx`` map so
        callers can translate species labels into flat indices.
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


def _build_cython_triplet_index(
    topo: dict,
    triplets: list[tuple[str, str, str]],
    rho: dict[tuple[str, str, str], tuple[float, float, float]],
) -> tuple[list[tuple[int, int, int]], list[tuple[float, float, float]]]:
    """Translate species triplets into flat node indices for the Cython DP.

    Triplets whose taxa are not all present in the network are silently
    dropped (mirrors the Python path).  Depends only on the *topology*
    (the ``label_to_idx`` map), so it is stable across branch-length /
    gamma edits and can be cached alongside the extracted topology.

    Args:
        topo: The dict returned by :func:`_extract_topology_for_cython`.
        triplets: Canonical species triplets to evaluate.
        rho: Gene-tree triplet frequency table (constant for the run).

    Returns:
        ``(trip_idx, rho_vals)`` lists aligned by position.
    """
    lbl_idx = topo["label_to_idx"]
    trip_idx: list[tuple[int, int, int]] = []
    rho_vals: list[tuple[float, float, float]] = []
    for t in triplets:
        x, y, z = t
        if x in lbl_idx and y in lbl_idx and z in lbl_idx:
            trip_idx.append((lbl_idx[x], lbl_idx[y], lbl_idx[z]))
            rho_vals.append(rho[t])
    return trip_idx, rho_vals


def _refresh_cython_params(engine: _TripleDPEngine, topo: dict) -> None:
    """Re-read branch lengths / gammas into a cached Cython topology dict.

    Lever 3 fast path: when only continuous parameters changed (the
    topology is unchanged), the structural arrays in ``topo`` are still
    valid -- only ``pa_bl`` / ``pa_gamma`` need updating from the live
    network edges.  This mirrors exactly the parameter logic in
    :func:`_extract_topology_for_cython` so a refreshed score is bit-for-bit
    identical to a full rebuild.  The engine's ``_edge_info`` cache is kept
    in sync for the pure-Python scoring path.

    Args:
        engine: The persistent engine bound to the (unchanged) topology.
        topo: The cached dict from :func:`_extract_topology_for_cython`,
            whose ``pa_bl`` / ``pa_gamma`` arrays are updated in place.
    """
    net = engine.network
    nodes = engine._topo_leaf_to_root
    pa_bl = topo["pa_bl"]
    pa_gamma = topo["pa_gamma"]
    for i, node in enumerate(nodes):
        parents = engine._node_parents[node]
        is_ret = engine._node_is_retic[node]
        n_par = len(parents)
        for j, parent in enumerate(parents[:_CY_MAX_PARENTS]):
            edge = net.get_edge(parent, node)
            bl = edge.get_length()
            gamma = edge.get_gamma()
            engine._edge_info[(parent, node)] = (bl, gamma)
            pa_bl[i, j] = bl if bl is not None else -1.0
            if is_ret:
                if gamma is not None and gamma > 0:
                    pa_gamma[i, j] = gamma
                else:
                    pa_gamma[i, j] = 1.0 / n_par
            else:
                pa_gamma[i, j] = 1.0


def _cy_score_from_topo(
    topo: dict,
    trip_idx: list[tuple[int, int, int]],
    rho_vals: list[tuple[float, float, float]],
) -> float:
    """Run the Cython DP from a (possibly cached) extracted topology."""
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


def _score_with_cython(
    engine: _TripleDPEngine,
    triplets: list[tuple[str, str, str]],
    rho: dict[tuple[str, str, str], tuple[float, float, float]],
) -> float:
    """Score a network against all triplets via the Cython DP engine.

    Triplets whose taxa are not all present in the engine's network
    are silently dropped -- this mirrors the Python path's behaviour
    and keeps the pruned-taxa case working.

    Args:
        engine: Python DP engine already bound to the target network.
        triplets: Canonical species triplets to evaluate.
        rho: Gene-tree triplet frequency table (constant for the run).

    Returns:
        Log pseudo-likelihood across all evaluable triplets.
    """
    topo = _extract_topology_for_cython(engine)
    trip_idx, rho_vals = _build_cython_triplet_index(topo, triplets, rho)
    return _cy_score_from_topo(topo, trip_idx, rho_vals)


# ======================================================================
# (F) Public triplet API
#
# These two functions are the entry points for users who want to
# bring-their-own-search-driver: enumerate triplets, precompute rho,
# then call score_species_network_triplets against any number of
# candidate species networks.  :class:`MPL` wraps both for the
# common case.
# ======================================================================

def compute_gene_tree_triplets(
    gene_trees: GeneTrees,
    mapping: dict[str, list[str]],
    species_labels: Optional[list[str]] = None,
) -> GeneTreeTripletResult:
    """Enumerate triplets and precompute rho from ``gene_trees``.

    Gene-tree statistics are constant across a network search, so
    this is called exactly once up front.

    Args:
        gene_trees: Collection of gene trees.
        mapping: Species -> list of allele labels.
        species_labels: Optional explicit species list; defaults to
            ``sorted(mapping.keys())``.  Provide this when scoring
            against a reduced taxon set (e.g. sub-sample experiments).

    Returns:
        :class:`GeneTreeTripletResult` carrying triplets and rho.

    Raises:
        ValueError: If fewer than three species labels are available
            (triplets require at least three taxa).
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
    """Score a species network against precomputed triplet frequencies.

    Builds a fresh :class:`_TripleDPEngine` bound to ``species_net``
    and walks every triplet, computing both predicted probabilities
    and the MPL log-sum.

    Args:
        species_net: Species network to evaluate.
        gene_triplet_result: Output of
            :func:`compute_gene_tree_triplets` (same gene-tree set).

    Returns:
        :class:`SpeciesNetworkTripletResult` with per-triplet
        probabilities and the final log pseudo-likelihood.

    Raises:
        ValueError: If a network triplet isn't present in
            ``gene_triplet_result`` (which indicates the rho table was
            built over a different species set than the network's
            leaves).
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


# ======================================================================
# (G) Scorer + adaptive proposal kernel
#
# :class:`MPLScorer` is the objective function (a thin callable that
# rebuilds the DP engine per call).  :class:`MPLKernel` is the
# proposal distribution driving the search (phased move selection +
# adaptive weights + adaptive sigmas for continuous proposals).
# :class:`_AdaptiveConfig` is a private dataclass-style container for
# the empirically-chosen tuning scalars that the kernel reads.
# ======================================================================


class MPLScorer:
    """Callable likelihood evaluator for use with :class:`Model`.

    Registered via ``model.set_likelihood_calculator(scorer)``.  On
    each call, rebuilds a fresh :class:`_TripleDPEngine` for the
    current network (topology moves invalidate the previous engine's
    cached articulation sets) and returns the log pseudo-likelihood.

    Gene-tree rho values never change during a search, so they are
    stored once on the instance.  Triplets whose gene-tree rho is
    identically zero contribute nothing to the score and are pruned
    at construction time to avoid redundant DP calls.

    Attributes:
        _rho: Full triplet -> rho table (kept even for pruned
            triplets in case the scorer is reused against a network
            with a different triplet mask).
        _triplets: Active (non-zero-rho) triplet list used by
            ``__call__``.
    """

    def __init__(
        self,
        rho: dict[tuple[str, str, str], tuple[float, float, float]],
        triplets: list[tuple[str, str, str]],
    ) -> None:
        """Initialise the scorer from precomputed gene-tree statistics.

        Args:
            rho: Mapping ``triplet -> (rho_XY|Z, rho_XZ|Y, rho_YZ|X)``.
            triplets: Canonical species triplets to iterate over.
        """
        self._rho = rho
        self._triplets = [
            t for t in triplets if any(rho[t][i] > 0.0 for i in range(3))
        ]

        # ---- Lever 3: incremental engine cache --------------------------
        # When consecutive scoring calls touch the *same* network object
        # and only its branch lengths / gammas changed (the dirty-set
        # protocol reports a non-``None`` touched-node set), we reuse the
        # cached engine + extracted Cython topology and refresh only the
        # parameter arrays, skipping the (dominant) topology rebuild.
        self._engine: _TripleDPEngine | None = None
        self._engine_net = None              # network object identity guard
        self._cy_topo: dict | None = None
        self._cy_trip_idx = None
        self._cy_rho_vals = None

    def _needs_rebuild(self, model: Model) -> bool:
        """Decide whether the cached engine must be rebuilt for ``model``.

        Rebuild when there is no cache yet, the network object changed
        (a different proposal / committed copy), or the move reported a
        full ("topology changed") invalidation via ``_dirty_nodes is
        None``.  A non-``None`` dirty set means parameters-only -> reuse.
        """
        if self._engine is None or self._engine_net is not model.network:
            return True
        dirty = getattr(model, "_dirty_nodes", None)
        return dirty is None

    def __call__(self, model: Model) -> float:
        """Return log pseudo-likelihood of ``model.network``.

        Uses the Cython DP path when available, otherwise falls back
        to the Python DP.  Both compute the same score.  Reuses a cached
        engine across calls on the same network when only continuous
        parameters changed (lever 3).

        Args:
            model: :class:`Model` whose ``network`` attribute is the
                current species network.

        Returns:
            Log pseudo-likelihood (higher is better; always <= 0).
        """
        net = model.network
        if self._needs_rebuild(model):
            self._engine = _TripleDPEngine(net)
            self._engine_net = net
            if _HAS_CYTHON_MPL:
                self._cy_topo = _extract_topology_for_cython(self._engine)
                self._cy_trip_idx, self._cy_rho_vals = (
                    _build_cython_triplet_index(
                        self._cy_topo, self._triplets, self._rho,
                    )
                )
        else:
            # Parameters-only change: refresh edge values on the cached
            # engine / topology rather than rebuilding the topology.
            if _HAS_CYTHON_MPL:
                _refresh_cython_params(self._engine, self._cy_topo)
            else:
                self._engine.refresh_params()

        # The scorer has consumed the dirty set; reset it so a subsequent
        # parameters-only call is recognised as such.
        clear = getattr(model, "clear_dirty_nodes", None)
        if callable(clear):
            clear()

        if _HAS_CYTHON_MPL:
            return _cy_score_from_topo(
                self._cy_topo, self._cy_trip_idx, self._cy_rho_vals,
            )

        engine = self._engine
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


class _AdaptiveConfig:
    """Internal adaptive-tuning constants for :class:`MPLKernel`.

    Grouped here so the kernel's class-level surface stays focused on
    its structural pieces (phase order, base-weight maps) rather than
    buried under fifteen tuning scalars.  None of these values are
    user-facing knobs; they're empirically-chosen defaults for the
    adaptive machinery.  Moving them here is a pure refactor -- the
    numbers are identical to the previous ``MPLKernel`` class
    constants.  When you want to tune the adaptive layer, this is the
    single place to look.
    """

    # ── Robbins-Monro continuous-sigma tuning ──────────────────────
    # Applied to ChangeNodeHeight and ChangeInheritanceProb.  The
    # scheme nudges ``log(sigma)`` toward the target acceptance rate;
    # step size shrinks as ``1 / (n + shift)**exp`` so later updates
    # make smaller corrections.
    SIGMA_TARGET_ACCEPT = 0.35           # Roberts-Rosenthal heuristic
    SIGMA_TUNE_DELAY = 20                # warm-up observations per move
    SIGMA_TUNE_EXPONENT = 0.6            # step decay rate (1/n^exp)
    SIGMA_TUNE_DENOM_SHIFT = 50          # dampen very early updates

    # ── ChangeNodeHeight sigma_frac bounds ─────────────────────────
    # sigma_frac is a *fraction of the feasible half-range*, so these
    # bounds are dimensionless.  0.4 gives a tighter proposal than the
    # prior uniform draw (which was effectively ~0.58).
    NH_SIGMA_INIT = 0.4
    NH_SIGMA_MIN = 0.02
    NH_SIGMA_MAX = 1.5

    # ── ChangeInheritanceProb sigma bounds ─────────────────────────
    # sigma acts directly on the gamma scale (inheritance probability).
    # Bounds keep proposals from degenerating to delta spikes or flat
    # uniforms.
    CIP_SIGMA_INIT = 0.1
    CIP_SIGMA_MIN = 0.005
    CIP_SIGMA_MAX = 0.4

    # ── SPR adaptive regraft radius ────────────────────────────────
    # ``SPR`` weights each candidate regraft edge by
    # ``1 / d**distance_decay``.  MAX keeps the move strongly local
    # (exploit current basin); MIN makes it near-flat (broad hops
    # across the network).  Decay interpolates from MAX -> MIN as
    # ``stagnation / phase_patience`` climbs toward 1.
    SPR_DECAY_MAX = 2.5
    SPR_DECAY_MIN = 0.5

    # ── Post-reheat wide-SPR window ────────────────────────────────
    # After each reheat notification, the next N SPR proposals are
    # forced to ``SPR_DECAY_MIN`` regardless of phase switches or
    # stagnation resets.  Without this, the phase switch triggered
    # by the reheat hook zeroes ``_stagnation`` before a single
    # broad SPR fires, cancelling the reheat's exploration intent
    # entirely.  Sized to roughly cover one post-reheat SA cooling
    # window (~250 iters at ~8% SPR density).
    SPR_BROAD_MODE_PROPOSALS = 20

    # ── efficiency-aware weight scaler blend ───────────────────────
    # Composite score = ACC_WEIGHT * acc_rate + EFF_WEIGHT * efficiency
    # where efficiency = imp_rate / max(acc_rate, EFF_EPSILON).  The
    # 0.4/0.6 split weights exploitation slightly over exploration --
    # SA itself handles exploration via temperature-driven uphill
    # accepts, so the adaptive layer focuses on "which move is
    # actually moving the needle".
    SCALER_ACC_WEIGHT = 0.4
    SCALER_EFF_WEIGHT = 0.6
    SCALER_EFF_EPSILON = 0.01


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

    Both phases keep a small bleed of the complementary phase's moves
    active (~10-15% weight) so the chain stays ergodic across phase
    flips and the adaptive scaler always has fresh observations for
    every class.

    Phase transitions fire after ``phase_patience`` consecutive
    proposals without accepted improvement.  Within each phase,
    selection weights adapt from phase-specific base distributions
    using a sliding window of recent acceptance rates.

    When explicit *weights* are supplied, the kernel falls back to
    flat (non-phased) mode for backward compatibility.
    """

    TOPOLOGY = "topology"
    RETICULATION = "reticulation"

    # Phase base weights.  Each phase keeps a low-rate bleed of the
    # complementary phase's moves so the chain is never strictly
    # topology-only or strictly reticulation-only.
    _TOPOLOGY_BASE: dict[type, float] = {
        SPR: 5.0,
        ChangeNodeHeight: 3.0,
        ChangeInheritanceProb: 1.0,
        # Bleed: ~10% aggregate weight on retic moves keeps Add/Remove
        # warm and lets small retic corrections happen mid-topology-phase.
        AddReticulation: 0.30,
        RemoveReticulation: 0.20,
        RelocateReticulation: 0.25,
        ChangeReticSource: 0.10,
        ChangeReticDest: 0.10,
        # FlipReticulation is deliberately weighted tiny.  Empirically
        # (see P4 50k run) it has a ~0.5% strict-improvement rate --
        # ~25x lower than other retic moves -- while its ~27% accept
        # rate (mostly uphill SA accepts) trips the adaptive scaler
        # into over-proposing it.  Keep it alive (not zero) so the
        # reticulation-orientation corner case is still reachable, but
        # don't let it consume a large slice of the proposal budget.
        FlipReticulation: 0.03,
    }

    _RETICULATION_BASE: dict[type, float] = {
        AddReticulation: 3.0,
        RemoveReticulation: 2.0,
        # FlipReticulation base weight: see TOPOLOGY_BASE comment.
        # Previously 1.5 (equal to source/dest moves); dropped to 0.5
        # based on observed improvement rate ~25x lower than peers.
        FlipReticulation: 0.5,
        ChangeReticSource: 1.5,
        ChangeReticDest: 1.5,
        RelocateReticulation: 2.5,
        ChangeInheritanceProb: 1.5,
        ChangeNodeHeight: 1.0,
        # Bleed: keep SPR alive in retic phase so local topology
        # adjustments can accompany reticulation placement.
        SPR: 1.5,
    }

    _PHASE_ORDER = [TOPOLOGY, RETICULATION]

    # All empirically-chosen adaptive-tuning scalars live on
    # ``_AdaptiveConfig`` (defined below this class) so the kernel's
    # visible surface stays focused on structural design decisions
    # (phases, base weights) rather than buried under a dozen magic
    # numbers.  Instance attributes below pull initial values from
    # that config at construction time.

    def __init__(self,
                 move_types: list[type[Move]] | None = None,
                 weights: list[float] | None = None,
                 max_reticulations: int | None = None,
                 max_level: int | None = None,
                 adaptive: bool = True,
                 window_size: int = 30,
                 min_weight: float = 0.05,
                 phase_patience: int = 150,
                 warmup: int = 8,
                 stagnation_reset_delta: float = 1.0,
                 rng: np.random.Generator | None = None) -> None:
        """
        Args:
            move_types: Move classes available to the kernel.  When
                ``None`` the full default set is used.
            weights: Fixed per-move selection weights.  Supplying this
                disables phased cycling and adaptive tuning.
            max_reticulations: Cap on reticulation nodes.  When the
                current network is at the cap, ``AddReticulation`` is
                dropped from the active move set so we don't waste
                proposals on guaranteed no-ops.
            max_level: Cap on network level (``max_lvl`` search flag).
                Passed to every level-raising move (Add / Relocate /
                ChangeReticSource / ChangeReticDest / Flip) so they
                self-reject proposals that would exceed the cap.
                ``None`` disables the per-move check (the accept-path
                level guard remains authoritative).
            adaptive: Enable within-phase adaptive weight scaling.
            window_size: Sliding-window length for acceptance stats.
            min_weight: Minimum scale factor (fraction of base weight)
                to prevent any move from being fully starved.
            phase_patience: Consecutive non-improving proposals before
                the kernel switches to the next phase.  Default 150
                keeps phase flips roughly aligned with the SA
                ``steps_per_temp`` cadence.
            warmup: Minimum observations per move class before the
                adaptive scaling activates for that class.
            stagnation_reset_delta: Minimum strict-improvement
                magnitude (in log-PL units) that counts as "real
                progress" for stagnation-reset purposes.  Any
                ``delta > stagnation_reset_delta`` resets the
                stagnation counter; smaller wiggles are ignored.
                Default 1.0 filters out the O(0.01-0.1) tweaks that
                late-stage NH/CIP moves generate, so the counter
                actually climbs during genuine plateaus.  Setting to
                0.0 restores the prior "any strict improvement resets"
                behaviour.  Required for phase cycling and adaptive
                SPR radius to track true plateaus rather than
                micro-noise.
            rng: ``numpy.random.Generator`` used to sample the move
                class.  When ``None``, a fresh generator is created
                from OS entropy; callers that care about
                reproducibility (e.g. ``MPL.search``) should pass one
                seeded from the search-wide ``SeedSequence``.
        """
        super().__init__()
        self._max_retics: int | None = max_reticulations
        self._max_level: int | None = max_level
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
        self._stagnation_reset_delta: float = float(stagnation_reset_delta)

        # Diagnostics for phase/SPR plumbing -- filled in as the run
        # progresses; surfaced via ``format_stats`` at the end.
        self._reheat_signals: int = 0
        self._stagnation_peak: int = 0

        self.rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

        from collections import deque
        self._history: dict[type[Move], deque] = {
            cls: deque(maxlen=window_size) for cls in self._all_moves
        }
        self._last_cls: type[Move] | None = None

        # Lifetime counters for end-of-run diagnostics.  ``proposed`` counts
        # every draw of a move class (regardless of validity).  ``accepted``
        # counts every draw that the SA/HC driver later committed to the
        # chain (strict improvement OR uphill-accepted).  ``improved``
        # counts strict-improvement commits (delta > 0 in maximisation).
        self._proposed: dict[type[Move], int] = {cls: 0 for cls in self._all_moves}
        self._accepted: dict[type[Move], int] = {cls: 0 for cls in self._all_moves}
        self._improved: dict[type[Move], int] = {cls: 0 for cls in self._all_moves}

        # Adaptive sigmas for the two continuous proposals.  They are
        # driven toward ``_SIGMA_TARGET_ACCEPT`` by a Robbins-Monro-style
        # update in ``report_outcome``.  ``_sigma_obs`` counts the
        # observations each move class has contributed so the step size
        # decays like 1/n^exponent.
        self._nh_sigma: float = _AdaptiveConfig.NH_SIGMA_INIT
        self._cip_sigma: float = _AdaptiveConfig.CIP_SIGMA_INIT
        self._sigma_obs: dict[type[Move], int] = {
            ChangeNodeHeight: 0,
            ChangeInheritanceProb: 0,
        }

        # SPR decay tracking.  ``_last`` holds the decay used on the most
        # recent SPR draw; ``_min_seen``/``_max_seen`` record the
        # realised range so ``format_stats`` can show how often the
        # kernel actually opened the SPR radius during the run.
        self._spr_decay_last: float = _AdaptiveConfig.SPR_DECAY_MAX
        self._spr_decay_min_seen: float = _AdaptiveConfig.SPR_DECAY_MAX
        self._spr_decay_max_seen: float = _AdaptiveConfig.SPR_DECAY_MAX

        # Post-reheat wide-SPR window.  ``on_reheat`` sets this to
        # ``SPR_BROAD_MODE_PROPOSALS``; each SPR draw decrements it.
        # While positive, ``_current_spr_decay`` returns ``SPR_DECAY_MIN``
        # regardless of the stagnation-derived decay.  This decouples
        # SPR radius adaptation from phase-switch stagnation resets:
        # even if the reheat triggers a phase switch (which zeroes
        # ``_stagnation``), the broad-mode counter still forces broad
        # regrafts for the next N SPR calls.  See ``_AdaptiveConfig``.
        self._spr_broad_remaining: int = 0
        self._spr_broad_activations: int = 0
        self._spr_broad_proposals: int = 0

    # ── phase helpers ───────────────────────────────────────────

    def _phase_base_map(self) -> dict[type, float]:
        """Base weight map for the current phase."""
        if self._phase == self.TOPOLOGY:
            return self._TOPOLOGY_BASE
        return self._RETICULATION_BASE

    def _at_retic_cap(self, network: Network | None) -> bool:
        """True when the current network already holds ``max_reticulations``."""
        if self._max_retics is None or network is None:
            return False
        return count_reticulations(network) >= self._max_retics

    def _active_moves_and_base(
        self, network: Network | None = None,
    ) -> tuple[list[type[Move]], list[float]]:
        """Active move classes and base weights for the current phase.

        When the caller supplies a ``network`` and we're at the
        reticulation cap, ``AddReticulation`` is omitted entirely so
        proposals aren't wasted on a guaranteed no-op.
        """
        base = self._phase_base_map()
        at_cap = self._at_retic_cap(network)
        moves: list[type[Move]] = []
        weights: list[float] = []
        for cls in self._all_moves:
            if cls not in base:
                continue
            if at_cap and cls is AddReticulation:
                continue
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

    # ── adaptive SPR regraft radius ─────────────────────────────

    def _current_spr_decay(self) -> float:
        """SPR distance-decay for the current stagnation level.

        Returns a value in ``[SPR_DECAY_MIN, SPR_DECAY_MAX]``.  Normal
        path: decay drops linearly from MAX (just improved / fresh)
        toward MIN (fully stuck) as ``stagnation / phase_patience``
        climbs.  Post-reheat path: while ``_spr_broad_remaining > 0``
        (set by :meth:`on_reheat`), decay is forced to MIN so the
        next ``SPR_BROAD_MODE_PROPOSALS`` SPR draws get broad hops
        regardless of intervening phase switches.  This is the fix
        for the "reheat triggers phase switch, phase switch zeroes
        stagnation, next SPR sees stuckness=0, wide mode never fires"
        pathology seen in long runs.
        """
        if self._spr_broad_remaining > 0:
            return _AdaptiveConfig.SPR_DECAY_MIN
        if self._phase_patience <= 0:
            return _AdaptiveConfig.SPR_DECAY_MAX
        stuckness = min(1.0, self._stagnation / float(self._phase_patience))
        span = _AdaptiveConfig.SPR_DECAY_MAX - _AdaptiveConfig.SPR_DECAY_MIN
        return _AdaptiveConfig.SPR_DECAY_MAX - stuckness * span

    # ── adaptive sigma (continuous proposals) ───────────────────

    def _tune_sigma(self,
                    current: float,
                    n_obs: int,
                    accepted: bool,
                    sigma_min: float,
                    sigma_max: float) -> float:
        """One Robbins-Monro step toward the acceptance-rate target.

        ``log(sigma)`` drifts by ``step * (accept_indicator - target)``,
        where ``step`` shrinks as ``1/(n + shift)^exponent``.  This is
        the standard adaptive Metropolis update (Roberts & Rosenthal,
        2009): accept > target widens proposals, accept < target
        tightens them.  Because SA also cools over time, ``sigma``
        tends to contract alongside temperature -- which is exactly
        what we want in the exploitation phase.
        """
        if n_obs < _AdaptiveConfig.SIGMA_TUNE_DELAY:
            return current
        step = 1.0 / ((n_obs + _AdaptiveConfig.SIGMA_TUNE_DENOM_SHIFT)
                      ** _AdaptiveConfig.SIGMA_TUNE_EXPONENT)
        indicator = 1.0 if accepted else 0.0
        log_adjust = step * (indicator - _AdaptiveConfig.SIGMA_TARGET_ACCEPT)
        new_sigma = current * math.exp(log_adjust)
        return max(sigma_min, min(sigma_max, new_sigma))

    # ── adaptive tuning ─────────────────────────────────────────

    def _adapt_weights(self,
                       moves: list[type[Move]],
                       base_weights: list[float]) -> list[float]:
        """Scale base weights by an efficiency-aware accept/improve blend.

        A pure acceptance-rate scaler over-rewards moves that get
        accepted often via uphill SA accepts but rarely make strict
        progress (e.g. ``RelocateReticulation`` or ``FlipReticulation``
        at cold T).  Meanwhile it starves moves that accept rarely but
        whose accepts are almost always strict improvements (e.g.
        ``SPR`` at cold T, ``AddReticulation`` when below cap).  That
        is the opposite of what we want for an MPL *optimiser*.

        The formula below reads two signals from each move's rolling
        ``window_size`` history:

            acc_rate        = accepts / window                # explore credit
            imp_rate        = strict_improvements / window    # exploit credit
            efficiency      = imp_rate / max(acc_rate, 0.01)  # per-accept quality

        ``efficiency`` is the probability that, given the move
        accepted, the accept was a strict improvement.  It's capped at
        1.0 to keep the math bounded when a warmup-era window happens
        to have only improvements.

        The composite score

            composite = 0.4 * acc_rate + 0.6 * efficiency

        then drives the same ``floor .. 2x`` multiplicative scale the
        old formula used -- so the knob count is unchanged; only the
        signal feeding it is different.  The 0.4 / 0.6 split
        deliberately weights exploitation (progress) more heavily than
        exploration (acceptance): SA itself already handles exploration
        through its temperature-driven uphill accepts.  We want the
        adaptive layer focused on "which move is actually moving the
        needle right now".

        Moves with fewer than ``warmup`` observations keep their full
        base weight (avoids early-window noise swinging the scaler).
        """
        floor = self._min_weight
        weights: list[float] = []
        for cls, base_w in zip(moves, base_weights):
            hist = self._history[cls]
            if len(hist) < self._warmup:
                weights.append(base_w)
                continue
            n = len(hist)
            n_acc = sum(1 for accepted, _ in hist if accepted)
            n_imp = sum(1 for accepted, delta in hist
                        if accepted and delta > 0.0)
            acc_rate = n_acc / n
            imp_rate = n_imp / n
            efficiency = min(
                1.0,
                imp_rate / max(acc_rate, _AdaptiveConfig.SCALER_EFF_EPSILON),
            )
            composite = (
                _AdaptiveConfig.SCALER_ACC_WEIGHT * acc_rate
                + _AdaptiveConfig.SCALER_EFF_WEIGHT * efficiency
            )
            scale = floor + (2.0 - floor) * composite
            weights.append(base_w * scale)
        return weights

    # ── public interface ────────────────────────────────────────

    def generate(self, model: "Model | None" = None) -> Move:
        """Sample a move from the current phase's weighted distribution.

        Args:
            model: Optional current model.  Used only for cap-aware
                filtering (``AddReticulation`` is dropped when the
                network is already at ``max_reticulations``).  Callers
                from ``SimulatedAnnealing``/``HillClimbing`` pass the
                live model; the abstract base signature keeps the
                parameter optional for back-compat.
        """
        network = model.network if model is not None else None

        # --- Decide which moves are eligible and their weights -----
        if self._phased:
            self._maybe_switch_phase()
            moves, base_weights = self._active_moves_and_base(network)
            weights = (self._adapt_weights(moves, base_weights)
                       if self._adaptive else base_weights)
        else:
            moves = self._all_moves
            weights = self._fixed_weights

        if not moves:
            # Defensive fallback: with cap filtering active and a weird
            # phase base, we might have no eligible moves.  Fall back to
            # the full move list so the search keeps progressing.
            moves = list(self._all_moves)
            weights = [1.0] * len(moves)

        # --- Sample a move class from the effective distribution ---
        w = np.asarray(weights, dtype=float)
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0.0:
            # Degenerate weights (all zero or NaN): fall back to
            # uniform sampling so the chain doesn't stall.
            idx = int(self.rng.integers(0, len(moves)))
        else:
            idx = int(self.rng.choice(len(moves), p=w / total))
        cls = moves[idx]
        self._last_cls = cls
        self._proposed[cls] = self._proposed.get(cls, 0) + 1

        # --- Instantiate the move with its current tuning knobs ----
        if cls is AddReticulation:
            return AddReticulation(
                max_reticulations=self._max_retics,
                max_level=self._max_level,
            )
        if cls in (FlipReticulation, ChangeReticSource,
                   ChangeReticDest, RelocateReticulation):
            # Level-raising relocation / endpoint moves: pass the level
            # cap so they can self-reject early.
            return cls(max_level=self._max_level)
        if cls is ChangeNodeHeight:
            return ChangeNodeHeight(sigma_frac=self._nh_sigma)
        if cls is ChangeInheritanceProb:
            return ChangeInheritanceProb(sigma=self._cip_sigma)
        if cls is SPR:
            decay = self._current_spr_decay()
            # Decrement the broad-mode window.  A single SPR draw
            # consumes one slot of post-reheat broad-radius credit
            # regardless of whether the move ultimately commits.
            # Tracking consumed slots separately from activations
            # lets format_stats distinguish "broad mode fired N
            # times" vs "M reheats activated it".
            if self._spr_broad_remaining > 0:
                self._spr_broad_remaining -= 1
                self._spr_broad_proposals += 1
            # Keep a trailing record so end-of-run stats can show the
            # realised decay range without sampling every tick.
            self._spr_decay_last = decay
            self._spr_decay_min_seen = min(self._spr_decay_min_seen, decay)
            self._spr_decay_max_seen = max(self._spr_decay_max_seen, decay)
            return SPR(distance_decay=decay)
        return cls()

    def report_outcome(self, accepted: bool, delta: float = 0.0) -> None:
        """Record a move outcome and update adaptive state.

        Called by the search driver (HC / SA) immediately after each
        ``generate()`` -> propose -> score pair, regardless of
        accept/reject.  Updates:

          * The rolling per-move history window (fed back to
            :meth:`_adapt_weights`).
          * Lifetime accepted/improved counters (for
            :meth:`format_stats`).
          * The Robbins-Monro sigma for the last move class when it
            was a continuous proposal (NH / CIP).
          * The magnitude-aware stagnation counter driving phase
            switching and adaptive SPR radius.

        Args:
            accepted: True iff the driver committed this move.
            delta: Score change ``new - old``; positive means strict
                improvement on the search objective (maximisation).
        """
        cls = self._last_cls
        if cls is not None and cls in self._history:
            self._history[cls].append((accepted, delta))
        if cls is not None:
            if accepted:
                self._accepted[cls] = self._accepted.get(cls, 0) + 1
            if accepted and delta > 0:
                self._improved[cls] = self._improved.get(cls, 0) + 1

            if cls is ChangeNodeHeight:
                self._sigma_obs[cls] += 1
                self._nh_sigma = self._tune_sigma(
                    self._nh_sigma,
                    self._sigma_obs[cls],
                    accepted,
                    _AdaptiveConfig.NH_SIGMA_MIN,
                    _AdaptiveConfig.NH_SIGMA_MAX,
                )
            elif cls is ChangeInheritanceProb:
                self._sigma_obs[cls] += 1
                self._cip_sigma = self._tune_sigma(
                    self._cip_sigma,
                    self._sigma_obs[cls],
                    accepted,
                    _AdaptiveConfig.CIP_SIGMA_MIN,
                    _AdaptiveConfig.CIP_SIGMA_MAX,
                )

        # Magnitude-aware reset: only genuine improvements reset
        # stagnation.  Tiny +0.01 tweaks from late-stage NH/CIP moves
        # don't count, so the counter actually climbs when the chain
        # is truly plateauing (which is what phase switching and
        # adaptive SPR radius need to see).
        if accepted and delta > self._stagnation_reset_delta:
            self._stagnation = 0
        else:
            self._stagnation += 1
            if self._stagnation > self._stagnation_peak:
                self._stagnation_peak = self._stagnation

    def on_reheat(self) -> None:
        """Notify the kernel that the driver just fired a reheat.

        Reheats are the SA layer's explicit "we're stuck" signal --
        they're computed from rolling improvement over a window, which
        is a *stronger* plateau signal than anything the kernel's own
        stagnation counter can see.  On notification we:

        1. Force ``_stagnation`` up to ``phase_patience`` so the next
           ``generate()`` call triggers a phase switch (if we aren't
           in the opposite phase already).  This flips to retic-focused
           moves to try a structurally different escape route.
        2. Arm the post-reheat broad-SPR window: the next
           ``SPR_BROAD_MODE_PROPOSALS`` SPR draws are forced to
           ``SPR_DECAY_MIN`` (near-flat, large hops) regardless of
           intervening phase switches or stagnation resets.

        The broad-SPR window is the critical half: without it, the
        phase switch from (1) resets ``_stagnation`` to 0 before a
        single broad SPR fires, reverting the move to local-only mode
        and cancelling the reheat's entire exploration intent.  With
        it, reheats actually reach the basin-escape topology moves
        they were fired to attempt.

        No-ops for the non-phased kernel (custom ``weights`` path).
        """
        self._reheat_signals += 1
        if not self._phased:
            return
        self._stagnation = max(self._stagnation, self._phase_patience)
        if self._stagnation > self._stagnation_peak:
            self._stagnation_peak = self._stagnation
        # Arm the broad-SPR window independently of the stagnation
        # counter.  If a prior reheat's window hasn't fully drained,
        # refresh to the full budget rather than accumulating -- the
        # cascade guard already limits how often this can happen.
        self._spr_broad_remaining = _AdaptiveConfig.SPR_BROAD_MODE_PROPOSALS
        self._spr_broad_activations += 1

    def get_weights(self) -> dict[str, float]:
        """Return the current effective selection probabilities.

        Honours the active phase (when phased) and the adaptive
        scaler (when enabled); falls back to the fixed weight list
        otherwise.  Useful for quick introspection or logging.

        Returns:
            ``dict`` mapping move class name to probability in [0, 1]
            (summing to 1).
        """
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

    def get_stats(self) -> dict[str, dict[str, int | float]]:
        """Lifetime per-move statistics across the search.

        Returns:
            Mapping ``move_class_name -> {proposed, accepted, improved,
            acceptance_rate, improvement_rate}``.  ``acceptance_rate``
            is accepted / proposed (fraction of draws that were
            committed to the chain, including uphill-accepted moves).
            ``improvement_rate`` is improved / proposed (strict
            log-PL gain).  Both rates default to 0 when a move was
            never sampled.
        """
        stats: dict[str, dict[str, int | float]] = {}
        for cls in self._all_moves:
            n_prop = self._proposed.get(cls, 0)
            n_acc = self._accepted.get(cls, 0)
            n_imp = self._improved.get(cls, 0)
            stats[cls.__name__] = {
                "proposed": n_prop,
                "accepted": n_acc,
                "improved": n_imp,
                "acceptance_rate": (n_acc / n_prop) if n_prop else 0.0,
                "improvement_rate": (n_imp / n_prop) if n_prop else 0.0,
            }
        return stats

    def format_stats(self) -> str:
        """Return a human-readable summary of lifetime kernel statistics.

        Intended for printing at the end of a search.  Includes
        per-move proposal / accept / improve counts, phase-switch
        summary, adaptive sigma end-state, SPR radius range, and the
        post-reheat broad-SPR window activation count.
        """
        stats = self.get_stats()
        name_width = max((len(k) for k in stats), default=8)
        lines = [
            f"{'move':<{name_width}}  {'proposed':>9}  {'accepted':>9}  "
            f"{'improved':>9}  {'accept%':>8}  {'improve%':>9}"
        ]
        total_prop = sum(s["proposed"] for s in stats.values())
        for name, s in sorted(
            stats.items(), key=lambda kv: -kv[1]["proposed"],
        ):
            lines.append(
                f"{name:<{name_width}}  "
                f"{s['proposed']:>9d}  "
                f"{s['accepted']:>9d}  "
                f"{s['improved']:>9d}  "
                f"{100.0 * s['acceptance_rate']:>7.2f}%  "
                f"{100.0 * s['improvement_rate']:>8.3f}%"
            )
        lines.append(f"  total proposals: {total_prop}")
        lines.append(
            f"  phase switches: {self._phase_switches}"
            f" (reheat notifications: {self._reheat_signals};"
            f" stagnation peak: {self._stagnation_peak}/{self._phase_patience})"
        )
        lines.append(
            "  adaptive sigma -- "
            f"ChangeNodeHeight.sigma_frac={self._nh_sigma:.4f} "
            f"(n={self._sigma_obs.get(ChangeNodeHeight, 0)}), "
            f"ChangeInheritanceProb.sigma={self._cip_sigma:.4f} "
            f"(n={self._sigma_obs.get(ChangeInheritanceProb, 0)})"
        )
        lines.append(
            "  adaptive SPR radius -- "
            f"distance_decay last={self._spr_decay_last:.3f}, "
            f"range=[{self._spr_decay_min_seen:.3f}, "
            f"{self._spr_decay_max_seen:.3f}] "
            f"(max={_AdaptiveConfig.SPR_DECAY_MAX}, "
            f"min={_AdaptiveConfig.SPR_DECAY_MIN}; "
            "low=broad, high=local)"
        )
        lines.append(
            "  post-reheat broad-SPR window -- "
            f"activated={self._spr_broad_activations}x, "
            f"SPR draws in broad mode={self._spr_broad_proposals} "
            f"(window size={_AdaptiveConfig.SPR_BROAD_MODE_PROPOSALS})"
        )
        return "\n".join(lines)



# ======================================================================
# (H) Orchestration class
#
# :class:`MPL` is the user-facing entry point.  It couples the
# gene-tree precomputation, the scorer, and a search driver behind a
# small and consistent API: ``MPL(net, gts, mapping).score()`` /
# ``.search(...)``.  The static-method wrappers below simply forward
# to the module-level public functions so existing callers don't have
# to import them separately.
# ======================================================================


class MPL:
    """Maximum pseudo-likelihood scorer and search driver for a species network.

    At construction time, rho values are precomputed from the input
    gene trees (constant for the run) and a DP engine is built for
    the initial species network.  Subsequent :meth:`score` calls
    reuse the cached rho values and rebuild only the network-side DP.
    :meth:`search` wires the scorer into a search driver (HC or SA)
    with an :class:`MPLKernel` and returns the best log-PL found.

    Typical use::

        mpl = MPL(species_net, gene_trees, mapping)
        log_pl = mpl.score()                    # one-off score
        best_log_pl = mpl.search(method="sa", num_iter=20000)

    Attributes:
        net: Current species network.  Updated in place by
            :meth:`search` to the best network found.
        gene_trees: Input gene-tree collection.
        mapping: Species -> allele labels mapping.
    """

    def __init__(
        self,
        species_net: Network,
        gene_trees: GeneTrees,
        mapping: dict[str, list[str]],
    ) -> None:
        """Initialise rho precomputation and the initial DP engine.

        Gene-tree-side work (rho table, active-triplet filter) runs
        exactly once here.  Network-side DP is rebuilt on every
        :meth:`score` call because topology moves invalidate the
        engine's cached articulation sets.

        Args:
            species_net: Initial species network topology.  Branch
                lengths and gammas should already be populated.
            gene_trees: Gene-tree collection to score against.
            mapping: Species -> list of allele labels (identity
                mappings, i.e. ``{species: [species]}``, are fine).
        """
        self.net = species_net
        self.gene_trees = gene_trees
        self.mapping = mapping

        # Precompute rho (constant across any search that reuses this
        # object).  Triplets missing from the gene trees will still
        # appear in self._triplets but have all-zero rho, so the
        # active-triplet filter below skips them for scoring.
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

        # Initial DP engine for the starting network.  Replaced by
        # :meth:`search` once the driver finishes (so ``self.net`` and
        # this engine always agree).
        self._triple_engine = _TripleDPEngine(self.net)

    @classmethod
    def from_nexus(
        cls,
        gt_file: str,
        st_file: str,
        mapping: dict[str, list[str]],
    ) -> "MPL":
        """Construct an :class:`MPL` from two NEXUS files.

        Args:
            gt_file: Path to a NEXUS file holding the gene trees.
            st_file: Path to a NEXUS file holding the starting species
                network.
            mapping: Species -> list of allele labels.

        Returns:
            A fully initialised :class:`MPL` ready to be scored or
            searched.
        """
        st: Network = io.read_nexus(st_file, return_type="networks")
        gts: GeneTrees = io.read_nexus(gt_file, return_type="genetrees")
        gts.species_gene_mapping(mapping)
        return cls(st, gts, mapping)

    @staticmethod
    def compute_gene_tree_triplets(
        gene_trees: GeneTrees,
        mapping: dict[str, list[str]],
        species_labels: Optional[list[str]] = None,
    ) -> GeneTreeTripletResult:
        """Thin wrapper around module-level :func:`compute_gene_tree_triplets`.

        Provided so callers that only have the :class:`MPL` class
        handy don't have to separately import the module-level
        function.  See that function's docstring for parameter
        details.
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
        """Thin wrapper around module-level :func:`score_species_network_triplets`.

        See that function's docstring for parameter details.
        """
        return score_species_network_triplets(
            species_net=species_net,
            gene_triplet_result=gene_triplet_result,
        )

    # ── Scoring ───────────────────────────────────────────────────

    def score(self) -> float:
        """Compute the total log pseudo-likelihood for ``self.net``.

        Returns:
            Log pseudo-likelihood (always <= 0, higher is better).
        """
        if _HAS_CYTHON_MPL:
            return _score_with_cython(
                self._triple_engine, self._active_triplets, self._rho,
            )

        total = 0.0
        for triplet in self._active_triplets:
            total += self._score_triplet(triplet)
        return total

    # ── Search ────────────────────────────────────────────────────

    def search(
        self,
        method: str = "hc",
        num_iter: int = 700,
        kernel: MPLKernel | None = None,
        max_reticulations: int | None = None,
        *,
        preset: str = "default",
        opt_bl: bool | None = None,
        fix_st: bool | None = None,
        max_lvl: int | None = None,
        pseudo: bool = True,
        backbone: Network | None = None,
        optimize_params: bool | None = None,
        optimize_scope: str | None = None,
        optimize_band: float = 5.0,
        save_best_path: str | os.PathLike[str] | None = None,
        reference_network: Network | None = None,
        comparison_report_path: str | os.PathLike[str] | None = None,
        comparison_top_k: int = 25,
        print_comparison: bool = True,
        **kwargs,
    ) -> float:
        """Search the network space for the highest-scoring topology.

        Builds an :class:`MPLScorer`, wires RNG seeds through the
        model/kernel/driver trio for reproducibility, and runs either
        a Hill Climbing or Simulated Annealing driver against the
        starting network ``self.net``.  ``self.net`` and
        ``self._triple_engine`` are updated in place to the best
        network found.

        End-of-run diagnostics (move kernel stats) are printed
        unconditionally.

        Args:
            preset: One-word search profile (see
                :data:`phynetpy._search_flags.SEARCH_PRESETS`):
                ``"default"`` (recommended -- accurate r>=1 results at
                ~baseline speed via near-free gamma optimisation),
                ``"fast"`` (raw climb), ``"accurate"`` (per-topology
                optimisation of gammas + incident branches), or
                ``"phylonet"`` (reproduce PhyloNet's optimise-everything
                behaviour).  Any flag below passed explicitly overrides the
                preset.
            method: Search driver -- ``"hc"`` (Hill Climbing, default)
                or ``"sa"`` (Simulated Annealing).
            num_iter: Number of proposed moves to evaluate.
            kernel: Custom :class:`MPLKernel` instance.  When
                ``None`` a default one is built with ``max_reticulations``.
            max_reticulations: Upper bound on reticulation count.
                ``None`` means unlimited.  Ignored when a custom
                ``kernel`` is supplied.
            opt_bl: Optimise branch lengths (``opt_bl`` flag).  When
                ``True`` the proposal kernel drops the continuous
                parameter moves (``ChangeNodeHeight`` /
                ``ChangeInheritanceProb``) during the topology search,
                and the best network's branch lengths + gammas are
                optimised once at the end via Brent coordinate ascent
                against the pseudo-likelihood objective.  Ignored when a
                custom ``kernel`` is supplied (except the final
                optimisation, which still runs).
            fix_st: Fix the starting-tree backbone (``fix_st`` flag).
                When ``True`` the kernel drops backbone-changing moves
                (``SPR``) so only reticulation add/remove/relocate/
                endpoint and gamma moves are proposed.  Ignored when a
                custom ``kernel`` is supplied.
            max_lvl: Maximum network level (``max_lvl`` flag).  When set
                (e.g. ``1`` for level-1), proposals that would push the
                network level above this are rejected by an accept-path
                guard, and level-raising moves self-reject early.
                ``None`` disables the cap.
            pseudo: Pseudo-likelihood scoring (``pseudo`` flag).  MPL is
                a pseudo-likelihood method, so this is always effectively
                ``True``; passing ``False`` emits a warning and is
                ignored (use ``MCMC_GT`` / ``InferNetwork_ML`` for full
                likelihood).
            backbone: Network the result must contain as a subgraph.  Drops
                every move that could destroy existing structure and
                enforces containment in the accept path.  Set by
                ``infer(start=Start(net, mode=StartMode.AUGMENT))``; leave
                ``None`` for an unconstrained search.
            optimize_params: When ``True`` (Hill-Climbing only), refine a
                proposal's continuous parameters with a Brent coordinate
                ascent *before* the accept/reject decision, so each
                reticulation topology is judged at (near) its parameter
                optimum -- matching PhyloNet's per-round behaviour.  This
                is the fix for systematically-worse r>=1 inferences: with
                it off, a new reticulation is only ever scored at its
                birth gamma (0.5) during the climb.  Defaults to ``None``
                -> taken from ``preset`` (``True`` for ``"default"``).
                Ignored for ``method="sa"`` and when a custom ``kernel``
                drops the relevant moves.
            optimize_scope: Which parameters the per-topology optimisation
                touches (see :func:`phynetpy._optimize.optimize_network_parameters`).
                ``"gamma"`` optimises only the inheritance probabilities
                -- the cheapest scope, a no-op for trees (r=0), and the
                direct fix for the "reticulation judged at gamma=0.5"
                bias.  ``"reticulation"`` also optimises the branches
                incident to reticulations; ``"all"`` optimises every
                identifiable branch length too (closest to a full
                per-topology optimisation, but markedly slower).  Branch
                lengths skipped here are still tuned by the one-time final
                optimisation.  Defaults to ``None`` -> taken from
                ``preset`` (``"gamma"`` for ``"default"``).  Only used
                when ``optimize_params`` resolves to ``True``.
            optimize_band: Lazy-gating tolerance (in log-pseudo-likelihood
                units).  A reticulation-bearing proposal is optimised only
                when its raw score is within ``optimize_band`` of the
                current score (i.e. ``raw >= current - optimize_band``), so
                clearly-bad random proposals never pay the optimisation
                cost.  Larger values optimise more proposals (more
                accurate, slower).  Only used when ``optimize_params`` is
                ``True``.
            save_best_path: Optional Newick output path for the best
                network found.
            reference_network: Optional ground-truth / baseline
                network.  When provided, a comparison report is
                printed after the search.
            comparison_report_path: Optional path to also write the
                comparison report as UTF-8 text.
            comparison_top_k: Number of worst-performing triplets
                listed in the comparison report.
            print_comparison: When ``reference_network`` is set,
                whether to echo the report to stdout.
            **kwargs: Forwarded to the chosen search driver.  For SA
                these include ``t_start``, ``t_end``, ``n_restarts``,
                ``seed``, ``plateau_frac``, ``progress_every`` (print
                every *n* iterations; ``0`` disables), and
                ``schedule``: ``"cool"`` (high-to-low T), ``"heat"``
                (low-to-high T), or ``"geometric_reheat"`` (geometric
                cooling every ``steps_per_temp`` steps down to
                ``t_min``, with multi-signal reheats: rate-based
                plateau -- ``reheat_window`` + ``reheat_min_improve``;
                strict-stall backstop -- ``reheat_threshold``;
                optional frozen-chain detection --
                ``reheat_on_no_uphill``).  A cascade guard
                (``reheat_max_consecutive``) suspends further reheats
                after N fire without any strict run-best improvement;
                the counter resets on improvement.  Also see
                ``cooling_alpha``, ``steps_per_temp``,
                ``reheat_factor``, ``reheat_cap_mult``, ``t_min``.

        Returns:
            Log pseudo-likelihood of the best network found.

        Raises:
            ValueError: If ``method`` is not ``"hc"`` or ``"sa"``.
        """
        if not pseudo:
            warnings.warn(
                "MPL only supports pseudo-likelihood scoring; the "
                "pseudo=False flag is ignored.  Use MCMC_GT or "
                "InferNetwork_ML for full-likelihood inference.",
                stacklevel=2,
            )

        # Resolve the preset into concrete behaviour; explicit flags win.
        settings = resolve_search_preset(
            preset,
            optimize_params=optimize_params,
            optimize_scope=optimize_scope,
            opt_bl=opt_bl,
            fix_st=fix_st,
        )
        opt_bl = settings.opt_bl
        fix_st = settings.fix_st
        optimize_params = settings.optimize_params
        optimize_scope = settings.optimize_scope

        scorer = MPLScorer(self._rho, self._active_triplets)

        # ----------------------------------------------------------------
        # RNG plumbing.
        #
        # One SeedSequence drives three independent generators:
        #   * SA accept/reject (``sa_ss``),
        #   * proposal-class sampling inside the kernel (``kernel_ss``),
        #   * proposal *content* inside each move (``model_ss``, read
        #     via ``model.rng``).
        # This is the only way to make seeded MPL runs actually
        # reproducible; every tuning study depends on it.  HC does not
        # consume its own accept/reject RNG so we only inject ``seed``
        # into kwargs for the SA path.
        # ----------------------------------------------------------------
        seed_in = kwargs.pop("seed", None)
        if isinstance(seed_in, np.random.SeedSequence):
            root_ss = seed_in
        else:
            root_ss = np.random.SeedSequence(seed_in)
        sa_ss, kernel_ss, model_ss = root_ss.spawn(3)
        if method == "sa":
            kwargs["seed"] = sa_ss

        # ----------------------------------------------------------------
        # Build a Model around a deep copy of the starting network so
        # the driver's in-place mutations don't touch self.net until
        # the run completes successfully.
        # ----------------------------------------------------------------
        model = Model(rng=model_ss)
        model.network = copy.deepcopy(self.net)
        model.set_likelihood_calculator(scorer)

        # ----------------------------------------------------------------
        # Build or fix up the proposal kernel.  If the caller supplied
        # their own kernel but didn't wire in an RNG, seed it from the
        # same SeedSequence so reproducibility still holds.
        # ----------------------------------------------------------------
        if kernel is None:
            kernel = MPLKernel(
                move_types=resolve_move_types(
                    opt_bl=opt_bl, fix_st=fix_st,
                    augment_only=backbone is not None,
                ),
                max_reticulations=max_reticulations,
                max_level=max_lvl,
                rng=np.random.default_rng(kernel_ss),
            )
        elif getattr(kernel, "rng", None) is None:
            kernel.rng = np.random.default_rng(kernel_ss)

        # ----------------------------------------------------------------
        # Dispatch to the chosen search driver.  The level validator is
        # the authoritative ``max_lvl`` guard: it level-checks every
        # accepted proposal regardless of which move produced it.  When a
        # backbone is required, containment is enforced in the same place
        # and for the same reason.
        # ----------------------------------------------------------------
        validate = make_level_validator(
            max_lvl, base=make_containment_validator(backbone),
        )

        # ----------------------------------------------------------------
        # Per-topology continuous-parameter optimisation (levers 1 + 2).
        # Lever 1: optimise only the reticulation parameters (gammas +
        # incident branches) so reticulation topologies are judged near
        # their optimum -- the fix for systematically-worse r>=1 results.
        # Lever 2: lazy gating -- only optimise reticulation-bearing
        # proposals that are already within ``optimize_band`` of the
        # incumbent, so the vast majority of rejected proposals stay cheap.
        # ----------------------------------------------------------------
        optimize_proposal = None
        should_optimize = None
        if optimize_params:
            if method != "hc":
                warnings.warn(
                    "optimize_params is only supported for method='hc'; "
                    "ignoring it for this run.",
                    stacklevel=2,
                )
            else:
                def optimize_proposal(opt_m: Model) -> float:
                    return optimize_network_parameters(
                        opt_m, scorer, self.mapping, scope=optimize_scope,
                    )

                def should_optimize(
                    opt_m: Model, raw_proposed: float, current: float, move,
                ) -> bool:
                    # Only re-optimise after a *topology* move (matching
                    # PhyloNet's per-round behaviour); pure continuous moves
                    # already explored their parameter, so re-optimising the
                    # whole reticulation neighbourhood after them is wasted
                    # work.
                    if isinstance(move, CONTINUOUS_MOVES):
                        return False
                    net = opt_m.network
                    if not any(v.is_reticulation() for v in net.V()):
                        return False
                    return raw_proposed >= current - optimize_band

        if method == "hc":
            searcher = HillClimbing(
                pkernel=kernel,
                model=model,
                num_iter=num_iter,
                validate=validate,
                optimize_proposal=optimize_proposal,
                should_optimize=should_optimize,
                **kwargs,
            )
        elif method == "sa":
            searcher = SimulatedAnnealing(
                pkernel=kernel,
                model=model,
                num_iter=num_iter,
                validate=validate,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown search method {method!r}; use 'hc' or 'sa'")

        end_state = searcher.run()

        # ----------------------------------------------------------------
        # End-of-run diagnostics.  Cheap, one-time, and the single
        # best tool for tuning kernel weights.  Callers that want to
        # silence it can patch ``MPLKernel.format_stats`` or capture
        # stdout.
        # ----------------------------------------------------------------
        if hasattr(kernel, "format_stats"):
            print("\nMove kernel stats (lifetime):", flush=True)
            print(kernel.format_stats(), flush=True)

        # ----------------------------------------------------------------
        # Adopt the best network into self.net and rebuild the DP
        # engine so subsequent :meth:`score` calls see the new
        # topology.  Optional Newick / comparison-report artifacts.
        # ----------------------------------------------------------------
        self.net = end_state.current_model.network
        self._triple_engine = _TripleDPEngine(self.net)

        # ----------------------------------------------------------------
        # Final Brent coordinate-ascent over *all* branch lengths and
        # gammas of the best network, against the pseudo-likelihood
        # objective (the same scorer the search used).  Runs whenever the
        # resolved preset requests a final optimisation (always for the
        # shipped presets) or ``opt_bl`` dropped the continuous moves.
        # ----------------------------------------------------------------
        final_score: float | None = None
        if settings.final_optimize or opt_bl:
            opt_model = Model(rng=np.random.default_rng())
            opt_model.network = self.net
            opt_model.set_likelihood_calculator(scorer)
            final_score = optimize_network_parameters(
                opt_model, scorer, self.mapping,
            )
            self.net = opt_model.network
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

        if final_score is not None:
            return final_score
        return end_state.likelihood()

    # ── Internal ──────────────────────────────────────────────────

    def _score_triplet(self, triplet: tuple[str, str, str]) -> float:
        """Contribution of one species triplet to the log-PL sum.

        Only used by the Python scoring fallback in :meth:`score`;
        the Cython path handles every triplet in a single call.

        Args:
            triplet: Canonical species triplet.

        Returns:
            ``sum_i rho_i * log(p_i)`` across the three resolutions,
            with ``p_i == 0`` terms replaced by ``_LOG_FLOOR`` and
            ``rho_i == 0`` terms skipped entirely.
        """
        x, y, z = triplet
        p_xy = self._triple_engine.calculate_triple_probability((x, y, z))
        p_xz = self._triple_engine.calculate_triple_probability((x, z, y))
        probs = (p_xy, p_xz, max(1.0 - p_xy - p_xz, 0.0))
        rho = self._rho[triplet]

        contribution = 0.0
        for rho_i, p_i in zip(rho, probs):
            if rho_i > 0.0:
                contribution += rho_i * (math.log(p_i) if p_i > 0.0 else _LOG_FLOOR)
        return contribution
