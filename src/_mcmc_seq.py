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
First Included in Version : 0.5.0

MCMC_SEQ -- Bayesian co-estimation of reticulate phylogenies and gene trees
directly from multilocus sequence alignments.

Faithful re-implementation of PhyloNet's ``MCMC_SEQ`` (Wen & Nakhleh 2018,
*Systematic Biology* 67(3):439-457, "Coestimating Reticulate Phylogenies and
Gene Trees from Multilocus Sequence Data").  The likelihood -- the part that is
verified to match PhyloNet bit-for-bit -- lives in
:mod:`phynetpy._seq_likelihood`; this module is the Bayesian wrapper around it:

* the **hierarchical posterior** (per-locus Felsenstein likelihood + timed
  MSNC density + network/parameter priors, all matching PhyloNet's defaults),
* an **incremental likelihood cache** so single-locus / single-parameter moves
  rescore only what changed,
* a **reversible RJMCMC kernel** that co-samples the per-locus gene trees
  (topology + coalescent times), the species network (node heights,
  reticulation inheritance probabilities, and reticulation count via
  add/delete), and the population mutation rate ``theta``,
* the public :class:`MCMC_SEQ` driver, mirroring the surface of
  :class:`phynetpy.infer.MCMC_GT`.

Units & defaults follow PhyloNet exactly: heights/branch lengths are in
expected substitutions per site, ``theta = 4 N mu`` (per-pair coalescent rate
``2/theta``), the default substitution model is JC69, the default population
size is constant across branches with prior mean 0.036 (starting value 0.02),
and the number of
reticulations carries a Poisson(1.0) prior truncated at ``max_reticulations``.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

import copy
import math
import multiprocessing as mp
import queue as _queue
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .Network import Network, Node, Edge
from . import _network_moves as _nm
from .GraphUtils import level as _network_level
from .MSA import MSA
from ._seq_likelihood import (
    SubstitutionModel,
    JC69,
    FelsensteinCalculator,
    gene_tree_msnc_log_density,
    _node_height,
)
from ._msnc_density import (
    build_network_msnc_index,
    build_gene_tree_msnc_index,
    build_gene_tree_topology_index,
    gene_tree_events,
    msnc_log_density_prebuilt,
)
from ._chain_analysis import (
    ChainSummary,
    summarize_traces,
    write_tracer_log,
    write_trees_nexus,
    gelman_rubin,
)


__all__ = [
    "MCMC_SEQ",
    "MCMCSeqPriors",
    "MCMCSeqSample",
    "MCMCSeqResult",
    "SeqState",
    "build_upgma_gene_tree",
    "run_parallel_chains",
    "MultiChainStatus",
    "MultiChainResult",
]


# ======================================================================
# Priors (PhyloNet defaults)
# ======================================================================

@dataclass
class MCMCSeqPriors:
    """Prior hyperparameters for :class:`MCMC_SEQ` (PhyloNet defaults).

    Attributes:
        poisson_mean: Mean of the Poisson prior on the number of
            reticulation nodes (PhyloNet ``-pp``; default 1.0).
        max_reticulations: Hard cap on reticulations (PhyloNet ``-mr``;
            default 4).
        max_level: Hard cap on network *level* -- the maximum number of
            reticulations in any single biconnected component ("blob").
            ``1`` restricts the sampler to level-1 (galled) networks,
            whose displayed-tree count grows linearly rather than
            combinatorially, so the MSNC ancestral-configuration DP stays
            cheap.  Reticulation-adding / relocating moves that would
            exceed the cap self-reject *before* the expensive coupled
            gene-tree re-proposal and scoring, so the constraint also acts
            as a runtime optimisation.  ``None`` (default) disables the
            cap (only ``max_reticulations`` applies).
        theta_shape: Shape of the Gamma prior on the population mutation
            rate ``theta`` (PhyloNet ``GAMMA_SHAPE = 2``).
        theta_prior_mean: Mean of the Gamma *prior* on ``theta`` (PhyloNet
            ``-ptheta``; default 0.036).  Distinct from the starting value:
            this is the regularising centre of the density, deliberately
            decoupled from where the chain begins so the prior actually
            resists the coalescent's ``theta -> 0`` degeneracy.
        theta_mean: Starting value for ``theta`` (PhyloNet ``-sps``; default
            0.02).  Used to initialise the chain only, not as the prior mean.
        diameter_mean: Mean of the exponential prior on reticulation-node
            diameters, in the network's own units (expected substitutions per
            site).  ``Exp(1/diameter_mean)`` *penalises large* diameters, so it
            favours compact, biologically-plausible, computationally-cheap
            reticulations (Wen et al. 2016).  It should be on the order of the
            network's root height; the previous default (10.0) was ~70x the
            tree scale and therefore numerically inert.  Note this soft prior
            is dominated by the sequence likelihood on large data and is *not*
            an identifiability guard -- an exponential density actually rewards
            tiny (degenerate) diameters, so the anti-degeneracy work is done by
            ``min_cycle_size`` / ``min_branch_length`` / the Beta gamma prior.
        use_diameter_prior: Whether the diameter prior is active.
        gamma_alpha: Alpha of the Beta prior on each inheritance probability.
        gamma_beta: Beta of the Beta prior on each inheritance probability.
            The default ``Beta(2, 2)`` vanishes at gamma = 0 and 1, keeping the
            sampler off the *unidentifiable* boundaries (gamma -> 0/1 makes a
            reticulation vestigial; Banos 2019, Allman-Banos-Rhodes 2024).
            Set both to ``1.0`` for the flat Uniform[0, 1] (PhyloNet default).
        min_cycle_size: Reject networks whose any reticulation sits on a cycle
            (in the underlying undirected graph) with fewer than this many
            nodes.  Level-1 identifiability requires cycle size > 3 (Solis-Lemus
            & Ane; Ane et al. "Extracting diamonds"), and 2-cycles are provably
            *totally undetectable*; the default ``4`` forbids the degenerate
            "bubble" reticulations the chain otherwise gets trapped on.  Set to
            ``0`` to disable the guard.
        min_branch_length: Reject networks with a hybrid edge shorter than this
            (in substitution units).  Zero-length hybrid edges (t -> 0) are an
            identifiability degeneracy; a tiny floor keeps proposals off it.
        branch_length_mean: Mean of the exponential prior ``p(C_lambda|delta) ~
            Exp(delta)`` placed on *every* branch length (Wen et al. 2016, PLoS
            Genetics, Eq. for ``p(C)``; PhyloNet default ``Exp(10)`` -> mean
            0.1 expected substitutions per site).  This is the piece of
            PhyloNet's network prior PhyNetPy previously omitted.  Without it the
            branch lengths carry only an implicit *flat/improper* prior, so the
            network posterior is improper in those dimensions and the
            reticulation's posterior *mass* is scaled by the arbitrary
            substitution-scale proposal window -- producing a spurious ~log(1/
            window) ~ 7-nat bias *against* every reticulation in the RJMCMC
            add/delete acceptance.  A proper ``Exp(delta)`` prior on the same
            scale as the proposal window replaces that ill-defined volume term
            with a scale-consistent density term, so trans-dimensional moves are
            governed by the likelihood + a proper prior rather than by the
            proposal geometry.
        use_branch_length_prior: Whether the ``Exp(delta)`` branch-length prior
            is active (default ``True``).
    """

    poisson_mean: float = 1.0
    max_reticulations: int = 4
    max_level: Optional[int] = None
    theta_shape: float = 2.0
    theta_prior_mean: float = 0.036
    theta_mean: float = 0.02
    diameter_mean: float = 0.5
    use_diameter_prior: bool = True
    gamma_alpha: float = 2.0
    gamma_beta: float = 2.0
    min_cycle_size: int = 4
    min_branch_length: float = 1e-6
    branch_length_mean: float = 0.1
    use_branch_length_prior: bool = True


def _log_poisson_pmf(k: int, mean: float) -> float:
    """Log of the Poisson(``mean``) pmf at ``k``."""
    if mean <= 0.0:
        return 0.0 if k == 0 else float("-inf")
    return k * math.log(mean) - mean - math.lgamma(k + 1)


def _log_gamma_pdf(x: float, shape: float, mean: float) -> float:
    """Log density of a Gamma distribution parameterised by shape + mean."""
    if x <= 0.0:
        return float("-inf")
    scale = mean / shape
    return (
        (shape - 1.0) * math.log(x)
        - x / scale
        - shape * math.log(scale)
        - math.lgamma(shape)
    )


def _log_beta_pdf(x: float, alpha: float, beta: float) -> float:
    """Log density of a Beta(``alpha``, ``beta``) distribution at ``x``.

    Returns ``-inf`` at (or outside) the open interval ``(0, 1)`` whenever the
    density genuinely vanishes there -- which for ``alpha, beta > 1`` is exactly
    the unidentifiable ``gamma -> 0/1`` boundary we want to forbid.
    """
    if x <= 0.0 or x >= 1.0:
        # Beta(1,1) is flat and finite on the closed interval; every other
        # shape with alpha,beta >= 1 vanishes at the boundary.
        if alpha == 1.0 and beta == 1.0:
            return 0.0
        return float("-inf")
    return (
        (alpha - 1.0) * math.log(x)
        + (beta - 1.0) * math.log1p(-x)
        - (math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta))
    )


def _ancestor_depths(net: Network, start: Node) -> dict[Node, int]:
    """Map every ancestor of ``start`` (inclusive) to its min edge-distance."""
    dist: dict[Node, int] = {start: 0}
    frontier = [start]
    while frontier:
        nxt: list[Node] = []
        for u in frontier:
            for p in net.get_parents(u):
                if p not in dist:
                    dist[p] = dist[u] + 1
                    nxt.append(p)
        frontier = nxt
    return dist


def _reticulation_cycle_size(net: Network, retic: Node) -> int:
    """Number of nodes on the reticulation's cycle (undirected graph).

    The cycle is the reticulation ``retic`` plus the two ancestral paths from
    its parents up to their lowest common ancestor.  A return of ``2`` means
    parallel edges (both parents identical); ``0`` means the node is not a
    proper 2-parent reticulation (treated as degenerate by the caller).
    """
    parents = net.get_parents(retic)
    if len(parents) < 2:
        return 0
    p1, p2 = parents[0], parents[1]
    if p1 is p2:
        return 2
    d1 = _ancestor_depths(net, p1)
    d2 = _ancestor_depths(net, p2)
    common = [n for n in d1 if n in d2]
    if not common:
        return 0
    lca = min(common, key=lambda n: d1[n] + d2[n])
    # r + (path p1..lca) + (path p2..lca), lca shared once.
    return d1[lca] + d2[lca] + 2


def log_prior_seq(
    species_net: Network,
    theta: float,
    priors: MCMCSeqPriors,
) -> float:
    """Log prior ``log p(Psi) + log p(theta)`` for the species network.

    Combines:

    * the Poisson prior on the reticulation count (with the
      ``1 / (E_i(E_i - 1))`` per-reticulation topology normaliser of Wen et
      al. 2016, where ``E_i`` is the running edge count),
    * the Gamma prior on the population mutation rate ``theta``,
    * the ``Exp(delta)`` prior ``p(C_lambda|delta)`` on every branch length
      (Wen et al. 2016; PhyloNet default ``Exp(10)``).  This makes the network
      posterior proper in the branch-length dimensions and supplies the
      scale-consistent density term that balances the RJMCMC add/delete
      proposal volume (without it, reticulations carry a spurious ~7-nat
      volume penalty at substitution scale),
    * (optionally) an exponential prior on reticulation-node diameters
      (the time spanned between a reticulation's two parents),
    * a ``Beta(gamma_alpha, gamma_beta)`` prior on each inheritance
      probability (default ``Beta(2, 2)``, which vanishes at the ``gamma ->
      0/1`` boundaries).

    Following Wen & Nakhleh (2018), any network that violates the phylogenetic
    identifiability conditions is assigned zero prior (``-inf``) so the proposal
    is rejected: a reticulation whose undirected cycle has ``< min_cycle_size``
    nodes (2-cycles are *totally* undetectable; ``> 3`` is required for generic
    level-1 identifiability), or a hybrid edge shorter than
    ``min_branch_length`` (the ``t -> 0`` degeneracy).

    Args:
        species_net: The species network.
        theta: Population mutation rate.
        priors: Hyperparameters.

    Returns:
        The log prior (``-inf`` if the reticulation cap is exceeded or the
        network sits in an unidentifiable/degenerate region).
    """
    reticulations = [v for v in species_net.V() if v.is_reticulation()]
    n_ret = len(reticulations)
    if n_ret > priors.max_reticulations:
        return float("-inf")

    # --- Identifiability guards: forbid degenerate reticulations ----------
    # These are hard constraints (prior = 0) because, unlike a soft prior, they
    # must still bite against a likelihood that is O(1e5) nats on large data.
    if n_ret:
        for r in reticulations:
            if priors.min_cycle_size > 0:
                csize = _reticulation_cycle_size(species_net, r)
                if csize < priors.min_cycle_size:
                    return float("-inf")
            if priors.min_branch_length > 0.0:
                for e in species_net.in_edges(r):
                    if float(e.get_length()) < priors.min_branch_length:
                        return float("-inf")

    lp = _log_poisson_pmf(n_ret, priors.poisson_mean)

    # Topology normaliser: E starts at 2n-2 (n leaves) and grows by 3 per
    # reticulation; each reticulation contributes -log(E(E-1)).
    n_leaves = len(species_net.get_leaves())
    edges = 2 * n_leaves - 2
    for _ in range(n_ret):
        if edges > 1:
            lp -= math.log(edges * (edges - 1))
        edges += 3

    lp += _log_gamma_pdf(theta, priors.theta_shape, priors.theta_prior_mean)

    # Exp(delta) prior on every branch length, p(C_lambda | delta) (Wen et al.
    # 2016).  This is the term PhyNetPy previously omitted; it makes the
    # posterior proper over branch lengths and, crucially, supplies the density
    # term that offsets the RJMCMC proposal-volume factor on add/delete.  A
    # uniform proposal on a window of width w has density 1/w; a matched Exp
    # prior contributes ~log(delta) per branch, so delta*w ~ O(1) rather than
    # the ~e^7 substitution-scale window blowing up the trans-dimensional ratio.
    # Lengths are derived from the ultrametric node heights (height(parent) -
    # height(child)) so the term is consistent with the geometry the MSNC /
    # Felsenstein factors read and avoids the stale-length-field warning.
    if priors.use_branch_length_prior and priors.branch_length_mean > 0.0:
        bl_rate = 1.0 / priors.branch_length_mean
        log_bl_rate = math.log(bl_rate)
        heights = _heights(species_net)
        for e in species_net.E():
            length = heights[e.src] - heights[e.dest]
            if length < 0.0:
                length = 0.0
            lp += log_bl_rate - bl_rate * length

    # Beta prior on each reticulation's inheritance probability.  Beta(2,2)
    # (default) -> -inf at gamma = 0/1, keeping the chain off the vestigial
    # (unidentifiable) reticulation boundary the freeze parked on.
    if n_ret and not (priors.gamma_alpha == 1.0 and priors.gamma_beta == 1.0):
        for r in reticulations:
            in_e = list(species_net.in_edges(r))
            if in_e:
                g = in_e[0].get_gamma()
                if g is not None:
                    lp += _log_beta_pdf(
                        float(g), priors.gamma_alpha, priors.gamma_beta
                    )

    if priors.use_diameter_prior and n_ret:
        height_cache: dict[Node, float] = {}
        rate = 1.0 / priors.diameter_mean
        for r in reticulations:
            parents = species_net.get_parents(r)
            if len(parents) >= 2:
                h0 = _node_height(species_net, parents[0], height_cache)
                h1 = _node_height(species_net, parents[1], height_cache)
                diam = abs(h0 - h1)
                lp += math.log(rate) - rate * diam

    return lp


# ======================================================================
# Height bookkeeping (ultrametric, substitution units)
# ======================================================================

def _heights(net: Network) -> dict[Node, float]:
    """Map every node to its height above the present (memoised post-order)."""
    cache: dict[Node, float] = {}
    for v in net.V():
        _node_height(net, v, cache)
    return cache


def _sync_lengths(net: Network, heights: dict[Node, float]) -> None:
    """Set every edge length to ``height(parent) - height(child)``.

    Keeps the network's stored branch lengths consistent with an explicit
    ultrametric height assignment so both likelihood factors (Felsenstein on
    edge lengths, MSNC on node heights) see the same geometry.

    Args:
        net: Network to update in place.
        heights: Node -> height map covering every node.
    """
    for e in net.E():
        src, dest = e.src, e.dest
        length = heights[src] - heights[dest]
        if length < 0.0:
            length = 0.0
        e.set_length(float(length))


def _clone_net(net: Network) -> Network:
    """Fast structural clone of a species network / gene tree.

    Uses :meth:`Network.copy` -- which rebuilds fresh ``Node`` / ``Edge``
    objects directly -- instead of :func:`copy.deepcopy`.  It is ~4x faster
    because it skips generic deepcopy's reflective, memoised object graph walk,
    and it is *exact* for sampling purposes: node labels, reticulation flags,
    edge lengths and inheritance probabilities (everything the Felsenstein and
    MSNC likelihood factors read) are all reproduced.  Node attribute dicts are
    shared with the source, which is safe throughout the samplers because they
    never mutate node attributes in place -- ultrametric heights live in
    external ``dict[Node, float]`` maps and are written back onto edge lengths.
    """
    return net.copy()[0]


def _descendant_species(
    net: Network, species_of: dict[str, str]
) -> dict[Node, frozenset]:
    """Map every node to the set of species labels at/below it.

    Args:
        net: A tree or network.
        species_of: Leaf-label -> species-label map; labels absent from the map
            are treated as their own species (identity).

    Returns:
        ``node -> frozenset(species labels)`` for every node.
    """
    memo: dict[Node, frozenset] = {}

    def desc(v: Node) -> frozenset:
        cached = memo.get(v)
        if cached is not None:
            return cached
        kids = net.get_children(v)
        if not kids:
            r = frozenset({species_of.get(v.label, v.label)})
        else:
            acc: set = set()
            for c in kids:
                acc |= desc(c)
            r = frozenset(acc)
        memo[v] = r
        return r

    for v in net.V():
        desc(v)
    return memo


def _enforce_embedding_consistency(
    species_net: Network,
    net_heights: dict[Node, float],
    gene_trees: list[Network],
    gt_heights: list[dict[Node, float]],
    species_of: dict[str, str],
    *,
    eps: float = 0.05,
) -> float:
    """Scale ``net_heights`` down until every gene tree embeds validly.

    A coalescence joining lineages from two or more species must occur at or
    above the height of their most-recent common ancestor in the species
    network (multispecies-coalescent constraint); otherwise the MSNC density is
    zero and the likelihood is ``-inf``.  Independently-built starting trees
    (UPGMA on concatenated data for the species tree, per-locus UPGMA for the
    gene trees) rarely satisfy this, so we uniformly shrink the species network
    -- which leaves the gene-tree branch lengths (and hence the Felsenstein
    likelihood) untouched -- until the deepest binding constraint clears with a
    safety margin ``eps``.

    Args:
        species_net: Starting species network (modified only via ``net_heights``).
        net_heights: Species-node heights (mutated in place when scaling).
        gene_trees: Per-locus starting gene trees.
        gt_heights: Per-locus gene-node heights.
        species_of: Allele-label -> species-label map.
        eps: Fractional safety margin kept below each constraint.

    Returns:
        The applied scale factor (``1.0`` if the start was already valid).
    """
    sp_desc = _descendant_species(species_net, {})
    sp_heights = net_heights
    # Species nodes in ascending height order, to find the lowest node whose
    # descendant-species set covers a given coalescence (its MRCA).
    nodes_by_h = sorted(species_net.V(), key=lambda v: sp_heights[v])

    # Upper bound on each species node's height = min height of any cross-
    # species coalescence whose species MRCA is that node.
    bounds: dict[Node, float] = {v: math.inf for v in species_net.V()}
    for gi, gt in enumerate(gene_trees):
        gh = gt_heights[gi]
        gdesc = _descendant_species(gt, species_of)
        for u in gt.V():
            if not gt.get_children(u):
                continue
            su = gdesc[u]
            if len(su) < 2:  # within-species coalescence: unconstrained
                continue
            hu = gh[u]
            for v in nodes_by_h:  # lowest node covering su == the MRCA
                if su <= sp_desc[v]:
                    if hu < bounds[v]:
                        bounds[v] = hu
                    break

    scale = 1.0
    for v, upper in bounds.items():
        h = sp_heights[v]
        if math.isfinite(upper) and h > 0.0:
            ratio = (1.0 - eps) * upper / h
            if ratio < scale:
                scale = ratio

    if scale < 1.0:
        for v in net_heights:
            net_heights[v] *= scale
    return scale


# ======================================================================
# UPGMA starting gene trees (PhyloNet's default -sgt)
# ======================================================================

def _jc_distance(s1: str, s2: str) -> float:
    """Jukes-Cantor corrected distance between two aligned sequences."""
    n = 0
    diff = 0
    for a, b in zip(s1, s2):
        a, b = a.upper(), b.upper()
        if a in "ACGT" and b in "ACGT":
            n += 1
            if a != b:
                diff += 1
    if n == 0:
        return 1.0
    p = diff / n
    if p >= 0.75:
        return 3.0  # saturated; cap the correction
    return -0.75 * math.log(1.0 - (4.0 / 3.0) * p)


def build_upgma_gene_tree(alignment: dict[str, str]) -> Network:
    """Build an ultrametric UPGMA starting gene tree from one locus.

    Mirrors PhyloNet's default ``-sgt`` (UPGMA on JC distances).  Node heights
    are half the cluster distance, so branch lengths are in substitution units
    and the tree is ultrametric -- a valid starting point for the coalescent
    density.

    Args:
        alignment: Map from allele label -> aligned nucleotide string.

    Returns:
        A rooted, ultrametric :class:`Network` (a tree).
    """
    labels = list(alignment.keys())
    if len(labels) == 1:
        net = Network()
        only = Node(name=labels[0])
        net.add_nodes(only)
        return net

    # Pairwise JC distance matrix.
    clusters: list[dict[str, Any]] = []
    for lab in labels:
        clusters.append({"newick": lab, "height": 0.0, "size": 1, "members": [lab]})
    dist: dict[tuple[int, int], float] = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            dist[(i, j)] = _jc_distance(alignment[labels[i]], alignment[labels[j]])

    active = list(range(len(labels)))
    next_id = len(labels)
    while len(active) > 1:
        # Find closest pair.
        best = None
        best_d = float("inf")
        for a in range(len(active)):
            for b in range(a + 1, len(active)):
                ia, ib = active[a], active[b]
                key = (min(ia, ib), max(ia, ib))
                d = dist[key]
                if d < best_d:
                    best_d = d
                    best = (a, b)
        a, b = best
        ia, ib = active[a], active[b]
        ca, cb = clusters[ia], clusters[ib]
        new_height = best_d / 2.0
        la = max(new_height - ca["height"], 1e-9)
        lb = max(new_height - cb["height"], 1e-9)
        new_newick = f"({ca['newick']}:{la:.10f},{cb['newick']}:{lb:.10f})I{next_id}"
        new_cluster = {
            "newick": new_newick,
            "height": new_height,
            "size": ca["size"] + cb["size"],
            "members": ca["members"] + cb["members"],
        }
        clusters.append(new_cluster)
        new_idx = len(clusters) - 1
        # Update distances (UPGMA average linkage).
        remaining = [x for k, x in enumerate(active) if k not in (a, b)]
        for ic in remaining:
            cc = clusters[ic]
            d_ac = dist[(min(ia, ic), max(ia, ic))]
            d_bc = dist[(min(ib, ic), max(ib, ic))]
            avg = (ca["size"] * d_ac + cb["size"] * d_bc) / (ca["size"] + cb["size"])
            dist[(min(new_idx, ic), max(new_idx, ic))] = avg
        active = remaining + [new_idx]
        next_id += 1

    return Network.from_newick(clusters[active[0]]["newick"] + ";")


# ======================================================================
# Likelihood engine with per-locus caching
# ======================================================================

class _SeqLikelihoodEngine:
    """Incremental posterior likelihood over all loci.

    Holds one :class:`FelsensteinCalculator` per locus plus per-locus caches
    of the phylogenetic likelihood ``P(S_i | g_i)`` and the MSNC density
    ``P(g_i | Psi)``.  Callers mark loci / the network dirty; only dirty
    components are rescored.

    Attributes:
        n_loci: Number of loci.
    """

    def __init__(
        self,
        loci: list[dict[str, str]],
        species_of: dict[str, str],
        model: SubstitutionModel,
    ) -> None:
        """Initialise calculators and empty caches.

        Args:
            loci: Per-locus alignments (allele label -> sequence string).
            species_of: Allele label -> species label.
            model: Nucleotide substitution model.
        """
        self._calcs = [FelsensteinCalculator(aln) for aln in loci]
        self._species_of = species_of
        self._model = model
        self.n_loci = len(loci)
        self._felsen: list[Optional[float]] = [None] * self.n_loci
        self._msnc: list[Optional[float]] = [None] * self.n_loci

        # ----- reusable MSNC indices ------------------------------------
        # Rebuilding ``_NetworkIndex`` / ``_GeneTreeIndex`` (and the O(V)
        # node-height walks + gene-coalescence-event sort) once per
        # (locus, network) pair dominated the SEQ profile.  Every MCMC_SEQ
        # move copy-and-swaps the object it mutates, so we can safely key
        # these caches by *object identity* (``is``): a fresh object means
        # a genuine change, and holding the reference here keeps the cached
        # object alive so its ``id`` can never be recycled under us.
        self._net_obj: Optional[Network] = None
        self._net_idx = None
        self._sp_heights: Optional[list[float]] = None
        self._gt_obj: list[Optional[Network]] = [None] * self.n_loci
        self._gti: list[object] = [None] * self.n_loci
        self._events: list[object] = [None] * self.n_loci

    def invalidate_locus(self, i: int) -> None:
        """Drop cached factors for locus ``i`` (after a gene-tree move)."""
        self._felsen[i] = None
        self._msnc[i] = None

    def invalidate_network(self) -> None:
        """Drop all MSNC caches (after a network / theta move)."""
        self._msnc = [None] * self.n_loci

    def _network_index(self, species_net: Network):
        """Fetch (or rebuild) the cached network view + node heights."""
        if species_net is not self._net_obj:
            self._net_idx, self._sp_heights = build_network_msnc_index(
                species_net
            )
            self._net_obj = species_net
        return self._net_idx, self._sp_heights

    def _gene_index(self, i: int, gene_tree: Network):
        """Fetch (or rebuild) locus ``i``'s cached gene-tree view + events."""
        if gene_tree is not self._gt_obj[i]:
            self._gti[i], self._events[i] = build_gene_tree_msnc_index(
                gene_tree, self._species_of
            )
            self._gt_obj[i] = gene_tree
        return self._gti[i], self._events[i]

    def log_likelihood(
        self,
        gene_trees: list[Network],
        species_net: Network,
        theta: float,
    ) -> float:
        """Total log likelihood ``sum_i [log P(S_i|g_i) + log P(g_i|Psi)]``.

        Args:
            gene_trees: Current per-locus gene trees.
            species_net: Current species network.
            theta: Current population mutation rate.

        Returns:
            The summed log likelihood, or ``-inf`` when any locus fails to
            embed in the network (a zero-probability state).  Returning
            ``-inf`` -- rather than a finite floor -- is essential: it makes
            Metropolis-Hastings *reject* invalid proposals instead of treating
            them as merely low-probability, which would let the chain wander
            into (and get trapped in) the vast region of invalid embeddings.
        """
        net_idx, sp_heights = self._network_index(species_net)
        total = 0.0
        for i in range(self.n_loci):
            if self._felsen[i] is None:
                self._felsen[i] = self._calcs[i].log_likelihood(
                    gene_trees[i], self._model
                )
            if self._msnc[i] is None:
                gti, events = self._gene_index(i, gene_trees[i])
                self._msnc[i] = msnc_log_density_prebuilt(
                    net_idx, sp_heights, gti, events, theta
                )
            total += self._felsen[i] + self._msnc[i]
        if not math.isfinite(total):
            return float("-inf")
        return total

    def clone_caches(self) -> tuple[list, list]:
        """Snapshot the caches (for cheap move rollback)."""
        return list(self._felsen), list(self._msnc)

    def restore_caches(self, snapshot: tuple[list, list]) -> None:
        """Restore caches from a :meth:`clone_caches` snapshot."""
        self._felsen, self._msnc = list(snapshot[0]), list(snapshot[1])


# ======================================================================
# State
# ======================================================================

class SeqState:
    """Full latent state of an MCMC_SEQ chain.

    Bundles the species network, the per-locus gene trees, the population
    mutation rate, the substitution model and the likelihood engine, and
    exposes :meth:`log_posterior` plus snapshot / restore for move rollback.

    Attributes:
        species_net: Current species network (ultrametric, substitution units).
        gene_trees: One gene tree per locus.
        theta: Current population mutation rate.
    """

    def __init__(
        self,
        species_net: Network,
        gene_trees: list[Network],
        species_of: dict[str, str],
        loci: list[dict[str, str]],
        priors: MCMCSeqPriors,
        model: SubstitutionModel,
        theta: float,
    ) -> None:
        """Initialise state and prime the likelihood caches.

        Args:
            species_net: Starting species network.
            gene_trees: Starting per-locus gene trees.
            species_of: Allele -> species map.
            loci: Per-locus alignments.
            priors: Prior hyperparameters.
            model: Substitution model.
            theta: Starting population mutation rate.
        """
        self.species_net = species_net
        self.gene_trees = gene_trees
        self.species_of = species_of
        self.priors = priors
        self.theta = theta
        self.model = model
        self._engine = _SeqLikelihoodEngine(loci, species_of, model)

        # Explicit ultrametric heights are the source of truth; edge lengths
        # are kept in sync so the likelihood factors see one consistent
        # geometry.
        self.net_heights: dict[Node, float] = _heights(species_net)
        self.gt_heights: list[dict[Node, float]] = [_heights(gt) for gt in gene_trees]
        # Independently-built starting trees may place gene coalescences below
        # species divergences (invalid embedding -> -inf likelihood).  Shrink
        # the species network until every locus embeds validly.
        _enforce_embedding_consistency(
            species_net, self.net_heights, gene_trees, self.gt_heights, species_of
        )
        self.resync_net()
        for i in range(len(gene_trees)):
            self.resync_gt(i)

    def resync_net(self) -> None:
        """Rewrite species-network edge lengths from ``net_heights``."""
        _sync_lengths(self.species_net, self.net_heights)

    def resync_gt(self, i: int) -> None:
        """Rewrite gene-tree ``i`` edge lengths from ``gt_heights[i]``."""
        _sync_lengths(self.gene_trees[i], self.gt_heights[i])

    def log_likelihood(self) -> float:
        """Current total phylogenetic + coalescent log likelihood."""
        return self._engine.log_likelihood(
            self.gene_trees, self.species_net, self.theta
        )

    def log_prior(self) -> float:
        """Current log prior on the network + ``theta``."""
        return log_prior_seq(self.species_net, self.theta, self.priors)

    def log_posterior(self) -> float:
        """Current unnormalised log posterior (likelihood + prior)."""
        return self.log_likelihood() + self.log_prior()

    def num_reticulations(self) -> int:
        """Reticulation-node count of the species network."""
        return sum(1 for v in self.species_net.V() if v.is_reticulation())


# ======================================================================
# Reversible MCMC operators
# ======================================================================
#
# Every operator returns either ``None`` (the proposal was illegal on the
# current state -- treated as an immediate reject) or a tuple
# ``(log_hastings_ratio, undo)`` where ``undo`` is a zero-argument callable
# that exactly reverses the proposal.  Operators use a copy-and-swap strategy:
# the move is built on a deep copy of the affected gene tree / network, then
# swapped into the state; ``undo`` simply swaps the original object (and the
# cached likelihood factors) back.  This makes rollback trivially correct.


def _internal_nodes(net: Network) -> list[Node]:
    """Non-root internal nodes (in-degree >= 1 and out-degree >= 1)."""
    return [
        v for v in net.V()
        if net.out_degree(v) > 0 and net.in_degree(v) > 0
    ]


def _edge_between(net: Network, parent: Node, child: Node) -> Edge:
    """The single directed edge ``parent -> child``."""
    e = net.get_edge(parent, child)
    return e[0] if isinstance(e, list) else e


def op_change_theta(
    state: SeqState, rng: np.random.Generator, *, window: float = 0.4
) -> Optional[tuple[float, Callable]]:
    """Scale the population mutation rate ``theta`` (single-parameter scaler).

    A multiplicative random walk ``theta' = theta * exp(window*(u-0.5))``; the
    Hastings ratio of a scale move on one positive parameter is ``log(f)``.
    """
    old = state.theta
    snap = state._engine.clone_caches()
    f = math.exp(window * (rng.random() - 0.5))
    state.theta = old * f
    state._engine.invalidate_network()

    def undo() -> None:
        state.theta = old
        state._engine.restore_caches(snap)

    return math.log(f), undo


def op_change_gamma(
    state: SeqState, rng: np.random.Generator, *, window: float = 0.2
) -> Optional[tuple[float, Callable]]:
    """Reflected random walk on a reticulation inheritance probability.

    Symmetric proposal -> Hastings ratio 0.
    """
    reticulations = [v for v in state.species_net.V() if v.is_reticulation()]
    if not reticulations:
        return None
    old_net = state.species_net
    old_heights = state.net_heights
    snap = state._engine.clone_caches()

    new_net = _clone_net(old_net)
    new_heights = _heights(new_net)
    rets = [v for v in new_net.V() if v.is_reticulation()]
    r = rets[int(rng.integers(len(rets)))]
    in_edges = list(new_net.in_edges(r))
    if len(in_edges) < 2:
        return None
    e0, e1 = in_edges[0], in_edges[1]
    g = e0.get_gamma()
    g_new = g + window * (rng.random() - 0.5)
    # Reflect into (0, 1).
    while g_new <= 0.0 or g_new >= 1.0:
        if g_new <= 0.0:
            g_new = -g_new
        if g_new >= 1.0:
            g_new = 2.0 - g_new
    e0.set_gamma(float(g_new))
    e1.set_gamma(float(1.0 - g_new))
    _sync_lengths(new_net, new_heights)
    state.species_net = new_net
    state.net_heights = new_heights
    state._engine.invalidate_network()

    def undo() -> None:
        state.species_net = old_net
        state.net_heights = old_heights
        state._engine.restore_caches(snap)

    return 0.0, undo


def _slide_height_move(
    net: Network,
    heights: dict[Node, float],
    rng: np.random.Generator,
    root_scale_window: float,
) -> Optional[float]:
    """Slide one internal node's height in place; return log Hastings ratio.

    Non-root internal nodes get a uniform slide within ``(max child height,
    min parent height)`` (symmetric, HR 0); the root is scaled up/down
    (HR ``log(f)``).  Returns ``None`` if no valid move exists.
    """
    internal = _internal_nodes(net)
    root = net.root()
    candidates = internal + [root]
    if not candidates:
        return None
    node = candidates[int(rng.integers(len(candidates)))]
    children = net.get_children(node)
    lower = max((heights[c] for c in children), default=0.0)

    if node is root or net.in_degree(node) == 0:
        f = math.exp(root_scale_window * (rng.random() - 0.5))
        new_h = heights[node] * f
        if new_h <= lower:
            return None
        heights[node] = new_h
        return math.log(f)

    parents = net.get_parents(node)
    upper = min(heights[p] for p in parents)
    if upper <= lower:
        return None
    heights[node] = lower + (upper - lower) * rng.random()
    return 0.0


def op_net_node_height(
    state: SeqState, rng: np.random.Generator, *, root_scale_window: float = 0.3
) -> Optional[tuple[float, Callable]]:
    """Slide / scale a species-network node height."""
    old_net = state.species_net
    old_heights = state.net_heights
    snap = state._engine.clone_caches()

    new_net = _clone_net(old_net)
    new_heights = _heights(new_net)
    loghr = _slide_height_move(new_net, new_heights, rng, root_scale_window)
    if loghr is None:
        return None
    _sync_lengths(new_net, new_heights)
    state.species_net = new_net
    state.net_heights = new_heights
    state._engine.invalidate_network()

    def undo() -> None:
        state.species_net = old_net
        state.net_heights = old_heights
        state._engine.restore_caches(snap)

    return loghr, undo


def op_gene_node_height(
    state: SeqState, rng: np.random.Generator, *, root_scale_window: float = 0.3
) -> Optional[tuple[float, Callable]]:
    """Slide / scale a gene-tree node height (random locus)."""
    if state._engine.n_loci == 0:
        return None
    i = int(rng.integers(state._engine.n_loci))
    old_gt = state.gene_trees[i]
    old_h = state.gt_heights[i]
    snap = state._engine.clone_caches()

    new_gt = _clone_net(old_gt)
    new_h = _heights(new_gt)
    loghr = _slide_height_move(new_gt, new_h, rng, root_scale_window)
    if loghr is None:
        return None
    _sync_lengths(new_gt, new_h)
    state.gene_trees[i] = new_gt
    state.gt_heights[i] = new_h
    state._engine.invalidate_locus(i)

    def undo() -> None:
        state.gene_trees[i] = old_gt
        state.gt_heights[i] = old_h
        state._engine.restore_caches(snap)

    return loghr, undo


def op_gene_tree_nni(
    state: SeqState, rng: np.random.Generator
) -> Optional[tuple[float, Callable]]:
    """Nearest-neighbour interchange on a gene tree (height-preserving).

    Picks an internal edge ``(p, v)`` (``v`` internal) and swaps the sibling
    subtree ``s`` of ``v`` with one child of ``v``.  Heights are preserved, so
    the proposal is symmetric (Hastings ratio 0); the move is rejected when the
    swap would violate the ultrametric ordering.
    """
    if state._engine.n_loci == 0:
        return None
    i = int(rng.integers(state._engine.n_loci))
    old_gt = state.gene_trees[i]
    old_h = state.gt_heights[i]
    snap = state._engine.clone_caches()

    new_gt = _clone_net(old_gt)
    new_h = _heights(new_gt)

    # Eligible internal edges: (p, v) where v is internal and p has a sibling.
    internal = [v for v in new_gt.V() if new_gt.out_degree(v) >= 2]
    eligible: list[tuple[Node, Node, Node, Node]] = []
    for v in internal:
        parents = new_gt.get_parents(v)
        if not parents:
            continue
        p = parents[0]
        sibs = [c for c in new_gt.get_children(p) if c is not v]
        if not sibs:
            continue
        s = sibs[0]
        v_children = new_gt.get_children(v)
        if len(v_children) < 2:
            continue
        c = v_children[int(rng.integers(len(v_children)))]
        eligible.append((p, v, s, c))
    if not eligible:
        return None
    p, v, s, c = eligible[int(rng.integers(len(eligible)))]

    # Validity: after swapping s under v and c under p, need h[v] > h[s].
    if new_h[v] <= new_h[s]:
        return None

    e_ps = _edge_between(new_gt, p, s)
    e_vc = _edge_between(new_gt, v, c)
    new_gt.remove_edge(e_ps)
    new_gt.remove_edge(e_vc)
    new_gt.add_edges(Edge(p, c))
    new_gt.add_edges(Edge(v, s))
    _sync_lengths(new_gt, new_h)

    state.gene_trees[i] = new_gt
    state.gt_heights[i] = new_h
    state._engine.invalidate_locus(i)

    def undo() -> None:
        state.gene_trees[i] = old_gt
        state.gt_heights[i] = old_h
        state._engine.restore_caches(snap)

    return 0.0, undo


def _split_edge(
    net: Network, heights: dict[Node, float], parent: Node, child: Node, t: float
) -> Node:
    """Insert a new degree-2 node at height ``t`` on edge ``(parent, child)``.

    Returns the new node (added to ``net`` and ``heights``).
    """
    e = _edge_between(net, parent, child)
    net.remove_edge(e)
    mid = net.add_uid_node()
    heights[mid] = t
    net.add_edges(Edge(parent, mid))
    net.add_edges(Edge(mid, child))
    return mid


def _has_parallel_edges(net: Network) -> bool:
    """True if any node has two incoming edges from the *same* parent.

    Two parallel edges between an ordered pair of nodes form a "bubble" (a
    length-2 reticulation cycle).  These are degenerate: they let a gene
    coalescence sit arbitrarily close above a species divergence, so the MSNC
    *density* diverges as ``theta -> 0`` and the sampler runs away to nonsense
    (unbounded log posterior, ``theta`` collapse).  PhyloNet's moves forbid
    these configurations; so do we.
    """
    for v in net.V():
        srcs = [e.src for e in net.in_edges(v)]
        if len(srcs) != len({id(s) for s in srcs}):
            return True
    return False


def _logsumexp(values: Sequence[float]) -> float:
    """Numerically stable ``log(sum(exp(v)))`` (``-inf`` for an empty list)."""
    m = float("-inf")
    for v in values:
        if v > m:
            m = v
    if m == float("-inf"):
        return float("-inf")
    acc = 0.0
    for v in values:
        acc += math.exp(v - m)
    return m + math.log(acc)


def _network_only_log_score(state: "SeqState", net: Network) -> float:
    """Network-dependent part of the log posterior for a *candidate* network.

    Returns ``sum_i log P(g_i | net) + log prior(net)`` using the chain's
    current gene trees (whose per-locus indices are reused from the engine
    cache -- gene trees never change inside a network move).  The Felsenstein
    factor ``sum_i log P(S_i | g_i)`` is deliberately omitted: it does not
    depend on the species network, so it cancels in every ratio that compares
    two candidate networks against the same gene trees.  This is the score the
    informed add/delete proposals rank placements by, and it is exactly the
    quantity whose difference the Metropolis-Hastings driver sees (up to that
    common Felsenstein constant).

    Height-scale marginalisation.  A freshly-added reticulation usually needs
    gene-tree coalescences *lifted* (or lowered) relative to a tree-adapted gene
    tree before the tree can embed at all -- the tree-adapted times are commonly
    a zero-probability configuration under the reticulated network (MSNC density
    ``-inf``).  Ranking placements by the raw current gene trees would therefore
    score every genuinely-useful reticulation ``-inf`` and never propose it.
    Instead each locus is scored by the log-sum-exp of its MSNC density over the
    :data:`_HEIGHT_SCALE_GRID` of global height rescalings -- i.e. the best the
    coupled move's gene-tree re-proposal could achieve on that placement.  This
    is cheap: the coalescence-event *times* are scaled directly (the topology
    and bit-index cache are untouched), so no gene tree is copied or re-indexed.
    Any positive deterministic weight preserves detailed balance; add and delete
    call this identical rule.

    Returns ``-inf`` only if some locus cannot embed at *any* scale (a genuinely
    zero-probability network).
    """
    eng = state._engine
    net_idx, sp_heights = build_network_msnc_index(net)
    if sp_heights is None:
        return float("-inf")
    total = 0.0
    for i in range(eng.n_loci):
        gti, events = eng._gene_index(i, state.gene_trees[i])
        terms: list[float] = []
        for s in _HEIGHT_SCALE_GRID:
            sev = events if s == 1.0 else [
                (t * s, a, b, c) for (t, a, b, c) in events
            ]
            d = msnc_log_density_prebuilt(net_idx, sp_heights, gti, sev, state.theta)
            if math.isfinite(d):
                terms.append(d)
        if not terms:
            return float("-inf")
        total += _logsumexp(terms)
    total += log_prior_seq(net, state.theta, state.priors)
    if not math.isfinite(total):
        return float("-inf")
    return total


def _find_edge_by_labels(
    net: Network, src_label: str, dest_label: str
) -> Optional[Edge]:
    """Locate the edge ``src_label -> dest_label`` in ``net`` (or ``None``).

    Used to re-identify an edge in a fresh deep copy: object identity is not
    preserved across :func:`copy.deepcopy`, but node *labels* are, and in a
    bubble-free network an ordered label pair identifies a unique edge.
    """
    for e in net.E():
        if e.src.label == src_label and e.dest.label == dest_label:
            return e
    return None


def _add_reticulation_placements(
    state: "SeqState", net: Network, heights: dict[Node, float]
) -> list[dict]:
    """Enumerate + score every valid single-reticulation placement on ``net``.

    A *placement* is an ordered edge pair ``(donor, reticulation)``: a new tree
    node ``v1`` is inserted at the midpoint of the feasible window on the donor
    edge and a new reticulation node ``v2`` at the midpoint of the reticulation
    edge, with a hybrid edge ``v1 -> v2`` (``gamma = 0.5``).  Placements that
    would create a cycle or a bubble, or whose donor edge has no portion above
    the reticulation point, are dropped.

    The *representative* midpoint / ``gamma = 0.5`` score ``s_rep`` is used only
    as the proposal weight (any positive weight function preserves detailed
    balance, provided the reverse move uses the identical rule on the identical
    network -- which it does, because the score is a deterministic function of
    ``(net, donor, reticulation)`` and the delete move re-enumerates on the
    exact network the add started from).

    Returns:
        A list of dicts, one per valid placement, with keys ``donor_labels``,
        ``retic_labels`` (ordered ``(src_label, dest_label)`` tuples), ``l2``
        (reticulation-edge length), and ``s_rep`` (representative log score).
    """
    edges = list(net.E())
    placements: list[dict] = []
    for d in edges:
        v3, v4 = d.src, d.dest
        l1 = heights[v3] - heights[v4]
        if l1 <= 0.0:
            continue
        for r in edges:
            if r is d:
                continue
            v5, v6 = r.src, r.dest
            l2 = heights[v5] - heights[v6]
            if l2 <= 0.0:
                continue
            t2 = heights[v6] + 0.5 * l2
            lo = max(heights[v4], t2)
            hi = heights[v3]
            if hi <= lo:
                continue
            t1 = 0.5 * (lo + hi)
            cand = _clone_net(net)
            ch = _heights(cand)
            cd = _find_edge_by_labels(cand, v3.label, v4.label)
            cr = _find_edge_by_labels(cand, v5.label, v6.label)
            if cd is None or cr is None:
                continue
            try:
                cv1 = _split_edge(cand, ch, cd.src, cd.dest, t1)
                cv2 = _split_edge(cand, ch, cr.src, cr.dest, t2)
                in_e = list(cand.in_edges(cv2))
                if len(in_e) != 1:
                    continue
                in_e[0].set_gamma(0.5)
                cv2.set_is_reticulation(True)
                cand.add_edges(Edge(cv1, cv2, gamma=0.5))
            except Exception:
                continue
            if _has_parallel_edges(cand):
                continue
            try:
                if not cand.is_acyclic():
                    continue
            except Exception:
                continue
            _sync_lengths(cand, ch)
            s_rep = _network_only_log_score(state, cand)
            if not math.isfinite(s_rep):
                continue
            placements.append({
                "donor_labels": (v3.label, v4.label),
                "retic_labels": (v5.label, v6.label),
                "l2": l2,
                "s_rep": s_rep,
            })
    return placements


def _geometric_placements(
    net: Network, heights: dict[Node, float]
) -> list[dict]:
    """Enumerate every *geometrically* valid single-reticulation placement.

    Unlike :func:`_add_reticulation_placements` this is a pure function of the
    network topology and node heights -- it does **not** score placements
    against the (mutable) gene trees.  That gene-tree independence is exactly
    what makes it safe to weight relocation proposals: the placement set (and so
    its cardinality) on a given base network is identical no matter which
    gene trees the chain currently holds, so the ``1 / K`` selection density
    cancels between a relocation and its reverse (which enumerate on the *same*
    base network but with different gene trees).

    A placement is an ordered edge pair ``(donor v3->v4, reticulation
    v5->v6)`` for which a hybrid can be inserted: both edges have positive
    length and the donor top ``h[v3]`` lies above ``max(h[v4], h[v6])`` so a
    non-empty height window exists.  Acyclicity/bubble feasibility is *not*
    pre-checked (it would require building each candidate); an unbuildable pick
    is simply declined at draw time, which -- because it inflates ``K`` equally
    in both directions -- leaves reversibility intact.

    Returns dicts with ``donor_labels`` and ``retic_labels`` only.
    """
    edges = list(net.E())
    out: list[dict] = []
    for d in edges:
        v3, v4 = d.src, d.dest
        if heights[v3] - heights[v4] <= 0.0:
            continue
        for r in edges:
            if r is d:
                continue
            v5, v6 = r.src, r.dest
            if heights[v5] - heights[v6] <= 0.0:
                continue
            if heights[v3] <= max(heights[v4], heights[v6]):
                continue
            out.append({
                "donor_labels": (v3.label, v4.label),
                "retic_labels": (v5.label, v6.label),
            })
    return out


def op_add_reticulation(
    state: SeqState, rng: np.random.Generator
) -> Optional[tuple[float, Callable]]:
    r"""Informed, exactly-reversible RJMCMC add-reticulation move.

    Rather than picking two edges blindly -- which almost never lands on a
    likelihood-improving reticulation and so is essentially never accepted --
    this move enumerates every valid ``(donor, reticulation)`` edge pair,
    scores each by the network-only log posterior of a representative midpoint
    placement, and selects a pair with probability proportional to
    ``exp(s_rep)``.  Conditioned on the chosen pair it then draws the actual
    reticulation height ``t2 ~ U`` along the reticulation edge, the donor point
    ``t1 ~ U`` in the feasible window above ``t2``, and ``gamma ~ U(0, 1)``.

    Detailed balance (Green 1995, reversible-jump).  With forward selection
    probability ``P_f = exp(s_rep*) / sum_k exp(s_rep_k)``, uniform reverse
    selection ``P_r = 1 / (2 (R + 1))`` (the delete move picks one of ``R + 1``
    reticulations and one of its two parents), and Jacobian ``l2 * win`` for the
    uniform ``t2`` / ``t1`` draws (``win`` = donor window width, ``gamma`` is a
    unit-scale auxiliary), the log Hastings ratio is

    .. math::

        \log\!\Big(\tfrac{P_r}{P_f}\Big) + \log(l_2 \cdot \mathrm{win})
        = \operatorname{logsumexp}_k s^{\mathrm{rep}}_k - s^{\mathrm{rep}}_*
          - \log(2 (R+1)) + \log l_2 + \log \mathrm{win}.

    This is the exact negative of :func:`op_delete_reticulation`'s ratio, so
    the pair is reversible; and because the acceptance collapses to
    ``exp(logsumexp_k s_rep_k - log(2(R+1)) - (prop - cur))`` the move is
    accepted with high probability whenever *any* placement improves the
    posterior -- the property blind proposals lack.
    """
    R = state.num_reticulations()
    if R >= state.priors.max_reticulations:
        return None
    old_net = state.species_net
    old_heights = state.net_heights

    placements = _add_reticulation_placements(state, old_net, old_heights)
    if not placements:
        return None

    s_reps = [p["s_rep"] for p in placements]
    lse = _logsumexp(s_reps)
    probs = np.asarray([math.exp(s - lse) for s in s_reps], dtype=np.float64)
    probs /= probs.sum()
    k = int(rng.choice(len(placements), p=probs))
    chosen = placements[k]
    s_star = chosen["s_rep"]

    snap = state._engine.clone_caches()
    new_net = _clone_net(old_net)
    new_heights = _heights(new_net)
    d = _find_edge_by_labels(new_net, *chosen["donor_labels"])
    r = _find_edge_by_labels(new_net, *chosen["retic_labels"])
    if d is None or r is None:
        return None
    v3, v4 = d.src, d.dest
    v5, v6 = r.src, r.dest
    l2 = new_heights[v5] - new_heights[v6]
    if l2 <= 0.0:
        return None
    t2 = new_heights[v6] + l2 * rng.random()
    lo = max(new_heights[v4], t2)
    hi = new_heights[v3]
    win = hi - lo
    if win <= 0.0:
        return None
    t1 = lo + win * rng.random()
    g = float(rng.random())
    try:
        v1 = _split_edge(new_net, new_heights, v3, v4, t1)
        v2 = _split_edge(new_net, new_heights, v5, v6, t2)
        in_e = list(new_net.in_edges(v2))
        if len(in_e) != 1:
            return None
        in_e[0].set_gamma(1.0 - g)
        v2.set_is_reticulation(True)
        new_net.add_edges(Edge(v1, v2, gamma=g))
    except Exception:
        return None
    if _has_parallel_edges(new_net):
        return None
    try:
        if not new_net.is_acyclic():
            return None
    except Exception:
        return None

    loghr = (
        lse
        - s_star
        - math.log(2.0 * (R + 1))
        + math.log(l2)
        + math.log(win)
    )

    _sync_lengths(new_net, new_heights)
    state.species_net = new_net
    state.net_heights = new_heights
    state._engine.invalidate_network()

    def undo() -> None:
        state.species_net = old_net
        state.net_heights = old_heights
        state._engine.restore_caches(snap)

    return loghr, undo


def op_delete_reticulation(
    state: SeqState, rng: np.random.Generator
) -> Optional[tuple[float, Callable]]:
    r"""Informed, exactly-reversible RJMCMC delete-reticulation move.

    Picks a reticulation ``v2`` uniformly among the ``R`` reticulations and one
    of its two parents ``v1`` (the donor) uniformly, removes the hybrid edge
    ``v1 -> v2`` and suppresses the two resulting degree-2 nodes -- merging
    ``v3 -> v1 -> v4`` into ``v3 -> v4`` and ``v5 -> v2 -> v6`` into
    ``v5 -> v6`` (exactly the two edges a reverse add would re-split).  The
    proposal is rejected for the degenerate configurations PhyloNet forbids
    (``v1`` not a tree node; the merge would create a bubble; the fully
    degenerate case).

    The log Hastings ratio is the exact negative of the informed
    :func:`op_add_reticulation` ratio evaluated at the reverse-add quantities:

    .. math::

        s^{\mathrm{rep}}_{\mathrm{rev}}
        - \operatorname{logsumexp}_k s^{\mathrm{rep}}_k
        + \log(2 R) - \log l_2 - \log \mathrm{win},

    where the ``s_rep`` are the representative placement scores of the
    *post*-delete network, ``s_rep_rev`` is the score of the placement that
    recreates the removed reticulation, ``l2`` / ``win`` are the reticulation
    edge length and donor window recovered from the pre-delete network, and
    ``R`` is the pre-move reticulation count.  Because add and delete share the
    identical scoring rule on the identical network, ``logsumexp_k s_rep_k``
    and ``s_rep_rev`` cancel exactly against the matching add, guaranteeing
    reversibility.
    """
    R = state.num_reticulations()
    if R == 0:
        return None
    old_net = state.species_net
    old_heights = state.net_heights

    rets = [v for v in old_net.V() if v.is_reticulation()]
    v2 = rets[int(rng.integers(len(rets)))]
    parents = old_net.get_parents(v2)
    if len(parents) != 2:
        return None
    v1 = parents[int(rng.integers(2))]
    v5 = parents[1] if parents[0] is v1 else parents[0]

    if v1.is_reticulation() or old_net.in_degree(v1) == 0:
        return None
    v3s = old_net.get_parents(v1)
    if len(v3s) != 1:
        return None
    v3 = v3s[0]
    other_children = [c for c in old_net.get_children(v1) if c is not v2]
    if len(other_children) != 1:
        return None
    v4 = other_children[0]
    v6s = old_net.get_children(v2)
    if len(v6s) != 1:
        return None
    v6 = v6s[0]

    if v3 in old_net.get_parents(v4):
        return None
    if v5 in old_net.get_parents(v6):
        return None
    if v3 is v5 and v4 is v6:
        return None

    # Reverse-add quantities, recovered from the pre-delete network: the merged
    # reticulation edge is v5 -> v6 (length l2), and the donor window is the
    # portion of the merged donor edge v3 -> v4 above the reticulation height.
    l2 = old_heights[v5] - old_heights[v6]
    if l2 <= 0.0:
        return None
    t2 = old_heights[v2]
    lo = max(old_heights[v4], t2)
    hi = old_heights[v3]
    win = hi - lo
    if win <= 0.0:
        return None
    rev_donor = (v3.label, v4.label)
    rev_retic = (v5.label, v6.label)

    snap = state._engine.clone_caches()
    new_net = _clone_net(old_net)
    new_heights = _heights(new_net)
    cv2 = None
    for v in new_net.V():
        if v.is_reticulation() and v.label == v2.label:
            cv2 = v
            break
    if cv2 is None:
        return None
    cparents = new_net.get_parents(cv2)
    cv1 = next((p for p in cparents if p.label == v1.label), None)
    if cv1 is None:
        return None

    try:
        del_edge = _edge_between(new_net, cv1, cv2)
        new_net.remove_edge(del_edge)
        cv2.set_is_reticulation(False)
        _suppress_degree2(new_net, new_heights, cv1)
        _suppress_degree2(new_net, new_heights, cv2)
    except Exception:
        return None

    E_after = len([e for e in new_net.E()])
    if E_after < 2:
        return None
    _sync_lengths(new_net, new_heights)

    # Reverse-add normaliser: enumerate placements on the post-delete network
    # (the exact network the reverse add would start from) with the identical
    # rule the add uses, and find the placement that recreates this reticulation.
    placements = _add_reticulation_placements(state, new_net, new_heights)
    if not placements:
        return None
    s_reps = [p["s_rep"] for p in placements]
    lse = _logsumexp(s_reps)
    s_rev = None
    for p in placements:
        if p["donor_labels"] == rev_donor and p["retic_labels"] == rev_retic:
            s_rev = p["s_rep"]
            break
    if s_rev is None:
        # The reverse add cannot regenerate this exact placement (e.g. the
        # merged donor edge dropped out of the feasible set); the move is not
        # reversible from here, so decline it.
        return None

    loghr = (
        s_rev
        - lse
        + math.log(2.0 * R)
        - math.log(l2)
        - math.log(win)
    )

    state.species_net = new_net
    state.net_heights = new_heights
    state._engine.invalidate_network()

    def undo() -> None:
        state.species_net = old_net
        state.net_heights = old_heights
        state._engine.restore_caches(snap)

    return loghr, undo


def _suppress_degree2(
    net: Network, heights: dict[Node, float], node: Node
) -> int:
    """Remove a degree-2 (1 in, 1 out) node, merging its two incident edges.

    Has no effect if ``node`` is not degree-2.  Returns the resulting edge
    count.
    """
    if net.in_degree(node) != 1 or net.out_degree(node) != 1:
        return len([e for e in net.E()])
    in_e = list(net.in_edges(node))[0]
    out_e = list(net.out_edges(node))[0]
    parent = in_e.src
    child = out_e.dest
    # The inheritance probability belongs to the edge entering ``child``; if
    # ``child`` is a reticulation that gamma must survive the merge (using the
    # *outgoing* edge's gamma, not the incoming tree edge's).
    gamma = out_e.get_gamma()
    net.remove_edge(in_e)
    net.remove_edge(out_e)
    net.remove_nodes(node)
    heights.pop(node, None)
    merged = Edge(parent, child)
    if child.is_reticulation():
        merged.set_gamma(gamma)
    net.add_edges(merged)
    return len([e for e in net.E()])


# ======================================================================
# Coupled network + gene-tree reticulation moves
# ======================================================================
#
# The informed add/delete above change ONLY the species network, holding the
# per-locus gene trees fixed.  On multilocus *sequence* data that hits a joint
# mode barrier: at a tree state the gene trees have already co-adapted to the
# tree (the MSNC prior squeezes out discordant signal), so bolting a
# reticulation onto those gene trees pays a network-prior cost with no
# coalescent gain -- even the best placement lowers the posterior at that
# instant -- and the move is essentially never accepted.  The reticulated
# network is the higher-posterior mode, yet it is unreachable one coordinate at
# a time.
#
# The COUPLED moves below cross the barrier by proposing the network change and
# a *guided* per-locus gene-tree re-proposal together, as one reversible-jump
# Metropolis-Hastings step.  When a reticulation is added, each locus's gene
# tree is re-drawn from a small candidate set (itself + its height-preserving
# NNI neighbours) with probability proportional to the joint per-locus score
# ``exp(logP(S_i|g) + logP(g|Psi',theta))`` under the NEW network, so loci
# whose sequences prefer a topology the reticulation now explains flip to it in
# the same step -- landing the chain directly in the reticulated mode.
#
# Rigour.  This is a bona-fide Green (1995) reversible-jump proposal; no part
# of the target is approximated or dropped.  Because the NNI candidates are
# height-preserving, the gene-tree re-proposal is a purely *discrete* choice
# (no continuous auxiliary, no extra Jacobian): the only Jacobian is the
# subdivide/merge ``l2 * win`` of the reticulation's continuous parameters,
# exactly as in the plain informed move.  The Hastings ratio is the honest
# ``log q_reverse - log q_forward + log|Jacobian|``.  The reverse gene-tree
# density is evaluated on the candidate set built around the *proposed* gene
# tree under the *reverse* network, which always contains the original because
# the NNI-neighbour relation is symmetric.  ``tests/test_mcmc_seq_coupled.py``
# checks the exact identity ``logH_add(x->x') == -logH_del(x'->x)``.


def _leaf_descendants(net: Network, node: Node) -> frozenset:
    """Frozenset of leaf labels reachable below ``node`` (small trees only)."""
    kids = net.get_children(node)
    if not kids:
        return frozenset({node.label})
    acc: set = set()
    for c in kids:
        acc |= _leaf_descendants(net, c)
    return frozenset(acc)


def _all_leaf_descendants(net: Network) -> dict[Node, frozenset]:
    """Descendant-leaf-set of *every* node in one memoised post-order pass.

    Equivalent to calling :func:`_leaf_descendants` on each node but O(n)
    instead of O(n^2): the per-node call recomputes each subtree from scratch,
    whereas this shares subtree results.  Used by :func:`_gt_signature`, which
    is on the coupled-move hot path (evaluated for every candidate gene tree).
    """
    memo: dict[Node, frozenset] = {}

    def rec(v: Node) -> frozenset:
        cached = memo.get(v)
        if cached is not None:
            return cached
        kids = net.get_children(v)
        if not kids:
            r = frozenset({v.label})
        else:
            acc: set = set()
            for c in kids:
                acc |= rec(c)
            r = frozenset(acc)
        memo[v] = r
        return r

    for v in net.V():
        rec(v)
    return memo


def _gt_signature(gt: Network, heights: dict[Node, float]) -> frozenset:
    """Topology + time signature of a rooted binary gene tree.

    Each internal node contributes ``(descendant-leaf-set, rounded height)``.
    Two gene trees with equal signatures are identical for both the Felsenstein
    and the timed-MSNC likelihood, so this is the correct key for de-duplicating
    the candidate set and for matching a gene tree across a ``deepcopy``
    boundary (object identity is not preserved by copying, but the signature
    is).
    """
    desc = _all_leaf_descendants(gt)
    return frozenset(
        (desc[v], round(float(heights.get(v, 0.0)), 12))
        for v in gt.V()
        if gt.get_children(v)
    )


def _node_by_leafset(net: Network, target: frozenset) -> Optional[Node]:
    """The unique node whose descendant-leaf-set equals ``target`` (or None)."""
    for v in net.V():
        if _leaf_descendants(net, v) == target:
            return v
    return None


def _gene_tree_nni_neighbors(
    gt: Network, heights: dict[Node, float]
) -> list[tuple[Network, dict[Node, float]]]:
    """All valid height-preserving single-NNI rearrangements of ``gt``.

    For every internal node ``v`` (with parent ``p`` and sibling ``s``) and
    every child ``c`` of ``v``, swapping ``s`` with ``c`` gives one neighbour,
    provided it keeps the tree ultrametric (``h[v] > h[s]``; ``h[p] > h[c]`` is
    automatic).  Node heights are preserved, so the neighbour relation is
    symmetric -- the property the coupled move needs so the reverse candidate
    set always contains the original gene tree.

    Returns ``(neighbour_gt, neighbour_heights)`` deep copies with edge lengths
    already synced; duplicates (by :func:`_gt_signature`) are dropped.
    """
    quads: list[tuple[frozenset, frozenset]] = []
    for v in gt.V():
        kids = gt.get_children(v)
        if len(kids) < 2:
            continue
        parents = gt.get_parents(v)
        if not parents:
            continue
        p = parents[0]
        sibs = [c for c in gt.get_children(p) if c is not v]
        if len(sibs) != 1:
            continue
        s = sibs[0]
        if heights[v] <= heights[s]:
            continue
        v_ls = _leaf_descendants(gt, v)
        s_ls = _leaf_descendants(gt, s)
        for c in kids:
            quads.append((s_ls, _leaf_descendants(gt, c), v_ls))

    neighbors: list[tuple[Network, dict]] = []
    seen: set = set()
    for s_ls, c_ls, _v_ls in quads:
        cand = _clone_net(gt)
        ch = _heights(cand)
        cs = _node_by_leafset(cand, s_ls)
        cc = _node_by_leafset(cand, c_ls)
        if cs is None or cc is None:
            continue
        cp_list = cand.get_parents(cs)
        cv_list = cand.get_parents(cc)
        if not cp_list or not cv_list:
            continue
        cp, cv = cp_list[0], cv_list[0]
        if cp is cv:
            continue
        try:
            cand.remove_edge(_edge_between(cand, cp, cs))
            cand.remove_edge(_edge_between(cand, cv, cc))
            cand.add_edges(Edge(cp, cc))
            cand.add_edges(Edge(cv, cs))
        except Exception:
            continue
        _sync_lengths(cand, ch)
        sig = _gt_signature(cand, ch)
        if sig in seen:
            continue
        seen.add(sig)
        neighbors.append((cand, ch))
    return neighbors


# Symmetric-under-reciprocal grid of gene-tree height-scale factors.  Because
# the grid is closed under ``s -> 1/s`` (and contains 1.0), scaling composes
# reversibly with the height-preserving NNI neighbourhood: the reverse candidate
# set around a chosen tree always contains the original.  Scaling every internal
# node height by a common factor keeps the tree ultrametric and only *shifts*
# coalescent times -- exactly the degree of freedom the plain NNI neighbourhood
# lacks, and the one that lets a tree-adapted gene tree become embeddable in a
# freshly-added reticulation (whose deeper structure needs coalescences lifted
# above the new hybrid height).
#
# CRITICAL -- the factors MUST be exact powers of two.  Reversibility of the
# coupled move requires the reverse candidate set (built around the chosen,
# rescaled tree and re-scaled by ``1/s``) to reproduce the *original* heights
# **bit-for-bit**, because :func:`_gt_signature` matches on ``round(height, 12)``.
# Only powers of two satisfy both ``1/s`` being exactly representable *and*
# ``s * (1/s) == 1.0`` exactly in IEEE-754.  A previous grid used +/- sqrt(2)
# steps (0.707106781 / 1.414213562); those are *not* exact reciprocals
# (0.707106781 * 1.414213562 = 0.999999999...), so any move that selected a
# sqrt(2) scale round-tripped to a height differing in the ~11th decimal, missed
# the signature match, and was silently declined.  Under a freshly-added
# reticulation the MSNC density *favours* the rescaled candidates, so add-
# reticulation picked a sqrt(2) scale almost every time and was therefore
# declined almost every time -- the chain could never leave a tree.  Powers of
# two fix this exactly (at the cost of a coarser factor-2 rather than factor-
# sqrt(2) step).
_HEIGHT_SCALE_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)


def _scaled_gene_tree(
    gt: Network, s: float
) -> tuple[Network, dict[Node, float]]:
    """Deep copy of ``gt`` with every internal node height multiplied by ``s``.

    Leaves stay at height 0, so the result is a valid ultrametric tree with the
    same topology and rescaled coalescent times.  ``s == 1.0`` is an identity
    (still a fresh copy so the caller owns an independent object).
    """
    cand = _clone_net(gt)
    ch = _heights(cand)
    if s != 1.0:
        ch = {v: h * s for v, h in ch.items()}
        _sync_lengths(cand, ch)
    return cand, ch


def _candidate_set(
    gt: Network, heights: dict[Node, float]
) -> list[tuple[Network, dict[Node, float]]]:
    """Discrete re-proposal support: NNI neighbourhood x height-scale grid.

    Starts from ``gt`` and its height-preserving NNI neighbours, then emits a
    height-rescaled copy of each at every factor in :data:`_HEIGHT_SCALE_GRID`.
    De-duplicated by :func:`_gt_signature`.  The identity (``gt`` itself at
    scale 1.0) is always present.  This is the finite set the coupled move
    re-draws a locus's gene tree from; because the set is reverse-reachable
    (NNI symmetry + reciprocal scale grid), the guided selection stays exactly
    reversible.
    """
    bases: list[tuple[Network, dict]] = [(gt, heights)]
    bases.extend(_gene_tree_nni_neighbors(gt, heights))
    out: list[tuple[Network, dict]] = []
    seen: set = set()
    for b_gt, _b_h in bases:
        for s in _HEIGHT_SCALE_GRID:
            cg, ch = _scaled_gene_tree(b_gt, s)
            sig = _gt_signature(cg, ch)
            if sig in seen:
                continue
            seen.add(sig)
            out.append((cg, ch))
    return out


def _candidate_logweights(
    state: "SeqState",
    i: int,
    net_idx,
    sp_heights: Optional[list[float]],
    theta: float,
    candidates: list[tuple[Network, dict[Node, float]]],
) -> np.ndarray:
    """Joint per-locus log score of each candidate under one network.

    ``logw[j] = log P(S_i | g_j) + log P(g_j | net, theta)`` -- Felsenstein
    phylogenetic likelihood plus timed MSNC density.  Candidates that do not
    embed (``-inf`` MSNC) or have non-finite Felsenstein get ``-inf`` weight,
    so they are never proposed and contribute zero reverse density.
    """
    eng = state._engine
    logw = np.full(len(candidates), -np.inf, dtype=np.float64)
    for j, (gt, _h) in enumerate(candidates):
        fels = eng._calcs[i].log_likelihood(gt, state.model)
        if not math.isfinite(fels):
            continue
        gti, events = build_gene_tree_msnc_index(gt, eng._species_of)
        msnc = msnc_log_density_prebuilt(net_idx, sp_heights, gti, events, theta)
        if not math.isfinite(msnc):
            continue
        logw[j] = fels + msnc
    return logw


def _enumerate_scored_candidates(
    state: "SeqState",
    i: int,
    net_idx,
    sp_heights: Optional[list[float]],
    theta: float,
    gt: Network,
    heights: dict[Node, float],
) -> tuple[list[frozenset], np.ndarray, Callable[[int], tuple[Network, dict]]]:
    """Enumerate + score the candidate set of ``gt`` without per-scale clones.

    The candidate set is ``({gt} + NNI(gt)) x`` :data:`_HEIGHT_SCALE_GRID`,
    de-duplicated by :func:`_gt_signature` -- identical to what
    :func:`_candidate_set` + :func:`_candidate_logweights` produce (same order,
    signatures and weights, so the guided selection is unchanged and remains
    exactly reversible).  The speed-up: only the distinct *base topologies*
    (the identity ``gt`` plus its NNI neighbours) are ever cloned; each height
    scale is scored by transiently rewriting the base's edge lengths and
    restoring them.  This is exact because the Felsenstein and MSNC factors read
    edge lengths / heights fresh on every call (no per-object cache to poison).

    Returns ``(signatures, logweights, rebuild)`` where ``rebuild(k)`` returns an
    independent ``(network, heights)`` for candidate ``k`` -- used to materialise
    the chosen candidate.
    """
    eng = state._engine
    # Identity base uses ``gt`` directly (transiently mutated + restored);
    # NNI neighbours are already independent clones.
    bases: list[tuple[Network, dict]] = [(gt, heights)]
    bases.extend(_gene_tree_nni_neighbors(gt, heights))

    sigs: list[frozenset] = []
    logw: list[float] = []
    descriptors: list[tuple[int, float]] = []
    seen: set = set()
    for bi, (b_gt, b_h) in enumerate(bases):
        # Snapshot base edge lengths so every scale restores exactly.
        base_lengths = [(e, e.get_length()) for e in b_gt.E()]
        # The gene-tree topology index (bit map, parent/children, and the
        # expensive coarsening cache) is scale-invariant -- rescaling all
        # heights preserves coalescence order -- so build it once per base
        # topology and reuse across every height scale, recomputing only the
        # timed events. This lets the coarsening cache carry over the scales
        # instead of being thrown away and rebuilt for each one.
        gti = None
        for s in _HEIGHT_SCALE_GRID:
            if s != 1.0:
                scaled_h = {v: h * s for v, h in b_h.items()}
                _sync_lengths(b_gt, scaled_h)
            else:
                scaled_h = b_h
            sig = _gt_signature(b_gt, scaled_h)
            if sig in seen:
                if s != 1.0:
                    for e, L in base_lengths:
                        e.set_length(float(L) if L is not None else 0.0)
                continue
            fels = eng._calcs[i].log_likelihood(b_gt, state.model)
            w = -np.inf
            if math.isfinite(fels):
                if gti is None:
                    gti = build_gene_tree_topology_index(b_gt, eng._species_of)
                events = gene_tree_events(b_gt, gti, scaled_h)
                msnc = msnc_log_density_prebuilt(
                    net_idx, sp_heights, gti, events, theta
                )
                if math.isfinite(msnc):
                    w = fels + msnc
            if s != 1.0:
                for e, L in base_lengths:
                    e.set_length(float(L) if L is not None else 0.0)
            seen.add(sig)
            sigs.append(sig)
            logw.append(w)
            descriptors.append((bi, s))

    def rebuild(k: int) -> tuple[Network, dict]:
        bi, s = descriptors[k]
        b_gt, _b_h = bases[bi]
        cg = _clone_net(b_gt)
        ch = _heights(cg)
        if s != 1.0:
            ch = {v: h * s for v, h in ch.items()}
        _sync_lengths(cg, ch)
        return cg, ch

    return sigs, np.asarray(logw, dtype=np.float64), rebuild


def _coupled_gene_tree_reproposal(
    state: "SeqState",
    target_net: Network,
    reverse_net: Network,
    rng: np.random.Generator,
) -> Optional[tuple[list[Network], list[dict], float, float]]:
    """Guided joint re-proposal of every locus's gene tree.

    For each locus, draw a new gene tree from ``{g_i} + NNI(g_i)`` with
    probability proportional to the joint per-locus score under ``target_net``
    (the network the move proposes), accumulating the forward selection
    log-density ``log q_f``.  Simultaneously accumulate the reverse selection
    log-density ``log q_r``: the probability that the identical rule, applied on
    the candidate set built around the *chosen* gene tree under ``reverse_net``
    (the network the reverse move starts from), re-selects the original gene
    tree.  The original always lies in that reverse set because NNI
    neighbourliness is symmetric.

    Returns ``(new_gene_trees, new_heights, log_qf, log_qr)`` or ``None`` when a
    locus's original gene tree cannot be matched in the reverse candidate set
    (a non-reversible corner -- decline the move) or when a candidate set has no
    finite-weight member.
    """
    eng = state._engine
    tgt_idx, tgt_h = build_network_msnc_index(target_net)
    rev_idx, rev_h = build_network_msnc_index(reverse_net)
    if tgt_h is None or rev_h is None:
        return None

    new_gts: list[Network] = []
    new_hs: list[dict] = []
    log_qf = 0.0
    log_qr = 0.0
    for i in range(eng.n_loci):
        g_i = state.gene_trees[i]
        h_i = state.gt_heights[i]
        target_sig = _gt_signature(g_i, h_i)

        fwd_sigs, fwd_lw, fwd_rebuild = _enumerate_scored_candidates(
            state, i, tgt_idx, tgt_h, state.theta, g_i, h_i
        )
        fwd_lse = _logsumexp([float(x) for x in fwd_lw])
        if not math.isfinite(fwd_lse):
            return None
        probs = np.exp(fwd_lw - fwd_lse)
        total = float(probs.sum())
        if not math.isfinite(total) or total <= 0.0:
            return None
        probs = probs / total
        j = int(rng.choice(len(fwd_sigs), p=probs))
        log_qf += float(fwd_lw[j]) - fwd_lse

        # Materialise an independent copy of the chosen candidate.
        chosen_gt, chosen_h = fwd_rebuild(j)

        # Reverse density: candidate set around the chosen tree, scored under
        # the reverse network; probability of re-selecting the original.
        rev_sigs, rev_lw, _rev_rebuild = _enumerate_scored_candidates(
            state, i, rev_idx, rev_h, state.theta, chosen_gt, chosen_h
        )
        rev_lse = _logsumexp([float(x) for x in rev_lw])
        if not math.isfinite(rev_lse):
            return None
        match = -1
        for jj, sig in enumerate(rev_sigs):
            if sig == target_sig:
                match = jj
                break
        if match < 0 or not math.isfinite(float(rev_lw[match])):
            return None
        log_qr += float(rev_lw[match]) - rev_lse

        new_gts.append(chosen_gt)
        new_hs.append(chosen_h)

    return new_gts, new_hs, log_qf, log_qr


def op_add_reticulation_coupled(
    state: SeqState, rng: np.random.Generator
) -> Optional[tuple[float, Callable]]:
    r"""Coupled (network + gene-tree) informed add-reticulation RJMCMC move.

    Extends :func:`op_add_reticulation` with a guided per-locus gene-tree
    re-proposal (see the section header above).  The log Hastings ratio is the
    plain informed-add ratio plus the gene-tree coupling term
    ``sum_i (log P_reverse_i - log P_forward_i)``:

    .. math::

        \log H = \operatorname{logsumexp}_k s^{\mathrm{rep}}_k - s^{\mathrm{rep}}_*
                 - \log(2(R+1)) + \log l_2 + \log \mathrm{win}
                 + \sum_i \big(\log P^{\mathrm r}_i - \log P^{\mathrm f}_i\big),

    which is the exact negative of :func:`op_delete_reticulation_coupled`'s
    ratio on the matching transition.
    """
    R = state.num_reticulations()
    if R >= state.priors.max_reticulations:
        return None
    old_net = state.species_net
    old_heights = state.net_heights
    old_gts = state.gene_trees
    old_gt_h = state.gt_heights

    placements = _add_reticulation_placements(state, old_net, old_heights)
    if not placements:
        return None
    s_reps = [p["s_rep"] for p in placements]
    lse = _logsumexp(s_reps)
    probs = np.asarray([math.exp(s - lse) for s in s_reps], dtype=np.float64)
    probs /= probs.sum()
    k = int(rng.choice(len(placements), p=probs))
    chosen = placements[k]
    s_star = chosen["s_rep"]

    snap = state._engine.clone_caches()
    new_net = _clone_net(old_net)
    new_heights = _heights(new_net)
    d = _find_edge_by_labels(new_net, *chosen["donor_labels"])
    r = _find_edge_by_labels(new_net, *chosen["retic_labels"])
    if d is None or r is None:
        return None
    v3, v4 = d.src, d.dest
    v5, v6 = r.src, r.dest
    l2 = new_heights[v5] - new_heights[v6]
    if l2 <= 0.0:
        return None
    t2 = new_heights[v6] + l2 * rng.random()
    lo = max(new_heights[v4], t2)
    hi = new_heights[v3]
    win = hi - lo
    if win <= 0.0:
        return None
    t1 = lo + win * rng.random()
    g = float(rng.random())
    try:
        v1 = _split_edge(new_net, new_heights, v3, v4, t1)
        v2 = _split_edge(new_net, new_heights, v5, v6, t2)
        in_e = list(new_net.in_edges(v2))
        if len(in_e) != 1:
            return None
        in_e[0].set_gamma(1.0 - g)
        v2.set_is_reticulation(True)
        new_net.add_edges(Edge(v1, v2, gamma=g))
    except Exception:
        return None
    if _has_parallel_edges(new_net):
        return None
    try:
        if not new_net.is_acyclic():
            return None
    except Exception:
        return None
    # Level cap (e.g. level-1 mode): reject a proposal whose new hybrid
    # lands in an already-occupied blob *before* paying for the coupled
    # gene-tree re-proposal and MSNC scoring.  This is the state-space
    # restriction that keeps the displayed-tree count (and thus the
    # ancestral-configuration DP) from blowing up.  Consistent with the
    # other early ``return None`` guards above (parallel edges, cycles),
    # a pruned proposal counts as a rejection.
    if (
        state.priors.max_level is not None
        and _network_level(new_net) > state.priors.max_level
    ):
        return None
    _sync_lengths(new_net, new_heights)

    repro = _coupled_gene_tree_reproposal(state, new_net, old_net, rng)
    if repro is None:
        return None
    new_gts, new_hs, log_qf_gt, log_qr_gt = repro

    loghr = (
        lse
        - s_star
        - math.log(2.0 * (R + 1))
        + math.log(l2)
        + math.log(win)
        + (log_qr_gt - log_qf_gt)
    )

    state.species_net = new_net
    state.net_heights = new_heights
    state.gene_trees = new_gts
    state.gt_heights = new_hs
    state._engine.invalidate_network()
    for i in range(state._engine.n_loci):
        state._engine.invalidate_locus(i)

    def undo() -> None:
        state.species_net = old_net
        state.net_heights = old_heights
        state.gene_trees = old_gts
        state.gt_heights = old_gt_h
        state._engine.restore_caches(snap)

    return loghr, undo


def op_delete_reticulation_coupled(
    state: SeqState, rng: np.random.Generator
) -> Optional[tuple[float, Callable]]:
    r"""Coupled (network + gene-tree) informed delete-reticulation RJMCMC move.

    Exact reverse of :func:`op_add_reticulation_coupled`.  Removes a uniformly
    chosen reticulation + parent, re-proposes every locus's gene tree guided by
    the *post-delete* network, and its log Hastings ratio is the negative of the
    coupled add's on the matching transition:

    .. math::

        \log H = s^{\mathrm{rep}}_{\mathrm{rev}}
                 - \operatorname{logsumexp}_k s^{\mathrm{rep}}_k
                 + \log(2 R) - \log l_2 - \log \mathrm{win}
                 + \sum_i \big(\log P^{\mathrm r}_i - \log P^{\mathrm f}_i\big).
    """
    R = state.num_reticulations()
    if R == 0:
        return None
    old_net = state.species_net
    old_heights = state.net_heights
    old_gts = state.gene_trees
    old_gt_h = state.gt_heights

    rets = [v for v in old_net.V() if v.is_reticulation()]
    v2 = rets[int(rng.integers(len(rets)))]
    parents = old_net.get_parents(v2)
    if len(parents) != 2:
        return None
    v1 = parents[int(rng.integers(2))]
    v5 = parents[1] if parents[0] is v1 else parents[0]

    if v1.is_reticulation() or old_net.in_degree(v1) == 0:
        return None
    v3s = old_net.get_parents(v1)
    if len(v3s) != 1:
        return None
    v3 = v3s[0]
    other_children = [c for c in old_net.get_children(v1) if c is not v2]
    if len(other_children) != 1:
        return None
    v4 = other_children[0]
    v6s = old_net.get_children(v2)
    if len(v6s) != 1:
        return None
    v6 = v6s[0]

    if v3 in old_net.get_parents(v4):
        return None
    if v5 in old_net.get_parents(v6):
        return None
    if v3 is v5 and v4 is v6:
        return None

    l2 = old_heights[v5] - old_heights[v6]
    if l2 <= 0.0:
        return None
    t2 = old_heights[v2]
    lo = max(old_heights[v4], t2)
    hi = old_heights[v3]
    win = hi - lo
    if win <= 0.0:
        return None
    rev_donor = (v3.label, v4.label)
    rev_retic = (v5.label, v6.label)

    snap = state._engine.clone_caches()
    new_net = _clone_net(old_net)
    new_heights = _heights(new_net)
    cv2 = None
    for v in new_net.V():
        if v.is_reticulation() and v.label == v2.label:
            cv2 = v
            break
    if cv2 is None:
        return None
    cparents = new_net.get_parents(cv2)
    cv1 = next((p for p in cparents if p.label == v1.label), None)
    if cv1 is None:
        return None
    try:
        del_edge = _edge_between(new_net, cv1, cv2)
        new_net.remove_edge(del_edge)
        cv2.set_is_reticulation(False)
        _suppress_degree2(new_net, new_heights, cv1)
        _suppress_degree2(new_net, new_heights, cv2)
    except Exception:
        return None
    if len([e for e in new_net.E()]) < 2:
        return None
    _sync_lengths(new_net, new_heights)

    # Coupled gene-tree re-proposal, guided by the post-delete network.  Called
    # while ``state.gene_trees`` is still the pre-delete set so the forward
    # density samples the new trees under ``new_net`` and the reverse density
    # rescores the old trees under ``old_net`` (the reverse add's network).
    repro = _coupled_gene_tree_reproposal(state, new_net, old_net, rng)
    if repro is None:
        return None
    new_gts, new_hs, log_qf_gt, log_qr_gt = repro

    # The reverse add enumerates placements on the post-delete network using the
    # *post-delete* gene trees, so install them before scoring.
    state.gene_trees = new_gts
    state.gt_heights = new_hs
    placements = _add_reticulation_placements(state, new_net, new_heights)
    if not placements:
        state.gene_trees = old_gts
        state.gt_heights = old_gt_h
        return None
    s_reps = [p["s_rep"] for p in placements]
    lse = _logsumexp(s_reps)
    s_rev = None
    for p in placements:
        if p["donor_labels"] == rev_donor and p["retic_labels"] == rev_retic:
            s_rev = p["s_rep"]
            break
    if s_rev is None:
        state.gene_trees = old_gts
        state.gt_heights = old_gt_h
        return None

    loghr = (
        s_rev
        - lse
        + math.log(2.0 * R)
        - math.log(l2)
        - math.log(win)
        + (log_qr_gt - log_qf_gt)
    )

    state.species_net = new_net
    state.net_heights = new_heights
    state._engine.invalidate_network()
    for i in range(state._engine.n_loci):
        state._engine.invalidate_locus(i)

    def undo() -> None:
        state.species_net = old_net
        state.net_heights = old_heights
        state.gene_trees = old_gts
        state.gt_heights = old_gt_h
        state._engine.restore_caches(snap)

    return loghr, undo


def op_add_reticulation_decoupled(
    state: SeqState, rng: np.random.Generator
) -> Optional[tuple[float, Callable]]:
    r"""Network-only informed add-reticulation move (PhyloNet-style).

    Identical placement machinery to :func:`op_add_reticulation_coupled` but it
    changes **only the species network** -- the per-locus gene trees are left
    untouched and are re-scored (MSNC) under the new network in place.  There is
    therefore no gene-tree coupling term; the log Hastings ratio is just the
    informed-add geometry

    .. math::

        \log H = \operatorname{logsumexp}_k s^{\mathrm{rep}}_k - s^{\mathrm{rep}}_*
                 - \log(2(R+1)) + \log l_2 + \log \mathrm{win},

    the exact negative of :func:`op_delete_reticulation_decoupled`.

    Rationale: the coupled move re-proposes every gene tree *at the instant* the
    dimension changes.  Once the gene trees have co-adapted to the current
    dimension that all-at-once re-proposal is strongly disfavoured in the reverse
    direction, so the chain sticks in whichever reticulation count it started in
    (severe trans-dimensional hysteresis).  Decoupling lets the network jump on
    its own and the gene trees migrate gradually via the ordinary per-locus
    NNI / height moves -- the mechanism PhyloNet's MCMC_SEQ relies on.
    """
    R = state.num_reticulations()
    if R >= state.priors.max_reticulations:
        return None
    old_net = state.species_net
    old_heights = state.net_heights

    placements = _add_reticulation_placements(state, old_net, old_heights)
    if not placements:
        return None
    s_reps = [p["s_rep"] for p in placements]
    lse = _logsumexp(s_reps)
    probs = np.asarray([math.exp(s - lse) for s in s_reps], dtype=np.float64)
    probs /= probs.sum()
    k = int(rng.choice(len(placements), p=probs))
    chosen = placements[k]
    s_star = chosen["s_rep"]

    snap = state._engine.clone_caches()
    new_net = _clone_net(old_net)
    new_heights = _heights(new_net)
    d = _find_edge_by_labels(new_net, *chosen["donor_labels"])
    r = _find_edge_by_labels(new_net, *chosen["retic_labels"])
    if d is None or r is None:
        return None
    v3, v4 = d.src, d.dest
    v5, v6 = r.src, r.dest
    l2 = new_heights[v5] - new_heights[v6]
    if l2 <= 0.0:
        return None
    t2 = new_heights[v6] + l2 * rng.random()
    lo = max(new_heights[v4], t2)
    hi = new_heights[v3]
    win = hi - lo
    if win <= 0.0:
        return None
    t1 = lo + win * rng.random()
    g = float(rng.random())
    try:
        v1 = _split_edge(new_net, new_heights, v3, v4, t1)
        v2 = _split_edge(new_net, new_heights, v5, v6, t2)
        in_e = list(new_net.in_edges(v2))
        if len(in_e) != 1:
            return None
        in_e[0].set_gamma(1.0 - g)
        v2.set_is_reticulation(True)
        new_net.add_edges(Edge(v1, v2, gamma=g))
    except Exception:
        return None
    if _has_parallel_edges(new_net):
        return None
    try:
        if not new_net.is_acyclic():
            return None
    except Exception:
        return None
    if (
        state.priors.max_level is not None
        and _network_level(new_net) > state.priors.max_level
    ):
        return None
    _sync_lengths(new_net, new_heights)

    loghr = (
        lse
        - s_star
        - math.log(2.0 * (R + 1))
        + math.log(l2)
        + math.log(win)
    )

    state.species_net = new_net
    state.net_heights = new_heights
    # Gene trees unchanged: only the network-dependent MSNC factors are stale.
    state._engine.invalidate_network()

    def undo() -> None:
        state.species_net = old_net
        state.net_heights = old_heights
        state._engine.restore_caches(snap)

    return loghr, undo


def op_delete_reticulation_decoupled(
    state: SeqState, rng: np.random.Generator
) -> Optional[tuple[float, Callable]]:
    r"""Network-only informed delete-reticulation move (PhyloNet-style).

    Exact reverse of :func:`op_add_reticulation_decoupled`: removes a uniformly
    chosen reticulation + parent, leaves every gene tree in place, and re-scores
    them under the post-delete network.  Its log Hastings ratio is the negative
    of the decoupled add's on the matching transition:

    .. math::

        \log H = s^{\mathrm{rep}}_{\mathrm{rev}}
                 - \operatorname{logsumexp}_k s^{\mathrm{rep}}_k
                 + \log(2 R) - \log l_2 - \log \mathrm{win}.
    """
    R = state.num_reticulations()
    if R == 0:
        return None
    old_net = state.species_net
    old_heights = state.net_heights

    rets = [v for v in old_net.V() if v.is_reticulation()]
    v2 = rets[int(rng.integers(len(rets)))]
    parents = old_net.get_parents(v2)
    if len(parents) != 2:
        return None
    v1 = parents[int(rng.integers(2))]
    v5 = parents[1] if parents[0] is v1 else parents[0]

    if v1.is_reticulation() or old_net.in_degree(v1) == 0:
        return None
    v3s = old_net.get_parents(v1)
    if len(v3s) != 1:
        return None
    v3 = v3s[0]
    other_children = [c for c in old_net.get_children(v1) if c is not v2]
    if len(other_children) != 1:
        return None
    v4 = other_children[0]
    v6s = old_net.get_children(v2)
    if len(v6s) != 1:
        return None
    v6 = v6s[0]

    if v3 in old_net.get_parents(v4):
        return None
    if v5 in old_net.get_parents(v6):
        return None
    if v3 is v5 and v4 is v6:
        return None

    l2 = old_heights[v5] - old_heights[v6]
    if l2 <= 0.0:
        return None
    t2 = old_heights[v2]
    lo = max(old_heights[v4], t2)
    hi = old_heights[v3]
    win = hi - lo
    if win <= 0.0:
        return None
    rev_donor = (v3.label, v4.label)
    rev_retic = (v5.label, v6.label)

    snap = state._engine.clone_caches()
    new_net = _clone_net(old_net)
    new_heights = _heights(new_net)
    cv2 = None
    for v in new_net.V():
        if v.is_reticulation() and v.label == v2.label:
            cv2 = v
            break
    if cv2 is None:
        return None
    cparents = new_net.get_parents(cv2)
    cv1 = next((p for p in cparents if p.label == v1.label), None)
    if cv1 is None:
        return None
    try:
        del_edge = _edge_between(new_net, cv1, cv2)
        new_net.remove_edge(del_edge)
        cv2.set_is_reticulation(False)
        _suppress_degree2(new_net, new_heights, cv1)
        _suppress_degree2(new_net, new_heights, cv2)
    except Exception:
        return None
    if len([e for e in new_net.E()]) < 2:
        return None
    _sync_lengths(new_net, new_heights)

    # Reverse-add placement enumeration uses the *current* (unchanged) gene
    # trees on the post-delete network.
    placements = _add_reticulation_placements(state, new_net, new_heights)
    if not placements:
        return None
    s_reps = [p["s_rep"] for p in placements]
    lse = _logsumexp(s_reps)
    s_rev = None
    for p in placements:
        if p["donor_labels"] == rev_donor and p["retic_labels"] == rev_retic:
            s_rev = p["s_rep"]
            break
    if s_rev is None:
        return None

    loghr = (
        s_rev
        - lse
        + math.log(2.0 * R)
        - math.log(l2)
        - math.log(win)
    )

    state.species_net = new_net
    state.net_heights = new_heights
    state._engine.invalidate_network()

    def undo() -> None:
        state.species_net = old_net
        state.net_heights = old_heights
        state._engine.restore_caches(snap)

    return loghr, undo


def _relocation_loghr(
    l2_new: float,
    win_new: float,
    l2_old: float,
    win_old: float,
    log_qf: float,
    log_qr: float,
) -> float:
    """Log Hastings ratio of a coupled reticulation relocation.

    The birth Jacobian of the newly attached hybrid (``l2_new * win_new``)
    minus that of the detached one (``l2_old * win_old``), plus the gene-tree
    coupling term ``log_qr - log_qf``.  Antisymmetric under swapping the new/old
    roles -- ``_relocation_loghr(a, b, c, d, qf, qr) ==
    -_relocation_loghr(c, d, a, b, qr, qf)`` -- which is exactly the reverse
    relocation's ratio, so the move is reversible.
    """
    return (
        math.log(l2_new)
        + math.log(win_new)
        - math.log(l2_old)
        - math.log(win_old)
        + (log_qr - log_qf)
    )


def op_relocate_reticulation_coupled(
    state: SeqState, rng: np.random.Generator
) -> Optional[tuple[float, Callable]]:
    r"""Coupled dimension-preserving reticulation *relocation* RJMCMC move.

    Add/delete change the reticulation *count*; neither can slide a hybrid from
    one clade to another without first passing through the tree -- the very
    barrier the coupled add is built to avoid -- so a chain seeded with a
    reticulation on the wrong clade (e.g. from a rough gene-tree bootstrap) gets
    stuck there.  This move detaches a reticulation and re-attaches it on a
    freshly chosen edge pair in a single step, keeping the count fixed.

    Construction.  Pick a reticulation ``A`` and one parent uniformly and remove
    it, suppressing the two degree-2 nodes to obtain the base network ``N0``
    (identical machinery to :func:`op_delete_reticulation_coupled`).  Draw a new
    placement ``B`` *uniformly* from the geometric placements of ``N0``
    (:func:`_geometric_placements`), draw its continuous parameters
    (``t2 ~ U(0, l2_B)``, ``t1 ~ U`` in the donor window ``win_B``,
    ``gamma ~ U(0, 1)``), and re-attach.  Finally re-propose every locus's gene
    tree guided by the new network (:func:`_coupled_gene_tree_reproposal`).

    Hastings ratio.  Because ``B`` is drawn uniformly from a placement set that
    depends only on ``N0`` (not on the gene trees), the ``1 / K`` selection
    density is identical in both directions and cancels; the base network ``N0``
    recovered by the reverse (remove ``B`` from the proposed net) is topology-
    and height-identical to the forward's, so ``A`` is always among its
    placements.  What remains is the birth Jacobian of each direction plus the
    gene-tree coupling term:

    .. math::

        \log H = \log l^{B}_2 + \log \mathrm{win}_B
                 - \log l^{A}_2 - \log \mathrm{win}_A
                 + \sum_i \big(\log P^{\mathrm r}_i - \log P^{\mathrm f}_i\big),

    which is exactly antisymmetric under swapping ``A`` and ``B`` -- the move is
    its own reverse, so ``logH(x->x') == -logH(x'->x)`` (checked numerically in
    ``tests/test_mcmc_seq_coupled.py``).
    """
    R = state.num_reticulations()
    if R == 0:
        return None
    old_net = state.species_net
    old_heights = state.net_heights
    old_gts = state.gene_trees
    old_gt_h = state.gt_heights

    # Pick reticulation A + parent to detach (defines the removed hybrid edge).
    rets = [v for v in old_net.V() if v.is_reticulation()]
    v2 = rets[int(rng.integers(len(rets)))]
    parents = old_net.get_parents(v2)
    if len(parents) != 2:
        return None
    v1 = parents[int(rng.integers(2))]
    v5 = parents[1] if parents[0] is v1 else parents[0]
    if v1.is_reticulation() or old_net.in_degree(v1) == 0:
        return None
    v3s = old_net.get_parents(v1)
    if len(v3s) != 1:
        return None
    v3 = v3s[0]
    other_children = [c for c in old_net.get_children(v1) if c is not v2]
    if len(other_children) != 1:
        return None
    v4 = other_children[0]
    v6s = old_net.get_children(v2)
    if len(v6s) != 1:
        return None
    v6 = v6s[0]
    if v3 in old_net.get_parents(v4):
        return None
    if v5 in old_net.get_parents(v6):
        return None
    if v3 is v5 and v4 is v6:
        return None

    # Reverse birth geometry for A, recovered from the pre-move network.
    l2_A = old_heights[v5] - old_heights[v6]
    if l2_A <= 0.0:
        return None
    t2_A = old_heights[v2]
    lo_A = max(old_heights[v4], t2_A)
    win_A = old_heights[v3] - lo_A
    if win_A <= 0.0:
        return None

    # Build N0 = old_net - A (deep copy; suppress the two degree-2 nodes).
    snap = state._engine.clone_caches()
    n0 = _clone_net(old_net)
    n0_h = _heights(n0)
    cv2 = next(
        (v for v in n0.V() if v.is_reticulation() and v.label == v2.label), None
    )
    if cv2 is None:
        return None
    cv1 = next(
        (p for p in n0.get_parents(cv2) if p.label == v1.label), None
    )
    if cv1 is None:
        return None
    try:
        n0.remove_edge(_edge_between(n0, cv1, cv2))
        cv2.set_is_reticulation(False)
        _suppress_degree2(n0, n0_h, cv1)
        _suppress_degree2(n0, n0_h, cv2)
    except Exception:
        return None
    if len([e for e in n0.E()]) < 2:
        return None
    _sync_lengths(n0, n0_h)

    # Draw a new placement B uniformly from N0's geometric placements.
    placements = _geometric_placements(n0, n0_h)
    if not placements:
        return None
    b = placements[int(rng.integers(len(placements)))]

    # Build N_B = N0 + B.
    new_net = _clone_net(n0)
    new_heights = _heights(new_net)
    d = _find_edge_by_labels(new_net, *b["donor_labels"])
    r = _find_edge_by_labels(new_net, *b["retic_labels"])
    if d is None or r is None:
        return None
    b3, b4 = d.src, d.dest
    b5, b6 = r.src, r.dest
    l2_B = new_heights[b5] - new_heights[b6]
    if l2_B <= 0.0:
        return None
    t2_B = new_heights[b6] + l2_B * rng.random()
    lo_B = max(new_heights[b4], t2_B)
    win_B = new_heights[b3] - lo_B
    if win_B <= 0.0:
        return None
    t1_B = lo_B + win_B * rng.random()
    g = float(rng.random())
    try:
        nv1 = _split_edge(new_net, new_heights, b3, b4, t1_B)
        nv2 = _split_edge(new_net, new_heights, b5, b6, t2_B)
        in_e = list(new_net.in_edges(nv2))
        if len(in_e) != 1:
            return None
        in_e[0].set_gamma(1.0 - g)
        nv2.set_is_reticulation(True)
        new_net.add_edges(Edge(nv1, nv2, gamma=g))
    except Exception:
        return None
    if _has_parallel_edges(new_net):
        return None
    try:
        if not new_net.is_acyclic():
            return None
    except Exception:
        return None
    # Level cap (e.g. level-1 mode): relocation keeps the reticulation
    # count fixed but can still slide a hybrid into an already-occupied
    # blob, raising the level.  Reject before the coupled gene-tree
    # re-proposal / scoring, mirroring the guard in the coupled add.
    if (
        state.priors.max_level is not None
        and _network_level(new_net) > state.priors.max_level
    ):
        return None
    _sync_lengths(new_net, new_heights)

    # Guided joint gene-tree re-proposal under the relocated network.
    repro = _coupled_gene_tree_reproposal(state, new_net, old_net, rng)
    if repro is None:
        return None
    new_gts, new_hs, log_qf_gt, log_qr_gt = repro

    loghr = _relocation_loghr(l2_B, win_B, l2_A, win_A, log_qf_gt, log_qr_gt)

    state.species_net = new_net
    state.net_heights = new_heights
    state.gene_trees = new_gts
    state.gt_heights = new_hs
    state._engine.invalidate_network()
    for i in range(state._engine.n_loci):
        state._engine.invalidate_locus(i)

    def undo() -> None:
        state.species_net = old_net
        state.net_heights = old_heights
        state.gene_trees = old_gts
        state.gt_heights = old_gt_h
        state._engine.restore_caches(snap)

    return loghr, undo


# ======================================================================
# Proposal kernel
# ======================================================================

class MCMCSeqKernel:
    """Weighted mixture of MCMC_SEQ operators.

    On each step :meth:`propose` selects an operator by weight and applies it.
    The default weights split effort across gene-tree moves and species-network
    moves in roughly the proportions PhyloNet uses (more weight on gene trees
    when there are few loci, since they carry most of the signal).

    Attributes:
        rng: Random generator driving operator selection and proposals.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        *,
        max_reticulations: int = 4,
        coupled_dimension_moves: bool = False,
    ) -> None:
        """Build the kernel.

        Args:
            rng: Random generator.
            max_reticulations: Reticulation cap (disables add when reached).
            coupled_dimension_moves: When ``True`` the add / delete-reticulation
                moves re-propose every gene tree at the instant the dimension
                changes (:func:`op_add_reticulation_coupled` /
                :func:`op_delete_reticulation_coupled`).  When ``False``
                (default) they change only the network and let the gene trees
                migrate gradually via the per-locus moves
                (:func:`op_add_reticulation_decoupled` /
                :func:`op_delete_reticulation_decoupled`).  The decoupled scheme
                (PhyloNet's) avoids the trans-dimensional hysteresis the coupled
                all-at-once re-proposal induces and mixes across reticulation
                counts far better.
        """
        self.rng = rng
        self.max_reticulations = max_reticulations
        self.coupled_dimension_moves = coupled_dimension_moves
        add_op = (
            op_add_reticulation_coupled
            if coupled_dimension_moves
            else op_add_reticulation_decoupled
        )
        del_op = (
            op_delete_reticulation_coupled
            if coupled_dimension_moves
            else op_delete_reticulation_decoupled
        )
        # (operator, weight) -- continuous gene-tree/network moves dominate;
        # topology + dimension changes are rarer, as in PhyloNet.
        self._ops: list[tuple[Callable, float]] = [
            (op_gene_node_height, 0.28),
            (op_gene_tree_nni, 0.18),
            (op_net_node_height, 0.18),
            (op_change_gamma, 0.07),
            (op_change_theta, 0.09),
            (add_op, 0.06),
            (del_op, 0.06),
            (op_relocate_reticulation_coupled, 0.08),
        ]
        self._weights = np.asarray([w for _, w in self._ops], dtype=np.float64)
        self._weights /= self._weights.sum()

    def propose(
        self, state: SeqState
    ) -> Optional[tuple[float, Callable]]:
        """Select and apply one operator.

        Returns:
            ``(log_hastings_ratio, undo)`` on success, or ``None`` if the
            chosen operator was inapplicable / illegal.
        """
        idx = int(self.rng.choice(len(self._ops), p=self._weights))
        op = self._ops[idx][0]
        try:
            return op(state, self.rng)
        except Exception:
            return None


# ======================================================================
# Result containers
# ======================================================================

@dataclass
class MCMCSeqSample:
    """One recorded posterior sample.

    Attributes:
        iteration: Chain iteration at which the sample was taken.
        log_posterior: Unnormalised log posterior.
        log_likelihood: Total phylogenetic + coalescent log likelihood.
        network_newick: Extended-Newick string of the species network.
        theta: Population mutation rate.
        num_reticulations: Reticulation count.
        gamma_major: Largest inheritance probability across the network's
            reticulations (``None`` for a tree state).
    """

    iteration: int
    log_posterior: float
    log_likelihood: float
    network_newick: str
    theta: float
    num_reticulations: int
    gamma_major: Optional[float] = None


@dataclass
class MCMCSeqResult:
    """Aggregate result of an :meth:`MCMC_SEQ.search` run.

    Attributes:
        map_network: Maximum-a-posteriori species network found.
        map_log_posterior: Its log posterior.
        map_theta: ``theta`` at the MAP state.
        samples: Recorded posterior samples (post burn-in, thinned).
        acceptance_rate: Fraction of proposals accepted.
        num_iterations: Total proposals attempted.
        wall_time_sec: Wall-clock duration.
    """

    map_network: Network
    map_log_posterior: float
    map_theta: float
    samples: list[MCMCSeqSample] = field(default_factory=list)
    acceptance_rate: float = 0.0
    num_iterations: int = 0
    wall_time_sec: float = 0.0
    map_log_likelihood: Optional[float] = None
    num_leaves: Optional[int] = None
    total_sites: Optional[int] = None

    # -- post-analysis (Tracer interop + native diagnostics) ----------

    def _sample_step(self) -> int:
        """Sampling interval (states between recorded samples), >= 1."""
        if len(self.samples) >= 2:
            step = self.samples[1].iteration - self.samples[0].iteration
            if step > 0:
                return int(step)
        return 1

    def trace_table(self) -> tuple[list[int], dict[str, list[float]]]:
        """Build a Tracer-style trace table from the recorded samples.

        Returns:
            ``(states, traces)`` where ``states`` are the per-sample iteration
            indices and ``traces`` maps column name to per-sample values.
            Columns: ``posterior``, ``likelihood``, ``prior`` (= posterior -
            likelihood), ``theta`` and ``reticulationCount`` -- mirroring the
            quantities PhyloNet's MCMC_SEQ logs.
        """
        states = [s.iteration for s in self.samples]
        traces: dict[str, list[float]] = {
            "posterior": [s.log_posterior for s in self.samples],
            "likelihood": [s.log_likelihood for s in self.samples],
            "prior": [s.log_posterior - s.log_likelihood for s in self.samples],
            "theta": [s.theta for s in self.samples],
            "reticulationCount": [
                float(s.num_reticulations) for s in self.samples
            ],
        }
        # Only a rectangular trace can be summarised; include the inheritance
        # probability only when every recorded state carried a reticulation.
        if self.samples and all(
            s.gamma_major is not None for s in self.samples
        ):
            traces["gammaMajor"] = [float(s.gamma_major) for s in self.samples]
        return states, traces

    def write_log(self, path: str) -> None:
        """Write a BEAST/Tracer-compatible ``.log`` of the sampled chain.

        Args:
            path: Output path (conventionally ``.log``); open it in Tracer.
        """
        states, traces = self.trace_table()
        write_tracer_log(
            states,
            traces,
            path,
            comments=[
                "PhyNetPy MCMC_SEQ posterior sample",
                f"iterations={self.num_iterations} "
                f"acceptance_rate={self.acceptance_rate:.4f} "
                f"wall_time_sec={self.wall_time_sec:.2f}",
            ],
        )

    def write_networks(self, path: str) -> None:
        """Write the sampled species networks as a NEXUS ``.trees`` file.

        Args:
            path: Output path (conventionally ``.trees``); for DensiTree /
                TreeAnnotator-style topology summaries.
        """
        states = [s.iteration for s in self.samples]
        newicks = [s.network_newick for s in self.samples]
        write_trees_nexus(states, newicks, path, prefix="STATE")

    def summary(self, *, hpd_prob: float = 0.95) -> ChainSummary:
        """Per-parameter diagnostics (mean, HPD, ESS, ...) for the chain.

        Args:
            hpd_prob: Coverage for the reported HPD interval (default 0.95).

        Returns:
            A :class:`~phynetpy._chain_analysis.ChainSummary`; ``print`` it for
            a Tracer-style table, or inspect ``.parameters`` / ``.min_ess``.
        """
        _, traces = self.trace_table()
        return summarize_traces(
            traces, step_size=self._sample_step(), hpd_prob=hpd_prob
        )

    def reticulation_posterior(self) -> dict[int, float]:
        """Posterior P(number of reticulations) from the sampled chain.

        This is the *mass*-based estimate to report instead of the MAP
        reticulation count: the joint MAP can sit on the coalescent
        ``theta -> 0`` degeneracy and over-count reticulations, whereas this
        marginal over the discrete dimension is what the chain actually visited.
        """
        return _reticulation_posterior(self.samples)

    def topology_posterior(
        self, top_n: Optional[int] = None
    ) -> list[tuple[str, float]]:
        """Ranked ``(representative_newick, posterior_probability)`` topologies.

        Topologies are compared ignoring branch lengths / gamma (see
        :func:`_topology_signature`).  The top entry is the posterior-mode
        ("MAP") *topology* -- a far more trustworthy point estimate than the
        single highest-density state.
        """
        return _topology_posterior(self.samples, top_n=top_n)

    def map_topology(self) -> Optional[str]:
        """Representative newick of the most-sampled topology (``None`` if empty)."""
        top = _topology_posterior(self.samples, top_n=1)
        return top[0][0] if top else None

    # -- model selection (AIC / BIC over reticulation count) ----------

    def _num_leaves(self) -> Optional[int]:
        """Leaf count of the MAP network (``num_leaves`` field, else derived)."""
        if self.num_leaves is not None:
            return self.num_leaves
        try:
            return len(self.map_network.get_leaves())
        except Exception:
            return None

    def information_criteria(self) -> Optional[dict[str, float]]:
        """AIC / AICc / BIC of the MAP network.

        Uses the MAP network's maximised log *likelihood* (not posterior) and
        its free-parameter count ``k = L + 3r`` (see
        :func:`_count_free_parameters`); the BIC sample size is the total number
        of aligned sites.  Returns ``None`` when the likelihood was not recorded
        (e.g. a hand-built result).  Lower AIC/BIC is better.
        """
        if self.map_log_likelihood is None:
            return None
        L = self._num_leaves()
        if L is None:
            return None
        r = sum(1 for v in self.map_network.V() if v.is_reticulation())
        k = _count_free_parameters(L, r)
        ic = _information_criteria(self.map_log_likelihood, k, self.total_sites)
        ic["num_reticulations"] = r
        return ic

    def model_selection_by_reticulation(self) -> list[dict[str, float]]:
        r"""AIC / AICc / BIC per reticulation count, from the sampled chain.

        Groups the posterior samples by reticulation count, takes the best
        (highest) log likelihood seen at each count as that model's maximised
        fit, and scores it with ``k = L + 3r`` parameters over the total site
        count.  Each row also carries ``dAIC`` / ``dBIC`` -- the increase over
        the best-scoring model -- so you can read off whether an *extra*
        reticulation earns its ``\Delta k = 3`` penalty or is just the chain
        exploring higher dimensions (a ``dAIC`` above ~2, or any positive
        ``dBIC``, means the simpler model is preferred despite its lower
        likelihood).  Sorted by reticulation count.
        """
        if not self.samples:
            return []
        L = self._num_leaves()
        if L is None:
            return []
        best_ll: dict[int, float] = {}
        for s in self.samples:
            r = s.num_reticulations
            if r not in best_ll or s.log_likelihood > best_ll[r]:
                best_ll[r] = s.log_likelihood
        rows: list[dict[str, float]] = []
        for r in sorted(best_ll):
            k = _count_free_parameters(L, r)
            ic = _information_criteria(best_ll[r], k, self.total_sites)
            ic["num_reticulations"] = r
            rows.append(ic)
        if rows:
            min_aic = min(row["AIC"] for row in rows)
            has_bic = all(math.isfinite(row.get("BIC", float("nan")))
                          for row in rows)
            min_bic = min(row["BIC"] for row in rows) if has_bic else None
            for row in rows:
                row["dAIC"] = row["AIC"] - min_aic
                if min_bic is not None:
                    row["dBIC"] = row["BIC"] - min_bic
        return rows


# ======================================================================
# Public driver
# ======================================================================

def _normalise_loci(
    loci: Sequence[Any],
) -> list[dict[str, str]]:
    """Coerce a sequence of loci to a list of ``{label: sequence}`` dicts.

    Accepts dicts directly, or :class:`phynetpy.MSA.MSA` objects (whose
    records are read out as ``name -> sequence string``).
    """
    out: list[dict[str, str]] = []
    for locus in loci:
        if isinstance(locus, dict):
            out.append({k: str(v) for k, v in locus.items()})
        elif isinstance(locus, MSA):
            out.append(
                {rec.get_name(): "".join(str(x) for x in rec.get_seq())
                 for rec in locus.get_records()}
            )
        else:
            raise TypeError(
                "Each locus must be a dict[label,seq] or an MSA instance."
            )
    return out


def _build_species_tree(
    loci: list[dict[str, str]],
    mapping: dict[str, list[str]],
) -> Network:
    """UPGMA species tree from per-species concatenated sequences.

    Uses one representative allele per species (the first listed) and the
    concatenation of all loci, mirroring a cheap PhyloNet-style starting
    species tree when none is supplied.
    """
    species = list(mapping.keys())
    concat: dict[str, str] = {}
    for sp in species:
        rep = mapping[sp][0]
        pieces = []
        for aln in loci:
            seq = aln.get(rep)
            if seq is None:
                # any allele of this species in this locus
                for allele in mapping[sp]:
                    if allele in aln:
                        seq = aln[allele]
                        break
            pieces.append(seq if seq is not None else "")
        concat[sp] = "".join(pieces)
    return build_upgma_gene_tree(concat)


class MCMC_SEQ:
    """Bayesian co-estimation of species networks and gene trees from sequences.

    Re-implements PhyloNet's ``MCMC_SEQ`` (Wen & Nakhleh 2018) on PhyNetPy's
    data structures.  The per-locus phylogenetic likelihood and the timed MSNC
    density -- the numerically load-bearing parts -- are
    :mod:`phynetpy._seq_likelihood` and are verified to match PhyloNet exactly;
    this class wires them into a reversible-jump Metropolis-Hastings sampler
    over the gene trees, species network, inheritance probabilities and
    population mutation rate.

    Mirrors the public surface of :class:`phynetpy.infer.MCMC_GT`:

    * :meth:`score` -- log posterior of the current state.
    * :meth:`search` -- run the sampler and return an :class:`MCMCSeqResult`.

    Attributes:
        loci: Per-locus alignments (label -> sequence).
        mapping: Species -> list of allele labels.
        species_net: Current / starting species network.
    """

    def __init__(
        self,
        loci: Sequence[Any],
        mapping: dict[str, list[str]],
        species_net: Optional[Network] = None,
        *,
        priors: Optional[MCMCSeqPriors] = None,
        model: Optional[SubstitutionModel] = None,
        theta: Optional[float] = None,
        gene_trees: Optional[list[Network]] = None,
    ) -> None:
        """Initialise the inference object.

        Args:
            loci: Sequence of per-locus alignments, each a ``dict`` mapping an
                allele label to its aligned nucleotide string, or an
                :class:`phynetpy.MSA.MSA`.
            mapping: Species -> list of allele labels (the allele labels are
                the alignment row names / gene-tree leaf labels).
            species_net: Optional starting species network (ultrametric, in
                substitution units).  When ``None`` a UPGMA species tree is
                built from the concatenated alignments.
            priors: Prior hyperparameters (defaults match PhyloNet).
            model: Nucleotide substitution model (default JC69).
            theta: Starting population mutation rate (default ``theta_mean``).
            gene_trees: Optional starting per-locus gene trees; when ``None``
                each is built by UPGMA on its locus (PhyloNet's ``-sgt``).
        """
        self.loci = _normalise_loci(loci)
        self.mapping = mapping
        self.priors = priors if priors is not None else MCMCSeqPriors()
        self.model = model if model is not None else JC69()
        self.theta = theta if theta is not None else self.priors.theta_mean
        self.species_of: dict[str, str] = {
            allele: sp for sp, alleles in mapping.items() for allele in alleles
        }

        if gene_trees is None:
            gene_trees = [build_upgma_gene_tree(aln) for aln in self.loci]
        self.gene_trees = gene_trees

        if species_net is None:
            species_net = _build_species_tree(self.loci, mapping)
        self.species_net = species_net

    def _new_state(self) -> SeqState:
        """Construct a fresh :class:`SeqState` from deep-copied inputs."""
        return SeqState(
            _clone_net(self.species_net),
            [_clone_net(gt) for gt in self.gene_trees],
            self.species_of,
            self.loci,
            self.priors,
            self.model,
            self.theta,
        )

    def _total_sites(self) -> Optional[int]:
        """Total aligned site count across all loci (BIC sample size)."""
        total = 0
        for aln in self.loci:  # normalised to {label: sequence} dicts
            if not aln:
                continue
            total += len(next(iter(aln.values())))
        return total if total > 0 else None

    def score(self) -> float:
        """Log posterior of the current (starting) state."""
        return self._new_state().log_posterior()

    def warm_start(
        self,
        *,
        gt_iters: int = 6_000,
        gt_burn_in: Optional[int] = None,
        gt_thin: int = 20,
        max_reticulations: Optional[int] = None,
        max_level: Optional[int] = None,
        seed: Any = None,
    ) -> Network:
        """Bootstrap the starting species network with a fast MCMC_GT search.

        Co-estimation from sequences faces a joint gene-tree/network mode
        barrier: from a plain species-tree start the per-locus gene trees
        co-adapt to the tree, after which adding a reticulation only costs prior
        (see :func:`op_add_reticulation_coupled`) and the chain collapses to a
        tree.  This method sidesteps the barrier by inferring a
        reticulation-bearing network *cheaply, from the gene-tree topologies
        alone* -- running :class:`~phynetpy.MCMC_GT` on this object's current
        per-locus gene-tree estimates -- and adopting it as the SEQ chain's
        starting network.  Seeded there, the coupled add/delete move holds and
        *refines* the reticulation (inheritance probabilities, node heights,
        co-estimated gene trees) instead of never discovering it.

        The gene-tree topologies drive the reticulation signal, so the quality
        of the seed is bounded by the quality of the starting gene-tree
        estimates and of the MCMC_GT search; the SEQ chain then does the
        sequence-aware refinement.  Idempotent-ish: sets and returns
        :attr:`species_net`.

        Args:
            gt_iters: MCMC_GT proposals for the bootstrap search.
            gt_burn_in: MCMC_GT burn-in (defaults to ``gt_iters // 4``).
            gt_thin: MCMC_GT sample thinning.
            max_reticulations: Reticulation cap for the bootstrap; defaults to
                the SEQ priors' ``max_reticulations``.
            max_level: Network-level cap for the bootstrap; defaults to the SEQ
                priors' ``max_level``.  Ensures a level-constrained SEQ chain is
                not seeded with an above-level network the sampler could never
                have proposed.
            seed: Seed for the bootstrap search.

        Returns:
            The seed :class:`~phynetpy.Network.Network` (also stored on
            :attr:`species_net`).
        """
        # Lazy import: MCMC_GT lives in a sibling module and pulls in the GT
        # likelihood stack; importing at call time keeps module import cheap and
        # avoids any import cycle.
        from ._mcmc_gt import MCMC_GT, MCMC_GTPriors
        from .GeneTrees import GeneTrees

        cap = (
            max_reticulations
            if max_reticulations is not None
            else self.priors.max_reticulations
        )
        lvl_cap = (
            max_level if max_level is not None else self.priors.max_level
        )
        gts = GeneTrees(
            gene_tree_list=[_clone_net(gt) for gt in self.gene_trees],
            species_gene_mapping=self.mapping,
        )
        booter = MCMC_GT.from_consensus(gts, self.mapping, priors=MCMC_GTPriors())
        gt_res = booter.search(
            method="mh",
            num_iter=gt_iters,
            burn_in=gt_iters // 4 if gt_burn_in is None else gt_burn_in,
            thin=gt_thin,
            max_reticulations=cap,
            max_lvl=lvl_cap,
            seed=seed,
        )
        self.species_net = gt_res.best_network

        # Validate the seed: on small data (or tight identifiability priors) the
        # bootstrap can return a network whose reticulation is degenerate /
        # unidentifiable (e.g. a sub-``min_cycle_size`` cycle), which scores
        # ``-inf`` under the SEQ prior and would poison the whole chain.  Fall
        # back to the plain species tree so the chain always starts valid; the
        # coupled add-reticulation move can still discover a *valid* reticulation
        # from there.
        try:
            seed_finite = math.isfinite(self._new_state().log_posterior())
        except Exception:
            seed_finite = False
        if not seed_finite:
            self.species_net = _build_species_tree(self.loci, self.mapping)
        return self.species_net

    def search(
        self,
        *,
        num_iter: int = 20_000,
        burn_in: int = 5_000,
        sample_freq: int = 100,
        seed: Any = None,
        kernel: Optional[MCMCSeqKernel] = None,
        progress: bool = False,
        check_every: int = 0,
        control: Optional[Callable[[dict], str]] = None,
        temperatures: Optional[Sequence[float]] = None,
        swap_interval: int = 1,
        warm_start: bool = False,
        warm_start_kwargs: Optional[dict] = None,
    ) -> MCMCSeqResult:
        """Run the RJMCMC sampler.

        Args:
            num_iter: Total proposals.
            burn_in: Proposals discarded before sampling begins.
            sample_freq: Thinning interval for recorded samples.
            seed: Seed for all randomness (any
                :class:`numpy.random.SeedSequence` seed).
            kernel: Custom proposal kernel; a default
                :class:`MCMCSeqKernel` honouring ``max_reticulations`` is built
                when ``None``.
            progress: Print a periodic status line when ``True``.
            check_every: When ``> 0`` (and ``control`` is given), invoke the
                ``control`` hook every ``check_every`` iterations so a caller
                can pause or halt the chain.  ``0`` disables the hook.
            control: Optional cooperative-control callback.  Called with a
                progress ``dict`` (keys: ``iteration``, ``num_iter``,
                ``log_posterior``, ``map_log_posterior``,
                ``num_reticulations``, ``acceptance_rate``, ``samples``) and
                must return one of ``"continue"`` (keep going), ``"pause"``
                (block here, re-polled every ~0.2 s until it stops returning
                ``"pause"``), or ``"stop"`` (halt now and return the partial
                result collected so far).  This is the hook
                :func:`run_parallel_chains` uses to drive each worker.
            temperatures: Optional Metropolis-coupled-MCMC (MC3) temperature
                ladder, e.g. ``[1.0, 2.0, 3.0]``.  ``None`` or a single ``1.0``
                runs an ordinary single (cold) chain.  When two or more are
                given, that many chains are run in lockstep -- each targets the
                posterior raised to ``1/T`` -- and adjacent chains periodically
                propose state swaps so the cold chain (``T = 1``) can escape
                local modes (e.g. the high-reticulation trap).  Only the cold
                chain is sampled and returned.
            swap_interval: MC3 only -- iterations between swap proposals.
            warm_start: When ``True``, first bootstrap the starting species
                network with :meth:`warm_start` (a fast MCMC_GT search on the
                current gene-tree estimates) so the chain begins from a
                reticulation-bearing network the coupled move can refine, rather
                than from a plain tree it would never leave.  Recommended for
                sequence co-estimation.
            warm_start_kwargs: Optional overrides forwarded to
                :meth:`warm_start` (e.g. ``{"gt_iters": 8000}``).  The bootstrap
                reuses ``seed`` unless one is given here.

        Returns:
            An :class:`MCMCSeqResult`; ``self.species_net`` / ``self.theta`` are
            updated in place to the MAP estimate.  When ``control`` requests an
            early stop the result holds the samples gathered up to that point
            and ``num_iterations`` reflects the iterations actually run.
        """
        _nm.warn_if_large_mcmc(len(self.mapping), method="MCMC_SEQ")

        if warm_start:
            ws_kwargs = dict(warm_start_kwargs or {})
            ws_kwargs.setdefault("seed", seed)
            ws_kwargs.setdefault(
                "max_reticulations", self.priors.max_reticulations
            )
            self.warm_start(**ws_kwargs)

        if temperatures is not None and len([t for t in temperatures]) > 1:
            return self._search_mc3(
                temperatures=list(temperatures),
                num_iter=num_iter, burn_in=burn_in, sample_freq=sample_freq,
                seed=seed, progress=progress, check_every=check_every,
                control=control, swap_interval=max(1, swap_interval),
            )

        rng = np.random.default_rng(seed)
        if kernel is None:
            kernel = MCMCSeqKernel(
                rng, max_reticulations=self.priors.max_reticulations
            )
        else:
            kernel.rng = rng

        state = self._new_state()
        cur = state.log_posterior()

        map_post = cur
        map_net = _clone_net(state.species_net)
        map_theta = state.theta
        map_ll = state.log_likelihood()

        samples: list[MCMCSeqSample] = []
        accepted = 0
        start = time.time()
        completed = num_iter

        for it in range(num_iter):
            proposal = kernel.propose(state)
            if proposal is not None:
                loghr, undo = proposal
                prop = state.log_posterior()
                # A non-finite proposal is a zero-probability (invalid) state;
                # reject it outright.  Guarding here also avoids ``inf - inf``
                # producing a NaN acceptance ratio.
                if not math.isfinite(prop):
                    undo()
                else:
                    log_alpha = (prop - cur) + loghr
                    if math.log(rng.random()) < log_alpha:
                        cur = prop
                        accepted += 1
                        if cur > map_post:
                            map_post = cur
                            map_net = _clone_net(state.species_net)
                            map_theta = state.theta
                            map_ll = state.log_likelihood()
                    else:
                        undo()

            if it >= burn_in and (it - burn_in) % sample_freq == 0:
                samples.append(
                    MCMCSeqSample(
                        iteration=it,
                        log_posterior=cur,
                        log_likelihood=state.log_likelihood(),
                        network_newick=_safe_newick(state.species_net),
                        theta=state.theta,
                        num_reticulations=state.num_reticulations(),
                        gamma_major=_major_gamma(state.species_net),
                    )
                )

            if progress and it % max(1, num_iter // 20) == 0:
                print(
                    f"[MCMC_SEQ] iter {it}/{num_iter} "
                    f"logP={cur:.4f} MAP={map_post:.4f} "
                    f"reti={state.num_reticulations()} "
                    f"acc={accepted / max(1, it + 1):.3f}",
                    flush=True,
                )

            # Cooperative pause / halt checkpoint.  ``control`` returns
            # "continue" | "pause" | "stop"; we block in place while paused and
            # break out (returning the partial result) on stop.
            if control is not None and check_every > 0 and it % check_every == 0:
                prog = {
                    "iteration": it,
                    "num_iter": num_iter,
                    "log_posterior": cur,
                    "map_log_posterior": map_post,
                    "num_reticulations": state.num_reticulations(),
                    "acceptance_rate": accepted / max(1, it + 1),
                    "samples": samples,
                }
                action = control(prog)
                while action == "pause":
                    time.sleep(0.2)
                    action = control(prog)
                if action == "stop":
                    completed = it + 1
                    break

        self.species_net = map_net
        self.theta = map_theta
        return MCMCSeqResult(
            map_network=map_net,
            map_log_posterior=map_post,
            map_theta=map_theta,
            samples=samples,
            acceptance_rate=accepted / max(1, completed),
            num_iterations=completed,
            wall_time_sec=time.time() - start,
            map_log_likelihood=map_ll,
            num_leaves=len(self.mapping),
            total_sites=self._total_sites(),
        )

    @staticmethod
    def _mh_step(kernel, state, cur: float, beta: float, rng) -> tuple:
        """One tempered Metropolis-Hastings step on ``state``.

        Targets ``pi^beta`` (``beta = 1`` is the ordinary posterior).  Applies
        the move in place and either keeps it or undoes it.

        Returns:
            ``(new_cur, accepted)`` -- the (untempered) log posterior of the
            state after the step and whether the proposal was accepted.
        """
        proposal = kernel.propose(state)
        if proposal is None:
            return cur, False
        loghr, undo = proposal
        prop = state.log_posterior()
        if not math.isfinite(prop):
            undo()
            return cur, False
        log_alpha = beta * (prop - cur) + loghr
        if math.log(rng.random()) < log_alpha:
            return prop, True
        undo()
        return cur, False

    def _search_mc3(
        self,
        *,
        temperatures: list,
        num_iter: int,
        burn_in: int,
        sample_freq: int,
        seed: Any,
        progress: bool,
        check_every: int,
        control: Optional[Callable[[dict], str]],
        swap_interval: int,
    ) -> MCMCSeqResult:
        """Metropolis-coupled MCMC (MC3): a temperature ladder with swaps.

        Runs ``len(temperatures)`` chains in lockstep, each targeting
        ``posterior ** (1 / T)``.  Adjacent chains attempt state swaps every
        ``swap_interval`` iterations (acceptance
        ``exp((beta_i - beta_j) (L_j - L_i))``), which lets the cold chain hop
        between modes the hot chains have flattened.  Only the cold (``T = 1``)
        chain is sampled; its draws are a valid posterior sample.
        """
        m = len(temperatures)
        betas = [1.0 / float(t) for t in temperatures]
        cold = int(min(range(m), key=lambda i: temperatures[i]))

        ss = (
            seed if isinstance(seed, np.random.SeedSequence)
            else np.random.SeedSequence(seed)
        )
        chain_seeds = ss.spawn(m)
        rngs = [np.random.default_rng(s) for s in chain_seeds]
        kernels = [
            MCMCSeqKernel(rngs[i], max_reticulations=self.priors.max_reticulations)
            for i in range(m)
        ]
        states = [self._new_state() for _ in range(m)]
        curs = [st.log_posterior() for st in states]

        cold_state = states[cold]
        map_post = curs[cold]
        map_net = _clone_net(cold_state.species_net)
        map_theta = cold_state.theta
        map_ll = cold_state.log_likelihood()

        samples: list[MCMCSeqSample] = []
        accepted = 0  # cold-chain acceptances
        swaps_accepted = 0
        swaps_proposed = 0
        start = time.time()
        completed = num_iter
        swap_rng = np.random.default_rng(ss.spawn(1)[0])

        for it in range(num_iter):
            # One MH step in every chain at its own temperature.
            for c in range(m):
                curs[c], acc = self._mh_step(
                    kernels[c], states[c], curs[c], betas[c], rngs[c]
                )
                if c == cold and acc:
                    accepted += 1
            cold_state = states[cold]

            # Track the cold chain's MAP.
            if curs[cold] > map_post:
                map_post = curs[cold]
                map_net = _clone_net(cold_state.species_net)
                map_theta = cold_state.theta
                map_ll = cold_state.log_likelihood()

            # Propose an adjacent swap on the temperature ladder.
            if m > 1 and it % swap_interval == 0:
                i = int(swap_rng.integers(0, m - 1))
                j = i + 1
                swaps_proposed += 1
                log_ratio = (betas[i] - betas[j]) * (curs[j] - curs[i])
                if math.log(swap_rng.random()) < log_ratio:
                    # Each temperature slot keeps its own kernel/RNG; only the
                    # states (and their cached log posteriors) migrate.
                    states[i], states[j] = states[j], states[i]
                    curs[i], curs[j] = curs[j], curs[i]
                    swaps_accepted += 1
                    cold_state = states[cold]

            if it >= burn_in and (it - burn_in) % sample_freq == 0:
                samples.append(
                    MCMCSeqSample(
                        iteration=it,
                        log_posterior=curs[cold],
                        log_likelihood=cold_state.log_likelihood(),
                        network_newick=_safe_newick(cold_state.species_net),
                        theta=cold_state.theta,
                        num_reticulations=cold_state.num_reticulations(),
                        gamma_major=_major_gamma(cold_state.species_net),
                    )
                )

            if progress and it % max(1, num_iter // 20) == 0:
                swap_rate = swaps_accepted / max(1, swaps_proposed)
                print(
                    f"[MC3] iter {it}/{num_iter} "
                    f"logP={curs[cold]:.4f} MAP={map_post:.4f} "
                    f"reti={cold_state.num_reticulations()} "
                    f"acc={accepted / max(1, it + 1):.3f} "
                    f"swap={swap_rate:.3f}",
                    flush=True,
                )

            if control is not None and check_every > 0 and it % check_every == 0:
                prog = {
                    "iteration": it,
                    "num_iter": num_iter,
                    "log_posterior": curs[cold],
                    "map_log_posterior": map_post,
                    "num_reticulations": cold_state.num_reticulations(),
                    "acceptance_rate": accepted / max(1, it + 1),
                    "samples": samples,
                }
                action = control(prog)
                while action == "pause":
                    time.sleep(0.2)
                    action = control(prog)
                if action == "stop":
                    completed = it + 1
                    break

        self.species_net = map_net
        self.theta = map_theta
        return MCMCSeqResult(
            map_network=map_net,
            map_log_posterior=map_post,
            map_theta=map_theta,
            samples=samples,
            acceptance_rate=accepted / max(1, completed),
            num_iterations=completed,
            wall_time_sec=time.time() - start,
            map_log_likelihood=map_ll,
            num_leaves=len(self.mapping),
            total_sites=self._total_sites(),
        )


def _safe_newick(net: Network) -> str:
    """Best-effort extended-Newick serialisation of ``net``."""
    try:
        return net.newick()
    except Exception:
        return "<unserialisable network>"


def _major_gamma(net: Network) -> Optional[float]:
    """Largest inheritance probability over all reticulations (``None`` if none)."""
    gammas: list[float] = []
    for v in net.V():
        if v.is_reticulation():
            for e in net.in_edges(v):
                g = e.get_gamma()
                if g is not None:
                    gammas.append(float(g))
    return max(gammas) if gammas else None


# ----------------------------------------------------------------------
# Posterior summaries (topology / reticulation-count), robust to the
# continuous coalescent degeneracy that makes the joint MAP unreliable.
# ----------------------------------------------------------------------

def _topology_signature(newick: str):
    """Branch-length-independent fingerprint of a sampled network.

    The signature is ``(reticulation_count, frozenset_of_clades)`` where each
    clade is the leaf-label set induced by an internal node.  Two samples with
    the same signature are the "same topology" for posterior-frequency
    purposes (ignoring branch lengths / heights / gamma).

    Returns ``None`` if the newick cannot be parsed.
    """
    try:
        net = Network.from_newick(newick)
    except Exception:
        return None
    leaves = {n.label for n in net.get_leaves()}
    n = len(leaves)
    cache: dict = {}

    def desc(v) -> frozenset:
        if v in cache:
            return cache[v]
        kids = net.get_children(v)
        if not kids:
            r = frozenset({v.label})
        else:
            acc: set = set()
            for c in kids:
                acc |= desc(c)
            r = frozenset(acc)
        cache[v] = r
        return r

    clades = set()
    n_ret = 0
    for v in net.V():
        if v.is_reticulation():
            n_ret += 1
        ds = desc(v)
        if 1 < len(ds) < n:
            clades.add(ds)
    return (n_ret, frozenset(clades))


def _reticulation_posterior(
    samples: Sequence["MCMCSeqSample"],
) -> dict[int, float]:
    """Posterior distribution over the reticulation count from ``samples``."""
    if not samples:
        return {}
    counts: dict[int, int] = {}
    for s in samples:
        counts[s.num_reticulations] = counts.get(s.num_reticulations, 0) + 1
    tot = len(samples)
    return {k: counts[k] / tot for k in sorted(counts)}


def _count_free_parameters(num_leaves: int, num_reticulations: int) -> int:
    r"""Free-parameter count ``k`` of an ultrametric species network.

    A binary time-tree on ``L`` leaves has ``L - 1`` internal node heights.
    Each reticulation adds two nodes (a hybrid and its donor) -- hence two
    heights -- plus one inheritance probability ``gamma``, i.e. three
    parameters.  One shared population size ``theta`` completes the model:

    .. math::  k = (L - 1) + 3 r + 1 = L + 3 r.

    Only differences in ``k`` across reticulation counts affect the AIC/BIC
    comparison, and each extra reticulation costs exactly ``\Delta k = 3`` --
    the penalty that decides whether its likelihood gain is worth it.
    """
    return num_leaves + 3 * num_reticulations


def _information_criteria(
    log_likelihood: float, k: int, n: Optional[int]
) -> dict[str, float]:
    r"""AIC, AICc and BIC for one fitted model.

    ``AIC = 2k - 2 logL``; ``BIC = k ln n - 2 logL``; the small-sample
    correction ``AICc = AIC + 2k(k+1) / (n - k - 1)`` (falls back to ``AIC``
    when ``n`` is unknown or ``n <= k + 1``).  Lower is better.
    """
    aic = 2.0 * k - 2.0 * log_likelihood
    out = {"log_likelihood": float(log_likelihood), "k": int(k), "AIC": aic}
    if n is not None and n > 0:
        out["n"] = int(n)
        out["BIC"] = k * math.log(n) - 2.0 * log_likelihood
        denom = n - k - 1
        out["AICc"] = aic + (2.0 * k * (k + 1) / denom) if denom > 0 else aic
    else:
        out["BIC"] = float("nan")
        out["AICc"] = aic
    return out


def _topology_posterior(
    samples: Sequence["MCMCSeqSample"], top_n: Optional[int] = None
) -> list[tuple[str, float]]:
    """Ranked ``(representative_newick, posterior_probability)`` topologies."""
    sig_count: dict = {}
    sig_rep: dict = {}
    n = 0
    for s in samples:
        sig = _topology_signature(s.network_newick)
        if sig is None:
            continue
        n += 1
        sig_count[sig] = sig_count.get(sig, 0) + 1
        sig_rep.setdefault(sig, s.network_newick)
    if n == 0:
        return []
    ranked = sorted(sig_count.items(), key=lambda kv: -kv[1])
    out = [(sig_rep[sig], cnt / n) for sig, cnt in ranked]
    return out[:top_n] if top_n else out


# ======================================================================
# Parallel multi-chain driver (process-per-chain, R-hat monitoring)
# ======================================================================
#
# Runs several independent MCMC_SEQ chains in separate processes (Python's
# GIL makes threads useless for this CPU-bound work).  Each worker pushes a
# compact status snapshot to a shared queue every ``check_every`` iterations
# and reads two shared events; the parent aggregates the snapshots, computes
# cross-chain Gelman-Rubin R-hat, and hands a :class:`MultiChainStatus` to an
# optional ``monitor`` callback.  The callback returns "continue", "pause"
# (block every chain until it stops asking to pause) or "stop" (halt every
# chain and return the partial samples gathered so far).  Ctrl-C in the parent
# also triggers a graceful stop.

# Map the human-readable trace names to MCMCSeqSample fields so the live R-hat
# uses the same columns as MCMCSeqResult.trace_table().
_TRACE_GETTERS: dict[str, Callable[["MCMCSeqSample"], float]] = {
    "posterior": lambda s: float(s.log_posterior),
    "likelihood": lambda s: float(s.log_likelihood),
    "prior": lambda s: float(s.log_posterior - s.log_likelihood),
    "theta": lambda s: float(s.theta),
    "reticulationCount": lambda s: float(s.num_reticulations),
}


def _extract_traces(
    samples: Sequence["MCMCSeqSample"], params: Sequence[str]
) -> dict[str, list[float]]:
    """Pull per-parameter value lists out of a sample list (draw order)."""
    out: dict[str, list[float]] = {}
    for p in params:
        getter = _TRACE_GETTERS.get(p)
        if getter is not None:
            out[p] = [getter(s) for s in samples]
    return out


@dataclass
class MultiChainStatus:
    """Live snapshot of a :func:`run_parallel_chains` run.

    Passed to the ``monitor`` callback every check interval.

    Attributes:
        per_chain: ``chain_id -> latest status dict`` with keys ``iteration``,
            ``num_iter``, ``log_posterior``, ``map_log_posterior``,
            ``num_reticulations``, ``acceptance_rate`` and ``n_samples``.
        rhat: ``param -> Gelman-Rubin R-hat`` across chains (``nan`` until at
            least two chains have >= 2 post-burn-in samples).  Values near 1.0
            (commonly ``< 1.05``) indicate the chains have mixed.
        elapsed_sec: Wall-clock seconds since launch.
        running: Chain ids still sampling.
        finished: Chain ids that have returned (completed or stopped).
    """

    per_chain: dict[int, dict]
    rhat: dict[str, float]
    elapsed_sec: float
    running: list[int]
    finished: list[int]

    def min_rhat_ok(self, target: float = 1.05) -> bool:
        """``True`` iff every finite R-hat is at or below ``target``."""
        finite = [v for v in self.rhat.values() if math.isfinite(v)]
        return bool(finite) and all(v <= target for v in finite)


@dataclass
class MultiChainResult:
    """Aggregate result of a :func:`run_parallel_chains` run.

    Attributes:
        chains: Per-chain :class:`MCMCSeqResult` (partial if stopped early),
            ordered by chain id.
        rhat: Final cross-chain R-hat per trace column.
        n_chains: Number of chains launched.
        stopped_early: Whether the run was halted (monitor "stop" or Ctrl-C)
            before all chains finished their iterations.
        errors: ``chain_id -> traceback`` for any chain that raised.
        wall_time_sec: Total wall-clock duration.
    """

    chains: list[MCMCSeqResult]
    rhat: dict[str, float]
    n_chains: int
    stopped_early: bool = False
    errors: dict[int, str] = field(default_factory=dict)
    wall_time_sec: float = 0.0

    def best(self) -> Optional[MCMCSeqResult]:
        """The chain with the highest MAP log posterior (``None`` if empty)."""
        if not self.chains:
            return None
        return max(self.chains, key=lambda r: r.map_log_posterior)

    def pooled_samples(self) -> list[MCMCSeqSample]:
        """All chains' post-burn-in samples concatenated."""
        out: list[MCMCSeqSample] = []
        for r in self.chains:
            out.extend(r.samples)
        return out

    def reticulation_posterior(self) -> dict[int, float]:
        """Posterior P(number of reticulations) pooled across all chains."""
        return _reticulation_posterior(self.pooled_samples())

    def topology_posterior(
        self, top_n: Optional[int] = None
    ) -> list[tuple[str, float]]:
        """Ranked ``(representative_newick, posterior_probability)`` pooled."""
        return _topology_posterior(self.pooled_samples(), top_n=top_n)

    def map_topology(self) -> Optional[str]:
        """Representative newick of the most-sampled topology across chains."""
        top = _topology_posterior(self.pooled_samples(), top_n=1)
        return top[0][0] if top else None


def _chain_worker(
    sampler: "MCMC_SEQ",
    chain_id: int,
    run_kwargs: dict,
    status_queue,
    stop_event,
    pause_event,
    check_every: int,
    rhat_params: tuple,
) -> None:
    """Worker entry point: run one chain, streaming status to the parent.

    Top-level (picklable) so it works under the Windows ``spawn`` start method.
    """
    last_iter = -1

    def control(prog: dict) -> str:
        nonlocal last_iter
        it = prog["iteration"]
        # Only push a fresh snapshot when the iteration advanced; this keeps a
        # paused chain from flooding the queue with identical frames.
        if it != last_iter:
            last_iter = it
            try:
                status_queue.put((
                    "status",
                    chain_id,
                    {
                        "iteration": it,
                        "num_iter": prog["num_iter"],
                        "log_posterior": prog["log_posterior"],
                        "map_log_posterior": prog["map_log_posterior"],
                        "num_reticulations": prog["num_reticulations"],
                        "acceptance_rate": prog["acceptance_rate"],
                        "n_samples": len(prog["samples"]),
                        "traces": _extract_traces(prog["samples"], rhat_params),
                    },
                ))
            except Exception:
                pass
        if stop_event.is_set():
            return "stop"
        if pause_event.is_set():
            return "pause"
        return "continue"

    try:
        result = sampler.search(
            control=control, check_every=check_every, **run_kwargs
        )
        status_queue.put(("result", chain_id, result))
    except Exception:
        status_queue.put(("error", chain_id, traceback.format_exc()))


def _live_rhat(
    per_chain: dict[int, dict], params: Sequence[str]
) -> dict[str, float]:
    """Cross-chain R-hat from the latest per-chain trace snapshots."""
    rhat: dict[str, float] = {}
    for p in params:
        traces = [
            st["traces"][p]
            for st in per_chain.values()
            if p in st.get("traces", {}) and len(st["traces"][p]) >= 2
        ]
        rhat[p] = gelman_rubin(traces) if len(traces) >= 2 else float("nan")
    return rhat


def _print_dashboard(status: MultiChainStatus) -> None:
    """Print a compact multi-chain status block."""
    print(
        f"\n[chains] t={status.elapsed_sec:6.1f}s  "
        f"running={len(status.running)} finished={len(status.finished)}",
        flush=True,
    )
    for cid in sorted(status.per_chain):
        st = status.per_chain[cid]
        print(
            f"  chain {cid}: it {st['iteration']}/{st['num_iter']} "
            f"logP={st['log_posterior']:.2f} MAP={st['map_log_posterior']:.2f} "
            f"reti={st['num_reticulations']} "
            f"acc={st['acceptance_rate']:.3f} n={st['n_samples']}",
            flush=True,
        )
    rhat_str = "  ".join(
        f"{p}={v:.3f}" if math.isfinite(v) else f"{p}=--"
        for p, v in status.rhat.items()
    )
    if rhat_str:
        print(f"  R-hat: {rhat_str}", flush=True)


def run_parallel_chains(
    sampler: "MCMC_SEQ",
    *,
    n_chains: int = 4,
    num_iter: int = 1_000_000,
    burn_in: int = 100_000,
    sample_freq: int = 100,
    seed: Any = None,
    check_every: int = 2_000,
    monitor: Optional[Callable[[MultiChainStatus], Optional[str]]] = None,
    rhat_params: Sequence[str] = ("posterior", "theta", "reticulationCount"),
    progress: bool = True,
    poll_interval: float = 0.5,
    shutdown_grace_sec: float = 30.0,
    temperatures: Optional[Sequence[float]] = None,
    swap_interval: int = 1,
) -> MultiChainResult:
    """Run ``n_chains`` independent MCMC_SEQ chains in parallel processes.

    Each chain is the same sampler seeded from an independent sub-stream, so
    the chains are over-dispersed and their cross-chain Gelman-Rubin R-hat
    (reported live and in the result) is a valid convergence diagnostic.

    Control: every ``check_every`` iterations each worker reports its state;
    the parent aggregates them into a :class:`MultiChainStatus` and, if
    ``monitor`` is given, calls ``monitor(status)``.  Return:

    * ``"continue"`` / ``None`` -- keep sampling (clears any pause);
    * ``"pause"``               -- block every chain in place (re-polled until
      the monitor stops returning ``"pause"``);
    * ``"stop"``                -- halt every chain and return partial results.

    A ``KeyboardInterrupt`` (Ctrl-C) in the parent also stops gracefully.

    Args:
        sampler: A fully constructed :class:`MCMC_SEQ` (must be picklable; it
            is copied to each worker).  Its ``priors.max_reticulations`` is
            honoured by each chain's default kernel.
        n_chains: Number of parallel chains / processes.
        num_iter: Iterations per chain.
        burn_in: Per-chain burn-in before sampling.
        sample_freq: Per-chain thinning interval.
        seed: Master seed; each chain gets an independent spawned sub-seed.
        check_every: Iterations between control/report checkpoints.
        monitor: Optional callback driving pause/stop (see above).
        rhat_params: Trace columns to track for live R-hat.  Valid names:
            ``posterior``, ``likelihood``, ``prior``, ``theta``,
            ``reticulationCount``.
        progress: Print a live dashboard when ``True``.
        poll_interval: Parent queue-poll cadence (seconds).
        shutdown_grace_sec: How long to wait for workers to return partial
            results after a stop before terminating them.
        temperatures: Optional MC3 temperature ladder (e.g. ``[1.0, 2.0, 3.0]``)
            forwarded to every chain; when given, each of the ``n_chains``
            processes runs a full Metropolis-coupled ensemble and contributes
            its cold chain to the cross-chain R-hat.  ``None`` runs plain
            single chains.
        swap_interval: MC3 swap cadence (iterations) forwarded to each ensemble.

    Returns:
        A :class:`MultiChainResult` with per-chain results, final R-hat, and
        any per-chain error tracebacks.
    """
    if n_chains < 1:
        raise ValueError("n_chains must be >= 1")
    if progress:
        _nm.warn_if_large_mcmc(len(sampler.mapping), method="MCMC_SEQ")

    ctx = mp.get_context("spawn")
    status_queue = ctx.Queue()
    stop_event = ctx.Event()
    pause_event = ctx.Event()

    master = (
        seed if isinstance(seed, np.random.SeedSequence)
        else np.random.SeedSequence(seed)
    )
    child_seeds = [int(s.generate_state(1)[0]) for s in master.spawn(n_chains)]
    rhat_params = tuple(rhat_params)

    procs: list = []
    for cid in range(n_chains):
        run_kwargs = dict(
            num_iter=num_iter,
            burn_in=burn_in,
            sample_freq=sample_freq,
            seed=child_seeds[cid],
            progress=False,
            temperatures=(list(temperatures) if temperatures is not None
                          else None),
            swap_interval=swap_interval,
        )
        procs.append(
            ctx.Process(
                target=_chain_worker,
                args=(
                    sampler, cid, run_kwargs, status_queue,
                    stop_event, pause_event, check_every, rhat_params,
                ),
                daemon=False,
            )
        )

    latest: dict[int, dict] = {}
    results: dict[int, MCMCSeqResult] = {}
    errors: dict[int, str] = {}
    remaining = set(range(n_chains))
    start = time.time()
    stopped_early = False

    def _handle(kind: str, cid: int, payload) -> None:
        if kind == "status":
            latest[cid] = payload
        elif kind == "result":
            results[cid] = payload
            remaining.discard(cid)
        elif kind == "error":
            errors[cid] = payload
            remaining.discard(cid)

    try:
        for p in procs:
            p.start()

        while remaining:
            # Drain at least one message (blocking briefly), then everything
            # else queued, so the monitor sees the freshest state each cycle.
            try:
                _handle(*status_queue.get(timeout=poll_interval))
                while True:
                    try:
                        _handle(*status_queue.get_nowait())
                    except _queue.Empty:
                        break
            except _queue.Empty:
                pass

            status = MultiChainStatus(
                per_chain=dict(latest),
                rhat=_live_rhat(latest, rhat_params),
                elapsed_sec=time.time() - start,
                running=sorted(remaining),
                finished=sorted(set(range(n_chains)) - remaining),
            )
            if progress:
                _print_dashboard(status)
            if monitor is not None:
                action = monitor(status)
                if action == "stop":
                    stop_event.set()
                    stopped_early = True
                elif action == "pause":
                    pause_event.set()
                else:  # "continue" / None
                    pause_event.clear()

            # Liveness guard: if every process died without sending a final
            # message (e.g. hard crash), stop waiting.
            if remaining and all(not p.is_alive() for p in procs):
                # give the queue a moment to flush, then bail
                time.sleep(0.2)
                try:
                    while True:
                        _handle(*status_queue.get_nowait())
                except _queue.Empty:
                    pass
                break
    except KeyboardInterrupt:
        stop_event.set()
        stopped_early = True
        if progress:
            print("\n[chains] Ctrl-C received -- halting chains...", flush=True)
    finally:
        deadline = time.time() + shutdown_grace_sec
        while remaining and time.time() < deadline:
            try:
                _handle(*status_queue.get(timeout=0.5))
            except _queue.Empty:
                if all(not p.is_alive() for p in procs):
                    break
        for p in procs:
            p.join(timeout=5)
        for p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

    chains = [results[c] for c in sorted(results)]
    final_rhat = _final_rhat(chains, rhat_params)
    if len(results) < n_chains:
        stopped_early = True
    return MultiChainResult(
        chains=chains,
        rhat=final_rhat,
        n_chains=n_chains,
        stopped_early=stopped_early,
        errors=errors,
        wall_time_sec=time.time() - start,
    )


def _final_rhat(
    chains: Sequence[MCMCSeqResult], params: Sequence[str]
) -> dict[str, float]:
    """Cross-chain R-hat from completed chains' full sample traces."""
    rhat: dict[str, float] = {}
    for p in params:
        traces = []
        for r in chains:
            vals = _extract_traces(r.samples, [p]).get(p, [])
            if len(vals) >= 2:
                traces.append(vals)
        rhat[p] = gelman_rubin(traces) if len(traces) >= 2 else float("nan")
    return rhat
