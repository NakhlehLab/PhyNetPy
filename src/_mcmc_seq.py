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
from .MSA import MSA
from ._seq_likelihood import (
    SubstitutionModel,
    JC69,
    FelsensteinCalculator,
    gene_tree_msnc_log_density,
    _node_height,
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
            diameters (PhyloNet default 10, on).
        use_diameter_prior: Whether the diameter prior is active.
    """

    poisson_mean: float = 1.0
    max_reticulations: int = 4
    theta_shape: float = 2.0
    theta_prior_mean: float = 0.036
    theta_mean: float = 0.02
    diameter_mean: float = 10.0
    use_diameter_prior: bool = True


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
    * (optionally) an exponential prior on reticulation-node diameters
      (the time spanned between a reticulation's two parents),
    * a Uniform[0,1] prior on each inheritance probability (contributes 0).

    Args:
        species_net: The species network.
        theta: Population mutation rate.
        priors: Hyperparameters.

    Returns:
        The log prior (``-inf`` if the reticulation cap is exceeded).
    """
    reticulations = [v for v in species_net.V() if v.is_reticulation()]
    n_ret = len(reticulations)
    if n_ret > priors.max_reticulations:
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

    def invalidate_locus(self, i: int) -> None:
        """Drop cached factors for locus ``i`` (after a gene-tree move)."""
        self._felsen[i] = None
        self._msnc[i] = None

    def invalidate_network(self) -> None:
        """Drop all MSNC caches (after a network / theta move)."""
        self._msnc = [None] * self.n_loci

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
        total = 0.0
        for i in range(self.n_loci):
            if self._felsen[i] is None:
                self._felsen[i] = self._calcs[i].log_likelihood(
                    gene_trees[i], self._model
                )
            if self._msnc[i] is None:
                self._msnc[i] = gene_tree_msnc_log_density(
                    gene_trees[i], species_net, self._species_of, theta=theta
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

    new_net = copy.deepcopy(old_net)
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

    new_net = copy.deepcopy(old_net)
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

    new_gt = copy.deepcopy(old_gt)
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

    new_gt = copy.deepcopy(old_gt)
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


def op_add_reticulation(
    state: SeqState, rng: np.random.Generator
) -> Optional[tuple[float, Callable]]:
    """RJMCMC add-reticulation move (mirrors PhyloNet ``AddReticulation``).

    Picks two distinct edges uniformly: a *donor* edge (``v3 -> v4``, split at
    height ``t1`` by a new tree node ``v1``) and a *reticulation* edge
    (``v5 -> v6``, split at height ``t2`` by the new reticulation node ``v2``).
    A hybrid edge ``v1 -> v2`` is added with ``gamma ~ U(0,1)``; ``v2``'s
    original in-edge keeps ``1 - gamma``.  The proposal is rejected when it
    would create a cycle or a parallel (bubble) edge.  The log Hastings ratio
    matches PhyloNet exactly::

        log( pda * l1 * l2 * E*(E-1) / (2*(R+1)) )

    where ``E`` is the pre-move edge count, ``l1/l2`` the donor/reticulation
    edge lengths, ``R`` the pre-move reticulation count and ``pda = 0.5`` only
    for the first reticulation (``R == 0``), else ``1.0``.  This is the exact
    negative of :func:`op_delete_reticulation`'s ratio, so add/delete are
    reversible.
    """
    R = state.num_reticulations()
    if R >= state.priors.max_reticulations:
        return None
    old_net = state.species_net
    old_heights = state.net_heights
    snap = state._engine.clone_caches()

    new_net = copy.deepcopy(old_net)
    new_heights = _heights(new_net)

    edges = [e for e in new_net.E()]
    E = len(edges)
    if E < 2:
        return None
    ia, ib = rng.choice(E, size=2, replace=False)
    e1, e2 = edges[int(ia)], edges[int(ib)]
    v3, v4 = e1.src, e1.dest          # donor edge (insert v1)
    v5, v6 = e2.src, e2.dest          # reticulation edge (insert v2)
    l1 = new_heights[v3] - new_heights[v4]
    l2 = new_heights[v5] - new_heights[v6]
    if l1 <= 0.0 or l2 <= 0.0:
        return None
    t1 = new_heights[v4] + l1 * rng.random()
    t2 = new_heights[v6] + l2 * rng.random()
    # The hybrid edge v1 -> v2 needs the donor point above the reticulation.
    if t1 <= t2:
        return None

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

    # Reject bubbles (parallel edges) and any cycle the hybrid edge created.
    if _has_parallel_edges(new_net):
        return None
    try:
        if not new_net.is_acyclic():
            return None
    except Exception:
        return None

    # Shared, single-source RJMCMC math (see phynetpy._network_moves); the
    # guards above guarantee E >= 2 and positive l1/l2 so this never raises.
    loghr = _nm.add_reticulation_log_hastings(
        edge_count_pre=E, retic_count_pre=R, l1=l1, l2=l2
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
    """RJMCMC delete-reticulation move (mirrors PhyloNet ``DeleteReticulation``).

    Picks a reticulation node ``v2`` and one of its parents ``v1`` (the donor)
    uniformly, removes the hybrid edge ``v1 -> v2``, then suppresses the two
    resulting degree-2 nodes -- merging ``v3 -> v1 -> v4`` into ``v3 -> v4`` and
    ``v5 -> v2 -> v6`` into ``v5 -> v6`` (exactly the two edges a reverse add
    would re-split).  The proposal is rejected for the same degenerate
    configurations PhyloNet forbids (``v1`` not a tree node; the merge would
    create a parallel/bubble edge; the fully-degenerate case).  The log
    Hastings ratio matches PhyloNet exactly::

        log( pad * 2*R / (l1 * l2 * E'*(E'-1)) )

    where ``R`` is the pre-move reticulation count, ``E'`` the *post*-move edge
    count, ``l1/l2`` the merged-edge lengths and ``pad = 2`` only when removing
    the last reticulation (``R == 1``), else ``1.0``.  This is the exact
    negative of :func:`op_add_reticulation`'s ratio.
    """
    R = state.num_reticulations()
    if R == 0:
        return None
    old_net = state.species_net
    old_heights = state.net_heights
    snap = state._engine.clone_caches()

    new_net = copy.deepcopy(old_net)
    new_heights = _heights(new_net)

    rets = [v for v in new_net.V() if v.is_reticulation()]
    v2 = rets[int(rng.integers(len(rets)))]
    parents = new_net.get_parents(v2)
    if len(parents) != 2:
        return None
    v1 = parents[int(rng.integers(2))]
    v5 = parents[1] if parents[0] is v1 else parents[0]

    # v1 must be an interior tree node (one parent, the other child distinct).
    if v1.is_reticulation() or new_net.in_degree(v1) == 0:
        return None
    v3s = new_net.get_parents(v1)
    if len(v3s) != 1:
        return None
    v3 = v3s[0]
    other_children = [c for c in new_net.get_children(v1) if c is not v2]
    if len(other_children) != 1:
        return None
    v4 = other_children[0]
    v6s = new_net.get_children(v2)
    if len(v6s) != 1:
        return None
    v6 = v6s[0]

    # Forbid merges that would create a parallel/bubble edge, or the fully
    # degenerate case (matches PhyloNet's hasParent rejections).
    if v3 in new_net.get_parents(v4):
        return None
    if v5 in new_net.get_parents(v6):
        return None
    if v3 is v5 and v4 is v6:
        return None

    l1 = new_heights[v3] - new_heights[v4]
    l2 = new_heights[v5] - new_heights[v6]
    if l1 <= 0.0 or l2 <= 0.0:
        return None

    try:
        del_edge = _edge_between(new_net, v1, v2)
        new_net.remove_edge(del_edge)
        v2.set_is_reticulation(False)
        _suppress_degree2(new_net, new_heights, v1)
        _suppress_degree2(new_net, new_heights, v2)
    except Exception:
        return None

    E_after = len([e for e in new_net.E()])
    if E_after < 2:
        return None
    # Shared, single-source RJMCMC math -- exact negative of the add ratio.
    loghr = _nm.remove_reticulation_log_hastings(
        edge_count_post=E_after, retic_count_pre=R, l1=l1, l2=l2
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
    ) -> None:
        """Build the kernel.

        Args:
            rng: Random generator.
            max_reticulations: Reticulation cap (disables add when reached).
        """
        self.rng = rng
        self.max_reticulations = max_reticulations
        # (operator, weight) -- continuous gene-tree/network moves dominate;
        # topology + dimension changes are rarer, as in PhyloNet.
        self._ops: list[tuple[Callable, float]] = [
            (op_gene_node_height, 0.30),
            (op_gene_tree_nni, 0.20),
            (op_net_node_height, 0.20),
            (op_change_gamma, 0.07),
            (op_change_theta, 0.10),
            (op_add_reticulation, 0.065),
            (op_delete_reticulation, 0.065),
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
    """

    iteration: int
    log_posterior: float
    log_likelihood: float
    network_newick: str
    theta: float
    num_reticulations: int


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
            copy.deepcopy(self.species_net),
            [copy.deepcopy(gt) for gt in self.gene_trees],
            self.species_of,
            self.loci,
            self.priors,
            self.model,
            self.theta,
        )

    def score(self) -> float:
        """Log posterior of the current (starting) state."""
        return self._new_state().log_posterior()

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

        Returns:
            An :class:`MCMCSeqResult`; ``self.species_net`` / ``self.theta`` are
            updated in place to the MAP estimate.  When ``control`` requests an
            early stop the result holds the samples gathered up to that point
            and ``num_iterations`` reflects the iterations actually run.
        """
        _nm.warn_if_large_mcmc(len(self.mapping), method="MCMC_SEQ")

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
        map_net = copy.deepcopy(state.species_net)
        map_theta = state.theta

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
                            map_net = copy.deepcopy(state.species_net)
                            map_theta = state.theta
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
        map_net = copy.deepcopy(cold_state.species_net)
        map_theta = cold_state.theta

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
                map_net = copy.deepcopy(cold_state.species_net)
                map_theta = cold_state.theta

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
        )


def _safe_newick(net: Network) -> str:
    """Best-effort extended-Newick serialisation of ``net``."""
    try:
        return net.newick()
    except Exception:
        return "<unserialisable network>"


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
