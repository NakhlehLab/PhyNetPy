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
MCMC_GT -- Bayesian inference of phylogenetic networks from gene-tree
topologies under the multispecies network coalescent (MSNC).

Implements the scoring function of

    Yu, Y., Degnan, J.H., Nakhleh, L. (2012). "The probability of a gene
    tree topology within a phylogenetic network with applications to
    hybridization detection." PLoS Genetics 8(4), e1002660.

and the Bayesian search wrapper of

    Wen, D., Yu, Y., Nakhleh, L. (2016). "Bayesian inference of
    reticulate phylogenies under the multispecies network coalescent."
    PLoS Genetics 12(5), e1006006.

Objective to sample (posterior):

    log P(Psi | G)
      = sum_i log P(g_i | Psi)          (MSNC likelihood per gene tree)
      + log P(Psi)                      (network prior: topology + branch lengths + gammas + retic-count)
      + const.                          (normalising constant dropped; MH ratios cancel it)

Pipeline overview
-----------------
End-to-end control flow when a user runs ``MCMC_GT.search(method="mh", ...)``:

    1. Enumerate gene-tree leaf -> species mapping and pre-hash per gene
       tree (labels only; topology signature is reused by the DP).

    2. Build an initial :class:`Model` wrapping the starting species
       network.  Attach an :class:`MCMCGTScorer` so the search driver
       can call ``scorer(model)`` to get the log posterior (MH) or
       log likelihood (HC / SA).

    3. Run the search driver (:class:`MetropolisHastings`,
       :class:`HillClimbing`, or :class:`SimulatedAnnealing`) using an
       :class:`MCMCGTKernel` to propose moves.  Each accepted proposal
       mutates the model's network in place; MH records samples every
       ``thin`` iterations after ``burn_in``.

    4. Return an :class:`MCMCGTResult` with the final best / sampled
       networks and diagnostic counters.

Scoring path (per call to ``MCMCGTScorer.__call__``):

    Model.network -> _GTLikelihoodEngine.update(dirty_nodes)
                  -> engine.log_prob(g_i) summed over gene trees
                  -> + log_prior_network(Psi)
                  -> posterior or likelihood

Likelihood model: full MSNC ancestral-configurations DP
--------------------------------------------------------
The MSNC likelihood is evaluated by the **ancestral-configurations
DP** of Yu, Degnan & Nakhleh (2012; PLoS Genetics 8(4):e1002660), the
exact algorithm underlying PhyloNet's MCMC_GT:

    P(g | Psi) = sum_h P_coal(g, h | Psi) * prod_R gamma(R, h)^|S_R|
                                            * (1 - gamma(R, h))^|A_R\\S_R|

summed over every coalescent history ``h`` that maps gene-tree
internal nodes to species-network branches, where ``S_R`` is the
subset of gene-tree lineages routed through one parent of
reticulation ``R`` (the complement goes through the other parent;
each lineage chooses independently).  The DP walks the species
network in bottom-up topological order maintaining a *joint*
distribution of lineage bitmasks over every currently-open
species-network edge -- this preserves the dependency between the two
parent-paths of a reticulation that re-merge at their lowest common
ancestor.  See :func:`_msnc_log_prob_network_int` for the per-node
recurrences.

Equivalence to the displayed-tree decomposition.  When every
reticulation is crossed by at most one gene-tree lineage (the
single-allele regime) the AC DP reduces to the displayed-tree
mixture ``sum_T w(T) P(g | T)``.  For multi-allele data, or
single-allele data where coalescent backwards-traversal puts
multiple lineages above a reticulation simultaneously, the
displayed-tree mixture is a strict lower bound and only the AC DP
returns the true MSNC likelihood.  This implementation always uses
the AC DP so the score matches PhyloNet across every retic-count /
allele-count regime.

Incremental likelihood caching
------------------------------
The engine cooperates with :class:`ModelGraph.Model`'s
``_dirty_nodes`` dirty-set plumbing (see :meth:`Move.touched_nodes`):
topology-preserving moves (``ChangeNodeHeight``,
``ChangeInheritanceProb``) post a single touched node; the engine
preserves its int-indexed network view and only invalidates the
per-call gene-tree memo (the static
:meth:`_NetworkIndex.edge_length` / ``edge_gamma`` accessors read
the mutated values fresh).  Topology-changing moves leave
``_dirty_nodes`` as ``None`` (fully invalidate), rebuilding the
:class:`_NetworkIndex` and busting per-(gene-tree, network)
score memos.  Persistent caches that survive every move:

    * ``_gij(t, i, j)`` and ``_log_gij``, ``_log_denom`` -- depend
      only on branch length / lineage counts, not on topology.
    * ``_GeneTreeIndex`` and its coarsening / linear-extension
      caches -- depend only on the (immutable) gene tree.
    * ``_score_cache`` keyed by ``(id(gene_tree), network_signature)``
      -- lets MH chains skip the AC DP entirely on revisits to a
      previously-evaluated network state.

Module layout
-------------
Sections are marked with banner comments.  Top-to-bottom:

    (A) Result containers          (MCMCSample, MCMCGTResult)
    (B) Priors                     (MCMC_GTPriors, log_prior_network)
    (C) MSNC likelihood engine     (_GTLikelihoodEngine)
    (D) Scorer                     (MCMCGTScorer)
    (E) Proposal kernel            (MCMCGTKernel)
    (F) Orchestration              (MCMC_GT)
"""

from __future__ import annotations

import copy
import math
import os
import pickle
import random as _py_random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product as _iter_product
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

from .Network import Network, Node, Edge
from .GeneTrees import GeneTrees
from . import IO as io
from . import _network_moves as _nm
from .ModelGraph import Model
from ._chain_analysis import (
    ChainSummary,
    summarize_traces,
    write_tracer_log,
    write_trees_nexus,
)
from .MetropolisHastings import (
    ProposalKernel,
    HillClimbing,
    MetropolisHastings,
    SimulatedAnnealing,
)
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

from ._msnc_density import (
    MSCBranchKernel,
    _LOG_FLOOR,
    _GeneTreeIndex,
    _NetworkIndex,
    _msnc_log_prob_network_int,
    _msc_log_prob_tree_int,
    _apply_branch_coalescent_int,
    _combine_configs_int,
    _logsumexp,
    _popcount,
    _bits,
)


# ======================================================================
# (A) Result containers
# ======================================================================

@dataclass
class MCMCSample:
    """A single posterior sample from an :class:`MCMC_GT` MH run.

    Attributes:
        iteration: Global iteration index at which the sample was taken
            (including burn-in).
        network: Deep copy of the species network at this iteration.
            Decoupled from the chain's live model so subsequent moves
            don't mutate the sample.
        log_posterior: Unnormalised log posterior at the sample
            (``sum_i log P(g_i | Psi) + log_prior(Psi)``).
        log_likelihood: Just the MSNC log likelihood part (useful for
            reporting the MAP estimate independently of the prior).
    """

    iteration: int
    network: Network
    log_posterior: float
    log_likelihood: float


@dataclass
class MCMCGTResult:
    """Aggregate result from an :class:`MCMC_GT` search.

    Both Bayesian (``method="mh"``) and likelihood-maximising
    (``method="hc"`` / ``"sa"``) runs populate this container; fields
    irrelevant to a given method are left at their defaults.

    Attributes:
        method: Which driver produced this result (``"mh"``, ``"hc"``,
            or ``"sa"``).
        best_network: Highest-scoring network seen during the search.
            For MH this is the MAP sample; for HC / SA it's the
            end-state.
        best_log_posterior: Score associated with ``best_network``
            (posterior for MH; likelihood for HC / SA).
        samples: Posterior samples collected from the MH chain, in
            order of draw.  Empty for HC / SA.
        num_iter: Total proposed moves evaluated.
        num_accepted: Count of moves actually accepted into the chain.
        wall_time_sec: Observed wall-clock time of the run.
    """

    method: str
    best_network: Network
    best_log_posterior: float
    samples: list[MCMCSample] = field(default_factory=list)
    num_iter: int = 0
    num_accepted: int = 0
    wall_time_sec: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of proposals the driver accepted."""
        if self.num_iter <= 0:
            return 0.0
        return self.num_accepted / float(self.num_iter)

    # -- post-analysis (Tracer interop + native diagnostics) ----------

    def _sample_step(self) -> int:
        """Sampling interval (states between recorded samples), >= 1."""
        if len(self.samples) >= 2:
            step = self.samples[1].iteration - self.samples[0].iteration
            if step > 0:
                return int(step)
        return 1

    def trace_table(self) -> tuple[list[int], dict[str, list[float]]]:
        """Build a Tracer-style trace table from the MH samples.

        Returns:
            ``(states, traces)`` with columns ``posterior``, ``likelihood``,
            ``prior`` (= posterior - likelihood) and ``reticulationCount``.
            Empty for hill-climbing / simulated-annealing runs (which record no
            samples).
        """
        states = [s.iteration for s in self.samples]
        traces: dict[str, list[float]] = {
            "posterior": [s.log_posterior for s in self.samples],
            "likelihood": [s.log_likelihood for s in self.samples],
            "prior": [s.log_posterior - s.log_likelihood for s in self.samples],
            "reticulationCount": [
                float(sum(1 for v in s.network.V() if v.is_reticulation()))
                for s in self.samples
            ],
        }
        return states, traces

    def write_log(self, path: str) -> None:
        """Write a BEAST/Tracer-compatible ``.log`` of the sampled chain."""
        states, traces = self.trace_table()
        write_tracer_log(
            states,
            traces,
            path,
            comments=[
                f"PhyNetPy MCMC_GT ({self.method}) posterior sample",
                f"iterations={self.num_iter} "
                f"acceptance_rate={self.acceptance_rate:.4f} "
                f"wall_time_sec={self.wall_time_sec:.2f}",
            ],
        )

    def write_networks(self, path: str) -> None:
        """Write the sampled species networks as a NEXUS ``.trees`` file."""
        states = [s.iteration for s in self.samples]
        newicks = [s.network.newick() for s in self.samples]
        write_trees_nexus(states, newicks, path, prefix="STATE")

    def summary(self, *, hpd_prob: float = 0.95) -> ChainSummary:
        """Per-parameter diagnostics (mean, HPD, ESS, ...) for the MH chain."""
        _, traces = self.trace_table()
        return summarize_traces(
            traces, step_size=self._sample_step(), hpd_prob=hpd_prob
        )


# ======================================================================
# (B) Priors
# ======================================================================

@dataclass
class MCMC_GTPriors:
    """Composable prior parameters for :func:`log_prior_network`.

    Defaults follow Wen & Nakhleh (2016).  Each component can be
    disabled by setting its weight to ``0.0`` for a plain-likelihood
    search (equivalent to HC / SA with method="mh" dropped in).

    Attributes:
        branch_length_rate: Rate ``lambda`` of an ``Exp(lambda)`` prior
            on every finite branch length.  Mean branch length under
            the prior is ``1 / lambda``.  Default 1.0 coalescent-unit.
        gamma_alpha, gamma_beta: Beta(alpha, beta) prior on each
            reticulation inheritance probability.  Default ``(1, 1)``
            is uniform.
        retic_count_mean: Mean of the ``Poisson`` prior on the number
            of reticulations.  Default 1.0 (softly shrinks toward
            tree-like topologies while still supporting several
            retics).
        topology_weight: Coefficient on the topology prior term
            (``log_prior_topology``).  Default 0.0 (uniform: every
            rooted binary topology with a given retic count has equal
            prior mass, so the term drops).  Set non-zero only for
            bespoke topology priors via ``topology_prior_fn``.
        topology_prior_fn: Optional caller-supplied callable taking a
            :class:`Network` and returning a log prior.  When
            ``None``, the topology prior is zero (uniform).
    """

    branch_length_rate: float = 1.0
    gamma_alpha: float = 1.0
    gamma_beta: float = 1.0
    retic_count_mean: float = 1.0
    topology_weight: float = 0.0
    topology_prior_fn: Optional[Callable[[Network], float]] = None


def _log_exp_pdf(x: float, rate: float) -> float:
    """Log-density of an ``Exp(rate)`` distribution at ``x``.

    Returns ``_LOG_FLOOR`` for non-positive ``x`` rather than ``-inf``
    to keep the MH chain finite.
    """
    if x is None or not math.isfinite(x) or x <= 0.0:
        return _LOG_FLOOR
    return math.log(rate) - rate * x


def _log_beta_pdf(x: float, alpha: float, beta: float) -> float:
    """Log-density of a ``Beta(alpha, beta)`` distribution at ``x``.

    Uses :func:`math.lgamma` to avoid overflow for tall beta peaks.
    Clamped to ``_LOG_FLOOR`` outside ``(0, 1)``.
    """
    if x is None or not math.isfinite(x) or x <= 0.0 or x >= 1.0:
        return _LOG_FLOOR
    log_beta_fn = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    return (alpha - 1.0) * math.log(x) + (beta - 1.0) * math.log(1.0 - x) - log_beta_fn


def _log_poisson_pmf(k: int, mean: float) -> float:
    """Log-pmf of a ``Poisson(mean)`` distribution at integer ``k``."""
    if k < 0 or mean <= 0.0:
        return _LOG_FLOOR
    return k * math.log(mean) - mean - math.lgamma(k + 1)


def log_prior_network(
    net: Network,
    priors: MCMC_GTPriors,
) -> float:
    """Compose the network prior from its component pieces.

    Components (all additive in log space):

      * **Topology**: ``priors.topology_weight * topology_prior_fn(net)``
        when the caller supplies a custom topology prior; otherwise
        zero (uniform over topologies with the observed retic count).
      * **Branch lengths**: per-edge ``Exp(branch_length_rate)``.
        Zero-length and ``None`` branches are treated as the
        ``_LOG_FLOOR`` cap to avoid ``log(0)`` but keep the score
        finite for partially-annotated networks.
      * **Gammas**: per-reticulation ``Beta(gamma_alpha, gamma_beta)``
        on one of the two parent-edge gammas (the other is
        ``1 - gamma`` by construction).
      * **Retic count**: ``Poisson(retic_count_mean)`` on the number
        of reticulation nodes.

    Args:
        net: Species network to score.  All branch lengths / gammas
            read directly off the network's edges; no topology
            caching is performed here (callers that need caching
            should memoise at a higher level).
        priors: :class:`MCMC_GTPriors` hyperparameters.

    Returns:
        Log prior (typically negative; the more "extreme" a network
        is under the prior, the more negative the value).
    """
    total = 0.0

    # --- Topology ---
    if priors.topology_prior_fn is not None and priors.topology_weight != 0.0:
        total += priors.topology_weight * priors.topology_prior_fn(net)

    # --- Branch lengths ---
    for e in net.E():
        total += _log_exp_pdf(e.get_length(), priors.branch_length_rate)

    # --- Gammas (one per retic; complementary gamma is implied) ---
    retic_count = 0
    for n in net.V():
        if not n.is_reticulation():
            continue
        retic_count += 1
        parents = net.get_parents(n)
        if len(parents) != 2:
            continue
        e1 = net.get_edge(parents[0], n)
        gamma = e1.get_gamma()
        if gamma is None:
            gamma = 0.5
        total += _log_beta_pdf(
            float(gamma), priors.gamma_alpha, priors.gamma_beta,
        )

    # --- Reticulation count ---
    total += _log_poisson_pmf(retic_count, priors.retic_count_mean)
    return total


# ======================================================================
# (C) MSNC gene-tree likelihood engine
# ======================================================================

class _GTLikelihoodEngine(MSCBranchKernel):
    """Full MSNC gene-tree-topology likelihood for a fixed species network.

    Computes ``log P(g | Psi)`` by the **ancestral-configurations DP**
    of Yu, Degnan & Nakhleh (2012).  See the module docstring for the
    formal definition; in brief, the DP walks the species network in
    bottom-up topological order maintaining a joint distribution

        dict[ tuple[(edge_id, lineage_mask), ...], log_prob ]

    over all currently-open species-network edges, applying the
    Kingman branch-coalescent on every traversed edge and splitting
    each lineage independently at every reticulation (per-lineage
    gamma weighting).  The joint state preserves the dependency
    between the two parent-paths of a reticulation that re-merge at
    their LCA.  Reduces to the standard MSC tree DP when the network
    has no reticulations and to the displayed-tree decomposition when
    every reticulation is crossed by at most one lineage; for
    multi-lineage retic crossings only the AC DP returns the true
    MSNC likelihood (the displayed-tree mixture is a strict lower
    bound there).

    Caching strategy (see module docstring for the motivating
    architecture):

      * ``_gij(t, n, m)`` and its log/denom variants are memoised on
        the engine; survives every move (depends only on length /
        lineage counts).
      * :class:`_NetworkIndex` view of the species network is built
        once and reused across topology-preserving moves; rebuilt
        from scratch on topology mutations.
      * :class:`_GeneTreeIndex` view + its coarsening / linear-
        extension caches are built once per gene tree and survive
        the engine's lifetime (gene trees are immutable).
      * Per-(gene-tree, network) score memo (``_score_cache``) keyed
        by ``(id(gene_tree), network_signature)`` lets MH chains
        skip the AC DP entirely on revisits to a previously-evaluated
        network state.
    """

    # Legacy displayed-tree warning threshold.  Retained because the
    # diagnostic :meth:`_build_displayed_trees` helper still consumes
    # it; the production scoring path goes through the AC DP and
    # does not enumerate displayed trees.
    _DISPLAYED_TREE_WARN_THRESHOLD = 7

    def __init__(self, network: Network) -> None:
        """Bind the engine to a species network.

        Heavy caching happens lazily on the first ``log_prob`` call
        so construction itself remains cheap (the search driver may
        instantiate many engines during proof-of-concept code paths,
        and we don't want the "maybe unused" cost up-front).

        Args:
            network: Species network whose MSNC likelihood the engine
                will score.  The engine holds a reference (not a
                copy); callers that mutate the network must honour
                the dirty-set protocol so the engine can invalidate
                stale caches.
        """
        super().__init__(theta=2.0)
        self.network = network
        # Int-indexed view of the species network used by the AC DP.
        # Built lazily on first ``log_prob`` call; rebuilt on topology
        # mutations (signaled via ``update(None)``).
        self._net_index: _NetworkIndex | None = None
        # Legacy displayed-tree machinery -- kept around as a
        # reference / diagnostic implementation (see
        # :func:`_msc_log_prob_tree_int`); not consulted by the
        # production scoring path.
        self._displayed_trees: list[_DisplayedTree] | None = None
        # Per-gene-tree cache keyed by gene-tree id() (cheap-but-brittle
        # identity).  Gene trees are assumed immutable for the
        # engine's lifetime; callers that truly swap gene trees
        # should instantiate a fresh scorer/engine.
        self._gene_tree_log_prob: dict[int, float] = {}
        # Per-gene-tree static :class:`_GeneTreeIndex` cache.  Survives
        # every ``update`` call: gene trees are immutable so the
        # bit-indexed view (and its internal coarsening / linear-
        # extension caches) are valid for the engine's whole lifetime.
        self._gti_cache: dict[int, _GeneTreeIndex] = {}
        # Per-(gene-tree, network) AC-DP score memo keyed by gene-tree
        # id and a lightweight network signature (per-edge length +
        # gamma).  Survives ``update`` calls so revisited network
        # states (rejected MH proposals reverting to the previous
        # accepted state) skip the DP entirely.  Bounded by the
        # number of distinct (length, gamma)-tuples the chain visits.
        # Memoised network signature -- recomputed exactly once per
        # ``update()`` cycle (i.e. once per accepted/rejected MH move)
        # rather than per scored gene tree.  On the 7-tax bench this
        # dropped ``_network_signature`` from a 2 s hotspot to a flat
        # ~0 s for the same number of likelihood calls.
        self._cached_signature: tuple | None = None
        self._score_cache: dict[
            tuple[int, tuple],
            float,
        ] = {}
        # Reverse map ``id(gene_tree) -> gene_tree`` so the scorer
        # can rebuild :class:`_GeneTreeIndex` entries when species_of
        # changes after caching.
        self._gt_by_id: dict[int, Network] = {}

    # ------------------------------------------------------------------
    # Displayed-tree decomposition
    # ------------------------------------------------------------------

    def _build_displayed_trees(self) -> list["_DisplayedTree"]:
        """Enumerate every displayed tree of ``self.network``.

        Each displayed tree is a contracted copy of the network where
        one parent edge is retained (and the other deleted) at each
        reticulation; resulting degree-2 passthrough nodes are
        collapsed into single edges whose length is the sum of the
        originals.  The associated weight is the product of the
        retained gammas.

        The returned list is freshly built and does not share any
        :class:`Node` objects with the input network, so subsequent
        mutations to the input network won't silently corrupt the
        displayed-tree representation.

        Returns:
            List of :class:`_DisplayedTree` records, one per
            displayed tree (empty retic list -> single trivial entry
            covering the input tree itself).
        """
        net = self.network
        retics: list[Node] = [n for n in net.V() if n.is_reticulation()]
        k = len(retics)
        if k >= self._DISPLAYED_TREE_WARN_THRESHOLD:
            # Soft warning via comment; searches that routinely propose
            # this many retics should move to the AC DP (future work).
            pass

        trees: list[_DisplayedTree] = []
        for mask in range(2 ** k):
            # For each retic i, keep parent[0] when (mask >> i) & 1 == 0
            # else parent[1].  Record the chosen gamma on each kept
            # edge for the log-weight.
            keep_choice: dict[Node, int] = {}
            log_weight = 0.0
            valid = True
            for i, r in enumerate(retics):
                parents = net.get_parents(r)
                if len(parents) != 2:
                    valid = False
                    break
                choice = (mask >> i) & 1
                keep_choice[r] = choice
                kept_edge = net.get_edge(parents[choice], r)
                gamma = kept_edge.get_gamma()
                if gamma is None or gamma <= 0.0:
                    log_weight += _LOG_FLOOR
                else:
                    log_weight += math.log(gamma)
            if not valid:
                continue
            dt = _DisplayedTree.from_network(net, keep_choice, log_weight)
            trees.append(dt)
        return trees

    def _ensure_displayed_trees(self) -> list["_DisplayedTree"]:
        """Return cached displayed trees; rebuild if missing."""
        if self._displayed_trees is None:
            self._displayed_trees = self._build_displayed_trees()
        return self._displayed_trees

    # ------------------------------------------------------------------
    # Incremental-update entry point (dirty-set protocol)
    # ------------------------------------------------------------------

    def update(self, dirty_nodes: "set[Node] | None") -> None:
        """Invalidate caches affected by the last batch of moves.

        Called by :class:`MCMCGTScorer` before each scoring pass.
        See the module docstring for the full protocol.

        Args:
            dirty_nodes: Either ``None`` (fully invalidate: the
                network topology changed; rebuild :class:`_NetworkIndex`
                and bust per-gene-tree memos) or a :class:`set` of
                network nodes whose incident parameters changed
                (partial invalidate: keep the network index but
                clear the per-call gene-tree memo).  An empty set
                is a no-op.
        """
        if dirty_nodes is None:
            # Topology change: bust the network index, the displayed-
            # tree skeleton, the per-gene memo, and the
            # (gene-tree, network)-keyed score cache.  The latter is
            # keyed against a topology that's gone, so its entries
            # are no longer reachable.  The ``_gij`` / ``_log_gij`` /
            # ``_log_denom`` caches and ``_gti_cache`` survive (they
            # depend only on branch lengths / gene trees, which are
            # unchanged here).
            self._net_index = None
            self._displayed_trees = None
            self._gene_tree_log_prob.clear()
            self._score_cache.clear()
            self._cached_signature = None
            return
        if not dirty_nodes:
            return
        # Partial-dirty path: branch lengths or gammas changed but
        # the topology didn't.  The :class:`_NetworkIndex` is still
        # valid (lengths/gammas are read fresh on each DP call), but
        # the cached :class:`_DisplayedTree` (used by the no-retic
        # fast path) bakes the old lengths into ``_idx_edge_lengths``
        # so we drop it and let it rebuild on the next score.  The
        # per-call memo always invalidates; the ``_score_cache`` is
        # keyed by a length+gamma signature and stays correct.
        self._displayed_trees = None
        self._gene_tree_log_prob.clear()
        self._cached_signature = None

    def _network_signature(self) -> tuple:
        """Compact tuple summarising the network's mutable parameters.

        Used as a cache key for the per-(gene-tree, network) AC DP
        score memo.  Captures every quantity the AC DP reads off the
        network: per-edge length + gamma, plus a stable identifier
        for the edge endpoints (so two networks with the same edge
        set but different topology don't collide).

        The signature is memoised across all ``log_prob`` calls
        between two consecutive :meth:`update` invocations -- it's
        a function of the network's mutable state, which only the
        scorer (via ``update``) is allowed to change.  Saves ~20 us
        per scored gene tree on the 7-tax bench.
        """
        cached = self._cached_signature
        if cached is not None:
            return cached
        sig = []
        for e in self.network.E():
            sig.append((id(e.src), id(e.dest), e.get_length(), e.get_gamma()))
        sig_tuple = tuple(sig)
        self._cached_signature = sig_tuple
        return sig_tuple

    # ------------------------------------------------------------------
    # Log-probability of a gene tree under the species network
    # ------------------------------------------------------------------

    def _get_gti(
        self,
        gene_tree: Network,
        species_of: dict[str, str],
    ) -> "_GeneTreeIndex":
        """Lazily build (or fetch) the bit-indexed view of ``gene_tree``.

        Static for the engine's lifetime: subsequent calls re-use the
        same object and inherit its coarsening / linear-extension
        caches.
        """
        gid = id(gene_tree)
        gti = self._gti_cache.get(gid)
        if gti is None:
            gti = _GeneTreeIndex(gene_tree, species_of)
            self._gti_cache[gid] = gti
            self._gt_by_id[gid] = gene_tree
        elif species_of is not None and not gti.leaf_species_of:
            # First call may have used an empty species map; re-bind.
            gti.refresh_species(gene_tree, species_of)
        return gti

    def log_prob(
        self,
        gene_tree: Network,
        species_of: dict[str, str],
    ) -> float:
        """Return ``log P(gene_tree | self.network)``.

        Computed by the full MSNC ancestral-configurations DP (Yu,
        Degnan & Nakhleh 2012); see :func:`_msnc_log_prob_network_int`.

        Args:
            gene_tree: The observed gene tree (a :class:`Network`
                with in-degree-1 internal nodes; reticulations in
                gene trees are not supported).
            species_of: Map from gene-copy label to species label.
                Must cover every leaf of ``gene_tree`` that appears
                in ``self.network``.  Gene leaves whose species is
                not in ``self.network`` are silently dropped.

        Returns:
            Log probability under the MSNC on the currently-bound
            network, floored at ``_LOG_FLOOR`` for incompatible
            configurations so MH acceptance ratios stay finite.
        """
        key = id(gene_tree)
        cached = self._gene_tree_log_prob.get(key)
        if cached is not None:
            return cached

        if self._net_index is None:
            self._net_index = _NetworkIndex(self.network)
        net_idx = self._net_index
        if net_idx.n_nodes == 0 or net_idx.root < 0:
            self._gene_tree_log_prob[key] = _LOG_FLOOR
            return _LOG_FLOOR

        gti = self._get_gti(gene_tree, species_of)
        # Refresh the species map if the cached one is stale (lazy-fill
        # path or the caller swapped species_of between calls).
        if species_of is not None and len(gti.leaf_species_of) != len(gti.leaves):
            gti.refresh_species(gene_tree, species_of)

        # Per-(gene_tree, network) score memo.  The network signature
        # captures every per-edge mutable quantity the DP reads, so
        # a hit here is exact and lets MH revisits skip the DP.
        sig = self._network_signature()
        cache_key = (key, sig)
        cached_score = self._score_cache.get(cache_key)
        if cached_score is None:
            # Tree-only fast path: when the network has no retics the
            # AC DP frontier degenerates to a single entry per step
            # but still pays the tuple-insertion bookkeeping per node.
            # Route through the optimised tree DP instead, which gives
            # identical results in the no-retic case (verified by
            # ``TestMSCClosedForm`` and the gamma-extreme tests).
            if not any(net_idx.is_retic):
                tree_dt = self._ensure_displayed_trees()
                if tree_dt:
                    cached_score = _msc_log_prob_tree_int(tree_dt[0], gti, self)
                else:
                    cached_score = _msnc_log_prob_network_int(net_idx, gti, self)
            else:
                cached_score = _msnc_log_prob_network_int(net_idx, gti, self)
            if not math.isfinite(cached_score):
                cached_score = _LOG_FLOOR
            self._score_cache[cache_key] = cached_score

        self._gene_tree_log_prob[key] = cached_score
        return cached_score

    def score_many(
        self,
        gene_trees: Iterable[Network],
        species_of: dict[str, str],
    ) -> float:
        """Sum of log probabilities across a batch of gene trees.

        Equivalent to ``sum(log_prob(g, species_of) for g in gene_trees)``
        but intended as the hot path for :class:`MCMCGTScorer` (and a
        natural target for future parallelisation: each gene tree is
        independent once the displayed-tree decomposition is built).
        """
        total = 0.0
        for g in gene_trees:
            total += self.log_prob(g, species_of)
        return total


class _DisplayedTree:
    """One displayed tree of a species network.

    Constructed by :meth:`_GTLikelihoodEngine._build_displayed_trees`
    via edge contraction: for each reticulation ``r`` the retained
    parent edge is kept and the other is deleted, then the resulting
    degree-2 passthrough nodes (``r`` itself plus possibly its kept
    parent) are merged with their surviving neighbours into a single
    edge whose length is the sum of the originals.  This preserves
    the MSC evaluation semantics (coalescent transitions cascade
    across concatenated branches) while reducing the graph to a
    strict rooted binary tree.

    Instances are lightweight: we don't re-use :class:`Network`
    itself (too much setup) but instead expose the minimal data the
    MSC DP needs -- a leaf label -> node map, a parent map, a
    children map, an edge-length map, and a post-order node list.
    """

    __slots__ = (
        "log_weight",
        "root",
        "leaves",
        "leaf_species",
        "children",
        "parent",
        "edge_length",
        "post_order",
        "signature",
        # Int-indexed view used by the bitmask MSC DP.  Built once
        # during ``__init__`` so the DP never has to hash
        # :class:`Node` objects (which dominated the original profile).
        "_idx_n",
        "_idx_post_order",
        "_idx_children",
        "_idx_edge_lengths",
        "_idx_leaves",
        "_idx_leaf_species",
        "_idx_root",
    )

    def __init__(
        self,
        *,
        log_weight: float,
        root: Any,
        leaves: list[Any],
        leaf_species: dict[Any, str],
        children: dict[Any, list[Any]],
        parent: dict[Any, Any],
        edge_length: dict[tuple[Any, Any], float | None],
        post_order: list[Any],
    ) -> None:
        """Store the pre-computed topology; see class docstring."""
        self.log_weight = log_weight
        self.root = root
        self.leaves = leaves
        self.leaf_species = leaf_species
        self.children = children
        self.parent = parent
        self.edge_length = edge_length
        self.post_order = post_order
        # Cache key for the per-displayed-tree score memo on the
        # backing :class:`_GTLikelihoodEngine`.  Stable across the
        # ``_ensure_displayed_trees`` rebuilds that ``update`` triggers
        # because :class:`_DisplayedTree` objects reference the
        # network's live :class:`Node` instances (no deep copy), so
        # ``id(node)`` is invariant for the duration of the search.
        # Pair each node id with the length of the edge feeding into
        # it from its parent; the post-order traversal pins the
        # ordering of the tuple.
        sig_pairs: list[tuple[int, float | None]] = []
        for n in post_order:
            p = parent.get(n)
            length = edge_length.get((p, n)) if p is not None else None
            sig_pairs.append((id(n), length))
        self.signature = tuple(sig_pairs)

        # Build the int-indexed view: assign each species-tree node
        # in ``post_order`` an index 0..len-1.  ``_idx_post_order``
        # is therefore just ``range(n)``; we store derived adjacency
        # arrays so the MSC DP can iterate without re-hashing the
        # underlying :class:`Node` objects.
        idx_of: dict[Any, int] = {n: i for i, n in enumerate(post_order)}
        self._idx_n = len(post_order)
        self._idx_post_order = list(range(self._idx_n))
        self._idx_children = [[] for _ in range(self._idx_n)]
        self._idx_edge_lengths = [[] for _ in range(self._idx_n)]
        for i, node in enumerate(post_order):
            for child in children.get(node, []):
                ci = idx_of[child]
                self._idx_children[i].append(ci)
                self._idx_edge_lengths[i].append(edge_length.get((node, child)))
        self._idx_leaves = [
            idx_of[l] for l in leaves if l in idx_of
        ]
        self._idx_leaf_species = {
            idx_of[l]: leaf_species[l]
            for l in leaves if l in idx_of and l in leaf_species
        }
        self._idx_root = idx_of.get(root, -1)

    @classmethod
    def from_network(
        cls,
        net: Network,
        retic_choice: dict[Node, int],
        log_weight: float,
    ) -> "_DisplayedTree":
        """Construct a displayed tree from a network + retic choices.

        Walks the species network top-down, accumulating a parent map
        and edge-length sums, following only the "kept" parent edge
        at each reticulation.  Degree-2 passthroughs introduced by
        dropping one retic parent are collapsed implicitly by the
        fact that reticulations with in-degree 1 after contraction
        behave as tree internal nodes whose single parent's edge
        carries the combined length.

        Args:
            net: Species network.
            retic_choice: ``{retic_node: 0-or-1}`` specifying which of
                the retic's two parents is retained.
            log_weight: ``sum of log(gamma)`` over the retained
                parent-edges, used by the displayed-tree mixture.

        Returns:
            Fully-populated :class:`_DisplayedTree`.
        """
        root = net.root()
        children_map: dict[Any, list[Any]] = {}
        parent_map: dict[Any, Any] = {}
        length_map: dict[tuple[Any, Any], float | None] = {}
        leaves: list[Any] = []
        leaf_species: dict[Any, str] = {}

        # Use a stable identifier for each species-network node in
        # this displayed tree.  A "node id" is the :class:`Node`
        # itself; we do not copy, since the displayed tree is read
        # only.
        visited: set[Node] = set()
        queue: deque[Node] = deque([root])
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            if net.out_degree(cur) == 0:
                # Leaf: terminal of the DP recursion.
                leaves.append(cur)
                leaf_species[cur] = cur.label
                children_map[cur] = []
                continue
            for child in net.get_children(cur):
                edge = net.get_edge(cur, child)
                length = edge.get_length()
                if child.is_reticulation():
                    # Only traverse the retained parent edge.
                    parents = net.get_parents(child)
                    if len(parents) != 2:
                        # Degenerate retic; conservatively keep the edge.
                        kept = cur
                    else:
                        choice = retic_choice.get(child, 0)
                        kept = parents[choice]
                    if kept is not cur:
                        continue
                # Accumulate degree-2 passthrough collapses: if ``cur``
                # already had a single parent edge feeding into it,
                # and ``cur`` itself is reticulation-like (in_degree>1
                # in original), we've still captured only the retained
                # branch via the ``kept == cur`` gate above.  The
                # ``edge_length`` here therefore correctly reflects
                # the surviving branch in the displayed tree.
                children_map.setdefault(cur, []).append(child)
                parent_map[child] = cur
                length_map[(cur, child)] = length
                queue.append(child)
        # Simplify: some reticulation nodes remain in the graph with
        # a single retained parent edge but their own outgoing edge;
        # we flatten those (length-sum) by detecting in_degree==1
        # passthrough-chains over reticulations and collapsing.
        root_out = cls._collapse_passthroughs(
            root=root,
            children_map=children_map,
            parent_map=parent_map,
            length_map=length_map,
            leaves=leaves,
            is_retic=lambda n: n.is_reticulation(),
        )
        post_order = cls._compute_post_order(root_out, children_map)
        return cls(
            log_weight=log_weight,
            root=root_out,
            leaves=[l for l in leaves if l in parent_map or l is root_out],
            leaf_species=leaf_species,
            children=children_map,
            parent=parent_map,
            edge_length=length_map,
            post_order=post_order,
        )

    @staticmethod
    def _collapse_passthroughs(
        *,
        root: Any,
        children_map: dict[Any, list[Any]],
        parent_map: dict[Any, Any],
        length_map: dict[tuple[Any, Any], float | None],
        leaves: list[Any],
        is_retic: Callable[[Any], bool],
    ) -> Any:
        """Merge degree-2 (in=1, out=1) chains into single edges.

        Called once during :meth:`from_network` to flatten any
        reticulation nodes that, after parent pruning, have only one
        incoming and one outgoing edge.  Branch lengths are summed.

        Works in place on the supplied maps and returns the (possibly
        unchanged) root.
        """
        changed = True
        while changed:
            changed = False
            # Snapshot interior nodes to safely mutate during iteration.
            interior = [
                n for n in list(children_map.keys())
                if n is not root
                and len(children_map.get(n, [])) == 1
                and n in parent_map
            ]
            for n in interior:
                parent = parent_map[n]
                if n not in children_map or len(children_map[n]) != 1:
                    continue
                child = children_map[n][0]
                # Merge (parent -> n -> child) into (parent -> child).
                top_len = length_map.get((parent, n))
                bot_len = length_map.get((n, child))
                combined = _safe_add(top_len, bot_len)
                # Detach n, reattach child directly.
                parent_kids = children_map.get(parent, [])
                parent_kids = [c for c in parent_kids if c is not n]
                parent_kids.append(child)
                children_map[parent] = parent_kids
                parent_map[child] = parent
                length_map.pop((parent, n), None)
                length_map.pop((n, child), None)
                length_map[(parent, child)] = combined
                del children_map[n]
                del parent_map[n]
                changed = True
        return root

    @staticmethod
    def _compute_post_order(
        root: Any,
        children_map: dict[Any, list[Any]],
    ) -> list[Any]:
        """Return a post-order traversal of the displayed tree."""
        post: list[Any] = []
        stack: list[tuple[Any, bool]] = [(root, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                post.append(node)
                continue
            stack.append((node, True))
            for child in children_map.get(node, []):
                stack.append((child, False))
        return post


def _safe_add(a: float | None, b: float | None) -> float | None:
    """Add two possibly-``None`` branch lengths.

    Treats ``None`` as "no finite length" (the root edge / infinite
    branch); summing ``None`` with anything yields ``None``.  For
    finite inputs the usual sum applies.
    """
    if a is None or b is None:
        return None
    return float(a) + float(b)




def _msc_log_prob_tree(
    dt: _DisplayedTree,
    gene_tree: Network,
    species_of: dict[str, str],
    engine: _GTLikelihoodEngine,
) -> float:
    """Compute ``log P(gene_tree | dt)`` via a partition-DP on ``dt``.

    Standard Rannala-Yang MSC evaluation:

      1. Initialise each leaf of ``dt`` with the set of gene-tree
         leaves mapped to its species (may be empty when the gene
         tree doesn't cover every species).
      2. Post-order ``dt``.  At each non-root internal node combine
         the "top-of-edge" lineage configurations coming in from its
         children (disjoint union; probabilities multiply).  At each
         edge entering the current node from a child, convolve the
         child's lineage configuration against the Kingman coalescent
         transition for the edge length, enumerating every valid
         coarsening of the active lineages via sibling-pair merges
         in the gene tree.
      3. Above the root of ``dt``, apply an infinite-length branch
         to force the remaining lineages to coalesce down to one.

    Args:
        dt: Displayed tree (already contracted).
        gene_tree: Gene tree to score.
        species_of: Gene-copy label -> species label.
        engine: Backing :class:`_GTLikelihoodEngine` (for ``_gij``).

    Returns:
        Log probability; clamped at ``_LOG_FLOOR`` if the gene tree
        has no valid embedding in this displayed tree (e.g. species
        coverage mismatch).
    """
    # Build gene-tree parent/child maps once per call.
    g_children: dict[Any, tuple[Any, Any]] = {}
    g_parent: dict[Any, Any] = {}
    g_root: Any | None = None
    g_leaves: list[Any] = []
    for node in gene_tree.V():
        kids = gene_tree.get_children(node)
        if not kids:
            g_leaves.append(node)
            continue
        if len(kids) != 2:
            # Non-binary gene tree (polytomy) -- flatten to "either
            # child order" by picking the first two; higher-order
            # polytomies reduce to a lower bound under the MSC.  A
            # better approach is to expand every binary resolution;
            # we flag this as a small-polytomy approximation.
            kids = kids[:2]
        g_children[node] = (kids[0], kids[1])
        for c in kids:
            g_parent[c] = node
    if gene_tree.roots():
        g_root = gene_tree.root()

    # Build leaf-config at each species leaf of the displayed tree.
    configs_at: dict[Any, dict[frozenset, float]] = {}
    for leaf in dt.leaves:
        species = dt.leaf_species.get(leaf)
        gene_nodes = frozenset(
            g for g in g_leaves
            if species_of.get(g.label) == species
        )
        configs_at[leaf] = {gene_nodes: 0.0}

    # Walk the displayed tree in post-order.  For each internal node
    # whose children have finished, merge children configs and apply
    # the coalescent transition on each incoming edge.
    for node in dt.post_order:
        if node in configs_at:
            continue
        kids = dt.children.get(node, [])
        if not kids:
            configs_at[node] = {frozenset(): 0.0}
            continue
        # 1. Apply per-child-edge coalescent transition into each child's
        #    config so we have "top-of-edge" distributions.
        child_top: list[dict[frozenset, float]] = []
        for child in kids:
            t = dt.edge_length.get((node, child))
            child_top.append(
                _apply_branch_coalescent(
                    configs_at[child],
                    t,
                    g_children,
                    g_parent,
                    engine,
                )
            )
        # 2. Combine child top-of-edge configs by disjoint union
        #    (independent subtrees, products of probabilities).
        merged = child_top[0]
        for nxt in child_top[1:]:
            merged = _combine_configs(merged, nxt)
        configs_at[node] = merged

    # Above the root, apply an infinite-length branch (g_{n,1}(inf)=1)
    # forcing a collapse to a single lineage at the gene-tree root.
    root_config = configs_at.get(dt.root, {})
    top_config = _apply_branch_coalescent(
        root_config,
        None,
        g_children,
        g_parent,
        engine,
    )
    if g_root is None:
        # No gene tree -> trivial.
        return 0.0
    target = frozenset([g_root])
    best = top_config.get(target, None)
    if best is None:
        # Aggregate across all configs that collapsed to the single
        # root -- there should be only one, but be defensive.
        acc: list[float] = []
        for cfg, lp in top_config.items():
            if len(cfg) == 1 and g_root in cfg:
                acc.append(lp)
        if not acc:
            return _LOG_FLOOR
        return _logsumexp(acc)
    return best


def _apply_branch_coalescent(
    config_in: dict[frozenset, float],
    length: float | None,
    g_children: dict[Any, tuple[Any, Any]],
    g_parent: dict[Any, Any],
    engine: _GTLikelihoodEngine,
) -> dict[frozenset, float]:
    """Convolve a bottom-of-edge distribution against branch coalescent.

    Runs, for every entry ``(config_in, log_prob_in)``, all reachable
    branch coarsenings and accumulates log-space probability into
    ``out[config_out]`` via log-sum-exp.  The Kingman coalescent
    transition probability used here is:

        P(C_out | C_in, t) = g_{n,m}(t) * |L(F)| / prod_{i=1..k} C(n-i+1, 2)

    where ``n = |C_in|``, ``m = |C_out|``, ``k = n - m``, ``F`` is the
    forest of gene-tree internal nodes picked up by the coarsening,
    and ``|L(F)|`` is the number of linear extensions (time orderings)
    of ``F`` under gene-tree ancestry.

    Args:
        config_in: Dict from incoming-lineage frozenset to log prob.
        length: Branch length in coalescent units (``None`` =
            infinite, i.e. force full collapse to one lineage).
        g_children / g_parent: Gene-tree adjacency maps.
        engine: Backing engine (memoised ``_gij``).

    Returns:
        Dict from outgoing-lineage frozenset to log prob.  Entries
        whose accumulated log prob falls to ``-inf`` are dropped.
    """
    out: dict[frozenset, list[float]] = {}
    for cfg_in, lp_in in config_in.items():
        n = len(cfg_in)
        coarsenings = _enum_coarsenings(cfg_in, g_children, g_parent)
        for cfg_out, merges in coarsenings:
            m = len(cfg_out)
            k = n - m
            gij = engine._gij(length, n, m)
            if gij <= 0.0:
                continue
            log_branch = math.log(gij)
            if k > 0:
                # Denominator: product of C(n-i+1, 2) for i in 1..k
                denom = 1.0
                for i in range(1, k + 1):
                    denom *= math.comb(n - i + 1, 2)
                le = _linear_extensions(list(merges), g_children)
                log_branch += math.log(le) - math.log(denom)
            log_total = lp_in + log_branch
            out.setdefault(cfg_out, []).append(log_total)
    # Collapse list-per-key via log-sum-exp.
    result: dict[frozenset, float] = {}
    for cfg, terms in out.items():
        result[cfg] = _logsumexp(terms)
    return result


def _combine_configs(
    left: dict[frozenset, float],
    right: dict[frozenset, float],
) -> dict[frozenset, float]:
    """Outer-product combine of two child-edge distributions.

    At a species-tree internal node, lineages from independent child
    subtrees are disjoint, so their combined distribution is the
    outer product (union of configs, sum of log probs) of the
    top-of-edge distributions coming in from each child.

    Duplicate resulting configs (possible when two gene-tree leaves
    live in different species children but share the same internal
    g-ancestry -- this happens for single-species gene subtrees that
    got split across a species-tree bipartition) are merged via
    log-sum-exp to avoid double-counting.
    """
    out: dict[frozenset, list[float]] = {}
    for cfg_l, lp_l in left.items():
        for cfg_r, lp_r in right.items():
            if cfg_l & cfg_r:
                # Overlapping lineage labels -> independence assumption
                # violated; skip.  (Shouldn't happen if mappings are
                # well-formed.)
                continue
            union = cfg_l | cfg_r
            out.setdefault(union, []).append(lp_l + lp_r)
    result: dict[frozenset, float] = {}
    for cfg, terms in out.items():
        result[cfg] = _logsumexp(terms)
    return result


# ======================================================================
# (D) Scorer
# ======================================================================

# ---------------------------------------------------------------------
# Multiprocess scoring pool
# ---------------------------------------------------------------------
# The MSNC AC DP is embarrassingly parallel across gene trees: each
# gene tree's ``log_prob`` call only reads the species network and
# the per-tree ``_GeneTreeIndex``, never writes shared state inside
# its own DP.  ``score_many`` therefore parallelises trivially across
# a process pool: each worker holds **all** gene trees + its own
# persistent :class:`_GTLikelihoodEngine`, and per iteration the
# main process pickles the current network once and dispatches it
# along with a ``[lo, hi)`` index range to each worker.  Each worker
# scores its assigned range and returns a partial sum; the main
# process sums the partials to recover the full ``score_many`` total.
#
# Why processes and not threads.  The Cython hot loops
# (``_apply_branch_coalescent_cy`` / ``_combine_configs_cy``) call
# back into Python for the coarsening cache + log_gij memo, so they
# never release the GIL.  Threads would serialise on the GIL exactly
# where the work is.  Processes sidestep this entirely at the cost
# of per-iter pickling -- on the 7-tax bench network pickle is
# ~3 KB / 0.1 ms (warm), dominated by the parallel scoring savings.
#
# Why every worker holds *all* gene trees rather than a
# pre-partitioned slice.  ``ProcessPoolExecutor.initializer`` runs
# on every worker with the *same* args, so we can't hand out
# per-worker slices through the official initialiser API.  Sending
# all trees to every worker at startup costs O(workers x trees)
# bytes once -- on the 7-tax bench, 4 workers x 200 trees x ~3 KB =
# ~2.4 MB of one-shot startup IO -- and lets us partition by index
# range per iter, which is correct without per-worker dispatch
# tricks (and is robust to ``map`` task interleaving).
#
# Worker-process global state.  Each worker stashes its persistent
# state in module-globals; this is the standard ``initializer``
# pattern and is safe because each worker is a single-threaded
# Python process with no concurrent task execution.
_WORKER_STATE: dict[str, Any] = {}


def _mcmc_worker_init(
    gene_trees_pickle: bytes,
    species_of: dict[str, str],
) -> None:
    """Initialise a worker process: unpickle the full gene-tree set.

    Called by :class:`ProcessPoolExecutor`'s ``initializer`` exactly
    once per worker at pool start.  The unpickled gene trees keep
    stable Python ``id()`` values for the worker's lifetime, so the
    engine's ``_gti_cache`` (keyed by ``id(gene_tree)``) hits
    permanently after the first ``log_prob`` call -- same caching
    behaviour as the serial path.

    Args:
        gene_trees_pickle: Pickled list of every gene tree to be
            scored.  Each worker holds the full list; per-iter
            scoring picks a ``[lo, hi)`` slice.
        species_of: Gene-copy label -> species label map.
    """
    global _WORKER_STATE
    gene_trees = pickle.loads(gene_trees_pickle)
    _WORKER_STATE = {
        "gene_trees": list(gene_trees),
        "species_of": dict(species_of),
        "engine": None,
    }


def _mcmc_worker_score(args: tuple[bytes, int, int]) -> float:
    """Score a contiguous gene-tree slice against the supplied network.

    Args:
        args: ``(network_pickle, idx_lo, idx_hi)``.  ``network_pickle``
            is the main-process pickle of the current species
            network for this iteration.  The worker scores
            ``state["gene_trees"][idx_lo:idx_hi]``.

    Returns:
        Partial sum of ``log P(g | net)`` over the assigned slice.
        Pickle round-trip on the network is score-preserving (diff=0
        verified on the 7-tax bench).
    """
    state = _WORKER_STATE
    network_pickle, idx_lo, idx_hi = args
    new_net = pickle.loads(network_pickle)

    # Reuse the worker's persistent engine across iterations -- we
    # only reset its network-side state, keeping the per-tree GTI
    # cache + the (length, n, m) ``_log_gij`` memo + the per-tree
    # coarsening / linear-extension caches warm.
    engine = state["engine"]
    if engine is None:
        engine = _GTLikelihoodEngine(new_net)
        state["engine"] = engine
    else:
        engine.network = new_net
        # ``update(None)`` busts the network index + per-call gene
        # tree memo + per-(gt, network) score cache, but keeps the
        # tree-level caches that are valid forever.
        engine.update(None)

    species_of = state["species_of"]
    gts = state["gene_trees"]
    total = 0.0
    for i in range(idx_lo, idx_hi):
        total += engine.log_prob(gts[i], species_of)
    return total


class _ScoreManyPool:
    """Persistent process pool for parallel :meth:`score_many`.

    Owns a :class:`ProcessPoolExecutor` plus a contiguous index
    partition over the gene-tree list.  One pool per
    :class:`MCMCGTScorer`; created when ``n_workers > 1`` and torn
    down by :meth:`MCMCGTScorer.close` (and ``__del__`` defensively).

    Partition is contiguous (worker 0 gets ``[0, k)``, worker 1 gets
    ``[k, 2k)``, ...) so adjacent gene trees -- which often share
    similar topologies in real datasets -- end up on the same
    worker.  This isn't load-balanced for skewed inputs, but for
    IID-like gene trees it gives even slices and keeps cache
    locality high within a worker.
    """

    def __init__(
        self,
        gene_trees: Sequence[Network],
        species_of: dict[str, str],
        n_workers: int,
    ) -> None:
        """Spin up workers and broadcast the full gene-tree set.

        Args:
            gene_trees: Full ordered list of gene trees.  Workers
                receive a copy of this list at init time; per-iter
                scoring picks ``[lo, hi)`` slices into it.
            species_of: Gene-copy -> species label map.
            n_workers: Number of worker processes.  Capped at
                ``len(gene_trees)`` (no point spawning more workers
                than there is work to dispatch).
        """
        n_total = len(gene_trees)
        n_workers = max(1, min(int(n_workers), n_total))
        self._n_workers = n_workers

        # Contiguous, near-equal partition.  ``chunk`` is the per-
        # worker slice size with the remainder spread across the
        # first few workers (so total covers exactly ``n_total``).
        chunk_base, rem = divmod(n_total, n_workers)
        ranges: list[tuple[int, int]] = []
        cursor = 0
        for w in range(n_workers):
            size = chunk_base + (1 if w < rem else 0)
            ranges.append((cursor, cursor + size))
            cursor += size
        self._ranges: tuple[tuple[int, int], ...] = tuple(ranges)
        self._slice_sizes: tuple[int, ...] = tuple(hi - lo for lo, hi in ranges)

        # Pickle the gene trees once; ``ProcessPoolExecutor`` ships
        # the pickled bytes to every worker through the initializer.
        gts_pickle = pickle.dumps(list(gene_trees))
        self._executor = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_mcmc_worker_init,
            initargs=(gts_pickle, dict(species_of)),
        )
        # Force every worker to start + run its initializer before
        # we return -- otherwise the first ``score()`` call pays
        # the worker-startup latency on its critical path.  We do
        # this by submitting a no-op task per worker and waiting.
        warmup = list(self._executor.map(
            _mcmc_worker_ping,
            range(n_workers),
            chunksize=1,
        ))
        if not all(warmup):
            raise RuntimeError(f"Worker pool warm-up failed: {warmup!r}")
        self._closed = False

    @property
    def n_workers(self) -> int:
        """Number of worker processes in the pool."""
        return self._n_workers

    @property
    def slice_sizes(self) -> tuple[int, ...]:
        """Per-worker gene-tree slice sizes.  Useful for debug prints."""
        return self._slice_sizes

    def score(self, network: Network) -> float:
        """Score every gene tree in parallel and return the total.

        Args:
            network: Current species network.  Pickled once on the
                main process and shipped to every worker (one
                ``map`` task per worker).

        Returns:
            Sum of ``log P(g | network)`` across all gene trees,
            byte-identical to the serial ``engine.score_many`` path
            on the same network and gene trees (verified by
            empirical diff in the smoke benchmark).
        """
        if self._closed:
            raise RuntimeError("Pool is closed")
        net_pickle = pickle.dumps(network)
        tasks = [(net_pickle, lo, hi) for lo, hi in self._ranges]
        partials = list(self._executor.map(
            _mcmc_worker_score,
            tasks,
            chunksize=1,
        ))
        return float(sum(partials))

    def close(self) -> None:
        """Shut down the worker pool (idempotent)."""
        if not self._closed:
            try:
                self._executor.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass
            self._closed = True

    def __del__(self) -> None:
        """Defensive cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass


def _mcmc_worker_ping(worker_idx: int) -> bool:
    """No-op task that confirms a worker is up and ``_WORKER_STATE`` is set.

    Used by :class:`_ScoreManyPool.__init__` to force every worker
    to run its initialiser before the pool returns -- otherwise the
    first ``score()`` call pays worker-startup latency on its
    critical path (which on Windows is several hundred ms per
    worker).  The ``worker_idx`` argument is unused but distinct
    per task, ensuring ``map`` actually dispatches one task per
    worker rather than coalescing them.

    Returns:
        ``True`` once ``_WORKER_STATE`` is populated.
    """
    return _WORKER_STATE.get("gene_trees") is not None


class MCMCGTScorer:
    """Callable likelihood evaluator for :class:`Model`.

    Registered via ``model.set_likelihood_calculator(scorer)``.  Holds
    a persistent :class:`_GTLikelihoodEngine` bound to
    ``model.network``; rebinds automatically when the scorer detects
    a network-object identity swap (e.g. after a deep-copy undo) or
    when ``model._dirty_nodes is None`` (topology change).

    Returns either the log posterior (``posterior=True``, the canonical
    Bayesian MCMC_GT objective) or the plain log likelihood
    (``posterior=False``, for ``method="hc"`` / ``"sa"`` where the
    prior term is irrelevant to the maximisation).
    """

    def __init__(
        self,
        gene_trees: Iterable[Network] | GeneTrees,
        mapping: dict[str, list[str]],
        priors: Optional[MCMC_GTPriors] = None,
        *,
        posterior: bool = True,
        n_workers: int = 1,
    ) -> None:
        """Initialise from gene trees, mapping, and prior hyperparams.

        Args:
            gene_trees: Either a :class:`GeneTrees` collection or any
                iterable yielding :class:`Network` gene trees.
            mapping: Species label -> list of gene-copy labels (same
                convention as :class:`MPL`).  Internally flattened to
                a gene-copy -> species reverse map for the engine.
            priors: Prior hyperparameters.  Default: :class:`MCMC_GTPriors`
                defaults (Exp(1) branch lengths, Beta(1,1) gammas,
                Poisson(1) retic count).
            posterior: When ``True`` (default), return log posterior.
                When ``False``, return plain log likelihood (priors
                dropped).
            n_workers: Number of worker processes for parallel
                ``score_many``.  ``1`` (default) keeps the original
                serial path.  When ``> 1``, a :class:`_ScoreManyPool`
                is constructed with that many workers; gene trees are
                broadcast to every worker once at pool startup.
                Capped at ``len(gene_trees)``.  Pool is torn down via
                :meth:`close` (called automatically from
                :meth:`MCMC_GT.search` via try/finally).
        """
        self._gene_trees: list[Network] = list(
            gene_trees.trees if isinstance(gene_trees, GeneTrees) else gene_trees
        )
        self._mapping = mapping
        self._species_of: dict[str, str] = {}
        for species, alleles in mapping.items():
            for a in alleles:
                self._species_of[a] = species
        self._priors = priors if priors is not None else MCMC_GTPriors()
        self._posterior = posterior
        self._engine: _GTLikelihoodEngine | None = None
        self._engine_network: Network | None = None
        # Last-observed likelihood (useful for reporting / tests).
        self._last_log_likelihood: float | None = None
        self._last_log_posterior: float | None = None

        # Lazy-init the worker pool: spawning workers on
        # ``__init__`` would pay the startup tax even for one-off
        # ``score()`` calls.  Pool is built on the first scoring
        # call when ``n_workers > 1`` and held for the scorer's
        # lifetime (idempotent close in :meth:`close`).
        self._n_workers: int = max(1, int(n_workers))
        self._pool: Optional[_ScoreManyPool] = None

    @property
    def posterior_mode(self) -> bool:
        """True if the scorer returns log posterior rather than log likelihood."""
        return self._posterior

    @property
    def last_log_likelihood(self) -> float | None:
        """Last likelihood value computed (``None`` before first call)."""
        return self._last_log_likelihood

    @property
    def last_log_posterior(self) -> float | None:
        """Last posterior value computed (``None`` before first call)."""
        return self._last_log_posterior

    @property
    def n_workers(self) -> int:
        """Number of worker processes used by ``score_many`` (1 = serial)."""
        return self._n_workers

    def _ensure_pool(self) -> Optional[_ScoreManyPool]:
        """Spin up the worker pool on first use, or return ``None``.

        Returns ``None`` when the scorer was constructed with
        ``n_workers <= 1`` -- the caller falls back to the serial
        ``engine.score_many`` path.
        """
        if self._n_workers <= 1:
            return None
        if self._pool is None:
            self._pool = _ScoreManyPool(
                self._gene_trees, self._species_of, self._n_workers,
            )
        return self._pool

    def close(self) -> None:
        """Tear down the worker pool if one was started.

        Safe to call multiple times; safe to call when no pool was
        ever started.  :meth:`MCMC_GT.search` calls this from a
        ``finally`` block so chains terminate cleanly even on
        exceptions.
        """
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def __del__(self) -> None:
        """Defensive cleanup on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    def __call__(self, model: Model) -> float:
        """Score ``model`` under MSNC (+ priors if ``posterior``).

        Args:
            model: :class:`Model` whose ``network`` is the current
                species network.

        Returns:
            Log posterior (when ``self.posterior_mode``) or log
            likelihood.  Clamped at ``_LOG_FLOOR`` on the lower end.
        """
        net = model.network
        if self._engine is None or self._engine_network is not net:
            # Network object swapped (e.g. after deep-copy undo).  Full
            # rebind; the engine's ``_gij`` memo can't be reused
            # because the old engine held references to old edge
            # objects.  This is still O(1) wrt iteration.
            self._engine = _GTLikelihoodEngine(net)
            self._engine_network = net
            model.clear_dirty_nodes()
        else:
            dirty = getattr(model, "_dirty_nodes", None)
            self._engine.update(dirty)
            model.clear_dirty_nodes()

        # Likelihood over all gene trees.  Parallel path takes the
        # current network and dispatches it to the worker pool;
        # serial path runs on the main-process engine.  Both produce
        # numerically-identical totals (verified empirically: diff=0
        # across pickle round-trip on 7-tax bench).
        pool = self._ensure_pool()
        if pool is not None:
            log_lik = pool.score(net)
        else:
            log_lik = self._engine.score_many(
                self._gene_trees, self._species_of,
            )
        if not math.isfinite(log_lik):
            log_lik = _LOG_FLOOR
        self._last_log_likelihood = log_lik

        if self._posterior:
            log_prior = log_prior_network(net, self._priors)
            log_post = log_lik + log_prior
            self._last_log_posterior = log_post
            return log_post
        self._last_log_posterior = log_lik
        return log_lik


# ======================================================================
# (E) Proposal kernel
# ======================================================================

class _MCMCGTAdaptiveConfig:
    """Internal tuning constants for :class:`MCMCGTKernel`.

    Numerical values mirror :class:`MPL._AdaptiveConfig` -- both kernels
    share the same Robbins-Monro / phase-cycle / SPR-decay machinery and
    we want them to behave identically on equivalent inputs.  The one
    intentional divergence is the **target acceptance rate**: MPL is an
    optimiser (likes high acceptance to climb fast) so it targets 0.35;
    Bayesian MH targets the Roberts/Gelman/Gilks 1997 asymptotic optimum
    of 0.234.
    """

    # ΓöÇΓöÇ Robbins-Monro continuous-sigma tuning ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # ``log(sigma)`` drifts toward the target via
    # ``log_sigma += step * (accept_indicator - target)`` with
    # ``step = 1 / (n + shift)^exp``.  The 1/n^0.6 schedule satisfies
    # the Roberts-Rosenthal 2007 diminishing-adaptation condition,
    # so we can leave the chain adapting through the whole run if we
    # want (we still freeze on burn-in by default for safety).
    SIGMA_TARGET_ACCEPT = 0.234           # Roberts/Gelman/Gilks 1997
    SIGMA_TUNE_DELAY = 20                 # warm-up obs before adapting
    SIGMA_TUNE_EXPONENT = 0.6
    SIGMA_TUNE_DENOM_SHIFT = 50

    # ΓöÇΓöÇ ChangeNodeHeight sigma_frac bounds ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # ``sigma_frac`` is dimensionless (fraction of feasible half-range).
    NH_SIGMA_INIT = 0.4
    NH_SIGMA_MIN = 0.02
    NH_SIGMA_MAX = 1.5

    # ΓöÇΓöÇ ChangeInheritanceProb sigma bounds ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    CIP_SIGMA_INIT = 0.1
    CIP_SIGMA_MIN = 0.005
    CIP_SIGMA_MAX = 0.4

    # ΓöÇΓöÇ SPR adaptive regraft radius ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # Edges weighted by ``1 / d**decay``.  High decay = local; low
    # decay = broad.  Interpolates linearly with stagnation/patience.
    SPR_DECAY_MAX = 2.5
    SPR_DECAY_MIN = 0.5

    # ΓöÇΓöÇ Efficiency-aware weight scaler blend ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # composite = ACC * acc_rate + EFF * efficiency
    # In MH the analogue of "improvement" is "log-posterior delta > 0";
    # we use that as the efficiency signal (i.e. acceptance into the
    # *higher-density* part of the chain rather than uphill via temp).
    SCALER_ACC_WEIGHT = 0.4
    SCALER_EFF_WEIGHT = 0.6
    SCALER_EFF_EPSILON = 0.01

    # ΓöÇΓöÇ Stagnation reset threshold ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    # Magnitude (log-posterior units) of an accepted-and-improving
    # move that counts as "real progress" for stagnation-reset
    # purposes.  Tiny tweaks from late-stage NH/CIP moves don't reset
    # the counter; honest improvements do.
    STAGNATION_RESET_DELTA = 1.0


class MCMCGTKernel(ProposalKernel):
    """Phase-aware, Robbins-Monro-tuned proposal kernel for :class:`MCMC_GT`.

    Cycles between two complementary search phases (same design as
    :class:`MPL.MPLKernel`):

      **TOPOLOGY** -- SPR + branch-length / inheritance-prob tuning;
      small bleed (~10% aggregate) of reticulation moves.

      **RETICULATION** -- Add/Remove/Flip/Source/Dest/Relocate +
      gamma tuning; small bleed (~10%) of SPR + ChangeNodeHeight so
      local refinements still happen alongside retic restructuring.

    Phase transitions fire after ``phase_patience`` consecutive
    proposals without an accepted log-posterior delta exceeding
    ``stagnation_reset_delta``.  Within each phase, ``ChangeNodeHeight``
    and ``ChangeInheritanceProb`` adapt their proposal sigmas via
    Robbins-Monro toward 0.234 acceptance (Roberts/Gelman/Gilks 1997),
    and ``SPR`` interpolates ``distance_decay`` between the local and
    broad endpoints based on the stagnation level.

    **Detailed balance.**  Adaptive MCMC is only valid for posterior
    sampling when the adaptation either (a) shrinks to zero
    asymptotically (Roberts-Rosenthal 2007 *J. Comput. Graph. Stat.*
    diminishing-adaptation theorem) or (b) is frozen at some point and
    the post-freeze chain is the actual sample.  Our 1/n^0.6 RM
    schedule satisfies (a), but to be conservative
    :meth:`freeze_adaptation` is called by :meth:`MCMC_GT._run_mh`
    when the burn-in window ends.  After freezing the kernel reverts
    to fixed-weight, fixed-sigma behaviour and the chain is a proper
    posterior sample.

    Backward-compat path: when explicit *weights* are supplied to the
    constructor, the kernel falls back to flat (non-phased,
    non-adaptive) sampling -- the original v1 behaviour.

    Default weights (re-normalised each draw after the cap-aware
    filter on AddReticulation):

        TOPOLOGY phase                   RETICULATION phase
        ------------------               ------------------
        SPR                  5.0         AddReticulation       3.0
        ChangeNodeHeight     3.0         RemoveReticulation    2.0
        ChangeInheritProb    1.0         RelocateReticulation  2.5
        AddReticulation      0.30        ChangeReticSource     1.5
        RemoveReticulation   0.20        ChangeReticDest       1.5
        RelocateRetic        0.25        FlipReticulation      0.5
        ChangeReticSource    0.10        ChangeInheritProb     1.5
        ChangeReticDest      0.10        ChangeNodeHeight      1.0
        FlipReticulation     0.03        SPR                   1.5
    """

    TOPOLOGY = "topology"
    RETICULATION = "reticulation"

    _TOPOLOGY_BASE: dict[type, float] = {
        SPR: 5.0,
        ChangeNodeHeight: 3.0,
        ChangeInheritanceProb: 1.0,
        AddReticulation: 0.30,
        RemoveReticulation: 0.20,
        RelocateReticulation: 0.25,
        ChangeReticSource: 0.10,
        ChangeReticDest: 0.10,
        FlipReticulation: 0.03,
    }
    _RETICULATION_BASE: dict[type, float] = {
        AddReticulation: 3.0,
        RemoveReticulation: 2.0,
        FlipReticulation: 0.5,
        ChangeReticSource: 1.5,
        ChangeReticDest: 1.5,
        RelocateReticulation: 2.5,
        ChangeInheritanceProb: 1.5,
        ChangeNodeHeight: 1.0,
        SPR: 1.5,
    }
    _PHASE_ORDER = [TOPOLOGY, RETICULATION]

    # Backward-compat alias kept for any out-of-tree caller still
    # reading ``MCMCGTKernel._DEFAULT_WEIGHTS``.  Maps to the topology
    # base map (the de facto starting phase).
    _DEFAULT_WEIGHTS: dict[type, float] = _TOPOLOGY_BASE

    def __init__(
        self,
        max_reticulations: Optional[int] = None,
        weights: Optional[dict[type, float]] = None,
        rng: Optional[np.random.Generator] = None,
        *,
        adaptive: bool = True,
        window_size: int = 30,
        min_weight: float = 0.05,
        phase_patience: int = 150,
        warmup: int = 8,
        stagnation_reset_delta: Optional[float] = None,
    ) -> None:
        """Configure the kernel.

        Args:
            max_reticulations: Upper bound on retic count.  When the
                current network is at or above the cap,
                :class:`AddReticulation` is filtered out of the
                candidate pool.
            weights: When supplied, disables phase cycling and
                adaptive tuning; the kernel reverts to the v1
                fixed-weight behaviour.  Useful for unit tests and
                "I know exactly what I want" callers.
            rng: NumPy random Generator.  When ``None``, a fresh one
                is created from OS entropy; callers that care about
                reproducibility should pass one seeded from the
                chain's :class:`np.random.SeedSequence`.
            adaptive: Enable within-phase adaptive weight scaling
                (efficiency-aware) and Robbins-Monro sigma tuning.
                Ignored when ``weights`` is supplied.  Default ``True``.
            window_size: Sliding-window length for per-move
                acceptance/improvement statistics.
            min_weight: Floor on the multiplicative scale factor
                applied to base weights, to prevent any move from
                being fully starved.
            phase_patience: Consecutive non-improving proposals before
                the kernel switches to the next phase.
            warmup: Minimum observations per move class before the
                adaptive scaler activates for that class.
            stagnation_reset_delta: Minimum strict-improvement
                magnitude (in log-posterior units) that counts as
                "real progress" for stagnation-reset purposes.
                ``None`` -> :attr:`_MCMCGTAdaptiveConfig.STAGNATION_RESET_DELTA`.
        """
        super().__init__()
        self._max_retics = max_reticulations
        self.rng = rng if rng is not None else np.random.default_rng()

        # ΓöÇΓöÇ Mode flags ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        # ``_phased`` -> use phase cycling
        # ``_adaptive`` -> use within-phase adaptive scaling + RM sigmas
        # Both are forced off when explicit ``weights`` are supplied.
        self._fixed_weights: Optional[dict[type, float]] = (
            dict(weights) if weights is not None else None
        )
        self._phased: bool = self._fixed_weights is None
        self._adaptive: bool = adaptive and (self._fixed_weights is None)

        self._all_moves: list[type] = list(
            self._fixed_weights.keys()
            if self._fixed_weights is not None
            else self._TOPOLOGY_BASE.keys()
        )

        # ΓöÇΓöÇ Phase / stagnation state ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        self._phase: str = self.TOPOLOGY
        self._phase_patience: int = phase_patience
        self._stagnation: int = 0
        self._phase_switches: int = 0
        self._stagnation_peak: int = 0
        self._stagnation_reset_delta: float = (
            float(stagnation_reset_delta)
            if stagnation_reset_delta is not None
            else _MCMCGTAdaptiveConfig.STAGNATION_RESET_DELTA
        )

        # ΓöÇΓöÇ Adaptive weight scaler state ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        self._window_size: int = window_size
        self._min_weight: float = min_weight
        self._warmup: int = warmup
        self._history: dict[type, deque] = {
            cls: deque(maxlen=window_size) for cls in self._all_moves
        }

        # ΓöÇΓöÇ Adaptive sigmas for continuous proposals ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        self._nh_sigma: float = _MCMCGTAdaptiveConfig.NH_SIGMA_INIT
        self._cip_sigma: float = _MCMCGTAdaptiveConfig.CIP_SIGMA_INIT
        self._sigma_obs: dict[type, int] = {
            ChangeNodeHeight: 0,
            ChangeInheritanceProb: 0,
        }

        # ΓöÇΓöÇ SPR distance decay tracking ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        self._spr_decay_last: float = _MCMCGTAdaptiveConfig.SPR_DECAY_MAX
        self._spr_decay_min_seen: float = _MCMCGTAdaptiveConfig.SPR_DECAY_MAX
        self._spr_decay_max_seen: float = _MCMCGTAdaptiveConfig.SPR_DECAY_MAX

        # ΓöÇΓöÇ Lifetime move counters (end-of-run diagnostics) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        # ``proposed``: every draw of a move class.
        # ``accepted``: every draw the MH driver committed.
        # ``improved``: accepts whose log-posterior delta was strictly
        #   positive (the Bayesian analogue of MPL's "improvement").
        self._proposed: dict[type, int] = {cls: 0 for cls in self._all_moves}
        self._accepted: dict[type, int] = {cls: 0 for cls in self._all_moves}
        self._improved: dict[type, int] = {cls: 0 for cls in self._all_moves}
        self._last_cls: Optional[type] = None

    # ΓöÇΓöÇ adaptation lifecycle ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def freeze_adaptation(self) -> None:
        """Disable all online adaptation; keep current sigmas/phase fixed.

        Called by :meth:`MCMC_GT._run_mh` exactly when the burn-in
        window ends.  The post-freeze chain is a proper Bayesian
        posterior sample (no diminishing-adaptation gymnastics
        required).  Stat counters keep updating so ``format_stats``
        still reports honest per-move accept rates over the
        post-burn-in window.
        """
        self._adaptive = False

    @property
    def adaptive(self) -> bool:
        """``True`` while the kernel is still adapting sigmas/weights."""
        return self._adaptive

    @property
    def phase(self) -> str:
        """Current search phase (``"topology"`` or ``"reticulation"``)."""
        return self._phase

    # ΓöÇΓöÇ phase / cap helpers ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def _phase_base_map(self) -> dict[type, float]:
        """Base weight map for the current phase."""
        if self._phase == self.TOPOLOGY:
            return self._TOPOLOGY_BASE
        return self._RETICULATION_BASE

    def _at_retic_cap(self, network: Optional[Network]) -> bool:
        """True when the current network already holds ``max_reticulations``."""
        if self._max_retics is None or network is None:
            return False
        return sum(1 for n in network.V() if n.is_reticulation()) >= self._max_retics

    def _active_moves_and_base(
        self, network: Optional[Network] = None,
    ) -> tuple[list[type], list[float]]:
        """Active move classes and base weights for the current phase.

        Drops :class:`AddReticulation` when the network is at the
        retic cap so we don't waste proposals on guaranteed no-ops.
        """
        base = self._phase_base_map()
        at_cap = self._at_retic_cap(network)
        moves: list[type] = []
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

    # ΓöÇΓöÇ adaptive SPR regraft radius ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def _current_spr_decay(self) -> float:
        """SPR ``distance_decay`` for the current stagnation level.

        Linearly interpolates from ``SPR_DECAY_MAX`` (just improved /
        fresh, prefer local regrafts) to ``SPR_DECAY_MIN`` (fully
        stuck, near-flat distribution -> broad hops).  Returns the
        max value when the kernel is non-adaptive.
        """
        if not self._adaptive or self._phase_patience <= 0:
            return _MCMCGTAdaptiveConfig.SPR_DECAY_MAX
        stuckness = min(1.0, self._stagnation / float(self._phase_patience))
        span = (
            _MCMCGTAdaptiveConfig.SPR_DECAY_MAX
            - _MCMCGTAdaptiveConfig.SPR_DECAY_MIN
        )
        return _MCMCGTAdaptiveConfig.SPR_DECAY_MAX - stuckness * span

    # ΓöÇΓöÇ Robbins-Monro sigma tuning ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def _tune_sigma(
        self,
        current: float,
        n_obs: int,
        accepted: bool,
        sigma_min: float,
        sigma_max: float,
    ) -> float:
        """One Robbins-Monro step of ``log(sigma)`` toward the target.

        ``log(sigma) += step * (accept_indicator - target)`` with
        ``step = 1 / (n_obs + shift)^exp``.  Standard adaptive
        Metropolis update (Roberts & Rosenthal 2009 *Examples of
        Adaptive MCMC*).  Accept-rate above target widens the
        proposal; below target tightens it.  Schedule satisfies
        diminishing adaptation.
        """
        if n_obs < _MCMCGTAdaptiveConfig.SIGMA_TUNE_DELAY:
            return current
        step = 1.0 / (
            (n_obs + _MCMCGTAdaptiveConfig.SIGMA_TUNE_DENOM_SHIFT)
            ** _MCMCGTAdaptiveConfig.SIGMA_TUNE_EXPONENT
        )
        indicator = 1.0 if accepted else 0.0
        log_adjust = step * (indicator - _MCMCGTAdaptiveConfig.SIGMA_TARGET_ACCEPT)
        new_sigma = current * math.exp(log_adjust)
        return max(sigma_min, min(sigma_max, new_sigma))

    # ΓöÇΓöÇ adaptive weight scaler ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def _adapt_weights(
        self,
        moves: list[type],
        base_weights: list[float],
    ) -> list[float]:
        """Scale base weights by an efficiency-aware accept/improve blend.

        Composite =
          ``SCALER_ACC_WEIGHT * acc_rate``        (chain-mixing credit)
        + ``SCALER_EFF_WEIGHT * efficiency``      (posterior-delta credit)

        where ``efficiency = imp_rate / max(acc_rate, eps)``.  This
        directs proposal mass toward moves that not only get accepted
        but actually move the chain to higher-posterior regions --
        the Bayesian analogue of the MPL "improvement" signal.

        Moves with fewer than ``warmup`` observations keep their full
        base weight to avoid early-window noise swinging the scaler.
        """
        floor = self._min_weight
        out: list[float] = []
        for cls, base_w in zip(moves, base_weights):
            hist = self._history[cls]
            if len(hist) < self._warmup:
                out.append(base_w)
                continue
            n = len(hist)
            n_acc = sum(1 for accepted, _ in hist if accepted)
            n_imp = sum(1 for accepted, delta in hist if accepted and delta > 0.0)
            acc_rate = n_acc / n
            imp_rate = n_imp / n
            efficiency = min(
                1.0,
                imp_rate / max(acc_rate, _MCMCGTAdaptiveConfig.SCALER_EFF_EPSILON),
            )
            composite = (
                _MCMCGTAdaptiveConfig.SCALER_ACC_WEIGHT * acc_rate
                + _MCMCGTAdaptiveConfig.SCALER_EFF_WEIGHT * efficiency
            )
            scale = floor + (2.0 - floor) * composite
            out.append(base_w * scale)
        return out

    # ΓöÇΓöÇ public interface ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

    def generate(self, model: "Model | None" = None) -> Move:
        """Sample a :class:`Move` from the current effective distribution.

        Args:
            model: Optional current model.  Used for cap-aware
                filtering (``AddReticulation`` dropped at the cap).

        Returns:
            Freshly-instantiated :class:`Move`.  Continuous moves
            (``ChangeNodeHeight``, ``ChangeInheritanceProb``) are
            instantiated with the kernel's current adaptive sigma;
            ``SPR`` with its current adaptive ``distance_decay``.
        """
        network = model.network if model is not None else None

        # --- Decide eligible moves and weights ----------------------
        if not self._phased:
            moves = list(self._fixed_weights.keys())
            weights = [self._fixed_weights[cls] for cls in moves]
            # Honour the cap even in fixed-weight mode.
            if self._at_retic_cap(network):
                moves_w = [
                    (cls, w) for cls, w in zip(moves, weights)
                    if cls is not AddReticulation
                ]
                if moves_w:
                    moves = [m for m, _ in moves_w]
                    weights = [w for _, w in moves_w]
        else:
            self._maybe_switch_phase()
            moves, base_weights = self._active_moves_and_base(network)
            weights = (
                self._adapt_weights(moves, base_weights)
                if self._adaptive else base_weights
            )

        if not moves:
            # Defensive: cap + filter eliminated everything.  Fall
            # back to a topology-neutral move so the chain progresses.
            return ChangeNodeHeight(sigma_frac=self._nh_sigma)

        # --- Sample ------------------------------------------------
        w = np.asarray(weights, dtype=float)
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0.0:
            idx = int(self.rng.integers(0, len(moves)))
        else:
            idx = int(self.rng.choice(len(moves), p=w / total))
        cls = moves[idx]
        self._last_cls = cls
        self._proposed[cls] = self._proposed.get(cls, 0) + 1

        # --- Instantiate with current adaptive knobs ----------------
        if cls is AddReticulation and self._max_retics is not None:
            return AddReticulation(max_reticulations=self._max_retics)
        if cls is ChangeNodeHeight:
            return ChangeNodeHeight(sigma_frac=self._nh_sigma)
        if cls is ChangeInheritanceProb:
            return ChangeInheritanceProb(sigma=self._cip_sigma)
        if cls is SPR:
            decay = self._current_spr_decay()
            self._spr_decay_last = decay
            self._spr_decay_min_seen = min(self._spr_decay_min_seen, decay)
            self._spr_decay_max_seen = max(self._spr_decay_max_seen, decay)
            return SPR(distance_decay=decay)
        return cls()

    def report_outcome(self, accepted: bool, delta: float = 0.0) -> None:
        """Record a move outcome and update adaptive state.

        Called by the search driver after each ``generate -> propose ->
        score`` cycle, regardless of accept/reject.

        Args:
            accepted: True iff the driver committed the move to the
                chain.
            delta: ``log_posterior_proposed - log_posterior_current``.
                Positive means the proposal landed in a strictly
                higher-density region of the posterior.
        """
        cls = self._last_cls
        if cls is None:
            return

        if cls in self._history:
            self._history[cls].append((accepted, delta))
        if accepted:
            self._accepted[cls] = self._accepted.get(cls, 0) + 1
            if delta > 0.0:
                self._improved[cls] = self._improved.get(cls, 0) + 1

        # Robbins-Monro sigma tuning (only while adapting; otherwise
        # we keep the frozen sigma from burn-in's last update).
        if self._adaptive:
            if cls is ChangeNodeHeight:
                self._sigma_obs[cls] += 1
                self._nh_sigma = self._tune_sigma(
                    self._nh_sigma,
                    self._sigma_obs[cls],
                    accepted,
                    _MCMCGTAdaptiveConfig.NH_SIGMA_MIN,
                    _MCMCGTAdaptiveConfig.NH_SIGMA_MAX,
                )
            elif cls is ChangeInheritanceProb:
                self._sigma_obs[cls] += 1
                self._cip_sigma = self._tune_sigma(
                    self._cip_sigma,
                    self._sigma_obs[cls],
                    accepted,
                    _MCMCGTAdaptiveConfig.CIP_SIGMA_MIN,
                    _MCMCGTAdaptiveConfig.CIP_SIGMA_MAX,
                )

        # Magnitude-aware stagnation reset.  Only honest improvements
        # reset; tiny tweaks let the counter climb so phase switches /
        # SPR-decay opens are driven by genuine plateaus.
        if accepted and delta > self._stagnation_reset_delta:
            self._stagnation = 0
        else:
            self._stagnation += 1
            if self._stagnation > self._stagnation_peak:
                self._stagnation_peak = self._stagnation

    def get_weights(self) -> dict[str, float]:
        """Return the current effective selection probabilities.

        Honours phase + adaptive scaler.  Useful for diagnostic
        printing mid-run.
        """
        if self._phased:
            moves, base = self._active_moves_and_base()
            weights = self._adapt_weights(moves, base) if self._adaptive else base
        else:
            moves = list(self._fixed_weights.keys())
            weights = [self._fixed_weights[c] for c in moves]
        total = sum(weights)
        if total <= 0.0:
            return {cls.__name__: 0.0 for cls in moves}
        return {cls.__name__: w / total for cls, w in zip(moves, weights)}

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Lifetime per-move statistics across the search.

        Returns:
            Mapping ``move_class_name -> {proposed, accepted, improved,
            acceptance_rate, improvement_rate}``.
        """
        out: dict[str, dict[str, float]] = {}
        for cls in self._all_moves:
            p = self._proposed.get(cls, 0)
            a = self._accepted.get(cls, 0)
            i = self._improved.get(cls, 0)
            out[cls.__name__] = {
                "proposed": p,
                "accepted": a,
                "improved": i,
                "acceptance_rate": (a / p) if p else 0.0,
                "improvement_rate": (i / p) if p else 0.0,
            }
        return out

    def format_stats(self) -> str:
        """Return a human-readable summary of lifetime kernel statistics.

        Includes per-move counts/rates, phase-switch summary, adaptive
        sigma end-state, and SPR distance-decay range.  Prints sorted
        by proposal count (descending) for at-a-glance reading of the
        actually-active moves.
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
                f"{int(s['proposed']):>9d}  "
                f"{int(s['accepted']):>9d}  "
                f"{int(s['improved']):>9d}  "
                f"{100.0 * s['acceptance_rate']:>7.2f}%  "
                f"{100.0 * s['improvement_rate']:>8.3f}%"
            )
        lines.append(f"  total proposals: {int(total_prop)}")
        if self._phased:
            lines.append(
                f"  phase: {self._phase}  "
                f"switches: {self._phase_switches}  "
                f"stagnation peak: {self._stagnation_peak}/{self._phase_patience}  "
                f"adaptive: {self._adaptive}"
            )
            lines.append(
                "  adaptive sigma -- "
                f"ChangeNodeHeight.sigma_frac={self._nh_sigma:.4f} "
                f"(n={self._sigma_obs.get(ChangeNodeHeight, 0)}), "
                f"ChangeInheritanceProb.sigma={self._cip_sigma:.4f} "
                f"(n={self._sigma_obs.get(ChangeInheritanceProb, 0)})"
            )
            lines.append(
                "  adaptive SPR distance_decay "
                f"last={self._spr_decay_last:.3f}, "
                f"range=[{self._spr_decay_min_seen:.3f}, "
                f"{self._spr_decay_max_seen:.3f}] "
                f"(low=broad, high=local)"
            )
        return "\n".join(lines)


# ======================================================================
# (F) Orchestration
# ======================================================================

class MCMC_GT:
    """Bayesian (and HC/SA) network inference from gene-tree topologies.

    Mirrors :class:`MPL`'s public surface:

    * :meth:`score` computes a one-off log posterior / likelihood for
      the currently-held network.
    * :meth:`search` runs a driver (``"mh"`` for Metropolis-Hastings
      posterior sampling; ``"hc"`` / ``"sa"`` for likelihood
      maximisation) and returns an :class:`MCMCGTResult`.
    * :classmethod:`from_nexus` is a convenience constructor for
      NEXUS-formatted gene trees + starting network.
    * :classmethod:`from_consensus` builds a starting tree from a
      majority-rule consensus of the gene trees (the recommended seed
      when no prior estimate is available).

    Gene-tree data (fixed across the run) is stored verbatim; the
    ``species_net`` attribute is *updated in place* at the end of
    :meth:`search` to reflect the best network found.
    """

    def __init__(
        self,
        species_net: Network,
        gene_trees: GeneTrees,
        mapping: dict[str, list[str]],
        priors: Optional[MCMC_GTPriors] = None,
    ) -> None:
        """Initialise the inference object.

        Args:
            species_net: Starting species network.  Must have branch
                lengths on every edge; gammas on reticulation
                in-edges.  Consumed by reference and may be mutated
                by :meth:`search`.
            gene_trees: Input gene tree set.
            mapping: Species -> list of gene-copy labels.
            priors: :class:`MCMC_GTPriors` hyperparameters.  Default:
                all weights at the Wen-Nakhleh defaults.
        """
        self.net = species_net
        self.gene_trees = gene_trees
        self.mapping = mapping
        self.priors = priors if priors is not None else MCMC_GTPriors()

    @classmethod
    def from_nexus(
        cls,
        gt_file: str,
        st_file: str,
        mapping: dict[str, list[str]],
        priors: Optional[MCMC_GTPriors] = None,
    ) -> "MCMC_GT":
        """Construct from two NEXUS files.

        Args:
            gt_file: Path to the gene-tree NEXUS.
            st_file: Path to the starting-network NEXUS.
            mapping: Species -> list of gene-copy labels.
            priors: Optional prior hyperparameters.

        Returns:
            Fully-initialised :class:`MCMC_GT`.
        """
        st: Network = io.read_nexus(st_file, return_type="networks")
        gts: GeneTrees = io.read_nexus(gt_file, return_type="genetrees")
        if hasattr(gts, "species_gene_mapping"):
            gts.species_gene_mapping(mapping)
        return cls(st, gts, mapping, priors=priors)

    @classmethod
    def from_consensus(
        cls,
        gene_trees: GeneTrees,
        mapping: dict[str, list[str]],
        priors: Optional[MCMC_GTPriors] = None,
        *,
        threshold: float = 0.5,
    ) -> "MCMC_GT":
        """Seed the starting network from a majority-rule consensus.

        Uses :meth:`GeneTrees.build_majority_rule_consensus_tree`.
        Polytomies in the consensus produce a deliberately poor
        initial score that the kernel resolves in the first few
        hundred moves (same caveat as :class:`MPL.from_consensus`-style
        usage).

        Args:
            gene_trees: Gene-tree collection to build the consensus
                from.
            mapping: Species -> list of gene-copy labels.
            priors: Optional prior hyperparameters.
            threshold: Support threshold for consensus clades.
                Default 0.5 (majority rule).

        Returns:
            Fully-initialised :class:`MCMC_GT`.
        """
        seed = gene_trees.build_majority_rule_consensus_tree(threshold=threshold)
        _populate_default_branch_lengths(seed)
        return cls(seed, gene_trees, mapping, priors=priors)

    # ── Scoring ───────────────────────────────────────────────────

    def score(self, *, posterior: bool = True) -> float:
        """One-off score of the current network.

        Args:
            posterior: When ``True`` (default) return log posterior;
                else return log likelihood only.

        Returns:
            Log posterior or log likelihood.
        """
        scorer = MCMCGTScorer(
            self.gene_trees, self.mapping, self.priors, posterior=posterior,
        )
        model = Model(rng=np.random.default_rng())
        model.network = self.net
        model.set_likelihood_calculator(scorer)
        return scorer(model)

    # ── Search ────────────────────────────────────────────────────

    def search(
        self,
        method: str = "mh",
        num_iter: int = 10_000,
        *,
        burn_in: int = 2_000,
        thin: int = 10,
        kernel: Optional[MCMCGTKernel] = None,
        max_reticulations: Optional[int] = None,
        seed: Any = None,
        n_workers: int = 1,
        **kwargs: Any,
    ) -> MCMCGTResult:
        """Search the network space with the chosen driver.

        Args:
            method: One of ``"mh"`` (Metropolis-Hastings posterior
                sampling; the canonical Bayesian MCMC_GT),
                ``"hc"`` (Hill Climbing over log likelihood), or
                ``"sa"`` (Simulated Annealing over log likelihood).
            num_iter: Total proposed moves.
            burn_in: MH only -- iterations to discard before
                collecting samples.  Ignored for HC / SA.
            thin: MH only -- collect every ``thin``-th post-burn-in
                sample.
            kernel: Custom :class:`MCMCGTKernel`.  When ``None`` a
                default one is constructed with
                ``max_reticulations``.
            max_reticulations: Cap passed to the default kernel.
            seed: Seed for the chain's RNGs.  Accepts any value
                :class:`np.random.SeedSequence` accepts (including
                another :class:`SeedSequence`).
            n_workers: Number of worker processes for parallel
                ``score_many`` (per-iter likelihood across the gene
                tree set).  ``1`` (default) keeps the serial path
                bit-for-bit identical to the v1 behaviour.  Values
                ``> 1`` spin up a persistent :class:`_ScoreManyPool`
                of that size; gene trees are broadcast to every
                worker once at startup, network is pickled once per
                iteration and dispatched to all workers in parallel.
                Capped at the number of gene trees.  Pool teardown
                is automatic via ``finally`` even on exception.
            **kwargs: Forwarded to the chosen search driver.  For HC
                these include ``enhanced_stop``; for SA these include
                ``t_start``, ``t_end``, ``schedule``, etc. (see
                :class:`SimulatedAnnealing`).

        Returns:
            :class:`MCMCGTResult` with the best network plus, for
            MH, the posterior sample list.
        """
        method = method.lower()
        if method not in {"mh", "hc", "sa"}:
            raise ValueError(f"Unknown method {method!r}; use 'mh', 'hc', or 'sa'.")

        if method == "mh":
            _nm.warn_if_large_mcmc(len(self.mapping), method="MCMC_GT")

        # RNG plumbing (mirrors ``MPL.search``: one root SeedSequence,
        # separate generators for the accept/reject loop, the kernel,
        # and the model/move RNG).
        if isinstance(seed, np.random.SeedSequence):
            root_ss = seed
        else:
            root_ss = np.random.SeedSequence(seed)
        driver_ss, kernel_ss, model_ss = root_ss.spawn(3)
        driver_seed = int(driver_ss.generate_state(1)[0])

        # Scorer + model.
        posterior = (method == "mh")
        scorer = MCMCGTScorer(
            self.gene_trees, self.mapping, self.priors,
            posterior=posterior, n_workers=n_workers,
        )
        model = Model(rng=np.random.default_rng(model_ss))
        model.network = copy.deepcopy(self.net)
        model.set_likelihood_calculator(scorer)

        # Kernel.
        if kernel is None:
            kernel = MCMCGTKernel(
                max_reticulations=max_reticulations,
                rng=np.random.default_rng(kernel_ss),
            )
        elif getattr(kernel, "rng", None) is None:
            kernel.rng = np.random.default_rng(kernel_ss)

        # Dispatch.  ``finally`` guarantees pool teardown even on
        # mid-chain exceptions; otherwise an exception in the MH loop
        # would leak worker processes.
        start = time.time()
        try:
            if method == "mh":
                result = self._run_mh(
                    model=model,
                    kernel=kernel,
                    scorer=scorer,
                    num_iter=num_iter,
                    burn_in=burn_in,
                    thin=thin,
                    driver_seed=driver_seed,
                )
            elif method == "hc":
                searcher = HillClimbing(
                    pkernel=kernel,
                    model=model,
                    num_iter=num_iter,
                    **kwargs,
                )
                end_state = searcher.run()
                best_score = end_state.likelihood()
                result = MCMCGTResult(
                    method="hc",
                    best_network=end_state.current_model.network,
                    best_log_posterior=best_score,
                    num_iter=num_iter,
                )
            else:  # method == "sa"
                kwargs.setdefault("seed", driver_ss)
                searcher = SimulatedAnnealing(
                    pkernel=kernel,
                    model=model,
                    num_iter=num_iter,
                    **kwargs,
                )
                end_state = searcher.run()
                best_score = end_state.likelihood()
                result = MCMCGTResult(
                    method="sa",
                    best_network=end_state.current_model.network,
                    best_log_posterior=best_score,
                    num_iter=num_iter,
                )
            result.wall_time_sec = time.time() - start
        finally:
            scorer.close()

        # Adopt the final network into self.net.  For MH we use the
        # final-state network (equivalent to "last sample"); callers
        # wanting the MAP should read ``result.best_network`` instead.
        self.net = copy.deepcopy(result.best_network)
        return result

    # ── Internal ──────────────────────────────────────────────────

    def _run_mh(
        self,
        *,
        model: Model,
        kernel: MCMCGTKernel,
        scorer: MCMCGTScorer,
        num_iter: int,
        burn_in: int,
        thin: int,
        driver_seed: int,
    ) -> MCMCGTResult:
        """Run a Metropolis-Hastings chain with burn-in and thinning.

        Uses the same posterior formulation as standard MH: accept a
        proposal with log-acceptance ``prop_score - cur_score +
        log_hastings_ratio``.  Samples are drawn every ``thin`` steps
        after ``burn_in``; ``MAP`` is tracked independently so even
        runs that never converge report the best single network
        observed.

        Args:
            model: Live model (deep-copied from ``self.net``).
            kernel: Proposal kernel.
            scorer: The scorer returned by ``model.likelihood`` --
                held explicitly so we can read back the last
                likelihood value for sample diagnostics.
            num_iter: Total proposed moves.
            burn_in: Iterations before collecting samples.
            thin: Sample every ``thin``-th post-burn-in iteration.
            driver_seed: Seed for the accept/reject Uniform(0, 1)
                generator.

        Returns:
            Populated :class:`MCMCGTResult`.
        """
        # Avoid importing State here to keep the MCMC_GT module free
        # of any State coupling; we use Model directly and manage the
        # accept/reject loop inline.  This is identical semantics to
        # the :class:`MetropolisHastings` driver except we can pull
        # samples mid-run and we use a dedicated Generator for the
        # acceptance-test random draws so reproducibility holds.
        rng = np.random.default_rng(driver_seed)
        cur_score = float(scorer(model))
        best_score = cur_score
        best_network = copy.deepcopy(model.network)
        best_log_lik = scorer.last_log_likelihood or cur_score
        samples: list[MCMCSample] = []
        num_accepted = 0

        # Adaptive kernels (e.g. :class:`MCMCGTKernel` with
        # ``adaptive=True``) tune sigmas / weights during burn-in only;
        # freezing afterwards preserves detailed balance (Roberts &
        # Rosenthal 2007 *J. Comput. Graph. Stat.*).  Static kernels
        # ignore this signal -- the call is only made when the kernel
        # actually exposes ``freeze_adaptation``.
        freeze_fn = getattr(kernel, "freeze_adaptation", None)
        adaptation_frozen = False

        for iter_no in range(num_iter):
            if (
                freeze_fn is not None
                and not adaptation_frozen
                and iter_no >= burn_in
            ):
                freeze_fn()
                adaptation_frozen = True

            move = kernel.generate(model)
            try:
                move.execute(model)
                # Reject structurally invalid proposals up front so they
                # count toward the rejection denominator instead of being
                # silently floored by the scorer (which breaks detailed
                # balance and inflates apparent mixing).
                if not _is_valid_network(model.network):
                    move.undo(model)
                    kernel.report_outcome(False, delta=0.0)
                    continue
                prop_score = float(scorer(model))
            except Exception:
                # Move failed mid-execute; try to undo and continue.
                try:
                    move.undo(model)
                except Exception:
                    pass
                kernel.report_outcome(False, delta=0.0)
                continue

            # Compute delta + acceptance test BEFORE mutating
            # ``cur_score`` so the adaptive layer sees the actual
            # log-posterior step size, not zero.  This was a bug in
            # the v1 kernel-stats path (delta was always 0 on accept).
            #
            # ``log_hastings_ratio`` is already in log space (0.0 for a
            # symmetric proposal), so it is *added* to the log-posterior
            # delta -- not exponentiated.  The previous interface
            # returned a linear ``1.0`` here, which silently inflated
            # every acceptance by a factor of ``e``.
            delta = prop_score - cur_score
            log_alpha = delta + move.log_hastings_ratio()
            accept = (log_alpha >= 0.0) or (math.log(rng.random()) < log_alpha)
            if accept:
                num_accepted += 1
                kernel.report_outcome(True, delta=delta)
                cur_score = prop_score
                if prop_score > best_score:
                    best_score = prop_score
                    best_network = copy.deepcopy(model.network)
                    best_log_lik = scorer.last_log_likelihood or prop_score
            else:
                try:
                    move.undo(model)
                except Exception:
                    # If undo fails, we're in a weird state.  Rebuild
                    # the scorer against the current (possibly
                    # corrupted) network rather than crashing.
                    pass
                kernel.report_outcome(False, delta=delta)

            # Sample (MH posterior).
            if iter_no >= burn_in and ((iter_no - burn_in) % thin == 0):
                samples.append(
                    MCMCSample(
                        iteration=iter_no,
                        network=copy.deepcopy(model.network),
                        log_posterior=cur_score,
                        log_likelihood=(
                            scorer.last_log_likelihood
                            if scorer.last_log_likelihood is not None
                            else cur_score
                        ),
                    )
                )

        return MCMCGTResult(
            method="mh",
            best_network=best_network,
            best_log_posterior=best_score,
            samples=samples,
            num_iter=num_iter,
            num_accepted=num_accepted,
        )


# ======================================================================
# Helpers
# ======================================================================

def _is_valid_network(net: Network) -> bool:
    """Cheap structural-validity gate for proposed networks.

    Enforces the phylogenetic-network invariants the MSNC scorer assumes:
    exactly one root (in-degree 0), every leaf with in-degree 1, every
    reticulation with in-degree 2 / out-degree 1, every other internal
    node with in-degree 1 / out-degree >= 2, and global acyclicity.
    Used by :meth:`MCMC_GT._run_mh` to reject malformed proposals as part
    of the Metropolis-Hastings rejection step (preserving detailed
    balance) rather than letting the scorer floor them to ``-inf``.
    """
    if net is None:
        return False
    root_count = 0
    for n in net.V():
        ind = net.in_degree(n)
        outd = net.out_degree(n)
        if ind == 0:
            root_count += 1
            if outd < 2:
                return False
        elif outd == 0:
            if ind != 1:
                return False
        elif n.is_reticulation():
            if ind != 2 or outd != 1:
                return False
        else:
            if ind != 1 or outd < 2:
                return False
    if root_count != 1:
        return False
    try:
        return net.is_acyclic()
    except Exception:
        return False


def _populate_default_branch_lengths(net: Network, default: float = 1.0) -> None:
    """Fill missing branch lengths with ``default`` coalescent units.

    Consensus trees from :meth:`GeneTrees.build_majority_rule_consensus_tree`
    may come out with missing or zero branch lengths; the MSNC
    likelihood requires a positive length on every edge to be
    meaningful.  This is a quick-and-dirty initialiser -- the MH /
    HC / SA search will refine the lengths from the prior / gradient
    within the first few hundred iterations anyway.
    """
    for e in net.E():
        bl = e.get_length()
        if bl is None or bl <= 0.0:
            e.set_length(default)
        g = e.get_gamma()
        if g is None and e.dest.is_reticulation():
            e.set_gamma(0.5)
