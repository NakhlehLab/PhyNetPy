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
Last Edit : 6/3/26
First Included in Version : 0.5.0

InferNetwork_ML -- maximum-likelihood phylogenetic-network inference from
gene-tree topologies.

This module re-implements PhyloNet's ``InferNetwork_ML`` command (Yu, Dong,
Liu & Nakhleh, *PNAS* 2014, "Maximum Likelihood Inference of Reticulate
Evolutionary Histories"), but built directly on top of PhyNetPy's existing,
heavily-optimised multispecies-network-coalescent (MSNC) machinery.

Why this is a *thin* layer
==========================
The likelihood that ``InferNetwork_ML`` maximises -- the probability of a set
of gene-tree topologies under the MSNC on a species network with branch
lengths (in coalescent units) and inheritance probabilities -- is exactly the
quantity computed by :class:`phynetpy._mcmc_gt._GTLikelihoodEngine`
(the ancestral-configurations DP of Yu, Degnan & Nakhleh 2012).  The Bayesian
:class:`~phynetpy.infer.MCMC_GT` method already drives that engine with an
adaptive proposal kernel and incremental cache invalidation.  Maximum
likelihood is then just MCMC_GT's objective *without the priors*
(``posterior=False``) plus three ML-specific behaviours:

1. **Per-topology continuous-parameter optimisation** (the PhyloNet ``-o``
   flag).  For every species-network *topology* examined, the branch lengths
   and inheritance probabilities are numerically optimised to their
   maximum-likelihood values before the topology is scored.  We use Brent's
   bounded line search per parameter (the same algorithm PhyloNet uses),
   wrapped in a coordinate-ascent loop.  Because the engine supports
   parameter-only ("partial dirty") cache invalidation, each line-search
   evaluation reuses the per-topology network index and only re-runs the DP.
2. **Hill-climbing topology search with random restarts** (``-x numRuns``).
3. **Consecutive-failure stopping** (``-f maxFailure``) and an optional cap on
   the number of topologies examined (``-m maxNetExamined``).

Everything else -- the move set, cap-aware reticulation proposals, and the
fast DP -- is shared verbatim with MCMC_GT, so this module stays small and the
two methods cannot drift apart numerically.

Public surface (mirrors :class:`~phynetpy.infer.MPL` /
:class:`~phynetpy.infer.MCMC_GT`)::

    from phynetpy.infer import InferNetwork_ML

    inf = InferNetwork_ML(starting_net, gene_trees, mapping, max_reticulations=1)
    result = inf.search(num_runs=5, optimize_params=True)
    print(result.best_network, result.best_log_likelihood)

Docs   - [x]
Tests  - [ ]
Design - [x]
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .Network import Network, Node
from .GeneTrees import GeneTrees
from .ModelGraph import Model
from . import IO as io

# Re-use MCMC_GT's likelihood scorer (posterior=False -> plain MSNC log
# likelihood) and proposal kernel.  ``_LOG_FLOOR`` keeps incompatible
# configurations finite so the hill-climb never compares against -inf.
from ._mcmc_gt import (
    MCMCGTScorer,
    MCMCGTKernel,
    _populate_default_branch_lengths,
    _LOG_FLOOR,
)

# Brent's bounded line search.  SciPy is a hard PhyNetPy dependency
# (``scipy>=1.5`` in ``setup.py``), but we degrade gracefully to a pure-Python
# golden-section search if it is somehow unavailable so the optimiser never
# hard-crashes an inference run.
try:  # pragma: no cover - import guard
    from scipy.optimize import minimize_scalar as _scipy_minimize_scalar

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - import guard
    _scipy_minimize_scalar = None
    _HAS_SCIPY = False


__all__ = [
    "InferNetwork_ML",
    "InferNetworkMLResult",
    "optimize_network_parameters",
]


# ======================================================================
# Numerical defaults
# ======================================================================
# Branch-length and gamma bounds mirror PhyloNet's defaults where one
# exists (``-l maxBL`` defaults to 6 coalescent units) and use small
# epsilons elsewhere so the optimiser never proposes a degenerate
# zero-length branch or a gamma pinned exactly at 0/1 (both send the MSNC
# log likelihood to -inf for some gene trees).
_MIN_BRANCH_LENGTH: float = 1e-6
_MAX_BRANCH_LENGTH: float = 6.0
_GAMMA_EPS: float = 1e-4

# Coordinate-ascent stopping.  ``_DEFAULT_MAX_ROUNDS`` matches PhyloNet's
# ``-r maxRounds`` (100); in practice the per-round improvement check exits
# far sooner.  ``_BRANCH_LINE_ITERS`` caps Brent evaluations per parameter so
# a single coordinate sweep stays cheap.
_DEFAULT_MAX_ROUNDS: int = 100
_DEFAULT_IMPROVE_THRESHOLD: float = 1e-4
_BRANCH_LINE_ITERS: int = 20
_BRANCH_LINE_XATOL: float = 1e-3


# ======================================================================
# Result container
# ======================================================================

@dataclass
class InferNetworkMLResult:
    """Aggregate result returned by :meth:`InferNetwork_ML.search`.

    Attributes:
        best_network: Highest-likelihood species network found across
            every run.  Branch lengths (coalescent units) and
            reticulation inheritance probabilities are the ML estimates
            for this topology (always optimised at least once at the end
            of the search).
        best_log_likelihood: MSNC log likelihood of ``best_network``.
            Higher (closer to 0) is better.
        num_reticulations: Reticulation-node count of ``best_network``.
        networks: Up to ``num_networks`` distinct top-scoring networks
            (best first), each paired with its log likelihood.  Useful
            for inspecting near-optimal alternatives / model selection.
        num_networks_examined: Total number of distinct topologies
            scored across all runs (the analogue of PhyloNet's
            ``-m`` budget consumption).
        num_runs: Number of independent hill-climbing runs executed.
        wall_time_sec: Observed wall-clock time of the whole search.
    """

    best_network: Network
    best_log_likelihood: float
    num_reticulations: int = 0
    networks: list[tuple[Network, float]] = field(default_factory=list)
    num_networks_examined: int = 0
    num_runs: int = 0
    wall_time_sec: float = 0.0


# ======================================================================
# Continuous-parameter optimisation (the "-o" behaviour)
# ======================================================================

def _line_minimize(
    neg_obj: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    max_iter: int,
    xatol: float,
) -> tuple[float, float]:
    """Minimise a unimodal-ish 1-D objective on ``[lo, hi]``.

    Wraps SciPy's bounded Brent minimiser (Brent, *Algorithms for
    Minimization without Derivatives*, 1973 -- the exact reference
    PhyloNet cites) when available, falling back to a fixed-iteration
    golden-section search otherwise.  Both return the minimiser and the
    objective value there.

    Args:
        neg_obj: The function to minimise.  Callers pass the *negative*
            log likelihood so that minimisation == likelihood
            maximisation.
        lo, hi: Inclusive search bounds.
        max_iter: Hard cap on objective evaluations.
        xatol: Absolute tolerance on the minimiser location.

    Returns:
        ``(x_best, neg_obj(x_best))``.
    """
    if hi <= lo:
        return lo, neg_obj(lo)

    if _HAS_SCIPY:
        res = _scipy_minimize_scalar(
            neg_obj,
            bounds=(lo, hi),
            method="bounded",
            options={"maxiter": max_iter, "xatol": xatol},
        )
        return float(res.x), float(res.fun)

    # Golden-section fallback: no derivatives, guaranteed bracket
    # contraction by the golden ratio each step.
    invphi = (math.sqrt(5.0) - 1.0) / 2.0          # 1/phi  ~ 0.618
    invphi2 = (3.0 - math.sqrt(5.0)) / 2.0         # 1/phi^2 ~ 0.382
    a, b = lo, hi
    c = a + invphi2 * (b - a)
    d = a + invphi * (b - a)
    fc = neg_obj(c)
    fd = neg_obj(d)
    for _ in range(max(1, max_iter)):
        if (b - a) < xatol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = a + invphi2 * (b - a)
            fc = neg_obj(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = neg_obj(d)
    if fc < fd:
        return c, fc
    return d, fd


def _allele_counts(mapping: dict[str, list[str]]) -> dict[str, int]:
    """Species label -> number of mapped gene copies (alleles)."""
    return {sp: len(alleles) for sp, alleles in mapping.items()}


def _collect_continuous_params(
    net: Network,
    allele_counts: dict[str, int],
) -> tuple[list[Any], list[tuple[Any, Any]]]:
    """Enumerate the free continuous parameters of ``net``.

    Returns two lists:

    * ``length_edges`` -- edges whose branch length is *identifiable*
      and therefore worth optimising.  An external (pendant) edge that
      leads to a leaf with a single sampled allele contributes nothing
      to the gene-tree-topology likelihood (a lineage cannot coalesce
      with itself), so we skip it.  This is a meaningful efficiency win
      on single-allele datasets where pendant edges dominate the edge
      count.
    * ``gamma_pairs`` -- for every reticulation node, the pair of
      in-edges ``(e0, e1)`` whose inheritance probabilities are
      constrained to sum to 1.  Only ``e0``'s gamma is a free parameter;
      ``e1`` is set to ``1 - gamma`` during optimisation.

    Args:
        net: Species network to scan.
        allele_counts: Species label -> allele count (from the mapping).

    Returns:
        ``(length_edges, gamma_pairs)``.
    """
    length_edges: list[Any] = []
    for e in net.E():
        dest = e.dest
        if net.out_degree(dest) == 0:
            # Pendant edge: only identifiable when >1 allele can coalesce
            # within the species branch.
            if allele_counts.get(dest.label, 1) <= 1:
                continue
        length_edges.append(e)

    gamma_pairs: list[tuple[Any, Any]] = []
    for v in net.V():
        if v.is_reticulation():
            # ``in_edges`` may return a set; materialise before indexing.
            in_edges = list(net.in_edges(v))
            if len(in_edges) >= 2:
                gamma_pairs.append((in_edges[0], in_edges[1]))

    return length_edges, gamma_pairs


def optimize_network_parameters(
    model: Model,
    scorer: MCMCGTScorer,
    mapping: dict[str, list[str]],
    *,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    improve_threshold: float = _DEFAULT_IMPROVE_THRESHOLD,
    min_branch: float = _MIN_BRANCH_LENGTH,
    max_branch: float = _MAX_BRANCH_LENGTH,
    branch_iters: int = _BRANCH_LINE_ITERS,
) -> float:
    """Maximise the likelihood of ``model.network`` over its continuous params.

    Holds the topology fixed and optimises every identifiable branch
    length and reticulation inheritance probability *in place* via
    Brent coordinate ascent, returning the optimised log likelihood.

    The objective is evaluated through ``scorer`` so it shares the one
    :class:`~phynetpy._mcmc_gt._GTLikelihoodEngine` the search already
    maintains.  Each evaluation flags a single network node dirty, which
    routes the engine through its *partial* invalidation path (keep the
    per-topology network index, drop only the length/gamma-dependent
    memo) -- the reason per-parameter line searches stay cheap.

    Args:
        model: Live model whose ``network`` is optimised in place.
        scorer: An :class:`MCMCGTScorer` (``posterior=False``) bound to
            ``model`` via ``model.set_likelihood_calculator``.
        mapping: Species -> allele-label map (used to skip
            unidentifiable single-allele pendant edges).
        max_rounds: Maximum coordinate-ascent sweeps over all params.
        improve_threshold: Stop once a full sweep improves the log
            likelihood by less than this.
        min_branch, max_branch: Inclusive bounds on each branch length
            (coalescent units).  ``max_branch`` mirrors PhyloNet's
            ``-l`` default of 6.
        branch_iters: Brent evaluation cap per parameter per sweep.

    Returns:
        The maximised MSNC log likelihood.
    """
    net = model.network
    allele_counts = _allele_counts(mapping)
    length_edges, gamma_pairs = _collect_continuous_params(net, allele_counts)

    # A stable, non-empty node set so the scorer takes the engine's
    # cheap partial-invalidation path on every re-score below.
    dirty_anchor = {net.root()} if net.V() else set()

    def evaluate() -> float:
        model.mark_touched(set(dirty_anchor))
        val = scorer(model)
        return val if math.isfinite(val) else _LOG_FLOOR

    best = evaluate()
    if not length_edges and not gamma_pairs:
        return best

    for _ in range(max(1, max_rounds)):
        round_start = best

        # ---- branch lengths -------------------------------------------------
        for e in length_edges:
            saved = e.get_length()

            def neg(x: float, _e=e) -> float:
                _e.set_length(float(x))
                return -evaluate()

            x_best, neg_val = _line_minimize(
                neg, min_branch, max_branch,
                max_iter=branch_iters, xatol=_BRANCH_LINE_XATOL,
            )
            cand = -neg_val
            if cand > best:
                e.set_length(float(x_best))
                best = cand
            else:
                # Brent found nothing better than the incumbent; restore.
                e.set_length(saved if saved is not None else float(x_best))

        # ---- inheritance probabilities -------------------------------------
        for e0, e1 in gamma_pairs:
            saved0 = e0.get_gamma()
            saved1 = e1.get_gamma()

            def neg(g: float, _e0=e0, _e1=e1) -> float:
                _e0.set_gamma(float(g))
                _e1.set_gamma(float(1.0 - g))
                return -evaluate()

            g_best, neg_val = _line_minimize(
                neg, _GAMMA_EPS, 1.0 - _GAMMA_EPS,
                max_iter=branch_iters, xatol=_BRANCH_LINE_XATOL,
            )
            cand = -neg_val
            if cand > best:
                e0.set_gamma(float(g_best))
                e1.set_gamma(float(1.0 - g_best))
                best = cand
            else:
                e0.set_gamma(saved0 if saved0 is not None else float(g_best))
                e1.set_gamma(
                    saved1 if saved1 is not None else float(1.0 - g_best)
                )

        if best - round_start < improve_threshold:
            break

    return best


# ======================================================================
# Topology de-duplication for the top-n leaderboard
# ======================================================================

def _topology_signature(net: Network) -> frozenset:
    """Branch-length-free signature identifying a network *topology*.

    Each directed edge is encoded as the pair of frozensets of leaf
    labels reachable below its source and below its destination.  Two
    networks share a signature iff they induce the same set of
    leaf-labelled edges (including reticulation edges), which is what we
    want for de-duplicating the top-n leaderboard regardless of the
    fitted branch lengths / gammas.

    Args:
        net: Network to fingerprint.

    Returns:
        A hashable signature suitable for use as a dict key.
    """
    # Memoised post-order leaf-descendant sets.
    below: dict[Node, frozenset] = {}

    def leaves_below(node: Node) -> frozenset:
        cached = below.get(node)
        if cached is not None:
            return cached
        out = net.out_edges(node)
        if not out:
            result = frozenset({node.label})
        else:
            acc: set = set()
            for e in out:
                acc |= leaves_below(e.dest)
            result = frozenset(acc)
        below[node] = result
        return result

    edges_sig = set()
    for e in net.E():
        edges_sig.add((leaves_below(e.src), leaves_below(e.dest)))
    return frozenset(edges_sig)


# ======================================================================
# Main inference driver
# ======================================================================

class InferNetwork_ML:
    """Maximum-likelihood species-network inference from gene-tree topologies.

    Faithful, streamlined re-implementation of PhyloNet's
    ``InferNetwork_ML`` (Yu et al. 2014) layered over PhyNetPy's MSNC
    likelihood engine.  Mirrors the public surface of :class:`MPL` and
    :class:`MCMC_GT`:

    * :meth:`score` -- one-off log likelihood of the held network
      (optionally after optimising its continuous parameters).
    * :meth:`search` -- hill-climbing search (with random restarts and
      per-topology parameter optimisation) returning an
      :class:`InferNetworkMLResult`.
    * :meth:`from_nexus` / :meth:`from_consensus` -- convenience
      constructors matching the sibling methods.

    Attributes:
        net: Current / best species network.  Updated in place by
            :meth:`search` to the maximum-likelihood network found.
        gene_trees: Input gene-tree collection (fixed across the run).
        mapping: Species -> list of gene-copy (allele) labels.
        max_reticulations: Upper bound on the reticulation count of the
            inferred network (PhyloNet's ``numReticulations``).
    """

    def __init__(
        self,
        species_net: Network,
        gene_trees: GeneTrees,
        mapping: dict[str, list[str]],
        max_reticulations: Optional[int] = None,
    ) -> None:
        """Initialise the inference object.

        Args:
            species_net: Starting species network.  Should carry branch
                lengths on every edge and gammas on reticulation
                in-edges; missing lengths are back-filled with a
                coalescent-unit default.  Consumed by reference and
                replaced in place by :meth:`search` with the best
                network found.
            gene_trees: Input gene-tree set.
            mapping: Species -> list of gene-copy labels.
            max_reticulations: Maximum number of reticulation nodes the
                search may add (``numReticulations``).  ``None`` leaves
                the count unbounded (not recommended for ML -- model
                complexity grows without penalty).
        """
        _populate_default_branch_lengths(species_net)
        self.net = species_net
        self.gene_trees = gene_trees
        self.mapping = mapping
        self.max_reticulations = max_reticulations

    # ── Constructors ──────────────────────────────────────────────

    @classmethod
    def from_nexus(
        cls,
        gt_file: str,
        st_file: str,
        mapping: dict[str, list[str]],
        max_reticulations: Optional[int] = None,
    ) -> "InferNetwork_ML":
        """Construct from a gene-tree NEXUS and a starting-network NEXUS.

        Args:
            gt_file: Path to the gene-tree NEXUS.
            st_file: Path to the starting-network NEXUS.
            mapping: Species -> list of gene-copy labels.
            max_reticulations: See :meth:`__init__`.

        Returns:
            A fully-initialised :class:`InferNetwork_ML`.
        """
        st: Network = io.read_nexus(st_file, return_type="networks")
        gts: GeneTrees = io.read_nexus(gt_file, return_type="genetrees")
        if hasattr(gts, "species_gene_mapping"):
            gts.species_gene_mapping = mapping
        return cls(st, gts, mapping, max_reticulations=max_reticulations)

    @classmethod
    def from_consensus(
        cls,
        gene_trees: GeneTrees,
        mapping: dict[str, list[str]],
        max_reticulations: Optional[int] = None,
        *,
        threshold: float = 0.5,
    ) -> "InferNetwork_ML":
        """Seed the starting topology from a majority-rule consensus tree.

        The consensus is the canonical PhyloNet starting point ("default
        value is the optimal MDC tree"); a consensus of the gene trees
        is a cheap, robust stand-in.  Polytomies are resolved by the
        first handful of search moves.

        Args:
            gene_trees: Gene-tree collection to build the consensus from.
            mapping: Species -> list of gene-copy labels.
            max_reticulations: See :meth:`__init__`.
            threshold: Support threshold for consensus clades (0.5 ==
                majority rule).

        Returns:
            A fully-initialised :class:`InferNetwork_ML`.
        """
        seed = gene_trees.build_majority_rule_consensus_tree(threshold=threshold)
        return cls(seed, gene_trees, mapping, max_reticulations=max_reticulations)

    # ── Scoring ───────────────────────────────────────────────────

    def score(
        self,
        *,
        optimize: bool = False,
        n_workers: int = 1,
        **opt_kwargs: Any,
    ) -> float:
        """Log likelihood of the currently-held network.

        Args:
            optimize: When ``True``, first optimise the network's branch
                lengths and inheritance probabilities to their ML values
                (mutating ``self.net`` in place), then report the
                optimised likelihood.  When ``False`` (default), score
                the network exactly as held.
            n_workers: Worker processes for parallel gene-tree scoring
                (forwarded to :class:`MCMCGTScorer`).  ``1`` keeps the
                serial path.
            **opt_kwargs: Forwarded to :func:`optimize_network_parameters`
                when ``optimize`` is ``True`` (e.g. ``max_rounds``,
                ``max_branch``).

        Returns:
            MSNC log likelihood (``<= 0``; higher is better).
        """
        scorer = MCMCGTScorer(
            self.gene_trees, self.mapping, posterior=False, n_workers=n_workers,
        )
        try:
            model = Model(rng=np.random.default_rng())
            model.network = self.net
            model.set_likelihood_calculator(scorer)
            if optimize:
                return optimize_network_parameters(
                    model, scorer, self.mapping, **opt_kwargs,
                )
            return float(scorer(model))
        finally:
            scorer.close()

    # ── Search ────────────────────────────────────────────────────

    def search(
        self,
        *,
        num_runs: int = 5,
        num_iter: int = 2_000,
        max_failures: int = 100,
        max_examinations: Optional[int] = None,
        optimize_params: bool = False,
        final_optimize: bool = True,
        num_networks: int = 1,
        kernel: Optional[MCMCGTKernel] = None,
        seed: Any = None,
        n_workers: int = 1,
        progress: bool = False,
        **opt_kwargs: Any,
    ) -> InferNetworkMLResult:
        """Search network space for the maximum-likelihood topology.

        Runs ``num_runs`` independent hill-climbing chains (PhyloNet's
        ``-x``).  Run 0 starts from ``self.net``; each subsequent run
        starts from a randomly perturbed copy of it (random restart) to
        escape the basin of the seed.  Within a run, moves come from a
        :class:`MCMCGTKernel`; a proposal is accepted iff it strictly
        improves the (optionally parameter-optimised) log likelihood.
        Each run stops after ``max_failures`` consecutive non-improving
        examinations (``-f``), after ``num_iter`` proposals, or once the
        global ``max_examinations`` budget (``-m``) is spent.

        Args:
            num_runs: Number of independent restarts.
            num_iter: Maximum proposals per run.
            max_failures: Consecutive non-improving examinations that end
                a run early (PhyloNet ``-f``; default 100).
            max_examinations: Optional global cap on the number of
                distinct topologies scored across all runs (PhyloNet
                ``-m``).  ``None`` == unbounded.
            optimize_params: When ``True`` (PhyloNet ``-o``), fully
                optimise branch lengths + gammas for every examined
                topology before scoring it -- slower per step, but each
                topology is judged at its own ML optimum.  When ``False``
                (default), branch lengths and gammas are *sampled* by the
                kernel's continuous moves during the climb and only the
                final best network is optimised (if ``final_optimize``).
            final_optimize: Optimise the continuous parameters of the
                overall best network once at the end (PhyloNet's
                post-search optimisation).  Always recommended so the
                returned network carries ML parameters.
            num_networks: Number of distinct top-scoring networks to
                return in ``result.networks`` (PhyloNet ``-n``).
            kernel: Custom proposal kernel.  When ``None`` a default
                :class:`MCMCGTKernel` is built honouring
                ``max_reticulations``.
            seed: Seed for all RNGs (any value accepted by
                :class:`numpy.random.SeedSequence`).
            n_workers: Worker processes for parallel gene-tree scoring.
            progress: Print a one-line summary per run when ``True``.
            **opt_kwargs: Forwarded to
                :func:`optimize_network_parameters`.

        Returns:
            An :class:`InferNetworkMLResult`.  ``self.net`` is updated in
            place to ``result.best_network``.
        """
        root_ss = (
            seed if isinstance(seed, np.random.SeedSequence)
            else np.random.SeedSequence(seed)
        )

        scorer = MCMCGTScorer(
            self.gene_trees, self.mapping, posterior=False, n_workers=n_workers,
        )

        # Leaderboard of distinct topologies, keyed by branch-length-free
        # signature -> (log_likelihood, deep-copied network).
        leaderboard: dict[frozenset, tuple[float, Network]] = {}
        examined = 0
        start = time.time()

        try:
            run_seeds = root_ss.spawn(max(1, num_runs))
            for run_idx, run_ss in enumerate(run_seeds):
                if (
                    max_examinations is not None
                    and examined >= max_examinations
                ):
                    break

                kernel_ss, model_ss, restart_ss = run_ss.spawn(3)

                # Build this run's kernel (fresh state per run unless the
                # caller supplied their own).
                run_kernel = kernel
                if run_kernel is None:
                    run_kernel = MCMCGTKernel(
                        max_reticulations=self.max_reticulations,
                        rng=np.random.default_rng(kernel_ss),
                    )
                elif getattr(run_kernel, "rng", None) is None:
                    run_kernel.rng = np.random.default_rng(kernel_ss)

                # Starting network: the seed for run 0, a perturbed copy
                # afterwards.
                start_net = copy.deepcopy(self.net)
                if run_idx > 0:
                    start_net = self._random_restart(
                        start_net,
                        run_kernel,
                        np.random.default_rng(restart_ss),
                    )

                model = Model(rng=np.random.default_rng(model_ss))
                model.network = start_net
                model.set_likelihood_calculator(scorer)

                run_examined = self._run_hill_climb(
                    model=model,
                    kernel=run_kernel,
                    scorer=scorer,
                    num_iter=num_iter,
                    max_failures=max_failures,
                    max_examinations=(
                        None if max_examinations is None
                        else max_examinations - examined
                    ),
                    optimize_params=optimize_params,
                    leaderboard=leaderboard,
                    num_networks=num_networks,
                    opt_kwargs=opt_kwargs,
                )
                examined += run_examined

                if progress:
                    best_so_far = max(
                        (v[0] for v in leaderboard.values()),
                        default=float("-inf"),
                    )
                    print(
                        f"[InferNetwork_ML] run {run_idx + 1}/{num_runs}: "
                        f"examined={run_examined} "
                        f"best_logL={best_so_far:.6f}",
                        flush=True,
                    )

            # Rank the leaderboard, optionally optimise the winner.
            ranked = sorted(
                leaderboard.values(), key=lambda t: t[0], reverse=True,
            )
            if not ranked:
                # Degenerate: no move ever improved -- score the seed.
                model = Model(rng=np.random.default_rng())
                model.network = copy.deepcopy(self.net)
                model.set_likelihood_calculator(scorer)
                base = float(scorer(model))
                ranked = [(base, model.network)]

            best_score, best_net = ranked[0]
            if final_optimize:
                model = Model(rng=np.random.default_rng())
                model.network = best_net
                model.set_likelihood_calculator(scorer)
                best_score = optimize_network_parameters(
                    model, scorer, self.mapping, **opt_kwargs,
                )
                best_net = model.network

            result = InferNetworkMLResult(
                best_network=best_net,
                best_log_likelihood=best_score,
                num_reticulations=self._count_reticulations(best_net),
                networks=[(n, s) for (s, n) in ranked[:max(1, num_networks)]],
                num_networks_examined=examined,
                num_runs=num_runs,
                wall_time_sec=time.time() - start,
            )
        finally:
            scorer.close()

        self.net = result.best_network
        return result

    # ── Internal ──────────────────────────────────────────────────

    def _run_hill_climb(
        self,
        *,
        model: Model,
        kernel: MCMCGTKernel,
        scorer: MCMCGTScorer,
        num_iter: int,
        max_failures: int,
        max_examinations: Optional[int],
        optimize_params: bool,
        leaderboard: dict[frozenset, tuple[float, Network]],
        num_networks: int,
        opt_kwargs: dict[str, Any],
    ) -> int:
        """Execute one hill-climbing run; update ``leaderboard`` in place.

        Greedy local search: propose a move, (optionally) optimise the
        proposal's continuous parameters, accept iff it strictly beats
        the incumbent log likelihood, otherwise undo.  Tracks consecutive
        failures for early stopping.

        Args:
            model: Live model for this run (network already set).
            kernel: Proposal kernel.
            scorer: Shared likelihood scorer.
            num_iter: Maximum proposals.
            max_failures: Consecutive non-improving examinations to stop.
            max_examinations: Remaining global examination budget (or
                ``None``).
            optimize_params: Optimise continuous params per examined
                topology when ``True``.
            leaderboard: Shared distinct-topology leaderboard.
            num_networks: Size of the leaderboard to retain.
            opt_kwargs: Forwarded to :func:`optimize_network_parameters`.

        Returns:
            Number of topologies examined in this run.
        """
        # Establish the incumbent score (optimise the start if requested
        # so run 0 begins from its own ML optimum).
        if optimize_params:
            cur_score = optimize_network_parameters(
                model, scorer, self.mapping, **opt_kwargs,
            )
        else:
            cur_score = float(scorer(model))

        self._record(leaderboard, model.network, cur_score, num_networks)

        failures = 0
        examined = 0

        for _ in range(num_iter):
            if max_failures is not None and failures >= max_failures:
                break
            if max_examinations is not None and examined >= max_examinations:
                break

            move = kernel.generate(model)
            try:
                move.execute(model)
            except Exception:
                # Illegal move on this topology: undo defensively and
                # count it as a (cheap) failure.
                try:
                    move.undo(model)
                except Exception:
                    pass
                kernel.report_outcome(False, delta=0.0)
                failures += 1
                continue

            examined += 1
            try:
                if optimize_params:
                    prop_score = optimize_network_parameters(
                        model, scorer, self.mapping, **opt_kwargs,
                    )
                else:
                    prop_score = float(scorer(model))
            except Exception:
                try:
                    move.undo(model)
                except Exception:
                    pass
                kernel.report_outcome(False, delta=0.0)
                failures += 1
                continue

            delta = prop_score - cur_score
            if delta > 0.0:
                # Strict improvement: commit (hill climbing).
                kernel.report_outcome(True, delta=delta)
                cur_score = prop_score
                failures = 0
                self._record(
                    leaderboard, model.network, prop_score, num_networks,
                )
            else:
                try:
                    move.undo(model)
                except Exception:
                    pass
                kernel.report_outcome(False, delta=delta)
                failures += 1

        return examined

    def _random_restart(
        self,
        net: Network,
        kernel: MCMCGTKernel,
        rng: np.random.Generator,
        *,
        n_moves: int = 6,
    ) -> Network:
        """Perturb ``net`` with a few random kernel moves for a restart.

        Applies up to ``n_moves`` accepted-blindly moves to diversify the
        starting topology of runs after the first.  Moves that fail to
        execute are skipped.  The kernel's own RNG drives proposal
        content; ``rng`` only gates how many succeed.

        Args:
            net: Network to perturb (mutated in place and returned).
            kernel: Proposal kernel to draw moves from.
            rng: Generator (reserved for future jitter; currently unused
                beyond seeding determinism via the caller).
            n_moves: Target number of random moves to apply.

        Returns:
            The perturbed network.
        """
        scratch = Model(rng=rng)
        scratch.network = net
        applied = 0
        attempts = 0
        while applied < n_moves and attempts < n_moves * 4:
            attempts += 1
            move = kernel.generate(scratch)
            try:
                move.execute(scratch)
                applied += 1
            except Exception:
                try:
                    move.undo(scratch)
                except Exception:
                    pass
        return scratch.network

    def _record(
        self,
        leaderboard: dict[frozenset, tuple[float, Network]],
        net: Network,
        score: float,
        num_networks: int,
    ) -> None:
        """Insert ``net`` into the distinct-topology leaderboard.

        Keeps the best score per distinct topology and trims the
        leaderboard to the ``num_networks`` highest-scoring entries.
        Networks are deep-copied on insertion so later in-place moves
        don't corrupt stored candidates.

        Args:
            leaderboard: Signature -> (score, network) map (mutated).
            net: Candidate network (will be deep-copied if retained).
            score: Its log likelihood.
            num_networks: Maximum leaderboard size to keep.
        """
        sig = _topology_signature(net)
        existing = leaderboard.get(sig)
        if existing is not None and existing[0] >= score:
            return
        leaderboard[sig] = (score, copy.deepcopy(net))

        # Trim to the top ``num_networks`` (keep a little slack so a
        # transiently-low topology can still climb back in).
        cap = max(1, num_networks)
        if len(leaderboard) > cap:
            worst = sorted(leaderboard.items(), key=lambda kv: kv[1][0])
            for sig_drop, _ in worst[: len(leaderboard) - cap]:
                del leaderboard[sig_drop]

    @staticmethod
    def _count_reticulations(net: Network) -> int:
        """Number of reticulation nodes in ``net``."""
        return sum(1 for v in net.V() if v.is_reticulation())
