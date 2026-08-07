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
:class:`~phynetpy._mcmc_gt.MCMC_GT` method already drives that engine with an
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

This class is an implementation detail, not the public API.  Reach it through
the ``(GeneTrees, MSC, Likelihood)`` cell of the inference matrix::

    from phynetpy.criteria import Likelihood
    from phynetpy.infer import infer

    result = infer(gene_trees, criterion=Likelihood(), max_reticulations=1)
    print(result.best, result.score)

Its own surface (mirroring :class:`~phynetpy._mpl.MPL` /
:class:`~phynetpy._mcmc_gt.MCMC_GT`) stays available for engine authors::

    from phynetpy._infernetworkml import InferNetwork_ML

    inf = InferNetwork_ML(starting_net, gene_trees, mapping, max_reticulations=1)
    result = inf.search(num_runs=5, optimize_params=True)
    print(result.best_network, result.best_log_likelihood)

Docs   - [x]
Tests  - [ ]
Design - [x]
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .Network import Network
from .GeneTrees import GeneTrees
from .GraphUtils import (
    count_reticulations,
    level as _network_level,
    network_clusters,
    _leaf_labels_below,
)
from .ModelGraph import Model
from . import IO as io

# Re-use MCMC_GT's likelihood scorer (posterior=False -> plain MSNC log
# likelihood) and proposal kernel.
from ._mcmc_gt import (
    MCMCGTScorer,
    MCMCGTKernel,
    _populate_default_branch_lengths,
)

# Continuous-parameter (branch length + gamma) optimiser.  Factored into a
# scorer-agnostic module (``_optimize``) so MPL / MCMC_GT / InferNetwork_ML
# can share the ``opt_bl`` behaviour without circular imports.  Re-exported
# here for backward compatibility
# (``from phynetpy.infer import optimize_network_parameters``).
from ._optimize import optimize_network_parameters
from ._search_flags import resolve_move_types, resolve_search_preset


__all__ = [
    "InferNetwork_ML",
    "InferNetworkMLResult",
    "optimize_network_parameters",
]


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
    below = _leaf_labels_below(net)
    return frozenset((below[e.src], below[e.dest]) for e in net.E())


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
        pseudo: bool = False,
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
            pseudo: Pseudo-likelihood scoring (``pseudo`` flag).  When
                ``True`` the full MSNC scorer is swapped for the triplet
                :class:`MPLScorer` (Yu & Nakhleh 2015); the returned value
                is then a log pseudo-likelihood.
            n_workers: Worker processes for parallel gene-tree scoring
                (forwarded to :class:`MCMCGTScorer`).  ``1`` keeps the
                serial path.
            **opt_kwargs: Forwarded to :func:`optimize_network_parameters`
                when ``optimize`` is ``True`` (e.g. ``max_rounds``,
                ``max_branch``).

        Returns:
            MSNC log likelihood (``<= 0``; higher is better), or the log
            pseudo-likelihood when ``pseudo=True``.
        """
        if pseudo:
            from ._mpl import compute_gene_tree_triplets, MPLScorer
            triplet_result = compute_gene_tree_triplets(
                self.gene_trees, self.mapping,
            )
            scorer = MPLScorer(
                triplet_result.rho_by_triplet, triplet_result.triplets,
            )
        else:
            scorer = MCMCGTScorer(
                self.gene_trees, self.mapping,
                posterior=False, n_workers=n_workers,
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
            close_fn = getattr(scorer, "close", None)
            if callable(close_fn):
                close_fn()

    # ── Search ────────────────────────────────────────────────────

    def search(
        self,
        *,
        num_runs: int = 5,
        num_iter: int = 2_000,
        max_failures: int = 100,
        max_examinations: Optional[int] = None,
        preset: str = "default",
        optimize_params: Optional[bool] = None,
        optimize_scope: Optional[str] = None,
        final_optimize: Optional[bool] = None,
        num_networks: int = 1,
        kernel: Optional[MCMCGTKernel] = None,
        opt_bl: Optional[bool] = None,
        fix_st: Optional[bool] = None,
        max_lvl: Optional[int] = None,
        pseudo: bool = False,
        backbone: Optional[Network] = None,
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
            preset: One-word search profile (see
                :data:`phynetpy._search_flags.SEARCH_PRESETS`):
                ``"default"`` (recommended), ``"fast"``, ``"accurate"``,
                or ``"phylonet"``.  Sets ``optimize_params`` /
                ``optimize_scope`` / ``opt_bl`` / ``fix_st`` /
                ``final_optimize`` as a coherent bundle; any of those
                passed explicitly overrides the preset.
            optimize_scope: Which parameters per-topology optimisation
                touches (``"gamma"`` / ``"reticulation"`` / ``"all"``);
                see :func:`phynetpy._optimize.optimize_network_parameters`.
                Defaults to ``None``; because ``InferNetwork_ML`` maximises
                the full MSNC likelihood (every branch length is
                identifiable), an unset scope resolves to the full
                ``"all"`` scope -- correctness over runtime -- rather than
                the cheaper ``"gamma"`` MPL default.  Pass an explicit
                value to narrow it.  The end-of-search optimisation always
                uses the full ``"all"`` scope regardless of this value.
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
            opt_bl: Optimise branch lengths (``opt_bl`` flag).  Forces
                ``final_optimize=True`` and drops the continuous-parameter
                moves from the default kernel so the climb is a pure
                topology search with one Brent optimisation of the winner
                at the end.  (InferNetwork_ML already optimises; this just
                makes that behaviour explicit and removes redundant
                continuous proposals.)  Ignored when a custom ``kernel``
                is supplied (final optimisation still runs).
            fix_st: Fix the starting-tree backbone (``fix_st`` flag).
                Drops ``SPR`` from the default kernel.  Ignored when a
                custom ``kernel`` is supplied.
            max_lvl: Maximum network level (``max_lvl`` flag).  Proposals
                exceeding this level are rejected by the inline hill-climb
                guard; level-raising moves also self-reject early.
                ``None`` disables the cap.
            pseudo: Pseudo-likelihood scoring (``pseudo`` flag).  When
                ``True`` the full MSNC scorer is swapped for the triplet
                :class:`MPLScorer` (Yu & Nakhleh 2015); the same scorer is
                used for the (final / per-topology) Brent optimisation.
            backbone: Network the result must contain as a subgraph.  Drops
                every move that could destroy existing structure and
                enforces containment in the accept path.  Set by
                ``infer(start=Start(net, mode=StartMode.AUGMENT))``; leave
                ``None`` for an unconstrained search.
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

        # Resolve the preset into concrete behaviour; explicit flags win.
        explicit_scope = optimize_scope
        settings = resolve_search_preset(
            preset,
            optimize_params=optimize_params,
            optimize_scope=optimize_scope,
            opt_bl=opt_bl,
            fix_st=fix_st,
            final_optimize=final_optimize,
        )
        optimize_params = settings.optimize_params
        optimize_scope = settings.optimize_scope
        opt_bl = settings.opt_bl
        fix_st = settings.fix_st
        final_optimize = settings.final_optimize

        # InferNetwork_ML maximises the *full* MSNC likelihood, where every
        # branch length is identifiable and matters -- so unless the caller
        # explicitly narrows the scope, per-topology optimisation uses the
        # full ``"all"`` scope (correctness over runtime).  MPL keeps the
        # cheaper preset scope (``"gamma"``) since its triplet objective is
        # far less branch-length-sensitive.
        if explicit_scope is None:
            optimize_scope = "all"

        # opt_bl: a pure-topology climb with one final optimisation.  It
        # forces final_optimize and drops the continuous-parameter moves
        # from the default kernel (see move_types below).
        if opt_bl:
            final_optimize = True

        # pseudo: swap the full MSNC scorer for the triplet
        # pseudo-likelihood.  The same scorer drives the Brent optimiser.
        if pseudo:
            from ._mpl import compute_gene_tree_triplets, MPLScorer
            triplet_result = compute_gene_tree_triplets(
                self.gene_trees, self.mapping,
            )
            scorer = MPLScorer(
                triplet_result.rho_by_triplet, triplet_result.triplets,
            )
        else:
            scorer = MCMCGTScorer(
                self.gene_trees, self.mapping,
                posterior=False, n_workers=n_workers,
            )

        move_types = resolve_move_types(
            opt_bl=opt_bl, fix_st=fix_st, augment_only=backbone is not None,
        )
        required_clusters = (
            network_clusters(backbone) if backbone is not None else None
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
                        move_types=move_types,
                        max_level=max_lvl,
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
                    optimize_scope=optimize_scope,
                    leaderboard=leaderboard,
                    num_networks=num_networks,
                    opt_kwargs=opt_kwargs,
                    max_lvl=max_lvl,
                    required_clusters=required_clusters,
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
                num_reticulations=count_reticulations(best_net),
                networks=[(n, s) for (s, n) in ranked[:max(1, num_networks)]],
                num_networks_examined=examined,
                num_runs=num_runs,
                wall_time_sec=time.time() - start,
            )
        finally:
            # ``MPLScorer`` (pseudo path) has no ``close``; guard it.
            close_fn = getattr(scorer, "close", None)
            if callable(close_fn):
                close_fn()

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
        max_lvl: Optional[int] = None,
        optimize_scope: str = "all",
        required_clusters: Optional[set] = None,
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
            max_lvl: Authoritative cap on network level.
            optimize_scope: Which continuous parameters per-topology
                optimisation touches.
            required_clusters: Leaf-label clusters every accepted proposal
                must still display (backbone containment).  ``None`` leaves
                the climb unconstrained.

        Returns:
            Number of topologies examined in this run.
        """
        # Establish the incumbent score (optimise the start if requested
        # so run 0 begins from its own ML optimum).
        if optimize_params:
            cur_score = optimize_network_parameters(
                model, scorer, self.mapping, scope=optimize_scope, **opt_kwargs,
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

            # Authoritative ``max_lvl`` guard: reject any proposal whose
            # network level exceeds the cap, regardless of which move
            # produced it (covers reticulation relocation / endpoint moves
            # that raise level without changing the reticulation count).
            if (
                max_lvl is not None and _network_level(model.network) > max_lvl
            ) or (
                required_clusters is not None
                and not required_clusters <= network_clusters(model.network)
            ):
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
                        model, scorer, self.mapping,
                        scope=optimize_scope, **opt_kwargs,
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
