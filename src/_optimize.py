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

Continuous-parameter (branch length + inheritance probability) optimisation
for phylogenetic networks.

This module holds the Brent coordinate-ascent optimiser that was originally
embedded in :mod:`phynetpy._infernetworkml`.  It is factored out into a
scorer-agnostic module so that *every* topology-based inference method
(``MPL``, ``MCMC_GT``, ``InferNetwork_ML``) can reuse the same
"optimise branch lengths at the end of the search" behaviour (the ``opt_bl``
flag) without circular imports.

The objective is supplied by the caller as any ``Callable[[Model], float]``
(the inference method's own scorer), so the optimiser maximises whatever the
search itself optimised -- the full MSNC likelihood, the triplet
pseudo-likelihood, etc.

Docs   - [x]
Tests  - [ ]
Design - [x]
"""

from __future__ import annotations

import math
from typing import Any, Callable

from .Network import Network
from .ModelGraph import Model

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
    "optimize_network_parameters",
    "snapshot_continuous_params",
    "restore_continuous_params",
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

# Likelihood floor shared with the scorers: keep degenerate evaluations
# finite so the optimiser never compares against -inf.  Defined locally
# (rather than imported) to avoid a circular import with the scorer
# modules that themselves call this optimiser.
_LOG_FLOOR: float = math.log(1e-200)


# ======================================================================
# Continuous-parameter optimisation (the "-o" / ``opt_bl`` behaviour)
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
    *,
    reticulation_only: bool = False,
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
        reticulation_only: When ``True`` restrict ``length_edges`` to the
            branches *incident to a reticulation node* (its in-edges and
            out-edges) rather than every identifiable edge.  This is the
            cheap per-topology optimisation scope: it tunes exactly the
            parameters a reticulation move just perturbed (whose optimum
            is what decides accept/reject for r>=1 topologies) while
            leaving the rest of the backbone to the one-time final
            optimisation.  ``gamma_pairs`` is unaffected.

    Returns:
        ``(length_edges, gamma_pairs)``.
    """
    retic_nodes = {v for v in net.V() if v.is_reticulation()}

    if reticulation_only:
        # Only branches touching a reticulation node (in- or out-edges).
        candidate_edges = [
            e for e in net.E()
            if e.src in retic_nodes or e.dest in retic_nodes
        ]
    else:
        candidate_edges = list(net.E())

    length_edges: list[Any] = []
    for e in candidate_edges:
        dest = e.dest
        if net.out_degree(dest) == 0:
            # Pendant edge: only identifiable when >1 allele can coalesce
            # within the species branch.
            if allele_counts.get(dest.label, 1) <= 1:
                continue
        length_edges.append(e)

    gamma_pairs: list[tuple[Any, Any]] = []
    for v in retic_nodes:
        # ``in_edges`` may return a set; materialise before indexing.
        in_edges = list(net.in_edges(v))
        if len(in_edges) >= 2:
            gamma_pairs.append((in_edges[0], in_edges[1]))

    return length_edges, gamma_pairs


# ======================================================================
# Reversible parameter snapshots
# ======================================================================
# Per-topology optimisation mutates branch lengths / gammas in place.
# When a search driver rejects the proposal it must restore the exact
# pre-optimisation continuous state (the structural ``Move.undo`` only
# reverses the topology edit, not the optimiser's parameter sweep).
# These helpers capture and replay that state by *edge identity* -- safe
# because optimisation never adds or removes edges, only re-values them.

def snapshot_continuous_params(
    net: Network,
) -> list[tuple[Any, Any, Any]]:
    """Capture every edge's ``(length, gamma)`` for later restoration.

    Args:
        net: Network whose continuous parameters should be snapshotted.

    Returns:
        A list of ``(edge, length, gamma)`` tuples keyed by live edge
        object -- pass it to :func:`restore_continuous_params` to undo a
        parameter sweep.
    """
    return [(e, e.get_length(), e.get_gamma()) for e in net.E()]


def restore_continuous_params(
    snapshot: list[tuple[Any, Any, Any]],
) -> None:
    """Restore edge lengths / gammas captured by :func:`snapshot_continuous_params`.

    Args:
        snapshot: The list returned by :func:`snapshot_continuous_params`.
            Edges absent from the current network (e.g. removed by a
            subsequent structural undo) are simply re-valued harmlessly.
    """
    for e, length, gamma in snapshot:
        if length is not None:
            e.set_length(length)
        if gamma is not None:
            e.set_gamma(gamma)


def optimize_network_parameters(
    model: Model,
    scorer: Callable[[Model], float],
    mapping: dict[str, list[str]],
    *,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    improve_threshold: float = _DEFAULT_IMPROVE_THRESHOLD,
    min_branch: float = _MIN_BRANCH_LENGTH,
    max_branch: float = _MAX_BRANCH_LENGTH,
    branch_iters: int = _BRANCH_LINE_ITERS,
    scope: str = "all",
) -> float:
    """Maximise the score of ``model.network`` over its continuous params.

    Holds the topology fixed and optimises every identifiable branch
    length and reticulation inheritance probability *in place* via
    Brent coordinate ascent, returning the optimised score.

    The objective is evaluated through ``scorer`` -- any
    ``Callable[[Model], float]`` (e.g. an ``MCMCGTScorer`` for the full
    MSNC likelihood or an ``MPLScorer`` for the triplet
    pseudo-likelihood).  Each evaluation flags a single network node
    dirty, which routes a likelihood engine that supports incremental
    invalidation (such as :class:`~phynetpy._mcmc_gt._GTLikelihoodEngine`)
    through its *partial* invalidation path -- the reason per-parameter
    line searches stay cheap.  Scorers without that fast path simply
    rescore from scratch (still correct, just not as fast).

    Args:
        model: Live model whose ``network`` is optimised in place.
        scorer: A callable bound to ``model`` via
            ``model.set_likelihood_calculator``.
        mapping: Species -> allele-label map (used to skip
            unidentifiable single-allele pendant edges).
        max_rounds: Maximum coordinate-ascent sweeps over all params.
        improve_threshold: Stop once a full sweep improves the score by
            less than this.
        min_branch, max_branch: Inclusive bounds on each branch length
            (coalescent units).  ``max_branch`` mirrors PhyloNet's
            ``-l`` default of 6.
        branch_iters: Brent evaluation cap per parameter per sweep.
        scope: Which parameters to optimise.  ``"all"`` (default)
            optimises every identifiable edge plus all gammas -- the
            behaviour used for the one-time final optimisation.
            ``"reticulation"`` optimises the gammas and the branches
            incident to reticulation nodes.  ``"gamma"`` optimises only
            the inheritance probabilities (no branch lengths) -- the
            cheapest per-topology scope, and a no-op for trees (r=0).
            Both restricted scopes leave the remaining branch lengths to
            the one-time final optimisation.

    Returns:
        The maximised score.
    """
    net = model.network
    allele_counts = _allele_counts(mapping)
    length_edges, gamma_pairs = _collect_continuous_params(
        net, allele_counts, reticulation_only=(scope == "reticulation"),
    )
    if scope == "gamma":
        # Gamma-only: leave every branch length to the final optimisation.
        # This is the cheapest in-loop scope and directly fixes the r>=1
        # "reticulation judged at gamma=0.5" bias.
        length_edges = []

    # A stable, non-empty node set so the scorer takes the engine's
    # cheap partial-invalidation path on every re-score below.
    dirty_anchor = {net.root()} if net.V() else set()

    def evaluate() -> float:
        """Re-score ``model.network`` after invalidating the dirty anchor, clamping non-finite results."""
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
                """Negated score with edge ``_e``'s length set to ``x`` (for Brent minimisation)."""
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
                """Negated score with the reticulation's gamma pair set to ``(g, 1-g)`` (for Brent minimisation)."""
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
