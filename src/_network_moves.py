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

Shared proposal-math for MCMC over phylogenetic-network space.

This module is the **single source of truth** for the proposal densities,
Hastings ratios and reversible-jump Jacobians used by every PhyNetPy
network sampler.  Previously this math lived in two unrelated places --
:mod:`phynetpy._mcmc_seq` (correct, log-space, with RJMCMC terms) and
:mod:`phynetpy.ModelMove` (an SA-oriented shortcut that returned ``1.0``
for everything).  Maintaining the math twice meant the gene-tree / MPL
samplers silently drifted out of correctness.  Both stacks now delegate
here.

Unified move contract
----------------------

Every proposal, regardless of which stack drives it, conceptually returns

    (log_hastings_ratio, undo)

where ``log_hastings_ratio`` is in **log space** (``0.0`` for a symmetric
proposal) and already folds in any reversible-jump log-Jacobian.  A driver
accepts a proposal with

    log_alpha = (log_pi_proposed - log_pi_current) + log_hastings_ratio
    accept    = log_alpha >= 0 or log(U(0,1)) < log_alpha

The sequence sampler (:mod:`phynetpy._mcmc_seq`) realises this contract via
``op_*`` closures; the gene-tree / MPL sampler realises it via
:class:`phynetpy.ModelMove.Move` objects that cache the value computed in
``execute`` on ``self._log_hr``.  Both call the pure functions below so the
formulas can never disagree.

Functions here are deliberately pure (numbers in, numbers out) and have no
dependency on ``Network``, ``SeqState`` or ``Model`` so they can be unit
tested in isolation and reused from either stack.
"""

from __future__ import annotations

import math
import warnings

__all__ = [
    "add_reticulation_log_hastings",
    "remove_reticulation_log_hastings",
    "normal_log_pdf",
    "standard_normal_cdf",
    "truncated_normal_log_norm",
    "random_walk_truncated_log_hastings",
    "gaussian_delta_log_hastings",
    "weighted_choice_log_hastings",
    "LargeAnalysisWarning",
    "MCMC_TAXA_ADVISORY",
    "warn_if_large_mcmc",
]

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


# ======================================================================
# Tractability advisory for large MCMC analyses
# ======================================================================
#
# MCMC sampling of phylogenetic *networks* scales badly with the number of
# taxa: the topology space grows super-exponentially, chains mix poorly
# (low ESS / high Gelman-Rubin R-hat), and the per-iteration MSNC cost
# climbs steeply.  Past roughly this many taxa a converged posterior in any
# practical number of iterations is unlikely.  We do **not** forbid such a
# run -- a user who knows what they are doing may proceed -- but we emit a
# loud one-time warning advising against it.

MCMC_TAXA_ADVISORY = 20


class LargeAnalysisWarning(UserWarning):
    """Advisory that an MCMC analysis is large enough to be intractable.

    A subclass of :class:`UserWarning` so callers can silence or escalate it
    selectively, e.g. ``warnings.filterwarnings("ignore",
    category=phynetpy._network_moves.LargeAnalysisWarning)``.
    """


def warn_if_large_mcmc(n_taxa: int, *, method: str = "MCMC") -> bool:
    """Warn (without blocking) when an MCMC run is likely intractable.

    Args:
        n_taxa: Number of taxa (species) in the analysis.
        method: Sampler name used in the message (e.g. ``"MCMC_SEQ"``).

    Returns:
        ``True`` when a warning was emitted (``n_taxa >= MCMC_TAXA_ADVISORY``),
        ``False`` otherwise.  The run is never blocked either way.
    """
    if n_taxa < MCMC_TAXA_ADVISORY:
        return False

    warnings.warn(
        f"{method}: {n_taxa} taxa requested. MCMC sampling over phylogenetic "
        f"network space is generally intractable beyond ~{MCMC_TAXA_ADVISORY} "
        f"taxa -- the topology space grows super-exponentially, chains tend to "
        f"mix poorly (low ESS / high R-hat), and the per-iteration likelihood "
        f"cost climbs steeply. The analysis will still run, but it may not "
        f"converge in any practical number of iterations and the posterior "
        f"may be unreliable. Strongly consider: fewer taxa, a constrained or "
        f"fixed backbone topology, or a non-MCMC method (e.g. maximum "
        f"pseudo-likelihood via MPL / InferNetworkML).",
        category=LargeAnalysisWarning,
        stacklevel=3,
    )
    return True


# ======================================================================
# Reversible-jump reticulation moves (PhyloNet MCMC_SEQ formulation)
# ======================================================================
#
# These mirror PhyloNet's ``AddReticulation`` / ``DeleteReticulation``
# acceptance terms exactly.  The add and delete ratios are constructed to
# be exact negatives of each other (so an add immediately followed by the
# matching delete has a combined log-Hastings of ~0), which is what makes
# the dimension-changing pair reversible.  See
# ``phynetpy._mcmc_seq.op_add_reticulation`` for the canonical derivation
# and ``tests/test_mcmc_seq.py::TestReticulationMoves`` for the invariance
# check.


def add_reticulation_log_hastings(
    *, edge_count_pre: int, retic_count_pre: int, l1: float, l2: float
) -> float:
    r"""Log Hastings ratio (incl. RJ Jacobian) for adding a reticulation.

    Two distinct edges are picked uniformly without replacement from the
    pre-move network (a *donor* edge of length ``l1`` and a *reticulation*
    edge of length ``l2``); each is split at a uniformly drawn point and a
    hybrid edge with ``gamma ~ U(0, 1)`` is inserted.  The log Hastings
    ratio is

    .. math::

        \log\!\Big( \mathrm{pda} \cdot l_1 \cdot l_2 \cdot
                    \frac{E (E - 1)}{2 (R + 1)} \Big)

    where ``E`` is the pre-move edge count, ``R`` the pre-move reticulation
    count, and ``pda = 0.5`` only when adding the first reticulation
    (``R == 0``), else ``1.0``.

    Args:
        edge_count_pre: ``E``, number of edges before the move.
        retic_count_pre: ``R``, number of reticulations before the move.
        l1: Length of the donor edge that was split.
        l2: Length of the reticulation edge that was split.

    Returns:
        The log Hastings ratio.  ``-inf`` is never returned; callers must
        reject degenerate (non-positive ``l1`` / ``l2`` or ``E < 2``)
        proposals *before* calling this.
    """
    E = int(edge_count_pre)
    R = int(retic_count_pre)
    if E < 2 or l1 <= 0.0 or l2 <= 0.0:
        raise ValueError(
            "add_reticulation_log_hastings requires E >= 2 and positive "
            f"l1, l2 (got E={E}, l1={l1}, l2={l2})."
        )
    pda = 0.5 if R == 0 else 1.0
    return (
        math.log(pda)
        + math.log(l1)
        + math.log(l2)
        + math.log(E)
        + math.log(E - 1)
        - math.log(2.0 * (R + 1))
    )


def remove_reticulation_log_hastings(
    *, edge_count_post: int, retic_count_pre: int, l1: float, l2: float
) -> float:
    r"""Log Hastings ratio (incl. RJ Jacobian) for deleting a reticulation.

    Exact negative of :func:`add_reticulation_log_hastings` evaluated at the
    reverse-add quantities:

    .. math::

        \log\!\Big( \mathrm{pad} \cdot \frac{2 R}
                    {l_1 \cdot l_2 \cdot E'(E' - 1)} \Big)

    where ``R`` is the pre-move reticulation count, ``E'`` the *post*-move
    edge count, ``l1`` / ``l2`` the lengths of the two edges that result
    from suppressing the degree-2 nodes left by the deletion (i.e. the two
    edges a reverse add would re-split), and ``pad = 2`` only when removing
    the last reticulation (``R == 1``), else ``1.0``.

    Args:
        edge_count_post: ``E'``, number of edges after the move.
        retic_count_pre: ``R``, number of reticulations before the move.
        l1: Length of the first merged edge.
        l2: Length of the second merged edge.

    Returns:
        The log Hastings ratio.
    """
    Ep = int(edge_count_post)
    R = int(retic_count_pre)
    if Ep < 2 or R < 1 or l1 <= 0.0 or l2 <= 0.0:
        raise ValueError(
            "remove_reticulation_log_hastings requires E' >= 2, R >= 1 and "
            f"positive l1, l2 (got E'={Ep}, R={R}, l1={l1}, l2={l2})."
        )
    pad = 2.0 if R == 1 else 1.0
    return (
        math.log(pad)
        + math.log(2.0 * R)
        - math.log(l1)
        - math.log(l2)
        - math.log(Ep)
        - math.log(Ep - 1)
    )


# ======================================================================
# Truncated-Gaussian random-walk helpers (bounded continuous parameters)
# ======================================================================


def normal_log_pdf(x: float, mu: float, sigma: float) -> float:
    """Log density of ``N(mu, sigma^2)`` evaluated at ``x``."""
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive (got {sigma}).")
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma) - _LOG_SQRT_2PI


def standard_normal_cdf(x: float) -> float:
    """Standard-normal CDF ``Phi(x)`` (via ``math.erf``; no SciPy needed)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def truncated_normal_log_norm(
    mu: float, sigma: float, lo: float, hi: float
) -> float:
    r"""Log normalising constant of ``N(mu, sigma^2)`` truncated to ``[lo, hi]``.

    ``Z = Phi((hi - mu)/sigma) - Phi((lo - mu)/sigma)``.  This is the term
    that makes a clamped/reflected Gaussian random walk into a *proper*
    truncated-Gaussian proposal: the Hastings ratio of such a walk is the
    ratio of these normalisers at the current vs proposed centre.
    """
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive (got {sigma}).")
    if hi <= lo:
        raise ValueError(f"require hi > lo (got lo={lo}, hi={hi}).")
    z = standard_normal_cdf((hi - mu) / sigma) - standard_normal_cdf(
        (lo - mu) / sigma
    )
    # Guard against catastrophic cancellation in the deep tails.
    z = max(z, 1e-300)
    return math.log(z)


def random_walk_truncated_log_hastings(
    *, old: float, new: float, sigma: float, lo: float, hi: float
) -> float:
    r"""Log Hastings ratio for a truncated-Gaussian random walk on ``[lo, hi]``.

    Proposal ``q(new | old) = N(new; old, sigma^2) / Z(old)`` truncated to
    ``[lo, hi]`` (and symmetrically for the reverse).  The Gaussian kernel
    is symmetric, so the Hastings ratio collapses to the ratio of the
    truncation normalisers:

    .. math::

        \log\frac{q(\text{old}\mid\text{new})}{q(\text{new}\mid\text{old})}
        = \log Z(\text{old}) - \log Z(\text{new}).

    This is the correct replacement for the old "draw a Gaussian, clamp to
    the bounds" behaviour, which piled proposal mass on the endpoints and
    used a Hastings ratio of 1.
    """
    return truncated_normal_log_norm(old, sigma, lo, hi) - truncated_normal_log_norm(
        new, sigma, lo, hi
    )


def gaussian_delta_log_hastings(
    *,
    delta: float,
    sigma_fwd: float,
    sigma_rev: float,
    lo_fwd: float,
    hi_fwd: float,
    lo_rev: float,
    hi_rev: float,
) -> float:
    r"""Log Hastings ratio for a symmetric-delta slide with state-dependent scale.

    Models the :class:`phynetpy.ModelMove.ChangeNodeHeight` proposal: a node
    is shifted by ``delta`` drawn from ``N(0, sigma_fwd^2)`` truncated to the
    forward feasible window ``[lo_fwd, hi_fwd]``.  The matching reverse move
    shifts by ``-delta`` drawn from ``N(0, sigma_rev^2)`` truncated to the
    *post*-move feasible window ``[lo_rev, hi_rev]`` (the scale and window
    both depend on the current state, so the proposal is only *mildly*
    asymmetric -- but not exactly symmetric, which is why a ratio of 1 was
    wrong).

    .. math::

        \log\frac{q_\text{rev}(-\delta)}{q_\text{fwd}(\delta)}
        = \big[\log N(\delta;0,\sigma_r) - \log Z_r\big]
        - \big[\log N(\delta;0,\sigma_f) - \log Z_f\big].
    """
    log_q_rev = normal_log_pdf(delta, 0.0, sigma_rev) - truncated_normal_log_norm(
        0.0, sigma_rev, lo_rev, hi_rev
    )
    log_q_fwd = normal_log_pdf(delta, 0.0, sigma_fwd) - truncated_normal_log_norm(
        0.0, sigma_fwd, lo_fwd, hi_fwd
    )
    return log_q_rev - log_q_fwd


# ======================================================================
# Discrete weighted selection (SPR regraft, edge picks)
# ======================================================================


def weighted_choice_log_hastings(
    *, fwd_weight: float, fwd_total: float, rev_weight: float, rev_total: float
) -> float:
    r"""Log Hastings ratio for an asymmetric weighted discrete choice.

    When a move selects an option with forward probability
    ``fwd_weight / fwd_total`` and the reverse move would select the
    inverse option with probability ``rev_weight / rev_total``, the
    selection contributes

    .. math::

        \log\frac{p_\text{rev}}{p_\text{fwd}}
        = \log\frac{\text{rev\_weight} / \text{rev\_total}}
                   {\text{fwd\_weight} / \text{fwd\_total}}.

    Used by :class:`phynetpy.ModelMove.SPR`, whose distance-weighted
    regraft is genuinely asymmetric.  Passing equal weights (uniform
    regraft) with equal totals yields ``0.0`` -- the symmetric special
    case.
    """
    if fwd_weight <= 0.0 or rev_weight <= 0.0 or fwd_total <= 0.0 or rev_total <= 0.0:
        raise ValueError(
            "weighted_choice_log_hastings requires strictly positive weights "
            f"and totals (got fwd={fwd_weight}/{fwd_total}, "
            f"rev={rev_weight}/{rev_total})."
        )
    return (
        math.log(rev_weight)
        - math.log(rev_total)
        - math.log(fwd_weight)
        + math.log(fwd_total)
    )
