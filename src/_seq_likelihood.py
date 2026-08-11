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

Likelihood foundation for :class:`phynetpy._mcmc_seq.MCMC_SEQ`.

This module implements -- bit-for-bit against PhyloNet's ``MCMC_SEQ``
(Wen & Nakhleh 2018, *Systematic Biology* 67(3):439-457, "Coestimating
Reticulate Phylogenies and Gene Trees from Multilocus Sequence Data") --
the two factors of the per-locus likelihood that the co-estimation sampler
maximises:

1. **Phylogenetic (Felsenstein) likelihood** ``P(S_i | g_i)`` -- the
   probability of the locus-``i`` alignment given its (timed, ultrametric)
   gene tree under a time-reversible nucleotide substitution model.  This
   is Felsenstein's (1981) pruning algorithm, vectorised across all sites of
   the alignment at once with site-pattern compression (the same trick
   BEAGLE uses), so a single matrix multiply scores every column.

2. **Multispecies-network-coalescent (MSNC) density** ``P(g_i | Psi)`` --
   the probability *density* (not the topology-only mass function used by
   :mod:`phynetpy._mcmc_gt`) of a gene tree with branch lengths embedded in
   a species network with node heights, per-branch population-mutation
   rates ``theta`` and reticulation inheritance probabilities ``gamma``.
   The per-branch kernel is exactly PhyloNet's
   ``GeneTreeBrSpeciesNetDistribution.calculateProbability``::

       per coalescence c (with u lineages just below it):
           (2/theta) * exp(-(c - prev) * u*(u-1) / theta)
       final no-coalescence segment on a finite branch [.., tau_high]:
           exp(-(tau_high - prev) * u_f*(u_f-1) / theta)
       and a gamma^u factor for the u lineages tracking a reticulation edge.

   The sum is over all coalescent histories (ancestral configurations);
   for a species *tree* there is a single history and the density reduces
   to the standard Rannala-Yang (2003) multispecies-coalescent density.

Units (critical for matching PhyloNet): node heights / branch lengths are in
**expected substitutions per site**; ``theta`` is the population mutation
rate ``4 N mu`` (so the per-pair coalescent rate is ``2/theta`` and the total
rate among ``u`` lineages is ``u(u-1)/theta``).

Conventions (also critical): nucleotides are ordered ``A,C,G,T`` -> ``0,1,2,3``
(the BEAST/BEAGLE order PhyloNet uses), and ambiguity / gap characters expand
to the standard IUPAC tip-partial 0/1 vectors.

The MSNC density delegates to the shared ancestral-configurations DP in
:mod:`phynetpy._msnc_density` (same kernel as :mod:`phynetpy._mcmc_gt`),
with explicit ``theta`` scaling for substitution-unit branch lengths.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .Network import Network, Node

# Re-exported: one MSNC density implementation, shared with the MCMC_GT stack.
from ._msnc_density import gene_tree_msnc_log_density  # noqa: F401


__all__ = [
    "SubstitutionModel",
    "JC69",
    "GTR",
    "HKY85",
    "FelsensteinCalculator",
    "gene_tree_msnc_log_density",
    "DNA_STATES",
    "tip_partials_for_char",
]


# ======================================================================
# Nucleotide encoding (A=0, C=1, G=2, T=3) with IUPAC ambiguity
# ======================================================================

DNA_STATES: str = "ACGT"
_STATE_INDEX: dict[str, int] = {"A": 0, "C": 1, "G": 2, "T": 3}

# IUPAC ambiguity codes -> the set of concrete nucleotides they cover.  Maps
# every character we might see in an aligned DNA matrix to a length-4 tip
# partial-likelihood vector with 1.0 in the allowed states and 0.0 elsewhere
# (gaps / N / ? / X are fully ambiguous -> all ones).
_IUPAC: dict[str, str] = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
    "R": "AG", "Y": "CT", "S": "CG", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG",
    "N": "ACGT", "?": "ACGT", "-": "ACGT", "X": "ACGT", ".": "ACGT",
    "O": "ACGT",
}


def tip_partials_for_char(ch: str) -> np.ndarray:
    """Length-4 tip partial-likelihood vector for a DNA character.

    Args:
        ch: A single (case-insensitive) IUPAC nucleotide / ambiguity / gap
            character.

    Returns:
        ``float64`` array of length 4 (order ``A,C,G,T``) with ``1.0`` in
        every state the character is compatible with and ``0.0`` elsewhere.
        Unknown characters are treated as fully ambiguous (all ones), matching
        PhyloNet's permissive tip handling.
    """
    out = np.zeros(4, dtype=np.float64)
    allowed = _IUPAC.get(str(ch).upper())
    if allowed is None:
        out[:] = 1.0
        return out
    for nt in allowed:
        out[_STATE_INDEX[nt]] = 1.0
    return out


# ======================================================================
# Substitution models
# ======================================================================

class SubstitutionModel:
    """Time-reversible 4-state nucleotide substitution model.

    Subclasses build the (un-normalised) instantaneous rate matrix ``Q`` via
    :meth:`_build_Q` from their stationary base frequencies ``pi`` and the six
    exchangeability rates.  The base class then normalises ``Q`` so that the
    expected substitution rate is 1 (``-sum_i pi_i Q_ii = 1``) -- this makes a
    branch length the expected number of substitutions per site, exactly as in
    PhyloNet/BEAST -- and computes ``P(t) = exp(Q t)`` via a single
    eigendecomposition that is reused for every branch length.

    Attributes:
        pi: Stationary base frequency vector (length 4, ``A,C,G,T``).
    """

    def __init__(self, pi: Sequence[float], rates: Sequence[float]) -> None:
        """Build and normalise the model.

        Args:
            pi: Stationary base frequencies (length 4; ``A,C,G,T``); must sum
                to 1.
            rates: The six exchangeability parameters in the order
                ``AC, AG, AT, CG, CT, GT``.
        """
        self.pi = np.asarray(pi, dtype=np.float64)
        if self.pi.shape != (4,):
            raise ValueError("Base frequencies must have length 4 (A,C,G,T).")
        if not math.isclose(float(self.pi.sum()), 1.0, abs_tol=1e-9):
            raise ValueError("Base frequencies must sum to 1.")
        self._rates = np.asarray(rates, dtype=np.float64)
        self._build_and_decompose()

    # -- construction -------------------------------------------------

    def _build_Q(self) -> np.ndarray:
        """Assemble the un-normalised, un-row-summed rate matrix.

        Returns:
            A ``4x4`` matrix whose off-diagonal entries are
            ``Q[i,j] = rate(i,j) * pi[j]`` (the time-reversible GTR
            parameterisation).  Diagonal is filled in by the caller.
        """
        ac, ag, at_, cg, ct, gt = self._rates
        r = np.array(
            [
                [0.0, ac, ag, at_],
                [ac, 0.0, cg, ct],
                [ag, cg, 0.0, gt],
                [at_, ct, gt, 0.0],
            ],
            dtype=np.float64,
        )
        Q = r * self.pi[np.newaxis, :]
        return Q

    def _build_and_decompose(self) -> None:
        """Normalise ``Q`` to unit mean rate and eigendecompose it."""
        Q = self._build_Q()
        np.fill_diagonal(Q, 0.0)
        diag = -Q.sum(axis=1)
        np.fill_diagonal(Q, diag)

        # Normalise so the expected number of substitutions per unit time is 1.
        mean_rate = -float(np.dot(self.pi, np.diag(Q)))
        if mean_rate <= 0.0:
            raise ValueError("Degenerate substitution model (non-positive rate).")
        Q /= mean_rate
        self.Q = Q

        # Symmetric eigendecomposition via pi^{1/2} similarity transform: this
        # is numerically far more stable than a general eig() and is exactly
        # how BEAST/PhyloNet decompose a reversible Q.
        sqrt_pi = np.sqrt(self.pi)
        inv_sqrt_pi = 1.0 / sqrt_pi
        B = (Q * sqrt_pi[:, None]) * inv_sqrt_pi[None, :]
        B = 0.5 * (B + B.T)  # symmetrise away round-off
        evals, evecs = np.linalg.eigh(B)
        self._evals = evals
        # Right/left eigenvectors of Q recovered from the symmetric ones.
        self._U = inv_sqrt_pi[:, None] * evecs
        self._Uinv = evecs.T * sqrt_pi[None, :]

    # -- transition probabilities ------------------------------------

    def p_matrix(self, t: float) -> np.ndarray:
        """Transition-probability matrix ``P(t) = exp(Q t)``.

        Args:
            t: Branch length in expected substitutions per site (``>= 0``).

        Returns:
            A ``4x4`` row-stochastic matrix; ``P[i,j]`` is the probability of
            ending in state ``j`` after time ``t`` starting from state ``i``.
        """
        if t < 0.0:
            t = 0.0
        expd = np.exp(self._evals * t)
        return (self._U * expd[None, :]) @ self._Uinv


class JC69(SubstitutionModel):
    """Jukes-Cantor (1969): equal base frequencies, equal rates.

    PhyloNet's default model.  Uses the closed-form transition matrix
    ``P_ii(t) = 1/4 + 3/4 e^{-4t/3}``, ``P_ij(t) = 1/4 - 1/4 e^{-4t/3}``,
    bypassing the eigendecomposition entirely for speed and exactness.
    """

    def __init__(self) -> None:
        """Construct the parameter-free JC69 model."""
        super().__init__([0.25, 0.25, 0.25, 0.25], [1.0] * 6)

    def p_matrix(self, t: float) -> np.ndarray:
        """Closed-form JC69 transition matrix (see class docstring)."""
        if t < 0.0:
            t = 0.0
        e = math.exp(-4.0 * t / 3.0)
        same = 0.25 + 0.75 * e
        diff = 0.25 - 0.25 * e
        P = np.full((4, 4), diff, dtype=np.float64)
        np.fill_diagonal(P, same)
        return P


class HKY85(SubstitutionModel):
    """Hasegawa-Kishino-Yano (1985): free ``pi``, one transition/transversion ratio.

    Exchangeabilities are ``[1, kappa, 1, 1, kappa, 1]`` in ``AC,AG,AT,CG,CT,GT``
    order (transitions ``A<->G`` and ``C<->T`` scaled by ``kappa``).
    """

    def __init__(self, kappa: float, pi: Sequence[float]) -> None:
        """Build an HKY85 model.

        Args:
            kappa: Transition/transversion rate ratio (``> 0``).
            pi: Stationary base frequencies (length 4, ``A,C,G,T``).
        """
        self.kappa = float(kappa)
        super().__init__(pi, [1.0, self.kappa, 1.0, 1.0, self.kappa, 1.0])


# ``GTR`` is just the general base class with all six rates free; alias it so
# user-facing code can name it explicitly.
class GTR(SubstitutionModel):
    """General time-reversible model: free ``pi`` and all six exchangeabilities.

    Args mirror PhyloNet's ``-gtr (piA piC piG piT, rAC rAG rAT rCG rCT rGT)``.
    """

    def __init__(self, pi: Sequence[float], rates: Sequence[float]) -> None:
        """Build a GTR model.

        Args:
            pi: Stationary base frequencies (length 4, ``A,C,G,T``).
            rates: Six exchangeabilities ``AC,AG,AT,CG,CT,GT``.
        """
        super().__init__(pi, rates)


# ======================================================================
# Felsenstein pruning likelihood (vectorised over sites)
# ======================================================================

class FelsensteinCalculator:
    """Per-locus phylogenetic likelihood ``P(S | g)`` via vectorised pruning.

    On construction the alignment for one locus is compressed to its distinct
    site patterns (columns), each carrying an integer weight (how many original
    sites share that pattern).  Scoring a gene tree then runs one post-order
    sweep in which every node's conditional likelihoods are an
    ``(n_patterns, 4)`` array, so a single ``partials @ P.T`` matrix product
    propagates all sites across a branch at once.  This is the standard
    BEAGLE-style batching and is the source of the "sequences are very
    vectorizable" speed-up.

    Attributes:
        taxa: Ordered list of leaf labels present in the alignment.
        n_patterns: Number of distinct site patterns after compression.
    """

    def __init__(self, alignment: dict[str, str]) -> None:
        """Compress a single-locus alignment.

        Args:
            alignment: Map from leaf label to its aligned nucleotide string
                (all strings the same length).  IUPAC ambiguity / gaps are
                allowed and expand to tip partials.
        """
        if not alignment:
            raise ValueError("Empty alignment passed to FelsensteinCalculator.")
        self.taxa: list[str] = list(alignment.keys())
        seqs = [alignment[t] for t in self.taxa]
        length = len(seqs[0])
        if any(len(s) != length for s in seqs):
            raise ValueError("All sequences in a locus must be equally long.")

        # Compress to distinct column patterns with multiplicities.
        pattern_index: dict[tuple, int] = {}
        weights: list[int] = []
        columns: list[tuple] = []
        for col in range(length):
            key = tuple(seqs[r][col].upper() for r in range(len(self.taxa)))
            idx = pattern_index.get(key)
            if idx is None:
                pattern_index[key] = len(columns)
                columns.append(key)
                weights.append(1)
            else:
                weights[idx] += 1

        self.n_patterns: int = len(columns)
        self._weights = np.asarray(weights, dtype=np.float64)

        # Pre-build the (n_patterns, 4) tip partials for every taxon.
        self._tip_partials: dict[str, np.ndarray] = {}
        for r, taxon in enumerate(self.taxa):
            mat = np.empty((self.n_patterns, 4), dtype=np.float64)
            for p, key in enumerate(columns):
                mat[p] = tip_partials_for_char(key[r])
            self._tip_partials[taxon] = mat

    def log_likelihood(
        self,
        gene_tree: Network,
        model: SubstitutionModel,
        *,
        site_rate: float = 1.0,
    ) -> float:
        """Felsenstein log-likelihood of the alignment on ``gene_tree``.

        Args:
            gene_tree: A rooted *tree* (one connected component, no
                reticulations) whose leaves are labelled with this locus's
                taxa and whose edges carry branch lengths in substitution
                units.
            model: The nucleotide substitution model.
            site_rate: Optional per-locus relative mutation rate (PhyloNet's
                ``-murate``); branch lengths are effectively scaled by this.
                Defaults to 1.0 (rate variation off).

        Returns:
            The summed log-likelihood over all (weighted) site patterns.
        """
        root = gene_tree.root()
        pi = model.pi
        partials = self._postorder_partials(gene_tree, root, model, site_rate)
        site_like = partials @ pi
        # Guard against underflow on pathological branch lengths.
        site_like = np.maximum(site_like, 1e-300)
        return float(np.dot(np.log(site_like), self._weights))

    def _postorder_partials(
        self,
        tree: Network,
        node: Node,
        model: SubstitutionModel,
        site_rate: float,
    ) -> np.ndarray:
        """Conditional likelihood array ``(n_patterns, 4)`` at ``node``."""
        children = tree.get_children(node)
        if not children:
            tip = self._tip_partials.get(node.label)
            if tip is None:
                # Leaf not in this locus's alignment -> fully ambiguous.
                return np.ones((self.n_patterns, 4), dtype=np.float64)
            return tip

        result = np.ones((self.n_patterns, 4), dtype=np.float64)
        for child in children:
            edges = tree.get_edge(node, child)
            edge = edges[0] if isinstance(edges, list) else edges
            t = edge.get_length()
            t = 0.0 if t is None else float(t) * site_rate
            child_partials = self._postorder_partials(tree, child, model, site_rate)
            P = model.p_matrix(t)
            # (n_patterns,4) @ (4,4)^T -> probability of child data given each
            # parent state; multiply across children (independent lineages).
            result *= child_partials @ P.T
        return result


# MSNC density: shared implementation in phynetpy._msnc_density (re-exported above).
