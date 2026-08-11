#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##############################################################################

"""
The criterion axis: the base class and the statistical objectives.

Three objectives plus one mode.  ``Bayesian`` deliberately *wraps* an
objective rather than sitting parallel to the likelihoods, because MCMC is
not a fourth thing to optimise -- it samples a posterior built on top of a
likelihood.  PhyloNet's own commands confirm the factoring:
``MLE_BiMarkers -pseudo`` makes the objective a flag on the method rather
than a separate method.

Which wrapped objectives are *available* is then a question about each
engine, not about the design.  No engine currently accepts a
pseudo-likelihood inside ``Bayesian``, because a pseudo-likelihood is not a
normalised probability of the data and so does not define a calibrated
posterior; those requests fail with an explanation rather than silently
sampling the wrong target.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Optional, Tuple

from ..data import Alignment, BiallelicMarkers, GeneTrees


class CriterionError(Exception):
    """Raised when a criterion is misconfigured."""


class Criterion(ABC):
    """Base class for a statistical objective.

    A criterion says *what number is being optimised*: a parsimony cost, a
    full likelihood, a pseudo-likelihood, or a posterior.  It is
    independent of both the data type and the biology, which is why it is
    its own axis.

    Two class attributes make the legality of a run checkable before any
    computation starts:

    * ``accepts_data`` -- the data types the objective is mathematically
      defined on.  Violating it is a :class:`TypeError`, not a missing
      feature: MDC over a sequence alignment is not unimplemented, it is
      undefined.
    * ``scorable`` -- whether the objective can score a single fixed
      network.  ``False`` for :class:`~phynetpy.criteria.Bayesian`, whose
      value on one network collapses to its wrapped objective.

    Attributes:
        use_branch_lengths: Tri-valued branch-length policy.  ``True``
            requires and uses gene-tree branch lengths, ``False`` ignores
            them (topology only), and ``None`` uses them when present.
            Validated against the data in :func:`phynetpy._registry.resolve`.
    """

    #: Data classes this objective is defined on.  Empty means "any".
    accepts_data: Tuple[type, ...] = ()

    #: Whether this objective can score a single fixed network.
    scorable: bool = True

    #: Default branch-length policy: topology only.
    use_branch_lengths: Optional[bool] = False

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class MDC(Criterion):
    """Minimizing deep coalescences: a parsimony criterion.

    Counts the extra gene lineages needed to reconcile the gene trees with
    the species network and prefers the network needing fewest.  Defined
    over gene-tree *topologies*, so branch lengths are ignored by
    construction -- ``use_branch_lengths`` is fixed at ``False`` and cannot
    be overridden.

    Lower is better, unlike every other criterion here; the verbs normalise
    this so that :attr:`~phynetpy.infer.InferenceResult.score` is always
    reported in the criterion's own natural direction and documented as
    such.
    """

    accepts_data = (GeneTrees,)
    use_branch_lengths = False

    def __init__(self, weighting: Optional[str] = None) -> None:
        """Configure the parsimony criterion.

        Args:
            weighting: Per-tree weighting scheme (``"bootstrap"``,
                ``"posterior"``, ...).  ``None`` weights every gene tree
                equally.

        Raises:
            CriterionError: If a weighting scheme is requested.
        """
        if weighting is not None:
            raise CriterionError(
                "confidence-weighted MDC is not implemented yet; the gene "
                "trees would need to carry per-tree weights. Use MDC()."
            )


class Likelihood(Criterion):
    """Full likelihood of the data under the model.

    Integrates over all coalescent histories rather than summarising them,
    so it is the most statistically efficient of the point-estimate
    criteria and the most expensive.  Defined on all three data types.

    Attributes:
        use_branch_lengths: ``True`` requires and uses gene-tree branch
            lengths, ``False`` scores topologies only, ``None`` uses them
            when present.
    """

    accepts_data = (GeneTrees, Alignment, BiallelicMarkers)

    def __init__(self, use_branch_lengths: Optional[bool] = None) -> None:
        """Configure the full-likelihood criterion.

        Args:
            use_branch_lengths: Branch-length policy.  ``None`` (default)
                uses gene-tree branch lengths when the data carries them.
        """
        self.use_branch_lengths = use_branch_lengths

    def __repr__(self) -> str:
        return f"Likelihood(use_branch_lengths={self.use_branch_lengths!r})"


class PseudoLikelihood(Criterion):
    """Pseudo-likelihood over subsets of the taxa.

    Replaces the full likelihood with a product of likelihoods over small
    subsets (by default rooted triples), which is far cheaper and scales to
    many more taxa at some cost in efficiency (Yu & Nakhleh 2015;
    Solis-Lemus & Ane 2016).

    Defined on gene-tree topologies and on biallelic markers, but not on
    alignments: a pseudo-likelihood over subsets of gene trees needs gene
    trees, and PhyNetPy does not estimate them from sequences.

    Attributes:
        subsets: Which subsets the product runs over.  Only ``"trinets"``
            (rooted triples) is implemented.
        use_branch_lengths: Branch-length policy; the triplet method
            consumes topologies only.
    """

    accepts_data = (GeneTrees, BiallelicMarkers)

    def __init__(
        self,
        subsets: str = "trinets",
        use_branch_lengths: Optional[bool] = None,
    ) -> None:
        """Configure the pseudo-likelihood criterion.

        Args:
            subsets: Subset family for the product.  ``"trinets"`` (3-taxon
                subnetworks) is the only supported value.
            use_branch_lengths: Branch-length policy.

        Raises:
            CriterionError: If *subsets* is not ``"trinets"``.
        """
        if subsets != "trinets":
            raise CriterionError(
                f"unsupported pseudo-likelihood subsets {subsets!r}; only "
                "'trinets' (rooted triples) is implemented."
            )
        self.subsets = subsets
        self.use_branch_lengths = use_branch_lengths

    def __repr__(self) -> str:
        return f"PseudoLikelihood(subsets={self.subsets!r})"


class Bayesian(Criterion):
    """Bayesian mode: sample a posterior built on an objective.

    Not a fourth objective but a *mode layered on one*.  The casual user
    writes ``Bayesian()`` and gets the full likelihood; the power user
    writes ``Bayesian(objective=PseudoLikelihood())``.  Which data types are
    legal follows from the wrapped objective, so ``accepts_data`` is
    delegated to it.

    Scoring a single fixed network under a Bayesian criterion collapses to
    computing its likelihood or posterior density, so ``scorable`` is
    ``False`` and :func:`phynetpy.infer.score` refuses it, pointing the
    caller at ``.objective`` instead.

    Attributes:
        objective: The wrapped likelihood (full or pseudo).
        prior: Prior hyperparameters, or ``None`` for the engine's
            defaults.
        chain_length: Total proposed moves.
        burnin: Iterations discarded before sampling begins.
        sample_freq: Thinning interval between retained samples.
        seed: Master RNG seed.
        temperatures: Optional Metropolis-coupled (MC3) temperature ladder.
    """

    scorable = False

    def __init__(
        self,
        objective: Optional[Criterion] = None,
        prior: Optional[Any] = None,
        chain_length: int = 1_000_000,
        burnin: int = 100_000,
        *,
        sample_freq: int = 100,
        seed: Any = None,
        temperatures: Optional[list] = None,
    ) -> None:
        """Configure the Bayesian mode.

        Args:
            objective: The likelihood the posterior is built on.  ``None``
                selects :class:`Likelihood`.
            prior: Prior hyperparameters
                (:class:`~phynetpy.infer.MCMC_GTPriors` for gene trees and
                markers, :class:`~phynetpy.infer.MCMCSeqPriors` for
                alignments).  ``None`` uses the engine's defaults.
            chain_length: Total proposed moves.
            burnin: Iterations discarded before sampling.
            sample_freq: Thinning interval for retained samples.
            seed: Master RNG seed.
            temperatures: MC3 temperature ladder (alignments only).

        Raises:
            CriterionError: If the objective is itself Bayesian, or the
                chain budget is inconsistent.
        """
        objective = objective if objective is not None else Likelihood()

        if isinstance(objective, Bayesian):
            raise CriterionError(
                "Bayesian cannot wrap another Bayesian; pass a likelihood "
                "(Likelihood() or PseudoLikelihood())."
            )
        if not isinstance(objective, Criterion):
            raise CriterionError(
                "Bayesian(objective=...) needs a Criterion; got "
                f"{type(objective).__name__}."
            )
        if chain_length <= 0:
            raise CriterionError(
                f"chain_length must be positive; got {chain_length}."
            )
        if burnin < 0:
            raise CriterionError(f"burnin cannot be negative; got {burnin}.")
        if burnin >= chain_length:
            raise CriterionError(
                f"burnin ({burnin}) must be smaller than chain_length "
                f"({chain_length}), or no samples would be retained."
            )

        self.objective = objective
        self.prior = prior
        self.chain_length = chain_length
        self.burnin = burnin
        self.sample_freq = sample_freq
        self.seed = seed
        self.temperatures = temperatures

    @property
    def accepts_data(self) -> tuple:
        """Data types the wrapped objective is defined on."""
        return self.objective.accepts_data

    @property
    def use_branch_lengths(self) -> Optional[bool]:
        """Branch-length policy of the wrapped objective."""
        return self.objective.use_branch_lengths

    def __repr__(self) -> str:
        return (
            f"Bayesian(objective={self.objective!r}, "
            f"chain_length={self.chain_length}, burnin={self.burnin})"
        )
