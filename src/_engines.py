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
Engine adapters: one registration site per implemented method.

This module is the seam between the declarative three-axis API and the
numerical code.  Every implemented cell of the validity matrix is exactly
one ``@register``-decorated class here, which makes the mapping from
PhyloNet commands to PhyNetPy code one-to-one and readable at a glance:

===============================================  ======================
Cell                                             PhyloNet command
===============================================  ======================
``(GeneTrees, MSC, MDC)``                        InferNetwork_MP
``(GeneTrees, MSC, Likelihood)``                 InferNetwork_ML / CalGTProb
``(GeneTrees, MSC, PseudoLikelihood)``           InferNetwork_MPL
``(GeneTrees, MSC, Bayesian)``                   MCMC_GT
``(Alignment, MSC, Bayesian)``                   MCMC_SEQ
``(BiallelicMarkers, MSC, Likelihood)``          MLE_BiMarkers
``(BiallelicMarkers, MSC, PseudoLikelihood)``    MLE_BiMarkers -pseudo
``(BiallelicMarkers, MSC, Bayesian)``            MCMC_BiMarkers
``(GeneTrees, Allopolyploid, MDC)``              MPAllopp / Polyphest
===============================================  ======================

Two of those rows are *not* registered below, and deliberately so.  MDC
under the MSC (``InferNetwork_MP``) and the biallelic pseudo-likelihood
(``MLE_BiMarkers -pseudo``) have no numerical implementation in PhyNetPy
yet: the only parsimony code here is defined on multiple-labelled trees for
the allopolyploid model, and the marker code computes the full likelihood
only.  Leaving those cells unregistered is what makes them fail as
``NotImplementedError`` -- an honest "nobody has built it yet" rather than a
misleading "that is impossible".

No engine reimplements a likelihood.  Each one resolves the starting
network, translates the axis objects into the arguments its implementation
module already takes, and normalises the return value onto
:class:`~phynetpy.infer.InferenceResult`.

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._registry import Engine, register
from .criteria import Bayesian, Likelihood, MDC, PseudoLikelihood
from .data import Alignment, BiallelicMarkers, GeneTrees
from .models import MSC, Allopolyploid
from .Network import Network
from ._units import BranchLengthUnit, require_branch_length_unit

__all__ = [
    "GTLikelihoodEngine",
    "GTPseudoLikEngine",
    "GTBayesEngine",
    "SeqBayesEngine",
    "MarkerLikelihoodEngine",
    "MarkerBayesEngine",
    "AllopolyParsimonyEngine",
]


# ======================================================================
# Shared helpers
# ======================================================================


def _best_of(scores: Dict[Network, float]) -> Tuple[Network, float]:
    """Pick the highest-scoring network from a ``{network: score}`` leaderboard.

    Args:
        scores: Candidate networks mapped to their scores.  Both leaderboards
            in this module are maximised -- the parsimony one holds negated
            costs precisely so that it can be.

    Returns:
        tuple[Network, float]: The winner and its score.

    Raises:
        RuntimeError: If no networks were scored.
    """
    if not scores:
        raise RuntimeError(
            "the search produced no scored networks; try more iterations or a "
            "different starting network."
        )
    net = max(scores, key=scores.get)
    return net, scores[net]


def _with_chain_controls(
    kwargs: Dict[str, Any], criterion: Bayesian, *, thin_key: str = "sample_freq",
) -> Dict[str, Any]:
    """Fill in the MCMC chain controls a ``Bayesian`` criterion carries.

    ``setdefault`` rather than assignment, so an explicit search kwarg still
    wins: the criterion supplies the chain budget, the caller may override it.

    Args:
        kwargs: The search kwargs assembled so far; mutated and returned.
        criterion: The Bayesian criterion holding the chain settings.
        thin_key: What the target sampler calls its thinning interval.

    Returns:
        dict[str, Any]: *kwargs*, with the chain controls filled in.
    """
    kwargs.setdefault("num_iter", criterion.chain_length)
    kwargs.setdefault("burn_in", criterion.burnin)
    kwargs.setdefault(thin_key, criterion.sample_freq)
    kwargs.setdefault("seed", criterion.seed)
    return kwargs


class _GeneTreeEngine(Engine):
    """Shared plumbing for the engines that consume gene trees.

    Factors out the two things every gene-tree method needs and does
    identically: turning a :class:`~phynetpy.infer.Start` into a concrete
    starting network (falling back to a majority-rule consensus), and
    splitting the caller's ``**search`` kwargs into the backbone constraint
    plus everything the implementation module already understands.
    """

    def _starting_network(self, data: GeneTrees) -> Network:
        """The network the search starts from.

        A supplied :class:`~phynetpy.infer.Start` is deep-copied so the
        caller's object is never mutated by the search -- the old
        method-class API consumed the starting network by reference, which
        made repeated runs from one seed quietly non-reproducible.  With no
        start, the seed is a majority-rule consensus of the gene trees.
        """
        from ._mcmc_gt import _populate_default_branch_lengths

        if self.start is not None:
            require_branch_length_unit(
                self.start.network,
                BranchLengthUnit.COALESCENT_2N,
                context="gene-tree inference",
            )
            return copy.deepcopy(self.start.network)

        seed = data.build_majority_rule_consensus_tree(threshold=0.5)
        seed.set_branch_length_unit(BranchLengthUnit.COALESCENT_2N)
        _populate_default_branch_lengths(seed)
        return seed

    def _search_kwargs(self, data: GeneTrees) -> Dict[str, Any]:
        """Assemble the implementation module's search kwargs.

        Adds the ``backbone`` constraint when the start is an
        ``AUGMENT`` start; everything else is forwarded verbatim.
        """
        kwargs = dict(self.search)
        if self.start is not None and self.start.augment_only:
            kwargs["backbone"] = self.start.network
        return kwargs

    def _score_with(
        self,
        scorer,
        network: Network,
        mapping: Dict[str, List[str]],
        *,
        optimize: bool,
    ) -> float:
        """Score ``network`` under ``scorer``, optionally optimising params.

        The generic implementation of ``score(..., optimize=True)``:
        :func:`phynetpy._optimize.optimize_network_parameters` is
        scorer-agnostic, so every likelihood-based engine gets the
        optimised-score path from the same code.
        """
        from ._optimize import optimize_network_parameters
        from .ModelGraph import Model as GraphModel

        require_branch_length_unit(
            network,
            BranchLengthUnit.COALESCENT_2N,
            context="gene-tree likelihood",
        )

        try:
            model = GraphModel(rng=np.random.default_rng())
            model.network = network
            model.set_likelihood_calculator(scorer)
            if optimize:
                return float(optimize_network_parameters(
                    model, scorer, mapping, **self.search,
                ))
            return float(scorer(model))
        finally:
            close_fn = getattr(scorer, "close", None)
            if callable(close_fn):
                close_fn()


# ======================================================================
# Gene trees x MSC
# ======================================================================


@register(GeneTrees, MSC, Likelihood)
class GTLikelihoodEngine(_GeneTreeEngine):
    """Full MSNC likelihood from gene-tree topologies (Yu et al. 2014).

    PhyloNet's ``InferNetwork_ML`` for :meth:`infer` and ``CalGTProb`` for
    :meth:`score` -- one engine, because scoring and searching are the same
    objective evaluated differently.
    """

    method = "InferNetwork_ML"

    def infer(self, data: GeneTrees):
        """Search for the network maximising the full MSNC likelihood.

        Args:
            data: Gene-tree topologies to fit.

        Returns:
            InferenceResult: The best network found, its log-likelihood,
            and the search trace.
        """
        from ._infernetworkml import InferNetwork_ML
        from .infer import InferenceResult

        kwargs = self._search_kwargs(data)
        max_reticulations = kwargs.pop("max_reticulations", None)

        driver = InferNetwork_ML(
            self._starting_network(data),
            data,
            data.resolved_mapping(),
            max_reticulations=max_reticulations,
        )
        result = driver.search(**kwargs)
        return InferenceResult(
            best=result.best_network,
            score=float(result.best_log_likelihood),
            method=self.method,
            trace=result.networks,
            raw=result,
        )

    def score(self, network: Network, data: GeneTrees, *, optimize: bool = False) -> float:
        """Score ``network`` under the full MSNC likelihood (PhyloNet's ``CalGTProb``).

        Args:
            network: The network to score.
            data: Gene-tree topologies.
            optimize: When ``True``, also optimise the network's branch
                lengths and inheritance probabilities before scoring.

        Returns:
            float: The log-likelihood.
        """
        from ._mcmc_gt import MCMCGTScorer

        mapping = data.resolved_mapping()
        scorer = MCMCGTScorer(
            data, mapping, posterior=False,
            n_workers=self.search.get("n_workers", 1),
        )
        return self._score_with(scorer, network, mapping, optimize=optimize)


@register(GeneTrees, MSC, PseudoLikelihood)
class GTPseudoLikEngine(_GeneTreeEngine):
    """Triplet pseudo-likelihood from gene trees (Yu & Nakhleh 2015).

    PhyloNet's ``InferNetwork_MPL``.  Replaces the full likelihood with a
    product over rooted triples, trading statistical efficiency for the
    ability to handle many more taxa.
    """

    method = "InferNetwork_MPL"

    def infer(self, data: GeneTrees):
        """Search for the network maximising the triplet pseudo-likelihood.

        Args:
            data: Gene-tree topologies to fit.

        Returns:
            InferenceResult: The best network found and its pseudo-likelihood
            score.
        """
        from ._mpl import MPL
        from .infer import InferenceResult

        driver = MPL(
            self._starting_network(data), data, data.resolved_mapping(),
        )
        best_score = driver.search(**self._search_kwargs(data))
        return InferenceResult(
            best=driver.net,
            score=float(best_score),
            method=self.method,
            raw=driver,
        )

    def score(self, network: Network, data: GeneTrees, *, optimize: bool = False) -> float:
        """Score ``network`` under the triplet pseudo-likelihood (PhyloNet's ``InferNetwork_MPL``).

        Args:
            network: The network to score.
            data: Gene-tree topologies.
            optimize: When ``True``, also optimise the network's branch
                lengths and inheritance probabilities before scoring.

        Returns:
            float: The pseudo-likelihood score.
        """
        from ._mpl import MPL, MPLScorer, compute_gene_tree_triplets

        mapping = data.resolved_mapping()
        if not optimize:
            return float(MPL(network, data, mapping).score())

        # The optimised path needs a Model-shaped scorer rather than MPL's
        # own cached DP engine, so build the triplet scorer directly.
        triplets = compute_gene_tree_triplets(data, mapping)
        scorer = MPLScorer(triplets.rho_by_triplet, triplets.triplets)
        return self._score_with(scorer, network, mapping, optimize=True)


@register(GeneTrees, MSC, Bayesian)
class GTBayesEngine(_GeneTreeEngine):
    """Posterior sampling from gene-tree topologies (Wen & Nakhleh 2018).

    PhyloNet's ``MCMC_GT``.

    Only ``Bayesian(objective=Likelihood())`` is available here.  Wrapping a
    pseudo-likelihood is refused rather than silently allowed: a triplet
    pseudo-likelihood is not a normalised probability of the data, so
    treating it as the data term of a Metropolis-Hastings target does not
    yield a calibrated posterior.  The pseudo-likelihood objective is
    reachable as a maximisation instead, via ``criterion=PseudoLikelihood()``.
    """

    method = "MCMC_GT"

    def _priors(self):
        """Prior hyperparameters from the criterion, or the defaults."""
        from ._mcmc_gt import MCMC_GTPriors

        prior = self.criterion.prior
        return prior if prior is not None else MCMC_GTPriors()

    def infer(self, data: GeneTrees):
        """Sample the posterior over networks given gene-tree topologies.

        Args:
            data: Gene-tree topologies to fit.

        Returns:
            InferenceResult: The MAP network, its log-posterior, and the
            posterior sample.

        Raises:
            NotImplementedError: If ``criterion.objective`` is a
                :class:`~phynetpy.criteria.PseudoLikelihood` (not a valid
                Bayesian data term; see class docstring).
        """
        from ._mcmc_gt import MCMC_GT
        from .infer import InferenceResult

        criterion: Bayesian = self.criterion
        if isinstance(criterion.objective, PseudoLikelihood):
            raise NotImplementedError(
                "Bayesian(objective=PseudoLikelihood()) is not available for "
                "gene trees: a triplet pseudo-likelihood is not a normalised "
                "probability of the data, so using it as the data term of an "
                "MH target does not give a calibrated posterior. Use "
                "criterion=PseudoLikelihood() to maximise it instead, or "
                "Bayesian(objective=Likelihood()) to sample."
            )

        kwargs = _with_chain_controls(
            self._search_kwargs(data), criterion, thin_key="thin",
        )

        driver = MCMC_GT(
            self._starting_network(data),
            data,
            data.resolved_mapping(),
            priors=self._priors(),
        )
        result = driver.search(method="mh", **kwargs)
        return InferenceResult(
            best=result.best_network,
            score=float(result.best_log_posterior),
            method=self.method,
            posterior=result.samples,
            trace=result.samples,
            raw=result,
        )

    def score(self, network: Network, data: GeneTrees, *, optimize: bool = False) -> float:
        """Score ``network`` under the MSNC posterior (log-likelihood + prior).

        Args:
            network: The network to score.
            data: Gene-tree topologies.
            optimize: When ``True``, also optimise the network's branch
                lengths and inheritance probabilities before scoring.

        Returns:
            float: The log-posterior.
        """
        # Unreachable through the public verb -- Bayesian.scorable is False,
        # so score() rejects it before dispatch -- but implemented so the
        # engine honours the Engine contract if called directly.
        from ._mcmc_gt import MCMCGTScorer

        mapping = data.resolved_mapping()
        scorer = MCMCGTScorer(data, mapping, self._priors(), posterior=True)
        return self._score_with(scorer, network, mapping, optimize=optimize)


# ======================================================================
# Sequence alignments x MSC
# ======================================================================


@register(Alignment, MSC, Bayesian)
class SeqBayesEngine(Engine):
    """Co-estimation of network and gene trees from sequences.

    PhyloNet's ``MCMC_SEQ`` (Wen & Nakhleh 2018).  Integrates over the gene
    trees rather than conditioning on point estimates of them, which is why
    the alignment never needs to be summarised first.
    """

    method = "MCMC_SEQ"

    def _driver(self, data: Alignment, network: Optional[Network]):
        """Build the sampler for this run."""
        from ._mcmc_seq import MCMC_SEQ

        if network is not None:
            require_branch_length_unit(
                network,
                BranchLengthUnit.SUBSTITUTIONS_PER_SITE,
                context="MCMC_SEQ",
            )

        return MCMC_SEQ(
            data.loci,
            data.resolved_mapping(),
            network,
            priors=self.criterion.prior,
            model=data.substitution_model,
            theta=self.model.theta,
            branch_thetas=self.model.branch_thetas,
        )

    def infer(self, data: Alignment):
        """Co-sample the network and gene trees from sequence alignments.

        Args:
            data: Multi-locus sequence alignment to fit.

        Returns:
            InferenceResult: The MAP network, its log-posterior, and the
            posterior sample.

        Raises:
            NotImplementedError: If ``criterion.objective`` is a
                :class:`~phynetpy.criteria.PseudoLikelihood`, or if
                ``self.start`` requests ``StartMode.AUGMENT`` (unsupported
                by this sampler's coupled moves).
        """
        from .infer import InferenceResult

        criterion: Bayesian = self.criterion
        if self.model.branch_thetas:
            raise NotImplementedError(
                "fixed branch_thetas can be used for sequence scoring, but "
                "not topology-changing MCMC_SEQ inference."
            )
        if isinstance(criterion.objective, PseudoLikelihood):
            raise NotImplementedError(
                "MCMC_SEQ samples the full sequence likelihood; a "
                "pseudo-likelihood objective over alignments is not "
                "implemented. Use Bayesian(objective=Likelihood())."
            )

        start_net = (
            copy.deepcopy(self.start.network) if self.start is not None else None
        )
        if self.start is not None and self.start.augment_only:
            raise NotImplementedError(
                "StartMode.AUGMENT is not supported by MCMC_SEQ: its kernel "
                "proposes coupled gene-tree and network moves that cannot be "
                "restricted to backbone-preserving ones. Use "
                "Start(net) for a free start."
            )

        kwargs = _with_chain_controls(dict(self.search), criterion)
        if criterion.temperatures is not None:
            kwargs.setdefault("temperatures", criterion.temperatures)

        result = self._driver(data, start_net).search(**kwargs)
        return InferenceResult(
            best=result.map_network,
            score=float(result.map_log_posterior),
            method=self.method,
            posterior=result.samples,
            trace=result.samples,
            raw=result,
        )

    def score(self, network: Network, data: Alignment, *, optimize: bool = False) -> float:
        """Score ``network`` under the sequence-likelihood posterior.

        Args:
            network: The network to score.
            data: Multi-locus sequence alignment.
            optimize: Must be ``False``; MCMC_SEQ has no fixed-topology
                parameter optimiser (see Raises).

        Returns:
            float: The log-posterior.

        Raises:
            NotImplementedError: If ``optimize`` is ``True``.
        """
        if optimize:
            raise NotImplementedError(
                "MCMC_SEQ has no fixed-topology parameter optimiser: its "
                "continuous parameters (gene trees, theta, gammas) are "
                "sampled rather than maximised. Use optimize=False."
            )
        return float(self._driver(data, copy.deepcopy(network)).score())


# ======================================================================
# Biallelic markers x MSC
# ======================================================================


class _MarkerEngine(Engine):
    """Shared plumbing for the biallelic-marker engines."""

    def score(
        self, network: Network, data: BiallelicMarkers, *, optimize: bool = False,
    ) -> float:
        """Score ``network`` under the Bryant et al. biallelic-marker likelihood.

        Args:
            network: The network to score.
            data: Biallelic-marker (SNP) alignment.
            optimize: Must be ``False``; no optimiser is implemented for
                this likelihood yet (see Raises).

        Returns:
            float: The log-likelihood.

        Raises:
            NotImplementedError: If ``optimize`` is ``True``.
        """
        from .BiMarkers import _snp_log_likelihood

        if optimize:
            raise NotImplementedError(
                "optimize=True is not implemented for biallelic markers: the "
                "Bryant likelihood has no coordinate-ascent optimiser wired "
                "up yet. Score the network as given (optimize=False), or "
                "infer to search over parameters."
            )

        return float(_snp_log_likelihood(
            network, data.alignment,
            self.model.u,
            self.model.v,
            self.model.theta if self.model.theta is not None else 0.02,
            data.samples,
            branch_thetas=self.model.branch_thetas,
            max_workers=self.search.get("max_workers", 8),
            sequential=self.search.get("sequential", True),
            verbose=self.search.get("verbose", False),
        ))


@register(BiallelicMarkers, MSC, Likelihood)
class MarkerLikelihoodEngine(_MarkerEngine):
    """Exact biallelic-marker likelihood (Bryant et al. 2012).

    PhyloNet's ``MLE_BiMarkers``.  Scoring is implemented; the maximisation
    search is not, so :meth:`infer` reports that honestly instead of
    quietly running something else.
    """

    method = "MLE_BiMarkers"

    def infer(self, data: BiallelicMarkers):
        """Unimplemented: maximum-likelihood search over biallelic markers.

        Args:
            data: Biallelic-marker (SNP) alignment.

        Raises:
            NotImplementedError: Always; only :meth:`score` is implemented
                for this engine. Use ``criterion=Bayesian()`` to search.
        """
        raise NotImplementedError(
            "maximum-likelihood search over networks from biallelic markers "
            "is not implemented; only scoring is. Use score(net, markers, "
            "criterion=Likelihood()) for a single network, or "
            "infer(markers, criterion=Bayesian()) to search."
        )


@register(BiallelicMarkers, MSC, Bayesian)
class MarkerBayesEngine(_MarkerEngine):
    """Posterior sampling from biallelic markers (Zhu et al. 2018).

    PhyloNet's ``MCMC_BiMarkers``.
    """

    method = "MCMC_BiMarkers"

    def infer(self, data: BiallelicMarkers):
        """Sample the posterior over networks given biallelic markers.

        Args:
            data: Biallelic-marker (SNP) alignment.

        Returns:
            InferenceResult: The best-scoring network among the samples
            and its log-posterior.

        Raises:
            NotImplementedError: If ``criterion.objective`` is a
                :class:`~phynetpy.criteria.PseudoLikelihood` (not
                implemented for biallelic markers).
        """
        from .BiMarkers import _snp_mcmc
        from .infer import InferenceResult

        criterion: Bayesian = self.criterion
        if self.model.branch_thetas:
            raise NotImplementedError(
                "fixed branch_thetas can be used for marker simulation and "
                "fixed-network scoring, but not topology-changing MCMC."
            )
        if isinstance(criterion.objective, PseudoLikelihood):
            raise NotImplementedError(
                "the biallelic pseudo-likelihood (PhyloNet's MLE_BiMarkers "
                "-pseudo, Zhu & Nakhleh 2018) is not implemented. Use "
                "Bayesian(objective=Likelihood())."
            )

        kwargs = _with_chain_controls(dict(self.search), criterion)
        kwargs.setdefault("priors", criterion.prior)

        if self.start is not None:
            kwargs["start_net"] = copy.deepcopy(self.start.network)
            if self.start.augment_only:
                kwargs["backbone"] = self.start.network

        scores = _snp_mcmc(
            data.alignment,
            self.model.u,
            self.model.v,
            self.model.theta if self.model.theta is not None else 0.02,
            samples=data.samples, **kwargs,
        )
        best_net, best_score = _best_of(scores)
        return InferenceResult(
            best=best_net,
            score=float(best_score),
            method=self.method,
            raw=scores,
        )


# ======================================================================
# Gene trees x Allopolyploid
# ======================================================================


@register(GeneTrees, Allopolyploid, MDC)
class AllopolyParsimonyEngine(_GeneTreeEngine):
    """Allopolyploid parsimony over MUL trees (Hejase et al.).

    PhyloNet's ``MPAllopp``; the same criterion underlies Polyphest.  Counts
    the extra gene lineages needed to reconcile the gene trees with the
    network's multiple-labelled tree, so lower is better -- the one
    criterion in PhyNetPy that is minimised.
    """

    method = "MP_Allop"

    def infer(self, data: GeneTrees):
        """Search for the allopolyploid network minimising extra gene lineages.

        Args:
            data: Gene-tree topologies to reconcile.

        Returns:
            InferenceResult: The best network found and its (positive,
            lower-is-better) extra-lineage count.

        Raises:
            NotImplementedError: If ``self.start`` requests
                ``StartMode.AUGMENT`` (unsupported by this search's only
                move, ``SwitchParentage``).
        """
        from ._infer_mp_allop import InferMPAllop, partition_gene_trees
        from .infer import InferenceResult

        if self.start is not None and self.start.augment_only:
            raise NotImplementedError(
                "StartMode.AUGMENT is not supported by the allopolyploid "
                "parsimony search: its only move (SwitchParentage) rewires "
                "subgenome parentage rather than adding structure, so "
                "'augment only' has no meaning for it. Use Start(net)."
            )

        subgenome_map = self.model.resolve_subgenome_map(data)
        gene_trees = list(data.trees)
        rng = np.random.default_rng(self.search.get("seed"))

        if self.start is not None:
            start_net = copy.deepcopy(self.start.network)
        else:
            start_net = partition_gene_trees(subgenome_map, rng=rng)

        driver = InferMPAllop(
            start_net,
            subgenome_map,
            gene_trees,
            self.search.get("num_iter", 500),
            rng=rng,
        )
        driver.run()

        # The leaderboard holds *negated* parsimony scores because the hill
        # climber maximises; report the natural (positive, lower-is-better)
        # extra-lineage count.
        best_net, best_negated = _best_of(driver.results)
        return InferenceResult(
            best=best_net,
            score=float(-best_negated),
            method=self.method,
            lower_is_better=True,
            trace=driver.results,
            raw=driver.results,
        )

    def score(self, network: Network, data: GeneTrees, *, optimize: bool = False) -> float:
        """Score ``network`` by its extra-lineage count against ``data``.

        Args:
            network: The (MUL-tree-compatible) network to score.
            data: Gene-tree topologies to reconcile against.
            optimize: Must be ``False``; parsimony has no continuous
                parameters to optimise (see Raises).

        Returns:
            float: The (positive, lower-is-better) extra-lineage count.

        Raises:
            ValueError: If ``optimize`` is ``True``.
        """
        from ._infer_mp_allop import allop_parsimony_score

        # Parsimony is defined on topologies, so there are no continuous
        # parameters to fit: silently ignoring the request would be worse
        # than refusing it.
        if optimize:
            raise ValueError(
                f"{type(self.criterion).__name__} scores topologies, so there "
                "are no continuous parameters to optimise; use optimize=False."
            )
        return float(allop_parsimony_score(
            network,
            list(data.trees),
            self.model.resolve_subgenome_map(data),
            rng=np.random.default_rng(self.search.get("seed")),
        ))
