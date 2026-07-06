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
Author : Mark Kessler
First Included in Version : 0.5.0

Data-oriented inference front door for PhyNetPy.

Historically each inference method had a bespoke name (``INFER_MP_ALLOP``,
``MCMC_SEQ``, ``MCMC_BIMARKERS``, ``MPL`` ...) that bundled *what data you
have* with *how you want to score it*.  This module untangles those two
axes into an orthogonal, discoverable API:

* **What data you have** picks the entry-point function
  (``*_GT`` for gene trees, ``*_MSA`` for sequence alignments, ``*_Sites``
  for biallelic SNP sites).
* **How you want to score** is a *strategy object* you pass in -- one of
  :class:`Parsimony`, :class:`PseudoLikelihood`, :class:`MaximumLikelihood`,
  or :class:`MCMC` -- each wrapping a :class:`Config` of the knobs that
  method needs (iteration count, search driver, burn-in, flags, ...).

There are two verbs per data type -- *score* an existing network, or
*infer* the best network::

    from phynetpy.LikelihoodStrategies import (
        Infer_Network_From_GT, Score_Network_Using_GT,
        Parsimony, PseudoLikelihood, MaximumLikelihood, MCMC, Config,
    )

    # Maximum-parsimony allopolyploid inference == the old INFER_MP_ALLOP.
    result = Infer_Network_From_GT(
        my_gene_trees, Parsimony(Config(num_iter=500)), allele_mapping,
    )

    # Bayesian co-estimation from sequences == MCMC_SEQ (or MCMC_BiMarkers
    # when the alignment is biallelic SNP data).
    result = Infer_Network_From_MSA(my_msa, MCMC(Config(num_iter=20_000)))

Each ``Infer_*`` call returns an :class:`InferenceResult` (best network +
objective value + the underlying method's raw result object); each
``Score_*`` call returns a ``float``.

The concrete numerical engines live in the underscore-prefixed
implementation modules (``_mpl``, ``_mcmc_gt``, ``_infernetworkml``,
``_mcmc_seq``, ``_infer_mp_allop``) and ``BiMarkers``; this module only
routes to them.  It never re-implements any likelihood.

Docs   - [x]
Tests  - [ ]
Design - [x]
"""

from __future__ import annotations

import os
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np

from .Network import Network
from .GeneTrees import GeneTrees
from .MSA import MSA
from .IO import read_nexus

# Core (always-available) engines.
from ._mpl import MPL
from ._mcmc_gt import MCMC_GT, MCMC_GTPriors, _populate_default_branch_lengths
from ._infernetworkml import InferNetwork_ML
from ._infer_mp_allop import (
    InferMPAllop,
    Allop_MUL,
    allele_map_set_ilp,
    partition_gene_trees,
)

__all__ = [
    "Config",
    "InferenceResult",
    "ScoringStrategy",
    "Parsimony",
    "PseudoLikelihood",
    "MaximumLikelihood",
    "MCMC",
    "UnsupportedStrategyError",
    "Score_Network_Using_GT",
    "Infer_Network_From_GT",
    "Score_Network_Using_MSA",
    "Infer_Network_From_MSA",
    "Score_Network_Using_Sites",
    "Infer_Network_From_Sites",
]


# ======================================================================
# Errors
# ======================================================================


class UnsupportedStrategyError(TypeError):
    """Raised when a (data type, scoring strategy) pair has no engine.

    e.g. asking for :class:`Parsimony` on a sequence alignment -- parsimony
    allopolyploid inference is only defined over gene-tree topologies.
    """


# ======================================================================
# Configuration
# ======================================================================


@dataclass
class Config:
    """Unified configuration for every scoring strategy.

    One flat container carries the knobs of all four strategies; each
    strategy reads only the fields it understands and ignores the rest,
    so the same ``Config`` can be reused across strategies.  Fields left
    at ``None`` fall through to the underlying engine's own default,
    so you only ever set what you want to change.

    Attributes:
        search: Search driver -- ``"hc"`` (hill climbing), ``"sa"``
            (simulated annealing) or ``"mh"`` (Metropolis-Hastings /
            Bayesian sampling).  ``None`` lets each strategy pick its
            natural default (``"hc"`` for likelihood maximisation,
            ``"mh"`` for :class:`MCMC`).
        num_iter: Number of proposed moves / sampler iterations.
        burn_in: Sampler iterations discarded before samples are kept
            (MCMC only).
        sample_freq: Thinning interval between kept samples (MCMC only).
        temperatures: Optional Metropolis-coupled (MC3) temperature
            ladder for :class:`MCMC` on sequence data.
        max_reticulations: Upper bound on the number of reticulations the
            search may add.
        max_lvl: Maximum network level (blob complexity) allowed.
        preset: One-word search profile for the topology-search methods
            (``"default"`` / ``"fast"`` / ``"accurate"`` / ``"phylonet"``).
        opt_bl: Optimise branch lengths at the end instead of sampling
            them during the climb.
        fix_st: Fix the starting-tree backbone (drop ``SPR``).
        optimize_scope: Which continuous parameters per-topology
            optimisation touches (``"gamma"`` / ``"reticulation"`` /
            ``"all"``).
        pseudo: Force triplet pseudo-likelihood scoring where the engine
            supports switching (e.g. :class:`MCMC` / :class:`MaximumLikelihood`
            on gene trees).
        u: Biallelic-marker red->green mutation rate (SNP methods).
        v: Biallelic-marker green->red mutation rate (SNP methods).
        coal: Coalescence rate (SNP methods).
        samples: Taxon -> sample-count map required by the one-off SNP
            likelihood (``Score_Network_Using_Sites``).
        seed: Random seed for reproducibility.
        n_workers: Worker processes for parallel gene-tree scoring.
        num_runs: Independent hill-climbing restarts
            (:class:`MaximumLikelihood` on gene trees).
        extra: Escape hatch -- extra keyword arguments forwarded verbatim
            to the underlying ``search`` / method call.
    """

    # Search driver + budget
    search: Optional[str] = None
    num_iter: Optional[int] = None

    # Sampling (MCMC)
    burn_in: Optional[int] = None
    sample_freq: Optional[int] = None
    temperatures: Optional[list[float]] = None

    # Topology-search flags (gene-tree methods)
    max_reticulations: Optional[int] = None
    max_lvl: Optional[int] = None
    preset: Optional[str] = None
    opt_bl: Optional[bool] = None
    fix_st: Optional[bool] = None
    optimize_scope: Optional[str] = None
    pseudo: Optional[bool] = None
    num_runs: Optional[int] = None

    # Biallelic-marker / SNP knobs
    u: float = 0.5
    v: float = 0.5
    coal: float = 1.0
    samples: Optional[dict[str, int]] = None

    # General
    seed: Optional[int] = None
    n_workers: int = 1

    # Verbatim passthrough to the underlying driver.
    extra: dict = field(default_factory=dict)


def _kw(**pairs: Any) -> dict[str, Any]:
    """Drop ``None`` values so underlying engine defaults still apply."""
    return {k: v for k, v in pairs.items() if v is not None}


# ======================================================================
# Result
# ======================================================================


@dataclass
class InferenceResult:
    """Uniform result of any ``Infer_*`` call.

    Attributes:
        network: Best network found by the search.
        score: The objective value the underlying method reports for
            ``network``.  For likelihood/posterior methods this is a log
            value (higher / closer to zero is better); for
            :class:`Parsimony` it is the extra-lineage parsimony score
            (lower is better).
        method: Human-readable name of the concrete engine that ran
            (e.g. ``"INFER_MP_ALLOP"``).
        raw: The underlying method's native return value
            (:class:`~phynetpy._mcmc_gt.MCMCGTResult`,
            :class:`~phynetpy._infernetworkml.InferNetworkMLResult`,
            a ``dict[Network, float]``, ...), for callers that need the
            samples, leaderboard, or diagnostics.
    """

    network: Network
    score: float
    method: str
    raw: Any = None


# ======================================================================
# Input normalisation
# ======================================================================


def _as_gene_trees(
    data: Union[GeneTrees, list, str, os.PathLike],
    mapping: Optional[dict[str, list[str]]],
) -> GeneTrees:
    """Coerce assorted gene-tree inputs into a :class:`GeneTrees`.

    Accepts a :class:`GeneTrees`, a ``list[Network]``, or a path to a
    NEXUS file.  ``mapping`` (species -> allele labels), when given, is
    attached as the collection's ``species_gene_mapping``.
    """
    if isinstance(data, GeneTrees):
        gts = data
    elif isinstance(data, (str, os.PathLike)):
        gts = read_nexus(str(data), return_type="genetrees")
    elif isinstance(data, (list, tuple, set)):
        gts = GeneTrees(gene_tree_list=list(data))
    else:
        raise TypeError(
            "gene-tree data must be a GeneTrees, a list of Networks, or a "
            f"path to a NEXUS file; got {type(data).__name__}."
        )
    if mapping is not None:
        gts.species_gene_mapping = mapping
    return gts


def _gene_tree_list(data: Union[GeneTrees, list, str, os.PathLike]) -> list[Network]:
    """Coerce gene-tree inputs into a plain ``list[Network]`` (MP-Allop)."""
    if isinstance(data, GeneTrees):
        return list(data.trees)
    if isinstance(data, (str, os.PathLike)):
        return list(read_nexus(str(data)))
    if isinstance(data, (list, tuple, set)):
        return list(data)
    raise TypeError(
        "gene-tree data must be a GeneTrees, a list of Networks, or a path "
        f"to a NEXUS file; got {type(data).__name__}."
    )


def _resolve_mapping(
    gts: GeneTrees, mapping: Optional[dict[str, list[str]]],
) -> dict[str, list[str]]:
    """Return an explicit mapping, else the collection's best guess."""
    if mapping is not None:
        return mapping
    return gts.mp_allop_map()


def _consensus_start(gts: GeneTrees) -> Network:
    """Majority-rule consensus starting network with default branch lengths."""
    seed = gts.build_majority_rule_consensus_tree(threshold=0.5)
    _populate_default_branch_lengths(seed)
    return seed


def _as_loci(data: Union[MSA, list, str, os.PathLike]) -> list[Any]:
    """Coerce sequence input into the per-locus list ``MCMC_SEQ`` expects."""
    if isinstance(data, MSA):
        return [data]
    if isinstance(data, (str, os.PathLike)):
        return [MSA(str(data))]
    if isinstance(data, (list, tuple)):
        return list(data)
    raise TypeError(
        "MSA data must be an MSA, a list of per-locus alignments, or a path "
        f"to an alignment file; got {type(data).__name__}."
    )


def _sites_filename(data: Union[MSA, str, os.PathLike]) -> str:
    """Extract a NEXUS filename for the file-based SNP engines."""
    if isinstance(data, (str, os.PathLike)):
        return str(data)
    if isinstance(data, MSA):
        fname = getattr(data, "filename", None)
        if fname and fname != "No Filename Given":
            return fname
    raise TypeError(
        "SNP/site data must be a path to a NEXUS file (or an MSA loaded from "
        "one): the biallelic-marker engine reads its matrix and network from "
        "the file directly."
    )


def _best_of(scores: dict[Network, float], *, lower_is_better: bool) -> tuple[Network, float]:
    """Pick the winning network from a ``{network: score}`` dict."""
    if not scores:
        raise RuntimeError("inference produced no scored networks.")
    picker = min if lower_is_better else max
    net = picker(scores, key=scores.get)
    return net, scores[net]


# ======================================================================
# Strategies
# ======================================================================


class ScoringStrategy(ABC):
    """Base class for a scoring method plus its :class:`Config`.

    A strategy is a small, declarative object: it names *how* to score a
    network and holds the configuration for that method.  The data-type
    entry-point functions dispatch to the ``_score_*`` / ``_infer_*``
    hooks below; each concrete strategy overrides only the hooks it
    supports and inherits a clear "unsupported" error for the rest.
    """

    #: Human-readable strategy name used in error messages.
    label = "scoring strategy"

    def __init__(self, config: Optional[Config] = None) -> None:
        """Store the configuration (defaults to an all-defaults :class:`Config`)."""
        self.config = config if config is not None else Config()

    # -- dispatch hooks (default: unsupported) ------------------------

    def _unsupported(self, data_kind: str, verb: str) -> "UnsupportedStrategyError":
        return UnsupportedStrategyError(
            f"{self.label} cannot {verb} a network from {data_kind}. "
            f"See LikelihoodStrategies for the supported (data, strategy) pairs."
        )

    def _score_gt(self, network, gene_trees, mapping):  # noqa: D401
        raise self._unsupported("gene trees", "score")

    def _infer_gt(self, gene_trees, mapping, start_network):
        raise self._unsupported("gene trees", "infer")

    def _score_msa(self, network, msa, mapping):
        raise self._unsupported("a sequence alignment", "score")

    def _infer_msa(self, msa, mapping, start_network):
        raise self._unsupported("a sequence alignment", "infer")

    def _score_sites(self, network, sites, mapping):
        raise self._unsupported("SNP sites", "score")

    def _infer_sites(self, sites, mapping, start_network):
        raise self._unsupported("SNP sites", "infer")


class Parsimony(ScoringStrategy):
    """Maximum-parsimony allopolyploid inference (INFER_MP_ALLOP).

    Scores a species network by the minimum number of extra lineages
    needed to reconcile the gene trees into it under an allopolyploid
    multiple-labelled (MUL) tree model (Hejase et al.).  Defined only for
    gene-tree data.

    Config fields used: ``num_iter`` (search iterations, default 500),
    ``seed``.
    """

    label = "Parsimony"

    def _score_gt(self, network, gene_trees, mapping):
        gts = _as_gene_trees(gene_trees, mapping)
        subgenome_map = _resolve_mapping(gts, mapping)
        gt_list = list(gts.trees)
        rng = np.random.default_rng(self.config.seed)

        mul = Allop_MUL(subgenome_map, rng)
        mul.to_mul(network)
        for gene_tree in gt_list:
            gene_tree.put_item(
                "allele maps", allele_map_set_ilp(gene_tree, subgenome_map),
            )
            gene_tree.put_item(
                "leaf descendants", gene_tree.leaf_descendants_all(),
            )
        return float(mul.score(gt_list))

    def _infer_gt(self, gene_trees, mapping, start_network):
        gts = _as_gene_trees(gene_trees, mapping)
        subgenome_map = _resolve_mapping(gts, mapping)
        gt_list = list(gts.trees)
        rng = np.random.default_rng(self.config.seed)

        if start_network is None:
            start_network = partition_gene_trees(subgenome_map, rng=rng)

        iter_ct = self.config.num_iter if self.config.num_iter is not None else 500
        model = InferMPAllop(
            start_network, subgenome_map, gt_list, iter_ct, rng=rng,
        )
        model.run()
        # The leaderboard stores *negated* parsimony scores (the hill climber
        # maximises), so the best network is the max; report it as a positive
        # extra-lineage parsimony score (lower is better).
        best_net, best_negated = _best_of(model.results, lower_is_better=False)
        return InferenceResult(
            best_net, float(-best_negated), "INFER_MP_ALLOP", model.results,
        )


class PseudoLikelihood(ScoringStrategy):
    """Maximum pseudo-likelihood inference from gene-tree triplets (MPL).

    Yu & Nakhleh (2015): scores a network by the product of per-triplet
    coalescent probabilities.  Defined only for gene-tree data.

    Config fields used: ``search`` (``"hc"``/``"sa"``), ``num_iter``,
    ``max_reticulations``, ``max_lvl``, ``preset``, ``opt_bl``,
    ``fix_st``, ``seed``, ``extra``.
    """

    label = "PseudoLikelihood"

    def _score_gt(self, network, gene_trees, mapping):
        gts = _as_gene_trees(gene_trees, mapping)
        mp = MPL(network, gts, _resolve_mapping(gts, mapping))
        return float(mp.score())

    def _infer_gt(self, gene_trees, mapping, start_network):
        gts = _as_gene_trees(gene_trees, mapping)
        resolved = _resolve_mapping(gts, mapping)
        if start_network is None:
            start_network = _consensus_start(gts)

        cfg = self.config
        mp = MPL(start_network, gts, resolved)
        best_score = mp.search(
            method=cfg.search or "hc",
            **_kw(
                num_iter=cfg.num_iter,
                max_reticulations=cfg.max_reticulations,
                preset=cfg.preset,
                opt_bl=cfg.opt_bl,
                fix_st=cfg.fix_st,
                max_lvl=cfg.max_lvl,
                seed=cfg.seed,
            ),
            **cfg.extra,
        )
        return InferenceResult(mp.net, float(best_score), "MPL", best_score)


class MaximumLikelihood(ScoringStrategy):
    """Full-likelihood inference under the MSNC / biallelic-marker model.

    For gene-tree data this is PhyloNet's ``InferNetwork_ML`` /
    ``CalGTProb`` (Yu et al. 2014).  For SNP sites, ``Score_*`` computes
    the exact biallelic-marker likelihood (Bryant et al., SNAPP-style).

    Config fields used (gene trees): ``num_iter``, ``num_runs``,
    ``max_reticulations``, ``max_lvl``, ``preset``, ``opt_bl``,
    ``fix_st``, ``optimize_scope``, ``pseudo``, ``seed``, ``n_workers``,
    ``extra``; (sites): ``u``, ``v``, ``coal``, ``samples``.
    """

    label = "MaximumLikelihood"

    def _score_gt(self, network, gene_trees, mapping):
        gts = _as_gene_trees(gene_trees, mapping)
        inf = InferNetwork_ML(network, gts, _resolve_mapping(gts, mapping))
        return float(inf.score(**_kw(pseudo=self.config.pseudo,
                                     n_workers=self.config.n_workers)))

    def _infer_gt(self, gene_trees, mapping, start_network):
        gts = _as_gene_trees(gene_trees, mapping)
        resolved = _resolve_mapping(gts, mapping)
        cfg = self.config
        if start_network is None:
            inf = InferNetwork_ML.from_consensus(
                gts, resolved, max_reticulations=cfg.max_reticulations,
            )
        else:
            inf = InferNetwork_ML(
                start_network, gts, resolved,
                max_reticulations=cfg.max_reticulations,
            )
        result = inf.search(
            **_kw(
                num_iter=cfg.num_iter,
                num_runs=cfg.num_runs,
                preset=cfg.preset,
                opt_bl=cfg.opt_bl,
                fix_st=cfg.fix_st,
                max_lvl=cfg.max_lvl,
                optimize_scope=cfg.optimize_scope,
                pseudo=cfg.pseudo,
                seed=cfg.seed,
                n_workers=cfg.n_workers,
            ),
            **cfg.extra,
        )
        return InferenceResult(
            result.best_network, float(result.best_log_likelihood),
            "InferNetwork_ML", result,
        )

    def _score_sites(self, network, sites, mapping):
        from .BiMarkers import SNP_LIKELIHOOD
        cfg = self.config
        if cfg.samples is None:
            raise ValueError(
                "Scoring SNP sites needs Config.samples (a taxon -> "
                "sample-count map)."
            )
        return float(SNP_LIKELIHOOD(
            _sites_filename(sites), cfg.u, cfg.v, cfg.coal, cfg.samples,
        ))


class MCMC(ScoringStrategy):
    """Bayesian (and HC/SA) network inference across every data type.

    Dispatches to the appropriate sampler for the data at hand:

    * gene trees      -> ``MCMC_GT`` (Wen & Nakhleh 2018).
    * sequence data   -> ``MCMC_SEQ``; when the alignment is biallelic
      SNP data, ``MCMC_BIMARKERS`` instead.
    * SNP sites       -> ``MCMC_BIMARKERS`` (Bryant et al.).

    Config fields used: ``search`` (gene trees: ``"mh"``/``"hc"``/``"sa"``,
    default ``"mh"``), ``num_iter``, ``burn_in``, ``sample_freq``,
    ``temperatures``, ``max_reticulations``, ``max_lvl``, ``preset``,
    ``opt_bl``, ``fix_st``, ``pseudo``, ``seed``, ``n_workers``,
    ``u``/``v``/``coal`` (SNP), ``extra``.
    """

    label = "MCMC"

    # -- gene trees ---------------------------------------------------

    def _score_gt(self, network, gene_trees, mapping):
        gts = _as_gene_trees(gene_trees, mapping)
        mc = MCMC_GT(network, gts, _resolve_mapping(gts, mapping))
        return float(mc.score())

    def _infer_gt(self, gene_trees, mapping, start_network):
        gts = _as_gene_trees(gene_trees, mapping)
        resolved = _resolve_mapping(gts, mapping)
        cfg = self.config
        if start_network is None:
            mc = MCMC_GT.from_consensus(gts, resolved)
        else:
            mc = MCMC_GT(start_network, gts, resolved)
        result = mc.search(
            method=cfg.search or "mh",
            **_kw(
                num_iter=cfg.num_iter,
                burn_in=cfg.burn_in,
                thin=cfg.sample_freq,
                max_reticulations=cfg.max_reticulations,
                preset=cfg.preset,
                opt_bl=cfg.opt_bl,
                fix_st=cfg.fix_st,
                max_lvl=cfg.max_lvl,
                pseudo=cfg.pseudo,
                seed=cfg.seed,
                n_workers=cfg.n_workers,
            ),
            **cfg.extra,
        )
        return InferenceResult(
            result.best_network, float(result.best_log_posterior),
            "MCMC_GT", result,
        )

    # -- sequence alignments ------------------------------------------

    def _score_msa(self, network, msa, mapping):
        if mapping is None:
            raise ValueError("Scoring an alignment needs a species -> allele mapping.")
        from ._mcmc_seq import MCMC_SEQ
        mc = MCMC_SEQ(_as_loci(msa), mapping, network, **_kw(theta=None))
        return float(mc.score())

    def _infer_msa(self, msa, mapping, start_network):
        if mapping is None:
            raise ValueError("Inference from an alignment needs a species -> allele mapping.")
        loci = _as_loci(msa)
        if _looks_like_snp(loci):
            return self._infer_sites(msa, mapping, start_network)

        from ._mcmc_seq import MCMC_SEQ
        cfg = self.config
        mc = MCMC_SEQ(loci, mapping, start_network)
        result = mc.search(
            **_kw(
                num_iter=cfg.num_iter,
                burn_in=cfg.burn_in,
                sample_freq=cfg.sample_freq,
                temperatures=cfg.temperatures,
                seed=cfg.seed,
            ),
            **cfg.extra,
        )
        return InferenceResult(
            result.map_network, float(result.map_log_posterior),
            "MCMC_SEQ", result,
        )

    # -- SNP sites ----------------------------------------------------

    def _infer_sites(self, sites, mapping, start_network):
        from .BiMarkers import MCMC_BIMARKERS
        cfg = self.config
        scores = MCMC_BIMARKERS(
            _sites_filename(sites), u=cfg.u, v=cfg.v, coal=cfg.coal,
        )
        best_net, best_score = _best_of(scores, lower_is_better=False)
        return InferenceResult(best_net, float(best_score), "MCMC_BIMARKERS", scores)

    def _score_sites(self, network, sites, mapping):
        # A one-off SNP score is the biallelic-marker likelihood; reuse the
        # MaximumLikelihood path (which requires Config.samples).
        return MaximumLikelihood(self.config)._score_sites(network, sites, mapping)


def _looks_like_snp(loci: list[Any]) -> bool:
    """Heuristic: does this alignment look like biallelic SNP data?

    True iff every observed character across the loci is a biallelic SNP
    token (``0``/``1``/``2`` or gaps/missing).  Anything with nucleotide
    letters is treated as DNA sequence.
    """
    snp_chars = set("012-?.N")
    for locus in loci:
        if isinstance(locus, MSA):
            seqs = ["".join(map(str, r.get_sequence()))
                    if hasattr(r, "get_sequence") else str(r)
                    for r in locus.get_records()]
        elif isinstance(locus, dict):
            seqs = [str(v) for v in locus.values()]
        else:
            return False
        for seq in seqs:
            if any(ch.upper() not in snp_chars for ch in seq):
                return False
    return True


# ======================================================================
# Top-level data-oriented entry points
# ======================================================================


def Score_Network_Using_GT(
    network: Network,
    gene_trees: Union[GeneTrees, list, str, os.PathLike],
    strategy: ScoringStrategy,
    mapping: Optional[dict[str, list[str]]] = None,
) -> float:
    """Score ``network`` against gene-tree data using ``strategy``.

    Args:
        network: Species network to score.
        gene_trees: A :class:`~phynetpy.GeneTrees.GeneTrees`, a list of
            gene-tree :class:`~phynetpy.Network.Network` objects, or a path
            to a NEXUS gene-tree file.
        strategy: :class:`Parsimony`, :class:`PseudoLikelihood`,
            :class:`MaximumLikelihood`, or :class:`MCMC`.
        mapping: Species -> allele/gene-copy labels.  ``None`` derives an
            identity/consensus mapping from the trees.

    Returns:
        The strategy's objective value for ``network`` (see
        :class:`InferenceResult` for sign conventions).
    """
    return strategy._score_gt(network, gene_trees, mapping)


def Infer_Network_From_GT(
    gene_trees: Union[GeneTrees, list, str, os.PathLike],
    strategy: ScoringStrategy,
    mapping: Optional[dict[str, list[str]]] = None,
    *,
    start_network: Optional[Network] = None,
) -> InferenceResult:
    """Infer the best network from gene-tree data using ``strategy``.

    ``Infer_Network_From_GT(gts, Parsimony(cfg), allele_mapping)`` is the
    modern spelling of ``INFER_MP_ALLOP``.

    Args:
        gene_trees: A :class:`~phynetpy.GeneTrees.GeneTrees`, a list of
            gene-tree networks, or a path to a NEXUS gene-tree file.
        strategy: :class:`Parsimony`, :class:`PseudoLikelihood`,
            :class:`MaximumLikelihood`, or :class:`MCMC`.
        mapping: Species -> allele/gene-copy labels.  ``None`` derives one
            from the trees.
        start_network: Optional starting topology.  ``None`` uses a
            method-appropriate default (parsimony partition or majority-rule
            consensus).

    Returns:
        An :class:`InferenceResult`.
    """
    return strategy._infer_gt(gene_trees, mapping, start_network)


def Score_Network_Using_MSA(
    network: Network,
    msa: Union[MSA, list, str, os.PathLike],
    strategy: ScoringStrategy,
    mapping: Optional[dict[str, list[str]]] = None,
) -> float:
    """Score ``network`` against a sequence alignment using ``strategy``.

    Args:
        network: Species network to score.
        msa: An :class:`~phynetpy.MSA.MSA`, a list of per-locus alignments,
            or a path to an alignment file.
        strategy: Typically :class:`MCMC` (sequence likelihood).
        mapping: Species -> allele labels (required).

    Returns:
        The strategy's objective value for ``network``.
    """
    return strategy._score_msa(network, msa, mapping)


def Infer_Network_From_MSA(
    msa: Union[MSA, list, str, os.PathLike],
    strategy: ScoringStrategy,
    mapping: Optional[dict[str, list[str]]] = None,
    *,
    start_network: Optional[Network] = None,
) -> InferenceResult:
    """Infer the best network from a sequence alignment using ``strategy``.

    ``Infer_Network_From_MSA(msa, MCMC(cfg))`` runs ``MCMC_SEQ`` for DNA
    alignments, or ``MCMC_BIMARKERS`` when the alignment is biallelic SNP
    data.

    Args:
        msa: An :class:`~phynetpy.MSA.MSA`, a list of per-locus alignments,
            or a path to an alignment file.
        strategy: Typically :class:`MCMC`.
        mapping: Species -> allele labels (required for sequence data).
        start_network: Optional starting network; ``None`` builds one
            (UPGMA species tree for ``MCMC_SEQ``).

    Returns:
        An :class:`InferenceResult`.
    """
    return strategy._infer_msa(msa, mapping, start_network)


def Score_Network_Using_Sites(
    network: Optional[Network],
    sites: Union[MSA, str, os.PathLike],
    strategy: ScoringStrategy,
    mapping: Optional[dict[str, list[str]]] = None,
) -> float:
    """Score a network against biallelic SNP sites using ``strategy``.

    The biallelic-marker engine reads both the matrix and the network from
    the NEXUS file, so ``sites`` must be a path (or an :class:`MSA` loaded
    from one) and ``network`` may be ``None``.  ``Config.samples`` (taxon
    -> sample count) is required.

    Args:
        network: Optional; ignored -- the network is read from ``sites``.
        sites: Path to a NEXUS SNP file (or an MSA loaded from one).
        strategy: :class:`MaximumLikelihood` or :class:`MCMC`.
        mapping: Unused for SNP sites.

    Returns:
        The biallelic-marker log likelihood.
    """
    return strategy._score_sites(network, sites, mapping)


def Infer_Network_From_Sites(
    sites: Union[MSA, str, os.PathLike],
    strategy: ScoringStrategy,
    mapping: Optional[dict[str, list[str]]] = None,
    *,
    start_network: Optional[Network] = None,
) -> InferenceResult:
    """Infer the best network from biallelic SNP sites using ``strategy``.

    ``Infer_Network_From_Sites(path, MCMC(cfg))`` runs ``MCMC_BIMARKERS``.

    Args:
        sites: Path to a NEXUS SNP file (or an MSA loaded from one).
        strategy: :class:`MCMC`.
        mapping: Unused for SNP sites.
        start_network: Unused (the SNP sampler builds its own start).

    Returns:
        An :class:`InferenceResult`.
    """
    return strategy._infer_sites(sites, mapping, start_network)
