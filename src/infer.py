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
Last Edit : 5/12/26
First Included in Version : 0.4.0

Public inference API for PhyNetPy.

This is the single front door for running phylogenetic-network inference.
Users should import everything they need from here rather than reaching into
the implementation modules (the underscore-prefixed ``_mpl``, ``_mcmc_gt``,
``_infer_mp_allop``)::

    from phynetpy.infer import MPL, MCMC_GT, INFER_MP_ALLOP

The module exposes:

* :class:`MPL` -- maximum pseudo-likelihood inference (Yu & Nakhleh, 2015).
* :class:`MCMC_GT` -- Bayesian / HC / SA inference from gene-tree topologies
  (Wen & Nakhleh, 2018).
* :class:`InferNetwork_ML` -- maximum-likelihood inference from gene-tree
  topologies under the MSNC (Yu, Dong, Liu & Nakhleh, 2014).
* :func:`INFER_MP_ALLOP`, :func:`INFER_MP_ALLOP_BOOTSTRAP`,
  :func:`ALLOP_SCORE` -- maximum-parsimony allopolyploid inference
  (Hejase et al.).
* :func:`MCMC_BIMARKERS`, :func:`SNP_LIKELIHOOD` -- biallelic-marker
  likelihood (Bryant et al. SNAPP-style).

Lower-level helpers (scorers, kernels, prior containers, result types) are
re-exported as well so that advanced users can wire the search drivers to a
custom :class:`~phynetpy.MetropolisHastings.ProposalKernel` or post-process
samples without importing from the private modules.
"""

from __future__ import annotations

# MPL
from ._mpl import (
    MPL,
    MPLScorer,
    MPLKernel,
    compute_gene_tree_triplets,
    score_species_network_triplets,
    format_mpl_reference_comparison,
    save_mpl_network_newick,
    GeneTreeTripletResult,
    SpeciesNetworkTripletResult,
)

# MCMC_GT
from ._mcmc_gt import (
    MCMC_GT,
    MCMCGTScorer,
    MCMCGTKernel,
    MCMC_GTPriors,
    MCMCSample,
    MCMCGTResult,
    log_prior_network,
)

# InferNetwork_ML -- maximum-likelihood inference (Yu et al. 2014).  Shares
# MCMC_GT's MSNC likelihood engine and proposal kernel; see ``_infernetworkml``.
from ._infernetworkml import (
    InferNetwork_ML,
    InferNetworkMLResult,
    optimize_network_parameters,
)

# MCMC_SEQ -- Bayesian co-estimation of reticulate phylogenies and gene trees
# directly from multilocus sequence alignments (Wen & Nakhleh 2018).  The
# per-locus Felsenstein likelihood and timed MSNC density live in
# ``_seq_likelihood`` and match PhyloNet exactly; ``_mcmc_seq`` is the sampler.
from ._mcmc_seq import (
    MCMC_SEQ,
    MCMCSeqPriors,
    MCMCSeqSample,
    MCMCSeqResult,
    MCMCSeqKernel,
    run_parallel_chains,
    MultiChainStatus,
    MultiChainResult,
)
from ._seq_likelihood import (
    SubstitutionModel,
    JC69,
    HKY85,
    GTR,
    FelsensteinCalculator,
    gene_tree_msnc_log_density,
)

# Coalescent simulation of multilocus sequence data (the generative model
# behind MCMC_SEQ): build a known-truth data set for recovery / calibration
# checks via ``MCMC_SEQ(**data.to_mcmc_seq_kwargs())``.
from ._sim_seq import (
    SimulatedData,
    simulate_gene_tree,
    simulate_sequences,
    simulate_multilocus,
)

# Post-analysis: MCMC chain diagnostics + Tracer / NEXUS interoperability.
# Sampler results (MCMCSeqResult, MCMCGTResult) also expose ``write_log``,
# ``write_networks`` and ``summary`` helpers built on these.
from ._chain_analysis import (
    effective_sample_size,
    autocorrelation_time,
    standard_error_of_mean,
    hpd_interval,
    geweke,
    summarize,
    summarize_traces,
    ParameterSummary,
    ChainSummary,
    write_tracer_log,
    read_tracer_log,
    write_trees_nexus,
)

# Maximum-parsimony allopolyploid inference
from ._infer_mp_allop import (
    INFER_MP_ALLOP,
    INFER_MP_ALLOP_BOOTSTRAP,
    ALLOP_SCORE,
    InferMPAllop,
    MPAllopComponent,
    MPAllopScorer,
    Allop_MUL,
    AlleleMap,
)

# Biallelic-marker likelihood (CPU implementation lives in BiMarkers; an
# optional CUDA-accelerated variant is wired up by ``phynetpy/__init__.py``
# when the optional dependencies are present).
from .BiMarkers import MCMC_BIMARKERS, SNP_LIKELIHOOD

__all__ = [
    # MPL
    "MPL",
    "MPLScorer",
    "MPLKernel",
    "compute_gene_tree_triplets",
    "score_species_network_triplets",
    "format_mpl_reference_comparison",
    "save_mpl_network_newick",
    "GeneTreeTripletResult",
    "SpeciesNetworkTripletResult",
    # MCMC_GT
    "MCMC_GT",
    "MCMCGTScorer",
    "MCMCGTKernel",
    "MCMC_GTPriors",
    "MCMCSample",
    "MCMCGTResult",
    "log_prior_network",
    # InferNetwork_ML
    "InferNetwork_ML",
    "InferNetworkMLResult",
    "optimize_network_parameters",
    # MCMC_SEQ
    "MCMC_SEQ",
    "MCMCSeqPriors",
    "MCMCSeqSample",
    "MCMCSeqResult",
    "MCMCSeqKernel",
    "run_parallel_chains",
    "MultiChainStatus",
    "MultiChainResult",
    "SubstitutionModel",
    "JC69",
    "HKY85",
    "GTR",
    "FelsensteinCalculator",
    "gene_tree_msnc_log_density",
    # Coalescent simulation (known-truth data for MCMC_SEQ)
    "SimulatedData",
    "simulate_gene_tree",
    "simulate_sequences",
    "simulate_multilocus",
    # Post-analysis / Tracer interop
    "effective_sample_size",
    "autocorrelation_time",
    "standard_error_of_mean",
    "hpd_interval",
    "geweke",
    "summarize",
    "summarize_traces",
    "ParameterSummary",
    "ChainSummary",
    "write_tracer_log",
    "read_tracer_log",
    "write_trees_nexus",
    # MP_Allop
    "INFER_MP_ALLOP",
    "INFER_MP_ALLOP_BOOTSTRAP",
    "ALLOP_SCORE",
    "InferMPAllop",
    "MPAllopComponent",
    "MPAllopScorer",
    "Allop_MUL",
    "AlleleMap",
    # BiMarkers
    "MCMC_BIMARKERS",
    "SNP_LIKELIHOOD",
]
