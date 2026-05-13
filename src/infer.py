#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##############################################################################

"""Public inference API for PhyNetPy.

This is the single front door for running phylogenetic-network inference.
Users should import everything they need from here rather than reaching into
the implementation modules (the underscore-prefixed ``_mpl``, ``_mcmc_gt``,
``_infer_mp_allop``)::

    from phynetpy.infer import MPL, MCMC_GT, INFER_MP_ALLOP

The module exposes:

* :class:`MPL` -- maximum pseudo-likelihood inference (Yu & Nakhleh, 2015).
* :class:`MCMC_GT` -- Bayesian / HC / SA inference from gene-tree topologies
  (Wen & Nakhleh, 2018).
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
