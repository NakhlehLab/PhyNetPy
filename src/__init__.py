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
PhyNetPy - Phylogenetic Network Python Library

A comprehensive library for phylogenetic network inference and analysis.
"""

# Core data structures
from .Network import Network, Node, Edge
from .MSA import MSA
from .Matrix import Matrix
from .Alphabet import Alphabet

# Parsing and I/O
from .Newick import get_labels, NexusTemplate, NewickParserError
from .IO import (read_fasta, write_fasta, read_fasta_records, 
                  write_fasta_from_network, read_vcf, write_vcf, 
                  read_vcf_metadata, read_newick, read_newick_file,
                  write_newick, write_newick_file, read_nexus,
                  read_nexus_msa, write_nexus, convert_newick,
                  detect_newick_standard)

# Models and utilities
from .GTR import GTR, JC, K80, HKY
from .BirthDeath import CBDP
from .GraphUtils import *
from .GeneTrees import GeneTrees

# Validation
from .Validation import (ValidationSummary, ValidationError, 
                          GeneTreeReport, GeneTreeAggregateSummary)

# New architecture (v1.1+)

from .BiMarkers import *
from .SNPSimulator import simulate as simulate_snp, random_network, SimulatedSNPData

# v1 architecture -- NOT CUDA-only.  This is the live Model/ModelGraph search
# infrastructure underneath every gene-tree-topology method (MPL, MCMC_GT,
# InferNetwork_ML, INFER_MP_ALLOP) as well as the CPU BiMarkers pipeline.
from .ModelGraph import *
from .ModelFactory import *
from .MetropolisHastings import MetropolisHastings, HillClimbing, SimulatedAnnealing, ProposalKernel
from .State import State
from .ModelMove import Move, SwitchParentage, AddReticulation, RemoveReticulation, FlipReticulation, SPR
from .ModelSelection import reticulation_sweep, SweepResult, SweepRow
from .Logger import Logger

# Inference methods.  The public, recommended import path is
# ``phynetpy.infer`` -- see ``src/infer.py`` for the curated surface.  The
# underscore-prefixed modules below hold the actual implementations and
# are not part of the public API.  ``infer.py``'s ``__all__`` already covers
# INFER_MP_ALLOP / MCMC_GT / InferNetwork_ML and friends, so they don't need
# to be re-imported directly here.
from . import infer
from .infer import *  # noqa: F401,F403  -- top-level re-export for ergonomics

# Data-oriented facade: choose an entry point by data type and pass a
# scoring strategy.  See ``src/LikelihoodStrategies.py``.
from .LikelihoodStrategies import (
    Config,
    InferenceResult,
    ScoringStrategy,
    Parsimony,
    PseudoLikelihood,
    MaximumLikelihood,
    MCMC,
    UnsupportedStrategyError,
    Score_Network_Using_GT,
    Infer_Network_From_GT,
    Score_Network_Using_MSA,
    Infer_Network_From_MSA,
    Score_Network_Using_Sites,
    Infer_Network_From_Sites,
)

# CUDA-accelerated BiMarkers (optional, currently unpackaged).
#
# A CUDA implementation (``MCMC_BiMarkers_CUDA.py``) exists in the pre-restructure
# ``1.1/`` snapshot but has not been ported into ``src/`` or validated against
# the current ``ModelGraph``/``Network`` API, so it is intentionally not wired
# up here.  Flip this back to a ``try: from .MCMC_BiMarkers_CUDA import ...``
# block once a vetted copy lands in ``src/``.
#
# NOTE: ``MCMC_BIMARKERS`` and ``SNP_LIKELIHOOD`` already have working CPU
# implementations re-exported above via ``from .infer import *`` -- they are
# deliberately *not* redefined here so that CUDA being unavailable doesn't
# clobber the CPU path.  Only the CUDA-exclusive names are stubbed.
CUDA_AVAILABLE = False
CUPY_AVAILABLE = False
NUMBA_CUDA_AVAILABLE = False

def SNP_LIKELIHOOD_DATA(*args, **kwargs):
    raise ImportError("CUDA BiMarkers is not yet packaged in phynetpy. Use the CPU path (phynetpy.infer.SNP_LIKELIHOOD) instead.")
def benchmark_cuda_vs_cpu(*args, **kwargs):
    raise ImportError("CUDA BiMarkers is not yet packaged in phynetpy.")
def get_cuda_device_info(*args, **kwargs):
    print("CUDA BiMarkers not available: not yet packaged in phynetpy.")
    return False

__version__ = "0.4.0"
__author__ = "Mark Kessler"

