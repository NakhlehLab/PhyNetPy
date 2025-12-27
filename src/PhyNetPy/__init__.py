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
from .NetworkParser import NetworkParser
from .Newick import Newick

# Models and utilities
from .GTR import GTR, JC, K80, HKY
from .BirthDeath import CBDP
from .GraphUtils import *
from .GeneTrees import GeneTrees

# Validation
from .Validation import ValidationReport, ValidationError

# New architecture (v1.1+)
from .ScoringStrategy import (
    ScoringStrategy,
    ScoringContext,
    SubtreeIndependentScoring,
    FullRecomputeScoring,
    CompositeScoring
)
from .NetworkAdapter import NetworkAdapter, NetworkObserver
from .ModelBuilder import (
    ModelBuilder,
    BuiltModel,
    BuildPhase,
    BuildContext,
    BuildError,
    NetworkPhase,
    DataPhase,
    ParameterPhase,
    ScoringPhase,
    ValidationPhase,
    CustomPhase
)
from .MCMC_BiMarkers_v2 import (
    BiMarkersQ,
    VPITracker,
    BiMarkersScoringV2,
    BiMarkersModel,
    BiMarkersState,
    BiMarkersScoringPhase,
    MSAPhase,
    build_bimarkers_model,
    SNP_LIKELIHOOD_V2,
    MCMC_BIMARKERS_V2,
    n_to_index,
    nr_to_index,
    index_to_nr
)

__version__ = "1.1.0"
__author__ = "Mark Kessler"

