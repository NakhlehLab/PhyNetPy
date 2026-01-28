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
from .Newick import get_labels, NexusTemplate, NewickParserError

# Models and utilities
from .GTR import GTR, JC, K80, HKY
from .BirthDeath import CBDP
from .GraphUtils import *
from .GeneTrees import GeneTrees

# Validation
from .Validation import ValidationSummary, ValidationError

# New architecture (v1.1+)

from .BiMarkers import *

# Old architecture (v1) - for CUDA BiMarkers support
from .ModelGraph2 import *
from .ModelFactory2 import *
from .MetropolisHastings import MetropolisHastings, HillClimbing, ProposalKernel
from .State import State
from .ModelMove import Move, SwitchParentage, AddReticulation, RemoveReticulation, FlipReticulation
from .Logger import Logger

# CUDA-accelerated BiMarkers (optional)
# CUDA_AVAILABLE: True if CuPy can access GPU (works with CUDA 13.x)
# NUMBA_CUDA_AVAILABLE: True if numba CUDA kernels work (requires compatible toolkit)
CUDA_AVAILABLE = False
CUPY_AVAILABLE = False
NUMBA_CUDA_AVAILABLE = False
try:
    from .MCMC_BiMarkers_CUDA import (
        MCMC_BIMARKERS, SNP_LIKELIHOOD, SNP_LIKELIHOOD_DATA,
        benchmark_cuda_vs_cpu, get_cuda_device_info, CUDAEvaluator,
        BiMarkersTransition, PartialLikelihoods,
        CUPY_AVAILABLE as _CUPY_AVAIL,
        NUMBA_CUDA_AVAILABLE as _NUMBA_CUDA_AVAIL,
        CUDA_IMPORTS_AVAILABLE
    )
    CUPY_AVAILABLE = _CUPY_AVAIL
    NUMBA_CUDA_AVAILABLE = _NUMBA_CUDA_AVAIL
    CUDA_AVAILABLE = CUPY_AVAILABLE  # CuPy is the primary GPU backend
except ImportError:
    # CUDA dependencies not installed - provide stub functions
    def MCMC_BIMARKERS(*args, **kwargs):
        raise ImportError("CUDA BiMarkers requires cupy and numba. Install with: pip install phynetpy[cuda]")
    def SNP_LIKELIHOOD(*args, **kwargs):
        raise ImportError("CUDA BiMarkers requires cupy and numba. Install with: pip install phynetpy[cuda]")
    def SNP_LIKELIHOOD_DATA(*args, **kwargs):
        raise ImportError("CUDA BiMarkers requires cupy and numba. Install with: pip install phynetpy[cuda]")
    def benchmark_cuda_vs_cpu(*args, **kwargs):
        raise ImportError("CUDA BiMarkers requires cupy and numba. Install with: pip install phynetpy[cuda]")
    def get_cuda_device_info(*args, **kwargs):
        print("CUDA BiMarkers not available. Install with: pip install phynetpy[cuda]")
        return False

__version__ = "1.1.0"
__author__ = "Mark Kessler"

