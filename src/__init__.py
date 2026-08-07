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
PhyNetPy -- a Python library for phylogenetic network inference and analysis.

Getting started
---------------
There are two verbs, ``infer`` and ``score``.  What used to be a menu of
fifteen command names is now three orthogonal axes, each a typed object: the
data, the model (biology), and the criterion (statistical objective)::

    from phynetpy.infer import infer, score
    from phynetpy.data import GeneTrees
    from phynetpy.models import MSC, Allopolyploid
    from phynetpy.criteria import MDC, Likelihood, PseudoLikelihood, Bayesian

    gts = GeneTrees.from_file("gene_trees.nex", mapping)

    result = infer(gts, model=MSC(), criterion=PseudoLikelihood())
    print(result.best, result.score)

    s = score(result.best, gts, criterion=MDC())

Build or read a network, then analyse it::

    from phynetpy import read_newick, Network, Node, Edge, compare_networks

    net = read_newick("((A,B),C);")[0]

Layout
------
The top-level ``phynetpy`` namespace, re-exported below, holds the data
structures and analysis helpers you need to prepare inputs and inspect
outputs. It is deliberately curated -- ``__all__`` lists everything that is
public and supported.

Submodules worth importing directly:

* :mod:`phynetpy.infer` -- the two verbs, ``InferenceResult``, ``Start``,
  MCMC diagnostics, and simulation. Start here.
* :mod:`phynetpy.data`, :mod:`phynetpy.models`, :mod:`phynetpy.criteria` --
  the three inference axes.
* :mod:`phynetpy.Network` -- the :class:`~phynetpy.Network.Network` graph type
  and its primitives.
* :mod:`phynetpy.GraphUtils` -- network decomposition, distances, and
  structural edits.
* :mod:`phynetpy.IO` -- readers and writers for Newick, NEXUS, FASTA, VCF.
* :mod:`phynetpy.GTR` -- substitution models for the model-graph engines. Note
  that the sequence-MCMC engine has its own model classes
  (``JC69``/``HKY85``/``GTR``) exported from :mod:`phynetpy.infer`.

.. note::
   ``phynetpy.Model`` is :class:`phynetpy.ModelGraph.Model`, the
   probabilistic graphical model the search drivers mutate.  The *biological
   process* on the inference model axis is ``phynetpy.models.Model``; the
   bare name is deliberately not re-exported here, so neither shadows the
   other.
"""

from ._version import __version__

__author__ = "Mark Kessler"

# =============================================================================
# Core data structures
# =============================================================================

from .Network import (
    Network,
    MUL,
    Node,
    Edge,
    Branch,
    NodeSet,
    EdgeSet,
    NetworkError,
    NodeError,
    EdgeError,
)
from .MSA import MSA, DataSequence
from .Matrix import Matrix
from .Alphabet import Alphabet
from .GeneTrees import GeneTrees

# =============================================================================
# Parsing and I/O
# =============================================================================

from .Newick import get_labels, NexusTemplate, NewickParserError
from .IO import (
    read_fasta,
    read_fasta_records,
    write_fasta,
    write_fasta_from_network,
    read_vcf,
    read_vcf_metadata,
    write_vcf,
    read_newick,
    read_newick_file,
    write_newick,
    write_newick_file,
    read_nexus,
    read_nexus_msa,
    write_nexus,
    convert_newick,
    detect_newick_standard,
)
from .Validation import (
    ValidationSummary,
    ValidationError,
    GeneTreeReport,
    GeneTreeAggregateSummary,
)

# =============================================================================
# Substitution models (model-graph engines)
# =============================================================================
# The bare name ``GTR`` is intentionally *not* re-exported here: it would
# collide with the ``phynetpy.GTR`` module, and the sequence-MCMC engine has a
# separate ``GTR`` class of its own.  Use ``from phynetpy.GTR import GTR`` for
# the model-graph class, or ``from phynetpy.infer import GTR`` for the
# sequence-likelihood class.

from .GTR import JC, K80, HKY, F81, K81, SYM, TN93, SubstitutionModelError

# =============================================================================
# Simulation
# =============================================================================

from .BirthDeath import CBDP
from .SNPSimulator import (
    simulate as simulate_snp,
    random_network,
    SimulatedSNPData,
)

# =============================================================================
# Network analysis: decomposition, distances, structural edits
# =============================================================================
# ``GraphUtils`` declares an explicit ``__all__``, so this star import brings in
# exactly its documented helpers (including the reticulation-comparison metrics
# it re-exports from ``ReticulationComparison``).

from .GraphUtils import *
from .GraphUtils import __all__ as _graphutils_all

# =============================================================================
# Search framework
# =============================================================================
# A Model pairs a network with the callable that scores it.  Import these to
# plug a custom likelihood into the shared search machinery: attach a scorer
# with Model.set_likelihood_calculator, propose topologies with Move subclasses
# or your own ProposalKernel, and drive it with HillClimbing or
# SimulatedAnnealing.

from .ModelGraph import Model, ModelError
from .MetropolisHastings import (
    ProposalKernel,
    HillClimbing,
    SimulatedAnnealing,
)
from .State import State
from .ModelMove import (
    Move,
    AddReticulation,
    RemoveReticulation,
    FlipReticulation,
    RelocateReticulation,
    SwitchParentage,
    SPR,
    ChangeNodeHeight,
    ChangeInheritanceProb,
    ChangeReticSource,
    ChangeReticDest,
)
from .ModelSelection import reticulation_sweep, SweepResult, SweepRow
from .Logger import Logger

# =============================================================================
# Inference
# =============================================================================
# ``phynetpy.infer`` is the full, curated inference surface (the two verbs,
# result types, scorers, proposal kernels, prior containers, MCMC diagnostics,
# and simulation).  The verbs and the result type are lifted to the top level;
# the three axes stay in their own subpackages so ``Model`` and ``Criterion``
# do not collide with ``ModelGraph.Model`` and ``ModelSelection.Criterion``.

# The two verbs live in ``phynetpy.infer`` and are *not* lifted to the top
# level: ``phynetpy.infer`` has to stay the module, or ``import
# phynetpy.infer`` would resolve to a function.  One import gets you both:
#
#     from phynetpy.infer import infer, score
#
# The result and starting-phylogeny types are lifted, since a caller that
# consumes an ``InferenceResult`` should not need a second import for it.

from . import criteria, data, infer, models
from .infer import InferenceResult, Start, StartMode


__all__ = [
    "__version__",
    # Core data structures
    "Network",
    "MUL",
    "Node",
    "Edge",
    "Branch",
    "NodeSet",
    "EdgeSet",
    "NetworkError",
    "NodeError",
    "EdgeError",
    "MSA",
    "DataSequence",
    "Matrix",
    "Alphabet",
    "GeneTrees",
    # Parsing and I/O
    "get_labels",
    "NexusTemplate",
    "NewickParserError",
    "read_fasta",
    "read_fasta_records",
    "write_fasta",
    "write_fasta_from_network",
    "read_vcf",
    "read_vcf_metadata",
    "write_vcf",
    "read_newick",
    "read_newick_file",
    "write_newick",
    "write_newick_file",
    "read_nexus",
    "read_nexus_msa",
    "write_nexus",
    "convert_newick",
    "detect_newick_standard",
    "ValidationSummary",
    "ValidationError",
    "GeneTreeReport",
    "GeneTreeAggregateSummary",
    # Substitution models
    "JC",
    "K80",
    "HKY",
    "F81",
    "K81",
    "SYM",
    "TN93",
    "SubstitutionModelError",
    # Simulation
    "CBDP",
    "simulate_snp",
    "random_network",
    "SimulatedSNPData",
    # Search framework
    "Model",
    "ModelError",
    "ProposalKernel",
    "HillClimbing",
    "SimulatedAnnealing",
    "State",
    "Move",
    "AddReticulation",
    "RemoveReticulation",
    "FlipReticulation",
    "RelocateReticulation",
    "SwitchParentage",
    "SPR",
    "ChangeNodeHeight",
    "ChangeInheritanceProb",
    "ChangeReticSource",
    "ChangeReticDest",
    "reticulation_sweep",
    "SweepResult",
    "SweepRow",
    "Logger",
    # Inference: the two verbs live in ``phynetpy.infer``; the three axes
    # live in ``phynetpy.data`` / ``.models`` / ``.criteria``.
    "infer",
    "data",
    "models",
    "criteria",
    "InferenceResult",
    "Start",
    "StartMode",
    # Network analysis (from GraphUtils.__all__)
    *_graphutils_all,
]

del _graphutils_all
