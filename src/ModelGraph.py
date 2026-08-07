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
Last Stable Edit : 3/11/25
First Included in Version : 0.1.0

The search-state container shared by every network inference method.

A :class:`Model` pairs a network with the callable that scores it, and tracks
which parts of the network a move has invalidated so scorers can rescore
incrementally.  Attaching a scorer via
:meth:`Model.set_likelihood_calculator` is how a new method plugs its own
likelihood into the shared search machinery (:mod:`~.ModelMove`,
:mod:`~.State`, :mod:`~.MetropolisHastings`).
"""

from __future__ import annotations
from typing import Any
import numpy as np

from .Network import Network, Node


#########################
#### EXCEPTION CLASS ####
#########################

class ModelError(Exception):
    """
    Class to handle any errors related to building the model or running 
    likelihoods computations on the model.
    """

    def __init__(self, message: str = "Model is Malformed") -> None:
        """
        Create a custom ModelError with a message.

        Args:
            message (str, optional): A custom error message. Defaults to 
                                     "Model is Malformed".
        Returns:
            N/A
        """
        super().__init__(message)

#####################
#### MODEL CLASS ####
#####################

class Model:
    """
    A network under evaluation, together with the calculator that scores it.

    Search drivers mutate ``network`` through :class:`~.ModelMove.Move` objects
    and read the resulting score back through :meth:`likelihood`, which caches
    until a move marks the model dirty.
    """

    def __init__(
        self,
        rng: np.random.Generator | np.random.SeedSequence | int | None = None,
    ) -> None:
        """
        Initialize a model object.

        Args:
            rng: Optional RNG context.  Accepts a ``numpy.random.Generator``
                (used directly), a ``numpy.random.SeedSequence`` (used to
                build a fresh generator), or an ``int`` seed.  ``None``
                falls back to ``numpy.random.default_rng()``, which seeds
                itself from OS entropy.  Callers that care about
                reproducibility (e.g. ``MPL.search``) should pass a
                ``SeedSequence`` spawned from the search-wide seed.
        Returns:
            N/A
        """
        self.network : Network = None
        if isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)
        self._likelihood_calculator: Any = None
        self._dirty: bool = True
        self._cached_likelihood: float | None = None
        # Fine-grained invalidation used by scorers that support
        # incremental likelihood updates (e.g. MCMC_GT).
        #
        # Semantics:
        #   * ``None``   -> "all dirty" (safe fallback); scorers rebuild
        #     any per-network-edge caches from scratch.
        #   * ``set()``  -> "nothing dirty"; the scorer may reuse every
        #     cached partial.
        #   * non-empty ``set[Node]`` -> only these network nodes (and
        #     their ancestor path) need recomputing.
        #
        # The coarse ``self._dirty`` boolean above is preserved verbatim
        # for backward compatibility with the existing MPL scorer (which
        # rebuilds the whole engine per call).  Scorers that know how to
        # honour ``_dirty_nodes`` should consume and then clear the set.
        self._dirty_nodes: set[Node] | None = None

    def set_likelihood_calculator(self, calculator: Any) -> None:
        """
        Set a callable that computes the model's likelihood/score.
        
        The callable should accept a single argument (the Model instance)
        and return a float.

        Args:
            calculator: A callable (Model) -> float.
        Returns:
            N/A
        """
        self._likelihood_calculator = calculator

    def update_network(self) -> None:
        """
        Called after any move modifies the network topology.
        Marks the model as needing recomputation.

        Treated as a *full* invalidation: any fine-grained dirty-node
        set that a move may have posted via :meth:`mark_touched` is
        promoted to ``None`` (all-dirty) so the scorer's fallback path
        kicks in.  Moves that know exactly which nodes they touched
        should call :meth:`mark_touched` *instead of* this method (or
        before it).

        Args:
            N/A
        Returns:
            N/A
        """
        self._dirty = True
        self._dirty_nodes = None

    def mark_touched(self, nodes: "set[Node] | None") -> None:
        """Record which network nodes were modified by the last move.

        Called by :class:`Move` subclasses from ``execute``/``undo`` to
        let scorers with per-network-edge caches (e.g. MCMC_GT's
        :class:`_GTLikelihoodEngine`) invalidate only the affected
        region instead of the entire likelihood.

        Semantics:
          * ``mark_touched(None)`` -> escalate to full invalidation,
            regardless of any previously-posted set.  Safe fallback
            used by moves whose impact is hard to localize.
          * ``mark_touched(set(...))`` -> merge into any existing
            dirty set (so multiple sequential mutations accumulate).
            If the model is already "all dirty" the call is a no-op.
          * ``mark_touched(set())`` -> initialise an empty set,
            signalling "nothing dirty this iteration" (useful for
            rebinding after a full rescore).

        Args:
            nodes: Nodes whose incident branches or gammas changed,
                or ``None`` for all-dirty.
        """
        self._dirty = True
        if nodes is None:
            self._dirty_nodes = None
            return
        if self._dirty_nodes is None:
            if not nodes:
                self._dirty_nodes = set()
            return
        self._dirty_nodes.update(nodes)

    def clear_dirty_nodes(self) -> None:
        """Reset the fine-grained dirty-node set to "nothing dirty".

        Called by scorers after they have consumed ``_dirty_nodes``
        and updated their caches accordingly.  Leaves the coarse
        ``_dirty`` flag alone; the caller controls that via
        :meth:`likelihood` gating.
        """
        self._dirty_nodes = set()

    def likelihood(self) -> float:
        """
        Compute the model likelihood/score using the registered calculator.

        Args:
            N/A
        Raises:
            ModelError: If no likelihood calculator has been set.
        Returns:
            float: The model likelihood or score.
        """
        if self._likelihood_calculator is None:
            raise ModelError("No likelihood calculator set on this model")
        if self._dirty or self._cached_likelihood is None:
            self._cached_likelihood = self._likelihood_calculator(self)
            self._dirty = False
        return self._cached_likelihood
