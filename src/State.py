#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author : Mark Kessler
Last Edit : 5/12/26
First Included in Version : 1.0.0

State -- accept/reject bookkeeping for MCMC and hill-climbing search.

Wraps a :class:`~.ModelGraph.Model` with snapshot / restore semantics so
that the search driver can:

* propose a topology or parameter move and tentatively evaluate it,
* atomically commit the move on acceptance, or
* revert it on rejection or model-validation failure.

The class is intentionally lightweight -- it holds references to the
mutable model and a small set of rollback hooks; the heavy lifting (move
execution, scoring) is delegated to :mod:`.ModelMove` and the inference
module's scorer.
"""

from __future__ import annotations

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

import copy
from typing import Callable

# Relative imports
from .BirthDeath import CBDP
from .GraphUtils import _valid_network_degrees
from .ModelGraph import Model
from .Matrix import Matrix
from .GTR import GTR
from .ModelMove import Move


def network_invariants_routine(model: Model) -> bool:
    """Validate that a network satisfies phylogenetic invariants.

    Checks:
      1. Exactly one root (in-degree 0, out-degree >= 2)
      2. Every leaf has (in=1, out=0)
      3. Every reticulation has (in=2, out=1)
      4. Every other internal node has (in=1, out >= 2)
      5. The graph is a DAG (acyclic)

    Args:
        model: A Model whose ``network`` attribute is the phylogenetic network.

    Returns:
        True when all invariants hold, False otherwise.
    """
    net = model.network
    if net is None or not _valid_network_degrees(net):
        return False

    return net.is_acyclic()


class State:
    """
    Accept/reject bookkeeping for the Metropolis-Hastings and annealing
    drivers, around one live model.

    A proposal is a :class:`Move` executed in place on ``model.network``.
    Rejection undoes the move in place. Acceptance records the proposal's
    score as the accepted score and nothing else: the live network simply
    stays as it is. There is no second copy of the network, so accepting a
    move costs nothing, and scorers that cache topology by network identity
    (see ``MPLScorer``) keep their caches across acceptances.

    The one invariant this rests on: :meth:`likelihood` returns the cached
    accepted score and never re-scores the live network, because the live
    network may be mid-proposal when it is read. Callers that want a
    network to keep (a best-so-far) must snapshot it with
    ``Network.copy()``; the live one moves on.

    ``current_model`` and ``proposed_model`` are both the live model. They
    remain as names for drivers written against the older two-model design.
    """

    def __init__(self,
                 model: Model | None = None,
                 validate: Callable[[Model], bool] = network_invariants_routine) -> None:
        """
        Initialize a State around ``model``.

        Args:
            model (Model | None, optional): A Phylogenetic Model. Defaults to
                                            None, only used for if bootstrapping
                                            is to be used.
            validate (Callable[[Model], bool]): A callable function that
                                                checks for model validity. The
                                                parameter for such a function
                                                should be a Model object, and
                                                return True if the Model is
                                                valid, False if not.
                                                Defaults to
                                                'network_invariants_routine',
                                                which checks that the Model's
                                                phylogenetic network is acyclic
                                                and has well-formed root, leaf,
                                                and reticulation degrees.
        Returns:
            N/A
        """
        self.model: Model | None = model
        self._accepted_score: float | None = None
        self.validation_routine = validate

    @property
    def current_model(self) -> Model | None:
        """The live model. Between proposals it is the accepted state."""
        return self.model

    @property
    def proposed_model(self) -> Model | None:
        """The live model, which a proposal mutates in place."""
        return self.model

    def likelihood(self) -> float:
        """
        The accepted score: the likelihood of the last accepted state.

        Computed from the model once, before any proposal, and afterwards
        updated only by :meth:`commit`. Never re-scores the live network.

        Args:
            N/A
        Returns: a float that is the model likelihood for the current
                 accepted state
        """
        assert(self.model is not None)
        if self._accepted_score is None:
            self._accepted_score = self.model.likelihood()
        return self._accepted_score

    def generate_next(self, move: Move) -> bool:
        """
        Apply one move to the live model, as a proposal.

        Relies on the move's own undo() for rollback instead of deep-copying
        the entire model as a snapshot.

        Args:
            move (Move): Any instantiated subclass of Move.

        Returns:
            bool: True if the network associated with the model is valid, False
                  otherwise.
        """
        # Pin the accepted score while the network still is the accepted
        # state. A driver may ask for it only after this proposal has been
        # applied, and likelihood() must never score a mid-proposal network.
        if self._accepted_score is None:
            self._accepted_score = self.model.likelihood()
        try:
            self.model = move.execute(self.model)
            valid = self.validate_proposed_network(move)
        except Exception:
            try:
                move.undo(self.model)
            except Exception:
                pass
            return False
        if not valid:
            return False
        return True

    def revert(self, move: Move) -> None:
        """
        Undo the proposal: the live model returns to the accepted state.

        Args:
            move (Move): Any instantiated subclass of Move.
        Returns:
            N/A
        """
        move.undo(self.model)

    def commit(self, move: Move, score: float | None = None) -> None:
        """
        Accept the proposal. The live network already is the new state;
        only the accepted score changes.

        Args:
            move (Move): The accepted move (kept for the driver API).
            score (float | None): The proposal's score, as the driver
                computed it. When omitted, the model's cached likelihood
                is read, which is the same value when nothing changed the
                parameters after it was scored.
        Returns:
            N/A
        """
        assert(self.model is not None)
        self._accepted_score = (
            float(score) if score is not None else self.model.likelihood()
        )
        # A topology move records its changes in an open Network
        # transaction so it can roll back on rejection; acceptance keeps
        # them, so drop the journal. No-op when the move used none.
        if self.model.network is not None:
            self.model.network.commit_transaction()

    def proposed(self) -> Model:
        """
        Grab the live model, which carries the current proposal.

        Args:
            N/A
        Returns:
            Model: The proposed / edited Model.
        """
        assert(self.model is not None)
        return self.model

    def bootstrap(self, data: Matrix, submodel: GTR) -> None:
        """
        Generate an initial state by
        1) simulating a network
        2) building a model based on that network and some input data.

        Currently only in use for the simplest case, a DNA/Felsenstein model.

        Args:
            data (Matrix): The nexus file data that has been preprocessed
                           by the Matrix class.
            submodel (GTR): Any substitution model (can be subtype of GTR)
        Returns:
            N/A
        """
        # base number of leaves to be the number of groups/taxa
        network = CBDP(1, .5, data.get_num_taxa()).generate_network()
        self.model = Model()
        self.model.network = network
        self._accepted_score = None

    def validate_proposed_network(self, prev_move: Move) -> bool:
        """
        Check certain conditions on a network to check for validity.

        Args:
            prev_move (Move): the move obj that was imposed on the current model

        Returns:
            bool: True if valid, False otherwise.
        """
        if self.validation_routine(self.model):
            return True
        else:
            self.revert(prev_move)
            return False
