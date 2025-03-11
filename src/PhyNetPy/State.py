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
Last Edit : 3/11/25
First Included in Version : 1.0.0
Approved for Release: No
"""

import copy
from typing import Callable

from BirthDeath import CBDP
from ModelGraph import Model
from Matrix import Matrix
from GTR import *
from ModelMove import Move
from Network import Network


def acyclic_routine(model : Model) -> bool:
    """
    Checks the Model's network for cycles. 

    Args:
        model (Model): A Model with a phylogenetic network.

    Returns:
        bool: True if the model's network is free of cycles, False if it 
              contains cycles.
    """
    assert(model.network is not None)
    if model.network.is_acyclic():
        return True
    return False


class State:
    """
    Class that implements accept/reject functionality for the 
    Metropolis-Hastings algorithm. There are 2 model objects, one of which 
    is the current state, and the other is the model used to test changes 
    and moves to the current accepted version.
    
    Rejection is implemented by reverting the change on the proposed model 
    (like git revert).
    
    Acceptance is implemented by duplicating the move on the current model 
    (like git merge).
    """

    def __init__(self, 
                 model : Model | None = None,
                 validate : Callable[[Model], bool] = acyclic_routine) -> None:
        """
        Initialize a State. A State contains two models-- one current model, 
        and one proposed model that contains one singular edit to the current 
        model. At the very beginning of the State and before the execution of a 
        method, the two models will be carbon copies of each other.

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
                                                Defaults to 'acyclic_routine',
                                                which checks that the Model's 
                                                phylogenetic network is 
                                                free of cycles.
        Returns:
            N/A                                   
        """
        if model is not None:
            self.current_model = model
            self.proposed_model = copy.deepcopy(model)
        
        self.validation_routine = validate
            
    def likelihood(self) -> float:
        """
        Calculates the likelihood of the current model
        
        Args:
            N/A
        Returns: a float that is the model likelihood for the current 
                 accepted state
        """
        assert(self.current_model is not None)
        return self.current_model.likelihood()

    def generate_next(self, move : Move) -> bool:
        """
        Set the proposed model to a new model that is the result of applying
        one move to the former proposed model

        Args:
            move (Move): Any instantiated subclass of Move.

        Returns:
            bool: True if the network associated with the model is valid, False
                  otherwise.
        """
        self.proposed_model = self.proposed_model.execute_move(move)
        return self.validate_proposed_network(move)

    def revert(self, move : Move) -> None:
        """
        Set the proposed model to the former proposed model, the move that was 
        made was not a beneficial one.

        Args:
            move (Move): Any instantiated subclass of Move.
        Returns:
            N/A
        """
        move.undo(self.proposed_model)

    def commit(self, move : Move) -> None:
        """
        The proposed change was beneficial. Make the same move on the current
        model as was made to the proposed model.
        
        Args:
            move (Move): Any instantiated subclass of Move.
        Returns:
            N/A
        """
        move.same_move(self.current_model)

    def proposed(self) -> Model:
        """
        Grab the proposed model.
        
        Args:
            N/A
        Returns:
            Model: The proposed / edited Model.
        """
        assert(self.proposed_model is not None)
        return self.proposed_model

    def bootstrap(self, data : Matrix, submodel : GTR) -> None:
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
        # TODO: Reconcile with ModelFactory concept. Models should generally 
        # be empty. Maybe a State should always require a model input
        
        # base number of leaves to be the number of groups/taxa
        network = CBDP(1, .5, data.get_num_taxa()).generate_network()  
        self.current_model = Model(network, data, submodel)
        self.proposed_model = copy.deepcopy(self.current_model)

    def write_line_to_summary(self, line : str) -> None:
        """
        Accumulate log output by appending line to the end of the
        current string.

        'line' need not be new line terminated.
        
        Args:
            line (str): logging information, plain text.
        Returns:
            N/A
        """
        self.current_model.summary_str += line.strip() + "\n"

    def validate_proposed_network(self, prev_move : Move) -> bool:
        """
        Check certain conditions on a network to check for validity (whatever
        that may mean for a given likelihood based method). If not valid, then 
        the proposed model will be reverted to the current model.

        Args:
            prev_move (Move): the move obj that was imposed on the current model

        Returns:
            bool: True if valid, False otherwise.
        """
        if self.validation_routine(self.proposed_model):
            return True
        else:
            self.revert(prev_move)
            return False








