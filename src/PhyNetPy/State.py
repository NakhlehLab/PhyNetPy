""" 
Author : Mark Kessler
Last Stable Edit : 7/16/23
First Included in Version : 0.1.0
Approved to Release Date : N/A
"""

import copy

from BirthDeath import CBDP
from ModelGraph import Model
from Matrix import Matrix
from GTR import *
from Move import Move



class State:
    """
    Class that implements accept/reject functionality for the Metropolis-Hastings algorithm.
    There are 2 model objects, one of which is the current state, and the other is the model used to test changes and moves
    to the current accepted version.
    
    Rejection is implemented by reverting the change on the proposed model (like git revert)
    Acceptance is implemented by duplicating the move on the current model (like git merge)
    """

    def __init__(self, model=None):
        if model is not None:
            self.current_model = model
            self.proposed_model = copy.deepcopy(model)
        

    def likelihood(self) -> float:
        """
        Returns: a float that is the model likelihood for the current accepted state
        """
        return self.current_model.likelihood()

    def generate_next(self, move) -> bool:
        """
        Set the proposed model to a new model that is the result of applying one move to the former proposed model
        """
        self.proposed_model = self.proposed_model.execute_move(move)
        return self.validate_proposed_network(move)

    def revert(self, move):
        """
        Set the proposed model to the former proposed model, the move that was made was not a beneficial one.
        """
        move.undo(self.proposed_model)

    def commit(self, move):
        """
        The proposed change was beneficial. Make the same move on the current model as was made to the proposed model.
        """
        move.same_move(self.current_model)

    def proposed(self):
        """
        Grab the proposed model
        """
        return self.proposed_model

    def bootstrap(self, data: Matrix, submodel: GTR):
        """
        Generate an initial state by 
        1) simulating a network
        2) building a model based on that network and some input data.
        
        Currently only in use for the simplest case, a DNA/Felsenstein model.
        
        Inputs:
        data (Matrix): the nexus file data that has been preprocessed by the Matrix class
        submodel (GTR): Any substitution model (can be subtype of GTR)

        """
        #TODO: Reconcile with ModelFactory concept. Models should generally be empty. Maybe a State should always require a model input
        
        network = CBDP(1, .5, data.get_num_taxa()).generate_tree()  # base number of leaves to be the number of groups/taxa
        self.current_model = Model(network, data, submodel)
        self.proposed_model = copy.deepcopy(self.current_model)

    def write_line_to_summary(self, line: str):
        """
        Accumulate log output by appending line to the end of the current string.

        line need not be new line terminated.
        """
        self.current_model.summary_str += line + "\n"

    def validate_proposed_network(self, prev_move : Move) -> bool:
        
        return True
        # if not self.proposed_model.network.is_acyclic():
        #     self.revert(prev_move)
        #     return False
        # return True


   
   
        
        