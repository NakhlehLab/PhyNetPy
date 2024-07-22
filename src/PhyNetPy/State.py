""" 
Author : Mark Kessler
Last Stable Edit : 4/9/24
First Included in Version : 1.0.0
Approved for Release: No
"""

import copy

from BirthDeath import CBDP
from ModelGraph import Model
from Matrix import Matrix
from GTR import *
from ModelMove import Move



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

    def __init__(self, model : Model = None):
        if model is not None:
            self.current_model = model
            self.proposed_model = copy.deepcopy(model)
        

    def likelihood(self) -> float:
        """
        Calculates the likelihood of the current model
        
        Returns: a float that is the model likelihood for the current 
                 accepted state
        """
        return self.current_model.likelihood()

    def generate_next(self, move : Move) -> bool:
        """
        Set the proposed model to a new model that is the result of applying
        one move to the former proposed model

        Args:
            move (Move): Any object of a subclass of Move.

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
            move (Move): Any object of a subclass of Move.
        """
        move.undo(self.proposed_model)

    def commit(self, move : Move) -> None:
        """
        The proposed change was beneficial. Make the same move on the current
        model as was made to the proposed model.
        
        Args:
            move (Move): Any object of a subclass of Move.
        """
        move.same_move(self.current_model)

    def proposed(self) -> None:
        """
        Grab the proposed model.
        """
        return self.proposed_model

    def bootstrap(self, data : Matrix, submodel : GTR) -> None:
        """
        Generate an initial state by 
        1) simulating a network
        2) building a model based on that network and some input data.
        
        Currently only in use for the simplest case, a DNA/Felsenstein model.
        
        Inputs:
        data (Matrix): the nexus file data that has been preprocessed
                       by the Matrix class
        submodel (GTR): Any substitution model (can be subtype of GTR)

        """
        #TODO: Reconcile with ModelFactory concept. Models should generally 
        # be empty. Maybe a State should always require a model input
        
        # base number of leaves to be the number of groups/taxa
        network = CBDP(1, .5, data.get_num_taxa()).generate_network()  
        self.current_model = Model(network, data, submodel)
        self.proposed_model = copy.deepcopy(self.current_model)

    def write_line_to_summary(self, line: str) -> None:
        """
        Accumulate log output by appending line to the end of the
        current string.

        "line" need not be new line terminated.
        """
        self.current_model.summary_str += line + "\n"

    def validate_proposed_network(self, prev_move : Move) -> bool:
        """
        Check certain conditions on a network to check for validity (whatever
        that may mean for a given likelihood based method.)

        Args:
            prev_move (Move): the move obj that was imposed on the current model

        Returns:
            bool: True if valid, False otherwise.
        """
        return True
        # if not self.proposed_model.network.is_acyclic():
        #     self.revert(prev_move)
        #     return False
        # return True


   
   
        
        