import copy

from BirthDeath import CBDP
from ModelGraph import Model


class State:

    def __init__(self, model=None):
        self.current_model = model
        self.proposed_model = None
        self.temp = None

    def likelihood(self) -> float:
        """
        Returns: a float that is the model likelihood for the current accepted state
        """
        #TODO: Delegate to correct model likelihood, set at felsenstein's
        return self.current_model.likelihood()

    def generate_next(self, move):
        """
        Set the proposed model to a new model that is the result of applying one move to the former proposed model
        """
        self.proposed_model = self.proposed_model.execute_move(move)

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

    def bootstrap(self, data, submodel):
        """
        Generate an initial state by simulating a network and building a model based on that network and some input data.

        """
        network = CBDP(1, .5, data.get_num_taxa()).generateTree()  # base number of leaves to be the number of groups/taxa
        self.current_model = Model(network, data, submodel, verbose=False)
        self.proposed_model = copy.deepcopy(self.current_model)

    def write_line_to_summary(self, line: str):
        """
        Accumulate log output by appending line to the end of the current string.

        line need not be new line terminated.
        """
        self.current_model.summary_str += line + "\n"
