from BirthDeath import CBDP
from GTR import *
from ModelGraph import Model


class State:

    def __init__(self, model=None):
        self.current_model = model
        self.proposed_model = None
        self.temp = None

    def likelihood(self):
        return self.current_model.likelihood()

    def generate_next(self, move):
        self.proposed_model = self.proposed_model.execute_move(move)

    def revert(self, move):
        move.undo(self.proposed_model)

    def commit(self, move):
        move.same_move(self.current_model)

    def proposed(self):
        return self.proposed_model

    def bootstrap(self, data, submodel):
        network = CBDP(1, .5,
                       data.get_num_taxa()).generateTree()  # base number of leaves to be the number of groups/taxa
        self.current_model = Model(network, data, submodel)
        self.proposed_model = copy.deepcopy(self.current_model)

    def write_line_to_summary(self, line):
        self.current_model.summary_str += line + "\n"
