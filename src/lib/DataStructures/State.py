from BirthDeath import Yule, CBDP
from GTR import *
from ModelGraph import Model

class State:

    def __init__(self, model=None):
        self.current_model = model
        self.cached_model = None

    def init_model(self):
        """
        Simulate a tree/network to be the starting state

        """
        return

    def likelihood(self):
        return self.current_model.likelihood()

    def generate_next(self, move):
        self.cached_model = self.current_model
        self.current_model = self.current_model.execute_move(move)

    def revert(self):
        self.current_model = self.cached_model
        self.cached_model = None

    def commit(self):
        self.cached_model = None

    def cached(self):
        return self.cached_model

    def bootstrap(self, data, submodel):
        network = CBDP(1, .5, 10).generateTree()

        print("SIM BOOTSTRAP")
        network.printGraph()
        self.current_model = Model(network, data, submodel)



