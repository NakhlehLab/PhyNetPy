class State:

    def __init__(self, stuff):
        self.stuffs = stuff
        self.saved_state = None

    def propose(self):
        new_state = 0
        return new_state

    def bootstrap(self):
        initial_state = 0
        return initial_state

    def reject(self):
        return 0

    def undo(self):
        return 0


class StateInfo:

    def __init__(self, net=None, likelihood=None):
        self.network = net
        self.likelihood = likelihood
