from Graph import ModelGraph
from abc import ABC, abstractmethod


def a_in_b(a, b):
    for elem in a:
        if elem in b:
            return True
    return False


class Model:

    def __init__(self):
        self.graph = ModelGraph()

    def build(self):
        return 0

    def likelihood(self):
        """
        Calculates the likelihood of the model graph lazily, by only
        calculating parts of the model that have been updated/state changed.

        Inputs:
        Outputs: A numerical likelihood value, the dot product of all root vector likelihoods
        """
        return 0


class CalculationNode(ABC):

    def __init__(self):
        self.updated = True  # on initialization, we should do the calculation
        self.cached = None


    @abstractmethod
    def update(self, *args, **kwargs):
        """
        This method should be implemented in each CalculationNode subclass.
        Updating internal data should be handled on an individual basis.

        When the model graph runs its calculate routine, this update method will have marked
        this calculation node and any calculation nodes upstream as needing recalculation.
        """
        pass

    @abstractmethod
    def get(self):
        """
        Either retrieves the cached calculation or redoes the calculation for this node
        This is an abstract method, due to the fact that the type of recalculation will vary.

        Returns: a vector of partial likelihoods
        """
        pass

    @abstractmethod
    def calc(self, *args, **kwargs):
        """
        This method should be implemented in each CalculationNode subclass.
        Doing a calculation should be a unique operation depending on the type of node.

        Returns: A vector of partial likelihoods.
        """
        pass

    def upstream(self, g):
        """
        Finds a path within the model graph from this node to the root, and marks each node along the way as updated
        using the switch_updated() method
        """
        self.switch_updated()

        neighbors = g.findDirectSuccessors()
        roots = g.findRoot()
        flag = True

        while flag:
            for neighbor in neighbors:
                neighbor.switch_updated()

            if a_in_b(roots, neighbors):
                flag = False

    def switch_updated(self):
        """
        A model node is updated if any of its calculation nodes downstream have been changed.

        This method will be called when a downstream node calls its upstream() method, setting this node
        as a node that needs to be recalculated.
        """
        self.updated = True


class StateNode(ABC):

    def __init__(self, data=None):
        self.data = data

    @abstractmethod
    def update(self, *args, **kwargs):
        pass


class BranchLengthNode(CalculationNode):

    def __init__(self, vector_index, branch_length):
        super().__init__()
        self.index = vector_index
        self.branch_length = branch_length

    def update(self, new_data, g):
        # update the branch length
        self.branch_length = new_data

        # Mark this node and any nodes upstream as needing to be recalculated
        self.upstream(g)

    def get(self):
        if self.updated:
            return self.calc()
        else:
            return self.cached

    def calc(self):
        pass

    def switch_index(self, new_index):
        self.index = new_index

    def get_index(self):
        return self.index


class TreeHeights(StateNode):

    def __init__(self, node_height_vec):
        self.heights = node_height_vec

    def update(self, new_vector, g):
        self.heights = new_vector

        for branch_node in g.findDirectSuccessors(self):
            branch_node.update(self.heights[branch_node.get_index()], g)


class TreeNode(CalculationNode):
    def __init__(self):
        super().__init__()

    def update(self, g):
        self.upstream(g)

    def get(self, g):
        if self.updated:
            return self.calc(g)
        else:
            return self.cached

    def calc(self, g):
        



class ExtantSpecies(StateNode):

    def update(self, new_data):
        pass


class SubstitutionModelParams(StateNode):

    def update(self, new_data):
        pass


class SubstitutionModel(CalculationNode):

    def update(self, new_data):
        # Set the new parameters

        # Mark this node and any nodes upstream as needing to be recalculated
        self.upstream()

    def get(self):
        if self.updated:
            return self.calc()
        else:
            return self.cached

    def calc(self):
        pass
