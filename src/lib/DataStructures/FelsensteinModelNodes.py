
from ModelGraph import CalculationNode
import numpy as np


class FelsensteinLeafNode(CalculationNode):

    def __init__(self, partials=None, branch=None, name: str = None):
        super().__init__()
        self.matrix = partials
        self.branch = branch
        self.name = name
        self.parent = None

    def node_move_bounds(self):
        return [0, self.parent.get_branch().get()]

    def update(self, new_partials, new_name):
        self.matrix = new_partials
        self.name = new_name
        self.upstream()

    def get(self):
        if self.updated:
            # print("Node <" + str(self.name) + "> needs to be recalculated!")
            return self.calc()
        else:
            # print("Node <" + str(self.name) + "> returning cached partials!")
            return self.cached

    def calc(self):
        # mark node as having been recalculated and cache the result
        if self.matrix is None:
            for child in self.get_predecessors():
                if type(child) is ExtantSpecies:
                    self.matrix = build_matrix_from_seq(child.get_seq())

        self.cached = self.matrix
        self.updated = False

        # return calculation
        return self.matrix

    def get_branch(self):
        if self.branch is None:
            for child in self.get_predecessors():
                if type(child) is BranchLengthNode:
                    self.branch = child
                    return child
        return self.branch

    def get_parent(self):
        return self.parent

    def get_children(self):
        return None

    def add_successor(self, model_node):
        """
        Adds a successor to this node.

        Input: model_node (type ModelNode)
        """
        if self.successors is None:
            self.successors = [model_node]
        else:
            self.successors.append(model_node)

        if type(model_node) is FelsensteinInternalNode:
            if self.parent is None:
                self.parent = model_node

    def remove_successor(self, model_node):
        """
        Removes a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if model_node in self.successors:
            self.successors.remove(model_node)
            if self.parent is not None:
                self.parent = None

class FelsensteinInternalNode(CalculationNode):
    def __init__(self, branch=None, name: str = None):
        super().__init__()
        self.partials = None
        self.branch = branch
        self.name = name
        self.network_children = None
        self.network_parent = None

    def node_move_bounds(self):

        if self.network_parent is None:
            # root node
            return None
        # Normal internal node
        upper_limit = self.network_parent.get_branch().get()
        lower_limit = max(0, max([child.get_branch().get() for child in self.network_children]))
        return [lower_limit, upper_limit]

    def update(self):
        self.upstream()

    def get(self):
        if self.updated:
            # print("Node <" + str(self.name) + "> needs to be recalculated!")
            # print(self.get_predecessors())
            return self.calc()
        else:
            # print("Node <" + str(self.name) + "> returning cached partials!")
            return self.cached

    def calc(self):

        children = self.get_predecessors()
        # print("CHILDREN of " + self.name + ": " + str(children))
        # print(self.predecessors)
        matrices = []

        for child in children:
            # type check
            if type(child) != FelsensteinInternalNode and type(child) != FelsensteinLeafNode:
                continue

            # get the child partial likelihood. Could be another internal node, but could be a leaf
            matrix = child.get()
            # print("RETRIEVED CHILD " + child.name + " PARTIALS")
            # print("CHILD PARTIAL = " + str(matrix))

            # compute matrix * Pij transpose
            step1 = np.matmul(matrix, child.get_branch().transition().transpose())

            # add to list of child matrices
            matrices.append(step1)

            # Element-wise multiply each matrix in the list
            result = np.ones(np.shape(matrices[0]))
            for matrix in matrices:
                result = np.multiply(result, matrix)
            self.partials = result

        # mark node as having been recalculated and cache the result
        self.cached = self.partials
        self.updated = False

        # return calculation
        return self.partials

    def get_children(self):
        return self.network_children

    def get_parent(self):
        return self.network_parent

    def get_branch(self):
        if self.branch is None:
            for child in self.get_predecessors():
                if type(child) is BranchLengthNode:
                    self.branch = child
                    return child
        return self.branch

    def add_successor(self, model_node):
        """
        Adds a successor to this node.

        Input: model_node (type ModelNode)
        """
        if self.successors is None:
            self.successors = [model_node]
        else:
            self.successors.append(model_node)

        if type(model_node) is FelsensteinInternalNode:
            self.network_parent = model_node

    def add_predecessor(self, model_node):
        """
        Adds a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if self.predecessors is None:
            self.predecessors = [model_node]
        else:
            self.predecessors.append(model_node)

        if type(model_node) is FelsensteinInternalNode or type(model_node) is FelsensteinLeafNode:
            if self.network_children is None:
                self.network_children = [model_node]
            else:
                self.network_children.append(model_node)

    def remove_successor(self, model_node):
        """
        Removes a successor to this node.

        Input: model_node (type ModelNode)
        """
        if model_node in self.successors:
            self.successors.remove(model_node)

    def remove_predecessor(self, model_node):
        """
        Removes a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if model_node in self.predecessors:
            self.predecessors.remove(model_node)
            if self.network_children is not None:
                if model_node in self.network_children:
                    self.network_children.remove(model_node)