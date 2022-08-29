from abc import ABC, abstractmethod
import numpy as np
from GTR import *



def a_in_b(a, b):
    for elem in a:
        if elem in b:
            return True
    return False


def build_matrix_from_seq(sequence):
    likelihoods = np.zeros((len(sequence), 4))

    # Map each character in the sequence to an array of length 4
    for row_no in range(len(sequence)):
        if sequence[row_no] == "A":
            row = np.array([1, 0, 0, 0])
        elif sequence[row_no] == "C":
            row = np.array([0, 1, 0, 0])
        elif sequence[row_no] == "G":
            row = np.array([0, 0, 1, 0])
        elif sequence[row_no] == "T":
            row = np.array([0, 0, 0, 1])

        # append row to matrix
        likelihoods[row_no, :] = row
    return likelihoods


class ModelGraphError(Exception):
    def __init__(self, message="Model Graph is Malformed"):
        super().__init__(message)


class Model:

    def __init__(self, network, data, submodel=JC()):
        self.network = network
        self.sub = submodel
        self.data = data
        self.nodes = []
        self.build_felsenstein()

    def build_felsenstein(self):

        submodelnode = SubstitutionModel(self.sub)
        self.nodes.append(submodelnode)

        branch_index = 0

        # map of the TreeNode objs to network nodes
        node_modelnode_map = {}

        for node in self.network.get_nodes():
            if self.network.outDegree(node) == 0:
                # A leaf node

                # Create branch
                branch = BranchLengthNode(branch_index, node.length())

                # Each branch has a substitution model for calculating transition matrix
                branch.add_predecessor(submodelnode)
                submodelnode.add_successor(branch)

                # Calculate the leaf likelihoods
                sequence = self.data.getSeq(node.get_name())
                new_leaf_node = FelsensteinLeafNode(partials=build_matrix_from_seq(sequence), branch=branch)

                # Point the branch length node to the leaf node
                branch.add_successor(new_leaf_node)
                new_leaf_node.add_predecessor(branch)

                # Add to list of model nodes
                self.nodes.append(new_leaf_node)
                self.nodes.append(branch)

                # Add to map
                node_modelnode_map[node] = new_leaf_node

            elif self.network.inDegree(node) != 0:
                # An internal node that is not the root

                # Create branch
                branch = BranchLengthNode(branch_index, node.length())

                # Link to the substitution model
                branch.add_predecessor(submodelnode)
                submodelnode.add_successor(branch)

                # Create internal node and link to branch
                new_internal_node = FelsensteinInternalNode(branch=branch)
                branch.add_successor(new_internal_node)
                new_internal_node.add_predecessor(branch)

                # Add to nodes list
                self.nodes.append(new_internal_node)
                self.nodes.append(branch)

                # Map node to the new internal node
                node_modelnode_map[node] = new_internal_node
            else:
                # The root. TODO: Add dependency on the base frequencies
                # The root doesn't have a branch.

                # Create root
                new_internal_node = FelsensteinInternalNode()

                # Add to nodes list
                self.nodes.append(new_internal_node)

                # Add to node map
                node_modelnode_map[node] = new_internal_node

            # inc branch length vector index
            branch_index += 1

        for edge in self.network.get_edges():
            # Handle network par-child relationships

            # Edge is from modelnode1 to modelnode2 in network, which means
            # modelnode2 is the parent
            modelnode1 = node_modelnode_map[edge[0]]
            modelnode2 = node_modelnode_map[edge[1]]

            # Add modelnode1 as the child of modelnode2
            modelnode1.add_predecessor(modelnode2)
            modelnode2.add_successor(modelnode1)

    def likelihood(self):
        """
        Calculates the likelihood of the model graph lazily, by only
        calculating parts of the model that have been updated/state changed.

        Inputs:
        Outputs: A numerical likelihood value, the dot product of all root vector likelihoods
        """
        return 0


class ModelNode:
    def __init__(self, successors=None, predecessors=None):
        self.successors = successors
        self.predecessors = predecessors

    def add_successor(self, model_node):
        self.successors.append(model_node)

    def add_predecessor(self, model_node):
        self.predecessors.append(model_node)

    def remove_successor(self, model_node):
        if model_node in self.successors:
            self.successors.remove(model_node)

    def remove_predecessor(self, model_node):
        if model_node in self.predecessors:
            self.predecessors.remove(model_node)

    def get_predecessors(self):
        return self.predecessors

    def get_successors(self):
        return self.successors

    def in_degree(self):
        return len(self.predecessors)

    def out_degree(self):
        return len(self.successors)

    def find_root(self):
        if self.out_degree() == 0:
            return self
        else:
            roots = set()
            for neighbor in self.successors():
                roots.update(neighbor.find_root())

            return roots


class CalculationNode(ABC, ModelNode):

    def __init__(self):
        super().__init__()
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

    def upstream(self):
        """
        Finds a path within the model graph from this node to the root, and marks each node along the way as updated
        using the switch_updated() method
        """
        self.switch_updated()

        neighbors = self.successors()
        roots = self.find_root()
        flag = True

        while flag:
            for neighbor in neighbors:
                neighbor.switch_updated()

            if a_in_b(list(roots), neighbors):
                flag = False

    def switch_updated(self):
        """
        A model node is updated if any of its calculation nodes downstream have been changed.

        This method will be called when a downstream node calls its upstream() method, setting this node
        as a node that needs to be recalculated.
        """
        self.updated = True


class StateNode(ABC, ModelNode):

    def __init__(self, data=None):
        super().__init__()
        self.data = data

    @abstractmethod
    def update(self, *args, **kwargs):
        pass


class BranchLengthNode(CalculationNode):

    def __init__(self, vector_index, branch_length):
        super().__init__()
        self.index = vector_index
        self.branch_length = branch_length
        self.sub = None
        self.updated_sub = True

    def update(self, new_bl):
        # update the branch length
        self.branch_length = new_bl

        # Mark this node and any nodes upstream as needing to be recalculated
        self.upstream()

    def update_sub(self, new_sub):
        self.sub = new_sub
        self.updated_sub = False
        self.upstream()

    def get(self):
        if self.updated:
            return self.calc()
        else:
            return self.cached

    def calc(self):
        # mark node as having been recalculated and cache the result
        self.cached = self.branch_length
        self.updated = False

        # return calculation
        return self.branch_length

    def switch_index(self, new_index):
        self.index = new_index

    def get_index(self):
        return self.index

    def transition(self):
        if self.updated_sub:
            for child in self.get_predecessors():
                if type(child) is SubstitutionModel:
                    self.sub = child
                    self.updated_sub = False
                    return child.expt(self.branch_length)
        else:
            return self.sub.expt(self.branch_length)


class TreeHeights(StateNode):

    def __init__(self, node_height_vec):
        super().__init__()
        self.heights = node_height_vec

    def update(self, new_vector):
        self.heights = new_vector

        ## NOT OPTIMAL, ONLY UPDATE VALUES THAT CHANGED...
        for branch_node in self.get_successors():
            branch_node.update(self.heights[branch_node.get_index()])

    def swap(self, index1, index2):
        pass


class FelsensteinInternalNode(CalculationNode):
    def __init__(self, branch=None):
        super().__init__()
        self.partials = None
        self.branch = branch

    def update(self):
        self.upstream()

    def get(self):
        if self.updated:
            return self.calc()
        else:
            return self.cached

    def calc(self):

        children = self.get_predecessors()
        matrices = []

        for child in children:
            # type check
            if type(child) != FelsensteinInternalNode or type(child) != FelsensteinLeafNode:
                continue

            # get the child partial likelihood. Could be another internal node, but could be a leaf
            matrix = child.get()

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
        self.cached = result
        self.updated = False

        # return calculation
        return self.partials

    def get_branch(self):
        if self.branch is None:
            for child in self.get_predecessors():
                if type(child) is BranchLengthNode:
                    self.branch = child
                    return child
        return self.branch


class FelsensteinLeafNode(CalculationNode):

    def __init__(self, partials=None, branch=None):
        super().__init__()
        self.matrix = partials
        self.branch = branch

    def update(self, new_partials):
        self.matrix = new_partials
        self.upstream()

    def get(self):
        if self.updated:
            return self.calc()
        else:
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


class ExtantSpecies(StateNode):

    def __init__(self, name, sequence):
        super().__init__()
        self.name = name
        self.seq = sequence

    def update(self, new_sequence):
        self.successors()[0].update(build_matrix_from_seq(new_sequence))

    def get_seq(self):
        return self.seq


class SubstitutionModelParams(StateNode):

    def update(self, new_data):
        pass


class SubstitutionModel(CalculationNode):

    def __init__(self, submodel):
        super().__init__()
        self.sub = submodel

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
