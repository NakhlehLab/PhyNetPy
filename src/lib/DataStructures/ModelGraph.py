from abc import ABC, abstractmethod
import numpy as np
from Bio import AlignIO

from GTR import *
import math
import typing
import copy

from src.lib.DataStructures.Matrix import Matrix
from src.lib.DataStructures.NetworkBuilder import NetworkBuilder


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
        self.tree_heights = None
        self.felsenstein_root = None
        self.submodel_node = None
        self.build_felsenstein()

    def change_branch(self, index, value):
        current_vec = self.tree_heights.heights
        print(current_vec)
        new_vec = copy.deepcopy(current_vec)
        new_vec[index] = value
        print(new_vec)
        self.tree_heights.update(new_vec)


    def build_felsenstein(self):

        tree_heights_node = TreeHeights()
        self.tree_heights = tree_heights_node

        tree_heights_vec = []

        submodelnode = SubstitutionModel(self.sub)
        self.submodel_node = submodelnode
        self.nodes.append(submodelnode)
        submodel_params = SubstitutionModelParams(self.sub.get_hyperparams()[0], self.sub.get_hyperparams()[1])
        self.nodes.append(submodel_params)
        submodel_params.add_successor(submodelnode)
        submodelnode.add_predecessor(submodel_params)

        branch_index = 0

        # map of the TreeNode objs to network nodes
        node_modelnode_map = {}

        for node in self.network.get_nodes():
            if self.network.outDegree(node) == 0:
                # A leaf node

                # Create branch
                branch = BranchLengthNode(branch_index, node.length())
                tree_heights_vec.append(node.length())
                branch_index += 1

                # Each branch has a substitution model for calculating transition matrix
                branch.add_predecessor(submodelnode)
                branch.add_predecessor(tree_heights_node)
                tree_heights_node.add_successor(branch)
                submodelnode.add_successor(branch)

                # Calculate the leaf likelihoods
                sequence = self.data.getSeq(node.get_name())
                new_leaf_node = FelsensteinLeafNode(partials=build_matrix_from_seq(sequence), branch=branch,
                                                    name=node.get_name())

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
                tree_heights_vec.append(node.length())
                branch_index += 1

                # Link to the substitution model
                branch.add_predecessor(submodelnode)
                branch.add_predecessor(tree_heights_node)
                tree_heights_node.add_successor(branch)
                submodelnode.add_successor(branch)

                # Create internal node and link to branch
                new_internal_node = FelsensteinInternalNode(branch=branch, name=node.get_name())
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
                new_internal_node = FelsensteinInternalNode(name=node.get_name())
                self.felsenstein_root = new_internal_node

                # Add to nodes list
                self.nodes.append(new_internal_node)

                # Add to node map
                node_modelnode_map[node] = new_internal_node

        for edge in self.network.get_edges():
            # Handle network par-child relationships
            # print("HANDLING EDGE FROM " + edge[0].get_name() + " TO " + edge[1].get_name())
            # Edge is from modelnode1 to modelnode2 in network, which means
            # modelnode2 is the parent
            modelnode1 = node_modelnode_map[edge[0]]
            modelnode2 = node_modelnode_map[edge[1]]

            # Add modelnode1 as the child of modelnode2
            modelnode1.add_predecessor(modelnode2)
            modelnode2.add_successor(modelnode1)

        tree_heights_node.update(tree_heights_vec)

        # for node in self.nodes:
        #
        #     print("----------------------")
        #     print("TYPE: " + str(type(node)))
        #     if type(node) is FelsensteinLeafNode or type(node) is FelsensteinInternalNode:
        #         print("NAME OF NODE: " + node.name)
        #     print("CHILDREN: " + str(node.get_predecessors()))
        #     print("PARENTS: " + str(node.get_successors()))
        #     print("----------------------")

    def likelihood(self):
        """
        Calculates the likelihood of the model graph lazily, by only
        calculating parts of the model that have been updated/state changed.

        Inputs:
        Outputs: A numerical likelihood value, the dot product of all root vector likelihoods
        """

        # calculate the root partials or get the cached values
        partials = self.felsenstein_root.get()

        # Should be the only child of the substitution model node
        params_state = self.submodel_node.get_predecessors()[0]
        base_freqs = params_state.base_freqs.reshape((4,))

        # tally up the logs of the dot products
        result = 0
        for row in range(np.shape(partials)[0]):
            logLikelihoodSite = math.log(np.dot(base_freqs, partials[row]))
            result += logLikelihoodSite

        # TODO: Include/multiply result with the kingman coalescent result
        # right now, simply the felsensteins likelihood
        return result


class ModelNode:
    def __init__(self, successors=None, predecessors=None):
        self.successors = successors
        self.predecessors = predecessors

    def add_successor(self, model_node):
        if self.successors is None:
            self.successors = [model_node]
        else:
            self.successors.append(model_node)

    def add_predecessor(self, model_node):
        if self.predecessors is None:
            self.predecessors = [model_node]
        else:
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
        if self.predecessors is None:
            return 0
        return len(self.predecessors)

    def out_degree(self):
        if self.successors is None:
            return 0
        return len(self.successors)

    def find_root(self):
        if self.out_degree() == 0:
            return {self}
        else:
            roots = set()
            for neighbor in self.successors:
                roots.update(neighbor.find_root())  # set update

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

        If all neighbors need to be recalculated, then so must every node upstream of it, and so we may stop updating
        """
        self.switch_updated()

        neighbors = self.get_successors()
        if neighbors is None:
            return

        roots = self.find_root()
        all_updated = True
        for neighbor in neighbors:
            if not neighbor.updated:
                all_updated = False

        if not all_updated:
            for neighbor in neighbors:
                if neighbor in roots:
                    neighbor.upstream()
                    return
                neighbor.upstream()
        else:
            print("Done looking")

        # flag = True
        #
        # while flag:
        #     all_updated = True
        #     for neighbor in neighbors:
        #         if neighbor.updated is False:
        #             all_updated = False
        #         neighbor.switch_updated()
        #     if all_updated:
        #         break
        #     if a_in_b(list(roots), neighbors):
        #         flag = False

    def switch_updated(self):
        """
        A model node is updated if any of its calculation nodes downstream have been changed.

        This method will be called when a downstream node calls its upstream() method, setting this node
        as a node that needs to be recalculated.
        """
        print("THIS NODE NOW NEEDS TO BE RECALCULATED")
        if type(self) is FelsensteinInternalNode or type(self) is FelsensteinLeafNode:
            print(self.name)
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
        print(self.branch_length)
        print(new_bl)
        self.branch_length = new_bl
        print(self.branch_length)

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
                    self.sub = child.get_submodel()
                    self.updated_sub = False
                    print("calculating Pij for branch length: "+ str(self.branch_length))
                    return child.get().expt(self.branch_length)
        else:
            print("calculating Pij for branch length: " + str(self.branch_length))
            return self.sub.expt(self.branch_length)


class TreeHeights(StateNode):

    def __init__(self, node_height_vec=None):
        super().__init__()
        self.heights = node_height_vec

    def update(self, new_vector):
        if self.heights is None:
            self.heights = new_vector
            for branch_node in self.get_successors():
                branch_node.update(self.heights[branch_node.get_index()])
        else:
            for branch_node in self.get_successors():
                if new_vector[branch_node.get_index()] != self.heights[branch_node.get_index()]:
                    branch_node.update(new_vector[branch_node.get_index()])

            self.heights = new_vector

    def swap(self, index1, index2):
        pass


class FelsensteinInternalNode(CalculationNode):
    def __init__(self, branch=None, name: str = None):
        super().__init__()
        self.partials = None
        self.branch = branch
        self.name = name

    def update(self):
        self.upstream()

    def get(self):
        if self.updated:
            print("Node <" + str(self.name) + "> needs to be recalculated!")
            # print(self.get_predecessors())
            return self.calc()
        else:
            print("Node <" + str(self.name) + "> returning cached partials!")
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

    def get_branch(self):
        if self.branch is None:
            for child in self.get_predecessors():
                if type(child) is BranchLengthNode:
                    self.branch = child
                    return child
        return self.branch


class FelsensteinLeafNode(CalculationNode):

    def __init__(self, partials=None, branch=None, name: str = None):
        super().__init__()
        self.matrix = partials
        self.branch = branch
        self.name = name

    def update(self, new_partials):
        self.matrix = new_partials
        self.upstream()

    def get(self):
        if self.updated:
            print("Node <" + str(self.name) + "> needs to be recalculated!")
            return self.calc()
        else:
            print("Node <" + str(self.name) + "> returning cached partials!")
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
        # should only have a single leaf calc node as the parent
        self.successors()[0].update(build_matrix_from_seq(new_sequence))

    def get_seq(self):
        return self.seq


class SubstitutionModelParams(StateNode):

    def __init__(self, freq: np.array, trans: np.array) -> None:
        super().__init__()
        self.base_freqs = freq
        self.transitions = trans

    def update(self, new_freqs=None, new_trans=None):

        # should only have the one parent
        submodel_node = self.get_successors()[0]

        if new_freqs is None and new_trans is None:
            raise ModelGraphError("Nonsensical update")
        elif new_freqs is not None and new_trans is not None:
            submodel_node.update(self.new_submodel(new_freqs, new_trans))
        elif new_freqs is not None:
            submodel_node.update(self.new_submodel(new_freqs))
        else:
            submodel_node.update(self.new_submodel(new_trans=new_trans))

    def new_submodel(self, new_freqs: np.array = None, new_trans: np.array = None):
        """
        Given a change in transitions and/or base_frequencies, determines the proper subclass of GTR
        to return
        """
        if new_freqs is None:
            proposed_freqs = self.base_freqs
        else:
            proposed_freqs = new_freqs

        if new_trans is None:
            proposed_trans = self.transitions
        else:
            proposed_trans = new_trans

        # At least check if we can expedite the expt calculation
        if proposed_freqs == np.array([.25, .25, .25, .25]) and proposed_trans == np.ones(6):
            return JC()
        elif proposed_freqs == np.array([.25, .25, .25, .25]) \
                and (proposed_trans[1] == proposed_trans[4]) \
                and (proposed_trans[0] == proposed_trans[2] == proposed_trans[3] == proposed_trans[5]) \
                and (proposed_trans[0] + proposed_trans[1] == 1):
            return K2P(proposed_trans[0], proposed_trans[1])
        else:
            return GTR(proposed_freqs, proposed_trans)


class SubstitutionModel(CalculationNode):

    def __init__(self, submodel):
        super().__init__()
        self.sub = submodel

    def update(self, new_sub_model):
        # Set the new parameters
        self.sub = new_sub_model
        # Mark this node and any nodes upstream as needing to be recalculated
        self.upstream()

    def get(self):
        if self.updated:
            return self.calc()
        else:
            return self.cached

    def calc(self):
        self.updated = False
        self.cached = self.sub
        return self.sub

    def get_submodel(self):
        return self.sub


#### TESTS ######

n2 = NetworkBuilder(
    "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxaMultipleSites.nex")
# n3 = NetworkBuilder(
# "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxa1Site.nex")

test2 = n2.getNetwork(0)
# test3 = n3.getNetwork(0)

msa2 = AlignIO.read(
    "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxaMultipleSites.nex",
    "nexus")
# msa3 = AlignIO.read(
# "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxa1Site.nex", "nexus")

data2 = Matrix(msa2)  # default is to use the DNA alphabet
# data3 = Matrix(msa3)

model = Model(test2, data2)  # JC
# model2 = Model(test3, data3)  # JC

print(model.likelihood())
model.change_branch(2, .5)
print(model.likelihood())

# print(model2.likelihood())
