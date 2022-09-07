from abc import ABC, abstractmethod
import numpy as np
from Bio import AlignIO

from GTR import *
import math
import typing
import copy
import time

from src.lib.DataStructures.Matrix import Matrix
from src.lib.DataStructures.NetworkBuilder import NetworkBuilder


def build_matrix_from_seq(sequence):
    """
    Given a char sequence of As, Cs, Gs, and Ts,
    build a matrix of likelihoods for each site.

    Inputs: sequence, a list of chars or a string
    Output: a numpy matrix with dimensions len(seq) x 4
    """
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
        else:
            raise ModelError("Unknown sequence letter")

        # append row to matrix
        likelihoods[row_no, :] = row

    return likelihoods


class ModelError(Exception):
    """
    Class to handle any errors related to building the model or running likelihoods computations
    on the model.
    """

    def __init__(self, message="Model is Malformed"):
        super().__init__(message)


class Model:
    """
    Class that describes a DAG structure that lazily computes a model likelihood.
    """

    def __init__(self, network, data, submodel=JC()):
        self.network = network
        self.sub = submodel
        self.data = data
        self.nodes = []
        self.tree_heights = None  # type TreeHeights
        self.felsenstein_root = None
        self.submodel_node = None  # type SubstitutionModel
        self.build_felsenstein()
        self.network_node_map = {}

    def change_branch(self, index, value):
        """
        Change a branch length in the model and update any nodes upstream from the changed node

        Inputs: index - index into the heights/lengths vector
                value - new height/length to replace the old one

        """

        # Grab current vector and make a copy TODO: more efficient way than copying fs
        current_vec = self.tree_heights.heights
        new_vec = copy.deepcopy(current_vec)

        # Make new list and give it to the tree height node to update
        new_vec[index] = value
        self.tree_heights.update(new_vec)

    def build_felsenstein(self, as_length=True):
        """
        Make a felsenstein likelihood model graph.

        Complexity: O(E + V).
        """

        # Initialize branch length/height vector and save it for update usage
        tree_heights_node = TreeHeights()
        self.tree_heights = tree_heights_node
        tree_heights_vec = []
        tree_heights_adj = []

        # Initialize substitution model node
        submodelnode = SubstitutionModel(self.sub)
        self.submodel_node = submodelnode
        self.nodes.append(submodelnode)

        # Initialize substitution model parameter state node
        submodel_params = SubstitutionModelParams(self.sub.get_hyperparams()[0], self.sub.get_hyperparams()[1])
        self.nodes.append(submodel_params)

        # Join state node to its parent (the substitution model node)
        submodel_params.join(submodelnode)

        # Keep track of which branch maps to what index
        branch_index = 0

        # Add parsed phylogenetic network into the model
        for node in self.network.get_nodes():
            if self.network.outDegree(node) == 0:  # This is a leaf

                # Create branch for this leaf and add it to the height/length vector
                branch = BranchLengthNode(branch_index, node.length())
                tree_heights_vec.append(node.length())
                branch_index += 1

                # Each branch has a substitution model and a link to the vector
                tree_heights_node.join(branch)
                submodelnode.join(branch)

                # Calculate the leaf likelihoods
                sequence = self.data.getSeq(node.get_name())  # Get char sequence from the matrix data
                new_leaf_node = FelsensteinLeafNode(partials=build_matrix_from_seq(sequence), branch=branch,
                                                    name=node.get_name())

                # Point the branch length node to the leaf node
                branch.join(new_leaf_node)

                # Add to list of model nodes
                self.nodes.append(new_leaf_node)
                self.nodes.append(branch)

                # Add to map
                self.network_node_map[node] = new_leaf_node

            elif self.network.inDegree(node) != 0:  # An internal node that is not the root

                # Create branch
                branch = BranchLengthNode(branch_index, node.length())
                tree_heights_vec.append(node.length())
                branch_index += 1

                # Link to the substitution model
                tree_heights_node.join(branch)
                submodelnode.join(branch)

                # Create internal node and link to branch
                new_internal_node = FelsensteinInternalNode(branch=branch, name=node.get_name())
                branch.join(new_internal_node)

                # Add to nodes list
                self.nodes.append(new_internal_node)
                self.nodes.append(branch)

                # Map node to the new internal node
                self.network_node_map[node] = new_internal_node
            else:  # The root. TODO: Add dependency on the base frequencies

                # Create root
                new_internal_node = FelsensteinInternalNode(name=node.get_name())
                self.felsenstein_root = new_internal_node

                # Add to nodes list
                self.nodes.append(new_internal_node)

                # Add to node map
                self.network_node_map[node] = new_internal_node

        if as_length is False:
            tree_heights_adj = np.array(len(tree_heights_vec))

        for edge in self.network.get_edges():
            # Handle network par-child relationships
            # Edge is from modelnode1 to modelnode2 in network, which means
            # modelnode2 is the parent
            modelnode1 = self.network_node_map[edge[0]]
            modelnode2 = self.network_node_map[edge[1]]

            if as_length is False:
                # Convert from node heights to branch lengths by subtracting the parent node height from the child node height
                branch1 = modelnode1.get_branch()
                branch2 = modelnode2.get_branch()
                tree_heights_adj[branch2.get_index()] = tree_heights_vec[branch2.get_index()] - tree_heights_vec[branch1.get_index()]

            # Add modelnode1 as the child of modelnode2
            modelnode2.join(modelnode1)

        # all the branches have been added, set the vector for the TreeHeight nodes
        if as_length is False:
            # Use the branch length adjusted version
            tree_heights_node.update(list(tree_heights_adj))
        else:
            # Passed in as branch lengths, no manipulation needed
            tree_heights_node.update(tree_heights_vec)

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

    def execute_move(self, move):
        """
        The operator move has asked for permission to work on this model.
        Pass the move this model and get a new, separate model that is the result of the operation on this model

        Input: move, a Move obj or any subtype
        Output: a new Model obj that is the result of doing Move on this Model obj
        """
        return move.execute(self)

    def get_tree_heights(self):
        return self.tree_heights

class ModelNode:
    """
    Class that defines the graphical structure and shared interactions between
    any node in the Model.
    """

    def __init__(self, successors=None, predecessors=None):
        self.successors = successors
        self.predecessors = predecessors

    def add_successor(self, model_node):
        """
        Adds a successor to this node.

        Input: model_node (type ModelNode)
        """
        if self.successors is None:
            self.successors = [model_node]
        else:
            self.successors.append(model_node)

    def add_predecessor(self, model_node):
        """
        Adds a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if self.predecessors is None:
            self.predecessors = [model_node]
        else:
            self.predecessors.append(model_node)

    def join(self, other_node):
        """
        Adds other_node as a parent, and adds this node as
        a child of other_node

        Input: other_node (type ModelNode)
        """
        self.add_successor(other_node)
        other_node.add_predecessor(self)

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

    def get_predecessors(self):
        """
        Returns: the list of child nodes to this node
        """
        return self.predecessors

    def get_successors(self):
        """
        Returns: the list of parent nodes to this node
        """
        return self.successors

    def in_degree(self):
        """
        Calculates the in degree of the current node (ie number of children)

        If 0, this node is a leaf
        """
        if self.predecessors is None:
            return 0
        return len(self.predecessors)

    def out_degree(self):
        """
        Calculates the out degree of the current node (ie number of parents

        If 0, this node is a root of the Model
        """
        if self.successors is None:
            return 0
        return len(self.successors)

    def find_root(self):
        """
        TODO: PLS MAKE MORE EFFICIENT THIS IS DUMB

        """
        if self.out_degree() == 0:
            return {self}
        else:
            roots = set()
            for neighbor in self.successors:
                roots.update(neighbor.find_root())  # set update

            return roots


class CalculationNode(ABC, ModelNode):
    """
    Subclass of a ModelNode that calculates a portion of the model likelihood.
    """

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
        # First update self
        self.switch_updated()

        # Get parent nodes and check that this node is not the root (in which case we're done
        neighbors = self.get_successors()
        if neighbors is None:
            return

        roots = self.find_root()

        # If all parent nodes are marked to be recalculated, then so must be each path from this node to the root,
        # so no further steps are required
        all_updated = True
        for neighbor in neighbors:
            if not neighbor.updated:
                all_updated = False

        # Otherwise, call upstream on each neighbor
        if not all_updated:
            for neighbor in neighbors:
                if neighbor in roots:
                    neighbor.upstream()
                    return
                neighbor.upstream()

    def switch_updated(self):
        """
        A model node is updated if any of its calculation nodes downstream have been changed.

        This method will be called when a downstream node calls its upstream() method, setting this node
        as a node that needs to be recalculated.
        """

        self.updated = True


class StateNode(ABC, ModelNode):
    """
    Model leaf nodes that hold some sort of data that calculation nodes use
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self, *args, **kwargs):
        pass


class BranchLengthNode(CalculationNode):
    """
    A calculation node that uses the substitution model to calculate the
    transition matrix Pij
    """

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
        """
        Calculate the Pij matrix
        """
        if self.updated_sub:
            # grab current substitution model
            for child in self.get_predecessors():
                if type(child) is SubstitutionModel:
                    self.sub = child.get_submodel()
                    self.updated_sub = False
                    print("calculating Pij for branch length: " + str(self.branch_length))
                    return child.get().expt(self.branch_length)
        else:
            print("calculating Pij for branch length: " + str(self.branch_length))
            # TODO: cache this?
            return self.sub.expt(self.branch_length)


class TreeHeights(StateNode):
    """
    State node that holds the node heights/branch lengths
    """

    def __init__(self, node_height_vec=None):
        super().__init__()
        self.heights = node_height_vec

    def update(self, new_vector):

        # Only update the parts of the vector that have changed
        if self.heights is None:
            self.heights = new_vector
            for branch_node in self.get_successors():
                branch_node.update(self.heights[branch_node.get_index()])
        else:
            for branch_node in self.get_successors():
                if new_vector[branch_node.get_index()] != self.heights[branch_node.get_index()]:
                    branch_node.update(new_vector[branch_node.get_index()])

            self.heights = new_vector

    def get_heights(self):
        return self.heights


class FelsensteinInternalNode(CalculationNode):
    def __init__(self, branch=None, name: str = None):
        super().__init__()
        self.partials = None
        self.branch = branch
        self.name = name
        self.network_children = None

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

    def get_children(self):
        if self.network_children is None:
            children = self.get_predecessors()
            network_children = []
            for child in children:
                if type(child) is FelsensteinInternalNode or type(child) is FelsensteinLeafNode:
                    network_children.append(child)

            self.network_children = network_children

        return self.network_children

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
            raise ModelError("Nonsensical update")
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

startFirst = time.perf_counter()
model.likelihood()
endFirst = time.perf_counter()

model.change_branch(2, .5)
startSecond = time.perf_counter()
model.likelihood()
endSecond = time.perf_counter()

print("WHOLE GRAPH: " + str(endFirst - startFirst))
print("RECALC GRAPH: " + str(endSecond - startSecond))

# print(model2.likelihood())
