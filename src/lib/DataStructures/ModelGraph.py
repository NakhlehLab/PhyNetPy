from abc import ABC, abstractmethod
from math import comb, pow
from GTR import *
import math
import copy

from src.lib.DataStructures.Alphabet import Alphabet
from src.lib.DataStructures.Graph import DAG
from src.lib.DataStructures.MSA import MSA
from src.lib.DataStructures.Matrix import Matrix


def vec_bin_array(arr, m):
    """
    Arguments:
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain (in the case of DNA, 4)

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's, read from left to right. [1, 0, 1, 1] is 13.
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[..., m - bit_ix - 1] = fetch_bit_func(strs).astype("int8")

    return ret


def convert_to_heights(node, adj_dict):
    """
    This is a recusive function that is used to take a model that is initialized
    with branch heights and turn it into a model based on node heights.

    Usage: convert_to_heights(root_node, {})

    The resulting heights are conditioned such that t=0 is at the root. Need to subtract dictionary value from
    max(heights of all leaves) to get heights such that the root is the time furthest in the past

    input: a ModelNode, to change the whole graph use the root.
    output: a dictionary that maps each model node to a float height value

    """
    if node.get_parent() is None:  # Root
        adj_dict[node] = 0  # Start root at t=0
    else:
        # For all other nodes, the height will be the branch length plus the node height of its parent
        adj_dict[node] = node.get_branch().get() + adj_dict[node.get_parent()]

    # Done at the leaves
    if type(node) is FelsensteinLeafNode:
        return adj_dict

    # Otherwise, recursively call on the children of this node
    if node.get_children() is not None:
        for child in node.get_children():
            # combine maps of all children
            adj_dict.update(convert_to_heights(child, adj_dict))

    # Return the built-up mapping
    return adj_dict


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
        self.netnodes_sans_root = []
        self.network_leaves = []
        self.tree_heights = None  # type TreeHeights
        self.felsenstein_root = None
        self.submodel_node = None  # type SubstitutionModel
        self.network_node_map = {}
        self.as_length = False
        if self.data.get_type() == "DNA":
            self.build_felsenstein()
        elif self.data.get_type() == "SNP":
            self.build_SNP(False, as_length=self.as_length)
        elif self.data.get_type() == "BINARY":
            self.build_SNP(True, as_length=self.as_length)
        self.internal = [item for item in self.netnodes_sans_root if item not in self.network_leaves]
        self.summary_str = ""

    def change_branch(self, index, value):
        """
        Change a branch length in the model and update any nodes upstream from the changed node

        Inputs: index - index into the heights/lengths vector
                value - new height/length to replace the old one

        """

        # Grab current vector and make a copy TODO: more efficient way than copying fs
        # current_vec = self.tree_heights.heights
        # new_vec = copy.deepcopy(current_vec)
        #
        # # Make new list and give it to the tree height node to update
        # new_vec[index] = value
        # # new_vec = self.tree_heights.heights
        # # new_vec[index] = value
        self.tree_heights.singular_update(index, value)

    def build_felsenstein(self):
        """
        Make a felsenstein likelihood model graph.

        Complexity: O(E + V).
        """

        # Initialize branch length/height vector and save it for update usage
        tree_heights_node = TreeHeights()
        self.tree_heights = tree_heights_node
        tree_heights_vec = []

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
            if self.network.outDegree(node, debug=False) == 0:  # This is a leaf

                # Create branch for this leaf and add it to the height/length vector
                branch = BranchLengthNode(branch_index, node.length(), heights= not self.as_length)
                tree_heights_vec.append(node.length())
                branch_index += 1

                # Each branch has a substitution model and a link to the vector
                tree_heights_node.join(branch)
                submodelnode.join(branch)

                # Calculate the leaf likelihoods
                sequence = self.data.get_number_seq(node.get_name())  # Get sequence from the matrix data
                new_leaf_node = FelsensteinLeafNode(partials=vec_bin_array(sequence, 4), branch=branch,
                                                    name=node.get_name())
                new_ext_species = ExtantSpecies(node.get_name(), sequence)

                new_ext_species.join(new_leaf_node)
                self.nodes.append(new_ext_species)
                self.network_leaves.append(new_leaf_node)

                # Point the branch length node to the leaf node
                branch.join(new_leaf_node)

                # Add to list of model nodes
                self.nodes.append(new_leaf_node)
                self.nodes.append(branch)
                self.netnodes_sans_root.append(new_leaf_node)

                # Add to map
                self.network_node_map[node] = new_leaf_node

            elif self.network.inDegree(node) != 0:  # An internal node that is not the root

                # Create branch
                branch = BranchLengthNode(branch_index, node.length(), heights= not self.as_length)
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
                self.netnodes_sans_root.append(new_internal_node)

                # Map node to the new internal node
                self.network_node_map[node] = new_internal_node
            else:  # The root. TODO: Add dependency on the base frequencies

                # Create root
                new_internal_node = FelsensteinInternalNode(name=node.get_name())
                self.felsenstein_root = new_internal_node

                if not self.as_length:
                    branch_height = BranchLengthNode(branch_index, 0, heights= not self.as_length)
                    branch_index += 1
                    tree_heights_vec.append(0)
                    branch_height.join(new_internal_node)
                    submodelnode.join(branch_height)
                    tree_heights_node.join(branch_height)

                # Add to nodes list
                self.nodes.append(new_internal_node)

                # Add to node map
                self.network_node_map[node] = new_internal_node

        for edge in self.network.get_edges():
            # Handle network par-child relationships
            # Edge is from modelnode1 to modelnode2 in network, which means
            # modelnode2 is the parent
            modelnode1 = self.network_node_map[edge[0]]
            modelnode2 = self.network_node_map[edge[1]]

            # Add modelnode1 as the child of modelnode2
            modelnode2.join(modelnode1)

        # all the branches have been added, set the vector for the TreeHeight nodes
        if self.as_length is False:
            # Use the branch length adjusted version
            tree_heights_adj = np.zeros(len(tree_heights_vec))
            adj_dict = convert_to_heights(self.felsenstein_root, {})

            # Keep track of the maximum leaf height, this is used to switch the node heights from root centric to leaf centric
            max_height = 0

            # Set each node height
            for node, height in adj_dict.items():
                tree_heights_adj[node.get_branch().get_index()] = height
                if height > max_height:
                    max_height = height

            # Subtract dict height from max child height
            tree_heights_adj = np.ones(len(tree_heights_adj)) * max_height - tree_heights_adj

            # Update all the branch length nodes to be the proper calculated heights
            tree_heights_node.update(list(tree_heights_adj))
        else:
            # Passed in as branch lengths, no manipulation needed
            tree_heights_node.update(tree_heights_vec)

    def build_SNP(self, phased, as_length=True):
        """
        Make a model graph for SNP likelihoods

        """

        # Initialize branch length/height vector and save it for update usage
        tree_heights_node = TreeHeights()
        self.tree_heights = tree_heights_node
        tree_heights_vec = []

        partials = SNP_compute_partials(self.data, phased=phased)

        # Keep track of which branch maps to what index
        branch_index = 0

        # Add parsed phylogenetic network into the model
        for node in self.network.get_nodes():
            if self.network.outDegree(node, debug=False) == 0:  # This is a leaf

                # Create branch for this leaf and add it to the height/length vector
                branch = BranchLengthNode(branch_index, node.length(), heights=not self.as_length)
                tree_heights_vec.append(node.length())
                branch_index += 1

                # Each branch has a link to the vector
                tree_heights_node.join(branch)

                # Calculate the leaf likelihoods
                sequence = self.data.get_number_seq(node.get_name())  # Get sequence from the matrix data
                new_leaf_node = SNPLeafNode(partials=partials[node.get_name()], branch=branch,
                                            name=node.get_name())
                new_ext_species = ExtantSpecies(node.get_name(), sequence)

                new_ext_species.join(new_leaf_node)
                self.nodes.append(new_ext_species)
                self.network_leaves.append(new_leaf_node)

                # Point the branch length node to the leaf node
                branch.join(new_leaf_node)

                # Add to list of model nodes
                self.nodes.append(new_leaf_node)
                self.nodes.append(branch)
                self.netnodes_sans_root.append(new_leaf_node)

                # Add to map
                self.network_node_map[node] = new_leaf_node

            elif self.network.inDegree(node) != 0:  # An internal node that is not the root

                # Create branch
                branch = BranchLengthNode(branch_index, node.length(), heights=not self.as_length)
                tree_heights_vec.append(node.length())
                branch_index += 1

                # Link to the substitution model
                tree_heights_node.join(branch)

                # Create internal node and link to branch
                new_internal_node = SNPInternalNode(branch=branch, name=node.get_name())
                branch.join(new_internal_node)

                # Add to nodes list
                self.nodes.append(new_internal_node)
                self.nodes.append(branch)
                self.netnodes_sans_root.append(new_internal_node)

                # Map node to the new internal node
                self.network_node_map[node] = new_internal_node
            else:  # The root.
                # Create root
                new_internal_node = SNPInternalNode(name=node.get_name())
                self.felsenstein_root = new_internal_node

                if not as_length:
                    branch_height = BranchLengthNode(branch_index, 0, heights=self.as_length)
                    branch_index += 1
                    tree_heights_vec.append(0)
                    branch_height.join(new_internal_node)
                    tree_heights_node.join(branch_height)

                # Add to nodes list
                self.nodes.append(new_internal_node)

                # Add to node map
                self.network_node_map[node] = new_internal_node

        for edge in self.network.get_edges():
            # Handle network par-child relationships
            # Edge is from modelnode1 to modelnode2 in network, which means
            # modelnode2 is the parent
            modelnode1 = self.network_node_map[edge[0]]
            modelnode2 = self.network_node_map[edge[1]]

            # Add modelnode1 as the child of modelnode2
            modelnode2.join(modelnode1)

        # all the branches have been added, set the vector for the TreeHeight nodes
        if as_length is False:
            # Use the branch length adjusted version
            tree_heights_adj = np.zeros(len(tree_heights_vec))
            adj_dict = convert_to_heights(self.felsenstein_root, {})

            # Keep track of the maximum leaf height, this is used to switch the node heights from root centric to leaf centric
            max_height = 0

            # Set each node height
            for node, height in adj_dict.items():
                tree_heights_adj[node.get_branch().get_index()] = height
                if height > max_height:
                    max_height = height

            # Subtract dict height from max child height
            tree_heights_adj = np.ones(len(tree_heights_adj)) * max_height - tree_heights_adj

            # Update all the branch length nodes to be the proper calculated heights
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
        return np.sum(np.log(np.matmul(partials, base_freqs)))

    def execute_move(self, move):
        """
        The operator move has asked for permission to work on this model.
        Pass the move this model and get the model that is the result of the operation on this model. IT IS THE SAME OBJ

        Input: move, a Move obj or any subtype
        Output: the !same! obj that is the result of doing Move on this Model obj
        """
        return move.execute(self)

    def summary(self, tree_filename, summary_filename):
        """
        Writes summary of calculations to a file, and gets the current state of the model
        and creates a network obj so that the newick format can be output.

        Inputs:
        1) tree_filename : a string that is the name of the file to output a newick string to.
                           if the filename does not exist, a new file will be created in the directory in which
                           one is operating in.

        2) summary_filename : a string that is the name of the file to output logging information.
                              if the filename does not exist, a new file will be created in the current directory

        TODO: TEST FILE CREATION

        """
        # Step 1: create network obj
        net = DAG()

        network_nodes = []
        network_nodes.extend([self.felsenstein_root])
        network_nodes.extend(self.network_leaves)
        network_nodes.extend(self.netnodes_sans_root)

        inv_map = {v: k for k, v in self.network_node_map.items()}
        net.addNodes([inv_map[node] for node in network_nodes])

        for node in network_nodes:

            # Change branch length to the branch length value from the branch node attached to node
            if self.as_length:
                try:
                    parent_height = node.get_parent().get_branch().get()
                    branch_len = parent_height - node.get_branch.get()
                finally:
                    pass
            else:
                branch_len = node.get_branch().get()

            inv_map[node].set_length(branch_len)

            # Add edges
            if node.get_children() is not None:
                for child in node.get_children():
                    net.addEdges((inv_map[node], inv_map[child]))  # switch order?

        net.printGraph()
        newick_str = net.newickString()

        # Write newick string to output file
        text_file = open(tree_filename, "w")
        n = text_file.write(newick_str)
        text_file.close()

        # Step 2: write iter summary to a file
        text_file2 = open(summary_filename, "w")
        n = text_file2.write(self.summary_str)
        text_file2.close()

    def get_tree_heights(self):
        return self.tree_heights

    def get_network_leaves(self):
        return self.network_leaves


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

    def unjoin(self, other_node):
        self.remove_successor(other_node)
        other_node.remove_predecessor(self)

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
        Calculates the out degree of the current node (ie number of parents)

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
        super(CalculationNode, self).__init__()
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


class NetworkNode(ABC, ModelNode):
    """
    Class that handles common functionality of all network nodes
    and all the height/branch length hookups.
    """

    def __init__(self, branch=None):
        super(NetworkNode, self).__init__()
        self.branch = branch
        self.parent = None
        self.children = None

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
            if self.parent is None:
                self.parent = model_node

    def remove_successor(self, model_node):
        """
        Removes a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if model_node in self.successors:
            self.successors.remove(model_node)
            if self.parent is not None and model_node == self.parent:
                self.parent = None

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
            if self.children is None:
                self.children = [model_node]
            else:
                self.children.append(model_node)

    def remove_predecessor(self, model_node):
        """
        Removes a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if model_node in self.predecessors:
            self.predecessors.remove(model_node)
            if self.children is not None:
                if model_node in self.children:
                    self.children.remove(model_node)

    @abstractmethod
    def node_move_bounds(self):
        pass

    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children


class BranchLengthNode(CalculationNode):
    """
    A calculation node that uses the substitution model to calculate the
    transition matrix Pij
    """

    def __init__(self, vector_index, branch_length, heights=True):
        super().__init__()
        self.index = vector_index
        self.branch_length = branch_length
        self.sub = None
        self.updated_sub = True
        self.as_height = True

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
        if self.as_height:
            try:
                node = self.get_successors()[0]
                parent_height = node.get_parent().get_branch().get()
                branch_len = parent_height - self.branch_length
            finally:
                pass
        else:
            branch_len = self.branch_length

        if self.updated_sub:
            # grab current substitution model
            for child in self.get_predecessors():
                if type(child) is SubstitutionModel:
                    self.sub = child.get_submodel()
                    self.updated_sub = False
                    # print("calculating Pij for branch length: " + str(branch_len))
                    return child.get().expt(branch_len)
        else:
            # print("calculating Pij for branch length: " + str(branch_len))
            # TODO: cache this?
            return self.sub.expt(branch_len)


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

    def singular_update(self, index, value):
        for branch_node in self.get_successors():
            if branch_node.get_index() == index:
                branch_node.update(value)


    def get_heights(self):
        return self.heights


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


class ExtantSpecies(StateNode):

    def __init__(self, name, sequence):
        super().__init__()
        self.name = name
        self.seq = sequence

    def update(self, new_sequence, new_name):
        # should only have a single leaf calc node as the parent
        # TODO: make SNP flexible
        self.seq = new_sequence
        self.name = new_name
        self.get_successors()[0].update(new_sequence, new_name)  # Delegate what to do with the sequence to the leaf node, DONT KNOW DONT CARE

    def get_seq(self):
        return self.seq

    def get_name(self):
        return self.name


class FelsensteinLeafNode(NetworkNode, CalculationNode):

    def __init__(self, partials=None, branch=None, name: str = None):
        super(FelsensteinLeafNode, self).__init__(branch=branch)
        super(CalculationNode).__init__()
        self.matrix = partials
        self.name = name

    def node_move_bounds(self):
        return [0, self.parent.get_branch().get()]

    def update(self, new_partials, new_name):
        self.matrix = vec_bin_array(new_partials, 4)
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
                    self.matrix = vec_bin_array(child.get_seq(), 4)

        self.cached = self.matrix
        self.updated = False

        # return calculation
        return self.matrix


class FelsensteinInternalNode(NetworkNode, CalculationNode):
    def __init__(self, branch=None, name: str = None):
        super(FelsensteinInternalNode, self).__init__(branch=branch)
        super(CalculationNode).__init__()
        self.partials = None
        self.name = name

    def node_move_bounds(self):

        if self.parent is None:
            # root node
            return None
        # Normal internal node
        upper_limit = self.parent.get_branch().get()
        lower_limit = max(0, max([child.get_branch().get() for child in self.children]))
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


#### TESTS ######

# n2 = NetworkBuilder(
#     "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxaMultipleSites.nex")
# # n3 = NetworkBuilder(
# # "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxa1Site.nex")
#
# test2 = n2.getNetwork(0)
# # test3 = n3.getNetwork(0)
#
# msa2 = AlignIO.read(
#     "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxaMultipleSites.nex",
#     "nexus")
# # msa3 = AlignIO.read(
# # "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxa1Site.nex", "nexus")
#
# data2 = Matrix(msa2)  # default is to use the DNA alphabet
# # data3 = Matrix(msa3)
#
# model = Model(test2, data2)  # JC
# # model2 = Model(test3, data3)  # JC
#
# startFirst = time.perf_counter()
# print(model.likelihood())
# endFirst = time.perf_counter()

# model.change_branch(2, .5)
# startSecond = time.perf_counter()
# model.likelihood()
# endSecond = time.perf_counter()
#
# print("WHOLE GRAPH: " + str(endFirst - startFirst))
# print("RECALC GRAPH: " + str(endSecond - startSecond))

# print(model2.likelihood())


def SNP_compute_partials(matrix: Matrix, phased=False):
    if phased:
        r = [matrix.get_num_taxa() - sum(matrix.getColumnAt(i)) for i in
             range(matrix.siteCount())]  # sum of the columns
        x = [r[i] / matrix.get_num_taxa() for i in range(matrix.siteCount())]  # r_i / n
    else:
        r = [2 * matrix.get_num_taxa() - sum(matrix.getColumnAt(i)) for i in range(matrix.siteCount())]
        x = [r[i] / (2 * matrix.get_num_taxa()) for i in range(matrix.siteCount())]

    print(r)
    print(x)
    partials = {}

    for taxa in range(matrix.get_num_taxa()):
        likelihoods = np.zeros(matrix.siteCount())
        for site in range(matrix.siteCount()):
            likelihoods[site] = comb(2 * matrix.get_num_taxa(), r[site]) * pow(x[site], r[site]) * pow((1 - x[site]),
                                                                                                       2 * matrix.get_num_taxa() -
                                                                                                       r[site])

        partials[matrix.name_given_row(taxa)] = likelihoods

    return partials


class SNPLeafNode(NetworkNode, CalculationNode):

    def __init__(self, partials=None, branch=None, name: str = None):
        super(SNPLeafNode, self).__init__(branch=branch)
        self.matrix = partials
        self.name = name

    def node_move_bounds(self):
        return [0, self.parent.get_branch().get()]

    def update(self, new_partials, new_name):
        self.matrix = SNP_compute_partials(new_partials)
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
                    self.matrix = vec_bin_array(child.get_seq(), 4)

        self.cached = self.matrix
        self.updated = False

        # return calculation
        return self.matrix


class SNPInternalNode(NetworkNode, CalculationNode):

    def __init__(self, branch=None, name: str = None):
        super(SNPInternalNode, self).__init__(branch=branch)
        self.partials = None
        self.name = name

    def node_move_bounds(self):

        if self.parent is None:
            # root node
            return None
        # Normal internal node
        upper_limit = self.parent.get_branch().get()
        lower_limit = max(0, max([child.get_branch().get() for child in self.children]))
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
        # TODO: CHANGE THIS FUNC
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


# msa = MSA("C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\SNPtests\\snptest1.nex")
# mat = Matrix(msa, Alphabet("SNP"))
# print(mat.charMatrix())
#
# print(SNP_compute_partials(mat))
