""" 
Author : Mark Kessler
Last Stable Edit : 7/16/23
First Included in Version : 0.1.0
Approved to Release Date : N/A
"""

import math
import warnings
import random
from abc import ABC, abstractmethod
from math import comb, pow
from GTR import *
from Graph import DAG
from Matrix import Matrix
from Move import Move
#from Bayesian.SNPTransition import SNPTransition
from scipy.special import binom
import scipy
#from Bayesian.SNPModule import *
from Node import Node
#from Bayesian.InferAllop import InferMPAllop
from typing import Callable


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


# def convert_to_heights(node, adj_dict):
#     """
#     This is a recursive function that is used to take a model that is initialized
#     with branch heights and turn it into a model based on node heights.

#     Usage: convert_to_heights(root_node, {})

#     The resulting heights are conditioned such that t=0 is at the root. Need to subtract dictionary value from
#     max(heights of all leaves) to get heights such that the root is the time furthest in the past

#     input: a ModelNode, to change the whole graph use the root.
#     output: a dictionary that maps each model node to a float height value

#     """
    
#     if node.get_parent() is None:  # Root
#         adj_dict[node] = 0  # Start root at t=0
#     else:
#         # For all other nodes, the height will be the branch length plus the node height of its parent
#         for branch in node.get_branches():
#             # Doesn't matter which parent is used to calculate node height, so just use the first one
#             if type(branch) is SNPBranchNode:
#                 if branch.net_parent in adj_dict.keys():
#                     adj_dict[node] = branch.get_length() + adj_dict[branch.net_parent]
#             else:
#                 adj_dict[node] = branch.get() + adj_dict[node.get_parent()]

#     # Done at the leaves
#     if type(node) is FelsensteinLeafNode or type(node) is SNPLeafNode:
#         return adj_dict

#     # Otherwise, recursively call on the children of this node
#     if node.get_children() is not None:
#         for child in node.get_children():
#             # combine maps of all children
#             adj_dict.update(convert_to_heights(child, adj_dict))

#     # Return the built-up mapping
#     return adj_dict


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

    def __init__(self): #, network: DAG = None, data: Matrix = None, submodel=JC()):
       
        ##-------ONLY USE THIS STUFF AS OF NOW-------##
        self.network = None #network
        self.network_container = None
        self.nodetypes = {"leaf":[], "internal": [], "reticulation":[], "root":[]}
        self.parameters : dict = {} #Maps parameter names (a string) to their parameter node (parent class) object
        
        rand_seed = random.randint(0, 1000)
        self.seed = rand_seed
        print(f"MODEL SEED: {rand_seed}")
        self.rng = np.random.default_rng(rand_seed)
        
        ##-------------------------------------------##
        
        # self.sub = submodel
        # self.data = data
        
        # self.verbose_out = False
        
        self.nodes = []
        self.netnodes_sans_root = []
        self.network_leaves = []
        
        self.tree_heights = None  # type TreeHeights
        self.submodel_node = None # type SubstitutionModel
       
        self.network_node_map : dict[Node, ModelNode] = {}
        self.snp_params = None #snp_params
                
        #self.internal = [item for item in self.netnodes_sans_root if item not in self.network_leaves]
        self.summary_str = ""

    def change_branch(self, index: int, value: float):
        """
        Change a branch length in the model and update any nodes upstream from the changed node

        Inputs: index - index into the heights/lengths vector
                value - new height/length to replace the old one

        """
        self.tree_heights.singular_update(index, value)

    
    def update_network(self):
        if self.network_container is not None:
            self.network_container.update(self.network)
                

    def update_parameter(self, param_name : str, param_value):
        self.parameters[param_name].update(param_value)
            

    def likelihood(self):
        """
        Calculates the likelihood of the model graph lazily, by only
        calculating parts of the model that have been updated/state changed.
        
        Delegates which likelihood based on the type of model. This method is the only 
        likelihood method that should be called outside of this module!!!

        Inputs:
        Outputs: A numerical likelihood value, the dot product of all root vector likelihoods
        """
        #TODO: this will change with root probability component
        return self.nodetypes["root"][0].get()

    def execute_move(self, move: Move):
        """
        The operator move has asked for permission to work on this model.
        Pass the move this model and get the model that is the result of the operation on this model. IT IS THE SAME OBJ

        Input: move, a Move obj or any subtype
        Output: the !same! obj that is the result of doing Move on this Model obj
        """
        return move.execute(self)

    def summary(self, tree_filename: str, summary_filename: str):
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
        if self.data.get_type() == "SNP":
            network_nodes.extend([self.snp_root])
        else:
            network_nodes.extend([self.felsenstein_root])
            
        network_nodes.extend(self.network_leaves)
        network_nodes.extend(self.netnodes_sans_root)

        inv_map = {v: k for k, v in self.network_node_map.items()}
        net.add_nodes([inv_map[node] for node in network_nodes])

        for node in network_nodes:
            for branch in node.get_branches():
                branch_len = branch.get()
                if node.parents is not None:
                    inv_map[node].set_length(branch_len, None)
                else:
                    inv_map[node].set_length(branch_len, None)

            # Add edges
            if node.get_children() is not None:
                for child in node.get_children():
                    net.add_edges([inv_map[node], inv_map[child]]) 

        newick_str = net.newick()

        # Write newick string to output file
        if tree_filename is not None:
            text_file = open(tree_filename, "w")
            text_file.write(newick_str)
            text_file.close()

        # Step 2: write iter summary to a file
        if summary_filename is not None:
            text_file2 = open(summary_filename, "w")
            text_file2.write(self.summary_str)
            text_file2.close()

class ModelNode:
    """
    Class that defines the graphical structure and shared interactions between
    any node in the Model.
    """

    def __init__(self, successors : list = None, predecessors : list = None, node_type : str = None):
        self.successors = successors
        self.predecessors = predecessors
        self.node_type = node_type

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
        self.add_predecessor(other_node)
        other_node.add_successor(self)

    def unjoin(self, other_node):
        self.remove_predecessor(other_node)
        other_node.remove_successor(self)

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

    def get_model_parents(self, of_type : type = None):
        """
        Returns: the list of parent nodes to this node
        """
        if of_type is None:
            return self.predecessors
        else:
            return [node for node in self.predecessors if type(node) == of_type]

    def get_model_children(self, of_type : type = None): 
        """
        Returns: the list of child nodes to this node
        """
        if of_type is None:
            return self.successors
        else:
            return [node for node in self.successors if type(node) == of_type]

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
        if self.in_degree() == 0:
            return {self}
        else:
            roots = set()
            for neighbor in self.predecessors:
                roots.update(neighbor.find_root())  # set update

            return roots
        
class CalculationNode(ABC, ModelNode):
    """
    Subclass of a ModelNode that calculates a portion of the model likelihood.
    """

    def __init__(self):
        super(CalculationNode, self).__init__()
        self.dirty = True  # defaults to dirty since calculation hasn't been done yet
        self.cached = None
        
    

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
    
    @abstractmethod
    def sim(self, *args, **kwargs):
        """
        This method should be implemented in each CalculationNode subclass.
        Doing a calculation should be a unique operation depending on the type of node.

        Returns: A vector of partial likelihoods.
        """
        pass
    
    def update(self, *args, **kwargs):
        """
        This method should be implemented in each CalculationNode subclass.
        Updating internal data should be handled on an individual basis.

        When the model graph runs its calculate routine, this update method will have marked
        this calculation node and any calculation nodes upstream as needing recalculation.
        """
        self.upstream()
        
    def upstream(self):
        """
        Finds a path within the model graph from this node to the root, and marks each node along the way as updated
        using the switch_updated() method

        If all neighbors need to be recalculated, then so must every node upstream of it, and so we may stop updating
        """
        # First update self
        self.make_dirty()

        # Get parent nodes and check that this node is not the root (in which case we're done)
        neighbors = self.get_model_parents()
        if neighbors is None:
            return

        roots = self.find_root()

        # If all parent nodes are marked to be recalculated, then so must be each path from this node to the root,
        # so no further steps are required
        all_dirty = True
        for neighbor in neighbors:
            if not neighbor.dirty:
                all_dirty = False

        # Otherwise, call upstream on each neighbor
        if not all_dirty:
            for neighbor in neighbors:
                if neighbor in roots:
                    neighbor.upstream()
                    return
                neighbor.upstream()

    def make_dirty(self):
        """
        A model node is updated if any of its calculation nodes downstream have been changed.

        This method will be called when a downstream node calls its upstream() method, setting this node
        as a node that needs to be recalculated.
        """

        self.dirty = True
        
    def cache(self, value):
        self.cached = value
        self.dirty = False
        return self.cached
    
    def get_parameters(self)-> dict[str, float]:
        """
        Retrieves any parameters that are attached to this calculation node

        Returns:
            dict[str, float]: a map from parameter names to their values
        """
        return {child.name : child.value for child in self.get_model_children(Parameter)}
        

class StateNode(ABC, ModelNode):
    """
    Model leaf nodes that hold some sort of data that calculation nodes use
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

class Parameter(StateNode):
    def __init__(self, name : str, value):
        super().__init__()
        self.name = name
        self.value = value
    
    def update(self, value):
        self.value = value
        
        for par in self.get_model_parents():
            par.update() #Any non-leaf node of a model should be a calculation node
            
class Accumulator(StateNode):
    def __init__(self, name : str, data_structure : object):
        super().__init__()
        self.data = data_structure
        self.name = name
    
    def update(self):
        pass
    
    def get_data(self):
        return self.data

class NetworkNode(ABC, ModelNode):
    """
    Class that handles common functionality of all network nodes
    and all the height/branch length hookups.
    """

    def __init__(self, branch=None):
        super(NetworkNode, self).__init__()
        self.branches = branch
        self.network_parents : list[NetworkNode]= None
        self.network_children : list[NetworkNode] = None

    def get_parent_branches(self):
        if self.network_parents is None:
            return None
        else:
            par_branches = {}
            for par in self.network_parents:
                par_branches[par] = [branch for branch in par.get_child_branches() if branch.dest() == self]
            return par_branches
    
    def get_child_branches(self):
        if self.branches is None:
            self.branches = {}
            all_branches = [child for child in self.get_model_parents(BranchNode)]
            for branch in all_branches:
                self.branches[branch.dest()] = branch
        return self.branches

    def add_successor(self, model_node):
        """
        Adds a successor to this node.

        Input: model_node (type ModelNode)
        """
        if self.successors is None:
            self.successors = [model_node]
        else:
            self.successors.append(model_node)

        if type(model_node) is NetworkNode:
            if self.parents is None:
                self.parents = [model_node]
            else:
                self.parents.append(model_node)

    def remove_successor(self, model_node):
        """
        Removes a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if model_node in self.successors:
            self.successors.remove(model_node)
            if self.parents is not None and model_node in self.parents:
                self.parents.remove(model_node)
                if len(self.parents) == 0:
                    self.parents = None

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
        #\
                #or type(model_node) is SNPInternalNode or type(model_node) is SNPLeafNode:
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

    def get_parent(self, return_all = False):
        if return_all:
            return self.parents
        else:
            if self.parents is not None:
                return self.parents[0]
            else:
                return None

    def get_children(self):
        return self.children

class BranchNode(ABC, ModelNode):
    def __init__(self, vector_index: int, branch_length: float) -> None:
        super().__init__()
        self.index : int = vector_index
        self.branch_length : float = branch_length
        self.net_parent : NetworkNode = None
        self.net_child : NetworkNode = None
        self.gamma : float = None

    def switch_index(self, new_index:int):
        """
        Change the lookup index of this branch in the TreeHeight node

        Args:
            new_index (int): a new index
        """
        self.index = new_index

    def get_index(self):
        """
        Returns:
            int: The index into the TreeHeight vector
        """
        return self.index

    def set_net_parent(self, parent: NetworkNode):
        """
        Set the network parent 

        Args:
            parent (NetworkNode): the source of this branch
        """
        self.net_parent = parent
    
    def set_net_child(self, child: NetworkNode):
        """
        Set the network child

        Args:
            child (NetworkNode): the destination of this branch
        """
        self.net_child = child
    
    def src(self):
        return self.net_parent
    
    def dest(self):
        return self.net_child

    def inheritance_probability(self)->float:
        """
        Return the gamma rate/ inheritance probability for a branch stemming from a hybridization node

        Returns:
            float: A number from [0,1], -1 if no inheritance probability is attached to this branch
        """
        if self.gamma is None:
            warnings.warn("An inheritance probability is not available for this node-- returning a value of -1")
            return -1
        return self.gamma
    
    def set_inheritance_probability(self, new_gamma : float)->None:
        self.gamma = new_gamma
    
class BranchLengthNode(BranchNode, CalculationNode):
    """
    A calculation node that uses the substitution model to calculate the
    transition matrix Pij
    """

    def __init__(self, vector_index: int, branch_length: float):
        super().__init__(vector_index, branch_length)
        self.as_height = True

    def update(self, new_bl: float):
        # update the branch length
        self.branch_length = new_bl
        # Mark this node and any nodes upstream as needing to be recalculated
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

    

    # def transition(self):
    #     """
    #     Calculate the Pij matrix
    #     """
    #     if self.as_height:
    #         try:
    #             node = self.get_model_children()[0]
    #             if node.get_parent() is None:
    #                 branch_len = 0
    #             else:
    #                 #Only trees, grab [0]
    #                 parent_height = node.get_parent().get_branches()[0].get()
    #                 branch_len = parent_height - self.branch_length
    #         finally:
    #             pass
    #     else:
    #         branch_len = self.branch_length

    #     if self.updated_sub:
    #         # grab current substitution model
    #         for child in self.get_model_parents():
    #             if type(child) is SubstitutionModel:
    #                 self.sub = child.get_submodel()
    #                 self.updated_sub = False
    #                 return child.get().expt(branch_len)
    #     else:
    #         # TODO: cache this?
    #         return self.sub.expt(branch_len)

class TreeHeights(StateNode):
    """
    State node that holds the node heights/branch lengths
    """

    def __init__(self, node_height_vec=None):
        super().__init__()
        self.heights = node_height_vec

    def update(self, new_vector: list):

        # Only update the parts of the vector that have changed
        if self.heights is None:
            self.heights = new_vector
            for branch_node in self.get_model_children():
                branch_node.update(self.heights[branch_node.get_index()])
        else:
            for branch_node in self.get_model_children():
                if new_vector[branch_node.get_index()] != self.heights[branch_node.get_index()]:
                    branch_node.update(new_vector[branch_node.get_index()])

            self.heights = new_vector

    def singular_update(self, index: int, value: float):
        for branch_node in self.get_model_children():
            if branch_node.get_index() == index:
                branch_node.update(value)

    def get_heights(self):
        return self.heights

class SubstitutionModelParams(StateNode):
    """
    TODO: Switch to multiple Param type nodes
    """
    def __init__(self, freq: np.array, trans: np.array) -> None:
        super().__init__()
        self.base_freqs = freq
        self.transitions = trans

    def update(self, new_freqs=None, new_trans=None):

        # should only have the one parent
        submodel_node = self.get_model_children()[0]

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
    """
    TODO: Make this consistent with having the two separate transition/transversion parameters
    """
    def __init__(self, submodel: GTR):
        super().__init__()
        self.sub = submodel

    def update(self, new_sub_model: GTR):
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

    def __init__(self, name: str, sequence: list):
        super().__init__()
        self.name = name
        #print("ADDING SEQ TO " + self.name + " : " + str(sequence))
        self.seq = sequence

    def update(self, new_sequence: list, new_name: str):
        # should only have a single leaf calc node as the parent
        self.seq = new_sequence
        self.name = new_name
        self.get_model_children()[0].update(new_sequence, new_name)

    def seq_len(self):
        if type(self.seq) is list:
            return len(self.seq[0].get_seq())
        else:
            return len(self.seq)

    def get_seq(self):
        return self.seq

    def get_name(self):
        return self.name



