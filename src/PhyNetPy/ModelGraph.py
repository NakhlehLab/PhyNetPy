""" 
Author : Mark Kessler
Last Stable Edit : 2/8/24
First Included in Version : 0.1.0
"""

from __future__ import annotations
from collections import defaultdict
import warnings
import random
from abc import ABC, abstractmethod
from GTR import *
from Network import Network, Edge, Node
from ModelMove import Move
from MSA import SeqRecord
from typing import Callable


##########################
#### HELPER FUNCTIONS ####
##########################

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


#########################
#### EXCEPTION CLASS ####
#########################

class ModelError(Exception):
    """
    Class to handle any errors related to building the model or running 
    likelihoods computations on the model.
    """

    def __init__(self, message="Model is Malformed"):
        super().__init__(message)

#####################
#### MODEL CLASS ####
#####################

class Model:
    """
    Class that implements a version of probabilistic graphical modeling, for 
    phylogenetics. Generally, it is made up of a network, along with various
    parameters and components that are attached to the model in order to 
    compute either a model likelihood or simulate data over a set of model
    parameters.
    """

    def __init__(self): 
        """
        Initialize an empty model.
        """
       
        # Maintain links to various internal structures and bookkeeping 
        # data 
        
        # Access all nodes of a given type
        self.all_nodes : dict[type, list[ModelNode]] = defaultdict(list)
        
        # Access the internal network
        self.network : Network = None 
        self.network_container = None
        
        # Access all network nodes of a certain kind 
        # (these are only explicitly defined by the in/out maps)
        self.nodetypes = {"leaf":[], 
                          "internal": [], 
                          "reticulation":[], 
                          "root":[]}
        
        # Maps parameter names (a string) to their parameter 
        # node (parent class) object
        self.parameters : dict[str, Parameter] = {} 
        
        # RNG object used to consistently select objects (useful mainly for 
        # debugging purposes, by setting a consistent seed instead of a random
        # seed).
        self.seed = random.randint(0, 1000)
        self.rng : np.random.Generator = np.random.default_rng(self.seed)
        
        # Map for the conversion between network Node objects and their 
        # associated model node object "wrapper"
        self.network_node_map : dict[Node, ModelNode] = {}
        
        #Log output
        self.summary_str = ""

    def change_branch(self, index: int, value: float) -> None:
        """
        Change a branch length in the model and update any nodes 
        upstream from the changed node.
        
        TODO: Edit for edge update

        Inputs: index - index into the heights/lengths vector
                value - new height/length to replace the old one

        """
        self.tree_heights.singular_update(index, value)

    def update_network(self) -> None:
        """
        Ensure that the network field and network container field
        are accessing the same network.
        """
        if self.network_container is not None:
            # Network container is a state node, and thus has an update method.
            self.network_container.update(self.network)
                
    def update_parameter(self, param_name : str, param_value : object) -> None:
        """
        Change the parameter value of the parameter with name 'param_name'
    
        Args:
            param_name (str): The name of the parameter to update. 
            param_value (object): A value, in whatever type the given
                                  parameter takes on.
        """
        self.parameters[param_name].update(param_value)
            
    def likelihood(self) -> float:
        """
        Calculates the likelihood of the model graph lazily, by only
        calculating parts of the model that have been updated/state changed.
        
        Delegates which likelihood based on the type of model. This method is 
        the only likelihood method that should be called outside of this 
        module!!!

        
        Returns: 
            float: A numerical likelihood value, the product of all root
                   vector likelihoods.
        """
        #TODO: this will change with root probability component
        return self.nodetypes["root"][0].get()

    def execute_move(self, move : Move) -> Model:
        """
        The operator move has asked for permission to work on this model.
        Pass the move this model and get the model that is the result of the 
        operation on this model. 

        Args:
            move (Move): A concrete subclass instance of Move.
        
        Returns: 
            Model: This !same! obj. The model will have changed based on the 
                   result of the move.
        """
        return move.execute(self)

    def summary(self, tree_filename : str, summary_filename : str) -> None:
        """
        Writes summary of calculations to a file, and gets the current state of 
        the model and creates a network obj so that the newick format 
        can be output.

        Inputs:
        1) tree_filename (str): A string that is the name of the file to output
                                 a newick string to. If the filename does not 
                                 exist, a new file will be created in the 
                                 directory in which one is operating in.

        2) summary_filename (str): A string that is the name of the file to 
                                   output logging information. If the filename 
                                   does not exist, a new file will be created 
                                   in the current directory.

        """
        newick_str = self.network.newick()

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


################################################
#### PROBABILISTIC GRAPHICAL MODELING NODES ####
################################################


class ModelNode:
    """
    Class that defines the graphical structure and shared interactions between
    any node in the Model.
    """

    def __init__(self, 
                 children : list = None, 
                 parents : list = None, 
                 node_type : str = None):
        
        self.children : list[ModelNode] = children
        self.parents : list[ModelNode] = parents
        self.node_type : str = node_type

    def add_child(self, model_node : ModelNode) -> None:
        """
        Adds a successor to this node.

        Input: model_node (type ModelNode)

        """
        if self.children is None:
            self.children = [model_node]
        else:
            self.children.append(model_node)

    def add_parent(self, model_node : ModelNode) -> None:
        """
        Adds a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if self.parents is None:
            self.parents = [model_node]
        else:
            self.parents.append(model_node)

    def join(self, other_node : ModelNode) -> None:
        """
        Adds other_node as a parent, and adds this node as
        a child of other_node.

        Args:
            other_node (ModelNode): A ModelNode to join this ModelNode to.
        """
        self.add_parent(other_node)
        other_node.add_child(self)

    def unjoin(self, other_node : ModelNode) -> None:
        self.remove_parent(other_node)
        other_node.remove_child(self)

    def remove_child(self, model_node):
        """
        Removes a successor to this node.

        Input: model_node (type ModelNode)
        """
        if model_node in self.children:
            self.children.remove(model_node)

    def remove_parent(self, model_node):
        """
        Removes a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if model_node in self.parents:
            self.parents.remove(model_node)

    def get_model_parents(self, of_type : type = None) -> list[ModelNode]:
        """
        Returns: the list of parent nodes to this node
        """
        if of_type is None:
            return self.parents
        else:
            return [node for node in self.parents if type(node) == of_type]

    def get_model_children(self, of_type : type = None) -> list[ModelNode]: 
        """
        Returns: the list of child nodes to this node
        """
        if of_type is None:
            return self.children
        else:
            return [node for node in self.children if type(node) == of_type]

    def in_degree(self) -> int:
        """
        Calculates the in degree of the current node (ie number of children)

        If 0, this node is a leaf
        """
        if self.parents is None:
            return 0
        return len(self.parents)

    def out_degree(self) -> int:
        """
        Calculates the out degree of the current node (ie number of parents)

        If 0, this node is a root of the Model
        """
        if self.children is None:
            return 0
        return len(self.children)

    def find_root(self) -> list[ModelNode]:
        """
        TODO: PLS MAKE MORE EFFICIENT THIS IS DUMB

        """
        if self.in_degree() == 0:
            return {self}
        else:
            roots = set()
            for neighbor in self.parents:
                roots.update(neighbor.find_root())  # set update

            return roots
        
class CalculationNode(ABC, ModelNode):
    """
    TODO: flush out some logic and return types. 
    
    Subclass of a ModelNode that calculates a portion of the model likelihood or
    data simulation.
    
    In probabilistic graphical modeling, this is also known as a deterministic 
    node.
    """

    def __init__(self) -> None:
        super(CalculationNode, self).__init__()
        
        # defaults to dirty since calculation hasn't been done yet
        self.dirty = True  
        self.cached = None
        
    @abstractmethod
    def get(self):
        """
        Either retrieves the cached calculation or redoes the calculation for 
        this node. This is an abstract method, due to the fact that the type of
        recalculation will vary.

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
    def sim(self, *args, **kwargs) -> None:
        """
        This method should be implemented in each CalculationNode subclass.
        Doing a calculation should be a unique operation depending on the type of node.

        Returns: A vector of partial likelihoods.
        """
        pass
    
    def update(self, *args, **kwargs) -> None:
        """
        This method should be implemented in each CalculationNode subclass.
        Updating internal data should be handled on an individual basis.

        When the model graph runs its calculate routine, this update method 
        will have marked this calculation node and any calculation nodes 
        upstream as needing recalculation.
        """
        self.upstream()
        
    def upstream(self) -> None:
        """
        Finds a path within the model graph from this node to the root, and 
        marks each node along the way as updated using the 
        switch_updated() method.

        If all neighbors need to be recalculated, then so must every node 
        upstream of it, and so we may stop updating.
        """
        # First update self
        self.make_dirty()

        # Get parent nodes and check that this node is not the root 
        # (in which case we're done). Only leaves may be of class other than
        # CalculationNode, so it is safe to assume a model parent has the 
        # upstream method implemented on it
        neighbors : list[CalculationNode] = self.get_model_parents()
        if neighbors is None:
            return

        roots = self.find_root()


        #TODO: Streamline this, this is bad logic
        # If all parent nodes are marked to be recalculated, then so must 
        # be each path from this node to the root, so no further steps are 
        # required
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

    def make_dirty(self) -> None:
        """
        A model node is updated if any of its calculation nodes downstream have 
        been changed.

        This method will be called when a downstream node calls its upstream() 
        method, setting this node as a node that needs to be recalculated.
        """
        self.dirty = True
        
    def cache(self, value : object) -> object:
        """
        Place some likelihood calculation or simulated data in the cache.

        Args:
            value (object): Some simulated data or likelihood computations.

        Returns:
            object: The value that was just cached.
        """
        self.cached = value
        self.dirty = False
        return self.cached
    
    def get_parameters(self) -> dict[str, float]:
        """
        Retrieves any parameters that are attached to this calculation node.

        Returns:
            dict[str, float]: A map from parameter names to their values.
        """
        params = self.get_model_children(Parameter)
        return {child.name : child.value for child in params}
        
class StateNode(ABC, ModelNode):
    """
    TODO: Make init and update docs.
    Model leaf nodes that hold some sort of data that calculation nodes use.
    
    In probabilistic graphical modeling, these are either clamped, constant, or
    observed values for parameters or data.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

class Parameter(StateNode):
    """
    A subtype of a StateNode, that is a parameter for the model.
    A parameter typically holds a numerical value that defines some sort of 
    prior distribution, or a value that defines behavior of 
    transition matrices, etc.
    """
    
    def __init__(self, name : str, value : object) -> None:
        """
        A parameter is defined by its name and value.

        Args:
            name (str): Name. ie, "u" for red -> green transition probability 
                        for SNPs.
            value (object): Value for the parameter. Ie, for "u", valid values 
                            would be numbers from 0 to 1.
        """
        super().__init__()
        
        #Define the name of the parameter, and its value.
        self.name : str = name
        self.value : object = value
    
    def update(self, value : object) -> None:
        """
        After changing the parameter, things that rely on the parameter for 
        their own computations need to be updated to reflect the change.
        
        Ie, for SNP models, if the parameter value u (red -> green transition
        probability) changes, then the Q matrix needs to be re-populated with 
        values.

        Args:
            value (object): A new value for the parameter.
        """
        self.value = value
        parents : list[ModelNode] = self.get_model_parents()
        
        for par in parents:
            #Any non-leaf node of a model should be a calculation node
            par.update() 
    
    def get_name(self) -> str:
        """
        Get the name of the parameter.

        Returns:
            str: The parameter name
        """
        return self.name
    
    def get_value(self) -> object:
        """
        Get the value of the parameter

        Returns:
            object: Some value.
        """
        return self.value
            
class Accumulator(StateNode):
    """
    Class that accumulates data from computations made across the model.
    Ie, for MCMC_Bimarkers, the vectors for partial likelihoods are defined by
    branches all across the network. Each network branch contributes to 
    maintaining bookkeeping for this data structure, and thus an Accumulator 
    node makes it easy to access from anywhere.
    
    Essentially a data bookkeeping structure.
    """
    def __init__(self, name : str, data_structure : object) -> None:
        """
        Accumulators are defined by name and the data they store.

        Args:
            name (str): Label for the accumulator
            data_structure (object): The data store.
        """
        super().__init__()
        self.data = data_structure
        self.name : str = name
    
    def update(self) -> None:
        """
        Update behaviors are defined in the subclass implementation.
        """
        pass
    
    def get_data(self) -> object:
        """
        Grab the data stored in this accumulator.

        Returns:
            object: The data store.
        """
        return self.data

#------#

class NetworkNode(ABC, ModelNode):
    """
    Class that handles common functionality of all network nodes
    and all the height/branch length hookups.
    """

    def __init__(self, branch = None):
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

    def add_child(self, model_node):
        """
        Adds a successor to this node.

        Input: model_node (type ModelNode)
        """
        if self.children is None:
            self.children = [model_node]
        else:
            self.children.append(model_node)

        if type(model_node) is NetworkNode:
            if self.parents is None:
                self.parents = [model_node]
            else:
                self.parents.append(model_node)

    def remove_child(self, model_node):
        """
        Removes a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if model_node in self.children:
            self.children.remove(model_node)
            if self.parents is not None and model_node in self.parents:
                self.parents.remove(model_node)
                if len(self.parents) == 0:
                    self.parents = None

    def add_parent(self, model_node):
        """
        Adds a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if self.parents is None:
            self.parents = [model_node]
        else:
            self.parents.append(model_node)

        if type(model_node) is FelsensteinInternalNode or type(model_node) is FelsensteinLeafNode:
        #\
                #or type(model_node) is SNPInternalNode or type(model_node) is SNPLeafNode:
            if self.children is None:
                self.children = [model_node]
            else:
                self.children.append(model_node)

    def remove_parent(self, model_node):
        """
        Removes a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if model_node in self.parents:
            self.parents.remove(model_node)
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
    def __init__(self, vector_index : int, branch_length : float) -> None:
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
    TODO: Make this consistent with having the two separate 
    transition/transversion parameters
    """
    def __init__(self, submodel : GTR) -> None:
        """
        Deterministic node that is often hooked up to transition, transversion,
        and base frequency parameters.

        Args:
            submodel (GTR): Any time reversible substitution model.
        """
        super().__init__()
        self.sub : GTR = submodel

    def update(self, new_sub_model : GTR) -> None:
        """
        Change the substitution model being used.

        Args:
            new_sub_model (GTR): The new type of substitution model to use.
                                 Be careful that the associated parameters 
                                 hooked up to this node still apply! If they do
                                 not, then the get() method will fail.
            
        """
        # Set the new parameters
        self.sub = new_sub_model
        # Mark this node and any nodes upstream as needing to be recalculated
        self.upstream()

    def get(self) -> GTR:
        """
        Based on the associated parameters, initialize a substitution model.

        Returns:
            GTR: The substitution model with the parameters values in the model 
                 graph.
        """
        if self.dirty:
            return self.calc()
        else:
            return self.cached

    def calc(self) -> GTR:
        #No longer in need of an update
        self.dirty = False
        
        param_dict : dict[str, object] = {}
        params : list[Parameter]= self.get_model_children(Parameter)
        for param_node in params:
            param_dict[param_node.name] = param_node.value
        
        self.sub.set_params(param_dict)
        self.cached = self.sub
        return self.sub

    def get_submodel(self) -> GTR:
        return self.sub

class ExtantSpecies(StateNode):
    """
    Node that links network leaf nodes to their MSA data. Falls under the 
    category of observed data.
    """

    def __init__(self, name : str, sequences: list[SeqRecord]) -> None:
        """
        Link a taxon name to its set of data sequences.

        Args:
            name (str): Taxon label.
            sequence (list[SeqRecord]): list of data sequences associated with 
                                        this taxa.
        """
        super().__init__()
        self.name : str = name
        self.seqs : list[SeqRecord]= sequences

    def update(self, new_sequence: list, new_name: str):
        # should only have a single leaf calc node as the parent
        self.seq = new_sequence
        self.name = new_name
        self.get_model_parents()[0].update(new_sequence, new_name)

    def seq_len(self) -> int:
        """
        Get the sequence length of all sequences associated with this taxon.

        Returns:
            int: Length of data sequence.
        """
        return len(self.seqs[0].get_seq())
        

    def get_seqs(self) -> list[SeqRecord]:
        """
        Get the list of sequence records associated with this taxon.

        Returns:
            list[SeqRecord]: List of sequence records.
        """
        return self.seqs

    def get_name(self) -> str:
        """
        Get the taxon label.

        Returns:
            str: Taxon label.
        """
        return self.name



