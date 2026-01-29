#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##
##  See "LICENSE.txt" for terms and conditions of usage.
##
##  If you use this work or any portion thereof in published work,
##  please cite it as:
##
##     Mark Kessler, Luay Nakhleh. 2025.
##
##############################################################################

""" 
Author : Mark Kessler
Last Stable Edit : 3/11/25
First Included in Version : 0.1.0

V1 Architecture - Model Graph for probabilistic graphical modeling in phylogenetics.
"""

from __future__ import annotations
from collections import defaultdict
import warnings
import random
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, TYPE_CHECKING
import numpy as np

# Relative imports
from .GTR import GTR, JC, K80
from .Network import Network, Edge, Node
from .MSA import DataSequence

if TYPE_CHECKING:
    from .ModelMove import Move


##########################
#### HELPER FUNCTIONS ####
##########################

def vec_bin_array(arr: np.array, m: int) -> np.array:
    """
    Convert an array of integers into a binary array of bits.
    
    Args:
        arr (np.array): Numpy array of positive integers
        m (int): Number of bits of each integer to retain (ie DNA, 4)

    Returns: 
        (np.array): a copy of arr with every element replaced with a bit vector.
                    Bits encoded as int8's, read from left to right. 
                    [1, 0, 1, 1] is 13.
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

    def __init__(self, message: str = "Model is Malformed") -> None:
        """
        Create a custom ModelError with a message.

        Args:
            message (str, optional): A custom error message. Defaults to 
                                     "Model is Malformed".
        Returns:
            N/A
        """
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
        Initialize a model object.
        
        Args:
            N/A
        Returns:
            N/A
        """
        # Maintain links to various internal structures and bookkeeping data 
        self.all_nodes: dict[type, list[ModelNode]] = defaultdict(list)
        self.network: Network = None 
        self.network_container = None
        self.nodetypes = {"leaf": [], "internal": [], "reticulation": [], "root": []}
        self.parameters: dict[str, Parameter] = {} 
        self.seed = random.randint(0, 1000)
        self.rng: np.random.Generator = np.random.default_rng(self.seed)
        self.network_node_map: dict[Node, ModelNode] = {}
        self.summary_str = ""
        self.tree_heights = None

    def change_branch(self, index: int, value: float) -> None:
        """
        Change a branch length in the model and update any nodes 
        upstream from the changed node.
        
        Args:
            index (int): Index into the heights/lengths vector.
            value (float): New height/length to replace the old one.
        Returns:
            N/A
        """
        if self.tree_heights is not None:
            self.tree_heights.singular_update(index, value)

    def update_network(self) -> None:
        """
        Ensure that the network field and network container field
        are accessing the same network.
        
        Args:
            N/A
        Returns:
            N/A
        """
        if self.network_container is not None:
            self.network_container.update(self.network)
                
    def update_parameter(self, param_name: str, param_value: object) -> None:
        """
        Change the parameter value of the parameter with name 'param_name'.
    
        Args:
            param_name (str): The name of the parameter to update. 
            param_value (object): A value, in whatever type the given
                                  parameter takes on.
        Returns:
            N/A
        """
        self.parameters[param_name].update(param_value)
            
    def likelihood(self) -> float:
        """
        Calculates the likelihood of the model graph lazily, by only
        calculating parts of the model that have been updated/state changed.
        
        Delegates which likelihood based on the type of model. This method is 
        the only likelihood method that should be called outside of this 
        module!!!
        
        Args:
            N/A
        Returns: 
            float: A numerical likelihood value, the product of all root
                   vector likelihoods.
        """
        return self.nodetypes["root"][0].get()

    def execute_move(self, move: Move) -> Model:
        """
        The operator move has asked for permission to work on this model.
        Pass the move this model and get the model that is the result of the 
        operation on this model. 

        Args:
            move (Move): A concrete subclass instance of Move.
        Returns: 
            Model: This same object. The model will have changed based on the 
                   result of the move.
        """
        return move.execute(self)

    def summary(self, tree_filename: str, summary_filename: str) -> None:
        """
        Writes summary of calculations to a file, and gets the current state of 
        the model and creates a network obj so that the newick format 
        can be output.

        Args:
            tree_filename (str): A string that is the name of the file to output
                                 a newick string to. If the filename does not 
                                 exist, a new file will be created in the 
                                 directory in which one is operating in.
            summary_filename (str): A string that is the name of the file to 
                                   output logging information. If the filename 
                                   does not exist, a new file will be created 
                                   in the current directory.
        Returns:
            N/A
        """
        newick_str = self.network.newick()

        # Write newick string to output file
        if tree_filename is not None:
            with open(tree_filename, "w") as text_file:
                text_file.write(newick_str)

        # Write iter summary to a file
        if summary_filename is not None:
            with open(summary_filename, "w") as text_file2:
                text_file2.write(self.summary_str)


################################################
#### PROBABILISTIC GRAPHICAL MODELING NODES ####
################################################


class ModelNode:
    """
    Class that defines the graphical structure and shared interactions between
    any node in the Model.
    """

    def __init__(self, 
                 children: list = None, 
                 parents: list = None, 
                 node_type: str = None) -> None:
        """
        Initialize a ModelNode object.

        Args:
            children (list, optional): Children of this node. Defaults to None.
            parents (list, optional): Parents of this node. Defaults to None.
            node_type (str, optional): A string that describes the type of node.
        """
        self.children: list[ModelNode] = children
        self.parents: list[ModelNode] = parents
        self.node_type: str = node_type

    def add_child(self, model_node: ModelNode) -> None:
        """
        Adds a successor to this node.

        Args:
            model_node (ModelNode): A ModelNode to add as a child.
        Returns:
            N/A
        """
        if self.children is None:
            self.children = [model_node]
        else:
            self.children.append(model_node)

    def add_parent(self, model_node: ModelNode) -> None:
        """
        Adds a predecessor to this node.

        Args:
            model_node (ModelNode): A ModelNode to add as a parent.
        Returns:
            N/A
        """
        if self.parents is None:
            self.parents = [model_node]
        else:
            self.parents.append(model_node)

    def join(self, other_node: ModelNode) -> None:
        """
        Adds other_node as a parent, and adds this node as
        a child of other_node.

        Args:
            other_node (ModelNode): A ModelNode to join this ModelNode to.
        Returns:
            N/A
        """
        self.add_parent(other_node)
        other_node.add_child(self)

    def unjoin(self, other_node: ModelNode) -> None:
        """
        Removes other_node as a parent, and removes this node as
        a child of other_node.

        Args:
            other_node (ModelNode): A ModelNode to unjoin this ModelNode from.
        Returns:
            N/A
        """
        self.remove_parent(other_node)
        other_node.remove_child(self)

    def remove_child(self, model_node: ModelNode) -> None:
        """
        Removes a successor to this node.

        Args:
            model_node (ModelNode): A ModelNode to remove as a child.
        Returns:
            N/A
        """
        if self.children is not None and model_node in self.children:
            self.children.remove(model_node)

    def remove_parent(self, model_node: ModelNode) -> None:
        """
        Removes a predecessor to this node.

        Args:
            model_node (ModelNode): A ModelNode to remove as a parent.
        Returns:
            N/A
        """
        if self.parents is not None and model_node in self.parents:
            self.parents.remove(model_node)

    def get_model_parents(self, of_type: type = None) -> list[ModelNode]:
        """
        Get the parent nodes to this node, but only the ModelNodes.
        
        Args:
            of_type (type, optional): A type to filter the parent nodes by.
        Returns: 
            list[ModelNode]: The list of parent nodes to this node
        """
        if of_type is None:
            return self.parents
        else:
            return [node for node in self.parents if type(node) == of_type]

    def get_model_children(self, of_type: type = None) -> list[ModelNode]: 
        """
        Get the children nodes to this node, but only the ModelNodes.

        Args:
            of_type (type, optional): A type to filter the child nodes by.
        Returns:
            list[ModelNode]: The list of child nodes to this node
        """
        if of_type is None:
            return self.children
        else:
            # Use isinstance to include subclasses
            return [node for node in self.children if isinstance(node, of_type)]

    def in_degree(self) -> int:
        """
        Calculates the in degree of the current node (ie number of parents)

        If 0, this node is a root of the Model.
        
        Args:
            N/A
        Returns:
            int: The number of parents to this node.
        """
        if self.parents is None:
            return 0
        return len(self.parents)

    def out_degree(self) -> int:
        """
        Calculates the out degree of the current node (ie number of children)

        If 0, this node is a leaf.

        Args:
            N/A
        Returns:
            int: The number of children to this node.
        """
        if self.children is None:
            return 0
        return len(self.children)

    def find_root(self) -> set[ModelNode]:
        """
        Find the root node of the model graph.

        Args:
            N/A
        Returns:
            set[ModelNode]: A set of root nodes.
        """
        if self.in_degree() == 0:
            return {self}
        else:
            roots = set()
            for neighbor in self.parents:
                roots.update(neighbor.find_root())

            return roots
        
class CalculationNode(ABC, ModelNode):
    """
    Subclass of a ModelNode that calculates a portion of the model likelihood or
    data simulation.
    
    In probabilistic graphical modeling, this is also known as a deterministic 
    node.
    """

    def __init__(self) -> None:
        """
        Initialize a CalculationNode object.
        
        Args:
            N/A
        Returns:
            N/A
        """
        super(CalculationNode, self).__init__()
        
        # defaults to dirty since calculation hasn't been done yet
        self.dirty = True  
        self.cached = None
        
    @abstractmethod
    def get(self) -> Any:
        """
        Either retrieves the cached calculation or redoes the calculation for 
        this node. This is an abstract method, due to the fact that the type of
        recalculation will vary.

        Args:
            N/A
        Returns: 
            Any: a vector of partial likelihoods
        """
        pass
        

    @abstractmethod
    def calc(self, *args, **kwargs) -> Any:
        """
        This method should be implemented in each CalculationNode subclass.
        Doing a calculation should be a unique operation depending on the type of node.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns: 
            Any: A vector of partial likelihoods.
        """
        pass
    
    @abstractmethod
    def sim(self, *args, **kwargs) -> None:
        """
        This method should be implemented in each CalculationNode subclass.
        Doing a calculation should be a unique operation depending on the type of node.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns: 
            N/A
        """
        pass
    
    def update(self, *args, **kwargs) -> None:
        """
        This method should be implemented in each CalculationNode subclass.
        Updating internal data should be handled on an individual basis.

        When the model graph runs its calculate routine, this update method 
        will have marked this calculation node and any calculation nodes 
        upstream as needing recalculation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:   
            N/A
        """
        self.upstream()
        
    def upstream(self) -> None:
        """
        Finds a path within the model graph from this node to the root, and 
        marks each node along the way as updated using the 
        switch_updated() method.

        If all neighbors need to be recalculated, then so must every node 
        upstream of it, and so we may stop updating.
        
        Args:
            N/A
        Returns:
            N/A
        """
        # First update self
        self.make_dirty()

        # Get parent nodes and check that this node is not the root 
        # (in which case we're done). Only leaves may be of class other than
        # CalculationNode, so it is safe to assume a model parent has the 
        # upstream method implemented on it
        neighbors: list[CalculationNode] = self.get_model_parents()
        if neighbors is None:
            return

        roots = self.find_root()

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
        
        Args:
            N/A
        Returns:
            N/A
        """
        self.dirty = True
        
    def cache(self, value: object) -> object:
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

        Args:
            N/A
        Returns:
            dict[str, float]: A map from parameter names to their values.
        """
        params = self.get_model_children(Parameter)
        if params is None:
            return {}
        return {child.name: child.value for child in params}
        
class StateNode(ABC, ModelNode):
    """
    Model leaf nodes that hold some sort of data that calculation nodes use.
    
    In probabilistic graphical modeling, these are either clamped, constant, or
    observed values for parameters or data.
    """

    def __init__(self) -> None:
        """
        Initialize a StateNode object.
        
        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__()

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Update behaviors are defined in the subclass implementation.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            N/A
        """
        pass

class Parameter(StateNode):
    """
    A subtype of a StateNode, that is a parameter for the model.
    A parameter typically holds a numerical value that defines some sort of 
    prior distribution, or a value that defines behavior of 
    transition matrices, etc.
    """
    
    def __init__(self, name: str, value: object) -> None:
        """
        A parameter is defined by its name and value.

        Args:
            name (str): Name. ie, "u" for red -> green transition probability 
                        for SNPs.
            value (object): Value for the parameter. Ie, for "u", valid values 
                            would be numbers from 0 to 1.
        Returns:    
            N/A
        """
        super().__init__()
        
        self.name: str = name
        self.value: object = value
    
    def update(self, value: object) -> None:
        """
        After changing the parameter, things that rely on the parameter for 
        their own computations need to be updated to reflect the change.
        
        Args:
            value (object): A new value for the parameter.
        Returns:
            N/A
        """
        self.value = value
        parents: list[ModelNode] = self.get_model_parents()
        
        if parents is not None:
            for par in parents:
                par.update() 
    
    def get_name(self) -> str:
        """
        Get the name of the parameter.

        Args:
            N/A
        Returns:
            str: The parameter name
        """
        return self.name
    
    def get_value(self) -> object:
        """
        Get the value of the parameter

        Args:
            N/A
        Returns:
            object: Some value.
        """
        return self.value
            
class Accumulator(StateNode):
    """
    Class that accumulates data from computations made across the model.
    """
    def __init__(self, name: str, data_structure: object) -> None:
        """
        Accumulators are defined by name and the data they store.

        Args:
            name (str): Label for the accumulator
            data_structure (object): The data store.
        Returns:
            N/A
        """
        super().__init__()
        self.data = data_structure
        self.name: str = name
    
    def update(self) -> None:
        """
        Update behaviors are defined in the subclass implementation.
        
        Args:
            N/A
        Returns:
            N/A
        """
        pass
    
    def get_data(self) -> object:
        """
        Grab the data stored in this accumulator.

        Args:   
            N/A
        Returns:
            object: The data store.
        """
        return self.data

class NetworkNode(ABC, ModelNode, Node):
    """
    Class that handles common functionality of all network nodes
    and all the height/branch length hookups.
    """

    def __init__(self, branch: Any = None) -> None:
        """
        Initialize a NetworkNode object.

        Args:
            branch (Any, optional): A BranchNode. Defaults to None.
        """
        super(NetworkNode, self).__init__()
        self.branches = branch
        self.network_parents: list[NetworkNode] = None
        self.network_children: list[NetworkNode] = None

    def get_parent_branches(self) -> dict[NetworkNode, list]:
        """
        Get the branches that connect this node to its parents.

        Args:
            N/A
        Returns:
            dict[NetworkNode, list]: A dictionary of parent nodes to
                                     incoming branches.
        """
        if self.network_parents is None:
            return None
        else:
            par_branches = {}
            for par in self.network_parents:
                par_branches[par] = [branch for branch in par.get_child_branches() if branch.dest() == self]
            return par_branches
    
    def get_child_branches(self) -> dict:
        """
        Get the branches that connect this node to its children.
        
        Args:
            N/A
        Returns:
            dict: A dict of branches that connect this node to its children.   
        """
        if self.branches is None:
            self.branches = {}
            all_branches = [child for child in self.get_model_parents(BranchNode)]
            for branch in all_branches:
                self.branches[branch.dest()] = branch
        return self.branches

    def add_child(self, model_node: ModelNode) -> None:
        """
        Adds a successor to this node.

        Args:
            model_node (ModelNode): A ModelNode to add as a child.
        Returns:    
            N/A
        """
        if self.children is None:
            self.children = [model_node]
        else:
            self.children.append(model_node)

        if isinstance(model_node, NetworkNode):
            if self.network_parents is None:
                self.network_parents = [model_node]
            else:
                self.network_parents.append(model_node)

    def remove_child(self, model_node: ModelNode) -> None:
        """
        Removes a child from this node.

        Args:
            model_node (ModelNode): A ModelNode to remove as a child.
        Returns:
            N/A
        """
        if self.children is not None and model_node in self.children:
            self.children.remove(model_node)
            if self.network_parents is not None and model_node in self.network_parents:
                self.network_parents.remove(model_node)
                if len(self.network_parents) == 0:
                    self.network_parents = None

    def add_parent(self, model_node: ModelNode) -> None:    
        """
        Adds a predecessor to this node.

        Args:
            model_node (ModelNode): A ModelNode to add as a parent.
        Returns:
            N/A
        """
        if self.parents is None:
            self.parents = [model_node]
        else:
            self.parents.append(model_node)

        if isinstance(model_node, NetworkNode):
            if self.network_children is None:
                self.network_children = [model_node]
            else:
                self.network_children.append(model_node)

    def remove_parent(self, model_node: ModelNode) -> None:
        """
        Removes a predecessor to this node.

        Args:
            model_node (ModelNode): A ModelNode to remove as a parent.
        Returns:
            N/A
        """
        if self.parents is not None and model_node in self.parents:
            self.parents.remove(model_node)
            if self.network_children is not None and model_node in self.network_children:
                self.network_children.remove(model_node)

    @abstractmethod
    def node_move_bounds(self) -> tuple[float, float]:
        """
        Get the bounds for the move on this node.
        
        Args:
            N/A
        Returns:
            tuple[float, float]: The lower and upper bounds for the move.
        """
        pass

    def get_parent(self, return_all: bool = False) -> Any:
        """
        Get the parent(s) node of this node.

        Args:
            return_all (bool, optional): If True, return all parents. Defaults to False.

        Returns:
            NetworkNode | list[NetworkNode]: The parent(s).
        """
        if return_all:
            return self.parents
        else:
            if self.parents is not None:
                return self.parents[0]
            else:
                return None

    def get_children(self) -> list[NetworkNode]:
        """
        Get the children of this node.

        Args:
            N/A
        Returns:
            list[NetworkNode]: The children of this node.
        """
        return self.children

class BranchNode(ABC, ModelNode):
    """
    A branch node is a node that represents a branch in a phylogenetic network.
    """
    def __init__(self, vector_index: int, branch_length: float) -> None:
        """
        Initialize a BranchNode object.
    
        Args:
            vector_index (int): index into the TreeHeights vector
            branch_length (float): The length of the branch
        Returns:
            N/A
        """
        super().__init__()
        
        self.index: int = vector_index
        self.branch_length: float = branch_length
        self.net_parent: NetworkNode = None
        self.net_child: NetworkNode = None
        self.gamma: float = None

    def switch_index(self, new_index: int) -> None:
        """
        Change the lookup index of this branch in the TreeHeight node

        Args:
            new_index (int): a new index
        Returns:
            N/A
        """
        self.index = new_index

    def get_index(self) -> int:
        """
        Gets the index of this branch in the TreeHeight vector
        
        Args:
            N/A
        Returns:
            int: The index into the TreeHeight vector
        """
        return self.index

    def set_net_parent(self, parent: NetworkNode) -> None:
        """
        Set the network parent 

        Args:
            parent (NetworkNode): the source of this branch
        Returns:
            N/A
        """
        self.net_parent = parent
    
    def set_net_child(self, child: NetworkNode) -> None:
        """
        Set the network child

        Args:
            child (NetworkNode): the destination of this branch
        Returns:
            N/A
        """
        self.net_child = child
    
    def src(self) -> NetworkNode:
        """
        Get the source of this branch

        Args:
            N/A
        Returns:
            NetworkNode: The source of this branch
        """
        return self.net_parent
    
    def dest(self) -> NetworkNode:
        """
        Get the destination of this branch

        Args:
            N/A
        Returns:
            NetworkNode: The destination of this branch
        """
        return self.net_child

    def inheritance_probability(self) -> float:
        """
        Return the gamma rate/ inheritance probability for a branch.

        Args:
            N/A
        Returns:
            float: A number from [0,1], -1 if no inheritance probability is attached
        """
        if self.gamma is None:
            warnings.warn("An inheritance probability is not available for this node")
            return -1
        return self.gamma
    
    def set_inheritance_probability(self, new_gamma: float) -> None:
        """
        Set the inheritance probability for a branch.

        Args:
            new_gamma (float): A number from [0,1]
        Returns:
            N/A
        """
        self.gamma = new_gamma
    
class BranchLengthNode(BranchNode, CalculationNode):
    """
    A calculation node that uses the substitution model to calculate the
    transition matrix Pij
    """

    def __init__(self, vector_index: int, branch_length: float) -> None:
        """
        Initialize a BranchLengthNode object.

        Args:
            vector_index (int): index into the TreeHeights vector
            branch_length (float): The length of the branch
        """
        super().__init__(vector_index, branch_length)
        self.as_height = True
        self.updated = False

    def update(self, new_bl: float) -> None:
        """
        Update the branch length

        Args:
            new_bl (float): The new branch length
        Returns:
            N/A
        """
        self.branch_length = new_bl
        self.upstream()

    def get(self) -> float:
        """
        Get the branch length

        Args:
            N/A
        Returns:
            float: the branch length
        """
        if self.updated:
            return self.calc()
        else:
            return self.cached

    def calc(self) -> float:
        """
        Calculate the branch length

        Args:
            N/A
        Returns:
            float: the branch length
        """
        self.cached = self.branch_length
        self.updated = False
        return self.branch_length
    
    def sim(self) -> None:
        """Simulation not implemented for BranchLengthNode."""
        pass

class TreeHeights(StateNode):
    """
    State node that holds the node heights/branch lengths
    """

    def __init__(self, node_height_vec: list = None) -> None:
        """
        Initialize a TreeHeights object.

        Args:
            node_height_vec (list, optional): A list of node heights. Defaults to None.
        """
        super().__init__()
        self.heights = node_height_vec

    def update(self, new_vector: list) -> None:
        """
        Update the heights vector

        Args:
            new_vector (list): A new vector of heights
        Returns:
            N/A
        """
        if self.heights is None:
            self.heights = new_vector
            children = self.get_model_children()
            if children:
                for branch_node in children:
                    branch_node.update(self.heights[branch_node.get_index()])
        else:
            children = self.get_model_children()
            if children:
                for branch_node in children:
                    if new_vector[branch_node.get_index()] != self.heights[branch_node.get_index()]:
                        branch_node.update(new_vector[branch_node.get_index()])
            self.heights = new_vector

    def singular_update(self, index: int, value: float) -> None:
        """
        Make an update to a single height/length in the vector

        Args:
            index (int): index into the heights/lengths vector
            value (float): The new height/length to replace the old one
        Returns:
            N/A
        """
        children = self.get_model_children()
        if children:
            for branch_node in children:
                if branch_node.get_index() == index:
                    branch_node.update(value)

    def get_heights(self) -> list:
        return self.heights

class SubstitutionModelParams(StateNode):
    """
    A state node that holds the parameters for a substitution model.
    """
    def __init__(self, freq: np.array, trans: np.array) -> None:
        """
        Initialize a SubstitutionModelParams object.

        Args:
            freq (np.array): The base frequencies for the model.
            trans (np.array): The transition matrix for the model.
        Returns:
            N/A
        """
        super().__init__()
        self.base_freqs = freq
        self.transitions = trans

    def update(self, new_freqs: np.array = None, new_trans: np.array = None) -> None:
        """
        Update the base frequencies and/or transitions with new values.

        Args:
            new_freqs (np.array, optional): The new base frequencies. Defaults to None.
            new_trans (np.array, optional): The new transition values. Defaults to None.
        Returns:
            N/A
        """
        submodel_node = self.get_model_children()[0]

        if new_freqs is None and new_trans is None:
            raise ModelError("Nonsensical update")
        elif new_freqs is not None and new_trans is not None:
            submodel_node.update(self.new_submodel(new_freqs, new_trans))
        elif new_freqs is not None:
            submodel_node.update(self.new_submodel(new_freqs))
        else:
            submodel_node.update(self.new_submodel(new_trans=new_trans))

    def new_submodel(self, new_freqs: np.array = None, new_trans: np.array = None) -> GTR:
        """
        Given a change in transitions and/or base_frequencies, determines the proper subclass of GTR
        to return.
        
        Args:
            new_freqs (np.array, optional): The new base frequencies. Defaults to None.
            new_trans (np.array, optional): The new transition values. Defaults to None.
        Returns:
            GTR: A new instance of a GTR model.
        """
        if new_freqs is None:
            proposed_freqs = self.base_freqs
        else:
            proposed_freqs = new_freqs

        if new_trans is None:
            proposed_trans = self.transitions
        else:
            proposed_trans = new_trans

        if np.array_equal(proposed_freqs, np.array([.25, .25, .25, .25])) and np.array_equal(proposed_trans, np.ones(6)):
            return JC()
        elif np.array_equal(proposed_freqs, np.array([.25, .25, .25, .25])) \
                and (proposed_trans[1] == proposed_trans[4]) \
                and (proposed_trans[0] == proposed_trans[2] == proposed_trans[3] == proposed_trans[5]) \
                and (proposed_trans[0] + proposed_trans[1] == 1):
            return K80(proposed_trans[0], proposed_trans[1])
        else:
            return GTR(proposed_freqs, proposed_trans)

class SubstitutionModel(CalculationNode):
    """
    Substitution model transition matrix calculation node.
    """
    def __init__(self, submodel: GTR) -> None:
        """
        Deterministic node that is often hooked up to transition, transversion,
        and base frequency parameters.

        Args:
            submodel (GTR): Any time reversible substitution model.
        Returns:
            N/A
        """
        super().__init__()
        self.sub: GTR = submodel

    def update(self, new_sub_model: GTR) -> None:
        """
        Change the substitution model being used.

        Args:
            new_sub_model (GTR): The new type of substitution model to use.
        Returns:
            N/A
        """
        self.sub = new_sub_model
        self.upstream()

    def get(self) -> GTR:
        """
        Based on the associated parameters, initialize a substitution model.

        Args:
            N/A
        Returns:
            GTR: The substitution model with the parameters values in the model 
                 graph.
        """
        if self.dirty:
            return self.calc()
        else:
            return self.cached

    def calc(self) -> GTR:
        """
        calculate the substitution model 

        Args:
            N/A
        Returns:
            GTR: the substitution model
        """
        self.dirty = False
        
        param_dict: dict[str, object] = {}
        params: list[Parameter] = self.get_model_children(Parameter)
        if params:
            for param_node in params:
                param_dict[param_node.name] = param_node.value
        
        self.sub.set_params(param_dict)
        self.cached = self.sub
        return self.sub
    
    def sim(self) -> None:
        """Simulation not implemented for SubstitutionModel."""
        pass

    def get_submodel(self) -> GTR:
        """
        Get the substitution model

        Args:
            N/A
        Returns:
            GTR: the substitution model 
        """
        return self.sub

class ExtantSpecies(StateNode):
    """
    Node that links network leaf nodes to their MSA data.
    """

    def __init__(self, name: str, sequences: list[DataSequence]) -> None:
        """
        Link a taxon name to its set of data sequences.

        Args:
            name (str): Taxon label.
            sequences (list[DataSequence]): list of data sequences associated with 
                                        this taxa.
        Returns:
            N/A
        """
        super().__init__()
        self.name: str = name
        self.label: str = name
        self.seqs: list[DataSequence] = sequences

    def update(self, new_sequences: list = None, new_name: str = None) -> None:
        """
        Update the extant species node with new data.

        Args:
            new_sequence (list): list of data sequences associated with this taxa.
            new_name (str): name of the taxa.
        Returns:
            N/A
        """
        if new_sequences is not None:
            self.seqs = new_sequences
        if new_name is not None:
            self.name = new_name
            self.label = new_name
        parents = self.get_model_parents()
        if parents:
            parents[0].update(self.seqs, self.label)

    def seq_len(self) -> int:
        """
        Get the sequence length of all sequences associated with this taxon.

        Args:
            N/A
        Returns:
            int: Length of data sequence.
        """
        return len(self.seqs[0].get_seq())

    def get_seqs(self) -> list[DataSequence]:
        """
        Get the list of sequence records associated with this taxon.

        Args:
            N/A
        Returns:
            list[DataSequence]: List of sequence records.
        """
        return self.seqs

    def get_name(self) -> str:
        """
        Get the taxon label.

        Args:
            N/A
        Returns:
            str: Taxon label.
        """
        return self.name

