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

from .Visitor import Visitor

# Relative imports
from .GTR import GTR, JC, K80
from .Network import Network, Edge, Node
from .MSA import *
from .Phylo import Branch

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
        self.network : Network = None
        self.nodetypes = {"leaf": [], "internal": [], "reticulation": [], "root": []}
        self.network_node_map: dict[Node, ModelNode] = {}
        self.seed = random.randint(0, 1000)
        self.rng: np.random.Generator = np.random.default_rng(self.seed)
        self.summary_str = ""
        self.root = None
  
    
    def get_root(self) -> ModelNode:
        """
        Returns the root node of the model.
        """
        return self.root

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


class ModelNode(ABC):
    """
    Class that defines the graphical structure and shared interactions between
    any node in the Model.
    """

    def __init__(self, 
                 children: list = None, 
                 parents: list = None) -> None:
        """
        Initialize a ModelNode object.

        Args:
            children (list, optional): Children of this node. Defaults to None.
            parents (list, optional): Parents of this node. Defaults to None.
            node_type (str, optional): A string that describes the type of node.
        """
        self.children: list[ModelNode] = children
        self.parents: list[ModelNode] = parents

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
        Removes a child to this node.

        Args:
            model_node (ModelNode): A ModelNode to remove as a child.
        Returns:
            N/A
        """
        if self.children is not None and model_node in self.children:
            self.children.remove(model_node)

    def remove_parent(self, model_node: ModelNode) -> None:
        """
        Removes a parent to this node.

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

    def get_node_type(self) -> str:
        """
        Get the type of this node.
        """
        return self.node_type

        
    def accept(self, visitor: Visitor) -> Any:
        """
        Accept a visitor and return the result of the visit.
        
        Args:
            visitor (ModelVisitor): A visitor to visit this node.
        Returns:
            Any: The result of the visit.
        """
        return visitor.visit(self)

############################
#### COMMON MODEL NODES ####
############################

class LeafNode(ModelNode):
    """
    A leaf node in the model graph.
    """
    def __init__(self, name: str, branch_length: Branch, data: list[DataSequence] = None, samples : int = 1) -> None:
        """
        Initialize a LeafNode object.
        
        Args:
            name (str): The name of this leaf node.
        """
        super().__init__()
        self.name : str = name
        self.node_type : str = "leaf"
        self.branch_info : Branch = branch_length
        self.data : list[DataSequence] = data    
        self.samples : int = samples 
   
    
    def get_samples(self) -> int:
        """
        Returns the number of samples for this leaf node.
        """
        return self.samples
    
    def get_name(self) -> str:
        """
        Returns the name of this leaf node.
        """
        return self.name
    
    def branch(self) -> Branch:
        """
        Returns information about the branch associated with this leaf node.
        Since it is a leaf node, there must be exactly one network parent and thus only one branch

        Args:
            N/A
        Returns:
            Branch: A Branch object containing information about the branch.
        """
        return self.branch_info
    
    def set_branch(self, branch_length: float) -> None:
        """
        Set the branch length of the branch associated with this leaf node.
        
        Args:
            branch_length (float): The new branch length.
        Returns:
            N/A
        """
        self.branch_info = Branch(branch_length)
    
    def set_data(self, data: list[DataSequence]) -> None:
        """
        Set the data for this leaf node.
        """
        self.data = data

class InternalNode(ModelNode):  
    """
    An internal node in the model graph.
    """
    def __init__(self, name: str, branch_length: Branch, disjoint_subnets: bool) -> None:
        """
        Initialize an InternalNode object.
        
        Args:
            branch_length (Branch, optional): The branch associated with this internal node. Defaults to a new Branch object.
        Returns:
            N/A
        """
        super().__init__()
        self.node_type = "internal"
        self.name : str = name
        self.branch_info : Branch = branch_length
        self.disjoint_subnets : bool = disjoint_subnets
    
    def branch(self) -> Branch:
        """
        Returns information about the branch associated with this internal node.
        Since it is a internal node, there must be exactly one network parent and thus only one branch

        Args:
            N/A
        Returns:
            Branch: A Branch object containing information about the branch.
        """
        return self.branch_info
    
    def set_branch(self, branch_length: float) -> None:
        """
        Set the branch length of the branch associated with this leaf node.
        
        Args:
            branch_length (float): The new branch length.
        Returns:
            N/A
        """
        self.branch_info.length = branch_length

    def get_name(self) -> str:
        """
        Returns the name of this leaf node.
        """
        return self.name

class ReticulationNode(ModelNode):
    """
    A reticulation node in the model graph.
    """
    def __init__(self, name: str, branch_1: Branch, branch_2: Branch) -> None:
        """
        Initialize a ReticulationNode object.
        
        Args:
            branch_1 (Branch, optional): The first branch. Defaults to 0.5.
            branch_2 (Branch, optional): The second branch. Defaults to 0.5.            
        Returns:
            N/A
        """
        super().__init__()
        self.node_type = "reticulation"
        self.name : str = name
        self.branch_info : tuple[Branch, Branch] = (branch_1, branch_2)
    
    def get_name(self) -> str:
        """
        Returns the name of this leaf node.
        """
        return self.name

    def branches(self) -> tuple[Branch, Branch]:
        """
        Returns information about the branch associated with this reticulation node.
        Since it is a reticulation node, there must be exactly two network parents and thus two branches

        Args:
            N/A
        Returns:
            tuple[Branch, Branch]: A tuple of Branch objects containing information about the branches.
        """
        return self.branch_info
    
    def set_branch(self, parent_id: str, branch_length: float = None, inheritance_probability: float = None) -> None:
        """
        Set the branch length and inheritance probability of the branch associated with this reticulation node.

        Args:
            parent_id (str): The id of the parent node.
            branch_length (float, optional): The new branch length. Defaults to None.
            inheritance_probability (float, optional): The new inheritance probability. Defaults to None.
        Returns:
            N/A
        """

        if parent_id == self.branch_info[0].parent_id:
            if branch_length is not None:
                self.branch_info[0].length = branch_length
            if inheritance_probability is not None:
                self.branch_info[0].inheritance_probability = inheritance_probability
        elif parent_id == self.branch_info[1].parent_id:
            if branch_length is not None:
                self.branch_info[1].length = branch_length
            if inheritance_probability is not None:
                self.branch_info[1].inheritance_probability = inheritance_probability
        else:
            raise ValueError(f"Parent id {parent_id} does not match any of the parent ids of the branches")

class RootNode(ModelNode):
    """
    A root node in the model graph.
    """
    def __init__(self, name: str) -> None:
        """
        Initialize a RootNode object.
        
        Args:
            N/A
        """
        super().__init__()
        self.node_type = "root" 
        self.name : str = name
    
    def get_name(self) -> str:
        """
        Returns the name of this leaf node.
        """
        return self.name

class RootAggregatorNode(ModelNode):
    """
    A root aggregator node in the model graph.
    """
    def __init__(self) -> None:
        """
        Initialize a RootAggregatorNode object.
        
        Args:
            N/A
        """
        super().__init__()
        self.result = None
        self.name : str = "root_aggregator"
        self.node_type = "root_aggregator"
    
    def get_name(self) -> str:
        """
        Returns the name of this leaf node.
        """
        return self.name
    
    def get_result(self) -> Any:
        """
        Returns the result of this root aggregator node.
        """
        return self.result
