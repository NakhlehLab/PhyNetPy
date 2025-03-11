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
Module that contains all functionality pertaining to Phylogenetic networks
(rooted and directed, or unrooted and undirected) and population genetics 
graphs. 

Release Version: 2.0.0

Author: Mark Kessler
"""

from __future__ import annotations
from typing import Any, Callable, Union
from Network import Network
from collections import defaultdict, deque
import copy
import math
import warnings
import numpy as np
import networkx as nx
from functools import singledispatchmethod
from DataStructures import DataSequence
from PhyloNet import run
from IO import *
import tempfile


"""START FILE"""

#########################
#### EXCEPTION CLASS #### 
#########################   
    
class GeneTreeError(Exception):
    """
    This exception is raised when an operation/initialization on/of a set of 
    gene trees fails.
    """
    def __init__(self, message : str = "Gene Tree Module Error") -> None:
        """
        Initialize a GeneTreeError instance.

        Args:
            message (str, optional): Error message. Defaults to "Gene Tree 
                                     Module Error".
        Returns:
            N/A
        """
        super().__init__(message)
        self.message = message

class NetworkError(Exception):
    """
    This exception is raised when a network is malformed, 
    or if a network operation fails.
    """
    def __init__(self, message : str = "Error operating on a Graph or Network"):
        """
        Initialize with an error message that will print upon the exception 
        being raised.

        Args:
            message (str, optional): Error message. Defaults to 
                                    "Error operating on a Graph or Network".
        Returns:
            N/A
        """
        self.message = message
        super().__init__(self.message)

class NodeError(Exception):
    """
    This exception is raised when a Node operation fails.
    """
    def __init__(self, message : str = "Error in Node Class"):
        """
        Initialize with an error message that will print upon the exception 
        being raised.

        Args:
            message (str, optional): Error message. Defaults to 
                                     "Error in Node Class".
        Returns:
            N/A
        """
        super().__init__(message)
        
class EdgeError(Exception):
    """
    This exception is raised when an Edge operation fails.
    """
    def __init__(self, message : str = "Error in Edge Class"):
        """
        Initialize with an error message that will print upon the exception 
        being raised.

        Args:
            message (str, optional): Error message. Defaults to 
                                     "Error in Edge Class".
        Returns:
            N/A
        """
        super().__init__(message)
        
##########################
#### HELPER FUNCTIONS #### 
##########################
  
def phynetpy_naming(taxa_name : str) -> str:
    """
    The default method for sorting taxa labels into groups

    Raises:
        GeneTreeError: if there is a problem applying the naming rule
    
    Args:
        taxa_name (str): a taxa label from a nexus file
    
    Returns:
        str: a string that is the key for this label
    """
    if not taxa_name[0:2].isnumeric():
        raise GeneTreeError("Error Applying PhyNetPy Naming Rule: \
                             first 2 digits is not numerical")
    
    if taxa_name[2].isalpha():
        return taxa_name[2].upper()
    else:
        raise GeneTreeError("Error Applying PhyNetPy Naming Rule: \
                             3rd position is not an a-z character")

def __random_object(mylist : list[Any], rng : np.random.Generator) -> object:
    """
    Select a random item from a list using an rng object 
    (for testing consistency and debugging purposes)

    Args:
        mylist (list[Any]): a list of any type
        rng (np.random.Generator) : the result of a .default_rng(seed) call

    Returns:
        object : an item from mylist
    """
    rand_index : int = rng.integers(0, len(mylist)) # type: ignore
    return mylist[rand_index]

def __dict_merge(a : dict[Any, Any], b : dict[Any, Any]) -> dict[Any, Any]:
    """
    Essentially the Union of two dictionaries, where if a key is in both 
    dictionaries, the 2 values are placed in a list -- [a value, b value].

    Args:
        a (dict[Any, Any]): A dictionary
        b (dict[Any, Any]): A dictionary

    Returns:
        dict[Any, Any]: A dictionary that contains all the unique keys of 'a' and 'b', 
              and where there is overlap, puts each value into a list.
    """
    merged : dict[Any, Any] = dict()
    
    for key in set(a.keys()).union(set(b.keys())):
        if key in a.keys() and key in b.keys():
            result = [a[key]]
            result.extend([b[key]])
            merged[key] = result
        elif key in a.keys():
            merged[key] = [a[key]]
        else:
            merged[key] = [b[key]]
    
    return merged
  
def print_graph(g : Union[Graph, Network]) -> None:
    """
    For each node in g's V set (node set), print out the node's information and
    attributes.

    Args:
        g (Union[Graph, Network]): A Graph or Network
    Returns:
        N/A
    """
    
    for node in g.V():
        print(node.to_string())

def pretty_print_edges(g : Union[Graph, Network]) -> None:
    """
    Prints the two node names as a tuple for all the edges in g's 
    E set (edge set). 

    Args:
        g (Union[Graph, Network]): A Graph or Network
    Returns:
        N/A
    """
    print("------------------")
    print("Printing edges...")
    for edge in g.E():
        print(edge.to_names())
    print("------------------") 
    
####################
#### GENE TREES ####
####################

class GeneTrees:
    """
    A container for a set of networks that are binary and represent a 
    gene tree.
    """
    
    def __init__(self, 
                 gene_tree_list : list[Network] | None = None, 
                 naming_rule : Callable[..., Any] = phynetpy_naming) -> None:
        """
        Wrapper class for a set of networks that represent gene trees

        Args:
            gene_tree_list (list[Network], optional): A list of networks, 
                                                      of the binary tree 
                                                      variety. Defaults to None.
            naming_rule (Callable[..., Any], optional): A function 
                                                        f : str -> str. 
                                                        Defaults to 
                                                        phynetpy_naming.
        
        Returns: N/A
        """
        
        self.trees : set[Network] = set()
        self.taxa_names : set[str] = set()
        self.naming_rule : Callable[..., Any] = naming_rule
        
        if gene_tree_list is not None:
            for tree in gene_tree_list:
                self.add(tree)
        
    def add(self, tree : Network) -> None:
        """
        Add a gene tree to the collection. Any new gene labels that belong to
        this tree will also be added to the collection of all 
        gene tree leaf labels.

        Args:
            tree (Network): A network that is a tree, must be binary.
        
        Returns:
            None
        """

        self.trees.add(tree)
        
        for leaf in tree.get_leaves():
            self.taxa_names.add(leaf.label)
        
    def mp_allop_map(self) -> dict[str, list[str]]:
        """
        Create a subgenome mapping from the stored set of gene trees

        Args:
            N/A
        Returns:
            dict[str, list[str]]: subgenome mapping
        """
        subgenome_map : dict[str, list[str]] = {}
        if len(self.taxa_names) != 0:
            for taxa_name in self.taxa_names:
                key = self.naming_rule(taxa_name)
                if key in subgenome_map.keys(): 
                    subgenome_map[key].append(taxa_name)
                else:
                    subgenome_map[key] = [taxa_name]
        return subgenome_map

class Node:
    """
    Node class that provides support for managing network constructs like 
    reticulation nodes and other phylogenetic attributes.
    """

    def __init__(self, 
                 name : str, 
                 is_reticulation : bool = False, 
                 attr : dict[Union[str, Any], Any] = dict(),
                 seq : Union[DataSequence, None] = None,
                 t : Union[float, None] = None) -> None:
        """
        Initialize a node with a name, attribute mapping, and a hybrid flag.

        Args:
            name (str): A Node label.
            
            is_reticulation (bool, optional): Flag that marks a node as a 
                                              reticulation node if set to True. 
                                              Defaults to False.
            
            attr (dict[Union[str, Any], Any], optional): Fill a mapping with any 
                                                         other user defined 
                                                         values. Defaults to an 
                                                         empty dictionary.
        Returns:
            N/A
        """
        
        self.__attributes : dict[Union[str, Any], Any] = attr
        self.__is_retic : bool = is_reticulation
        self.__name : str = name
        self.__seq : Union[DataSequence, None] = seq
        self.__t : Union[float, None] = t
        self.__is_dirty : bool = False
    
    def get_attributes(self) -> dict[Union[str, Any], Any]:
        """
        Retrieve the attribute mapping.

        Args:
            N/A
        Returns:
            dict[Union[str, Any], Any]: A storage of key value pairs that 
                                        correspond to user defined node 
                                        attributes. Typically keys are    
                                        string labels, but they can be anything.
        """
        return self.__attributes
    
    def set_attributes(self, new_attr : dict[Union[str, Any], Any]) -> None:
        """
        Set a node's attributes to a mapping of key labels to attribute values.

        Args:
            new_attr (dict[Union[str, Any], Any]): Attribute storage mapping.
            
        Returns:
            N/A
        """
        self.__attributes = new_attr
    
    def get_time(self) -> float:
        """
        Get the speciation time for this node.
        
        Closer to 0 implies a time closer to the origin (the root). A larger 
        time implies a time closer to the present (leaves). 
        
        Args:
            N/A
        Returns:
            float: Speciation time, typically in coalescent units.
        """
        if self.__t is None:
            raise NodeError("No time has been set for this node!")
        return self.__t
    
    def set_time(self, new_t : float) -> None:
        """
        Set the speciation time for this node. The arg 't' must be a 
        non-negative number.

        Args:
            new_t (float): The new speciation/hybridization time for this node.
        
        Returns:
            N/A
        """
        if new_t >= 0:
            self.__t = new_t 
        else:
            raise NodeError("Please set speciation time, t, to a non-negative\
                             number!")
        
    def to_string(self) -> str:
        """
        Create a description of a node and summarize its attributes.

        Args:
            N/A
        Returns:
            str: A string description of the node.
        """
        my_str = "Node " + str(self.__name) + ": "
        if self.__t is not None:
            my_str += "t = " + str(round(self.__t, 4)) + " "
        my_str += " is a reticulation node? " + str(self.__is_retic)
        my_str += " has attributes: " + str(self.__attributes)

        return my_str

    @property
    def label(self) -> str:
        """
        Returns the name of the node

        Args:
            N/A
        Returns:
            str: Node label.
        """
        return self.__name

    def set_name(self, new_name : str) -> None:
        """
        Sets the name of the node to new_name.
    
        Args:
            new_name (str): A new string label for this node.
        Returns:
            N/A
        """
        self.__name = new_name
        self.__is_dirty = True

    def set_is_reticulation(self, new_is_retic : bool) -> None:
        """
        Sets whether a node is a reticulation Node (or not).

        Args:
            new_is_retic (bool): Hybrid flag. True if this node is a 
                                 reticulation node, false otherwise
        
        Returns:
            N/A
        """
        self.__is_retic = new_is_retic

    def is_reticulation(self) -> bool:
        """
        Retrieves whether a node is a reticulation Node (or not)

        Args:
            N/A
        Returns:
            bool: True, if this node is a reticulation. False otherwise.
        """
        return self.__is_retic

    def add_attribute(self, 
                      key : Any, 
                      value : Any, 
                      append : bool = False) -> None:
        """
        Put a key and value pair into the node attribute dictionary.

        If the key is already present, it will overwrite the old value.
        
        Args:
            key (Any): Attribute key.
            value (Any): Attribute value for the key.
            append (bool, optional): If True, appends the given value to the 
                                     existing value for the key.   
                                     If false, simply replaces. 
                                     Defaults to False.
        Returns:
            N/A
        """
        
        if append:
            if key in self.__attributes.keys():
                content = self.__attributes[key]
        
                if type(content) is dict[Any, Any]:
                    content = __dict_merge(content, value)
                    self.__attributes[key] = content
                elif type(content) is list[Any]:
                    content.extend(value)
                    self.__attributes[key] = content
        else:
            self.__attributes[key] = value

    def attribute_value(self, key : Any) -> object:
        """
        If key is a key in the attributes mapping, then
        its value will be returned.

        Otherwise, returns None.

        Args:
            key (Any): A lookup key.

        Returns:
            object: The value of key, if key is present.
        """
        if key in self.__attributes.keys():
            return self.__attributes[key]
        else:
            return None
    
    def set_seq(self, new_sequence : DataSequence) -> None:
        """
        Associate a data sequence with this node, if this node is a leaf in a 
        network.

        Args:
            new_sequence (DataSequence): A data sequence wrapper. Grab from MSA 
                                      object upon parsing.
        
        Returns:
            N/A
        """
        
        self.__seq = new_sequence
    
    def get_seq(self) -> DataSequence:
        """
        Gets the data sequence associated with this node.

        Args:
            N/A
        Returns:
            DataSequence: Data sequence wrapper.
        """
        if self.__seq is None:
            raise NodeError("No sequence record has been associated with this\
                             node!")
        return self.__seq
    
    def copy(self) -> Node:
        """
        Duplicate this node by copying all data into a separate Node object.
        
        Useful for crafting copies of networks without having to deep copy. 
        
        Args:
            N/A
        Returns:
            Node: An equivalent node to this node, with all the same data but 
                  technically are not "=="
        """
        dopel = Node(self.__name, self.__is_retic, self.__attributes)
        if self.__seq is not None:
            dopel.set_seq(self.__seq)
            
        if self.__t is not None:
            dopel.set_time(self.__t)
            
        dopel.__is_dirty = self.__is_dirty
        
        return dopel
    
class NodeSet:
    """
    Data structure that is in charge of managing the nodes that are in a given
    Network
    """
    
    def __init__(self, directed : bool = True) -> None:
        """
        Initialize an empty set of network nodes.

        Args:
            directed (bool, optional): Flag that indicates how to bookkeep 
                                       items. Undirected Graphs vs Directed
                                       Networks requires different treatment.
                                       Defaults to True (directed context).
        Returns:
            N/A
        """
        self.__nodes : set[Node] = set()
        self.__in_degree : dict[Node, int] = defaultdict(int)
        self.__out_degree : dict[Node, int] = defaultdict(int)
        self.__in_map : dict[Node, list[Any]] = defaultdict(list)
        self.__out_map : dict[Node, list[Any]] = defaultdict(list)
        self.__node_names : dict[str, Node] = {}
        self.__directed : bool = directed
    
    def __contains__(self, n : Node) -> bool:
        """
        Overrides the built in contains method. Can now say:
        
        n : Node
        
        if n in my_network.V()...

        Args:
            n (Node): A NodeSet contains Node objects

        Returns:
            bool: True if node is in the set, False otherwise.
        """
        return n in self.__nodes

    def add(self, *nodes : Node | list[Node]) -> None:
        """
        Add a list of nodes to the network node set.

        Args:
            node (Node | list[Node]): A list of new nodes to put in the network.
        Returns:
            N/A
        """
        for node in nodes:
            if type(node) is list[Node]:
                for n in node:
                    if n not in self.__nodes:
                        self.__nodes.add(n)
                        self.__node_names[n.label] = n
            if type(node) is Node:
                if node not in self.__nodes:
                    self.__nodes.add(node)
                    self.__node_names[node.label] = node
    
    def ready(self, edge : Union[Edge, UEdge]) -> bool:
        """
        Check if an edge is allowed to be added to the network (both nodes must
        be in the node set before an edge can be added).
        
        A undirected edge is not allowed to be inserted into a directed network,
        and a directed edge is not allowed to be inserted into an undirected 
        graph. False will be returned in such instances.
 
        Args:
            edge (Edge | UEdge): A potential new network edge.

        Returns:
            bool: True if edge can be safely added, False otherwise.
        """
        if self.__directed:
            if type(edge) is Edge:
                return edge.src in self.__nodes and edge.dest in self.__nodes
            return False
        
        if type(edge) is UEdge:
            return edge.n1 in self.__nodes and edge.n2 in self.__nodes
        return False
            
    
    def in_deg(self, node : Node) -> int:
        """
        Gets the in degree of a node in this set.
        Returns 0 if the node has no in edges, or if the node is not in the 
        node set. It is up to the user to make sure the parameter
        node is in the node set.

        Args:
            node (Node): any Node obj

        Returns:
            int: the in degree of the node
        """
        return self.__in_degree[node]

    def out_deg(self, node : Node) -> int:
        """
        Gets the out degree of a node in this set.
        Returns 0 if the node has no out edges, or if the node is not in the 
        node set. It is up to the user to make sure the parameter
        node is in the node set.

        Args:
            node (Node): any Node obj

        Returns:
            int: the out degree of the node
        """
        return self.__out_degree[node]
    
    def in_edges(self, node : Node) -> list[Any]:
        """
        Gets the in edges of a node in this set.
        Returns an empty list if the node has no in edges, or if the node 
        is not in the node set.It is up to the user to make sure the parameter
        node is in the node set.

        Args:
            node (Node): any Node obj

        Returns:
            list[Edge]: the in edges of the node
        """
        return self.__in_map[node]
    
    def out_edges(self, node : Node) -> list[Any]:
        """
        Gets the out edges of a node in this set.
        Returns an empty list if the node has no out edges, or if the node 
        is not in the node set. It is up to the user to make sure the parameter
        node is in the node set.

        Args:
            node (Node): any Node obj

        Returns:
            list[Edge]: the out edges of the node
        """
        return self.__out_map[node]
    
    
    def process(self, edge : Union[Edge, UEdge], removal : bool = False) -> None:
        """
        Keep track of network data (in/out degrees, in/out edge maps) upon 
        the addition or removal of an edge for a network.

        Args:
            edge (Union[Edge, UEdge]): The edge that is being added or removed
            removal (bool, optional): False if edge is being added, True if 
                                      edge is being removed. Defaults to False.
        Returns:
            N/A
        """
        n1 : Node
        n2 : Node
        
        if type(edge) is Edge and self.__directed:
            n1 = edge.src
            n2 = edge.dest
        elif type(edge) is UEdge and not self.__directed:
            n1 = edge.n1
            n2 = edge.n2
        else:
            raise EdgeError("Tried to process wrong type of edge!")
            
            
        if self.ready(edge):
            if not removal:
                self.__out_degree[n1] += 1
                self.__in_degree[n2] += 1
                self.__out_map[n1].append(edge)
                self.__in_map[n2].append(edge)
                if not self.__directed:
                    self.__out_degree[n2] += 1
                    self.__in_degree[n1] += 1
                    self.__out_map[n2].append(edge)
                    self.__in_map[n1].append(edge)
            else:
                self.__out_degree[n1] -= 1
                self.__in_degree[n2] -= 1
                self.__out_map[n1].remove(edge)
                self.__in_map[n2].remove(edge)
                
                if not self.__directed:
                    self.__out_degree[n2] -= 1
                    self.__in_degree[n1] -= 1
                    self.__out_map[n2].remove(edge)
                    self.__in_map[n1].remove(edge)
        else:
            raise EdgeError("Tried to add edge to the network, and the edge \
                             contains a node that is not part of the network. \
                             Please add the node first and retry!")
                
    def get_set(self) -> set[Node]:
        """
        Grab the set of nodes.
        
        Args:
            N/A
        Returns:
            set[Node]: V, the node set of a network.
        """
        return self.__nodes
    
    def remove(self, node : Node) -> None:
        """
        Remove a node from V, and update necessary mappings.

        Args:
            node (Node): Node to remove from the network.
        Returns:
            N/A
        """
        if node in self.__nodes:
            self.__nodes.remove(node)
            del self.__in_degree[node]
            del self.__out_degree[node]
            
            # There is a world where a node may not ever be a key in either one 
            # of these maps. Ie, a leaf node that is subsequently detached from
            # its parent will not be in the out_map (or a root that gets 
            # detached from all children... but this isn't likely).
            if node in self.__out_map.keys():
                del self.__out_map[node]
            if node in self.__in_map.keys():
                del self.__in_map[node] 
                 
            del self.__node_names[node.label] 
    
    def update(self, node : Node, new_name : str) -> None:
        """
        Protected method that processes updates to node labels within the 
        NodeSet.
        
        You may not set a Node's name to None.
       
        Raises:
            NodeError: If @new_name is None or if @node is not present in this 
                       NodeSet.
        Args:
            node (Node): The Node that is being renamed 
            new_name (str): The new name.
        Returns:
            N/A
        """
    
        if node not in self:
            raise NodeError(f"Error updating the name of node \
                            {node.label}, node could not be found in this \
                            NodeSet.")
        
        if node.label in self.__node_names.keys():
            del self.__node_names[node.label]
            
        node.set_name(new_name)
        self.__node_names[new_name] = node
    
    def get(self, name : str) -> Union[Node, None]:
        """
        Retrieves the Node with label 'name' if one exists in the set.

        Args:
            name (str): a Node label

        Returns:
            Union[Node, None]: Returns the Node object if one is found, None if 
                               no Node exists with label 'name'.
        """
        if name in self.__node_names.keys():
            return self.__node_names[name]
        return None

class UEdge:
    """
    Undirected dge class that is essentially a wrapper class for a set
    {a, b} where a and b are Node objects. There is no ordering/direction.
    """
    
    def __init__(self, 
                 n1 : Node,
                 n2 : Node,
                 length : float | None = None, 
                 weight : float | None = None) -> None:
        """
        Initialize an undirected Edge.

        Raises:
            EdgeError: If 'length' does not match the difference in node times
                       should both n1 and n2 have set time attributes.
                       
        Args:
            n1 (Node): A Node (designated member #1 for efficient retrieval)
            n2 (Node): A Node (designated member #2 for efficient retrieval)
            length (float, optional): _description_. Defaults to None.
            weight (float, optional): _description_. Defaults to None.

        Returns:
            N/A
        
        """
        super().__init__()

        self._n1 = n1
        self._n2 = n2
        
        if weight is not None:
            self.set_weight(weight)
        
        if length is not None:
            self.set_length(length, False)
    
    @property
    def n1(self) -> Node:
        """
        Retrieves the first node given as a parameter to this undirected edge
        
        Args:
            N/A
        Returns:
            Node: The first node given as a parameter to this undirected edge
        """
        return self._n1
    
    @property
    def n2(self) -> Node:
        """
        Retrieves the second node given as a parameter to this undirected edge
        
        Args:
            N/A
        Returns:
            Node: The second node given as a parameter to this undirected edge
        """
        return self._n2
    
    def __contains__(self, x : Node) -> bool:
        """
        An undirected edge contains two member nodes.
        
        IE.
        
        e : UEdge
        n : Node
        
        "n in e" gives True or False.
        
        Args:
            x (Node): A Node
        Returns:
            bool: True if 'x' is a member node of this UEdge, False if not.
        """
        return x == self._n1 or x == self._n2 
    
    def get_length(self) -> float:
        """
        Gets the branch length of this edge. If it is not equivalent to the
        current times of the source and destination nodes, then a warning
        is raised that the branch lengths are not equivalent, and that either
        the times are outdated or the branch length field is outdated.

        Args:
            N/A
        Returns:
            float: branch length.
        """
        n1t = self._n1.get_time() 
        n2t = self._n2.get_time()
        #if n1t is not None and n2t is not None and self.__length is not None:
        if self.__length != abs(n2t - n1t):
            warnings.warn("This edge has a length field set to a number \
                            that is not equal to the differences in time for\
                            its node members!")
        return self.__length

    def set_length(self, 
                   branch_length : float, 
                   enforce_times : bool = True) -> None:
        """
        Sets the branch length of this edge.

        Args:
            length (float): a branch length value (>0).
        Returns:
            N/A
        """
        #If the source and destination nodes already have defined times, 
        # go ahead and use them.
        # if self._n1.get_time() is not None \
        #         and self._n2.get_time() is not None:
            
        # Get difference in speciation times.
        new_len : float = abs(self._n2.get_time() - self._n1.get_time())
        matches : bool = branch_length == new_len
        #They should match!
        if enforce_times:
            if not matches:
                #and branch_length is not None:
                    
                raise EdgeError("Provided length is not equivalent to \
                                provided n1 and n2 times!")
        else:
            if not matches: 
                warnings.warn("Setting branch length of an edge to a value\
                                that is not equivalent to the difference \
                                in their set speciation times")      
        
        self.__length : float = new_len
    
    def copy(self, 
             new_n1 : Node | None = None, 
             new_n2 : Node | None = None) -> UEdge:
        """
        Make an equivalent UEdge (same as a deepcopy essentially).

        Args:
            new_n1 (Node | None, optional): Provide new nodes in
                                            case you'd like finer control 
                                            over naming. Defaults to None.
            new_n2 (Node | None, optional): Provide new nodes in
                                            case you'd like finer control 
                                            over naming. Defaults to None.

        Returns:
            UEdge: A new but functionally equivalent UEdge
        """
        # If new nodes are provided, great. If not, duplicate the current nodes
        if new_n1 is None or new_n2 is None: 
            new_edge = UEdge(self._n1.copy(), self._n2.copy())
        else:
            new_edge = UEdge(new_n1, new_n2)
       
        # Copy over the data
        new_edge.set_length(self.__length)
        new_edge.set_weight(self.__weight)
        
        return new_edge
    
    def to_directed(self, src : Node) -> Edge:
        """
        Convert this edge to a directed edge. 

        Raises: 
            EdgeError: If 'src' is not a node in this Edge.
            
        Args:
            src (Node): Must be one of the Nodes that is in this edge. Sets this
                        Node as the source node, and the other node as the 
                        destination
                                 
        Returns:
           Edge: An equivalent edge to this undirected edge, but with a 
                    direction now enforced.
        """
        if src == self._n1:
            new_edge = Edge(self._n1.copy(), self._n2.copy())
        elif src == self._n2:
            new_edge = Edge(self._n2.copy(), self._n1.copy())
        else:
            raise EdgeError("'src' parameter not one of the nodes that is a \
                             member of this edge")
        
        # Copy over the data
        new_edge.set_length(self.__length)
        new_edge.set_weight(self.__weight)
        
        return new_edge
    
    def get_weight(self) -> float:
        """
        Get the weight value for this UEdge (not! equivalent to its length in a 
        phylogenetic context, this is for any other potential use for the
        inclusion of weighting edges).

        Args:
            N/A
        Returns:
            float: The weight of this UEdge
        """
        return self.__weight
    
    def set_weight(self, new_weight : float) -> None:
        """
        Set a weight value for this UEdge (not! equivalent to its length in a 
        phylogenetic context, this is for any other potential use for the
        inclusion of weighting edges).

        Args:
            new_weight (float): A new weight value
        Returns:
            N/A
        """
        self.__weight = new_weight
    
    def to_names(self) -> tuple[str, str]:
        """
        Return this UEdge as a two-tuple of the two member nodes' labels.
        While you should not presume an order, the order will be n1 then n2 if 
        you remember which order you passed them in as during initialization.

        Args:
            N/A
        Returns:
            tuple[str, str]: a two-tuple of names, (n1 name , n2 name)
        """
        return (self._n1.label, self._n2.label)
           
class Edge:
    """
    Class for directed edges. 
    
    Instead of being a wrapper for a "set" of member nodes, we can now think 
    about an Edge as a wrapper for a tuple of member nodes (a, b), where 
    now the direction is encoded in the ordering (where a is the source, and b 
    the destination).
    """
    
    def __init__(self,
                 source : Node, 
                 destination : Node, 
                 length : float | None = None,
                 gamma : float | None = None,
                 weight : float | None = None,
                 tag : str | None = None
                 ) -> None:
        """
        An Edge has a source (parent) and a destination (child). Edges in a 
        phylogenetic context are *generally* directed.
        
        source -----> destination
        
        Raises:
            ValueError: If @gamma is not a probabilistic value between 0 and 1 
                        (inclusive, but if a hybrid edge pair has 0 and 1 then
                        those hybrid edges may as well not exist).
            EdgeError: If @length does not match the difference between 
                       @source.get_time() and @destination.get_time().

        Args:
            source (Node): The parent node.
            destination (Node): The child node.
            length (float, optional): Branch length value. Defaults to None.
            gamma (float, optional): Inheritance Probability, MUST be from 
                                     [0,1]. Defaults to None.
            weight (float, optional): Edge weight, can be any real number. 
                                      Defaults to None.
            tag (str, optional): A name tag for identifiability of hybrid edges
                                 should each have a gamma of 0.5. 
                                 Defaults to None.
        
        Returns:
            N/A
        """
        super().__init__()
        
        self._src = source
        self._dest = destination
        
        #Set all fields
        if length is not None:
            self.set_length(length, False)
        else:
            self.set_length(1, False)
        
        if gamma is not None:
            self.set_gamma(gamma)
        else:
            self.set_gamma(0.0)
       
        if tag is not None:
            self.set_tag(tag)
        else:
            self.set_tag("no tag assigned yet")
        
        if weight is not None:
            self.set_weight(weight)
        else:
            self.set_weight(0.0)
    
    @property
    def src(self) -> Node:
        """
        Get the source (parent) node.
        
        Args:
            N/A
        Returns:
            Node: source Node obj
        """
        return self._src
    
    @property
    def dest(self) -> Node:
        """
        Get the dest (child) node.
        
        Args:
            N/A
        Returns:
            Node: destination Node obj
        """
        return self._dest
    
    def set_tag(self, new_tag : str) -> None:
        """
        Set the name/identifiability tag of this edge.

        Args:
            new_tag (str): a unique string identifier.
        Returns:
            N/A
        """
        self.__tag = new_tag
        
    def get_tag(self) -> str:
        """
        Get the name/identifiability tag for this edge.
        Args:
            N/A
        Returns:
            str: The name/identifiabity tag for this edge.
        """
        return self.__tag
    
    def set_gamma(self, gamma : float) -> None:
        """
        Set the inheritance probability of this edge. Only applicable to 
        hybrid edges, but no warning will be raised if you attempt to set the 
        probability of a non-hybrid edge.

        Args:
            gamma (float): A probability (between 0 and 1 inclusive).
        Returns:
            N/A
        """
        
        if gamma < 0 or gamma > 1:
            raise ValueError("Please provide a probabilistic value for \
                                gamma (between 0 and 1, inclusive)!")
        self.__gamma = gamma
    
    def get_gamma(self) -> float:
        """
        Gets the inheritance probability for this edge.

        Args:
            N/A
        Returns:
            float: A probability (between 0 and 1).
        """
        return self.__gamma
    
    def copy(self, 
             new_src : Node | None = None, 
             new_dest : Node | None = None) -> Edge:
        """
        Craft an identical edge to this edge object, just in a new object.
        Useful in building subnetworks of a network that is in hand.
        
        Args:
            new_src (Node | None, optional): _description_. Defaults to None.
            new_dest (Node | None, optional): _description_. Defaults to None.

        Returns:
            Edge: An identical edge to this one, with respect to the data they 
                  hold.
        """
        # If new nodes are provided, great. If not, duplicate the current nodes
        new_edge : Edge
        if new_src is None or new_dest is None: # ?? why would this be the case
            new_edge = Edge(self._src.copy(), self._dest.copy())
        else:
            new_edge = Edge(new_src, new_dest)
       
        # Copy over the data
        new_edge.set_length(self.__length)
        new_edge.set_gamma(self.__gamma)
        new_edge.set_weight(self.__weight)
        
        return new_edge

    def set_length(self, 
                   branch_length : float, 
                   warn_times : bool = False,
                   enforce_times : bool = False) -> None:
        """
        Set the length of this Edge, and optionally let the user decide if they
        allow possible discrepancies between Edge lengths and the Edge's 
        src/dest .t (time) attribute values.
        
        Raises:
            EdgeError: if user chooses to enforce that the branch length and .t 
                       attributes of src and dest match, but they don't.
        Args:
            branch_length (float): The new length/weight of the Edge
            warn_times (bool, optional): Option that, if enabled, warns instead 
                                         of crashes the program in the event 
                                         that the branch_length parameter does 
                                         not match the time attributes of the 
                                         edge's src and dest Node objects.
                                         Defaults to False (will not warn).
            enforce_times (bool, optional): Option that, if enabled, will allow 
                                            this method to raise an exception if
                                            the branch_length parameter does 
                                            not match the time attributes of the 
                                            edge's src and dest Node objects.
                                            Defaults to False (will not throw).
        Returns:
            N/A

        """
        
        #If the source and destination nodes already have defined times, 
        # go ahead and use them.
        
  
        # Get difference in speciation times.
        # Children always (or should always) have a larger time 
        # since root = 0
        
        try:
            check_len = self._dest.get_time() - self._src.get_time()
        except NodeError:
            check_len = 0
            
        
        #They should match!
        if enforce_times:
            if abs(branch_length - check_len) >= 1e-5: 
                raise EdgeError("Provided length is not equivalent to \
                                provided n1 and n2 times!")
        elif warn_times:
            if branch_length != check_len:
                warnings.warn("Setting branch length of an edge to a value\
                                that is not equivalent to the difference \
                                in their set speciation times")     
        
        self.__length : float = branch_length
    
    def get_length(self) -> float:
        """
        Get the Edge length.

        Args:
            N/A
        Returns:
            float: Edge length/branch length
        """
        return self.__length
    
    def set_weight(self, new_weight : float) -> None:
        """
        Set a weight value for this Edge (not! equivalent to its length in a 
        phylogenetic context, this is for any other potential use for the
        inclusion of weighting edges).

        Args:
            new_weight (float): A new weight value
        Returns:
            N/A
        """
        self.__weight = new_weight
    
    def get_weight(self) -> float:
        """
        Get the weight value for this Edge (not! equivalent to its length in a 
        phylogenetic context, this is for any other potential use for the
        inclusion of weighting edges).

        Args:
            N/A
        Returns:
            float: The weight of this Edge
        """
        return self.__weight
    
    def to_names(self) -> tuple[str, str]:
        """
        Return this Edge as a two-tuple where the first element is the
        src Node object's label, and the second element is the dest Node
        object's label

        Args:
            N/A
        Returns:
            tuple[str, str]: a two-tuple of names, (src name , dest name)
        """
        return (self._src.label, self._dest.label)
    
class EdgeSet:
    """
    Data structure that serves the purpose of keeping track of edges that belong
    to a network. We call this set E.
    """
    
    def __init__(self, directed : bool = True) -> None:
        """
        Initialize the set of edges, E, for a network.

        Args:
            directed (bool, optional): If using for a directed Network, use 
                                       True. If using for an undirected Graph, 
                                       use False. Defaults to True.
        Returns:
            N/A
        """
        
        # Map (src, dest) tuples to a list of edges. this list will have 1 
        # element for most, but in the case of bubbles will contain 2. This 
        # exists purely to deal with bubbles 
        self.__hash : dict[tuple[Node, Node], list[Edge]] = dict()
        self.__uhash : dict[tuple[Node, Node], list[UEdge]] = dict()
        
        # Edge set, E
        self.__edges : set[Edge] = set()
        self.__uedges : set[UEdge] = set()
        self.__directed : bool = directed
    
    def __contains__(self, e : Union[Edge, UEdge]) -> bool:
        """
        An undirected edge (Edge) is in the EdgeSet if there is an equivalent 
        edge object OR if there is anUEdgeobject with equivalent node members.
        
        A directed edge (DiEdge) is in the EdgeSet only if the EdgeSet has a 
        reference to @e.

        Args:
            e (Union[Edge, UEdge]): An undirected or directed edge.

        Returns:
            bool: If the Edge @e is represented in the graph
        
        """
        if self.__directed:
            if type(e) is Edge:
                return e in self.__edges
            return False
        else:
            if type(e) is UEdge:
                return e in self.__uedges #self.__retrieve(e.n1, e.n2) != []
            return False
    
    def __add_to_hash(self, 
                      n1 : Node,
                      n2 : Node, 
                      e : Union[Edge, UEdge]):
        """
        Add an edge to the hash table for lookup
        Args:
            n1 (Node): _description_
            n2 (Node): _description_
            e (Union[Edge, UEdge]): _description_
        Returns:
            N/A
        """
        
        if type(e) is UEdge:
            if (n1, n2) in self.__uhash.keys():
                self.__uhash[(n1, n2)].append(e)
            else:
                self.__uhash[(n1, n2)] = [e]
        elif type(e) is Edge:
            if (n1, n2) in self.__hash.keys():
                self.__hash[(n1, n2)].append(e)
            else:
                self.__hash[(n1, n2)] = [e]
            
  
        
    # @singledispatchmethod
    def add(self, *edges : Union[Edge, UEdge]) -> None:
        """
        Add any number of edges to E.
        
        Raises:
            TypeError: If the edge type (undirected or directed) doesn't
                       match the designated type of network/graph associated 
                       with this edge set.

        Args:
            *edges (Edge): An amount of new edges to add to E.
        Returns:
            N/A
        """
        
        for edge in edges:
            if self.__directed and type(edge) is not Edge:
                raise TypeError("This edge set is associated with a directed \
                                graph. An undirected edge was provided.")
            
            if not self.__directed and type(edge) is Edge:
                raise TypeError("This edge set is associated with an undirected \
                                graph. A directed edge was provided.")
                
            
            if type(edge) is UEdge:
                if edge not in self.__uedges:
                    if (edge.n1, edge.n2) in self.__uhash.keys() \
                        or (edge.n2, edge.n1) in self.__uhash.keys():
                        warnings.warn("Adding duplicate edge to undirected Graph. \
                                    This function call will have no effect.")
                        return
                    
                    self.__add_to_hash(edge.n1, edge.n2, edge) 
                    self.__uedges.add(edge) 
            elif type(edge) is Edge:
                if edge not in self.__edges:
                    self.__add_to_hash(edge.src, edge.dest, edge)
                    self.__edges.add(edge)
            
    
    # @add.register
    # def _(self, edges : list) -> None: # type: ignore
    #     """
    #     Add all edges from a list of edges to E.

    #     Args:
    #         edges (list[AEdge]): A list of new edges to add to E.
        
    #     Raises:
    #         TypeError: If the edge type (undirected or directed) doesn't
    #                    match the designated type of network/graph associated 
    #                    with this edge set.
    #     """
        
    #     for edge in edges:
    #         if self.__directed and type(edge) is not Edge:
    #             raise TypeError("This edge set is associated with a directed \
    #                             graph. An undirected edge was provided.")
            
    #         if not self.__directed and type(edge) is Edge:
    #             raise TypeError("This edge set is associated with an undirected \
    #                             graph. A directed edge was provided.")
                
            
    #         if type(edge) is UEdge:
    #             if edge not in self.__uedges:
    #                 if (edge.n1, edge.n2) in self.__uhash.keys() \
    #                     or (edge.n2, edge.n1) in self.__uhash.keys():
    #                     warnings.warn("Adding duplicate edge to undirected Graph. \
    #                                 This function call will have no effect.")
    #                     return
                    
    #                 self.__add_to_hash(edge.n1, edge.n2, edge) 
    #                 self.__uedges.add(edge) 
    #         elif type(edge) is Edge:
    #             if edge not in self.__edges:
    #                 self.__add_to_hash(edge.src, edge.dest, edge)
    #                 self.__edges.add(edge)
            
            
    def remove(self, edge : Union[Edge, UEdge]) -> None:
        """
        Remove an edge from E.

        Args:
            edge (Edge): An edge that is currently in E.
        Returns:
            N/A
        """
        if edge in self.__edges:
            self.__hash[(edge.src, edge.dest)].remove(edge)
            
            # Delete the key from the hash if there is no bubble / same edge
            if self.__retrieve(edge.src, edge.dest) == []:
                del self.__hash[(edge.src, edge.dest)]
                
            self.__edges.remove(edge)
        if edge in self.__uedges:
            self.__uhash[(edge.n1, edge.n2)].remove(edge)
        
            # Delete the key from the hash if there is no bubble / same edge
            if self.__retrieve(edge.n1, edge.n2) == []:
                del self.__uhash[(edge.n1, edge.n2)]
                
            self.__uedges.remove(edge)
                
            
    def __retrieve(self, n1 : Node, n2 : Node) -> Union[list[Edge], list[UEdge]]:
        """
        Private method. If (n1, n2) is not a key in the hash map,
        then just return an empty list instead of throwing a key error.

        Args:
            n1 (Node): 1st node in the edge
            n2 (Node): 2nd node in the edge

        Returns:
            list[Union[Edge, UEdge]]: A list of edges that have n1 and n2 as their nodes
        """
        try:
            hashed = self.__hash[(n1, n2)]
            return hashed
        except KeyError:
            try:
                uhashed = self.__uhash[(n1, n2)]
                return uhashed
            except KeyError:
                return []
        
        
          
    def get(self,
            n1 : Node, 
            n2 : Node, 
            gamma : float | None = None,
            tag : str | None = None
            ) -> Union[Edge, UEdge]:
        """
        Given the nodes that make up the edge and an inheritance probability,
        get the edge in E that matches the data. Inheritance probability is only
        required for when a directed network has two edges with the same source 
        and destination nodes, and it is known that a bubble edge is being 
        looked up. Inheritance probabilities are insufficient for 
        identifiability of directed bubble edges should gamma 
        be .5, in which case please tag your edges with some sort of key. 
        
        If a bubble edge is being looked up and there is no gamma and/or tag 
        provided, then one of the two bubble edges will be returned at random.
        
        There are no identifiability problems with undirected graphs, since 
        duplicate edges are not a thing, and there exists only one edge with
        a given pair of member nodes.
        
        Raises:
            EdgeError: If there are any problems looking up the desired edge,
                       or if there is no such edge in the graph/network.
        
        Args:
            n1 (Node): If directed, this is "src" / the parent. If undirected,
                       node ordering does not matter.
            destination (Node): If directed, this is "dest" / the child. If 
                                undirected, node ordering does not matter.
            gamma (float, optional): Inheritance probability, for bubble 
                                     identifiability. Defaults to None.
            tag (str, optional): In the event that gamma is .5 AND THERE IS A 
                                 BUBBLE, then edges should be tagged with a 
                                 unique identifier string.
        Returns:
            Edge: The edge in E that matches the given data.
        """
        valid_edges = self.__retrieve(n1, n2)
        
           
        if len(valid_edges) == 0:
            raise EdgeError("Found 0 matching edges in this network or \
                                graph")
        elif len(valid_edges) == 1:
            return valid_edges[0]
        elif len(valid_edges) == 2:
            if gamma is None:
                warnings.warn("No gamma provided, but a bubble is being \
                                looked up. Returning a random bubble edge!")
                return valid_edges[0]
            else:
                assert(type(valid_edges[0]) is Edge)
                assert(type(valid_edges[1]) is Edge)
                
                if valid_edges[0].get_gamma() == gamma\
                    and valid_edges[1].get_gamma() == gamma:
                        
                    tag0 = valid_edges[0].get_tag()
                    tag1 = valid_edges[1].get_tag()
                    
                    
                    if tag0 == tag:
                        return valid_edges[0]
                    elif tag1 == tag:
                        return valid_edges[1]
                    else:
                        raise EdgeError(f"Tags of the edges \
                                            {(tag0, tag1)} do not \
                                            match passed in tag parameter \
                                            : {tag} !")
                elif valid_edges[0].get_gamma() == gamma:
                    return valid_edges[0]
                elif valid_edges[1].get_gamma() == gamma:
                    return valid_edges[1]
                else:
                    raise EdgeError("Error looking up an edge. Inheritance\
                                    probability is not a match for any Edge\
                                    in this set.")
        else:
            raise EdgeError("Found more than 2 eligible edges... something\
                            is wrong with the network topology. Bubbles \
                            can only have 2 edges.")
    
        # elif type(valid_edges) is list[UEdge]:
        #     # No use for tags and gamma here, in an undirected context there are
        #     # no duplicate edges or bubbles.
        #     valid_edges_set : set[UEdge] = set(valid_edges)
        #     other_valid_edges : set[UEdge] = set(self.__retrieve(n2, n1))
        #     valid_edges = valid_edges.union(other_valid_edges)

        #     if len(valid_edges) > 1:
        #         raise EdgeError("Found a duplicate edge in an undirected graph")
            
        #     return valid_edges
            
    def get_set(self) -> set[Union[Edge, UEdge, Any]]:
        """
        Get the set, E, for a network.

        Args:
            N/A
        Returns:
            set[Edge]: Edge set, E.
        """
        if self.__directed:
            return self.__edges
        else:
            return self.__uedges
    
class Graph:
    """
    Superclass that defines the general constructs for working with graphs
    in a phylogenetic context. This object assumes that edges are undirected 
    and the graph is therefore unrooted.
    
    Nodes may have as many neighbors as needed, and no topological restrictions
    are assumed.
    """
    
    def __init__(self, 
                 edges : EdgeSet | None = None, 
                 nodes : NodeSet | None = None) -> None:
        """
        Initialize a Network object.
        You may initialize with any combination of edges/nodes,
        or provide none at all.

        Args:
            edges (EdgeSet, optional): A set of Edges. 
                                       Defaults to an empty EdgeSet.
            nodes (NodeSet, optional): A set of Nodes. 
                                       Defaults to an empty NodeSet.
        Returns:
            N/A
        """
        
        # Blob storage for anything that you want to associate with 
        # this network. Just give it a string key!
        self.__items : dict[str, object] = {}

        if edges is not None:
            self._edges : EdgeSet = edges
        else:
            self._edges : EdgeSet = EdgeSet(directed = False)
        
        if nodes is not None:
            self._nodes : NodeSet = nodes
        else:
            self._nodes : NodeSet = NodeSet(directed = False)
        
        
        # Initialize the unique id count
        self.__uid : int = 0
    
        
        #Free floater nodes/edges are allowed
        for edge in list(self._edges.get_set()):
            self._nodes.process(edge)
        
        self.__leaves : list[Node] = [node for node in self.V()
                                      if self._nodes.out_deg(node) == 0]
    
    def __contains__(self, obj : Union[Node, UEdge, Edge]) -> bool:
        """
        Allows a simple pythonic "n in graph" or "e in graph" check.
        
        Raises: 
            TypeError: If @obj is of any type other than Node, Edge, orEdge.
            
        Args:
            obj (Node | UEdge |Edge): A node or edge.

        Returns:
            bool: True if obj is a node or edge in the graph
        """
        if type(obj) is UEdge or type(obj) is Edge:
            return obj in self._edges
        elif type(obj) is Node:
            return obj in self._nodes
        else:
            raise TypeError("Graphs and Networks only contain Node, Edge, \
                            orEdge objects.")
    
    def __reclassify_node(self, node : Node) -> None:
        """
        
        PRIVATE. FOR IN CLASS USE ONLY
        Whenever an edge is added or removed from a network, the nodes that make
        up the edge need to be reclassified. 

        Args:
            node (Node): A node in the graph
        Returns:
            N/A
        """
        
        # If out degree now = 2, then the node was previously a leaf, 
        # and is not anymore
        if self._nodes.out_deg(node) == 2 or \
            self._nodes.out_deg(node) == 0:
            if node in self.__leaves:
                self.__leaves.remove(node)
            
        if self._nodes.out_deg(node) == 1:
            if node not in self.__leaves:
                self.__leaves.append(node)            
    
    @singledispatchmethod
    def add_nodes(self, *nodes : Node) -> None:
        """
        Add any amount of nodes to this graph (or Network).
        
        Args:
            *nodes (Node): A comma delimited list of Node objects.
        
        Returns:
            N/A
        """
        for node in nodes:
            self._nodes.add(node)
    
    @add_nodes.register
    def _(self, nodes : list) -> None: # type: ignore
        """
        Add any amount of nodes to this graph (or Network). This
        version of the method takes a list of Nodes.
        
        Args:
            nodes (list[Node]): A list of Node objs.
        
        Returns:
            N/A
        """
        for n in nodes: # type: ignore
            self._nodes.add(n) # type: ignore
            
                
    def add_uid_node(self, node : Node | None = None) -> Node:
        """
        Ensure a node has a unique name that hasn't been used before/is 
        not currently in use for this graph.
        
        May be used with a node that is or is not yet an element of V.
        The node will be added to V as a result of this function.

        Args:
            node (Node): Any node object. Defaults to None.
        Returns:
            (Node): the added/edited node that has been added to the graph.
        """
        if node is None:
            new_node : Node = Node(name = "UID_" + str(self.__uid))
            self.add_nodes(new_node)
            self.__uid += 1
            return new_node
        else:
            if node not in self._nodes:
                self.add_nodes(node)
            self.update_node_name(node, "UID_" + str(self.__uid))
            self.__uid += 1
            return node
    
    # @singledispatchmethod
    # def add_edges(self, *edges : UEdge) -> None:
    #     """
    #     Add any amount of edges to the graph.
        
    #     Duplicate edges, in an undirected context, is not allowed (will have
    #     no effect).
        
    #     Note: Each edge that you attempt to add must be between two nodes that
    #     exist in the network. Otherwise, an error will be thrown.
        
    #     Args:
    #         *edges (Edge): Any amount of edge objects.

    #     Raises:
    #         NetworkError: If any edge provided contains nodes that are not in  
    #                       the graph. 
    #         TypeError: If any edge provided is undirected when it should be
    #                    directed, or directed when it should be undirected.
    #     """
        
    #     for edge in edges: 
    #         if self._nodes.ready(edge): 
    #             if edge not in self._edges:             
    #                 self._edges.add(edge)
    #                 self._nodes.process(edge)  
    #                 self.__reclassify_node(edge.n1)
    #                 self.__reclassify_node(edge.n2)
    #         else:
    #             raise NetworkError("Tried to add an edge between two nodes,\
    #                                 at least one of which does not belong\
    #                                 to this network.")
    
    # @add_edges.register
    # def _(self, edges : list) -> None: # type: ignore
    #     """
    #     Add all edges from a list of edges to the graph.
        
    #     Duplicate edges, in an undirected context, is not allowed (will have
    #     no effect).
        
    #     Note: Each edge that you attempt to add must be between two nodes that
    #     exist in the network. Otherwise, an error will be thrown.
        
    #     Args:
    #         edges (list[AEdge]): A list of edges

    #     Raises:
    #         NetworkError: If any edge provided contains nodes that are not in  
    #                       the graph. 
    #         TypeError: If any edge provided is undirected when it should be
    #                    directed, or directed when it should be undirected.
    #     """
    #     for edge in edges: 
    #         if self._nodes.ready(edge):
    #             if edge not in self._edges:              
    #                 self._edges.add(edge)
    #                 self._nodes.process(edge)  
    #                 self.__reclassify_node(edge.n1)
    #                 self.__reclassify_node(edge.n2)
    #         else:
    #             raise NetworkError("Tried to add an edge between two nodes,\
    #                                 at least one of which does not belong\
    #                                 to this network.")
          
    @singledispatchmethod
    def remove_nodes(self, *nodes : Node) -> None:
        """
        Removes any amount of nodes from the list of nodes.
        Also prunes all edges from the graph that are connected to the removed 
        node(s).
        
        Has no effect if a node is not in this network.
        
        Args:
            *nodes (Node): Any amount of Node objs
        Returns:
            N/A
        """
        node : Node
        for node in nodes:
            if node in self._nodes:
                #in_edges are the same as outedges in the undirected context
        
                for edge in self._nodes.in_edges(node):
                    self.remove_edge(edge)
                
                
                self._nodes.remove(node)
    
         
    def remove_edge(self, edge : Union[Edge, UEdge, list[Node]]) -> None:
        """
        Removes an edge from the EdgeSet. Does not delete nodes with no edges
        Has no effect if 'edge' is not in the graph.

        Raises: 
            NetworkError: If given as a list, and the list of nodes is malformed 
                          in any way.
        Args:
            edge (UEdge | list[Node]): An edge to remove from the graph, either 
                                      represented as an Edge object or as a list
                                      of Nodes (length 2).
        Returns:
            N/A                            
        
        """
        
        if type(edge) is list[Node]:
            if len(edge) == 2:
                edge = self.get_edge(edge[0], edge[1]) 
                
            else:
                raise NetworkError("Please provide a list of two nodes.")
    
        if edge in self.E():
            if type(edge) is UEdge:
                # Remove the edge from the edge set
                self._edges.remove(edge)
            
                #Make the node set aware of the edge removal
                self._nodes.process(edge, removal = True)
                
                # Reclassify the nodes, as they may be leaves/roots/etc now.
                
                self.__reclassify_node(edge.n1)
                self.__reclassify_node(edge.n2)
        else:
            raise NetworkError("Attempted to remove edge fr")

            
    def get_edge(self, n1 : Node, n2 : Node) -> Any:
        """
        Gets the edge in the graph with the given members.
        
        Args:
            n1 (Node): A Node that is a member of an edge with @n2
            n2 (Node): A Node that is a member of an edge with @n1
        Returns:
            UEdge: the edge that has member nodes 'n1' and 'n2'
        """
        e = self._edges.get(n1, n2)
        if type(e) is UEdge:
            return e   
        else:
            raise NetworkError("Retrieved an edge from a Graph object that is \
                                not undirected") 
   
    def V(self) -> list[Node]:
        """
        Get all nodes in V.

        Args:
            N/A
        Returns:
            list[Node]: the set V, in list form.
        """
        return list(self._nodes.get_set())
    
    def E(self) -> list[Any]:
        """
        Get the set E (in list form).

        Args:
            N/A
        Returns:
            list[Edge]: The list of all edges in the graph
        """
        return list(self._edges.get_set()) 
    
    def get_item(self, key : str) -> object:
        """
        Access the blob storage with a key. 

        Args:
            key (str): A string key associated with the item you wish 
                       to retrieve

        Returns:
            object: The object in storage that is associated with 'key'
        """
        return self.__items[key]
    
    def put_item(self, key : str, item : Any) -> None:
        """
        Associate an item with this Graph and give it a key for later lookup.

        Args:
            key (str): a lookup key
            item (Any): literally any object you wish to associate with this 
                        Graph/Network
        """
        if key not in self.__items:
            self.__items[key] = item

    def update_node_name(self, node : Node, name : str) -> None:
        """
        Rename a node and update the bookkeeping.

        Args:
            node (Node): a node in the graph
            name (str): the new name for the node.
        Returns:
            N/A
        """
        self._nodes.update(node, name)
    
    def has_node_named(self, name : str) -> Union[Node, None]:
        """
        Check whether the graph has a node with a certain name.
        Strings must be exactly equal (same white space, capitalization, etc.)

        Args:
            name (str): the name to search for

        Returns:
            Node (or None): the node with the given name, if one exists.
        """
        try:
            return self._nodes.get(name) 
        except:
            return None
        
    def in_degree(self, node : Node) -> int:
        """
        Get the in-degree of a node.

        Args:
            node (Node): A node in V

        Returns:
            int: the in degree count
        """
        if node in self._nodes:
            return self._nodes.in_deg(node)
        else:
            warnings.warn("Attempting to get the in-degree of a node that is \
                not in the graph-- returning 0")
            return 0

    def out_degree(self, node : Node) -> int:
        """
        Get the out-degree(number of edges where the given node is a parent)
        of a node in the graph.

        Args:
            node (Node): a node in V

        Returns:
            int: the out-degree count
        """
        if node in self._nodes:
            return self._nodes.out_deg(node)
        else:
            warnings.warn("Attempting to get the out-degree of a node that is\
                not in the graph-- returning 0")
            return 0

    def in_edges(self, node : Node) -> list[Any]:
        """
        Get the in-edges of a node in V. The in-edges are the edges in E, where
        the given node is the child.

        Args:
            node (Node): a node in V

        Returns:
            list[Edge]: the list of in-edges
        """
        if node in self._nodes:
            return self._nodes.in_edges(node)
        else:
            warnings.warn("Attempting to get the in-edges of a node that is\
                not in the graph-- returning an empty list")
            return []
            
    def out_edges(self, node : Node) -> list[Any]:
        """
        Get the out-edges of a node in V. The out-edges are the edges in E,
        where the given node is the parent.

        Args:
            node (Node): a node in V

        Returns:
            list[Edge]: the list of out-edges
        """
        if node in self._nodes:
            return self._nodes.out_edges(node)
        else:
            warnings.warn("Attempting to get the out-edges of a node that is\
                not in the graph-- returning an empty list")
            return []

class Network(Graph):
    """
    This class represents a directed (and potentially acyclic) graph containing 
    nodes and edges.
    
    An 'Edge' object is a wrapper class for a tuple of two nodes, (a, b),
    where a and b are Node objects, and the direction of the edge is from 
    a to b (a is b's parent) -- thus (a, b) is NOT the same as (b, a).
    
    Use of 'UEdge' objects is strictly prohibited here.

    Notes and Allowances:
    
    1) You may create cycles -- however we have provided a method to check if 
       this graph object is acyclic. This method is internally called on 
       methods that assume that a network has no cycles, so be mindful of the 
       state of networks that are passed as arguments.
    
    2) You may have multiple roots. Be mindful of whether this graph is 
       connected and what root you wish to operate on.
    
    3) You may end up with floater nodes/edges, ie this may be an unconnected 
       network with multiple connected components. We will provide a method to 
       check for whether your object is one single connected component. 
       We have also provided methods to remove such artifacts.      
    """

    def __init__(self,
                 edges : Union[EdgeSet, None] = None, 
                 nodes : Union[NodeSet, None] = None) -> None:
        """
        Initialize a Network object.
        You may initialize with any combination of edges/nodes,
        or provide none at all.
        
        If you provide an EdgeSet and no nodes, each node present in the 
        EdgeSet *WILL* be added to the network.

        Args:
            edges (EdgeSet, optional): A set of Edges. 
                                       Defaults to None.
            nodes (NodeSet, optional): A set of Nodes. 
                                       Defaults to None.
        Returns:
            N/A
        """
        
        if edges is not None and nodes is not None:
            super().__init__(edges, nodes)
        elif edges is None and nodes is not None:
            super().__init__(EdgeSet(), nodes)
        elif edges is not None and nodes is None:
            ns : NodeSet = NodeSet()
            for e in edges.get_set():
                if type(e) is not Edge:
                     raise TypeError("Network objects takeEdge objects. \
                                    Gave an undirected edge (Edge obj).")
                ns.add(e.src, e.dest)
            super().__init__(edges, ns)
        else:
            super().__init__(EdgeSet(), NodeSet())
        
        self.__leaves : list[Node] = [node for node in list(self._nodes.get_set())
                        if self._nodes.out_deg(node) == 0]
        self.__roots : list[Node] = [node for node in list(self._nodes.get_set()) 
                        if self._nodes.in_deg(node) == 0]
        
    def add_edges(self, edges : Union[Edge, list[Edge]]) -> None:
        """
        If edges is a list of Edges, then add each Edge to the list of edges.
        
        If edges is a singleton Edge then just add to the edge array.
        
        Note: Each edge that you attempt to add must be between two nodes that
        exist in the network. Otherwise, an error will be thrown.
        
        Raises:
            NetworkError: if input edge/edges are malformed in any way
        
        Args:
            edges (Edge | list[Edge]): a single edge, or multiple.

        Returns:
            N/A
        """
        
        # Determine whether the param is a list of edges, or a single edge. 
        
        if type(edges) is list:
            for edge in edges: 
                if self._nodes.ready(edge):              
                    self._edges.add(edge)
                    self._nodes.process(edge)  
                    self.__reclassify_node(edge.src, True, True)
                    self.__reclassify_node(edge.dest, False, True)
                else:
                    raise NetworkError("Tried to add an edge between two nodes,\
                                        at least one of which does not belong\
                                        to this network.")
        elif type(edges) is Edge:
            if self._nodes.ready(edges):
                self._edges.add(edges)
                self._nodes.process(edges)
                self.__reclassify_node(edges.src, True, True)
                self.__reclassify_node(edges.dest, False, True)  
            else:
                raise NetworkError("Tried to add an edge between two nodes,\
                                    at least one of which does not belong\
                                    to this network.") 
    
    @singledispatchmethod
    def remove_nodes(self, node : Node) -> None:
        """
        Removes node from the list of nodes.
        Also prunes all edges from the graph that are connected to the node.
        
        Has no effect if node is not in this network.
        
        Args:
            node (Node): a Node obj
        
        Returns:
            N/A
        """
        
        if node in self._nodes:
            in_edges = self._nodes.in_edges(node)
            out_edges = self._nodes.out_edges(node)
            
            for edge in in_edges:
                self.remove_edge(edge)
            for edge in out_edges:
                self.remove_edge(edge)
            
            self._nodes.remove(node)
                     
    def remove_edge(self, 
                    edge : Union[Edge, UEdge, list[Node]], 
                    gamma : float | None = None) -> None:
        """
        Removes edge from the list of edges. Does not delete nodes with no edges
        Has no effect if 'edge' is not in the graph.
        
        Args:
            edge (Edge | UEdge | list[Node]): an edge to remove from the graph
            gamma (float): an inheritance probability from [0,1], if the edge is
                           provided as a list of nodes, and there is an 
                           identifiability issue that needs resolving (ie,
                           the edge that needs to be removed is a bubble
                           edge). Optional. Defaults to None.
            
        Returns:
            N/A
        """
    
        if type(edge) == list:
            if len(edge) == 2:
                if gamma is not None:
                    edge = self.get_edge(edge[0], edge[1], gamma) 
                else:
                    edge = self.get_edge(edge[0], edge[1])
            else:
                raise NetworkError("Please provide a list of two nodes,\
                                 in the format [src, dest]")
        
        
        if edge in self.E() and type(edge) is Edge:
            # Remove the edge from the edge set
            self._edges.remove(edge)
        
            #Make the edge set aware of the edge removal
            self._nodes.process(edge, removal = True)
            
            # Reclassify the nodes, as they may be leaves/roots/etc now.
            self.__reclassify_node(edge.src, True, False)
            self.__reclassify_node(edge.dest, False, False)
        elif edge not in self.E() and type(edge) is Edge:
            return
        else:
            raise NetworkError("Tried to remove undirected edge object from a\
                                directed Network")
    
    def get_edge(self, 
                 n1 : Node, 
                 n2 : Node, 
                 gamma : float | None = None, 
                 tag : str | None = None) -> Edge:
        """
        Note, that in the event of bubbles, 2 edges will exist with the same 
        source and destination. If this is possible, please supply the 
        inheritance probability of the correct branch. If both edges are known 
        to be identical (gamma = 0.5), then one will be chosen at random.

        Args:
            n1 (Node): parent node
            n2 (Node): child node
            gamma (float): inheritance probability. Optional. Defaults to None
            tag (str): A name/identifiability tag for hybrid edges should both
                       gammas be = .5. Optional. Defaults to None.                   
        Returns:
           Edge: the edge containing n1 and n2 and has the proper gamma value 
                  (if applicable).
        """
        e = self._edges.get(n1, n2, gamma, tag) 
        assert(type(e) is Edge)
        return e
             
    def __reclassify_node(self, 
                        node : Node, 
                        is_par : bool,
                        is_addition : bool) -> None:
        """
        Whenever an edge is added or removed from a network, the nodes that make
        up the edge need to be reclassified. 

        Args:
            node (Node): A node in the graph
            is_par (bool): flag that tells the method whether the node is being 
                           operated on as a parent (true) or child (false)
            is_addition (bool): flag that tells the method whether the node arg 
                                is an addition (true) or subtraction (false)
        
        Returns:
            N/A
        """
        if is_addition:
            if is_par:
                # If out degree now = 1, then the node was previously a leaf, 
                # and is not anymore
                if self._nodes.out_deg(node) == 1:
                    if node in self.__leaves:
                        self.__leaves.remove(node)
                if self._nodes.in_deg(node) == 0:
                    if node not in self.__roots:
                        self.__roots.append(node)
            else:
                # If in_degree now = 1, then the node was previously a root,
                # and is not anymore
                if self._nodes.in_deg(node) == 1:
                    if node in self.__roots:
                        self.__roots.remove(node)
                if self._nodes.out_deg(node) == 0:
                    if node not in self.__leaves:
                        self.__leaves.append(node)                
        else:
            if is_par:
                # if out degree is now = 0, then the node is now a leaf
                if self._nodes.out_deg(node) == 0:
                    self.__leaves.append(node)     
            else:
                # if in degree is now = 0, the node is now a root
                if self._nodes.in_deg(node) == 0:
                    self.__roots.append(node)
                
    def root(self) -> Node:
        """
        Return the root of the Network. Phylogenetic networks only have one 
        root, but for generality and practical use, multiple roots have been 
        allowed. To get all roots, should multiple exist, call the function
        "roots". This function only returns 1 root.

        Raises:
            NetworkError: If there are no roots in the network (cycle, or empty)
            Warning: If there is more than 1 root in the network.

        Args:
            N/A
        Returns:
            Node: root Node object
        """
        roots = [root for root in self.__roots 
                if self._nodes.out_deg(root) != 0]
        
        if len(roots) > 1:
            warnings.warn("Asked for singular root, but there are more than\
                            one. Returning the first one, but double check to\
                            make sure this is the root \
                            you intended to get!")
            
        if len(roots) != 0:
            return roots[0]
        else:
            raise NetworkError("There are no roots in this network. There\
                                is either a cycle, or nothing has been added\
                                and this is an empty network.")
    
    def roots(self) -> list[Node]:
        """
        Return the root(s) of the Network. Phylogenetic networks only have one 
        root, but for generality and practical use, multiple roots have been 
        allowed.

        Raises:
            NetworkError: If there are no roots in the network (cycle, or empty)
        Args:
            N/A
        Returns:
            list[Node]: a list of root Node objects.
        """
        roots = [root for root in self.__roots 
                if self._nodes.out_deg(root) != 0]
        return roots
        
    def get_leaves(self) -> list[Node]:
        """
        Returns the set X (a subset of V), the set of all leaves (nodes with
        out-degree 0). Only returns the leaves that are connected/reachable from
        the root.

        Args:
            N/A
        Returns:
            list[Node]: the connected elements of X, in list format.
        """
        #why not "return self.leaves?"
        return [leaf for leaf in self.__leaves 
               if self._nodes.in_deg(leaf) != 0]
        
    def get_parents(self, node : Node) -> list[Node]:
        """
        Returns a list of the parents of a node. 
        There are no assumptions placed on the length of this array.
        
        Raises:
            NetworkError: if the node is not in the network.

        Args:
            node (Node): any node in V.

        Returns:
            list[Node]: the list of nodes in the network that have the given    
                        node as a child node via an edge.
        """
        try:
            return [edge.src for edge in self._nodes.in_edges(node)]
        except:
            raise NetworkError("Attempted to calculate parents of a node that \
                is not in the graph.")
        
    def get_children(self, node : Node) -> list[Node]:
        """
        Returns a list of the children of a node.
        There are no assumptions placed on the length of this array.

        Raises:
            NetworkError: if the node is not in the network.

        Args:
            node (Node): any node in V.

        Returns:
            list[Node]: the list of nodes in the network are child nodes of the
                        given node.
        """
        try:
            return [edge.dest for edge in self._nodes.out_edges(node)]
        except:
            raise NetworkError("Attempted to calculate children of a node that \
                is not in the graph.")
            
    def clean(self, options : list[bool] = [True, True, True]) -> None:
        """
        All the various ways that the graph can be cleaned up and streamlined
        while not altering topology or results of algorithms.
        
        Algorithm Indeces:
        0) Remove nodes that have in/out degree of 0 (floater nodes)
        1) Remove a spurious root/root edge (root node with only one out edge)
        2) Consolidate all chains of nodes with in/out degree of 1 into 1 edge.
        
        Default behavior is to run all three. To not run a certain routine, 
        set the options list at the indeces listed above to False.
        
        Ie. To run the first and third algo, use [True, False, True].

        Args:
            options (list[bool], optional): a list of booleans that designate 
                                            which of the cleaning algorithms to 
                                            run. Defaults to [True, True, True],
                                            aka runs all 3.
        Returns:
            N/A
        """
        if options[0]:
            #Delete floater nodes
            floaters = [node for node in self.V() \
                        if self.in_degree(node) == 0 \
                        and self.out_degree(node) == 0]
            
            for floater in floaters:
                self._nodes.remove(floater)
    
        if options[1]:
            #Delete spurious root/root edge combo
            root : Node = self.root()
            if self.out_degree(root) == 1:
                spurious_edge = self.get_edge(root, self.get_children(root)[0])
                self.remove_edge(spurious_edge)
                self.remove_nodes(root)
        if options[2]:
            #Delete spurious "speciation events" marked by nodes with in/out = 1 
            spacers = [n for n in self.V() \
                       if self.in_degree(n) == 1 and self.out_degree(n) == 1]
            
            while len(spacers) != 0:
                cur = spacers[0]
                spacer_par = self.get_parents(cur)[0]
                spacer_child = self.get_children(cur)[0]
                
                self.remove_edge(self.get_edge(spacer_par, cur))
                self.remove_edge(self.get_edge(cur, spacer_child))
                self.remove_nodes(cur)
                
                self.add_edges(Edge(spacer_par, spacer_child))
                
                spacers = [n for n in self.V() if self.in_degree(n) == 1 and \
                           self.out_degree(n) == 1]
        
    def mrca(self, set_of_nodes: set[Node] | set[str]) -> Node:
        """
        Computes the Least Common Ancestor of a set of graph nodes

        Args:
            set_of_nodes (set[Node] | set[str]): A set of Nodes, or node names.

        Returns:
            Node: The node that is the LCA of the set.
        """
        format_set : set[Node] = set()
        for item in set_of_nodes:
            if type(item) is str:
                node_version = self.has_node_named(item)
                if node_version is None:
                    raise NetworkError("A node in 'set_of_nodes' is not \
                                      in the graph")
                else:
                    format_set.add(node_version)
            elif type(item) is Node:
                if item in self.V():
                    format_set.add(item)
                else:
                    raise NetworkError("A node in 'set_of_nodes' is not \
                                      in the graph")
            else:
                raise NetworkError(f"Wrong type for parameter set_of_nodes. \
                                  Expected set[Node] or set[str].")
        
        set_of_nodes = format_set
                
        # mapping from each node in set_of_nodes to 
        # a mapping from ancestors to dist from node.
        leaf_2_parents : dict[Node, dict[Node, int]] = {} 

        for leaf in set_of_nodes:
            #Run bfs upward from each node 
            node_2_lvl : dict[Node, int] = {}
            
            # queue for bfs
            q : deque[Node] = deque()
            q.append(leaf)
            visited : set[Node] = set()
            node_2_lvl[leaf] = 0

            while len(q) != 0:
                cur = q.popleft()

                #Seach cur's parents
                for neighbor in self.get_parents(cur):
                    if neighbor not in visited:
                        node_2_lvl[neighbor] = node_2_lvl[cur] + 1
                        q.append(neighbor)
                        visited.add(neighbor)
            
            leaf_2_parents[leaf] = node_2_lvl
        
        
        #Compare each leaf's parents
        intersection = self._nodes.get_set()
        for leaf, par_level in leaf_2_parents.items():
            intersection = intersection.intersection(set(par_level.keys()))
        
        # Map potential LCA's to cumulative distance from all the nodes
        additive_level = {} 
        
        for node in intersection: # A LCA has to be in each node's ancestor set
            lvl = 0
            for leaf in set_of_nodes:
                try:
                    lvl += leaf_2_parents[leaf][node]
                except KeyError:
                    continue
            
            additive_level[node] = lvl
        
        #The LCA is the node that minimizes cumulative distance
        return min(additive_level, key=additive_level.get) # type: ignore
                      
    def leaf_descendants(self, node : Node) -> set[Node]:
        """
        Compute the set of all leaf nodes that are descendants of the parameter 
        node. Uses DFS to find paths to leaves.

        Args:
            node (Node): The node for which to compute leaf children

        Returns:
            set[Node]: The list of all leaves that descend from 'node'
        """
        if node not in self.V():
            raise NetworkError("Node not found in graph.")
        
        root = node

        # stack for dfs
        q : deque[Node]= deque()
        q.appendleft(root)
        leaves : set[Node] = set()

        while len(q) != 0:
            cur = q.popleft()
        
            if self.out_degree(cur) == 0:
                leaves.add(cur)
                
            for neighbor in self.get_children(cur): #Continue path to a leaf
                q.append(neighbor)
        
        return leaves    
        
    def diff_subtree_edges(self, rng : np.random.Generator) -> list[Edge]:
        """
        Returns 2 random edges such that there does not exist a directed path 
        from one edge source node to the other edge source node.

        Args:
            rng (np.random.Generator): an rng object.
        
        Returns:
            list[Edge]: a list of 2 edges such that neither edge 
                        is reachable from either starting point.
        """
        #Grab a random edge
        first_edge = __random_object(self.E(), rng)
        assert(type(first_edge) is Edge)
        
        #Find another edge while excluding descendants of the first edge
        first_edge_subtree = self.leaf_descendants(first_edge.dest)
       
        #Accumulate pairs of edges that satisfy the requirement
        valid_edges : list[Edge] = []
        for edge in self.E():
            leaf_desc_edge : set[Node] = self.leaf_descendants(edge.dest)
            # If the intersection of leaf sets is null, then neither edge
            # can be found from the other.
            if len(leaf_desc_edge.intersection(first_edge_subtree)) == 0:
                valid_edges.append(edge)
            
        second_edge = __random_object(valid_edges, rng)
        assert(type(second_edge) is Edge)
        
        return [first_edge, second_edge]
    
    def subgenome_count(self, n : Node) -> int:
        """
        Given a node in this graph, return the subgenome count.
         
        Args:
            n (Node): Any node in the graph. 
                      It is an error to input a node that is not in the graph.

        Returns:
            int: subgenome count
        """
        
        if n not in self.V():
            raise NetworkError("Input node is not in the graph")
        
        if self.root() == n:
            return 1
        else:
            parents = self.get_parents(n)
            return sum([self.subgenome_count(parent) for parent in parents])
            
    def edges_downstream_of_node(self, n : Node) -> list[Edge]:
        """
        Returns the set (as a list) of edges that are in the subgraph of a node.

        Args:
            n (Node): A node in a graph.
        Returns:
            list[Edge]: The set of all edges in the subgraph of n.
        """
        if n not in self.V():
            raise NetworkError("Input node is not in the graph.")
        
        q : deque[Node] = deque()
        q.appendleft(n)
        
        edges : list[Edge] = list()
        
        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            edge : Edge
            for edge in self.out_edges(cur):
                edges.append(edge)
                q.append(edge.dest)
        
        return edges
    
    def edges_upstream_of_node(self, n : Node) -> list[Edge]:
        """
        Returns the set (as a list) of edges that are in all paths from the root
        to this node.
        
        Useful in avoiding the creation of cycles when adding edges.

        Args:
            n (Node): A node in a graph.
        Returns:
            list[Edge]: The set of all edges in the subgraph of n.
        """
        if n not in self.V():
            raise NetworkError("Input node is not in the graph.")
        
        q : deque[Node] = deque()
        q.appendleft(n)
        
        edges : list[Edge] = list()
        
        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            edge : Edge
            for edge in self.in_edges(cur):
                edges.append(edge)
                q.append(edge.src)
        
        return edges
    
    def subgenome_ct_edges(self, 
                           downstream_node : Node | None= None, 
                           delta : float = math.inf, 
                           start_node : Node | None = None) \
                           -> dict[Edge, int]:
        """
        Maps edges to their subgenome counts.
        
        Raises:
            NetworkError: If the graph has more than one root to start.
            
        Args:
            downstream_node (Node, optional): No edges will be included in the
                                              map that are in a subgraph of this 
                                              node. Defaults to None.
            delta (float, optional): Only include edges in the mapping that have
                                     subgenome counts <= delta. 
                                     Defaults to math.inf.
            start_node (Node, optional): Provide a node only if you don't want 
                                         to start at the root. 
                                         Defaults to None.

        Returns:
            dict[Edge, int]: a map from edges to subgenome counts
        """
    
        old_map = self.edges_to_subgenome_count(downstream_node, 
                                                delta, 
                                                start_node)
        rev_map : dict[Edge, int] = {}
        
        for key, edges in old_map.items():
            for edge in edges:
                rev_map[edge] = key
        
        return rev_map
                
    def edges_to_subgenome_count(self, 
                                 downstream_node : Node | None= None, 
                                 delta : float = math.inf, 
                                 start_node : Node | None = None) \
                                 -> dict[int, list[Edge]]:
        """
        Maps edges to their subgenome counts.
        
        Raises:
            NetworkError: If the graph has more than one root to start.
        Args:
            downstream_node (Node, optional): No edges will be included in the
                                              map that are in a subgraph of this 
                                              node. Defaults to None.
            delta (float, optional): Only include edges in the mapping that have
                                     subgenome counts <= delta. 
                                     Defaults to math.inf.
            start_node (Node, optional): Provide a node only if you don't want 
                                         to start at the root. 
                                         Defaults to None.

        Returns:
            dict[Edge, int]: a map from edges to subgenome counts
        """
    
        if start_node is None:
            start_node = self.root()
            
        q : deque[Node] = deque()
        q.appendleft(start_node)
        
        edges_2_sub = {edge : 0 for edge in self.E()}
        
        
        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            for neighbor in self.get_children(cur):
                
                edges_2_sub[self._edges.get(cur, neighbor)] += 1
                
                # Resume search from the end of the chain if one existed, 
                # or this is neighbor if nothing was done
                q.append(neighbor)
    
        
        partition : dict[int, list[Edge]] = {}
        for edge, value in edges_2_sub.items():
            if value not in partition.keys():
                partition[value] = [edge]
            else:
                partition[value].append(edge)
        
        #Filter out invalid keys
        filter1 = {key : value for (key, value) in partition.items() 
                   if key <= delta}
        
        #Filter out edges that would create a cycle from param edge
        if downstream_node is not None:
            filter2 : dict[int, list[Edge]] = {}
            for subct, edges in filter1.items():
                for target in edges:
                    downstream_edges = self.edges_downstream_of_node(downstream_node)
                    if target not in downstream_edges:
                        if subct not in filter2.keys():
                            filter2[subct] = [target]
                        else:
                            filter2[subct].append(target)
            return filter2
        else:
            return filter1

    def __leaf_desc_help(self, 
                         node : Node, 
                         leaves : list[Node], 
                         desc_map : dict[Node, set[Node]]) -> set[Node]:
        """
        Helper function for "leaf_descedants_all".  

        Args:
            net (Network): A network
            node (Node): a node in 'net'
            leaves (list[Node]): _description_
            desc_map (dict[Node, set[Node]]): _description_

        Returns:
            set[Node]: the leaf descendents of the 'node' param
        """
        if node not in desc_map.keys():
            if node in leaves:
                desc_map[node] = {node}
            else:
                desc_map[node] = set()
                for child in self.get_children(node):
                    child_desc = self.__leaf_desc_help(child, leaves, desc_map)
                    #A node's leaf descendant set is the union of all its children's
                    #leaf descendant sets
                    desc_map[node] = desc_map[node].union(child_desc)
                        
        return desc_map[node]  

    def leaf_descendants_all(self) -> dict[Node, set[Node]]:
        """
        Map each node in the graph to its set of leaf descendants
       
        Args:
            N/A
        Returns:
            dict[Node, set[Node]]: map from graph nodes to their 
                                   leaf descendants
        """
        desc_map : dict[Node, set[Node]] = {}
        
        #Mutates desc_map
        self.__leaf_desc_help(self.root(), self.get_leaves(), desc_map)
        
        return desc_map
    
    def __newick_help(self, 
                      node : Node, 
                      processed_retics : set[Node]) -> str:
        """
        Helper function to "newick". Generates the newick string (sans ending 
        semicolon) of the subnetwork defined by 'node'.

        Args:
            net (Network): A Network
            node (Node): a Node in 'net'
            processed_retics (set[Node]): A set of all reticulation nodes that
                                        have already been seen by the search 
                                        function. Helps to avoid rewriting the
                                        subnetwork string of a reticulation.

        Returns:
            str: The newick representation of a subnetwork.
        """
            
        if node in self.get_leaves():
            return node.label
        else:
            if self.in_degree(node) >= 2 and node in processed_retics:
                if node.label[0] != "#":
                    return "#" + node.label
                return node.label
            else:
                if self.in_degree(node) >= 2:
                    processed_retics.add(node)
                    if node.label[0] != "#":
                        node_name = "#" + node.label
                    else:
                        node_name = node.label
                else:
                    node_name = node.label    
                    
                substr = "("
                for child in self.get_children(node):
                    substr += self.__newick_help(child, processed_retics)
                    substr += ","
                substr = substr[0:-1]
                substr += ")"
                substr += node_name
                
                return substr
    
    def newick(self) -> str:
        """
        Generates the extended newick representation of this Network
        
        Args:
            N/A
        Returns:
            str: a newick string
        """
        return self.__newick_help(self.root(), set()) + ";"
    
    def _is_cyclic_util(self, 
                       v : Node,
                       visited : dict[Node, bool], 
                       rec_stack : dict[Node, bool]) -> bool:
        """
        is_acyclic helper function

        Args:
            v (Node): _description_
            visited (dict[Node, bool]): _description_
            rec_stack (dict[Node, bool]): _description_

        Returns:
            bool: _description_
        """
        visited[v] = True
        rec_stack[v] = True

        for neighbor in self.get_children(v):
            if not visited[neighbor]:
                if self._is_cyclic_util(neighbor, visited, rec_stack):
                    return True
            elif rec_stack[neighbor]:
                return True

        rec_stack[v] = False
        return False

    def is_acyclic(self) -> bool:
        """
        Checks if each of this graph's connected components is acyclic

        Args:
            N/A
        Returns:
            bool: True if acyclic, False if cyclic. 
        """
        
        #Maintain structures for checking nodes that are visited or in the recursive stack
        visited = {node : False for node in self.V()}
        rec_stack = {node : False for node in self.V()}

        #Call recursive dfs on each root node / each connected component
        for node in self.roots():
            if not visited[node]:
                if self._is_cyclic_util(node, visited, rec_stack):
                    return False

        return True
    
    def bfs_dfs(self, 
                start_node : Node | None = None,
                dfs : bool = False, 
                is_connected : bool = False, 
                accumulator : Callable[..., None] | None = None, 
                accumulated : Any = None) -> tuple[dict[Node, int], Any]:
        """
        General bfs-dfs routine, with the added utility of checking 
        whether or not this graph is made up of multiple connected components.

        Args:
            start_node (Node, optional): Give a node to start the search from. 
                                         Defaults to None, in which case the 
                                         search will start at the root.
            dfs (bool, optional): Flag that specifies whether to use bfs or dfs. 
                                  Defaults to False (bfs), if true is passed, 
                                  will run dfs.
            is_connected (bool, optional): Flag that, if enabled, will check for 
                                           the connected component status. 
                                           Defaults to False (won't run).
            accumulator (Callable, optional): A function that takes the 
                                              currently searched Node in the 
                                              graph and does some sort 
                                              of bookkeeping.
            accumulated (Any): Any type of structure that stores the data 
                               given by the accumulator function.

        Returns:
            dict[Node, int]: Mapping from nodes to their distance 
                             from the start node.
        """
        q : deque[Node] = deque()
        visited : set[Node] = set()
        
        
        if start_node is not None:
            q.append(start_node)
            dist = {start_node : 0}
            visited.add(start_node)
        else:
            root : Node = self.root()
            q.append(root)
            dist = {root : 0}
            visited.add(root)
        
        while len(q) != 0:
            if dfs: 
                #Adding to left, so popleft is LIFO behavior
                cur : Node = q.popleft() 
            else: 
                #Popright is FIFO behavior
                cur : Node = q.pop() #Popright is FIFO behavior
            
            if accumulator is not None and accumulated is not None:
                accumulated = accumulator(cur, accumulated)
            
            for neighbor in self.get_children(cur):
                dist[neighbor] = dist[cur] + 1
                q.appendleft(neighbor)
                visited.add(neighbor)
        
        if is_connected:
            if len(set(self.V()).difference(visited)) != 0:
                print("GRAPH HAS MORE THAN 1 CONNECTED COMPONENT")
            else:
                print("GRAPH IS FULLY CONNECTED")
        
        return dist, accumulated
         
    # def rootpaths(self, start : Node) -> list[list[Edge]]:
    #     """
    #     Get all paths (list of edges)

    #     Args:
    #         start (Node): Start the search from this node

    #     Returns:
    #         list[list[Edge]]: a list of all paths (lists of edges) to the root  
    #                           from 'start'
    #     """
    #     #A list of paths, each path is a list of edges.
    #     paths : list[list[Edge]] = [] 
        
    #     for par in self.get_parents(start):
    #         for path in self.rootpaths(par):
    #             paths.append(path.append(self._edges.get(par, start)))
    #     return paths
    
    def subnet(self, retic_node : Node) -> Network:
        """
        Make a copy of a subnetwork of this Network, rooted at 'retic_node', 
        with unique node names.
        
        Args:
            retic_node (Node): A node in this network that is a reticulation 
                               node

        Returns:
            Network: A subnetwork of the DAG being operated on
        """
    
        q : deque[Node] = deque()
        q.appendleft(retic_node)
        net_copy = Network()
        
        new_node = Node(name = retic_node.label + "_copy")
        net_copy.add_nodes(new_node)
        net_2_mul = {retic_node : new_node}
        

        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            for neighbor in self.get_children(cur):
                new_node = Node(name = neighbor.label + "_copy")
                net_copy.add_nodes(new_node)
                net_2_mul[neighbor] = new_node
                net_copy.add_edges(Edge(net_2_mul[cur], new_node))
                
                # Resume search from the end of the chain if one 
                # existed, or this is neighbor if nothing was done
                q.append(neighbor)
        
        return net_copy 
    
    def copy(self) -> tuple[Network, dict[Node, Node]]:
        """
        Copy this network into a new network object, also with new node and 
        edge objects.

        Args:
            N/A
        Returns:
            tuple[Network, dict[Node, Node]]: A carbon copy of this Network, 
                                              along with a mapping from the old
                                              network's nodes to the nodes in 
                                              the new network. 
        """
        net_copy : Network = Network()
        
        old_new : dict[Node, Node] = {}
        
        for node in self.V():
            new = node.copy()
            old_new[node] = new
            net_copy.add_nodes(new)
        
        for edge in self.E():
            new_src = old_new[edge.src]
            new_dest = old_new[edge.dest]
            new = edge.copy(new_src, new_dest)
            
            net_copy.add_edges(new)
        
        return net_copy, old_new
    
    def to_networkx(self) -> nx.Graph:
        """
        Generates a networkx object from this network, for viewing purposes!
        
        Args:
            N/A
        Returns:
            nx.Graph: _description_
        """
        nx_network = nx.MultiDiGraph()
        nx_network.add_nodes_from([node.label for node in self.V()]) # type: ignore
        nx_network.add_edges_from([edge.to_names() for edge in self.E()]) # type: ignore
        return nx_network

    def compare_network(self, net : Network, measure : str) -> float:
        """
        Compares the topology of this network compared to another network.
        
        The following are options for the distance measure:
        [tree|tri|cluster|luay|rnbs|apd|normapd|wapd|normwapd]
        
        If [tree | tri | cluster] is used, the return value will be the average
        of the false positive and false negative rates.
        
        If [luay] is used, the return value will be the distance between the
        two networks.
        
        If [rnbs | apd | normapd | wapd | normwapd] is used, the return value
        is the dissimilarity between the two networks.

        Args:
            net (Network): The other network.
            measure (str): The key for using one of the available distance 
                           measures, must be selected from 
                           [tree|tri|cluster|luay|rnbs|apd|normapd|wapd|normwapd]

        Returns:
            float: Measure of the two networks similarity/distance.
        """
        
        temp = tempfile.NamedTemporaryFile(suffix='.nex')
        temp_dir = tempfile.TemporaryDirectory()

        nex_file = NexusTemplate()
        net1_name = self.newick()
        net2_name = net.newick()
       
        nex_file.add(net1_name)
        nex_file.add(net2_name)
        
        ret_type : str
        flag : str = measure.strip().lower()
        if flag in ["tree", "tri", "cluster"]:
            ret_type = "avg"
        elif flag == "luay":
            ret_type = "dist"
        elif flag in ["rnbs", "apd", "normapd", "wapd", "normwapd"]:
            ret_type = "dissim"
        else:
            raise NetworkError(f"Unrecognized compare network option: {flag}. \
                                 Please select from [tree|tri|cluster|luay|\
                                 rnbs|apd|normapd|wapd|normwapd]")
        
        nex_file.add_phylonet_cmd(f"Cmpnets net1 net2 -m {flag}")
        nex_file.generate(temp_dir.name, temp.name)
        
        location : str = temp_dir.name + "/" + temp.name
        return_stream : list[Any] = run(location)
        
        print(return_stream)
        
        if ret_type == "avg":
            pass
        elif ret_type == "dist":
            pass
        else:
            pass
        
        return 0.0
    
class MUL(Network):
    """
    A subclass of a Network, that is a binary tree that results from the 
    transformation of a standard network into a Multilabeled Species Tree.
    """
    def __init__(self, gene_map : dict[str, list[str]], rng : np.random.Generator) -> None:
        """
        Initialize a MUL tree (Multilabeled Species Tree) with a map of species
        to the genes associates with them, and a rng.

        Args:
            gene_map (dict[str, list[str]]): A subgenome mapping.
            rng (np.random.Generator): A random seed generator
        Returns:
            N/A
        """
        self.net : Network | None = None
        self.mul : Network | None = None
        self.gene_map : dict[str, list[str]] = gene_map
        self.rng : np.random.Generator = rng
                        
    def to_mul(self, net : Network) -> Network:
        """
        Creates a (MU)lti-(L)abeled Species Tree from a network
        
        Raises:
            NetworkError: If the network is malformed with regards to ploidy
            
        Args:
            net (Network): A Network

        Returns:
            Network: a MUL tree (as a Network obj)
        """
       
        # Number of network leaves must match the number of gene map keys
        if len(net.get_leaves()) != len(self.gene_map.keys()):
            raise NetworkError(f"Input network has incorrect amount of \
                leaves. Given : {len(net.get_leaves())} \
                Expected : { len(self.gene_map.keys())}")
       
        copy_gene_map = copy.deepcopy(self.gene_map)
        mul_tree = Network()
        
        # Create copies of all the nodes in net and keep track of the conversion
        network_2_mul : dict[Node, Node] = {node : node.copy() for node in net.V()}
        
        # Add all nodes and edges from net into the mul tree
        
        #NODES
        mul_tree.add_nodes(list(network_2_mul.values()))
        
        #EDGES
        for edge in net.E():
            new_edge : Edge = edge.copy(network_2_mul[edge.src],
                                        network_2_mul[edge.dest])
            mul_tree.add_edges(new_edge)
        
        
        
        # Bottom-Up traversal starting at leaves. Algorithm from STEP 1 in (2)
        
        # Starting at leaves...
        # push onto queue when all children have been moved to the processed set
        processed : set[Node] = set()
        traversal_queue = deque(mul_tree.get_leaves())
        
        while len(traversal_queue) != 0:
            cur = traversal_queue.pop()
            
            original_pars = [node for node in mul_tree.get_parents(cur)]
            
            if mul_tree.in_degree(cur) == 2:
                #reticulation node. make a copy of subgraph
                subtree = mul_tree.subnet(cur)
                
                retic_pars = mul_tree.get_parents(cur)
                a = retic_pars[0]
                b = retic_pars[1]
            
                mul_tree.remove_edge([b, cur])
                mul_tree.add_nodes(subtree.V())
                for edge in subtree.E():
                    mul_tree.add_edges(edge)
                #mul_tree.add_edges(subtree.E())
                mul_tree.add_edges(Edge(b, subtree.root()))
                processed.add(subtree.root())
        
            processed.add(cur)
            
            for par in original_pars:
                cop = set(mul_tree.get_children(par))
                if cop.issubset(processed):
                    traversal_queue.append(par)
        
        #Get rid of excess connection nodes
        
        mul_tree.clean([False, False, True])
        
        #Rename tips based on gene mapping
        
        for leaf in mul_tree.get_leaves():
            
            new_name : str = copy_gene_map[leaf.label.split("_")[0]].pop()
            mul_tree.update_node_name(leaf, new_name)

        self.mul = mul_tree 
     
        return mul_tree  