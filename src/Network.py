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
Last Edit : 3/11/25
First Included in Version : 1.0.0
Docs   - [ ]
Tests  - [ ] 
Design - [ ]
"""

from __future__ import annotations
import copy
import math
import re
import warnings
from typing import Any, Callable, Union

import networkx as nx
import numpy as np

from .MSA import DataSequence

# ``NodeSet`` / ``EdgeSet`` are the compiled adjacency structures that back
# every Network. They are required, not an optional accelerator: keeping a
# second pure-Python implementation in sync by hand had already let the two
# drift apart on four behaviours.
try:
    from .cython.graph_core_cy import NodeSet, EdgeSet
except ImportError as exc:  # pragma: no cover - build configuration issue
    raise ImportError(
        "PhyNetPy requires its compiled Cython extensions. Build them with "
        "`pip install -e .` from the repository root. If PhyNetPy is already "
        "installed, the extensions were most likely built against a different "
        "Python version than the interpreter now running."
    ) from exc

#############################
####  EXCEPTION CLASSES  ####
#############################

class NetworkError(Exception):
    """
    This exception is raised when a network is malformed, 
    or if a network operation fails.
    """
    def __init__(self, message : str = "Error operating on a Network") -> None:
        """
        Initialize with an error message that will print upon the exception 
        being raised.

        Args:
            message (str, optional): Error message. Defaults to 
                                    "Error operating on a Network".
        Returns:
            N/A
        """
        self.message = message
        super().__init__(self.message)

class NodeError(Exception):
    """
    This exception is raised when a Node operation fails.
    """
    def __init__(self, message : str = "Error in Node Class") -> None:
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
    def __init__(self, message : str = "Error in Edge Class") -> None:
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
#### NODES AND EDGES #####
##########################

class Branch:
    """
    A lightweight, detached view of a single branch (edge) of a phylogenetic
    tree or network.
    
    Holds the branch length, inheritance probability, and the label of the
    parent node. Unlike :class:`Edge`, a ``Branch`` is not wired into a
    :class:`Network`; it is the value type handed to model nodes and other
    consumers that need branch parameters without the surrounding topology.
    """
    def __init__(self,
                 length : float = None,
                 inheritance_probability : float = None,
                 parent_id : str = None) -> None:
        """
        Initialize a Branch.
        
        Args:
            length (float, optional): Branch length. Defaults to None.
            inheritance_probability (float, optional): Inheritance probability
                                                       (gamma) for a
                                                       reticulation branch.
                                                       Defaults to None.
            parent_id (str, optional): Label of the parent node. Defaults to
                                       None.
        Returns:
            N/A
        """
        self.length = length
        self.inheritance_probability = inheritance_probability
        self.parent_id = parent_id
    
    def __len__(self) -> float:
        """
        Get the length of the branch.
        
        Args:
            N/A
        Returns:
            float: The branch length.
        """
        return self.length

    @property
    def length(self) -> float:
        """
        Get the length of the branch.
        
        Args:
            N/A
        Returns:
            float: The branch length.
        """
        return self._length
    
    @length.setter
    def length(self, value : float) -> None:
        """
        Set the length of the branch. Negative values are rejected with a
        warning and leave the current value untouched.
        
        Args:
            value (float): The new branch length.
        Returns:
            N/A
        """
        if value is not None and value < 0:
            warnings.warn("Branch length cannot be negative, the length will "
                          "not be changed")
            return

        self._length = value
    
    @property
    def inheritance_probability(self) -> float:
        """
        Get the inheritance probability (gamma) of the branch.
        
        Args:
            N/A
        Returns:
            float: The inheritance probability.
        """
        return self._inheritance_probability
    
    @inheritance_probability.setter
    def inheritance_probability(self, value : float) -> None:
        """
        Set the inheritance probability of the branch. Values outside [0, 1]
        are rejected with a warning and leave the current value untouched.
        
        Args:
            value (float): The new inheritance probability.
        Returns:
            N/A
        """
        if value is not None and (value < 0 or value > 1):
            warnings.warn("Inheritance probability must be between 0 and 1, "
                          "the inheritance probability will not be changed")
            return
        self._inheritance_probability = value
    
    @property
    def parent_id(self) -> str:
        """
        Get the label of the branch's parent node.
        
        Args:
            N/A
        Returns:
            str: The parent node label.
        """
        return self._parent_id
    
    @parent_id.setter
    def parent_id(self, value : str) -> None:
        """
        Set the label of the branch's parent node.
        
        Args:
            value (str): The new parent node label.
        Returns:
            N/A
        """
        self._parent_id = value
    
    def __str__(self) -> str:
        """
        Return a string representation of the branch.
        
        Args:
            N/A
        Returns:
            str: Human readable description of the branch.
        """
        return (f"Branch(length={self.length}, "
                f"inheritance_probability={self.inheritance_probability})")
    
    def __repr__(self) -> str:
        """
        Return a string representation of the branch.
        
        Args:
            N/A
        Returns:
            str: Human readable description of the branch.
        """
        return self.__str__()


class Node:
    """
    Node class that provides support for managing network constructs like 
    reticulation nodes and other phylogenetic attributes.
    """
    __slots__ = ('_Node__attributes', '_Node__is_retic', '_Node__name', 
                 '_Node__hash', '_Node__seq', '_Node__t', '_Node__is_dirty')

    def __init__(self, 
                 name : str, 
                 is_reticulation : bool = False, 
                 attr : Union[dict[Any, Any], None] = None,
                 seq : Union[DataSequence, None] = None,
                 t : Union[float, None] = None) -> None:
        """
        Initialize a node with a name, attribute mapping, and a hybrid flag.

        Args:
            name (str): A Node label.
            
            is_reticulation (bool, optional): Flag that marks a node as a 
                                              reticulation node if set to True. 
                                              Defaults to False.
            
            attr (dict[Any, Any], optional): Fill a mapping with any other user
                                             defined values. Defaults to a new
                                             empty dictionary.
            seq (Union[DataSequence, None], optional): A data sequence wrapper.
            t (Union[float, None], optional): A speciation time for this node.
        Returns:
            N/A
        """
        # A shared ``{}`` default would alias one dictionary across every Node
        # in every Network, so attributes written on one node would appear on
        # all of them.
        self.__attributes : dict[Any, Any] = {} if attr is None else attr
        self.__is_retic : bool = is_reticulation
        self.__name : str = name
        # Cache the (name-based) hash. ``__hash__`` is called tens of millions
        # of times per MCMC chain via node dict/set lookups; recomputing
        # ``hash(name)`` each time (a builtin call) was ~12% of chain runtime
        # in profiling. The cache is refreshed in ``set_name`` -- the sole
        # rename path (NodeSet.update -> set_name) -- so it stays consistent,
        # and collections already discard/re-add around renames.
        self.__hash : int = hash(name)
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

        Raises:
            NodeError: If the new time is negative.
        Args:
            new_t (float): The new speciation/hybridization time for this node.
        Returns:
            N/A
        """
        if new_t >= 0:
            self.__t = new_t 
        else:
            raise NodeError("Please set speciation time, t, to a non-negative number!")

    def __str__(self) -> str:
        """
        Create a string representation of a node.

        Args:
            N/A
        Returns:
            str: A string representation of the node.
        """
        return self.__name
    
    def _get_comparison_key(self) -> tuple[float, str]:
        """
        Get the comparison key for this node.
        
        For biological meaningfulness, nodes are ordered by:
        1. Primary: Time from root (t attribute) - closer to root (smaller t) comes first
        2. Fallback: Node name for deterministic ordering when times unavailable
        
        Note: For edge count fallback, use network.set_node_times_from_root() to establish
        times based on distances before comparison.
        
        Returns:
            tuple[float, str]: (time_or_inf, name) for comparison
        """
        if self.__t is not None:
            # Use actual time - closer to root (smaller time) sorts first
            return (self.__t, self.__name)
        else:
            # If no time is set, use infinity to sort after all timed nodes
            # Then use name for deterministic ordering among untimed nodes
            return (float('inf'), self.__name)
    
    def __lt__(self, other) -> bool:
        """
        Less than comparison for Node objects.
        Used by heapq and other sorting algorithms.
        
        Compares nodes by biological distance from root:
        1. Primary: Time from root (t attribute) - closer to root comes first
        2. Fallback: Node name for deterministic ordering when times unavailable
        
        Args:
            other: Another Node object to compare against
        Returns:
            bool: True if this node is closer to root than other node
        """
        if not isinstance(other, Node):
            return NotImplemented
        return self._get_comparison_key() < other._get_comparison_key()
    
    def __le__(self, other) -> bool:
        """
        Less than or equal comparison for Node objects.
        
        Args:
            other: Another Node object to compare against
        Returns:
            bool: True if this node is closer to or equal distance from root
        """
        if not isinstance(other, Node):
            return NotImplemented
        return self._get_comparison_key() <= other._get_comparison_key()
    
    def __gt__(self, other) -> bool:
        """
        Greater than comparison for Node objects.
        
        Args:
            other: Another Node object to compare against
        Returns:
            bool: True if this node is farther from root than other node
        """
        if not isinstance(other, Node):
            return NotImplemented
        return self._get_comparison_key() > other._get_comparison_key()
    
    def __ge__(self, other) -> bool:
        """
        Greater than or equal comparison for Node objects.
        
        Args:
            other: Another Node object to compare against
        Returns:
            bool: True if this node is farther from or equal distance from root
        """
        if not isinstance(other, Node):
            return NotImplemented
        return self._get_comparison_key() >= other._get_comparison_key()
    
    def __eq__(self, other) -> bool:
        """
        Equality comparison for Node objects.
        Two nodes are equal if they have the same name (identity).
        
        Note: This is based on node identity, not biological distance.
        
        Args:
            other: Another object to compare against
        Returns:
            bool: True if both are Node objects with the same name
        """
        if not isinstance(other, Node):
            return NotImplemented
        return self.__name == other.__name
    
    def __hash__(self) -> int:
        """
        Hash function for Node objects.
        Uses the node name for hashing to maintain consistency with equality.
        
        Returns:
            int: Hash value based on the node name
        """
        return self.__hash
    
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
        self.__hash = hash(new_name)
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

    def add_attribute(self, key : Any, value : Any) -> None:
        """
        Put a key and value pair into the node attribute dictionary.

        If the key is already present, it will overwrite the old value.
        
        Args:
            key (Any): Attribute key.
            value (Any): Attribute value for the key.
        Returns:
            N/A
        """
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
        return self.__attributes.get(key)
    
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
        Raises:
            NodeError: If no sequence record has been associated with this
                       node.
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
        # Shallow-copy the attribute mapping: sharing it would couple the
        # copy's attributes to the original's, which defeats the point.
        dopel = Node(self.__name, self.__is_retic, dict(self.__attributes))
        if self.__seq is not None:
            dopel.set_seq(self.__seq)
            
        if self.__t is not None:
            dopel.set_time(self.__t)
            
        dopel.__is_dirty = self.__is_dirty
        
        return dopel
    
class Edge:
    """
    Class for directed edges. 
    
    Instead of being a wrapper for a "set" of member nodes, we can now think 
    about an Edge as a wrapper for a tuple of member nodes (a, b), where 
    now the direction is encoded in the ordering (where a is the source, and b 
    the destination).
    """
    
    __slots__ = ('_src', '_dest', '_Edge__length', '_Edge__gamma', 
                 '_Edge__weight', '_Edge__tag')
    
    def __init__(self,
                 source : Node, 
                 destination : Node, 
                 length : float | None = None,
                 gamma : float | None = None,
                 weight : float | None = None,
                 tag : str | None = None
                 ) -> None:
        """
        An edge has a source (parent) and a destination (child). Edges in a 
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
        Get the source node of this edge.

        Args:
            N/A
        Returns:
            Node: The source node.
        """
        return self._src
    
    @property
    def dest(self) -> Node:
        """
        Get the destination node of this edge.

        Args:
            N/A
        Returns:
            Node: The destination node.
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
            str: The name/identifiabity tag for this edge. High change of being 
                 None!
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
        Raises:
            ValueError: If gamma is not between 0 and 1 inclusive.
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
            new_src (Node | None, optional): A new source node for this edge.
            new_dest (Node | None, optional): A new destination node for this
                                              edge.
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
        Set the branch length of this edge. If enforce_times is True, then the
        branch length must be equivalent to the difference in times of the
        source and destination nodes. If warn_times is True, then a warning
        will be raised if the branch length is not equivalent to the difference
        in times of the source and destination nodes.

        Raises:
            EdgeError: If enforce_times is True and the branch length is not 
                       equivalent to the difference in times of the source and 
                       destination nodes.
            Warning: If warn_times is True and the branch length is not
                     equivalent to the difference in times of the source and
                     destination nodes.
        Args:
            branch_length (float): The new branch length.
            warn_times (bool, optional): If True, raises a warning if the branch
                                         length is not equivalent to the difference
                                         in times of the source and destination
                                         nodes. Defaults to False.
            enforce_times (bool, optional): If True, raises an error if the branch
                                            length is not equivalent to the difference
                                            in times of the source and destination
                                            nodes. Defaults to False.
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
        Get the branch length of this edge. 
        
        Args:
            N/A
        Returns:
            float: The branch length of this edge.
        """
        return self.__length
    
    def set_weight(self, new_weight : float) -> None:
        """
        Set the weight of this edge.

        Args:
            new_weight (float): The new weight of this edge.
        Returns:
            N/A
        """
        self.__weight = new_weight
    
    def get_weight(self) -> float:
        """
        Get the weight of this edge.
    
        Args:
            N/A
        Returns:
            float: The weight of this edge.
        """
        return self.__weight
    
    def to_names(self) -> tuple[str, str]:
        """
        Get the names of the nodes in this edge.
        
        Args:
            N/A
        Returns:
            tuple[str, str]: A tuple of the names of the nodes in this edge.
        """
        return (self._src.label, self._dest.label)
    
    # NOTE: ``Edge`` deliberately does NOT implement ``__len__``.  Python's
    # data model requires ``__len__`` to return a non-negative integer;
    # returning the branch length (a float) caused ``bool(edge)`` and
    # ``if edge:`` expressions to raise ``TypeError`` wherever an ``Edge``
    # was truth-tested.  Use ``edge.get_length()`` for the branch length
    # and ``edge is not None`` for identity tests.

    def to_branch(self) -> Branch:
        """
        Convert this edge to a Branch object for use in phylogenetic
        computations.

        Returns:
            Branch: A Branch with this edge's length, gamma, and source
                    node label.
        """
        return Branch(self.__length, self.__gamma, self.src.label)
    
#########################
#### NETWORK CLASSES ####
#########################
class Network:
    """
    A directed (and potentially acyclic) graph of Nodes joined by Edges.
    
    An 'Edge' object is a wrapper class for a tuple of two nodes, (a, b),
    where a and b are Node objects, and the direction of the edge is from 
    a to b (a is b's parent) -- thus (a, b) is NOT the same as (b, a).

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
                 edges : Union[EdgeSet, set[Edge], None] = None, 
                 nodes : Union[NodeSet, set[Node], None] = None) -> None:
        """
        Initialize a Network object.
        You may initialize with any combination of edges/nodes,
        or provide none at all.
        
        If you provide edges and no nodes, each node present in those edges
        *WILL* be added to the network.

        Raises:
            TypeError: If any provided edge is not an Edge object.
        Args:
            edges (EdgeSet | set[Edge], optional): The edges of the network.
                                                   Defaults to no edges.
            nodes (NodeSet | set[Node], optional): The nodes of the network.
                                                   Defaults to no nodes.
        Returns:
            N/A
        """
        if isinstance(edges, set):
            edge_set = EdgeSet()
            edge_set.add(*edges)
            edges = edge_set
        
        if isinstance(nodes, set):
            node_set = NodeSet()
            node_set.add(*nodes)
            nodes = node_set
        
        self._edges : EdgeSet = EdgeSet() if edges is None else edges
        
        if nodes is not None:
            self._nodes : NodeSet = nodes
        else:
            # No node set given: derive V from the endpoints of E.
            self._nodes = NodeSet()
            for edge in self._edges.get_set():
                if type(edge) is not Edge:
                    raise TypeError("Networks are directed and take Edge "
                                    f"objects. Got a {type(edge).__name__}.")
                self._nodes.add(edge.src, edge.dest)
        
        # Blob storage for anything that you want to associate with 
        # this network. Just give it a string key!
        self.__items : dict[str, object] = {}
        
        # Counter behind add_uid_node's "UID_<n>" names.
        self.__uid : int = 0
        
        # Free floater nodes/edges are allowed.
        for edge in list(self._edges.get_set()):
            self._nodes.process(edge)
        
        self.__leaves : set[Node] = {node for node in self._nodes.get_set()
                                     if self._nodes.out_deg(node) == 0}
        self.__roots : set[Node] = {node for node in self._nodes.get_set()
                                    if self._nodes.in_deg(node) == 0}
    
    def __contains__(self, obj : Union[Node, Edge]) -> bool:
        """
        Allows a simple pythonic "n in network" or "e in network" check.
        
        Raises: 
            TypeError: If @obj is neither a Node nor an Edge.
        Args:
            obj (Node | Edge): A node or edge.
        Returns:
            bool: True if obj is a node or edge in the network.
        """
        if type(obj) is Edge:
            return obj in self._edges
        elif type(obj) is Node:
            return obj in self._nodes
        else:
            raise TypeError("Networks only contain Node and Edge objects.")
    
    def add_nodes(self, *nodes : Union[Node, list[Node]]) -> None:
        """
        Add any amount of nodes to this network.
        
        Accepts nodes as separate arguments, as lists, or as a mixture of
        both.
        
        Args:
            *nodes (Node | list[Node]): Any amount of node objects.
        Returns:
            N/A
        """
        self._nodes.add(*nodes)
                         
    def add_uid_node(self, node : Union[Node, None] = None) -> Node:
        """
        Ensure a node has a unique name that hasn't been used before/is 
        not currently in use for this graph.
        
        May be used with a node that is or is not yet an element of V.
        The node will be added to V as a result of this function.

        Args:
            node (Node | None): Any node object. Defaults to None.
        Returns:
            Node: the added/edited node that has been added to the graph.
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
    
    def uid_count(self) -> int:
        """
        Return the current value of the unique-id counter used by
        :meth:`add_uid_node`.

        Structural copies (see :meth:`Network.copy`) must carry this
        counter forward so that regenerated ``UID_*`` node names never
        collide with names already present in the graph.

        Args:
            N/A
        Returns:
            int: The next unique id that :meth:`add_uid_node` will use.
        """
        return self.__uid

    def set_uid_count(self, value : int) -> None:
        """
        Set the unique-id counter used by :meth:`add_uid_node`.

        Args:
            value (int): The value to assign to the counter.
        Returns:
            N/A
        """
        self.__uid = value
    
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
        Access the blob storage with a key. CONSIDER REMOVAL OF THIS FUNCTION...

        Args:
            key (str): a string key to access the blob storage
        Returns:
            object: the object stored in the blob storage
        """
        return self.__items[key]
    
    def put_item(self, key : str, item : Any) -> None:
        """
        Add an item to the blob storage.

        Args:
            key (str): a string key to access the blob storage
            item (Any): the object to store in the blob storage
        Returns:
            N/A
        """
        if key not in self.__items:
            self.__items[key] = item

    def has_node_named(self, name : str) -> Union[Node, None]:
        """
        Check whether the graph has a node with a certain name.
        Strings must be exactly equal (same white space, capitalization, etc.)

        Args:
            name (str): the name to search for

        Returns:
            Node | None: the node with the given name, or None if there is no 
                         such node
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
            list[Any]: the list of in-edges
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
            list[Any]: the list of out-edges
        """
        if node in self._nodes:
            return self._nodes.out_edges(node)
        else:
            warnings.warn("Attempting to get the out-edges of a node that is\
                not in the graph-- returning an empty list")
            return []

    @classmethod
    def from_newick(cls, newick : str) -> Network:
        """
        Construct a Network object by passing in a newick string and parsing out
        the topology from there.

        Args:
            newick (str): An extended newick string.
        Returns:
            Network: An initialized network object
        """
        def parse_newick_string(newick_str: str) -> Network:
            """
            Parse a Newick string directly into a Network object.
            
            This function handles:
            - Standard Newick format: (A,B)C;
            - Branch lengths: (A:0.1,B:0.2)C:0.0;
            - Internal node labels: (A,B)Internal1;
            - Reticulation nodes marked with #: ((A,#H1)B,(C,#H1)D)E;
            - Inheritance probabilities: #H1[&gamma=0.7]
            
            The parsing proceeds in three phases:
              1. **Tokenize** – split the raw string into structural tokens
                 (parentheses, commas, colons), names, branch lengths, and
                 bracket-comment blocks.
              2. **Parse** – recursively consume tokens to build an
                 intermediate ``NewickNode`` tree that mirrors the nested
                 parenthetical structure.
              3. **Build** – walk the ``NewickNode`` tree to create
                 ``Node`` / ``Edge`` objects and assemble the
                 ``Network``.  Reticulation nodes (names starting with
                 ``#``) are deduplicated via a shared ``node_map`` so that
                 the same ``Node`` instance receives edges from both
                 parents.

            Args:
                newick_str (str): A Newick-formatted string representing a phylogenetic network
                
            Returns:
                Network: A Network object representing the parsed structure
                
            Raises:
                NewickParserError: If the Newick string is malformed
            """
            # Imported here because :mod:`Newick` reaches this module
            # through :mod:`GeneTrees`.
            from .Newick import NewickParserError

            class NewickNode:
                """Helper class to represent nodes during parsing"""
                def __init__(self, name=None, length=None, children=None):
                    self.name = name
                    self.length = length if length is not None else 1.0
                    self.children = children if children is not None else []
                    self.comment = None
                    self.gamma = None
                    
            def tokenize(s):
                """Tokenize the Newick string"""
                tokens = []
                i = 0
                while i < len(s):
                    if s[i] in '(),;':
                        tokens.append(s[i])
                        i += 1
                    elif s[i] == ':':
                        # Branch length follows
                        tokens.append(':')
                        i += 1
                        j = i
                        while j < len(s) and s[j] not in '(),;:[':
                            j += 1
                        tokens.append(s[i:j])
                        i = j
                    elif s[i] == '[':
                        # Comment block
                        j = s.find(']', i)
                        if j == -1:
                            raise NewickParserError("Unclosed comment block")
                        tokens.append(s[i:j+1])
                        i = j + 1
                    elif s[i].isspace():
                        i += 1
                    else:
                        # Node name
                        j = i
                        while j < len(s) and s[j] not in '(),;:[]' and not s[j].isspace():
                            j += 1
                        if i < j:
                            tokens.append(s[i:j])
                        i = j
                return tokens
            
            def parse_comment(comment_str):
                """Parse comment block for gamma values"""
                if comment_str.startswith('[') and comment_str.endswith(']'):
                    content = comment_str[1:-1]
                    if content.startswith('&gamma='):
                        try:
                            gamma = float(content[7:])
                            return gamma
                        except ValueError:
                            pass
                return None
            
            def parse_tokens(tokens, idx=0):
                """Recursively parse tokenized Newick string"""
                if idx >= len(tokens):
                    raise NewickParserError("Unexpected end of string")
                    
                node = NewickNode()
                
                if tokens[idx] == '(':
                    # Internal node with children
                    idx += 1
                    children = []
                    
                    while True:
                        child, idx = parse_tokens(tokens, idx)
                        children.append(child)
                        
                        if idx >= len(tokens):
                            raise NewickParserError("Unexpected end of string")
                            
                        if tokens[idx] == ',':
                            idx += 1
                        elif tokens[idx] == ')':
                            idx += 1
                            break
                        else:
                            raise NewickParserError(f"Unexpected token: {tokens[idx]}")
                    
                    node.children = children
                    
                    # Parse internal node label if present
                    if idx < len(tokens) and tokens[idx] not in ':,);[':
                        node.name = tokens[idx]
                        idx += 1
                else:
                    # Leaf node
                    if tokens[idx] not in ':,);[':
                        node.name = tokens[idx]
                        idx += 1
                
                # Parse comment if present
                if idx < len(tokens) and tokens[idx].startswith('['):
                    node.gamma = parse_comment(tokens[idx])
                    node.comment = tokens[idx]
                    idx += 1
                
                # Parse branch length if present
                if idx < len(tokens) and tokens[idx] == ':':
                    idx += 1
                    if idx >= len(tokens):
                        raise NewickParserError("Expected branch length after ':'")
                    try:
                        node.length = float(tokens[idx])
                    except ValueError:
                        raise NewickParserError(f"Invalid branch length: {tokens[idx]}")
                    idx += 1
                    
                # Parse comment after branch length if present
                if idx < len(tokens) and tokens[idx].startswith('['):
                    if node.gamma is None:
                        node.gamma = parse_comment(tokens[idx])
                    node.comment = tokens[idx]
                    idx += 1
                
                return node, idx
            
            def build_network(newick_node, parent_phynet_node=None, network=None, node_map=None, time=0.0):
                """Convert parsed Newick structure to Network"""
                if network is None:
                    network = Network()
                    node_map = {}
                
                # Create or retrieve PhyNet node
                node_name = newick_node.name if newick_node.name else f"Internal{len(node_map)}"
                
                # Check if this is a reticulation node (already exists in map)
                if node_name in node_map:
                    phynet_node = node_map[node_name]
                else:
                    is_retic = node_name.startswith('#')
                    phynet_node = Node(name=node_name, is_reticulation=is_retic)
                    phynet_node.set_time(time)
                    network.add_nodes(phynet_node)
                    node_map[node_name] = phynet_node
                
                # Add edge from parent if exists
                if parent_phynet_node is not None:
                    edge = Edge(parent_phynet_node, phynet_node, length=newick_node.length)
                    if newick_node.gamma is not None:
                        edge.set_gamma(newick_node.gamma)
                    network.add_edges(edge)
                
                # Process children
                for child in newick_node.children:
                    build_network(child, phynet_node, network, node_map, time + child.length)
                
                return network
            
            # Main parsing logic
            try:
                # Clean the string
                newick_str = newick_str.strip()
                if not newick_str.endswith(';'):
                    raise NewickParserError("Newick string must end with semicolon")
                newick_str = newick_str[:-1]  # Remove semicolon
                
                # Tokenize and parse
                tokens = tokenize(newick_str)
                if not tokens:
                    raise NewickParserError("Empty Newick string")
                    
                root, _ = parse_tokens(tokens)
                
                # Build network
                network = build_network(root)
                
                # Clean up the network
                network.clean([True, True, True])
                
                return network
                
            except Exception as e:
                if isinstance(e, NewickParserError):
                    raise
                else:
                    raise NewickParserError(f"Failed to parse Newick string: {str(e)}")
        #parse nodes and edges from topology
        return parse_newick_string(newick)
    
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
            # Copy the edge sets to avoid modifying during iteration
            in_edges = self._nodes.in_edges(node).copy()
            out_edges = self._nodes.out_edges(node).copy()

            for edge in in_edges:
                self.remove_edge(edge)
            for edge in out_edges:
                self.remove_edge(edge)

            self._nodes.remove(node)

            # Drop the node from the leaf/root caches.  Edge removal above
            # reclassifies a soon-to-be-deleted node as a leaf and/or root
            # (out-deg 0 / in-deg 0); without this discard those stale
            # references pile up in ``__leaves`` / ``__roots`` across a long
            # search, bloating every ``copy.deepcopy`` in the MCMC undo path.
            self.__leaves.discard(node)
            self.__roots.discard(node)
                     
    def remove_edge(self, 
                    edge : Union[Edge, list[Node]], 
                    gamma : float | None = None) -> None:
        """    
        Removes edge from the list of edges. Does not delete nodes with no edges
        Has no effect if 'edge' is not in the graph.
        
        Raises:
            NetworkError: If 'edge' is neither an Edge nor a [src, dest] pair.
        Args:
            edge (Edge | list[Node]): an edge to remove from the graph
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
        # O(1) EdgeSet membership instead of building the full edge list
        # (O(E)) up to twice per call; remove_edge is called on every move.
        edge_present = edge in self._edges
        if edge_present and type(edge) is Edge:
            # Remove the edge from the edge set
            self._edges.remove(edge)
        
            #Make the edge set aware of the edge removal
            self._nodes.process(edge, removal = True)
            
            # Reclassify the nodes, as they may be leaves/roots/etc now.
            self.__reclassify_node(edge.src, True, False)
            self.__reclassify_node(edge.dest, False, False)
        elif not edge_present and type(edge) is Edge:
            return
        else:
            raise NetworkError(f"Tried to remove a {type(edge).__name__} from "
                               "a Network, which holds only Edge objects.")
    
    def update_node_name(self, node : Node, name : str) -> None:
        """
        Rename a node and update *all* bookkeeping (NodeSet, EdgeSet,
        leaf/root sets).

        Node.__hash__ is name-based, so the node must be pulled out of every
        hash-keyed collection *before* the rename and put back *after*.

        Args:
            node (Node): a node in the graph
            name (str): the new name for the node.
        Returns:
            N/A
        """
        is_leaf = node in self.__leaves
        is_root = node in self.__roots
        self.__leaves.discard(node)
        self.__roots.discard(node)

        affected = self._nodes.in_edges(node) | self._nodes.out_edges(node)
        self._nodes.update(node, name)
        self._edges.rehash_node(node, affected)

        if is_leaf:
            self.__leaves.add(node)
        if is_root:
            self.__roots.add(node)

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
                    self.__leaves.discard(node)
                if self._nodes.in_deg(node) == 0:
                    self.__roots.add(node)
            else:
                # If in_degree now = 1, then the node was previously a root,
                # and is not anymore
                if self._nodes.in_deg(node) == 1:
                    self.__roots.discard(node)
                if self._nodes.out_deg(node) == 0:
                    self.__leaves.add(node)
        else:
            if is_par:
                # if out degree is now = 0, then the node is now a leaf
                if self._nodes.out_deg(node) == 0:
                    self.__leaves.add(node)
            else:
                # if in degree is now = 0, the node is now a root
                if self._nodes.in_deg(node) == 0:
                    self.__roots.add(node)
                
    def root(self) -> Node:
        """
        Return the root of the Network. Phylogenetic networks only have one 
        root, but for generality and practical use, multiple roots have been 
        allowed. To get all roots, should multiple exist, call the function
        "roots". This function only returns 1 root.

        Raises:
            NetworkError: If there are no roots in the network (cycle, or empty),
                          or if there are multiple roots in the network.
        Args:
            N/A
        Returns:
            Node: The root of the network.
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
        Return all root(s) of the Network. Phylogenetic networks typically have
        one root, but for generality and practical use, multiple roots have
        been allowed. Use ``root()`` to retrieve a single root.

        Returns:
            list[Node]: A list of all root nodes (nodes with in-degree 0 and
                        out-degree > 0).
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
        There is no hard cap on the length of this array.

        Args:
            node (Node): any node in V.
        Returns:
            list[Node]: the parents of the node.
        Raises:
            NetworkError: If the node is not in the graph.
        """
        try:
            return [edge.src for edge in self._nodes.in_edges(node)]
        except:
            raise NetworkError("Attempted to calculate parents of a node that \
                is not in the graph.")
        
    def get_children(self, node : Node) -> list[Node]:
        """
        Returns a list of the children of a node.
        There is no hard cap on the length of this array.

        Args:
            node (Node): any node in V.
        Returns:
            list[Node]: the children of the node.
        Raises:
            NetworkError: If the node is not in the graph.
        """
        try:
            return [edge.dest for edge in self._nodes.out_edges(node)]
        except:
            raise NetworkError("Attempted to calculate children of a node that \
                is not in the graph.")
    
    def get_branches(self, node : Node) -> dict[str, Any]:
        """
        Returns a dictionary of branches connected to the node.
        
        Args:
            node (Node): a node in the graph.
        Returns:
            dict[str, Any]: a dictionary of branches connected to the node.
                            The keys are "parent_branches" and "child_branches".
                            The values are lists of branches.
        """
        return {
            "parent_branches": [edge.to_branch() for edge in self._nodes.in_edges(node)],
            "child_branches": [edge.to_branch()for edge in self._nodes.out_edges(node)]
        }

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
            options (list[bool], optional): A list of booleans that signal 
                                            which algorithms to run. Defaults to 
                                            [True, True, True].
        Returns:
            N/A
        """
        _nopt.clean(self, options)

    def mrca(self, set_of_nodes: set[Node] | set[str]) -> Node:
        """
        Computes the Least Common Ancestor of a set of graph nodes.

        Args:
            set_of_nodes (set[Node] | set[str]): A set of Nodes, or node names.
        Returns:
            Node: The node that is the LCA of the set.
        Raises:
            NetworkError: If any node in the set is not in the graph, or if
                          elements are of an unexpected type.
        """
        return _nopt.mrca(self, set_of_nodes)
                      
    def leaf_descendants(self, node : Node) -> set[Node]:
        """
        Compute the set of all leaf nodes that are descendants of the parameter 
        node. Uses DFS to find paths to leaves.

        Args:
            node (Node): The node for which to compute leaf children.
        Returns:
            set[Node]: The set of all leaves that descend from 'node'.
        Raises:
            NetworkError: If node is not found in the graph.
        """
        return _nopt.leaf_descendants(self, node)
        
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
        return _nopt.diff_subtree_edges(self, rng)
    
    def subgenome_count(self, n : Node) -> int:
        """
        Given a node in this graph, return the subgenome count.
         
        Args:
            n (Node): Any node in the graph. 
                      It is an error to input a node that is not in the graph.
        Returns:
            int: subgenome count
        Raises:
            NetworkError: If the input node is not in the graph.
        """
        
        return _nopt.subgenome_count(self, n)
            
    def edges_downstream_of_node(self, n : Node) -> list[Edge]:
        """
        Returns the set (as a list) of edges that are in the subgraph of a node.

        Args:
            n (Node): A node in a graph.
        Returns:
            list[Edge]: The set of all edges in the subgraph of n.
        Raises:
            NetworkError: If the input node is not in the graph.
        """
        return _nopt.edges_downstream_of_node(self, n)
    
    def edges_upstream_of_node(self, n : Node) -> list[Edge]:
        """
        Returns the set (as a list) of edges that are in all paths from the root
        to this node.
        
        Useful in avoiding the creation of cycles when adding edges.

        Args:
            n (Node): A node in a graph.
        Returns:
            list[Edge]: The set of all edges on paths from the root to n.
        Raises:
            NetworkError: If the input node is not in the graph.
        """
        return _nopt.edges_upstream_of_node(self, n)
    
    def subgenome_ct_edges(self, 
                           downstream_node : Union[Node, None]= None, 
                           delta : float = math.inf, 
                           start_node : Union[Node, None] = None) \
                           -> dict[Edge, int]:
        """
        Maps edges to their subgenome counts.
        
        Raises:
            NetworkError: If the graph has more than one root to start.
        Args:
            downstream_node (Node | None, optional): No edges will be included in the
                                              map that are in a subgraph of this 
                                              node. Defaults to None.
            delta (float, optional): Only include edges in the mapping that have
                                     subgenome counts <= delta. 
                                     Defaults to math.inf.
            start_node (Node | None, optional): Provide a node only if you don't want 
                                         to start at the root. 
                                         Defaults to None.
        Returns:
            dict[Edge, int]: a map from edges to subgenome counts
        """
    
        return _nopt.subgenome_ct_edges(self, downstream_node, delta,
                                        start_node)
                
    def edges_to_subgenome_count(self, 
                                 downstream_node : Union[Node, None]= None, 
                                 delta : float = math.inf, 
                                 start_node : Union[Node, None] = None) \
                                 -> dict[int, list[Edge]]:
        """
        Maps edges to their subgenome counts.
        
        
        Raises:
            NetworkError: If the graph has more than one root to start.
        Args:
            downstream_node (Node | None, optional): No edges will be included in the
                                              map that are in a subgraph of this 
                                              node. Defaults to None.
            delta (float, optional): Only include edges in the mapping that have
                                     subgenome counts <= delta. 
                                     Defaults to math.inf.
            start_node (Node | None, optional): Provide a node only if you don't want 
                                         to start at the root. 
                                         Defaults to None.
        Returns:
            dict[int, list[Edge]]: a map from edges to subgenome counts
        """
    
        return _nopt.edges_to_subgenome_count(self, downstream_node, delta,
                                              start_node)

    def leaf_descendants_all(self) -> dict[Node, set[Node]]:
        """
        Map each node in the graph to its set of leaf descendants
        
        Args:
            N/A
        Returns:
            dict[Node, set[Node]]: map from graph nodes to their 
                                   leaf descendants
        """
        return _nopt.leaf_descendants_all(self)
    
    def newick(self) -> str:
        """
        Generate the newick string of the network.

        Args:
            N/A
        Returns:
            str: The newick representation of the network.
        """
        return _nopt.newick(self)
    
    def is_acyclic(self) -> bool:
        """
        Checks if each of this graph's connected components is acyclic

        Args:
            N/A
        Returns:
            bool: True if acyclic, False if cyclic. 
        """
        return _nopt.is_acyclic(self)
    
    def bfs_dfs(self, 
                start_node : Union[Node, None] = None,
                dfs : bool = False, 
                is_connected : bool = False, 
                accumulator : Callable[..., None] | None = None, 
                accumulated : Any = None) -> tuple[dict[Node, int], Any]:
        """
        General bfs-dfs routine, with the added utility of checking 
        whether or not this graph is made up of multiple connected components.

        Args:
            start_node (Node | None, optional): Give a node to start the search from. 
                                         Defaults to None, in which case the 
                                         search will start at the root.
            dfs (bool, optional): Flag that specifies whether to use bfs or dfs. 
                                  Defaults to False (bfs), if true is passed, 
                                  will run dfs.
            is_connected (bool, optional): Flag that, if enabled, will check for 
                                           the connected component status. 
                                           Defaults to False (won't run).
            accumulator (Callable[..., None] | None, optional): A function that takes the 
                                              currently searched Node in the 
                                              graph and does some sort 
                                              of bookkeeping.
            accumulated (Any): Any type of structure that stores the data 
                               given by the accumulator function.

        Returns:
            tuple[dict[Node, int], Any]: Mapping from nodes to their distance 
                             from the start node.
        """
        return _nopt.bfs_dfs(self, start_node, dfs, is_connected,
                             accumulator, accumulated)
         
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
        Make a copy of a subnetwork of this DAG, rooted at 'retic_node', 
        with unique node names.
        
        Args:
            retic_node (Node): A node in this network that is a reticulation 
                               node
        Returns:
            Network: A subnetwork of the DAG being operated on
        """
        return _nopt.subnet(self, retic_node)
    
    def copy(self) -> tuple[Network, dict[Node, Node]]:
        """
        Copy this network into a new network object, also with new node and 
        edge objects.

        Args:
            N/A
        Returns:
            tuple[Network, dict[Node, Node]]: A tuple of the copied network 
                                              and a map from old nodes to new 
                                              nodes.
        """
        return _nopt.copy(self)
    
    def to_networkx(self) -> nx.Graph:
        """
        Convert this network to a NetworkX graph object.
        
        Args:
            N/A
        Returns:
            nx.Graph: A NetworkX graph object.
        """
        return _nopt.to_networkx(self)

    def is_isomorphic(self, other: Network) -> bool:
        """
        Check if this network is topologically isomorphic to another network.
        
        Two networks are isomorphic if they have the same topology, regardless of
        node labels or branch lengths. This method calls the GraphUtils.is_isomorphic
        function.
        
        Args:
            other (Network): The other network to compare against.
        
        Returns:
            bool: True if the networks are topologically isomorphic, False otherwise.
        """
        from .GraphUtils import is_isomorphic
        return is_isomorphic(self, other)
    
    def rooted_triplet_distance(self, other: Network) -> float:
        """
        Computes the rooted triplet distance between two networks.
        
        This distance measure compares the topological relationships between all
        triplets of leaves in both networks.  For each triplet of leaves
        {x, y, z}, the pair sharing the deepest MRCA are identified as
        siblings, producing a canonical encoding.  The distance is the
        normalized symmetric difference of these triplet sets.
        
        Args:
            other (Network): The other network to compare against.
        
        Returns:
            float: The rooted triplet distance (0 = identical, 1 = completely different).
        """
        
        return _nopt.rooted_triplet_distance(self, other)

    def get_subtree_at(self, node: Node) -> set[Node]:
        """
        Collect all nodes in the subtree rooted at the given node.

        Args:
            node (Node): The root of the subtree.

        Returns:
            set[Node]: A set of nodes in the subtree.
        """
        return _nopt.get_subtree_at(self, node)

    def dist_from_root(self, node : Node) -> int:
        """
        Get the distance from the root to the given node, measured in edge
        count (hop count) via BFS.

        Args:
            node (Node): A node in this network.
        Returns:
            int: The number of edges on the BFS-shortest path from the root
                 to the given node.
        """
        return _nopt.dist_from_root(self, node)
    
    def topological_order(self) -> list[Node]:
        """
        Compute a topological order of nodes in an acyclic network.

        Returns:
            list[Node]: Nodes in topological order from roots to leaves.
        Raises:
            NetworkError: If the network contains a directed cycle and no
                          valid topological order exists.
        """
        return _nopt.topological_order(self)
    
    def distance_from_root(self, node: Node, use_time: bool = True) -> float:
        """
        Compute distance from root to a given node.
        
        Args:
            node (Node): The target node
            use_time (bool): If True, use branch lengths/times. If False, use edge count.
            
        Returns:
            float: Distance from root (time units or edge count)
            
        Raises:
            NetworkError: If node is not in the network or no path to root exists
        """
        return _nopt.distance_from_root(self, node, use_time)
    
    def set_node_times_from_root(self) -> None:
        """
        Set time attributes for all nodes based on their distance from root.
        Uses cumulative branch lengths from root. Useful for enabling 
        biologically meaningful node comparisons.

        The root node is assigned time 0.0 and each descendant's time is the
        sum of branch lengths along the BFS path from the root.

        Returns:
            None
        """
        _nopt.set_node_times_from_root(self)
    
    def set_node_times_by_edge_count(self) -> None:
        """
        Set time attributes for all nodes based on their edge count from root.
        This provides biologically meaningful comparison when actual times are
        not available. Edge counts are converted to time values for comparison
        purposes.

        The root node is assigned time 0.0 and each descendant's time equals
        the number of edges on the BFS path from the root.

        Returns:
            None
        """
        _nopt.set_node_times_by_edge_count(self)

class MUL(Network):
    """
    A subclass of a Network, that is a binary tree that results from the 
    transformation of a standard network into a Multilabeled Species Tree.
    """
    def __init__(self, 
                 gene_map : dict[str, list[str]],
                 rng : np.random.Generator) -> None:
        """
        Initializes a MUL object.

        Args:
            gene_map (dict[str, list[str]]): A mapping from gene names to 
                                             a set of gene copy names.
            rng (np.random.Generator): A numpy random number generator.
        Returns:    
            N/A
        """
        super().__init__()
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

        # Work on an independent copy of the network so the input is untouched.
        mul_tree, _ = net.copy()

        # Expand reticulations until none remain. The MUL tree of a network is
        # unique, so the construction MUST be deterministic and independent of
        # set-iteration order. The previous bottom-up, leaf-driven traversal
        # chose which parent kept the original subnetwork (vs. a duplicate) by
        # the iteration order of the parent set, and pushed work back onto a
        # queue using the iteration order of child sets. For reticulations that
        # sit directly above a single leaf this is harmless, but for a
        # reticulation above a non-trivial (and possibly nested) clade -- e.g.
        # the (t,u) and (w,x,y,z) hybrid clades in DEFJ scenario J -- different
        # orderings produced structurally different (and incorrect) MUL trees,
        # making the parsimony score random rather than a function of the
        # network. We instead expand reticulations deterministically.
        #
        # Key invariant: always expand a *lowest* reticulation -- one with no
        # reticulation among its descendants -- so the subnetwork being copied
        # is already a pure tree. This is what makes a single ``subnet`` copy a
        # faithful duplicate and keeps the result order-independent.
        def _descendants(node : Node) -> set[Node]:
            """Strict descendants of 'node' in the working MUL tree."""
            return mul_tree.get_subtree_at(node) - {node}

        while True:
            retics = [n for n in mul_tree.V() if mul_tree.in_degree(n) >= 2]
            if not retics:
                break

            # Pick a reticulation with no reticulation descendants (a "lowest"
            # one). Tie-break by label for a fully deterministic order.
            retic_set = set(retics)
            lowest = sorted(
                (r for r in retics if not (_descendants(r) & retic_set)),
                key=lambda r: r.label,
            )
            target = lowest[0]

            # Keep the subnetwork under the first parent (sorted by label) and
            # graft an independent copy of it under every additional parent.
            parents = sorted(mul_tree.get_parents(target), key=lambda p: p.label)
            for parent in parents[1:]:
                subtree = mul_tree.subnet(target)
                mul_tree.remove_edge([parent, target])
                mul_tree.add_nodes(subtree.V())
                for edge in subtree.E():
                    mul_tree.add_edges(edge)
                mul_tree.add_edges(Edge(parent, subtree.root()))

        # Get rid of excess degree-2 connection nodes left by the expansion.
        mul_tree.clean([True, True, True])

        # Rename tips based on the gene mapping. Iterate leaves in sorted label
        # order so the (arbitrary) assignment of homeolog labels to the
        # duplicated copies of a species is reproducible across runs.
        #
        # Reticulation expansion duplicates hybrid subtrees, so a species that
        # sits under (or is) a reticulation needs as many allele/subgenome
        # labels in ``gene_map`` as MUL tips of that species. A 1:1 identity
        # map is not enough when the network has hybrids.
        #
        # Network leaf labels are species names; ``subnet`` appends
        # ``_copyN`` when duplicating. Allele labels in ``gene_map`` values
        # may be any strings and are applied only in this rename step.
        for leaf in sorted(mul_tree.get_leaves(), key=lambda l: l.label):
            species = _mul_tip_species(leaf.label, self.gene_map)
            if species is None:
                raise NetworkError(
                    f"Species tip '{leaf.label}' (after reticulation "
                    f"expansion) does not match any key in the "
                    f"allele/subgenome map. Map keys: "
                    f"{sorted(self.gene_map.keys())}."
                )
            remaining = copy_gene_map[species]
            if not remaining:
                provided = len(self.gene_map.get(species, []))
                raise NetworkError(
                    f"Allele/subgenome map has too few copies for species "
                    f"'{species}' (provided {provided}). Reticulation "
                    f"expansion created more MUL tips for this taxon than "
                    f"the map supplies. Provide one distinct allele label "
                    f"per parental lineage of each hybrid (e.g. "
                    f"'{species}': ['{species}_a', '{species}_b'] for a "
                    f"single reticulation involving '{species}')."
                )
            new_name: str = remaining.pop()
            mul_tree.update_node_name(leaf, new_name)

        leftovers = {
            sp: copies for sp, copies in copy_gene_map.items() if copies
        }
        if leftovers:
            detail = ", ".join(
                f"'{sp}' has {len(copies)} unused "
                f"({'/'.join(copies)})" for sp, copies in sorted(leftovers.items())
            )
            raise NetworkError(
                f"Allele/subgenome map has more copies than MUL tips after "
                f"reticulation expansion: {detail}. Reduce the map to one "
                f"allele label per MUL tip for each species."
            )

        self.mul = mul_tree

        return mul_tree


def _mul_tip_species(leaf_label: str,
                     gene_map: dict[str, list[str]]) -> str | None:
    """
    Recover the species key for a MUL tip label produced by reticulation
    expansion.

    Tips are either the original network leaf name (a ``gene_map`` key) or
    that name plus a ``_copyN`` suffix from :meth:`Network.subnet`. Allele
    labels in ``gene_map`` values are unrelated and are not consulted here.
    """
    if leaf_label in gene_map:
        return leaf_label

    # Strip trailing ``_copy<digits>`` suffixes (may nest for nested retics).
    label = leaf_label
    while True:
        match = re.fullmatch(r"(.*)_copy\d+", label)
        if match is None:
            break
        label = match.group(1)
        if label in gene_map:
            return label

    return None


# ---------------------------------------------------------------------------
# Backdoor optimization module. Imported at the *bottom* of the file so that
# ``_network_optimizations`` can ``from .Network import NetworkError`` at its
# own top level without a circular-import failure (all class/exception names it
# needs are already defined by the time we get here). ``Network`` methods
# delegate their heavy graph algorithms to ``_nopt`` -- keeping this module a
# lightweight data-structure + public-API layer.
from . import _network_optimizations as _nopt  # noqa: E402

