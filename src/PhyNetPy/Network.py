""" 
Author : Mark Kessler
Last Edit : 5/10/24
First Included in Version : 1.0.0
Docs   - [x]
Tests  - [ ] 
Design - [ ]
"""

from __future__ import annotations
from collections import defaultdict, deque
import copy
import math
import random
from typing import Any, Callable
import warnings
import numpy as np
import sys
import networkx as nx

sys.setrecursionlimit(300)


##########################           
#### HELPER FUNCTIONS ####
##########################

def leaf_desc_help(net : Network, node : Node, leaves : list[Node], 
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
            for child in net.get_children(node):
                child_desc = leaf_desc_help(net, child, leaves, desc_map)
                #A node's leaf descendant set is the union of all its children's
                #leaf descendant sets
                desc_map[node] = desc_map[node].union(child_desc)
                    
    return desc_map[node]    

def newick_help(net : Network, node : Node, processed_retics : set[Node]):
    """
    Helper function to "newick". Generates the newick string (sans ending 
    semicolon) of the subnetwork defined by 'node'.

    Args:
        net (Network): A Network
        node (Node): a Node in 'net'
        processed_retics (set[Node]): _description_

    Returns:
        _type_: _description_
    """
        
    if node in net.get_leaves():
        return node.get_name()
    else:
        if net.in_degree(node) >= 2 and node in processed_retics:
            if node.get_name()[0] != "#":
                return "#" + node.get_name()
            return node.get_name()
        else:
            if net.in_degree(node) >= 2:
                processed_retics.add(node)
                if node.get_name()[0] != "#":
                    node_name = "#" + node.get_name()
                else:
                    node_name = node.get_name()
            else:
                node_name = node.get_name()    
                
            substr = "("
            for child in net.get_children(node):
                substr += newick_help(net, child, processed_retics)
                substr += ","
            substr = substr[0:-1]
            substr += ")"
            substr += node_name
            
            return substr

def random_object(mylist : list[object], rng : np.random.Generator) -> object:
    """
    Select a random item from a list using an rng object 
    (for testing consistency and debugging purposes)

    Args:
        mylist (list[object]): a list of any type
        rng (np.random.Generator) : the result of a .default_rng(seed) call

    Returns:
        object : an item from mylist
    """
    rand_index = rng.integers(0, len(mylist))
    return mylist[rand_index]

def dict_merge(a : dict, b: dict):
    merged = {}
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

#############################
#### EXCEPTION SPECIFICS ####
#############################

class NetworkError(Exception):
    """
    This exception is raised when a network is malformed, 
    or if a network operation fails.
    """
    def __init__(self, message = "Error with a Graph Instance"):
        self.message = message
        super().__init__(self.message)

class NodeError(Exception):
    """
    This exception is raised when a Node operation fails.
    """
    def __init__(self, message = "Error in Node Class"):
        super().__init__(message)
        
class EdgeError(Exception):
    """
    This exception is raised when an Edge operation fails.
    """
    def __init__(self, message = "Error in Edge Class"):
        super().__init__(message)

##########################
#### NODES AND EDGES #####
##########################

class Node:
    """
    Node class that provides support for managing network constructs like 
    reticulation nodes and other phylogenetic attributes.
    """

    def __init__(self, name : str = None, is_reticulation : bool = False, 
                 attr : dict = dict()) -> None:
        """
        Initialize a node with a name, attribute mapping, and a hybrid flag.

        Args:
            name (str, optional): A Node name. Defaults to None, but nodes need
                                  to be named, in general.
            is_reticulation (bool, optional): Flag that marks a node as a 
                                              reticulation node if set to True. 
                                              Defaults to False.
            attr (dict, optional): Fill a mapping with any other user defined 
                                   values. Defaults to an empty dictionary.
        """
        
        self.attributes = attr
            
        self.is_retic : bool = is_reticulation
        self.name : str = name
        self.seq : list[str] = None
        self.t : float = None
        self.is_dirty : bool = False
    
    def get_time(self) -> float:
        """
        Get the speciation time for this node.
        
        Closer to 0 implies a time closer to the origin (the root). A larger 
        time implies a time closer to the present (leaves). 
        
        Returns:
            float: Speciation time, typically in coalescent units.
        """
        return self.t
    
    def set_time(self, t : float) -> None:
        """
        Set the speciation time for this node. The arg 't' must be a 
        non-negative number.

        Args:
            t (float): _description_
        """
        if t >= 0:
            self.t = t 
        else:
            raise NodeError("Please set speciation time, t, to a non-negative\
                             number!")
        
    def as_string(self) -> str:
        """
        Create a description of a node and summarize its attributes.

        Returns:
            str: A string description of the node.
        """
        myStr = "Node " + str(self.name) + ": "
        myStr += "t = " + str(round(self.t, 4)) + " "
        myStr += " is a reticulation node? " + str(self.is_retic)
        myStr += " has attributes: " + str(self.attributes)

        return myStr

    def get_name(self) -> str:
        """
        Returns the name of the node

        Returns:
            str: Node label.
        """
        return self.name

    def set_name(self, new_name : str) -> None:
        """
        Sets the name of the node to new_name.
    
        Args:
            new_name (str): A new string label for this node.
        """
        self.name = new_name
        self.is_dirty = True

    def set_is_reticulation(self, is_retic : bool) -> None:
        """
        Sets whether a node is a reticulation Node (or not).

        Args:
            is_retic (bool): Hybrid flag. True if this node is a 
                             reticulation node, false otherwise
        """
        self.is_retic = is_retic

    def is_reticulation(self) -> bool:
        """
        Retrieves whether a node is a reticulation Node (or not)

        Returns:
            bool: True, if this node is a reticulation. False otherwise.
        """
        return self.is_retic

    def add_attribute(self, key : Any, value : Any, 
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
        """
        
        if append:
            if key in self.attributes.keys():
                content = self.attributes[key]
        
                if type(content) is dict:
                    content = dict_merge(content, value)
                    self.attributes[key] = content
                elif type(content) is list:
                    content.extend(value)
                    self.attributes[key] = content
        else:
            self.attributes[key] = value

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
        if key in self.attributes.keys():
            return self.attributes[key]
        else:
            return None
    
    def set_seq(self, sequence : str) -> None:
        """
        Associate a data sequence with this node, if this node is a leaf in a 
        network.

        Args:
            sequence (str): A data sequence, in characters. It is recommended to
                            use the alphabet and matrix classes to obtain this.
        """
        self.seq = sequence
    
    def get_seq(self) -> str:
        """
        Gets the data sequence associated with this node.

        Returns:
            str: Data sequence.
        """
        return self.seq
    
    def duplicate(self) -> Node:
        """
        Duplicate this node by copying all data into a separate Node object.
        
        Useful for crafting copies of networks without having to deep copy. 
        NOTE: A node may only be a member of one network.

        Returns:
            Node: _description_
        """
        dopel = Node(self.name, self.is_retic, self.attributes)
        dopel.set_seq(self.seq)
        if self.t is not None:
            dopel.set_time(self.t)
        dopel.is_dirty = self.is_dirty
        
        return dopel
    
class NodeSet:
    """
    Data structure that is in charge of managing the nodes that are in a given
    Network
    """
    
    def __init__(self) -> None:
        """
        Initialize an empty set of network nodes
        """
        self.nodes : set[Node] = set()
        self.in_degree : dict[Node, int] = defaultdict(int)
        self.out_degree : dict[Node, int] = defaultdict(int)
        self.in_map : dict[Node, list[Edge]] = defaultdict(list)
        self.out_map : dict[Node, list[Edge]] = defaultdict(list)
        self.node_names : dict[str, Node] = {}

    def add(self, node : Node) -> None:
        """
        Add a node to the network node set.

        Args:
            node (Node): A new node to put in the network.
        """
        if node not in self.nodes:
            self.nodes.add(node)
            self.node_names[node.get_name()] = node
    
    def ready(self, edge : Edge) -> bool:
        """
        Check if an edge is allowed to be added to the network (both nodes must
        be in the node set before an edge can be added)

        Args:
            edge (Edge): A potential new network edge.

        Returns:
            bool: True if edge can be safely added, False otherwise.
        """
        return edge.src in self.nodes and edge.dest in self.nodes
    
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
        return self.in_degree[node]

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
        return self.out_degree[node]
    
    def in_edges(self, node : Node) -> list[Edge]:
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
        return self.in_map[node]
    
    def out_edges(self, node : Node) -> list[Edge]:
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
        return self.out_map[node]
    
    def process(self, edge : Edge, removal : bool = False) -> None:
        """
        Keep track of network data (in/out degrees, in/out edge maps) upon 
        the addition or removal of an edge for a network.

        Args:
            edge (Edge): The edge that is being added or removed
            removal (bool, optional): False if edge is being added, True if 
                                      edge is being removed. Defaults to False.
        """
        if self.ready(edge):
            if not removal:
                self.out_degree[edge.src] += 1
                self.in_degree[edge.dest] += 1
                self.out_map[edge.src].append(edge)
                self.in_map[edge.dest].append(edge)
            else:
                self.out_degree[edge.src] -= 1
                self.in_degree[edge.dest] -= 1
                self.out_map[edge.src].remove(edge)
                self.in_map[edge.dest].remove(edge)
                
    def get_set(self) -> set[Node]:
        """
        Grab the set of nodes.

        Returns:
            set[Node]: V, the node set of a network.
        """
        return self.nodes
    
    def remove(self, node : Node) -> None:
        """
        Remove a node from V, and update necessary mappings.

        Args:
            node (Node): Node to remove from the network.
        """
        if node in self.nodes:
            self.nodes.remove(node)
            del self.in_degree[node]
            del self.out_degree[node]
            del self.out_map[node]
            del self.in_map[node]  
            del self.node_names[node.get_name()] 

class Edge:
    """
    Class that encodes node to node relationships, as well as any applicable 
    bookkeeping data and parameters that are applicable to edges.
    """
    
    def __init__(self, source : Node, destination : Node) -> None:
        """
        An edge has a source (parent) and a destination (child). Edges in a 
        phylogenetic context are always directed.

        Args:
            source (Node): Parent node.
            destination (Node): Child node.
        """
        self.src : Node = source
        self.dest : Node = destination
        self.gamma : float = None
        
        #If the source and destination nodes already have defined times, 
        # go ahead and use them.
        if source.get_time() is not None and destination.get_time() is not None:
            self.length : float  = destination.get_time() - source.get_time()
        else:
            self.length : float = None
        
    def ready(self, node : Node | str) -> bool:
        """
        Check to see if a node is in an edge. Checking via node name is also 
        supported.
        
        Args:
            node (Node | str): Look up based on object or name.

        Returns:
            bool: True if the node is in the edge either as the source node or 
                  destination node, False if not.
        """
        if type(node) == str:
            if node == self.src.get_name() or node == self.dest.get_name():
                return True
            return False
        else:
            if node == self.src or node == self.dest:
                return True
            return False
    
    def set_length(self, length : float) -> None:
        """
        Sets the branch length of this edge.

        Args:
            length (float): a branch length value (>0).
        """
        self.length = length
    
    def get_length(self) -> float:
        """
        Gets the branch length of this edge. If it is not equivalent to the
        current times of the source and destination nodes, then a warning
        is raised that the branch lengths are not equivalent, and that either
        the times are outdated or the branch length field is outdated.

        Returns:
            float: branch length.
        """
        srct = self.src.get_time() 
        destt = self.dest.get_time()
        if srct is not None and destt is not None:
            if self.length != destt - srct:
                warnings.warn("This edge has a length field set to a number \
                               that is not equal to the differences in time for\
                               its source and destination!")
        return self.length

    def set_gamma(self, gamma : float) -> None:
        """
        Set the inheritance probability of this edge. Only applicable to 
        hybrid edges, but no warning will be raised if you attempt to set the 
        probability of a non-hybrid edge.

        Args:
            gamma (float): A probability (between 0 and 1).
        """
        self.gamma = gamma
    
    def get_gamma(self) -> float:
        """
        Gets the inheritance probability for this edge.

        Returns:
            float: A probability (between 0 and 1).
        """
        return self.gamma
    
    def to_names(self) -> tuple[str]:
        """
        Get the names of the source and destination nodes. 

        Returns:
            tuple[str]: 2-tuple, in the format (source name, destination name).
        """
        return (self.src.get_name(), self.dest.get_name())
    
    def duplicate(self, new_src : Node = None, new_dest : Node = None) -> Edge:
        """
        Craft an identical edge to this edge object, just in a new object.
        Useful in building subnetworks of a network that is in hand.

        Returns:
            Edge: An identical edge to this one, with respect to the data they 
                  hold.
        """
        if new_src is None or new_dest is None:
            new_edge = Edge(self.src, self.dest)
        else:
            new_edge = Edge(new_src, new_dest)
        new_edge.set_length(self.length)
        new_edge.set_gamma(self.gamma)
        return new_edge
          
class EdgeSet:
    """
    Data structure that serves the purpose of keeping track of edges that belong
    to a network. We call this set E.
    """
    
    def __init__(self) -> None:
        """
        Initialize the set of edges, E, for a network.
        """
        
        # Map (src, dest) tuples to a list of edges. this list will have 1 
        # element for most, but in the case of bubbles will contain 2. This 
        # exists purely to deal with bubbles 
        self.hash : dict[tuple[Node], list[Edge]] = defaultdict(list)
        
        # Edge set, E
        self.edges : set[Edge] = set()
    
    def add(self, edge : Edge) -> None:
        """
        Add an edge to E.

        Args:
            edge (Edge): A new edge to add to E.
        """
        if edge not in self.edges:
            self.hash[(edge.src, edge.dest)].append(edge)
            self.edges.add(edge)
            
    def remove(self, edge : Edge) -> None:
        """
        Remove an edge from E.

        Args:
            edge (Edge): An edge that is currently in E.
        """
        if edge in self.edges:
            self.hash[(edge.src, edge.dest)].remove(edge)
            self.edges.remove(edge)
            
    def get(self, source : Node, destination : Node, 
            gamma : float = None) -> Edge:
        """
        Given a source node, destination node, and an inheritance probability,
        get the edge in E that matches the data.

        Args:
            source (Node): Parent node.
            destination (Node): Child node.
            gamma (float, optional): Inheritance probability, for bubble 
                                     identifiability reasons. Defaults to None.

        Raises:
            EdgeError: If no edges in E satisfy the given data.

        Returns:
            Edge: The edge in E that matches the given data.
        """
        
        valid_edges : list[Edge] = self.hash[(source, destination)]
        
        if len(valid_edges) == 1:
            return valid_edges[0]
        elif len(valid_edges) == 2:
            if gamma is None:
                return valid_edges[0]
            else:
                if valid_edges[0].gamma == gamma:
                    return valid_edges[0]
                elif valid_edges[1].gamma == gamma:
                    return valid_edges[1]
        elif len(valid_edges)==0:
            raise EdgeError("No edges found with the given source and \
                              destination")
        else:
            raise EdgeError(">2 edges are detected with the given source and \
                              destination nodes. Networks are not allowed to \
                              have such topology.")
    
    def get_all(self, source : Node, destination : Node) -> list[Edge]:
        """
        Retrieves all edges in E with the given data.

        Args:
            source (Node): Parent node.
            destination (Node): Child node.

        Returns:
            list[Edge]: List of all edges that match the source and destination.
        """
        return self.hash[(source, destination)]
    
    def get_set(self) -> set[Edge]:
        """
        Get the set, E, for a network.

        Returns:
            set[Edge]: Edge set, E.
        """
        return self.edges
    

#########################
#### NETWORK CLASSES ####
#########################

class Network():
    """
    This class represents a directed acyclic graph containing nodes and edges.
    An edge is a list [a,b] where a and b are nodes in the graph, 
    and the direction of the edge is from a to b (a is b's parent). 
    [a,b] is NOT the same as [b,a], as this is a DIRECTED graph.

    Allowances:
    1) You may create cycles-- BUT we have provided a method to check if this 
       graph object is acyclic
    2) You may have multiple roots. Be mindful of whether this graph is 
       connected and what root you wish to operate on.
    3) You may end up with floater nodes/edges, ie this may be an unconnected 
       graph with multiple connected components. We will provide a method to 
       check for whether your graph object is one single connected component. 
       We have also provided methods to remove such artifacts.
       
       
    Formulation:
    Network = (E, V), where E is the set of all edges [a,b], where a is 
    b's parent, and a and b are elements of V, the set of all nodes. 
    
    """

    def __init__(self, edges : EdgeSet = None, 
                 nodes : NodeSet = None) -> None:
        """
        Initialize a Network object.
        You may initialize with any combination of edges/nodes,
        or provide none at all.

        Args:
            edges (EdgeSet, optional): A set of Edges. 
                                       Defaults to an empty EdgeSet.
            nodes (NodeSet, optional): A set of Nodes. 
                                       Defaults to an empty NodeSet.
        """
        
        # Outgroup is a node that is a leaf and whose parent is the root node.
        self.outgroup : Node = None
        
        # Blob storage for anything that you want to associate with 
        # this network. Just give it a string key!
        self.items : dict[str, object] = {}

        if edges is not None:
            self.edges : EdgeSet = edges
        else:
            self.edges : EdgeSet = EdgeSet()
        
        if nodes is not None:
            self.nodes : NodeSet = nodes
        else:
            self.nodes : NodeSet = NodeSet()
        
        # Initialize the unique id count
        self.UID = 0
        
        self.node_edge_consistency_check()
        
        #Free floater nodes/edges are allowed
        for edge in list(self.edges.get_set()):
            self.nodes.process(edge)
        
        self.leaves : list[Node] = [node for node in list(self.nodes.get_set())
                        if self.nodes.out_degree[node] == 0]
        self.roots : list[Node] = [node for node in list(self.nodes.get_set()) 
                        if self.nodes.in_degree[node] == 0]
    
    def node_edge_consistency_check(self) -> None:
        
        node_set : set[Node] = self.nodes.get_set()
        for edge in list(self.edges.get_set()):
            if edge.src not in node_set:
                self.nodes.add(edge.src) 
            if edge.dest not in node_set:
                self.nodes.add(edge.dest)
        
    def get_edge(self, src : Node, dest : Node, gamma : float = None) -> Edge:
        """
        Gets the edge in the graph with the given source, destination, and any 
        additional info that can help identify the edge. 
        
        Note, that in the event of bubbles, 2 edges will exist with the same 
        source and destination. If this is possible, please supply the 
        inheritance probability of the correct branch. If both edges are known 
        to be identical (gamma = 0.5), then one will be chosen at random.

        Args:
            src (Node): parent node
            dest (Node): child node
            gamma (float): inheritance probability. Optional. Defaults to None
                                    
        Returns:
            Edge: the edge from src to dest.
        """
        
        return self.edges.get(src, dest, gamma)
                        
    def add_nodes(self, nodes : Node | list[Node]) -> None:
        """
        If nodes is a list of nodes, then add each node point to the list
        If nodes is simply a node, then just add the one node to the nodes list.

        Args:
            nodes (Node | list[Node]): any amount of nodes, either a singleton,
                                       or a list
                                    
        """

        if type(nodes) == list:
            for node in nodes:
                self.nodes.add(node)     
        else:
            self.nodes.add(nodes)
        
    def add_edges(self, edges : Edge | list[Edge]) -> None:
        """
        If edges is a list of Edges, then add each Edge to the list of edges.
        
        If edges is a singleton Edge then just add to the edge array.
        
        Note: Each edge that you attempt to add must be between two nodes that
        exist in the network. Otherwise, an error will be thrown.
        
        Args:
            edges (Edge | list[Edge]): a single edge, or multiple.

        Raises:
            NetworkError: if input edge/edges are malformed in any way
        """
        
        # Determine whether the param is a list of edges, or a single edge. 
        if type(edges) == list:
            for edge in edges: 
                if self.nodes.ready(edge):              
                    self.edges.add(edge)
                    self.nodes.process(edge)  
                    self.reclassify_node(edge.src, True, True)
                    self.reclassify_node(edge.dest, False, True)
                else:
                    raise NetworkError("Tried to add an edge between two nodes,\
                                        at least one of which does not belong\
                                        to this network.")
        else:
            if self.nodes.ready(edges):
                self.edges.add(edges)
                self.nodes.process(edges)
                self.reclassify_node(edges.src, True, True)
                self.reclassify_node(edges.dest, False, True)  
            else:
                raise NetworkError("Tried to add an edge between two nodes,\
                                    at least one of which does not belong\
                                    to this network.") 
        
    def update_node_name(self, node : Node, name : str) -> None:
        """
        Rename a node and update the bookkeeping.

        Args:
            node (Node): a node in the graph
            name (str): the new name for the node.
        """
        if node.get_name() is not None:
            del self.nodes.node_names[node.get_name()]
        node.set_name(name)
        self.nodes.node_names[name] = node
        
    def remove_node(self, node : Node) -> None:
        """
        Removes node from the list of nodes.
        Also prunes all edges from the graph that are connected to the node.
        
        Has no effect if node is not in this network.
        
        Args:
            node (Node): a Node obj
        """
        
        if node in self.nodes.get_set():
            for edge in self.nodes.in_map[node]:
                self.remove_edge(edge)
            for edge in self.nodes.out_map[node]:
                self.remove_edge(edge)
            
            self.nodes.remove(node)
                     
    def remove_edge(self, edge : Edge | list[Node], 
                    gamma : float = None) -> None:
        """
        Args:
            edge (Edge | list[Node]): an edge to remove from the graph
            gamma (float): an inheritance probability from [0,1], if the edge is
                           provided as a list of nodes, and there is an 
                           identifiability issue that needs resolving (ie,
                           the edge that needs to be removed is a bubble
                           edge). Optional. Defaults to None.
            
            
        Removes edge from the list of edges. Does not delete nodes with no edges
        Has no effect if 'edge' is not in the graph.
        """
    
        if type(edge) == list:
            if len(edge) == 2:
                edge = self.get_edge(edge[0], edge[1], gamma) 
            else:
                raise NetworkError("Please provide a list of two nodes,\
                                 in the format [src, dest]")
                
        if edge in self.get_edges():
            # Remove the edge from the edge set
            self.edges.remove(edge)
        
            #Make the edge set aware of the edge removal
            self.nodes.process(edge, removal = True)
            
            # Reclassify the nodes, as they may be leaves/roots/etc now.
            self.reclassify_node(edge.src, True, False)
            self.reclassify_node(edge.dest, False, False)
        
    def reclassify_node(self, node : Node, is_par : bool,
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
        """
        if is_addition:
            if is_par:
                # If out degree now = 1, then the node was previously a leaf, 
                # and is not anymore
                if self.nodes.out_degree[node] == 1:
                    try:
                        self.leaves.remove(node)
                    except:
                        pass
                if self.nodes.in_degree[node] == 0:
                    if node not in self.roots:
                        self.roots.append(node)
            else:
                # If in_degree now = 1, then the node was previously a root,
                # and is not anymore
                if self.nodes.in_degree[node] == 1:
                    try:
                        self.roots.remove(node)
                    except:
                        pass
                if self.nodes.out_degree[node] == 0:
                    if node not in self.leaves:
                        self.leaves.append(node)
                        pars = self.get_parents(node)
                        if len(pars) == 1 and self.roots == 1:
                            if pars[0] in self.roots:
                                self.outgroup = node
        else:
            if is_par:
                # if out degree is now = 0, then the node is now a leaf
                if self.nodes.out_degree[node] == 0:
                    self.leaves.append(node)
                    pars = self.get_parents(node)
                    if len(pars) == 1 and self.roots == 1:
                        if pars[0] in self.roots:
                            self.outgroup = node
            else:
                # if in degree is now = 0, the node is now a root
                if self.nodes.in_degree[node] == 0:
                    self.roots.append(node)
                    
    def get_outgroup(self) -> Node:
        """
        Return the network's outgroup, returns None if there is none

        Returns:
            Node: the outgroup node
        """
        return self.outgroup
        
    def get_nodes(self) -> list[Node]:
        """
        Get all nodes in V.

        Returns:
            list[Node]: the set V, in list form.
        """
        return list(self.nodes.get_set())

    def get_edges(self) -> list[Edge]:
        """
        Get the set E (in list form).

        Returns:
            list[Edge]: The list of all edges in the graph
        """
        return list(self.edges.get_set())
        
    def get_item(self, key : str) -> object:
        """
        Access the blob storage with a key. CONSIDER REMOVAL OF THIS FUNCTION...

        Args:
            key (str): _description_

        Returns:
            object: _description_
        """
        return self.items[key]
    
    def put_item(self, key : str, item):
        if key not in self.items:
            self.items[key] = item

    def add_uid_node(self, node : Node = None) -> Node:
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
            new_node : Node = Node(name = "UID_" + str(self.UID))
            self.add_nodes(new_node)
            self.UID += 1
            return new_node
        else:
            if node not in self.nodes:
                self.add_nodes(node)
            self.update_node_name(node, "UID_" + str(self.UID))
            self.UID += 1
            return node
        
    def in_degree(self, node: Node) -> int:
        """
        Get the in-degree of a node

        Args:
            node (Node): A node in V

        Returns:
            int: the in degree count
        """
        if node in self.nodes.get_set():
            return self.nodes.in_degree[node]
        else:
            warnings.warn("Attempting to get the in-degree of a node that is \
                not in the graph-- returning 0")
            return 0

    def out_degree(self, node: Node) -> int:
        """
        Get the out-degree(number of edges where the given node is a parent)
        of a node in the graph.

        Args:
            node (Node): a node in V

        Returns:
            int: the out-degree count
        """
        if node in self.nodes.get_set():
            return self.nodes.out_degree[node]
        else:
            warnings.warn("Attempting to get the out-degree of a node that is\
                not in the graph-- returning 0")
            return 0

    def in_edges(self, node: Node) -> list[Edge]:
        """
        Get the in-edges of a node in V. The in-edges are the edges in E, where
        the given node is the child.

        Args:
            node (Node): a node in V

        Returns:
            list[Edge]: the list of in-edges
        """
        if node in self.nodes.get_set():
            return self.nodes.in_map[node]
        else:
            warnings.warn("Attempting to get the in-edges of a node that is\
                not in the graph-- returning an empty list")
            return []
            
    def out_edges(self, node: Node) -> list[Edge]:
        """
        Get the out-edges of a node in V. The out-edges are the edges in E,
        where the given node is the parent.

        Args:
            node (Node): a node in V

        Returns:
            list[Edge]: the list of out-edges
        """
        if node in self.nodes.get_set():
            return self.nodes.out_map[node]
        else:
            warnings.warn("Attempting to get the out-edges of a node that is\
                not in the graph-- returning an empty list")
            return []
    
    def root(self) -> list[Node]:
        """
        Return the root(s) of a network.
        
        In general, there may be multiple roots. 
        """
        #TODO: Why is this not just: "return self.roots"???
        return [root for root in self.roots 
                if self.nodes.out_degree[root] != 0]
        
    def get_leaves(self) -> list[Node]:
        """
        Returns the set X (a subset of V), the set of all leaves (nodes with
        out-degree 0). Only returns the leaves that are connected/reachable from
        the root.

        Returns:
            list[Node]: the connected elements of X, in list format.
        """
        #why not "return self.leaves?"
        return [leaf for leaf in self.leaves 
                if self.nodes.in_degree[leaf] != 0]

    def get_parents(self, node : Node) -> list[Node]:
        """
        Returns a list of the parents of a node. 
        There is no hard cap on the length of this array.

        node (Node): any node in V.
        """
        if node in self.get_nodes():
            return [edge.src for edge in self.nodes.in_map[node]]
        else:
            raise NetworkError("Attempted to calculate parents of a node that \
                is not in the graph.")
        
    def get_children(self, node: Node) -> list[Node]:
        """
        Returns a list of the children of a node.
        There is no hard cap on the length of this array.

        node (Node): any node in V.
        """
        if node in self.get_nodes():
            return [edge.dest for edge in self.nodes.out_map[node]]
        else:
            raise NetworkError("Attempted to calculate children of a node that \
                is not in the graph.")

    def has_node_named(self, name : str) -> Node:
        """
        Check whether the graph has a node with a certain name.
        Strings must be exactly equal (same white space, capitalization, etc.)

        Args:
            name (str): the name to search for

        Returns:
            Node: the node with the given name
        """
        try:
            return self.nodes.node_names[name]
        except:
            return None

    def print_graph(self) -> None:
        """
        Simply prints out information for each node in V.
        """
        for node in self.get_nodes():
            print(node.as_string())

    def pretty_print_edges(self) -> None:
        """
        Prints all the edges in E.
        """
        for edge in self.get_edges():
            print(f"<{edge.src.get_name()}, {edge.dest.get_name()}>")
            print(edge)
            print("------")
        
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
        """
        if options[0]:
            #Delete floater nodes
            floaters = [node for node in self.get_nodes() \
                        if self.in_degree(node) == 0 \
                        and self.out_degree(node) == 0]
            for floater in floaters:
                self.nodes.remove(floater)
    
        if options[1]:
            #Delete spurious root/root edge combo
            root = self.root()[0]
            if self.out_degree(root) == 1:
                spurious_edge = self.get_edge(root, self.get_children(root)[0])
                self.remove_edge(spurious_edge)
                self.remove_node(root)
        if options[2]:
            #Delete spurious "speciation events" marked by nodes with in/out = 1 
            root = self.root()[0]
        
            q = deque()
            q.appendleft(root)
            
            while len(q) != 0:
                cur = q.pop() #pop right for bfs

                for neighbor in self.get_children(cur):
                    current_node : Node = neighbor
                    previous_node : Node = cur
                    node_removed = False
                    
                    # There could be a chain of nodes with in/out degree = 1. 
                    # Resolve the whole chain before moving on 
                    while self.in_degree(current_node) == 1 \
                        and self.out_degree(current_node) == 1:
                            
                        self.remove_edge([previous_node, current_node])
                        
                        previous_node = current_node
                        temp = self.get_children(current_node)[0]
                        self.remove_node(current_node)
                        current_node = temp
                        node_removed = True
                    
                    # We need to connect cur to its new successor
                    if node_removed:
                        # self.remove_edge([previous_node, current_node])
                        self.add_edges(Edge(cur, current_node))    
        
                    
                    # Resume search from the end of the chain if one existed, 
                    # or this is neighbor if nothing was done
                    q.append(current_node)
            
           
    # def generate_branch_lengths(self) -> None:
    #     """
    #     Assumes that each node in the graph does not yet have a branch length associated with it,
    #     but has a defined "t" attribute.

    #     Raises:
    #         Exception: If a node is encountered that does not have a "t" value defined in its attribute dictionary
    #     """
    #     root = self.root()[0]
    #     root.add_length(0, None)
        
    #     # stack for dfs
    #     q = deque()
    #     q.append(root)
    #     visited = set()

    #     while len(q) != 0:
    #         cur = q.pop()

    #         for neighbor in self.get_children(cur):
    #             if neighbor not in visited:
    #                 t_par = cur.attribute_value("t")
    #                 t_nei = neighbor.attribute_value("t")
                    
    #                 #Handle case that a "t" value doesn't exist
    #                 if t_par is None or t_nei is None:
    #                     raise NetworkError("Assumption that t attribute exists for all nodes has been violated")
    #                 neighbor.add_length(t_par - t_nei, cur)
                
    #                 q.append(neighbor)
    #                 visited.add(neighbor)

    def mrca(self, set_of_nodes: set[Node] | set[str])-> Node:
        """
        Computes the Least Common Ancestor of a set of graph nodes

        Args:
            set_of_nodes (set[Node] | set[str]): A set of Nodes, or node names.

        Returns:
            Node: The node that is the LCA of the set.
        """
        format_set = set()
        for item in set_of_nodes:
            if type(item) is str:
                node_version = self.has_node_named(item)
                if node_version is None:
                    raise NetworkError("A node in 'set_of_nodes' is not \
                                      in the graph")
                else:
                    format_set.add(node_version)
            elif type(item) is Node:
                if item in self.get_nodes():
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
        leaf_2_parents = {} 

        for leaf in set_of_nodes:
            #Run bfs upward from each node 
            node_2_lvl : dict = {}
            
            # queue for bfs
            q = deque()
            q.append(leaf)
            visited = set()
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
        intersection = self.nodes.get_set()
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
        return min(additive_level, key=additive_level.get)
                      
    def leaf_descendants(self, node : Node) -> set[Node]:
        """
        Compute the set of all leaf nodes that are descendants of the parameter 
        node. Uses DFS to find paths to leaves.

        Args:
            node (Node): The node for which to compute leaf children

        Returns:
            set[Node]: The list of all leaves that descend from 'node'
        """
        if node not in self.get_nodes():
            raise NetworkError("Node not found in graph.")
        
        root = node

        # stack for dfs
        q = deque()
        q.appendleft(root)
        leaves = set()

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
        first_edge : Edge = random_object(self.get_edges(), rng)
        
        #Find another edge while excluding descendants of the first edge
        first_edge_subtree = self.leaf_descendants(first_edge.dest)
       
        #Accumulate pairs of edges that satisfy the requirement
        valid_edges = []
        for edge in self.get_edges():
            leaf_desc_edge : set[Node] = self.leaf_descendants(edge.dest)
            # If the intersection of leaf sets is null, then neither edge
            # can be found from the other.
            if len(leaf_desc_edge.intersection(first_edge_subtree)) == 0:
                valid_edges.append(edge)
            
        second_edge : Edge = random_object(valid_edges, rng)
        
        return [first_edge, second_edge]
    
    def subgenome_count(self, n : Node) -> int:
        """
        THINK ABOUT REFACTORING!
        
        Given a node in this graph, return the subgenome count.
         
        Args:
            n (Node): Any node in the graph. 
                      It is an error to input a node that is not in the graph.

        Returns:
            int: subgenome count
        """
        
        if n not in self.get_nodes():
            raise NetworkError("Input node is not in the graph")
        
        if self.root()[0] == n:
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
            edges (list[Edge]): The set of all edges in the subgraph of n.
        """
        if n not in self.get_nodes():
            raise NetworkError("Input node is not in the graph.")
        
        q = deque()
        q.appendleft(n)
        
        edges : list[Edge] = list()
        
        while len(q) != 0:
            cur = q.pop() #pop right for bfs

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
            edges (list[Edge]): The set of all edges in the subgraph of n.
        """
        if n not in self.get_nodes():
            raise NetworkError("Input node is not in the graph.")
        
        q = deque()
        q.appendleft(n)
        
        edges : list[Edge] = list()
        
        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            for edge in self.in_edges(cur):
                edges.append(edge)
                q.append(edge.src)
        
        return edges
    
    def edges_to_subgenome_count(self, downstream_node : Node = None, 
                                 delta : float = math.inf, 
                                 start_node : Node = None) -> dict[int, list[Edge]]:
        """
        Maps edges to their subgenome counts.
        
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

        Raises:
            NetworkError: If the graph has more than one root to start.

        Returns:
            dict[Edge, int]: a map from edges to subgenome counts
        """
    
        if start_node is None:
            start_nodes = self.root()
            if len(self.root()) != 1:
                raise NetworkError("Please specify a start node for this \
                                    network, there is more than one root \
                                    (or none)")
            start_node = start_nodes[0]
            
        q = deque()
        q.appendleft(start_node)
        
        edges_2_sub = {edge : 0 for edge in self.get_edges()}
        
        
        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            for neighbor in self.get_children(cur):
                
                edges_2_sub[self.edges.get(cur, neighbor)] += 1
                
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
            filter2 : dict = {}
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

    def leaf_descendants_all(self) -> dict[Node, set[Node]]:
        """
        Map each node in the graph to its set of leaf descendants
        Returns:
            dict[Node, set[Node]]: map from graph nodes to their 
                                   leaf descendants
        """
        desc_map : dict[Node, set[Node]] = {}
        
        #Mutates desc_map
        leaf_desc_help(self, self.root()[0], self.get_leaves(), desc_map)
        
        return desc_map
        
    def newick(self):
        return newick_help(self, self.root()[0], set()) + ";"
    
    def is_cyclic_util(self, v, visited, rec_stack):
        visited[v] = True
        rec_stack[v] = True

        for neighbor in self.get_children(v):
            if not visited[neighbor]:
                if self.is_cyclic_util(neighbor, visited, rec_stack):
                    return True
            elif rec_stack[neighbor]:
                return True

        rec_stack[v] = False
        return False

    def is_acyclic(self) -> bool:
        """
        Checks if each of this graph's connected components is acyclic

        Returns:
            bool: True if acyclic, False if cyclic. 
        """
        
        #Maintain structures for checking nodes that are visited or in the recursive stack
        visited = {node : False for node in self.get_nodes()}
        rec_stack = {node : False for node in self.get_nodes()}

        #Call recursive dfs on each root node / each connected component
        for node in self.root():
            if not visited[node]:
                if self.is_cyclic_util(node, visited, rec_stack):
                    return False

        return True
    
    def bfs_dfs(self, start_node : Node = None, dfs : bool = False, 
                is_connected : bool = False, accumulator : Callable = None, 
                accumulated = None) -> dict[Node, int]:
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
        q : deque = deque()
        visited : set[Node] = set()
        
        
        if start_node is not None:
            q.append(start_node)
            dist = {start_node : 0}
            visited.add(start_node)
        else:
            root : Node = self.root()[0]
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
            if len(set(self.get_nodes()).difference(visited)) != 0:
                print("GRAPH HAS MORE THAN 1 CONNECTED COMPONENT")
            else:
                print("GRAPH IS FULLY CONNECTED")
        
        return dist, accumulated
        
    def reset_outgroup(self, new_outgroup : Node) -> None:
        """
        Change the root of the network such that 'new_outgroup' 
        (a leaf) is now the outgroup of the network.
        
        Assumes that:
        
        1) The parent of new outgroup must not be a parent to a reticulation 
        2) The path to the root must not contain any reticulation nodes
        
        Args:
            new_outgroup (Node): A node that is a leaf of the current network.

        Raises:
            NetworkError: If the graph is unable to be transformed
        """
        if new_outgroup == self.get_outgroup():
            return
        
        pars = self.get_parents(new_outgroup)
        if len(pars) != 1:
            raise NetworkError("Could not set the outgroup to the given node. \
                              Violates assumption 2 \
                              (see function documentation).")
        
        # Check Assumption 2
        children : list[Node] = self.get_children(pars[0])
        
        for child in children:
            if child.is_reticulation():
                raise NetworkError("Could not set the outgroup to the given node.\
                                  Violates assumption 2")
        
        # Check Assumption 1
        path_to_root = self.rootpaths(pars[0])
        if len(path_to_root) != 1:
            raise NetworkError("Could not set the outgroup to the given node. \
                              Violates assumption 1")
        
        # Valid outgroup, now switch roots 
        #(by reversing all the edges on the path to the root)
        for edge in path_to_root[0]:
            self.remove_edge(edge)
            self.add_edges(Edge(edge.dest, edge.src)) #Reverse the edge
        
        #There may be excess nodes as a result of the edge flipping process
        self.clean() 
        
        #These better match
        print(self.outgroup)
        print(new_outgroup.get_name())
        
    def rootpaths(self, start : Node) -> list[list[Edge]]:
        """
        Get all paths (list of edges)

        Args:
            start (Node): Start the search from this node

        Returns:
            list[list[Edge]]: a list of all paths (lists of edges) to the root  
                              from 'start'
        """
        #A list of paths, each path is a list of edges.
        paths : list[list[Edge]] = [] 
        
        for par in self.get_parents(start):
            for path in self.rootpaths(par):
                paths.append(path.append(self.edges.get(par, start)))
        return paths
    
    def subtree_copy(self, retic_node : Node) -> Network:
        """
        Make a copy of a subnetwork of this DAG, rooted at 'retic_node', 
        with unique node names.
        
        Args:
            retic_node (Node): A node in this network that is a reticulation 
                               node

        Returns:
            Network: A subnetwork of the DAG being operated on
        """
    
        q = deque()
        q.appendleft(retic_node)
        net_copy = Network()
        
        new_node = Node(name = retic_node.get_name() + "_copy")
        net_copy.add_nodes(new_node)
        net_2_mul = {retic_node : new_node}
        

        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            for neighbor in self.get_children(cur):
                new_node = Node(name = neighbor.get_name() + "_copy")
                net_copy.add_nodes(new_node)
                net_2_mul[neighbor] = new_node
                net_copy.add_edges(Edge(net_2_mul[cur], new_node))
                
                # Resume search from the end of the chain if one 
                # existed, or this is neighbor if nothing was done
                q.append(neighbor)
        
        return net_copy 
    
    def duplicate(self) -> list:
        """
        Copy this network into a new network object, also with new node and 
        edge objects.

        Returns:
            Network: A carbon copy of this Network.
        """
        net_copy : Network = Network()
        
        old_new = {}
        
        for node in self.get_nodes():
            new = node.duplicate()
            old_new[node] = new
            net_copy.add_nodes(new)
        
        for edge in self.get_edges():
            new_src = old_new[edge.src]
            new_dest = old_new[edge.dest]
            new = edge.duplicate(new_src, new_dest)
            
            net_copy.add_edges(new)
        
        return net_copy, old_new
    
    def to_networkx(self) -> nx.Graph:
        nx_network = nx.MultiDiGraph()
        nx_network.add_nodes_from([node.get_name() for node in self.get_nodes()])
        nx_network.add_edges_from([edge.to_names() for edge in self.get_edges()])
        return nx_network
    
    
class DAGNetwork(Network):
    """
    This DAG is enforced to be acyclic and singularly rooted. 
    
    If at any time an edge is removed such that a node has no in-edges, that 
    subgraph starting with that node is all removed, stored as a unique network,
    and is placed in a data structure that holds DAGs.
    """
    
    def __init__(self, edges = EdgeSet(), nodes = NodeSet()) -> None:
        super().__init__(edges, nodes)   
        self.roots = []
        if not self.is_acyclic():
            raise NetworkError("This network contains a cycle. Please refactor\
                             or place into standard DAG object")
        if len(self.roots) > 1:
            raise NetworkError("This network has more than one root!\
                Please refactor or place into a standard DAG object.")
        
    def root(self) -> Node:
        if len(self.roots) >1:
            raise NetworkError("This network has more than one root!\
                Please refactor or place into a standard DAG object.")
        return self.roots[0]   
        
class BinaryNetwork(DAGNetwork):
    """
    This Network is enforced with the constraint that any node may have at most
    2 parents and 1 child or 1 parent and 2 children, in addition to the 
    requirement that this graph be acyclic and singularly rooted.
    """    
    
    def __init__(self, edges = EdgeSet(), nodes = NodeSet()) -> None:
        super().__init__(edges, nodes)
        
class MUL(Network):
    """
    A subclass of a Network, that is a binary tree that results from the 
    transformation of a standard network into a Multilabeled Species Tree.
    """
    def __init__(self, gene_map : dict, rng : np.random.Generator) -> None:
        
        self.net : Network = None
        self.mul : Network = None
        self.gene_map : dict = gene_map
        self.rng : np.random.Generator = rng
                        
    def to_mul(self, net : Network) -> Network:
        """
        Creates a (MU)lti-(L)abeled Species Tree from a network

        Args:
            net (Network): A Network

        Raises:
            NetworkError: If the network is malformed with regards to ploidy
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
        network_2_mul : dict[Node, Node] = {node : node.duplicate() for node in net.get_nodes()}
        
        # Add all nodes and edges from net into the mul tree
        
        #NODES
        mul_tree.add_nodes(list(network_2_mul.values()))
        
        #EDGES
        for edge in net.get_edges():
            new_edge = edge.duplicate(network_2_mul[edge.src],
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
                subtree = mul_tree.subtree_copy(cur)
                
                retic_pars = mul_tree.get_parents(cur)
                a = retic_pars[0]
                b = retic_pars[1]
            
                mul_tree.remove_edge([b, cur])
                mul_tree.add_nodes(subtree.get_nodes())
                mul_tree.add_edges(subtree.get_edges())
                mul_tree.add_edges(Edge(b, subtree.root()[0]))
                processed.add(subtree.root()[0])
        
            processed.add(cur)
            
            for par in original_pars:
                cop = set(mul_tree.get_children(par))
                if cop.issubset(processed):
                    traversal_queue.append(par)
        
        
        
        #Get rid of excess connection nodes
        
        mul_tree.clean([False, False, True])
        
        
        
        #Rename tips based on gene mapping
        for leaf in mul_tree.get_leaves():
            new_name = copy_gene_map[leaf.get_name().split("_")[0]].pop()
            mul_tree.update_node_name(leaf, new_name)

        self.mul = mul_tree 
     
        return mul_tree  




def test():
    
    net = Network()
    
    n1 = Node("n1")
    n2 = Node("n2")
    e1 = Edge(n1, n2)
    e1.set_gamma(.4)
    e2 = Edge(n1, n2)
    e2.set_gamma(.6)
    
    net.add_nodes([n1, n2])
    net.add_edges([e1, e2])
    

    net.pretty_print_edges()
    
