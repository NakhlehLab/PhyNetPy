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

from Network import NetworkError, Node, Network, Edge
import random
from GeneTrees import *

"""
For now, the four moves I have chosen to implement for network editing are as
follows:

1) Add reticulation -- Needed in order to increase network level. With proper 
                       implementation, this function is able to introduce genome
                       duplications (bubbles)

2) Remove reticulation -- Needed in order to decrease network level.

3) NNI -- Needed in order to change topologies within a given network level.
 
4) Node height change -- Needed in order to change the height of a node within
                         the network. This is useful for changing the time of
                         speciation events.
"""

#########################
#### Network Changes ####
#########################

def add_hybrid(net : Network,
               source : Edge = None, 
               destination : Edge = None, 
               t_src : float = None, 
               t_dest : float = None) -> None:
    """
    Adds a hybrid edge from src to dest. If no source or destination edge is 
    provided, edges will be chosen at random. Edges are connected at the midway
    point, unless times are given.
    
    src:                dest:
    a                   x
    |                   |
    |                   |
    |                   |
    v                   v
    n1- - - - - - - - ->n2
    |                   |
    |                   |
    v                   v
    b                   y
    
    Args:
        net (Network): A network.
        source (Edge, optional): A source edge, the origin of the hybrid edge.
                                 Defaults to None.
        destination (Edge, optional): A destination edge. Defaults to None.
        t_src (float, optional): From the diagram, the speciation time at n1.
                                 Defaults to None.
        t_dest (float, optional): From the diagram, the speciation time at n2. 
                                  Defaults to None.
    Returns:
        N/A
    """
    src : Edge
    dest : Edge
    a : Node
    b : Node
    x : Node
    y : Node
    n1 : Node
    n2 : Node
    
    if source is None or destination is None:
        src : Edge = random.choice(net.E())
        a = src.src
        b = src.dest
        valid_dests = [edge for edge in net.E()
                       if edge not in net.edges_downstream_of_node(b)]
        dest = random.choice(valid_dests)
        x = dest.src
        y = dest.dest
    else:
        src = source
        dest = destination

        a = src.src
        b = src.dest
        x = dest.src
        y = dest.dest

        if src in net.edges_downstream_of_node(y):
            raise Exception("Destination for hybrid edge is upstream of \
                            selected Source!")
    
    n1 : Node = net.add_uid_node()
    n2 : Node = net.add_uid_node() #This is the reticulation node
    n2.set_is_reticulation(True)
    
    #Rewire edges
    net.remove_edge(src)
    net.remove_edge(dest)

    # Set times
    if t_src is not None and a.get_time() < t_src < b.get_time():
        n1.set_time(t_src)
    else:
        n1.set_time((a.get_time() + b.get_time()) / 2.0)

    if t_dest is not None and x.get_time() < t_src < y.get_time():
        n2.set_time(t_dest)
    else:
        n2.set_time((x.get_time() + y.get_time()) / 2.0)  
        
    # Add back new edges 
    edges : list[Edge] = [Edge(a, n1), 
                          Edge(n1, b), 
                          Edge(x, n2), #x to n2 is a hybrid edge
                          Edge(n2, y), 
                          Edge(n1, n2)] #n1 to n2 is a hybrid edge
    
    net.add_edges(edges)
        
def remove_hybrid(net : Network, 
                  hybrid_edge : Edge) -> None:
    """
    Removes a hybrid edge from a network.

    src:                dest:
    a                   x
    |                   |
    |                   |
    |                   |
    v                   v
    n1- - - - - - - - ->n2
    |                   |
    |                   |
    v                   v
    b                   y
    
    Raises:
        Exception: If the given edge destination is not a reticulation, or if 
                   the given edge source is a reticulation node (In which case
                   the resulting network will be malformed).
    Args:
        net (Network): A network.
        hybrid_edge (Edge): An edge in E, whose destination node is a 
                            reticulation node.
    Returns:
        N/A
    """
    
    #Assumption 1
    if not hybrid_edge.dest.is_reticulation():
        raise Exception("Given edge parameter is not a hybrid edge!")

    #Assumption 2
    if hybrid_edge.src.is_reticulation():
        raise Exception("The source of the given edge is a reticulation!")
    
    
    n1 : Node = hybrid_edge.src
    n2 : Node = hybrid_edge.dest
    
    #May assume the following operations work given that only one parent exists
    #and there must be a child of n1 that is not n2.
    a : Node = net.get_parents(n1)[0] 
    b : Node = [node for node in net.get_children(n1) if node != n2][0]
    x : Node = [node for node in net.get_parents(n2) if node != n1][0]
    y : Node = net.get_children(n2)[0]


    #Remove all edges by removing the nodes.
    net.remove_nodes(n2)
    net.remove_nodes(n1)

    # Add back new edges 
    net.add_edges([Edge(a, b),Edge(x, y)])       

def nni(net: Network) -> None:
    """
    Perform a nearest neighbor interchange (NNI) on the network.
    
    Args:
        net (Network): A network.
    Returns:
        N/A
    """
    # Select an internal edge at random
    internal_edges = [edge for edge in net.E() if not edge.dest in net.get_leaves()]
    if not internal_edges:
        raise Exception("No internal edges available for NNI.")
    
    edge = random.choice(internal_edges)
    a, b = edge.src, edge.dest
    
    # Get the neighbors of the nodes connected by the edge
    a_neighbors = [node for node in net.get_children(a) if node != b]
    b_neighbors = [node for node in net.get_children(b) if node != a]
    
    if not a_neighbors or not b_neighbors:
        raise Exception("Not enough neighbors for NNI.")
    
    # Select one neighbor from each set
    c = random.choice(a_neighbors)
    d = random.choice(b_neighbors)
    
    # Remove the original edges
    net.remove_edge(edge)
    net.remove_edge(Edge(a, c))
    net.remove_edge(Edge(b, d))
    
    # Add the new edges
    net.add_edge(Edge(a, d))
    net.add_edge(Edge(b, c))
    net.add_edge(Edge(a, b))

def node_height_change(n : Node, 
                       net : Network, 
                       height : float,
                       extend : bool = False) -> None:
    """
    Alter the node height of a node in a network, without altering the 
    node heights of any surrounding nodes (parents, children).
    
    A node height can be moved up or down within the range
    min child height < new h < max parent height (refer to assumptions.txt). 
    It is an error to attempt to move a node outside of these bounds -- the 
    operation will abort and no changes to the network will be made. 
    
    If extend is set (True), then the subtree branch lengths will be retained 
    instead of altered. The new height of 'n', then, must not be closer to the 
    root than the parent of 'n' whose height is furthest from the root. 
    Attempting this is an error -- the operation will abort and no changes to 
    the network will be made.

    Args:
        n (Node): A node whose height needs to be changed.
        net (Network): The network that contains node n.
        height (float): The new height of node n.
        extend (bool, optional): Flag to retain subtree branch lengths. Defaults to False.
    Returns:
        N/A
    """
    parents = net.get_parents(n)
    children = net.get_children(n)

    if not parents or not children:
        raise NetworkError("Node must have both parents and children.")

    max_parent_height = max(parent.get_time() for parent in parents)
    min_child_height = min(child.get_time() for child in children)

    if not (min_child_height < height < max_parent_height):
        raise NetworkError("New height is out of bounds.")

    if extend:
        if height < max_parent_height:
            raise NetworkError("New height is too close to the root when extending.")
        for child in children:
            child.set_time(child.get_time() + (height - n.get_time()))

    n.set_time(height)

