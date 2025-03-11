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

Docs   - [x]
Tests  - [ ]
Design - [ ]
"""
from Network import Network, Edge, Node
from collections import deque
from NetworkParser import *

def _retic_edge_choice(retic_map : dict[Node, list[Edge]], 
                      funcs : list[dict[Node, Edge]]) -> list[dict[Node, Edge]]:
    """
    Recursive helper function that generates all combinations of hybrid edges
    to remove from a network to create an underlying tree structure.

    Args:
        retic_map (dict[Node, list[Edge]]): dictionary of reticulation nodes to 
                                            their hybrid in-edges
        funcs (list[dict[Node, Edge]]): list of dictionaries that map reticulation
                                        nodes to their chosen hybrid in-edges   
    Returns:
        list[dict[Node, Edge]]: list of dictionaries that map reticulation nodes
                                to their chosen hybrid in-edges
    """
    if retic_map != {}:
        node = list(retic_map.keys())[0]
        hybrid_edges = retic_map[node]
       
        for index in range(len(funcs) // 2):
            funcs[index][node] = hybrid_edges[0]
        for index in range(len(funcs) // 2, len(funcs)):
            funcs[index][node] = hybrid_edges[1]
         
        del retic_map[node]
        
        _retic_edge_choice(retic_map, funcs)
        
    return funcs

def get_all_clusters(net : Network,
                     node : Node | None = None, 
                     include_trivial : bool = False)\
                     -> set[tuple[Node, ...]] | set[str]:
    """
    Compile a list of clusters that make up this graph.
    Ie: for a graph ((A, B)C, D); , set of all clusters is {(A,B), (A,B,C)}.
    
    Can optionally allow the trivial leaf clusters in the set as desired.

    Args:
        net (Network): the network to operate on
        node (Node): For any user call, this should be the root. 
                     For internal calls, it is the starting point for search.
        include_trivial (bool): If set to True, includes clusters of size 1. 
                                Defaults to False.

    Returns:
        set: A set of all clusters in this graph. Each cluster is represented 
             as a set of either names or nodes.
    """
    if node is None:
        node = net.root()
    
    cluster_set : set[Node] = set()
    graph_leaves = net.get_leaves()
    children = net.get_children(node)
    
    # Each leaf_descendant set of a child is a cluster, 
    # so long as it is not trivial
    for child in children:
        if child not in graph_leaves:
            #Get potential cluster
            leaf_descendant_set = net.leaf_descendants(child)
            
            #Check for size 
            if len(leaf_descendant_set) > 1: 
                cluster_set.add(tuple(leaf_descendant_set))
            
            #Recurse over the next subtree
            cluster_set = cluster_set.union(get_all_clusters(net, child))
    
    if include_trivial:
        for leaf in graph_leaves:
            cluster_set.add(tuple([leaf]))
        
    return cluster_set

def combine(left : Network, right : Network) -> Network:
    """
    Combine two networks into a single
    network object by making the roots of each network
    the children of a new root node.

    Args:
        left (Network): Left subnetwork
        right (Network): Right subnetwork

    Returns:
        Network: The resulting network object, containing copies of the nodes
                 and edges of the original networks.
    """
    merger = Network()
    
    left_copy, oldnew_left = left.copy()
    right_copy, oldnew_right = right.copy()
    
    # Create a new root node
    new_root = Node()
    merger.add_nodes(new_root)
    
    # Add left and right roots as children of the new root
    left_root = oldnew_left[left.root()]
    right_root = oldnew_right[right.root()]
    
    merger.add_edges(Edge(new_root, left_root))
    merger.add_edges(Edge(new_root, right_root))
    
    # Add all nodes and edges from left_copy and right_copy to merger
    for node in left_copy.V():
        merger.add_nodes(node)
    for edge in left_copy.E():
        merger.add_edges(edge)
    
    for node in right_copy.V():
        merger.add_nodes(node)
    for edge in right_copy.E():
        merger.add_edges(edge)
    
    return merger

def subnet_given_leaves(net : Network, leaf_set : list[Node]) -> Network:
    """
    Compute the minimally sized subnetwork of a network such that the leaf set 
    of the subnetwork is a subset of the original network's leaf set.

    Args:
        net (Network): A network
        leaf_set (list[Node]): A set of leaf nodes of the given network

    Returns:
        Network: A new Network object with node and edge copies of the original.
    """
    subnet : Network = Network()
    
    sub_root = net.mrca(set(leaf_set))
    new_sub_root = sub_root.copy()
    subnet.add_nodes(new_sub_root)
    old_new_map = {sub_root : new_sub_root}
    q = deque(sub_root)
    
    while len(q) != 0:
        cur = q.popleft()
    
        for child in net.get_children(cur):
            new_child = child.copy()
            old_new_map[child] = new_child
            subnet.add_nodes(new_child)
            
            #Copy edge info
            old_edge = net.get_edge(cur, child)
        
            #Add equivalent edge
            new_edge = Edge(old_new_map[cur], new_child)
            new_edge.set_gamma(old_edge.get_gamma())
            new_edge.set_length(old_edge.get_length())

            subnet.add_edges(new_edge)
            
            #Add child to queue
            q.appendleft(child)
    
    return subnet
    
def get_all_subtrees(net : Network) -> list[Network]:
    """
    Generate all possible trees that can be derived from the given network by
    removing hybrid edges and creating copies with subtrees that start at each 
    non-reticulation node.

    Args:
        net (Network): A network object
    Returns:
        list[Network]: A list of network objects, each representing a tree that
                       is derived from the original network.
    """
    retics = [node for node in net.V() if node.is_reticulation()]
    retic2edges = {node : net.in_edges(node) for node in retics}
    retic_maps : list[dict[Node, Edge]] = _retic_edge_choice(retic2edges, [{} for _ in range(2 ** len(retics))])
    
    trees = []
    
    for func in retic_maps:
        preop, old_new = net.copy()
        
        for edge in func.values():
            preop.remove_edge(old_new[edge.src], old_new[edge.dest])
        
        preop.clean()
        trees.append(preop)
    
    return trees

def dominant_tree(net : Network) -> Network:
    """
    Generate the dominant tree from a given network by retaining only the 
    reticulation edges with the highest inheritance probability and removing 
    all other reticulation edges.

    Args:
        net (Network): A network object
    Returns:
        Network: A new network object representing the dominant tree derived 
                 from the original network.
    """
    dom : Network = Network()
    
    edges_2_remove = []
    old_new_node_map = {}
    
    #Only include reticulation edges that are the maximum inheritance prob
    for node in [retic for retic in net.V() if retic.is_reticulation()]:
        retic_edges = [e for e in net.in_edges(node)]
        sorted_by_gamma = retic_edges.sort(key=lambda x: x.gamma, reverse=False)
        #remove all but maximum inheritance probability
        edges_2_remove.extend(sorted_by_gamma[:-1])
    
    #Add all nodes from original network
    for node in net.V():
        new_node = node.copy()
        dom.add_nodes(new_node)
        old_new_node_map[node] = new_node
    
    #Add only dominant reticulation edges and all other normal edges
    for edge in net.E():
        if edge not in edges_2_remove:
            new_src = old_new_node_map[edge.src]
            new_dest = old_new_node_map[edge.dest]
            dom.add_edges(edge.copy(new_src, new_dest))
      
    #Clean artifacts created by removing some of the retic edges      
    dom.clean()
    
    return dom



