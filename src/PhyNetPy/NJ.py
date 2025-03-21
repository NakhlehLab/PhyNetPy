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
File with classes and functions that implement the Neighbor Joining algorithm
for trees.

Source : https://en.wikipedia.org/wiki/Neighbor_joining

Author: Mark Kessler
Date Last Edited: 3/11/25
Ready for Release:
    Docs    - [ X ]
    Design  - [ X ]
    Testing - [ X ]
"""

from typing import Union
from Network import *
from collections import deque
from NetworkParser import *
import numpy as np
from MSA import *


class NJException(Exception):
    """
    This exception is raised when there is an error either in parsing data
    into the matrix object, or if there is an error during any sort of 
    operation
    """

    def __init__(self, message : str = "NJ Error") -> None:
        """
        Initialize the error with a message

        Args:
            message (str, optional): The error message. Defaults to "NJ Error".
        Returns:
            N/A
        """
        self.message = message
        super().__init__(self.message)

def _find_distance_seq(aln : MSA, 
                       leaves : list[Node]) -> dict[tuple[Node], float]:
    """
    Compute the distance between two leaves in a network based on the sequence
    similarity.

    Raises:
        NJException -- if the Network and MSA are not associated or linked 
                       properly
    Args:
        aln (MSA): A sequence alignment for a network.
        leaves (list[Node]): The leaf nodes of a network that is associated with
                             the given alignment

    Returns:
        dict[tuple[Node], float]: "hamming"-esque distances for each pair of 
                                  leaves
    """
    D = dict()
    
    reqs : list[DataSequence] = aln.get_records()
    
    # Compute the pairwise distances between DataSequence objects
    d : dict[tuple[DataSequence], float] = aln.distance_matrix()
    
    # Map nodes to the index in the reqs index for which the node's seqrecord 
    # is located
    index_to_leaf: dict[int, Node] = dict()
    for leaf in leaves:
        if leaf.get_seq() is None:
            raise NJException(f"Leaf {leaf.get_seq()} does not have a DataSequence \
                               associated with it")
        try:
            index = reqs.index(leaf.get_seq())
            index_to_leaf[index] = leaf
        except ValueError:
            raise NJException(f"Sequence associated with the leaf \
                              {leaf.label} is not in the given MSA!")
     
    # Translate aln distance matrix. Change DataSequence tuple to the equivalent
    # Node tuple.
    for seqtup, val in d.items():
        n1 : Node = index_to_leaf[reqs.index(seqtup[0])]
        n2 : Node = index_to_leaf[reqs.index(seqtup[1])]
        new_tup = (n1, n2)
        D[new_tup] = val
    
    return D
        
def _find_distance(N : Network, node_a : Node, node_b : Node) -> float:
    """
    Compute the distance between two leaf nodes, a and b, in a simulated network
    based on their speciation times.
    
    Args:
        N (Network): A directed, rooted, acyclic network with branch lengths.
        node_a (Node): A node in N
        node_b (Node): A node in N

    Returns:
        float: The time "distance" between nodes A and D
    """
    parents_a, parents_b = {}, {}
    explore_a, explore_b = deque([node_a]), deque([node_b])
    while len(set(parents_a.keys()) & set(parents_b.keys())) == 0:
        n = len(explore_a)
        i = 0
        while i < n:
            c = explore_a.popleft()
            for v in N.get_parents(c):
                parents_a[v] = node_a.get_time() - v.get_time()
                explore_a.append(v)
            i += 1

        n = len(explore_b)
        i = 0
        while i < n:
            c = explore_b.popleft()
            for v in N.get_parents(c):
                parents_b[v] = node_b.get_time() - v.get_time()
                explore_b.append(v)
            i += 1
    
    vs = set(parents_a.keys()) & set(parents_b.keys())
    dist = np.inf
    for v in vs:
        if dist > parents_a[v] + parents_b[v]:
            dist = parents_a[v] + parents_b[v]

    return dist

def _create_matrix(N : Network, aln : MSA = None) -> dict[tuple[Node], float]:
    """
    Create a pairwise distance mapping from all pairs of leaves in N to their
    distances.

    Args:
        N (Network): A network.
        aln (MSA, optional): An MSA. Defaults to None.

    Returns:
        dict[tuple[Node], float]: A mapping from all pairs of leaves to their
                                  distances.
    """
    leaves = N.get_leaves()
    D = dict()
    
    if aln is None:
        for i in range(len(leaves)):
            for j in range(i+1, len(leaves)):
                u = leaves[i]
                v = leaves[j]
                d = _find_distance(N, u, v)
                D[(i,j)] = d
                D[(j,i)] = d 
    else:
        D = _find_distance_seq(aln, leaves)

    return D

def _compute_q(d : dict[tuple[Node], float], 
               nodes : list[Node]) -> dict[tuple[Node], float]:
    """
    Given the distance matrix/mapping, generate the Q matrix.

    Args:
        d (dict[tuple[Node], float]): Pairwise distance matrix/map
        nodes (list[Node]): Nodes that remain and need to be connected.

    Returns:
        dict[tuple[Node], float]: Q
    """
    Q = dict()
    
    for i in nodes:
        for j in nodes:
            if i != j: #only for different nodes
                Q[(i, j)] = _q_entry(i, j, d, nodes)
    
    return Q
                    
def _q_entry(i : Node, 
             j : Node, 
             d : dict[tuple[Node], float], 
             nodes : list[Node]) -> float:
    """
    Compute the entry into the Q matrix given the distance mapping,
    the number of nodes to connect, and the 2 nodes in question.

    Args:
        i (Node): A Node that needs to be connected
        j (Node): A Node that needs to be connected
        d (dict[tuple[Node], float]): Pairwise distance mapping
        nodes (list[Node]): The list of nodes that remain

    Returns:
        float: Q matrix entry for (i, j)
    """
    num_nodes = len(nodes)
    
    #EQ 1 
    term1 = (num_nodes - 2) * d[(i, j)]
    term2 = sum([d[(i, k)] for k in nodes if k != i])
    term3 = sum([d[(j, k)] for k in nodes if k != j])
    
    return term1 - term2 - term3

def _new_dist(i : Node,
              j : Node, 
              d : dict[tuple[Node], float],
              nodes : list[Node]) -> float:
    """
    Delta function. Computes the edge lengths from i and j to the newly created
    node that is connecting to them.

    Uses EQ 2 from the source material.
    
    Args:
        i (Node): One node that connects to the new node
        j (Node): The other node that connects to the new node
        d (dict[tuple[Node], float]): pairwise distance map
        nodes (list[Node]): List of nodes left to be connected

    Returns:
        float: The edge distance from node i to node j
    """
    num_nodes = len(nodes)

    #EQ 2
    term1 = d[(i, j)] / 2
    sum1 = sum([d[(i, k)] for k in nodes if i != k])
    sum2 = sum([d[(j, k)] for k in nodes if j != k])
    term2 =  (sum1 - sum2) / (2 * num_nodes - 4)
    
    return term1 + term2

def _minmaxkey(mapping : dict[object, Union[int, float]],
               mini : bool = True) -> object:
    """
    Return the object in a mapping with the minimum or maximum value associated
    with it.

    Args:
        mapping (dict[object, int  |  float]): A mapping from objects to 
                                               numerical values
        mini (bool, optional): If True, return the object with the minimum
                               value. If False, return the object with the
                               maximum value. Defaults to True.
    Returns:
        object: The object with the minimimum or maximum value.
    """

    cur = math.inf
    cur_key = None
    if not mini:
        cur = cur * -1
        
    for key, value in mapping.items():
        if mini:
            if value < cur:
                cur = value
                cur_key = key
        else:
            if value > cur:
                cur = value
                cur_key = key
    
    return cur_key

def _nj_help(leaves : list[Node], d : dict[tuple[Node], float]) -> Graph:
    """
    Run the Neighbor Joining Algorithm on a set of leaves and a mapping from 
    these leaves to their pairwise distances.

    Args:
        leaves (list[Node]): List of leaves of a network.
        d (dict[tuple[Node], float]): Pairwise distances from leaf to leaf
    Returns:
        Graph: The inferred/joined network
    """
    nj_net = Graph()
    cur_nodes = leaves
    nj_net.add_nodes(cur_nodes)
    cur_d = d
    
    # Run the algorithm until 3 taxa remain. There is only one possible topology
    # with 3 taxa for unrooted and undirected networks.
    
    while len(cur_nodes) != 3:
        #Compute Q
        Q = _compute_q(cur_d, cur_nodes)
        
        #get i,j of nodes that have minimum distance 
        (n_i, n_j) = _minmaxkey(Q)
        
        #compute distance to new connecting node
        d_ik = _new_dist(n_i, n_j, cur_d, cur_nodes)
        d_jk = cur_d[(n_i, n_j)] - d_ik

        #Make network connections
        new_node = nj_net.add_uid_node()
        e1 = Edge(new_node, n_i)
        e1.set_length(d_ik)
        e2 = Edge(new_node, n_j)
        e2.set_length(d_jk)
        nj_net.add_edges([e1, e2])
        
        cur_nodes.remove(n_i)
        cur_nodes.remove(n_j)
        cur_nodes.append(new_node)
        
        #Recalculate the dist mapping
        
        old_nodes = [node for node in cur_nodes if node != new_node]
        
        new_d = {key : value for (key, value) in cur_d.items()
                 if key[0] in old_nodes and key[1] in old_nodes}
        
        for node in old_nodes:
            val = .5 * (cur_d[(n_i, node)] + 
                        cur_d[(n_j, node)] - 
                        cur_d[(n_i, n_j)])
            new_d[(new_node, node)] = val
            new_d[(node, new_node)] = val
            
            if val < 0:
                warnings.warn("Computed a negative distance between nodes \
                               during the computation of the pairwise distance \
                               matrix.")
        
        cur_d = new_d
        
    # Connect final 3 nodes, based on the naming convention 
    # in the source material
    w : Node = nj_net.add_uid_node()
    v : Node = cur_nodes[0]
    d : Node = cur_nodes[1]
    e : Node = cur_nodes[2]
    
    edge_v = Edge(v, w)
    edge_d = Edge(d, w)
    edge_e = Edge(e, w)
    
    #calculate edge lengths
    v_w = _new_dist(v, d, cur_d, cur_nodes)
    d_w = cur_d[(v, d)] - v_w 
    e_w = cur_d[(v, e)] - v_w
    
    edge_v.set_length(v_w) 
    edge_d.set_length(d_w)
    edge_e.set_length(e_w)
    
    #Add all edges
    nj_net.add_edges([edge_v, edge_d, edge_e])
    
    return nj_net

def NJ(net : Network, 
       aln : MSA = None, 
       d : dict[tuple[Node], float] = None) -> Graph:
    """
    Perform Neighbor Joining.

    A method for calculating pairwise leaf distances will be selected based on
    the provided arguments.
    
    Args:
        net (Network): A network.
        aln (MSA, optional): If there is an MSA associated with the leaves of 
                             the given network, and it is desired to calculate 
                             distances via the sequences, provide this. 
                             Defaults to None.
        d (dict[tuple[Node], float], optional): If there is no alignment 
                                                associated with the network, or 
                                                if distances have been 
                                                calculated via a customized 
                                                algorithm, provide this mapping.
                                                Defaults to None.

    Returns:
        Graph: The inferred unrooted and undirected NJ graph.
    """
    
    netleaves_to_leaves = {leaf : leaf.copy() for leaf in net.get_leaves()}
    
    if aln is None and d is not None:
        d_network = d
    elif aln is not None:
        d_network = _create_matrix(net, aln = aln)
    else:
        d_network = _create_matrix(net) 
        
    d = {(netleaves_to_leaves[key[0]], netleaves_to_leaves[key[1]]) : dist 
         for (key, dist) in d_network.items()}
    
    new_leaves = list(netleaves_to_leaves.values())
    
    return _nj_help(new_leaves, d)
