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

import io
import re
import traceback

import networkx as nx
from Bio import Phylo
from Network import Network, Node, NetworkError
from NetworkParser import NetworkParser, NetworkParserError
from collections import deque
import dendropy

def generate_branch_lengths(network : Network) -> None:
    """
    Assumes that each node in the graph does not yet have a branch length 
    associated with it, but has a defined "t" attribute.
    
    Calculates and assigns a value to a nodes "length" field based on the t vals

    Raises:
        Exception: If a node is encountered that does not have a "t" 
                   value defined in its attribute dictionary
    """
    root = network.root()
    root.add_length(0, None)
    
    # stack for dfs
    q = deque()
    q.append(root)
    visited = set()

    while len(q) != 0:
        cur : Node = q.pop()

        for neighbor in network.get_children(cur):
            if neighbor not in visited:
                t_par : float = cur.attribute_value("t")
                t_nei : float = neighbor.attribute_value("t")
                
                #Handle case that a "t" value doesn't exist
                if t_par is None or t_nei is None:
                    raise NetworkError("Assumption that t attribute exists\
                                      for all nodes has been violated")
                neighbor.add_length(t_par - t_nei, cur)
            
                q.append(neighbor)
                visited.add(neighbor)
    
    

def add_parallel_edge(network : Network, parent: Node, child: Node) -> None:
    """
    Adds a parallel edge between parent and child in network
                   parent
                     |
                    nd1
                     | \
                     |  |
                     | /
                    nd2
                     |
                   child
    """
    #Double check to make sure the passed parameters are actually in the network
    if [parent, child] not in network.edges:
        raise NetworkError(
            f"Edge {parent.label}->{child.label} does not exist \
                in network {network.print_adjacency()}")
    
    #Create new nodes that are required to make the parallel edge
    
    inserted_node1 = network.add_uid_node()
    inserted_node2 = network.add_uid_node()
    
    network.remove_edge([parent, child])
    network.add_edges([
        [parent, inserted_node2],
        [inserted_node2, inserted_node1],
        [inserted_node2, inserted_node1],
        [inserted_node1, child]
    ])




def bfs(network: Network):
    q = [network.root()]
    visited = set()
    while q:
        cur = q.pop()
        yield cur
        visited.add(cur)

        for child in network.get_children(cur):
            if child not in visited and child not in q:
                q.append(child)


def get_leaf_name_set(network: Network):
    return set([node.label for node in network.get_leaves()])

def convert_to_networkx(network: Network):
    G = nx.DiGraph()
    edges = []
    for edge in network.E():
        edges.append((edge[0].label, edge[1].label))
    G.add_edges_from(edges)
    return G

def plot_network(network: Network):
    import matplotlib.pyplot as plt
    G = convert_to_networkx(network)
    indegrees = G.in_degree()
    edge_colors = ['red' if indegrees[edge[1]] > 1 else 'black' for edge in G.edges()]
    nx.draw_networkx(G, edge_color=edge_colors, with_labels=True)
    plt.savefig("network.png")


def remove_binary_nodes(net: Network):
    """Modified based on DAG.prune_excess_nodes()"""

    def prune(net: Network) -> bool:
        root = net.root()
        q = deque([root])
        net_updated = False

        while q:
            cur = q.pop()  # pop right for bfs

            for neighbor in net.get_children(cur):
                current_node: Node = neighbor
                previous_node: Node = cur
                node_removed = False

                # There could be a chain of nodes with in/out degree = 1. Resolve the whole chain before moving on to search more nodes
                while net.in_degree(current_node) == net.out_degree(current_node) == 1:
                    net.remove_edge([previous_node, current_node])

                    previous_node = current_node
                    temp = net.get_children(current_node)[0]
                    net.remove_nodes(current_node)
                    current_node = temp
                    node_removed = True

                # We need to connect cur to its new successor
                if node_removed:
                    # self.removeEdge([previous_node, current_node])
                    net.add_edges([cur, current_node])
                    current_node.set_parent([cur])
                    net_updated = True

                # Resume search from the end of the chain if one existed, or this is neighbor if nothing was done
                q.append(current_node)

        return net_updated

    while True:
        update = prune(net)
        if not update:
            break


def print_topology_newick(net: Network):
    newick = re.sub(r':\[.*?\]', '', net.newick())
    newick = newick.replace(" ", "")
    print(newick)

def is_tree(graph: Network):
    visited = set()
    stack = [graph.root()]   
    while stack:
        node = stack.pop()
        if node in visited:
            return False # cycle detected
        visited.add(node)

        for child in graph.get_children(node):
            if child in visited:
                return False # cycle detected
            stack.append(child)

    return len(visited) == len(graph.nodes)



def preorder_traversal(tree: Network):
    if not is_tree(tree):
        raise Exception("Not a valid tree")

    root = tree.root()
    if root is None:
        return []

    result = []
    stack = [root[0]]

    while stack:
        node = stack.pop()
        result.append(node)

        stack.extend(reversed(tree.get_children(node)))

    return result

def postorder_traversal(net: Network):
    root = net.root()
    stack = [root]
    searched_nodes = []
    node2index = {root: 0}

    while stack:
        top_node = stack[-1]
        index = node2index[top_node]
        if index == net.out_degree(top_node):
            searched_nodes.append(stack.pop())
        else:
            it = iter(net.get_children(top_node))
            for _ in range(index):
                next(it)
            child = next(it)

            if child in searched_nodes:
                node2index[top_node] = index + 1
            else:
                stack.append(child)
                node2index[child] = 0
    return searched_nodes


def init_node_heights(graph: Network):
    nodes = postorder_traversal(graph)
    for node in nodes:
        if not node.attribute_value('t'):
            node.add_attribute('t', 0)

        for par in node.get_parent(return_all=True):
            branch_length = node.length()[par][0]
            if not par.attribute_value('t'):
                par.add_attribute('t', branch_length + node.attribute_value('t'))














