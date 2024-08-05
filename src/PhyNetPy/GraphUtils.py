""" 
Author : Mark Kessler
Last Edit : 3/28/24
First Included in Version : 1.0.0

Docs   - [x]
Tests  - [ ]
Design - [ ]
"""
from Network import Network, Edge, Node
from collections import deque
from NetworkParser import *
import networkx as nx
import matplotlib.pyplot as plt 


def _retic_edge_choice(retic_map : dict[Node, list[Edge]], 
                      funcs : list[dict[Node, Edge]]) -> list[dict[Node, Edge]]:
    """
    Recursive helper function that generates all combinations of hybrid edges
    to remove from a network to create an underlying tree structure.

    Args:
        retic_map (dict[Node, list[Edge]]): _description_
        funcs (list[dict[Node, Edge]]): _description_

    Returns:
        list[dict[Node, Edge]]: _description_
    """
    if retic_map != {}:
        node = list(retic_map.keys())[0]
        hybrid_edges = retic_map[node]
       
        for index in range(len(funcs) / 2):
            funcs[index][node] = hybrid_edges[0]
        for index in range(len(funcs) / 2, len(funcs)):
            funcs[index][node] = hybrid_edges[1]
         
        del retic_map[node]
        
        _retic_edge_choice(retic_map, funcs)
        
    return funcs



def get_all_clusters(net : Network, node : Node) -> set:
    """
    Compile a list of non-trivial clusters (size > 1) that make up this graph.
    Ie: for a graph ((A, B)C, D); , set of all clusters is {(A,B), (A,B,C)}

    Args:
        net (Network): the network to operate on
        node (Node): For any user call, this should be the root. 
                     For internal calls, it is the starting point for search.

    Returns:
        set: A set of all clusters in this graph. Each cluster is represented 
             as a set of either names or nodes.
    """
    
    cluster_set = set()
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
            cluster_set = cluster_set.union(net.get_all_clusters(net, child))
    
    return cluster_set

def combine(left : Network, right : Network) -> Network:
    merger = Network()
    
    left_copy, oldnew_left = left.duplicate()
    right_copy, oldnew_right = right.duplicate()
    
    pass

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
    new_sub_root = sub_root.duplicate()
    subnet.add_nodes(new_sub_root)
    old_new_map = {sub_root : new_sub_root}
    q = deque(sub_root)
    
    while len(q) != 0:
        cur = q.popleft()
    
        for child in net.get_children(cur):
            new_child = child.duplicate()
            old_new_map[child] = new_child
            subnet.add_nodes(new_child)
            
            #Copy edge info
            old_edge = net.get_edge(cur, child)
        
            #Add equivalent edge
            new_edge = DiEdge(old_new_map[cur], new_child)
            new_edge.set_gamma(old_edge.get_gamma())
            new_edge.set_length(old_edge.get_length())

            subnet.add_edges(new_edge)
            
            #Add child to queue
            q.appendleft(child)
    
    return subnet
    
def get_trees(net : Network) -> list[Network]:
    """_summary_

    Args:
        net (Network): _description_

    Returns:
        list[Network]: _description_
    """
    retics = [node for node in net.get_nodes() if node.is_reticulation()]
    retic2edges = {node : net.in_edges(node) for node in retics}
    retic_maps : list[dict[Node, Edge]] = _retic_edge_choice(retic2edges, [])
    
    trees = []
    
    preop : Network
    for i in range(len(retic_maps)):
        preop, old_new = net.duplicate()
        func : dict[Node, Edge] = retic_maps[i]
        
        for edge in func.values():
            preop.remove_edge(old_new[edge.src], old_new[edge.dest])
        
        preop.clean()
        trees.append(preop)
    
    return trees
        
        
def dominant_tree(net : Network) -> Network:
    """
    

    Args:
        net (Network): _description_

    Returns:
        Network: _description_
    """
    dom : Network = Network()
    
    edges_2_remove = []
    old_new_node_map = {}
    
    #Only include reticulation edges that are the maximum inheritance prob
    for node in [retic for retic in net.get_nodes() if retic.is_reticulation()]:
        retic_edges = [e for e in net.in_edges(node)]
        sorted_by_gamma = retic_edges.sort(key=lambda x: x.gamma, reverse=False)
        #remove all but maximum inheritance probability
        edges_2_remove.extend(sorted_by_gamma[:-1])
    
    #Add all nodes from original network
    for node in net.get_nodes():
        new_node = node.duplicate()
        dom.add_nodes(new_node)
        old_new_node_map[node] = new_node
    
    #Add only dominant reticulation edges and all other normal edges
    for edge in net.get_edges():
        if edge not in edges_2_remove:
            new_src = old_new_node_map[edge.src]
            new_dest = old_new_node_map[edge.dest]
            dom.add_edges(edge.duplicate(new_src, new_dest))
      
    #Clean artifacts created by removing some of the retic edges      
    dom.clean()
    
    return dom


     
        
        
        
def test():
    net : Network = NetworkParser('/Users/mak17/Documents/PhyNetPy/src/bubble_J.nex').get_all_networks()[0]
    G = net.to_networkx()
    for layer, nodes in enumerate(nx.topological_generations(G)):
    # `multipartite_layout` expects the layer as a node attribute, so add the
    # numeric layer value as a node attribute
        for node in nodes:
            G.nodes[node]["layer"] = layer

    # Compute the multipartite_layout using the "layer" node attribute
    pos = nx.multipartite_layout(G, subset_key="layer")

    fig, ax = plt.subplots()
    nx.draw_networkx(G, pos=pos, ax=ax)
    ax.set_title("DAG layout in topological order")
    fig.tight_layout()
    plt.show()
    
        
