""" 
Author : Mark Kessler
Last Edit : 3/28/24
First Included in Version : 1.0.0
Approved for Release: Yes, post test-suite final inspection, if included.
"""
from Network import Network, Edge, Node

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