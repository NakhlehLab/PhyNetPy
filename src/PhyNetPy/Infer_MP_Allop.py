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
Author: Mark Kessler

Description: This file contains the method Infer_MP_Allop_2.0, which 
is a maximum parsimony approach to inferring phylogenetic networks that contain
allopolyploid, polyploid, and autopolyploid species, given a set of gene trees.

Last Edit: 3/11/25
Included in version : 1.0.0

Docs   - [ ]
Tests  - [x]
Design - [x]

"""
from __future__ import annotations
import copy
from Network import *
import GraphUtils as utils
from collections import defaultdict, deque
import numpy as np
from NetworkParser import NetworkParser
from BirthDeath import Yule
from ModelGraph import *
from ModelFactory import *
from MetropolisHastings import *
from State import *
from NetworkMoves import *
import random

"""
Sources:

1)
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000501

2)
https://doi.org/10.1371/journal.pgen.1002660

3)
https://academic.oup.com/sysbio/article/71/3/706/6380964
"""

"""
Clusters are represented a few different ways in this file.

1) frozen set of strings 
2) frozen set of Nodes
3) set of Nodes 
4) set of strings 

(where the strings are node names)


The ILP (integer linear programming) Algorithm used to compute an MDC tree can 
be found here:

https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000501

The rest of this file is based on the algorithms described here: 

https://academic.oup.com/sysbio/article/71/3/706/6380964

"""

###########################
#### EXCEPTION HANDLER ####
###########################

class InferAllopError(Exception):
    """
    Exception class for Infer_MP_Allop related operation errors.
    """
    def __init__(self, message : str = "Error during the execution of \
                                        Infer_MP_Allop_2.0") -> None:
        """
        Initialize a new error message

        Args:
            message (str, optional): The error message. Defaults to "Error
                                    during the execution of Infer_MP_Allop_2.0".
        Returns:
            N/A
        """
        
        self.message = message
        super().__init__(self.message)

#########################
#### HELPER FUNCTIONS ###
#########################

def _dict_subtract(cur : dict[Node, int],
                   goal : dict[Node, int]) -> dict[Node, int]:
    """
    Subtract the current ploidy for each leaf in the network from the goal
    ploidy (derived from the subgenome mapping), to see which leaves need to 
    have increased ploidy.

    Raises:
        Exception: if cur and goal have unequivalent leaf sets. 
    Args:
        cur (dict[Node, int]): map from leaves to their ploidy levels 
        goal (dict[Node, int]): map from leaves to their desired ploidy level.
    Returns:
        dict[Node, int]: a mapping of leaves to the amount of ploidy that needs
                         to be added to achieve the goal ploidy.
    """
    if cur.keys() == goal.keys():
        dif = {key : goal[key] - cur[key] for key in cur.keys()}
        return dif
    raise Exception("Error subtracting dicts. Key sets are not the same!")    

def _nodes_to_improve(net : Network, 
                      n : Node,
                      nti : dict[Node, int], 
                      lti : dict[Node, int]) -> int:
    """
    Compute the maximum amount of ploidy that needs to be added for
    each leaf and internal node. This number, for any node, is the minimum over
    its set of child nodes.

    Args:
        net (Network): Network
        n (Node): Node for which to compute the amount of ploidy needed
        nti (dict[Node, int]): Map that accumulates results for each node.
        lti (dict[Node, int]): Map that contains starter values for each leaf.

    Returns:
        int: The amount of ploidy needed for @n
    """
    #base case
    if net.out_degree(n) == 0:
        nti[n] = lti[n]
    else: #recursive case
        nti[n] = min([_nodes_to_improve(net, node, nti, lti) \
                      for node in net.get_children(n)])
    return nti[n]

def _valid_hyb_src(net : Network, 
                   cluster_par : Node,
                   ploidy : int) -> list[Edge]:
    """
    Gather the valid source edges for a hybrid edge to connect to an in-edge
    of 'cluster_par'.
    
    Edges are also filtered by their ploidy values, any valid edge must be <=
    'ploidy'

    Args:
        net (Network): Network
        cluster_par (Node): Parent node of the clade that will be given 
                            additional ploidy via the new hybrid edge.
        ploidy (int): Maximum amount of ploidy that can be added to the clade.

    Returns:
        list[Edge]: List of valid edges.
    """
    valid_edges = []
    edges_2_ploidy = net.subgenome_ct_edges(delta = ploidy)
    
    subnet_edges = net.edges_downstream_of_node(cluster_par)
    for edge in net.E():
        if edge not in subnet_edges and edge in edges_2_ploidy.keys():
            valid_edges.append(edge)
    
    return valid_edges
                  
def _ploidy_dif(goal : dict[Node, int], cur : dict[Node, int]) -> bool:
    """
    Checks if cur has achieved goal ploidy.

    Args:
        goal (dict[Node, int]): map of leaf nodes to their desired ploidy
        cur (dict[Node, int]): map of leaf nodes to their current ploidy

    Returns:
        bool: if goal and cur are "equivalent" mappings.
    """
    #if even one value is different, maps are not equivalent.
    for leaf in goal.keys():
        if cur[leaf] != goal[leaf]:
            return False
    return True
    
def _attach(net : Network, nti : dict[Node, int]) -> None:
    """
    Given a set of nodes that need to be edited as to increase ploidy, add a 
    hybrid edge in such a location as to increase the ploidy of a maximally 
    sized set of nodes that need to be improved.
    
    Leads to a starting network with a greedily minimum amount of reticulations
    and that groups as many leaves into one "polyploid" clade as possible.

    Args:
        net (Network): The Network.
        nti (dict[Node, int]): A map of nodes in @net to the amount of ploidy
                               that the node needs to gain.
    Return:
        N/A

    """
    
    #Map all nodes to their current subgenome counts
    subgenome_count_nodes = {node : net.subgenome_count(node) 
                             for node in net.V()}
    
    #Get all clusters, including clusters of size 1
    clusters : set[tuple[Node]] = utils.get_all_clusters(net, 
                                                         include_trivial = True)
    
    #compute largest cluster in the set of clusters where each node in the 
    #cluster is a node that we need to increase the ploidy
    max_c : tuple[Node] = tuple()
    for cluster in clusters:
        is_valid = True
        for node in cluster:
            if node not in nti.keys():
                is_valid = False
            else:
                if nti[node] == 0:
                    is_valid = False
        
        if is_valid and len(cluster) > len(max_c):
            max_c = cluster
    
    #Get MRCA of cluster, this node is the node whose in edge is the hybrid 
    #destination, so that we can edit the ploidy for all nodes under it.
    
    par_of_maxc = net.mrca(set(max_c))

    # We can only increase the ploidy by the minimum. IE cluster (A, B, C).
    # If A needs 2, B needs 2, but C only needs 1, we can only add 1 ploidy 
    # value to mrca(A, B, C).
    min_ploidy_of_max_c = min([subgenome_count_nodes[n] for n in max_c])
    
    # Take a random (most likely has one edge anyways) in edge of the parent 
    # and attach that edge to another edge.
    hyb_dest_edge : Edge = random.choice(net.in_edges(par_of_maxc))
    
    # Calculate valid hybrid source edges as to 1) not make a cycle 2) add the
    # correct amount of ploidy
    potential_edges = _valid_hyb_src(net, par_of_maxc, min_ploidy_of_max_c)
    
    # Choose any of the valid sources
    hyb_src_edge : Edge = random.choice(potential_edges)
    
    # Finally, add the hybrid edge. An acceptable amount of ploidy will have 
    # been added to the network such that other leaves will remain unaffected.
    add_hybrid(net, hyb_src_edge, hyb_dest_edge)
     
def _resolve_ploidy(net : Network, 
                    subgenomes : dict[str, list[str]]) -> Network:
    """
    Given a tree and a subgenome mapping of network leaves to genes, add 
    reticulation edges such that each leaf has the desired ploidy.

    Args:
        net (Network): A standard binary tree. Each leaf will have ploidy 1.
        subgenomes (dict[str, list[str]]): A subgenome mapping. Maps names of 
                                           subgenomes to the gene names.
    Returns:
        Network: A reconciled network with correct ploidy.
    """
    
    # Compute the current amount of ploidy, and the desired amount of ploidy
    cur_ploidy = {node : net.subgenome_count(node) 
                  for node in net.get_leaves()}
    goal_ploidy = {node : len(subgenomes[node.label]) 
                   for node in net.get_leaves()}
    
    # Keep connecting and adding ploidy to maximally sized clusters, until
    # each leaf has the desired ploidy.
    while not _ploidy_dif(goal_ploidy, cur_ploidy):
        
        #Figure out all the nodes that need increases in ploidy
        lti = _dict_subtract(cur_ploidy, goal_ploidy)
        nti = {}
        _nodes_to_improve(net, net.root(), nti, lti)
    
        #Determine valid reticulation spots and add the hybrid edge
        _attach(net, nti)
       
        #Recalculate leaf ploidy
        cur_ploidy = {node : net.subgenome_count(node) 
                  for node in net.get_leaves()} 

    return net

def random_object(mylist : list[Any], rng : np.random.Generator) -> object:
    """
    Select a random object from a list using a default rng object from numpy

    Args:
        mylist (list[Any]): any list of objects
        rng (np.random.Generator): numpy default rng object

    Returns:
        object: could be anything that is contained in mylist
    """
    rand_index : int = rng.integers(0, len(mylist)) # type: ignore
    return mylist[rand_index]
    
def cluster_as_name_set(cluster : set[Node]) -> frozenset[str]:
    """
    Convert cluster from a set of nodes to a set of strings (names).

    Args:
        cluster (set[Node]): One form of a "cluster"

    Returns:
        frozenset[str]: The set of node names in the cluster.
    """
    return frozenset([node.label for node in cluster])
    
def clusters_contains(cluster : set[Node], 
                      set_of_clusters : set[set[Node]]) -> bool:
    """
    Check if a cluster is in a set of clusters by checking names
    (the objects can be different, but two clusters are equal if their 
    names are equal).
    
    Args:
        cluster (set[Node]): a cluster
        set_of_clusters (set[set[Node]]): a set of clusters

    Returns:
        bool: True, if cluster is an element of set_of_clusters. False if not.
    """
    names = cluster_as_name_set(cluster)
    
    for item in set_of_clusters:
        names_item = cluster_as_name_set(item)
        if names == names_item:
            return True
    return False

def cluster_partition(cluster : frozenset[Node],
                      processed : dict[set[Node], Node]) -> frozenset:
    """
    Given a cluster such as ('A', 'B', 'C'), if a cluster such as ('A', 'B') 
    has already been processed, split the original cluster into subsets -- 
    {('A', 'B'), ('C')}.

    Args:
        cluster (frozenset[Node]): A cluster.
        processed (dict[set[Node], Node]): A mapping from clusters to the Node 
                                           obj that is the root of that cluster.

    Returns:
        frozenset: the partitioned cluster
    """
    # Allow mutations
    editable_cluster = set(cluster) 
    
    # Build up partioned cluster from scratch
    new_cluster = set() 
    
    # Search already processed clusters for a cluster
    # that is a subset of the original cluster
    for subcluster in processed.keys():
        if subcluster.issubset(editable_cluster):
            #add subset cluster to the partion set
            new_cluster.add(subcluster)
            
            #remove the items from the subset 
            for item in subcluster:
                editable_cluster.remove(item)
    
    #Add the remaining items that are not apart of a pre-existing cluster
    for item in editable_cluster:
        new_cluster.add(item)
    
    # convert partioned cluster to an immutable version for hashing
    return frozenset(new_cluster)
          
def generate_tree_from_clusters(tree_clusters : set[str]) -> Network:
    """
    Given a set of clusters (given by the taxa labels, ie {('A','B','C'), 
    ('B','C'), ('D','E','F'), ('D','E')}), reconstruct the tree it represents.

    Args:
        tree_clusters (set[str]): A set of tree clusters, as described above.
    Returns:
        Network: the MDC network.
    """
    
    net : Network = Network()
    root : Node = Node(name = "ROOT")
    net.add_nodes(root)
    
    #start with clusters of size 2
    i = 2 
    
    #internal node UIDs
    j = 1
    
    is_root = False
    
    processed_clusters : dict[set, Node] = {}
    root_children : list[Node] = []
    
    # remove clusters as they are processed
    while len(tree_clusters) != 0:
        
        # get all clusters of a certain size
        clusters_len_i = [c for c in tree_clusters if len(c) == i]
        
        # last round, so the 2 clusters need to point to the root instead of two 
        # different parents!
        if len(tree_clusters) == len(clusters_len_i):
            is_root = True
            
        # Process smallest clusters first
        for cluster in clusters_len_i:
            
            cluster_parent = Node(name = f"Internal_{j}")
            net.add_nodes(cluster_parent)
            
            j += 1
            
            if is_root:
                root_children.append(cluster_parent)
            
            #detect previously encountered clusters in current cluster
            partitioned_cluster = cluster_partition(cluster, processed_clusters)
            
            for subtree in partitioned_cluster:
                # Already found the cluster before
                if type(subtree) == frozenset: 
                    #connect previous cluster to this current cluster parent
                    new_edge =Edge(cluster_parent, processed_clusters[subtree])
                    net.add_edges(new_edge)
                else: 
                    # subtree is simply a taxa label (a string), 
                    # so create a new node
                    taxa_node = Node(name = subtree)
                    net.add_nodes(taxa_node)
                    net.add_edges(Edge(cluster_parent, taxa_node))
                
            processed_clusters[cluster] = cluster_parent
            tree_clusters.remove(cluster)
        
        #process next sized set of clusters
        i += 1 


    #connect the 2 disjoint clusters together with the root
    for root_child in root_children:   
        net.add_edges(Edge(root, root_child))
        
    return net

def partition_gene_trees(gene_map : dict[str, list[str]], 
                         rng : np.random.Generator = None) -> Network:
    """
    Generate a starting network given a subgenome mapping.
    
    Turn this map:
     
    {"B": ["01bA"], "A": ["01aA"], "C": ["01cA"], "X": ["01xA", "01xB"], 
    "Y": ["01yA", "01yB"], "Z": ["01zA", "01zB"]}
    
    into a network in which B, A, and C are all diploid while X, Y, and Z have 
    ploidy 2.

    Args:
        gene_map (dict[str, list[str]]): Map from gene names to a list of 
                                         copy names.
        rng (np.random.Generator): A random generator instance.
                                   Defaults to None.

    Returns:
        Network : A bootstrapped starting network with correct ploidy values.
    """
  
    #Start with a network with the number of diploid leaves
    yule_generator : Yule = Yule(.1, len(gene_map.keys()), rng = rng)
    simple_network : Network = yule_generator.generate_network()

    # Change the names of the leaves to match the keys of the gene map
    name_idx = 0
    names = list(gene_map.keys())
    for leaf in simple_network.get_leaves():
        simple_network.update_node_name(leaf, names[name_idx])
        name_idx += 1
            
    # Network is now a tree with the right names and right amount of leaves.
    # Now use the maxcluster-minploidy algorithm to finish building the network.
    return _resolve_ploidy(simple_network, gene_map)

def get_other_copies(gene_tree_leaf : Node, 
                     gene_map : dict[str, list[str]]) -> list[str]:
    """
    Given a gene tree leaf, get all other gene copy names for the
    Taxon for which gene_tree_leaf is a value.
    
    Raises:
        InferAllopError: raised if gene_tree_leaf is not listed in the taxon map 
                         as an item of any value

    Args:
        gene_tree_leaf (Node): Leaf node of a gene tree
        gene_map (dict[str, list[str]]): a taxon map.
        
    Returns:
        list[str]: List of other gene copy names
    """
    for copy_names in gene_map.values():
        if gene_tree_leaf.label in copy_names:
            return copy_names
    
    raise InferAllopError(f"Leaf name '{gene_tree_leaf.label}' not found \
                            in gene copy mapping")
          
def allele_map_set(g : Network, 
                   gene_map : dict[str, list[str]]) -> list[AlleleMap]: 
    """
    Let a MUL tree, T', have taxa labels drawn from the set X 
    (keys of gene_map input dict). Calculate all possible mappings from taxa 
    labels of g (values of gene_map input dict) into X.
    
    Args:
        g (Network): A gene tree
        gene_map (dict[str, list]): A taxon map.

    Returns:
        list[AlleleMap]: a list of functions that map labels of g to 
                         labels of a MUL tree.
    """
    
    funcs : list[AlleleMap] = []
    funcs.append(AlleleMap())
    
    #Map each gene tree leaf to a mul tree leaf
    for gleaf in g.get_leaves():
        
        new_funcs = []
        
        #Only consider options for gleaf that correspond to the gene_map.
        other_copies = get_other_copies(gleaf, gene_map)
        
        for mul_leaf in other_copies:
            
            copy_funcs = copy.deepcopy(funcs)
            for func in copy_funcs:
                status = func.put(gleaf, mul_leaf)
                #map was successful, function is valid
                if status == 1: 
                    new_funcs.append(func)
                
        funcs = new_funcs  
            
    return funcs      

##############################
### MUL TREE & Allele Maps ###
##############################

class Allop_MUL(MUL):
    """
    A Standard MUL tree, but contains relevant methods for calculating the 
    maximum parsimony score for Infer_MP_Allop_2.0
    """
    def extra_lineages(self, 
                       coal_event_map : dict[tuple, str], 
                       f : AlleleMap) -> int:
        """
        Computes the number of extra lineages in a mapping from a gene tree T, 
        into a MUL tree, T'.

        Args:
            coal_event_map (dict[tuple[str, str], str]): A mapping from edges in 
                                                         T' to a list of nodes 
                                                         of T that have been 
                                                         mapped into that edge.
                                                         All nodes are by name, 
                                                         not object. Edges here
                                                         are represented by 
                                                         tuples with node names.
            
            f (AlleleMap): An allele map. Used to ensure the correct number of 
                           lineages for each leaf branch. Some leaves may not be 
                           included in the map.

        Returns:
            int: number of extra lineages
        """
        # Map edges to the number of lineages present in the branch
        edge_2_xl = {}
        
        root = self.mul.root()
        
        # Populate the edge to lineage mapping by calling the xl_helper function
        # at the root (propagates through the whole network)
        self.xl_helper(root, edge_2_xl, coal_event_map, f)
        
        #START COUNTING EXTRA LINEAGES 
        
        # Root of g will be mapped to this edge, 
        # but doesn't count as a coal event
        root_xl = edge_2_xl[(None, root.label)][0] - len(coal_event_map[(None, root.label)]) - 1 
    
        extra_lin_total = 0
        mul_leaves = self.mul.get_leaves()
        
        # Process each edge in the mul tree and tabulate 
        # the extra lineage count.
        for edge in self.mul.E():

            # Code follows Lemma 1 from (1):
            # The number of extra lineages in a branch is equal to the number of 
            # lineages exiting it, minus the number of coalescent events within 
            # the branch, minus 1 (minus 0 if its a leaf branch).
            if edge.src is not None:
                extra_lin_total += edge_2_xl[edge.to_names()][0] 
                extra_lin_total -= 1
                
                # branches connected to leaves will have no coalescent events
                if edge.dest in mul_leaves: 
                    extra_lin_total -= 0
                elif edge.dest is not None:
                    extra_lin_total -= len(coal_event_map[edge.to_names()]) 
    
        # Add in any extra lineages in the root branch. 
        extra_lin_total += root_xl 
        
        return extra_lin_total
    
    def xl_helper(self, 
                  start_node : Node, 
                  edge_xl_map : dict,
                  coal_map : dict[tuple[str, str], str], 
                  f : AlleleMap) -> None:
        """
        Modifies the edge_xl_map parameter. This is a recursive function that 
        processes an edge in T'. The number of extra lineages exiting a branch 
        is equal to the sum of its child lineages entering the branch, minus 
        the number of coalescent events, minus 1.

        Args:
            start_node (Node): Node to calculate the number of 
                               lineages exiting/entering it
            edge_xl_map (dict[tuple[str, str], list[int]): Mapping from edges to
                                                           the number of 
                                                           lineages 
                                                           exiting/entering it
            coal_map (dict[tuple[str, str], str]): Mapping from edges to gene 
                                                   tree internal node names.
            f (AlleleMap): allele mapping (gene tree leaf names to mul tree leaf 
                           names -- types are strings).
        Returns:
            N/a
        """
        # Pull the mapping from the AlleleMap object
        fmap : dict[str, str] = f.map
        
        if start_node in self.mul.get_leaves():
            # should definitely not be None, and there should only be 1.
            par : Node  = self.mul.get_parents(start_node)[0]
            
            if start_node.label in fmap.values():
                #bottom, top of branch will each be 1
                edge_xl_map[(par.label, start_node.label)] = [1, 1] 
            else:
                #There is no gene tree leaf mapped to this mul tree leaf :(
                edge_xl_map[(par.label, start_node.label)] = [0, 0] 
        else:
            if start_node == self.mul.root():
                par : Node = None
            else:
                par : Node  = self.mul.get_parents(start_node)[0] 
                
            sum_of_child_tops = 0
            
            #Get each child's top lineage value 
            for child in self.mul.get_children(start_node):
                self.xl_helper(child, edge_xl_map, coal_map, f)
                e = (start_node.label, child.label)
                sum_of_child_tops += edge_xl_map[e][1]
            
            # Special case that start_node == root is not important since the 
            # top value is never used. Each node that is mapped to the 
            # par->start edge is a coal event that combines 2 lineages into 1
            par_name : str = None
            if par is not None:
                par_name = par.label
                
            bottom = sum_of_child_tops
            coal_events = len(coal_map[(par_name, start_node.label)])
            top = sum_of_child_tops - coal_events
            
            edge_xl_map[(par_name, start_node.label)] = [bottom, top]
            
    def gene_tree_map(self,
                      g : Network, 
                      leaf_map : AlleleMap, 
                      mrca_cache: dict[frozenset[Node], Node]) -> dict[tuple, list[str]]:
        """
        Maps a gene tree (T) into a MUL tree (T'), where each have taxa 
        from the set X.

        Args:
            g (Network): The gene tree
            leaf_map (AlleleMap): A function f: string -> string from gene tree 
                                  leaf names to MUL tree leaf names. 
                                  An allele map.
            mrca_cache (dict[frozenset[Node], Node]): A cache for MRCA
                                                      computations.
        Returns:
            dict[tuple, list[str]]: A mapping from edges in T' to a list of nodes 
                                    of T that have been mapped into that edge.
                                    All nodes are by name, not object.

        """
        # Let v′ = MRCA_T′(C_T(v)), and let u′ be the parent node of v′. 
        # Then, v is mapped to any point pv, excluding node u′, 
        # in the branch (u′, v′) in T′
        tnode_2_edgeloc = {} 
        
        # Map of edges to the set of nodes that get mapped within that edge.
        edgeloc_2_tnode = defaultdict(list)
        
        # Map each internal node of the gene tree into an edge of the MUL tree
        gene_tree_leaves = g.get_leaves()
        mul_root = self.mul.root()
        
        #grab results of the prior leaf_descendants_all() call
        leaf_desc_map = g.get_item("leaf descendants")
        
        for node in g.V():
            #only map internal nodes
            if node not in gene_tree_leaves: 
                c_t_ofv = leaf_desc_map[node] 

                cluster_names = [leaf_map.map[leaf.label] 
                                 for leaf in c_t_ofv]
                cluster = frozenset(cluster_names)
                
                # Calculate the most recent common ancestor of all the nodes in
                # the cluster in the mul tree.
                try:
                    v_prime : Node = mrca_cache[cluster]
                except:
                    v_prime : Node = self.mul.mrca(cluster) 
                    mrca_cache[cluster] = v_prime 
                
                edge = [None, v_prime.label]
                
                # Calculate v_prime's parent
                if v_prime == mul_root:
                    u_prime = None
                    e = (None, v_prime.label)
                    tnode_2_edgeloc[node.label] = e
                else:
                    u_prime : Node = self.mul.get_parents(v_prime)[0] 
                    edge[0] = u_prime.label
                    e = (u_prime.label, v_prime.label)
                    tnode_2_edgeloc[node.label] = e
                
                
                #Add node to edge mapping
                if tuple(edge) in edgeloc_2_tnode.keys():
                    edgeloc_2_tnode[tuple(edge)].append(node.label)
                else:
                    edgeloc_2_tnode[tuple(edge)] = [node.label]
        
        return edgeloc_2_tnode           
            
    def XL(self, g : Network, mrca_cache : dict[frozenset[Node], Node]) -> int:
        """
        Computes the number of extra lineages in the map from the gene tree g, 
        into MUL tree T.
        EQ1 from (3)
        
        Args:
            g (Network): A gene tree
            mrca_cache (dict[frozenset[Node], Node]): A cache for MRCA
                                                      computations.
        Returns:
            int: the minimum number of extra lineages over all 
                 possible allele maps
        """
        xl = [self.XL_Allele(g, allele_map, mrca_cache) 
                    for allele_map in g.get_item("allele maps")]

        return min(xl)

    def XL_Allele(self, 
                  g : Network, 
                  f : AlleleMap, 
                  mrca_cache : dict[frozenset[Node], Node]) -> int:
        """
        Compute the extra lineages given a MUL tree T, a gene tree, g, 
        and an allele mapping f.

        Args:
            g (Network): A gene tree
            f (AlleleMap): An allele map from leaves of g to leaves of T
            mrca_cache (dict[frozenset[Node], Node]): A cache for MRCA
                                                      computations. 
        Returns:
            int: number of extra lineages
        """
        #map the gene tree into the MUL tree
        edge_2_nodes = self.gene_tree_map(g, f, mrca_cache)
        
        #compute the extra lineages
        return self.extra_lineages(edge_2_nodes, f)

    def score(self, gt_list : list[Network]) -> int:
        """
        Compute the total score of a MUL tree given a list of gene trees.
        Right side of EQ 2 from EQ1 from (3)

        Args:
            gt_list (list[Network]): a list of gene trees

        Returns:
            int: The MUL tree score 
        """
        # Sum the XL values over all gene trees
        mrca_cache = {}    
        gt_scores = [self.XL(gt, mrca_cache) for gt in gt_list]
        return sum(gt_scores)

class AlleleMap:
    """
    Data structure that holds a mapping from gene tree leaf names to mul tree 
    leaf names. Internally handles the mechanism for making sure gene copies 
    are not mapped to the same subgenome.
    """
    
    def __init__(self) -> None:
        """
        Initialize an empty allele mapping.
        
        Args:
            N/A
        Returns:
            N/A
        """
        self.map = dict()
        self.disallowed = set() 

    def put(self, g_leaf : Node, mul_leaf : str) -> int:
        """
        Map a gene tree leaf to a MUL tree leaf

        Args:
            g_leaf (Node): a gene tree leaf
            mul_leaf (str): a MUL tree leaf label

        Returns:
            int: 0 if the mapping was unsuccessful, 1 if it was.
        """
        
        # If a gene tree leaf has already been mapped to a mul tree leaf, 
        # disallow the mapping
        if mul_leaf in self.disallowed:
            return 0
        
        # Otherwise, set the mapping and avoid mapping a gene tree leaf to 
        # this mul leaf in the future
        self.map[g_leaf.label] = mul_leaf
        self.disallowed.add(mul_leaf)
        return 1

#####################
#### MODEL BUILD ####
#####################

class InferMPAllop:
    """
    Class that is used to instantiate a new usage of the Infer_MP_Allop_2.0 
    method.
    """
    
    def __init__(self, 
                 network : Network, 
                 gene_map : dict[str, list[str]], 
                 gene_trees : list[Network], 
                 iter : int, 
                 rng : np.random.Generator) -> None:
        """
        Initialize the parameters for this method call.

        Args:
            network (Network): A starting network.
            gene_map (dict[str, str]): A subgenome mapping, from subgenome names
                                       to the list of names of genes that are 
                                       present in that subgenome.
            gene_trees (list[Network]): A set of gene trees.
            iter (int): Number of iterations to run the chain.
            rng (np.random.Generator): random number generator, for consistent 
                                       testing.
        Returns:
            N/A
        """
        
        for gene_tree in gene_trees:
            #Calculate the allele maps for this gene tree and store the results.
            allele_funcs = allele_map_set(gene_tree, gene_map)
            gene_tree.put_item("allele maps", allele_funcs)
            
            #Calculate key pieces of network data for use later
            leaf_descendants = gene_tree.leaf_descendants_all()
            gene_tree.put_item("leaf descendants", leaf_descendants)

        # Build the model using the model factory.
        mp_allop_comp = MPAllopComponent(network, gene_map, gene_trees, rng)
        model_fac : ModelFactory = ModelFactory(mp_allop_comp)
        self.mp_allop_model : Model = model_fac.build()
        self.iter = iter
        self.results = {}
        
    def run(self) -> float:
        """
        Computes the network with the lowest (highest likelihood) parsimony 
        score over the set of given gene trees.

        Args:
            N/A
        Returns:
            float : parsimony score of the most likely Network
        """
        hc = HillClimbing(Infer_MP_Allop_Kernel(),
                          num_iter = self.iter, 
                          model = self.mp_allop_model)
        end_state : State = hc.run()
        self.results = hc.nets_2_scores
        return end_state.likelihood()
    
class MPAllopComponent(ModelComponent):
    """
    Model Component that sets up the model for Infer MP Allop 2.0.
    """
    
    def __init__(self, 
                 network : Network, 
                 gene_map : dict[str, list[str]],
                 gene_trees : list[Network], 
                 rng : np.random.Generator) -> None:
        """
        Initialize the MP Allop model component

        Args:
            network (Network): A starting network.
            gene_map (dict[str, str]): Subgenome mapping.
            gene_trees (list[Network]): Set of gene trees.
            rng (np.random.Generator): Random number generator.
        Returns:
            N/A
        """
        
        super().__init__(set())
        
        self.network : Network = network
        self.gene_map : dict[str, list[str]] = gene_map
        self.gene_trees : list[Network] = gene_trees
        self.rng : np.random.Generator = rng
        
    
    def build(self, model : Model) -> None:
        """
        Attaches the MP allop component to the main model.

        Args:
            model (Model): A model, under construction.
        Returns:
            N/A
        """
        # Add mul tree 
        mul_node : MULNode = MULNode(self.gene_map, self.rng)
        
        # Add start network
        net_node : NetworkContainer = NetworkContainer(self.network)
        
        # Add gene trees
        gene_trees_node : GeneTreesNode = GeneTreesNode(self.gene_trees)
        
        # Add scoring function
        score_root_node : ParsimonyScore = ParsimonyScore()
        
        # Book-keeping TODO: Fix this.
        model.nodetypes["root"] = [score_root_node]
        model.nodetypes["internal"] = [mul_node]
        model.nodetypes["leaf"] = [gene_trees_node, net_node]
        model.network = self.network
        model.network_container = net_node
        
        # Connect the nodes.
        gene_trees_node.join(score_root_node)
        mul_node.join(score_root_node)
        net_node.join(mul_node)
        
class NetworkContainer(StateNode):
    """
    Instead of integrating the network into the model with each network node
    getting its own model node, use one model node to represent the network.
    In this case, the network represents "observed" data -- observed is in 
    quotes, due to that in most cases the network will be edited iteration over
    iteration.
    """
    def __init__(self, network : Network):
        """
        Put a network into this container.

        Args:
            network (Network): Any Network obj.
        Returns:
            N/A
        """
        super().__init__()
        self.network : Network = network
    
    def update(self, new_net : Network) -> None:
        """
        Replace the network that is currently stored with a new one, and update 
        any model nodes that rely on this data.

        Args:
            new_net (Network): The new Network obj to be stored.
        Returns:
            N/A
        """
        self.network = new_net
        model_parents : list[CalculationNode] = self.get_model_parents()
        for model_parent in model_parents:
            model_parent.upstream()
        
    def get(self) -> Network:
        """
        Grab the network obj stored in this container.

        Args:
            N/A
        Returns:
            Network: The stored Network obj.
        """
        return self.network

class MULNode(CalculationNode):
    """
    Node that stores a MUL tree network that is calculated based on a stored 
    network.
    """
    
    def __init__(self, 
                 gene_map : dict[str, list[str]],
                 rng : np.random.Generator) -> None:
        """
        Initialize a MUL tree and store it in this model node.

        Args:
            gene_map (dict[str, str]): A subgenome mapping.
            rng (np.random.Generator): A random number generator.
        Returns:
            N/A
        """
        super().__init__()
        self.multree : Allop_MUL = Allop_MUL(gene_map, rng)
        
    def calc(self) -> Allop_MUL:
        """
        Regenerate the mul tree, if the underlying network has been edited.

        Raises:
            InferAllopError: If the MUL node is not correctly linked to a
                             NetworkContainer.

        Args:
            N/A
        Returns:
            Allop_MUL: The newly generated MUL tree.
        """
        
        model_children = self.get_model_children()
        
        if len(model_children) == 1:
            if type(model_children[0]) is NetworkContainer:
                self.multree.to_mul(model_children[0].get())
            else:
                raise InferAllopError("Malformed MP Allop Model Graph. \
                                       Expected MUL Node to have Network \
                                       Container Child")
        self.cache(self.multree)
        return self.multree
    
    def sim(self) -> None:
        """
        No implementation for a MUL node for simulations.
        
        Args:
            N/A
        Returns:
            N/A
        """
        pass
    
    def update(self) -> None:
        """
        Flag this node and all upstream nodes as being in need of recalculation.
        
        Args:
            N/A
        Returns:
            N/A
        """
        self.upstream()
    
    def get(self) -> Allop_MUL:
        """
        Get the MUL tree if the underlying network has not been changed 
        recently, otherwise generate a new one, cache it 
        (via the calc function), and return it.

        Args:
            N/A
        Returns:
            Allop_MUL: A MUL tree that is up to date with the underlying network.
        """
        if self.dirty:
            return self.calc()
        else:
            return self.cached
    
class GeneTreesNode(StateNode):
    """
    A set of observed gene trees.
    """
    
    def __init__(self, gene_tree_list : list[Network]) -> None:
        """
        Store a set of gene trees in this model node.

        Args:
            gene_tree_list (list[Network]): A list of gene trees.
        Returns:
            N/A
        """
        super().__init__()
        self.gene_trees : list[Network] = gene_tree_list
    
    def update(self, new_tree : Network, index : int) -> None:
        """
        Replace a gene tree at index 'index' in the gene tree list, and updates
        any model nodes that rely on this data.

        Args:
            new_tree (Network): a new gene tree to take the place of an old one.
            index (int): The index into the gene tree list that is the spot for
                         the new gene tree.
        Returns:
            N/A
        """
        self.gene_trees[index] = new_tree
        model_parents : list[CalculationNode] = self.get_model_parents()
        for par in model_parents:
            par.upstream()
        
    def get(self) -> list[Network]:
        """
        Grab the list of gene trees.

        Args:
            N/A
        Returns:
            list[Network]: List of gene trees.
        """
        return self.gene_trees
    
class ParsimonyScore(CalculationNode):
    """
    Class that implements parsimony scoring for Infer MP Allop 2.0 based on
    the mp allop model.
    """
    def __init__(self) -> None:
        """
        Initialize the Parsimony Score node.
        
        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__()
        
    def calc(self) -> float:
        """
        The MP Allop score for a network over a set of gene trees.

        Raises:
            InferAllopError: If there is an error in the model formation or if 
                             something went wrong computing the score. 
        Args:
            N/A
        Returns:
            float: Parsimony score.
        """
        model_children = self.get_model_children()
        
        if len(model_children) == 2:
            g_trees : list[Network] = [child for child in model_children 
                                       if type(child) == GeneTreesNode][0].get()
            
            mul : Allop_MUL = [child for child in model_children 
                               if type(child) == MULNode][0].get()
            
            if mul.gene_map is None:
                # Invalid Network, abort.
                raise InferAllopError("An invalid network has been proposed\
                                       somehow")
            else:  
                # We are returning -1 times the score, since hill climbing uses 
                # the max function, and parsimony scores need to minimized.
                # Minimizing is the same as maximizing the negative of the 
                # score.
                return self.cache(-1 * mul.score(g_trees))
        else:
            raise InferAllopError("Malformed Model. Parsimony Score function \
                                   for MP ALLOP should only have 2 feeder \
                                   nodes")
    
    def sim(self) -> None:
        """
        Simulations not applicable for this node.
        
        Args:
            N/A
        Returns:
            N/A
        """
        pass
    
    def update(self) -> None:
        """
        Flag this node as being in need of recalculation.
        
        Args:
            N/A
        Returns:
            N/A
        """
        self.dirty = True
    
    def get(self) -> float:
        """
        If nothing has changed in the model, then return the cached score.
        If something has changed, then recompute the parsimony score.

        Args:
            N/A
        Returns:
            float: The mp allop 2.0 parsimony score.
        """
        if self.dirty:
            return self.calc()
        else:
            return self.cached  


#################
#### METHODS ####
#################

def INFER_MP_ALLOP_BOOTSTRAP(start_network_file : str, 
                             gene_tree_file : str,
                             subgenome_assign : dict[str, list[str]], 
                             iter_ct : int = 500, 
                             seed : int = None) -> dict[Network, float]:
    """
    Infer_MP_Allop_2.0, with a provided starting network.
    
    Given a set of gene trees, a subgenome assignment, and a starting network,
    infer the network that minimizes the parsimony score.

    Args:
        start_network_file (str): A nexus file that contains a starting network
        gene_tree_file (str): A nexus file that contains the gene trees
        subgenome_assign (dict[str, list[str]]): A mapping from genomes to the set
                                           of genes in them.
        iter_ct (int, optional): Number of iterations to run
                                 the inference chain. Defaults to 500.
        seed (int, optional): Random seed value. Defaults to None, in which
                                 case a random integer will be used.

    Returns:
        dict[Network, float]: A map from a small number of Networks to their    
                              parsimony scores
    """
    
    #init rng object
    if seed is None:
        rng = np.random.default_rng(random.randint(0, 1000))
    else:
        rng = np.random.default_rng(seed = seed)
    
    #Parse gene trees and starting network from files
    gene_tree_list : list[Network] = NetworkParser(gene_tree_file).get_all_networks()
    net_parser = NetworkParser(start_network_file)
    start_net = net_parser.get_all_networks()[0]

    #initialize model
    mp_model = InferMPAllop(start_net, 
                            subgenome_assign, 
                            gene_tree_list, 
                            iter_ct, 
                            rng = rng)
    
    #run the method
    mp_model.run()
    
    #return the results
    return mp_model.results

def INFER_MP_ALLOP(gene_tree_file : str, 
                   subgenome_assign : dict[str, list[str]] = None, 
                   iter_ct : int = 500,
                   seed : int  = None) -> dict[Network, float]:
    """
    Infer_MP_Allop_2.0.
    
    Given a set of gene trees, and a subgenome assignment, infer the network
    that minimizes the parsimony score.

    Args:
        gene_tree_file (str): A nexus file containing the gene trees
        subgenome_assign (dict[str, list[str]]): a map from genomes to their genes
        iter_ct (int, optional): Number of iterations to run the inference 
                                 chain. Defaults to 500.
        seed (int, optional): Random number generator seed value. 
                              Defaults to None, in which case a random value 
                              will be chosen.

    Returns:
        dict[Network, float]: a mapping from a small number of Networks to 
                              their parsimony scores.
    """
    gene_tree_list : list[Network] = NetworkParser(gene_tree_file).get_all_networks()
    
    if subgenome_assign is None:
        gts = GeneTrees(gene_tree_list)
        subgenome_assign = gts.mp_allop_map()
    
    if seed is None:
        rng = np.random.default_rng(random.randint(0, 1000))
    else:
        rng = np.random.default_rng(seed = seed)
        
    start_net = partition_gene_trees(subgenome_assign, rng = rng)
    
    mp_model = InferMPAllop(start_net, 
                            subgenome_assign, 
                            gene_tree_list, 
                            iter_ct, 
                            rng = rng)
    
    mp_model.run()
    
    return mp_model.results

def ALLOP_SCORE(net_filename : str, 
                gene_trees_filename : str, 
                subgenome_map : dict[str, list[str]]) -> int: 
    """
    Given a network, a set of gene trees, and a subgenome mapping, compute
    the parsimony score over all the gene trees.

    Args:
        net_filename (str): Network that will be scored
        gene_trees_filename (str): filename for a nexus file that contains 
                                   gene trees
        subgenome_map (dict[str, list[str]]): a mapping from subgenomes to their
                                              genes.

    Returns:
        int: parsimony score
    """
    rng = np.random.default_rng()
    
    T = Allop_MUL(subgenome_map, rng)
    T.to_mul(NetworkParser(net_filename).get_all_networks()[0])
   
    gene_trees = NetworkParser(gene_trees_filename).get_all_networks()
    
    for gene_tree in gene_trees:
        allele_funcs =  allele_map_set(gene_tree, subgenome_map)
        gene_tree.put_item("allele maps", allele_funcs)
        gene_tree.put_item("leaf descendants", gene_tree.leaf_descendants_all())
    
    return T.score(gene_trees)
