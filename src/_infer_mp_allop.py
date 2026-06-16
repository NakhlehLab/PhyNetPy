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

Last Edit: 3/4/26
Included in version : 1.0.0

Docs   - [ ]
Tests  - [x]
Design - [x]

"""
from __future__ import annotations
import copy
from collections import defaultdict
from typing import Any

import numpy as np

from .Network import Network, Node, Edge, MUL, NetworkError
from .IO import read_nexus
from .BirthDeath import Yule
from .GeneTrees import GeneTrees
from .ModelGraph import Model
from .ModelFactory import ModelFactory, ModelComponent
from .MetropolisHastings import HillClimbing, Infer_MP_Allop_Kernel
from .State import State
from .NetworkMoves import add_hybrid

try:
    from pulp import (
        LpVariable, LpProblem, LpMaximize, lpSum, LpStatus,
        value, PULP_CBC_CMD,
    )
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

from . import GraphUtils as utils

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
seed = np.random.randint(0, 1000)
RNG = np.random.default_rng(seed)

###########################
#### EXCEPTION HANDLER ####
###########################

class InferAllopError(Exception):
    """
    Exception class for Infer_MP_Allop related operation errors.
    """
    def __init__(self, message: str = "Error during the execution of "
                                       "Infer_MP_Allop_2.0") -> None:
        self.message = message
        super().__init__(self.message)

#########################
#### HELPER FUNCTIONS ###
#########################

def _nodes_to_improve(net: Network,
                      n: Node,
                      nti: dict[Node, int],
                      lti: dict[Node, int]) -> int:
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
        int: The amount of ploidy needed for n
    """
    if net.out_degree(n) == 0:
        nti[n] = lti[n]
    else:
        child_needs = [_nodes_to_improve(net, node, nti, lti)
                       for node in net.get_children(n)]
        nti[n] = min(child_needs)
    return nti[n]
            
def _ploidy_dif(goal: dict[Node, int], cur: dict[Node, int]) -> bool:
    """
    Checks if cur has achieved goal ploidy.

    Args:
        goal (dict[Node, int]): map of leaf nodes to their desired ploidy
        cur (dict[Node, int]): map of leaf nodes to their current ploidy

    Returns:
        bool: True if goal and cur are equivalent mappings.
    """
    for leaf in goal.keys():
        if cur[leaf] != goal[leaf]:
            return False
    return True
    
def _attach(net: Network,
            nti: dict[Node, int],
            goal_ploidy: dict[Node, int],
            rng: np.random.Generator = None) -> None:
    """
    Given a set of nodes that need to be edited as to increase ploidy, add a 
    hybrid edge in such a location as to increase the ploidy of a maximally 
    sized set of nodes that need to be improved.

    Args:
        net (Network): The Network.
        nti (dict[Node, int]): A map of nodes in net to the amount of ploidy
                               that the node needs to gain.
        goal_ploidy (dict[Node, int]): Target ploidy per leaf.
    """
    clusters: set[tuple[Node]] = utils.get_all_clusters(net,
                                                        include_trivial=True)
    # Deterministic ordering: largest clusters first, ties broken by the
    # sorted tuple of member labels. Without the secondary key the order of
    # equal-size clusters depends on set iteration order (and thus on
    # per-process string-hash randomization), making start networks
    # irreproducible across runs even with a fixed seed.
    sorted_clusters = sorted(
        clusters,
        key=lambda c: (-len(c), tuple(sorted(node.label for node in c))),
    )
    
    def amt_allowed(cluster: tuple[Node], lti: dict) -> int:
        return min([lti[node] for node in cluster])
    
    chosen_cluster = None
    ploidy_amt = 0
    
    for candidate in sorted_clusters:
        max_ploidy = amt_allowed(candidate, nti)
        if max_ploidy > 0:
            chosen_cluster = candidate
            ploidy_amt = max_ploidy
            break
    
    if chosen_cluster is None:
        raise InferAllopError("No clusters whose taxa needed more ploidy")
    
    top_of_cluster: Node = net.mrca(set(chosen_cluster))
    valid_edges = net.subgenome_ct_edges(downstream_node=top_of_cluster,
                                         delta=ploidy_amt)
    
    valid_edge_list = list(valid_edges.keys())
    if not valid_edge_list:
        raise InferAllopError("No valid edges for ploidy attachment")
    
    src: Edge = random_object(valid_edge_list, rng if rng is not None else RNG)
    if top_of_cluster != net.root():
        in_edges = list(net.in_edges(top_of_cluster))
        if in_edges:
            add_hybrid(net, src, in_edges[0])
        else:
            raise InferAllopError("top_of_cluster has no parent edges")
    else:
        raise NetworkError("Cannot attach hybrid at root")
    
         
def _resolve_ploidy(net: Network,
                    subgenomes: dict[str, list[str]],
                    rng: np.random.Generator = None) -> Network:
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
    cur_ploidy = {node: net.subgenome_count(node)
                  for node in net.get_leaves()}
    goal_ploidy = {node: len(subgenomes[node.label])
                   for node in net.get_leaves()}
    
    max_iterations = 25
    iteration = 0
   
    while not _ploidy_dif(goal_ploidy, cur_ploidy) and iteration < max_iterations:
        iteration += 1
        
        lti: dict[Node, int] = {}
        for node in net.get_leaves():
            lti[node] = goal_ploidy[node] - cur_ploidy[node]
        
        _attach(net, lti, goal_ploidy, rng=rng)
       
        cur_ploidy = {node: net.subgenome_count(node)
                      for node in net.get_leaves()}
    
    return net

def random_object(mylist: list[Any], rng: np.random.Generator) -> object:
    """
    Select a random object from a list using a numpy rng.

    Args:
        mylist (list[Any]): any list of objects
        rng (np.random.Generator): numpy default rng object

    Returns:
        object: randomly selected element from mylist
    """
    if not mylist:
        return None
    rand_index: int = rng.integers(0, len(mylist))
    return mylist[rand_index]
    
def cluster_as_name_set(cluster: set[Node]) -> frozenset[str]:
    """
    Convert cluster from a set of nodes to a set of strings (names).

    Args:
        cluster (set[Node]): One form of a "cluster"

    Returns:
        frozenset[str]: The set of node names in the cluster.
    """
    return frozenset([node.label for node in cluster])
    
def clusters_contains(cluster: set[Node],
                      set_of_clusters: set[set[Node]]) -> bool:
    """
    Check if a cluster is in a set of clusters by checking names.
    
    Args:
        cluster (set[Node]): a cluster
        set_of_clusters (set[set[Node]]): a set of clusters

    Returns:
        bool: True if cluster is an element of set_of_clusters.
    """
    names = cluster_as_name_set(cluster)
    for item in set_of_clusters:
        if cluster_as_name_set(item) == names:
            return True
    return False

def cluster_partition(cluster: frozenset[Node],
                      processed: dict[set[Node], Node]) -> frozenset:
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
    editable_cluster = set(cluster)
    new_cluster = set()
    
    for subcluster in processed.keys():
        if subcluster.issubset(editable_cluster):
            new_cluster.add(subcluster)
            for item in subcluster:
                editable_cluster.remove(item)
    
    for item in editable_cluster:
        new_cluster.add(item)
    
    return frozenset(new_cluster)
          
def generate_tree_from_clusters(tree_clusters: set[str]) -> Network:
    """
    Given a set of clusters (given by the taxa labels), reconstruct the tree
    it represents.

    Args:
        tree_clusters (set[str]): A set of tree clusters.
    Returns:
        Network: the MDC network.
    """
    net: Network = Network()
    root: Node = Node(name="ROOT")
    net.add_nodes(root)
    
    i = 2
    j = 1
    is_root = False
    processed_clusters: dict[set, Node] = {}
    root_children: list[Node] = []
    
    while len(tree_clusters) != 0:
        clusters_len_i = [c for c in tree_clusters if len(c) == i]
        
        if len(tree_clusters) == len(clusters_len_i):
            is_root = True
            
        for cluster in clusters_len_i:
            cluster_parent = Node(name=f"Internal_{j}")
            net.add_nodes(cluster_parent)
            j += 1
            
            if is_root:
                root_children.append(cluster_parent)
            
            partitioned_cluster = cluster_partition(cluster, processed_clusters)
            
            for subtree in partitioned_cluster:
                if type(subtree) == frozenset:
                    new_edge = Edge(cluster_parent, processed_clusters[subtree])
                    net.add_edges(new_edge)
                else:
                    taxa_node = Node(name=subtree)
                    net.add_nodes(taxa_node)
                    net.add_edges(Edge(cluster_parent, taxa_node))
                
            processed_clusters[cluster] = cluster_parent
            tree_clusters.remove(cluster)
        
        i += 1

    for root_child in root_children:   
        net.add_edges(Edge(root, root_child))
        
    return net

def partition_gene_trees(gene_map: dict[str, list[str]],
                         rng: np.random.Generator = None) -> Network:
    """
    Generate a starting network given a subgenome mapping.

    Args:
        gene_map (dict[str, list[str]]): Map from gene names to a list of 
                                         copy names.
        rng (np.random.Generator): A random generator instance.

    Returns:
        Network: A bootstrapped starting network with correct ploidy values.
    """
    if rng is None:
        rng = RNG
    yule_generator: Yule = Yule(.1, len(gene_map.keys()), rng=rng)
    simple_network: Network = yule_generator.generate_network()

    name_idx = 0
    names = list(gene_map.keys())
    for leaf in simple_network.get_leaves():
        simple_network.update_node_name(leaf, names[name_idx])
        name_idx += 1
            
    return _resolve_ploidy(simple_network, gene_map, rng=rng)

def get_other_copies(gene_tree_leaf: Node,
                     gene_map: dict[str, list[str]]) -> list[str]:
    """
    Given a gene tree leaf, get all other gene copy names for the
    Taxon for which gene_tree_leaf is a value.
    
    Raises:
        InferAllopError: raised if gene_tree_leaf is not listed in the taxon map 

    Args:
        gene_tree_leaf (Node): Leaf node of a gene tree
        gene_map (dict[str, list[str]]): a taxon map.
        
    Returns:
        list[str]: List of other gene copy names
    """
    for copy_names in gene_map.values():
        if gene_tree_leaf.label in copy_names:
            return copy_names
    
    raise InferAllopError(f"Leaf name '{gene_tree_leaf.label}' not found "
                          f"in gene copy mapping")
          
def allele_map_set(g: Network,
                   gene_map: dict[str, list[str]]) -> list[AlleleMap]: 
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
    funcs: list[AlleleMap] = [AlleleMap()]
    
    for gleaf in g.get_leaves():
        new_funcs = []
        other_copies = get_other_copies(gleaf, gene_map)
        
        for mul_leaf in other_copies:
            copy_funcs = copy.deepcopy(funcs)
            for func in copy_funcs:
                status = func.put(gleaf, mul_leaf)
                if status == 1:
                    new_funcs.append(func)
                
        funcs = new_funcs  
            
    return funcs      

def allele_map_set_ilp(g: Network,
                       gene_map: dict[str, list[str]],
                       max_solutions: int = 10) -> list[AlleleMap]:
    """
    Compute valid allele maps using Integer Linear Programming.
    Falls back to brute-force enumeration if PuLP is not installed.
    
    Args:
        g (Network): A gene tree
        gene_map (dict[str, list[str]]): A taxon map.
        max_solutions (int): Maximum number of solutions to find.

    Returns:
        list[AlleleMap]: a list of functions that map labels of g to 
                         labels of a MUL tree.
    """
    if not PULP_AVAILABLE:
        return allele_map_set(g, gene_map)
    
    gene_leaves = g.get_leaves()
    mul_leaves = set()
    for copies in gene_map.values():
        mul_leaves.update(copies)
    
    x = LpVariable.dicts("map",
                         ((i.label, j) for i in gene_leaves for j in mul_leaves),
                         cat='Binary')
    
    solutions = []
    
    while len(solutions) < max_solutions:
        prob = LpProblem("Allele_Mapping", LpMaximize)
        prob += lpSum(x[i.label, j] for i in gene_leaves for j in mul_leaves)
        
        for i in gene_leaves:
            prob += lpSum(x[i.label, j] for j in mul_leaves) == 1
        
        for j in mul_leaves:
            prob += lpSum(x[i.label, j] for i in gene_leaves) <= 1
        
        for i in gene_leaves:
            valid_mul_leaves = get_other_copies(i, gene_map)
            for j in mul_leaves:
                if j not in valid_mul_leaves:
                    prob += x[i.label, j] == 0
        
        for prev_solution in solutions:
            diff_constraint = []
            for i in gene_leaves:
                for j in mul_leaves:
                    if prev_solution.map.get(i.label) == j:
                        diff_constraint.append(1 - x[i.label, j])
                    else:
                        diff_constraint.append(x[i.label, j])
            prob += lpSum(diff_constraint) >= 1
        
        prob.solve(PULP_CBC_CMD(msg=False))
        
        if LpStatus[prob.status] != 'Optimal':
            break
        
        allele_map = AlleleMap()
        for i in gene_leaves:
            for j in mul_leaves:
                if value(x[i.label, j]) == 1:
                    allele_map.put(i, j)
        
        solutions.append(allele_map)
    
    if not solutions:
        raise InferAllopError("No valid allele map found")
    
    return solutions

##############################
### MUL TREE & Allele Maps ###
##############################

class Allop_MUL(MUL):
    """
    A Standard MUL tree with methods for calculating the maximum parsimony
    score for Infer_MP_Allop_2.0.
    """
    # Per-score caches (set in ``XL``); avoid recomputing the MUL leaf set
    # and root inside the hot parsimony traversal.
    _mul_leaf_set: set = set()
    _mul_root = None

    def extra_lineages(self,
                       coal_event_map: dict[tuple, str],
                       f: AlleleMap) -> int:
        """
        Computes the number of extra lineages in a mapping from a gene tree T, 
        into a MUL tree, T'.

        Args:
            coal_event_map (dict[tuple[str, str], str]): A mapping from edges in 
                                                         T' to a list of nodes 
                                                         of T that have been 
                                                         mapped into that edge.
            f (AlleleMap): An allele map.

        Returns:
            int: number of extra lineages
        """
        edge_2_xl = {}
        root = self._mul_root
        
        self.xl_helper(root, edge_2_xl, coal_event_map, f)
        
        root_xl = (edge_2_xl[(None, root.label)][0]
                   - len(coal_event_map[(None, root.label)]) - 1)
    
        extra_lin_total = 0
        mul_leaves = self._mul_leaf_set
        
        for edge in self.mul.E():
            if edge.src is not None:
                extra_lin_total += edge_2_xl[edge.to_names()][0] 
                extra_lin_total -= 1
                
                if edge.dest in mul_leaves: 
                    extra_lin_total -= 0
                elif edge.dest is not None:
                    extra_lin_total -= len(coal_event_map[edge.to_names()]) 
    
        extra_lin_total += root_xl 
        
        return extra_lin_total
    
    def xl_helper(self,
                  start_node: Node,
                  edge_xl_map: dict,
                  coal_map: dict[tuple[str, str], str],
                  f: AlleleMap) -> None:
        """
        Recursive helper that computes lineage counts for extra lineage
        calculation.

        Args:
            start_node (Node): Node to process
            edge_xl_map (dict): Mapping from edges to lineage counts [bottom, top]
            coal_map (dict): Mapping from edges to gene tree internal node names
            f (AlleleMap): allele mapping
        """
        fmap: dict[str, str] = f.map
        
        if start_node in self._mul_leaf_set:
            par: Node = self.mul.get_parents(start_node)[0]
            
            if start_node.label in fmap.values():
                edge_xl_map[(par.label, start_node.label)] = [1, 1] 
            else:
                edge_xl_map[(par.label, start_node.label)] = [0, 0] 
        else:
            if start_node == self._mul_root:
                par = None
            else:
                par = self.mul.get_parents(start_node)[0]
                
            sum_of_child_tops = 0
            
            for child in self.mul.get_children(start_node):
                self.xl_helper(child, edge_xl_map, coal_map, f)
                e = (start_node.label, child.label)
                sum_of_child_tops += edge_xl_map[e][1]
            
            par_name = par.label if par is not None else None
                
            bottom = sum_of_child_tops
            coal_events = len(coal_map[(par_name, start_node.label)])
            top = sum_of_child_tops - coal_events
            
            edge_xl_map[(par_name, start_node.label)] = [bottom, top]
            
    def gene_tree_map(self,
                      g: Network,
                      leaf_map: AlleleMap,
                      mrca_cache: dict[frozenset[Node], Node]
                      ) -> dict[tuple, list[str]]:
        """
        Maps a gene tree (T) into a MUL tree (T').

        Args:
            g (Network): The gene tree
            leaf_map (AlleleMap): Allele map from gene tree to MUL tree.
            mrca_cache (dict): A cache for MRCA computations.
        Returns:
            dict[tuple, list[str]]: Mapping from edges in T' to gene tree nodes.
        """
        edgeloc_2_tnode = defaultdict(list)
        
        gene_tree_leaves = g.get_leaves()
        mul_root = self._mul_root
        
        leaf_desc_map = g.get_item("leaf descendants")
        
        for node in g.V():
            if node not in gene_tree_leaves: 
                c_t_ofv = leaf_desc_map[node]

                cluster_names = [leaf_map.map[leaf.label]
                                 for leaf in c_t_ofv]
                cluster = frozenset(cluster_names)
                
                try:
                    v_prime: Node = mrca_cache[cluster]
                except KeyError:
                    v_prime: Node = self.mul.mrca(cluster)
                    mrca_cache[cluster] = v_prime
                
                if v_prime == mul_root:
                    e = (None, v_prime.label)
                else:
                    u_prime: Node = self.mul.get_parents(v_prime)[0]
                    e = (u_prime.label, v_prime.label)
                
                edgeloc_2_tnode[e].append(node.label)
        
        return edgeloc_2_tnode
            
    def XL(self, g: Network, mrca_cache: dict[frozenset[Node], Node]) -> int:
        """
        Computes the minimum number of extra lineages over all possible
        allele maps for gene tree g mapped into this MUL tree.

        Args:
            g (Network): A gene tree
            mrca_cache (dict): A cache for MRCA computations.
        Returns:
            int: the minimum number of extra lineages (>= 0)
        """
        allele_maps = g.get_item("allele maps")
        if not allele_maps:
            return 0
        # Cache the MUL leaf set and root once per gene tree. Both are
        # invariant for the lifetime of the current MUL tree and were
        # previously recomputed inside xl_helper for every visited node
        # (the dominant runtime hotspot for larger scenarios).
        self._mul_leaf_set = set(self.mul.get_leaves())
        self._mul_root = self.mul.root()
        xl = [max(0, self.XL_Allele(g, allele_map, mrca_cache))
              for allele_map in allele_maps]
        return min(xl)

    def XL_Allele(self,
                  g: Network,
                  f: AlleleMap,
                  mrca_cache: dict[frozenset[Node], Node]) -> int:
        """
        Compute the extra lineages for a specific allele mapping.

        Args:
            g (Network): A gene tree
            f (AlleleMap): An allele map from leaves of g to leaves of T
            mrca_cache (dict): A cache for MRCA computations.
        Returns:
            int: number of extra lineages
        """
        edge_2_nodes = self.gene_tree_map(g, f, mrca_cache)
        return self.extra_lineages(edge_2_nodes, f)

    def score(self, gt_list: list[Network]) -> int:
        """
        Compute the total parsimony score over a list of gene trees.

        Args:
            gt_list (list[Network]): a list of gene trees

        Returns:
            int: The MUL tree score 
        """
        mrca_cache = {}    
        gt_scores = [self.XL(gt, mrca_cache) for gt in gt_list]
        return sum(gt_scores)

class AlleleMap:
    """
    Data structure that holds a mapping from gene tree leaf names to MUL tree 
    leaf names. Internally handles the mechanism for making sure gene copies 
    are not mapped to the same subgenome.
    """
    
    def __init__(self) -> None:
        self.map: dict[str, str] = dict()
        self.disallowed: set[str] = set()

    def put(self, g_leaf: Node, mul_leaf: str) -> int:
        """
        Map a gene tree leaf to a MUL tree leaf.

        Args:
            g_leaf (Node): a gene tree leaf
            mul_leaf (str): a MUL tree leaf label

        Returns:
            int: 0 if the mapping was unsuccessful, 1 if it was.
        """
        if mul_leaf in self.disallowed:
            return 0
        
        self.map[g_leaf.label] = mul_leaf
        self.disallowed.add(mul_leaf)
        return 1

######################################
#### LIKELIHOOD SCORER FOR MODEL  ####
######################################

class MPAllopScorer:
    """
    Callable scorer that integrates with Model.set_likelihood_calculator().
    Computes the MP Allop parsimony score (negated for maximization).
    """
    
    def __init__(self,
                 gene_map: dict[str, list[str]],
                 gene_trees: list[Network],
                 rng: np.random.Generator) -> None:
        self.mul = Allop_MUL(gene_map, rng)
        self.gene_trees = gene_trees
    
    def __call__(self, model: Model) -> float:
        """
        Compute the parsimony score. Returns negated score since hill climbing
        maximizes but parsimony should be minimized.

        If the network is malformed (e.g. wrong leaf count after a move),
        returns ``-inf`` so the hill climber will reject the proposal.
        """
        net = model.network
        if net is None:
            return float("-inf")
        
        try:
            self.mul.to_mul(net)
        except Exception:
            return float("-inf")
        
        if self.mul.gene_map is None:
            return float("-inf")
        
        try:
            return -1 * self.mul.score(self.gene_trees)
        except Exception:
            return float("-inf")

#####################
#### MODEL BUILD ####
#####################

class MPAllopComponent(ModelComponent):
    """
    Model Component that sets up the model for Infer MP Allop 2.0.
    Registers the parsimony scorer on the model and attaches the network.
    """
    
    def __init__(self,
                 network: Network,
                 gene_map: dict[str, list[str]],
                 gene_trees: list[Network],
                 rng: np.random.Generator) -> None:
        super().__init__(set())
        self.network = network
        self.gene_map = gene_map
        self.gene_trees = gene_trees
        self.rng = rng
    
    def build(self, model: Model) -> None:
        """
        Attaches the MP allop component to the model.

        Args:
            model (Model): A model, under construction.
        """
        model.network = self.network
        scorer = MPAllopScorer(self.gene_map, self.gene_trees, self.rng)
        model.set_likelihood_calculator(scorer)


class InferMPAllop:
    """
    Class that sets up and runs the Infer_MP_Allop_2.0 inference method.
    """
    
    def __init__(self,
                 network: Network,
                 gene_map: dict[str, list[str]],
                 gene_trees: list[Network],
                 iter_ct: int,
                 rng: np.random.Generator) -> None:
        """
        Initialize the parameters for this method call.

        Args:
            network (Network): A starting network.
            gene_map (dict[str, list[str]]): A subgenome mapping.
            gene_trees (list[Network]): A set of gene trees.
            iter_ct (int): Number of iterations to run the chain.
            rng (np.random.Generator): random number generator.
        """
        for gene_tree in gene_trees:
            allele_funcs = allele_map_set_ilp(gene_tree, gene_map)
            gene_tree.put_item("allele maps", allele_funcs)
            
            leaf_descendants = gene_tree.leaf_descendants_all()
            gene_tree.put_item("leaf descendants", leaf_descendants)

        mp_allop_comp = MPAllopComponent(network, gene_map, gene_trees, rng)
        model_fac: ModelFactory = ModelFactory(mp_allop_comp)
        self.mp_allop_model: Model = model_fac.build()
        self.iter_ct = iter_ct
        self.results: dict = {}
        
    def run(self) -> float:
        """
        Computes the network with the lowest parsimony score over the set
        of given gene trees.

        Returns:
            float: parsimony score of the most likely Network
        """
        hc = HillClimbing(Infer_MP_Allop_Kernel(),
                          num_iter=self.iter_ct,
                          model=self.mp_allop_model,
                          enhanced_stop=True)
        
        end_state: State = hc.run()
        
        self.results = hc.nets_2_scores
        return end_state.likelihood()


#################
#### METHODS ####
#################

def INFER_MP_ALLOP_BOOTSTRAP(start_network_file: str,
                             gene_tree_file: str,
                             subgenome_assign: dict[str, list[str]],
                             iter_ct: int = 500,
                             seed: int = None) -> dict[Network, float]:
    """
    Infer_MP_Allop_2.0, with a provided starting network.
    
    Given a set of gene trees, a subgenome assignment, and a starting network,
    infer the network that minimizes the parsimony score.

    Args:
        start_network_file (str): A nexus file that contains a starting network
        gene_tree_file (str): A nexus file that contains the gene trees
        subgenome_assign (dict[str, list[str]]): A mapping from genomes to the
                                                  set of genes in them.
        iter_ct (int, optional): Number of iterations. Defaults to 500.
        seed (int, optional): Random seed value. Defaults to None.

    Returns:
        dict[Network, float]: A map from Networks to their parsimony scores
    """
    rng = np.random.default_rng(seed if seed is not None else
                                 np.random.randint(0, 10000))
    
    gene_tree_list: list[Network] = read_nexus(gene_tree_file)
    start_net = read_nexus(start_network_file)[0]

    mp_model = InferMPAllop(start_net,
                            subgenome_assign,
                            gene_tree_list,
                            iter_ct,
                            rng=rng)
    
    mp_model.run()
    return mp_model.results

def INFER_MP_ALLOP(gene_tree_file: str,
                   subgenome_assign: dict[str, list[str]] = None,
                   iter_ct: int = 500,
                   seed: int = None) -> dict[Network, float]:
    """
    Infer_MP_Allop_2.0.
    
    Given a set of gene trees, and a subgenome assignment, infer the network
    that minimizes the parsimony score.

    Args:
        gene_tree_file (str): A nexus file containing the gene trees
        subgenome_assign (dict[str, list[str]]): a map from genomes to genes
        iter_ct (int, optional): Number of iterations. Defaults to 500.
        seed (int, optional): Random seed value. Defaults to None.

    Returns:
        dict[Network, float]: a mapping from Networks to their parsimony scores.
    """
    rng = np.random.default_rng(seed if seed is not None else
                                 np.random.randint(0, 10000))
    
    gene_tree_list: list[Network] = read_nexus(gene_tree_file)
    
    if subgenome_assign is None:
        gts = GeneTrees(gene_tree_list)
        subgenome_assign = gts.mp_allop_map()
    
    start_net = partition_gene_trees(subgenome_assign, rng=rng)
    
    mp_model = InferMPAllop(start_net,
                            subgenome_assign,
                            gene_tree_list,
                            iter_ct,
                            rng=rng)
    mp_model.run()
    return mp_model.results

def ALLOP_SCORE(net_filename: str,
                gene_trees_filename: str,
                subgenome_map: dict[str, list[str]]) -> int: 
    """
    Given a network, a set of gene trees, and a subgenome mapping, compute
    the parsimony score over all the gene trees.

    Args:
        net_filename (str): Network nexus file
        gene_trees_filename (str): Gene trees nexus file
        subgenome_map (dict[str, list[str]]): subgenome-to-genes mapping

    Returns:
        int: parsimony score
    """
    rng = np.random.default_rng()
    
    T = Allop_MUL(subgenome_map, rng)
    T.to_mul(read_nexus(net_filename)[0])
   
    gene_trees = read_nexus(gene_trees_filename)
    
    for gene_tree in gene_trees:
        allele_funcs = allele_map_set_ilp(gene_tree, subgenome_map)
        gene_tree.put_item("allele maps", allele_funcs)
        gene_tree.put_item("leaf descendants", gene_tree.leaf_descendants_all())
    
    return T.score(gene_trees)
