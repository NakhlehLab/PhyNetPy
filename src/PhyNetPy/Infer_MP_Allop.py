"""
Author: Mark Kessler

Description: This file contains the method Infer_MP_Allop_2.0, which 
is a maximum parsimony approach to inferring phylogenetic networks that contain
allopolyploid, polyploid, and autopolyploid species, given a set of gene trees.

Last Edit: 4/2/24
Included in version : 1.0.0
Approved for Release: NO.

"""
from __future__ import annotations
from cProfile import Profile
import copy
import io
import math
from Network import Network, Edge, Node, MUL
import GraphUtils as utils
from collections import defaultdict, deque
import pulp as p
import numpy as np
from NetworkParser import NetworkParser
from BirthDeath import Yule
from typing import Callable
from ModelGraph import *
from ModelFactory import *
from MetropolisHastings import *
from State import *
import pstats, io


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
    def __init__(self, message = "Error during the execution of \
                                  Infer_MP_Allop_2.0"):
        self.message = message
        super().__init__(self.message)

#########################
#### HELPER FUNCTIONS ###
#########################

def random_object(mylist : list, rng : np.random.Generator) -> object:
    """
    Select a random object from a list using a default rng object from numpy

    Args:
        mylist (list): any list of objects
        rng (np.random.Generator): numpy default rng object

    Returns:
        object : could be anything that is contained in mylist
    """
    rand_index = rng.integers(0, len(mylist))
    return mylist[rand_index]
    
def cluster_as_name_set(cluster : set[Node]) -> set[str]:
    """
    Convert cluster from a set of nodes to a set of strings (names).

    Args:
        cluster (set[Node]): One form of a "cluster"

    Returns:
        set[str]: The set of node names in the cluster.
    """
    return frozenset([node.get_name() for node in cluster])
    
def clusters_contains(cluster : set, set_of_clusters : set) -> bool:
    """
    Check if a cluster is in a set of clusters by checking names
    (the objects can be different, but two clusters are equal if their 
    names are equal).
    
    Args:
        cluster (set): a cluster
        set_of_clusters (set): a set of clusters

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
                      processed : dict[set, Node]) -> frozenset:
    """
    Given a cluster such as ('A', 'B', 'C'), if a cluster such as ('A', 'B') 
    has already been processed, split the original cluster into subsets -- 
    {('A', 'B'), ('C')}.

    Args:
        cluster (frozenset): A cluster.
        processed (dict): A mapping from clusters to the Node obj that is the 
                          root of that cluster.

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
                    new_edge = Edge(cluster_parent, processed_clusters[subtree])
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
                         num_retic : int = 1, 
                         rng = None) -> Network:
    """
    TODO : SOMETHING AINT RIGHT as of 4.2.24, pls investigate
    ex: 
    {"B": ["01bA"], "A": ["01aA"], "C": ["01cA"], "X": ["01xA", "01xB"], 
    "Y": ["01yA", "01yB"], "Z": ["01zA", "01zB"]}

    Args:
        gene_map (dict[str, list[str]]): 

    Returns:
        Network : _description_
    """
    retic_ct = 0
    
    std_leaves = [leaf for leaf in gene_map.keys() if len(gene_map[leaf]) == 1]
    std_leaves.sort()
    
    ploidy_leaves = [leaf for leaf in gene_map.keys() \
                    if len(gene_map[leaf]) != 1]
    
    yule_generator : Yule = Yule(.1, len(std_leaves), rng = rng)
    simple_network : Network = yule_generator.generate_network()

    #Change the names of the leaves to match 
    for leaf_pair in zip(simple_network.get_leaves(), std_leaves):
        net_leaf : Node = leaf_pair[0]
        simple_network.update_node_name(net_leaf, leaf_pair[1])
    
    #Partition the ploidy samples
    partitions : list[set[str]] = []
    for dummy in range(num_retic):
        partitions.append(set())
    
    #partition randomly
    for ploidy_leaf in ploidy_leaves:
        rand_set : set[str] = random_object(partitions, rng = rng)
        rand_set.add(ploidy_leaf)
    
    # Make clades out of each partition and hook them up to the simple network 
    # via a reticulation node.
    for partition in partitions:
        yule_generator = Yule(.1, len(partition), rng = rng)
        if len(partition) > 0:
            
            clade : Network = yule_generator.generate_network()
            
            for node in clade.get_nodes():
                new_name : str = node.get_name() + "_c" + str(retic_ct)
                clade.update_node_name(node, new_name)  
            
            for leaf_pair in zip(clade.get_leaves(), partition):
                net_leaf : Node = leaf_pair[0]
                clade.update_node_name(net_leaf, leaf_pair[1])
                
            #Connect to graph
            
            #Make new internal nodes for proper attachment
            node1 = Node(name = "#H" + str(retic_ct), is_reticulation = True)
            node2 = Node(name = "H" + str(retic_ct) + "_par1")
            node3 = Node(name = "H" + str(retic_ct) + "_par2")
            clade_root : Node = clade.root()[0]

            clade.add_nodes([node1, node2, node3])
            clade.add_edges(Edge(node1, clade_root))
            clade.add_edges(Edge(node2, node1))
            clade.add_edges(Edge(node3, node1))

            #Get the connection points
            connecting_edges = simple_network.diff_subtree_edges(rng = rng)
            a : Node = connecting_edges[0].dest
            b : Node = connecting_edges[0].src
            c : Node = connecting_edges[1].dest
            d : Node = connecting_edges[1].src
            
            # Add the clade's nodes/edges 
            # now that the connection points have been selected
            simple_network.add_nodes(clade.get_nodes())
            simple_network.add_edges(clade.get_edges())
            
            # Remove and add back edges
            simple_network.remove_edge([b, a])
            simple_network.remove_edge([d, c])
           
            simple_network.add_edges([Edge(node2, a), Edge(b, node2), 
                                      Edge(node3, c), Edge(d, node3)])
            
            retic_ct += 1

    return simple_network 

def get_other_copies(gene_tree_leaf : Node, gene_map : dict) -> list[str]:
    """
    Given a gene tree leaf, get all other gene copy names for the
    Taxon for which gene_tree_leaf is a value.

    Args:
        gene_tree_leaf (Node): Leaf node of a gene tree
        gene_map (dict): a taxon map 

    Raises:
        InferAllopError: raised if gene_tree_leaf is not listed in the taxon map 
                         as an item of any value

    Returns:
        list[str]: List of other gene copy names
    """
    for copy_names in gene_map.values():
        if gene_tree_leaf.get_name() in copy_names:
            return copy_names
    
    raise InferAllopError(f"Leaf name '{gene_tree_leaf.get_name()}' not found \
                            in gene copy mapping")
          
def allele_map_set(g : Network, gene_map : dict[str, list[str]]) -> list: 
    """
    Let a MUL tree, T', have taxa labels drawn from the set X 
    (keys of gene_map input dict). Calculate all possible mappings from taxa 
    labels of g (values of gene_map input dict) into X.
    
    Args:
        g (Network): A gene tree

    Returns:
        list: a list of functions that map labels of g to labels of a MUL tree.
    """
    
    funcs = []
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


########################
#### MDC START TREE  ###        TODO: Approve this class
########################

class MDC_Tree:
    
    def __init__(self, gene_trees : list[Network]) -> None:
        """
        Given a set of gene trees, construct the species tree using the 
        MDC criterion.

        Args:
            net_list (list[Network]): a list of gene trees 
        """
        
        self.gene_trees = gene_trees
        self.validate_gene_trees()
        self.cg_edges : set = set() # Set of compatibility graph edges
        self.cluster2index = {} # Map clusters to unique integers
        self.index2cluster = {} # the reverse map
        self.wv : list = [] # the function wv(A) where A is a cluster
    
        #Compute compatibility graph
        self.compatibility_graph() # self.wv set in call to this function
        
        indices = [str(num) for num in range(len(self.wv))]
        
        # Each decision variable corresponds to a cluster. value = 0 if not 
        # included in MDC tree, 1 if it is included.
        self.x : np.array = np.array(p.LpVariable.matrix("X", 
                                                         indices, 
                                                         lowBound=0, 
                                                         upBound=1, 
                                                         cat="Integer"))
        
        #Now we can solve the maximization problem using the pulp library
        self.mdc : Network = self.solve()
        
    
    def compute_wv(self, cluster_set : set) -> list:
        """
        Compute the number of extra lineages (per gene tree) for each cluster 
        found in the set of gene trees. Calling this function means that the 
        index2cluster mapping has been filled out properly.
        
        Args:
            cluster_set (set): A set of clusters 

        Returns:
            list: An implicit map from clusters (defined by the list index) to 
                  their wv values defined by equation 3 in (1)
                  
        """
        
        m = 0
        
        #{cluster_as_name_set(cluster) : 0 for cluster in cluster_set}
        wv_unadjusted = defaultdict(int) 
        
        # Each cluster has |{gene trees}| alpha(A, T) values to add up
        for A in cluster_set:
            for T in self.gene_trees:
                #compute extra lineages contributed by cluster A in gene tree T
                extra = self.eq2(T, A)
                
                # Compute m value. defined as the maximum extra lineages 
                # over all clusters/gene trees
                if extra > m:
                    m = extra
                    
                wv_unadjusted[cluster_as_name_set(A)] += extra
        
        #adjust the wv map per eq3 for ILP reasons
        wv_map = {key : m - wv_unadjusted[key] + 1 
                  for key in wv_unadjusted.keys()}  
        
        #convert to correct list
        wv = []
        for index in range(len(list(wv_map.keys()))):
            wv.append(wv_map[self.index2cluster(index)])
        
        return wv
        
    def eq2(T : Network, A : set) -> int:
        """
        For a cluster, A, and a gene tree T, compute the added lineages based on
        equation 2 of (1):
        xl = k - 1, where k is the number of maximal clades in T such that 
        leaf_descendants(C) is a subset of A
        
        Args:
            T (Network): A gene tree.
            A (set): A cluster.

        Raises:
            Exception: if the computation fails and no maximal clades are found

        Returns:
            int: number of extra lineages for A and T.
        """
        
        num_maximal_clades = 0
        
        # visited set. remove elements as maximal clades are found
        cluster = set(cluster_as_name_set(A)) 
        cur = T.root()[0]
        q = deque()
        
        q.append(cur)
        
        # bfs - If you start at the root, you start at the largest possible leaf 
        # set and work down
        while len(q) != 0 or len(cluster) > 0:
            leaf_desc : set[Node] = T.leaf_descendants(cur)
            break_subtree = False
            
            # Check if the current clade in T is a maximal one, ie contains only 
            # elements from cluster A.
            if cluster.issuperset(set([leaf.get_name() for leaf in leaf_desc])):
                num_maximal_clades += 1
                break_subtree = True
                for n in leaf_desc:
                    cluster.remove(n.get_name())
            
            q.pop()
            
            # Don't search subtrees after finding a maximal clade
            if not break_subtree: 
                for child in T.get_parents(cur):
                    q.appendleft(child) 
        
        if num_maximal_clades == 0:
            raise Exception("Something went wrong computing xl for a cluster \
                             and a gene tree")
        
        return num_maximal_clades - 1 

    def compatibility_graph(self) -> None:
        """
        Computes the compatability graph described in (1)
        
        Also is in charge of populating the mapping wv: A -> N, where A is the 
        set of encountered clusters, and N is the set of natural numbers.
        """
        #keep track of encountered clusters
        clusters = set() 
        
        #map clusters to variable indeces for the ilp step
        index = 0 
        
        for T in self.gene_trees:
            
            clusters_T = utils.get_all_clusters(T, T.root()[0])
            
            #convert each tuple element to a frozenset
            clusters_T = set([frozenset(tup) for tup in clusters_T])
            
            #process each cluster in the gene tree
            for A in clusters_T:
                if clusters_contains(A, clusters):
                    #This cluster has been seen before in another gene tree
                    self.wv[self.cluster2index[cluster_as_name_set(A)]] += 1
                else:
                    #This cluster hasn't been seen before
                    self.wv.append(1)
                    self.cluster2index[cluster_as_name_set(A)] = index
                    self.index2cluster[index] = cluster_as_name_set(A)
                    clusters.add(A)
                    index += 1
            
            # Add edge c1--c2 to compatability graph. 
            # Each cluster in T gets connected to every other cluster in T.
            for c1 in clusters_T:
                for c2 in clusters_T:
                    if c1 != c2:
                        c1_names = cluster_as_name_set(c1)
                        c2_names = cluster_as_name_set(c2)
                        self.cg_edges.add(frozenset([c1_names, c2_names]))
        
        self.wv = self.compute_wv(clusters)
        
    def solve(self) -> Network:
        """
        Uses the PULP integer linear programming library to solve the 
        optimization problem. Calculates the clique that minimizes extra 
        lineages by solving the problem described by formulation 4 in (1)
        
        Then, using the selected clusters, the mdc tree is reconstructed.
        """
        
        #init the model
        find_tree_model = p.LpProblem("MDC-Tree-Construction", p.LpMaximize)
        
        #add the objective function to the model (4)
        find_tree_model += p.lpSum(self.x * np.array(self.wv))
        
        # Add the constraints. If two clusters are not connected in the 
        # compatability graph, then only one can appear in the mdc tree
        # and thus x_i + x_j must not exceed 1.
        for i in range(len(self.x)):
            for j in range(i, len(self.x)):
                e = frozenset([self.index2cluster[i], self.index2cluster[j]])
                if e not in self.cg_edges and i != j:
                    find_tree_model += p.lpSum([self.x[i], self.x[j]]) <= 1 \
                        , "Compatibility Graph Constraints" + f'<{i},{j}>'

        #solve the problem
        find_tree_model.solve()
        
        #collect all clusters with value 1, they are in the mdc tree
        tree_clusters = set()
        for v in find_tree_model.variables():
            try:
                #if cluster is included in the optimal solution
                if v.value() == 1: 
                    cluster = self.index2cluster[int(v.name.split("_")[1])]
                    tree_clusters.add(cluster)
            except:
                print("error couldnt find value")
        
        #reconstruct the tree and return  
        return generate_tree_from_clusters(tree_clusters)
    
    def get(self) -> Network:
        return self.mdc
        
    def validate_gene_trees(self) -> Network:
        """
        Validates that each gene tree has the same number of taxa and 
        same taxa labels
        """
        taxa : list[Node] = self.gene_trees[0].get_leaves()
        taxa_names = [t.get_name() for t in taxa]
        for T in self.gene_trees[1:]:
            leaves = T.get_leaves()
            assert(len(leaves) == len(taxa))
            leaf_names = [l.get_name() for l in leaves]
            for name in leaf_names:
                assert(name in taxa_names)
        
##############################
### MUL TREE & Allele Maps ###
##############################

class Allop_MUL(MUL):
    """
    A Standard MUL tree, but contains relevant methods for calculating the 
    maximum parsimony score for Infer_MP_Allop_2.0
    """
    def extra_lineages(self, coal_event_map : dict[tuple, str], 
                       f : AlleleMap) -> int:
        """
        Computes the number of extra lineages in a mapping from a gene tree T, 
        into a MUL tree, T'.

        Args:
            coal_event_map (dict[tuple, str]): A mapping from edges in T' to a 
                                              list of nodes of T that have been 
                                              mapped into that edge. All nodes 
                                              are by name, not object. Edges 
                                              here are represented by tuples
                                              with node names.
            
            f (AlleleMap): An allele map. Used to ensure the correct number of 
                           lineages for each leaf branch. Some leaves may not be 
                           included in the map.

        Returns:
            int: number of extra lineages
        """
        # Map edges to the number of lineages present in the branch
        edge_2_xl = {}
        
        root = self.mul.root()[0]
        
        # Populate the edge to lineage mapping by calling the xl_helper function
        # at the root (propagates through the whole network)
        self.xl_helper(root, edge_2_xl, coal_event_map, f)
        
        #START COUNTING EXTRA LINEAGES 
        
        # Root of g will be mapped to this edge, 
        # but doesn't count as a coal event
        root_xl = edge_2_xl[(None, root.get_name())][0] \
                  - len(coal_event_map[(None, root.get_name())]) - 1 
    
        extra_lin_total = 0
        mul_leaves = self.mul.get_leaves()
        
        # Process each edge in the mul tree and tabulate 
        # the extra lineage count.
        for edge in self.mul.get_edges():

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
    
    def xl_helper(self, start_node : Node, edge_xl_map : dict,
                  coal_map : dict, f : AlleleMap) -> None:
        """
        Modifies the edge_xl_map parameter. This is a recursive function that 
        processes an edge in T'. The number of extra lineages exiting a branch 
        is equal to the sum of its child lineages entering the branch, minus 
        the number of coalescent events, minus 1.

        Args:
            start_node (Node): Node to calculate the number of 
                               lineages exiting/entering it
            edge_xl_map (dict): Mapping from edges to the number of 
                                lineages exiting/entering it
            coal_map (dict): Mapping from edges to gene tree internal nodes.
            f (AlleleMap): allele mapping (gene tree leaf names to mul tree leaf 
                           names -- types are strings).
        """
        # Pull the mapping from the AlleleMap object
        fmap : dict[str, str] = f.map
        
        if start_node in self.mul.get_leaves():
            # should definitely not be None, and there should only be 1.
            par : Node  = self.mul.get_parents(start_node)[0]
            
            if start_node.get_name() in fmap.values():
                #bottom, top of branch will each be 1
                edge_xl_map[(par.get_name(), start_node.get_name())] = [1, 1] 
            else:
                #There is no gene tree leaf mapped to this mul tree leaf :(
                edge_xl_map[(par.get_name(), start_node.get_name())] = [0, 0] 
        else:
            if start_node == self.mul.root()[0]:
                par : Node = None
            else:
                par : Node  = self.mul.get_parents(start_node)[0] 
                
            sum_of_child_tops = 0
            
            #Get each child's top lineage value 
            for child in self.mul.get_children(start_node):
                self.xl_helper(child, edge_xl_map, coal_map, f)
                e = (start_node.get_name(), child.get_name())
                sum_of_child_tops += edge_xl_map[e][1]
            
            # Special case that start_node == root is not important since the 
            # top value is never used. Each node that is mapped to the 
            # par->start edge is a coal event that combines 2 lineages into 1
            par_name : str = None
            if par is not None:
                par_name = par.get_name()
                
            bottom = sum_of_child_tops
            coal_events = len(coal_map[(par_name, start_node.get_name())])
            top = sum_of_child_tops - coal_events
            
            edge_xl_map[(par_name, start_node.get_name())] = [bottom, top]
            
    def gene_tree_map(self, g : Network, leaf_map : dict, 
                      mrca_cache: dict[frozenset[Node], Node]) -> dict:
        """
        Maps a gene tree (T) into a MUL tree (T'), where each have taxa 
        from the set X.

        Args:
            g (Network): The gene tree
            leaf_map (dict): A function f: string -> string from gene tree leaf
                             names to MUL tree leaf names. Aka an allele map.
        """
        # Let v′ = MRCA_T′(C_T(v)), and let u′ be the parent node of v′. 
        # Then, v is mapped to any point pv, excluding node u′, 
        # in the branch (u′, v′) in T′
        tnode_2_edgeloc = {} 
        
        # Map of edges to the set of nodes that get mapped within that edge.
        edgeloc_2_tnode = defaultdict(list)
        
        # Map each internal node of the gene tree into an edge of the MUL tree
        gene_tree_leaves = g.get_leaves()
        mul_root = self.mul.root()[0]
        
        #grab results of the prior leaf_descendants_all() call
        leaf_desc_map = g.get_item("leaf descendants")
        
        for node in g.get_nodes():
            #only map internal nodes
            if node not in gene_tree_leaves: 
                c_t_ofv = leaf_desc_map[node] 

                cluster_names = [leaf_map[leaf.get_name()] for leaf in c_t_ofv]
                cluster = frozenset(cluster_names)
                
                # Calculate the most recent common ancestor of all the nodes in
                # the cluster in the mul tree.
                try:
                    v_prime : Node = mrca_cache[cluster]
                except:
                    v_prime : Node = self.mul.mrca(cluster) 
                    mrca_cache[cluster] = v_prime 
                
                edge = [None, v_prime.get_name()]
                
                # Calculate v_prime's parent
                if v_prime == mul_root:
                    u_prime = None
                    e = (None, v_prime.get_name())
                    tnode_2_edgeloc[node.get_name()] = e
                else:
                    u_prime : Node = self.mul.get_parents(v_prime)[0] 
                    edge[0] = u_prime.get_name()
                    e = (u_prime.get_name(), v_prime.get_name())
                    tnode_2_edgeloc[node.get_name()] = e
                
                
                #Add node to edge mapping
                if tuple(edge) in edgeloc_2_tnode.keys():
                    edgeloc_2_tnode[tuple(edge)].append(node.get_name())
                else:
                    edgeloc_2_tnode[tuple(edge)] = [node.get_name()]
        
        return edgeloc_2_tnode           
            
    def XL(self, g : Network, mrca_cache : dict[frozenset[Node], Node]) -> int:
        """
        Computes the number of extra lineages in the map from the gene tree g, 
        into MUL tree T.
        EQ1 from (3)
        
        Args:
            g (Network): A gene tree

        Returns:
            int: the minimum number of extra lineages over all 
                 possible allele maps
        """
    
        return min([self.XL_Allele(g, allele_map, mrca_cache) 
                    for allele_map in g.get_item("allele maps")])

    def XL_Allele(self, g : Network, f : dict, 
                  mrca_cache : dict[frozenset[Node], Node]) -> int:
        """
        Compute the extra lineages given a MUL tree T, a gene tree, g, 
        and an allele mapping f.

        Args:
            g (Network): A gene tree
            f (dict): An allele map from leaves of g to leaves of T

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
    
    TODO: Add in constraint flexibility
    """
    
    def __init__(self) -> None:
        """
        Initialize an empty allele mapping.
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
        self.map[g_leaf.get_name()] = mul_leaf
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
                 gene_map : dict[str, str], 
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
        self.results = None
        
    def run(self) -> float:
        """
        Computes the network with the lowest (highest likelihood) parsimony 
        score over the set of given gene trees.

        Returns:
            Network : Network that maximizes the maximum parsimony score over
                      the set of gene trees.
        """
        hc = HillClimbing(Infer_MP_Allop_Kernel(), None, None, 
                          self.iter, self.mp_allop_model)
        end_state : State = hc.run()
        self.results = hc.nets_2_scores
        return end_state.likelihood()
    
class MPAllopComponent(ModelComponent):
    """
    Model Component that sets up the model for Infer MP Allop 2.0.
    """
    
    def __init__(self, 
                 network : Network, 
                 gene_map : dict[str, str],
                 gene_trees : list[Network], 
                 rng : np.random.Generator) -> None:
        """
        Initialize the MP Allop model component

        Args:
            network (Network): A starting network.
            gene_map (dict[str, str]): Subgenome mapping.
            gene_trees (list[Network]): Set of gene trees.
            rng (np.random.Generator): Random number generator.
        """
        
        super().__init__(set())
        
        self.network : Network = network
        self.gene_map : dict[str, str] = gene_map
        self.gene_trees : list[Network] = gene_trees
        self.rng : np.random.Generator = rng
        
    
    def build(self, model : Model) -> None:
        """
        Attaches the MP allop component to the main model.

        Args:
            model (Model): A model, under construction.
        """
        # Add mul tree 
        mul_node : MULNode = MULNode(self.gene_map, self.rng)
        
        # Add start network
        net_node : NetworkContainer = NetworkContainer(self.network)
        
        # Add gene trees
        gene_trees_node : GeneTreesNode = GeneTreesNode(self.gene_trees)
        
        # Add scoring function
        score_root_node : ParsimonyScore = ParsimonyScore()
        
        # Book-keeping
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
        """
        super().__init__()
        self.network : Network = network
    
    def update(self, new_net : Network) -> None:
        """
        Replace the network that is currently stored with a new one, and update 
        any model nodes that rely on this data.

        Args:
            new_net (Network): The new Network obj to be stored.
        """
        self.network = new_net
        model_parents : list[CalculationNode] = self.get_model_parents()
        for model_parent in model_parents:
            model_parent.upstream()
        
    def get(self) -> Network:
        """
        Grab the network obj stored in this container.

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
                 gene_map : dict[str, str],
                 rng : np.random.Generator) -> None:
        """
        Initialize a MUL tree and store it in this model node.

        Args:
            gene_map (dict[str, str]): A subgenome mapping.
            rng (np.random.Generator): A random number generator.
        """
        super().__init__()
        self.multree : Allop_MUL = Allop_MUL(gene_map, rng)
        
    def calc(self) -> Network:
        """
        Regenerate the mul tree, if the underlying network has been edited.

        Raises:
            InferAllopError: If the MUL node is not correctly linked to a
                             NetworkContainer.

        Returns:
            Network: The newly generated MUL tree.
        """
        
        model_children = self.get_model_children()
        
        if len(model_children) == 1:
            if type(model_children[0]) is NetworkContainer:
                self.multree.to_mul(model_children[0].get())
            else:
                raise InferAllopError("Malformed MP Allop Model Graph. \
                                       Expected MUL Node to have Network \
                                       Container Child")
        
        return self.cache(self.multree)
    
    def sim(self) -> None:
        """
        No implementation for a MUL node for simulations.
        """
        pass
    
    def update(self):
        self.upstream()
    
    def get(self) -> Network:
        """
        Get the MUL tree if the underlying network has not been changed 
        recently, otherwise generate a new one, cache it 
        (via the calc function), and return it.

        Returns:
            Network: A MUL tree that is up to date with the underlying network.
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

        """
        self.gene_trees[index] = new_tree
        model_parents : list[CalculationNode] = self.get_model_parents()
        for par in model_parents:
            par.upstream()
        
    def get(self) -> list[Network]:
        """
        Grab the list of gene trees.

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
        No parameters required.
        """
        super().__init__()
        
    def calc(self) -> float:
        """
        The MP Allop score for a network over a set of gene trees.

        Raises:
            InferAllopError: If there is an error in the model formation or if 
                             something went wrong computing the score. 

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
        """
        pass
    
    def update(self) -> None:
        """
        Flag this node as being in need of recalculation.
        """
        self.dirty = True
    
    def get(self) -> float:
        """
        If nothing has changed in the model, then return the cached score.
        If something has changed, then recompute the parsimony score.

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

def INFER_MP_ALLOP_BOOTSTRAP(start_network_file: str, 
                             gene_tree_file : str,
                             subgenome_assign : dict[str, str], 
                             iter_ct : int = 500, 
                             seed = None) -> dict:
    """
    Infer_MP_Allop_2.0, with a provided starting network.
    
    Given a set of gene trees, a subgenome assignment, and a starting network,
    infer the network that minimizes the parsimony score.

    Args:
        start_network_file (str): A nexus file that contains a starting network
        gene_tree_file (str): A nexus file that contains the gene trees
        subgenome_assign (dict[str, str]): A mapping from genomes to the set
                                           of genes in them.
        iter_ct (int, optional): Number of iterations to run
                                 the inference chain. Defaults to 500.
        seed (_type_, optional): Random seed value. Defaults to None, in which
                                 case a random integer will be used.

    Returns:
        dict: A map from a small number of Networks to their parsimony scores
    """
    
    #init rng object
    rng = np.random.default_rng(seed = seed)
    
    #Parse gene trees and starting network from files
    gt_parser = NetworkParser(gene_tree_file)
    gene_tree_list : list[Network] = gt_parser.get_all_networks()
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
                   subgenome_assign : dict[str, str], 
                   iter_ct : int = 500,
                   seed : int = None):
    """
    Infer_MP_Allop_2.0.
    
    Given a set of gene trees, and a subgenome assignment, infer the network
    that minimizes the parsimony score.

    Args:
        gene_tree_file (str): A nexus file containing the gene trees
        subgenome_assign (dict[str, str]): a map from genomes to their genes
        iter_ct (int, optional): Number of iterations to run the inference 
                                 chain. Defaults to 500.
        seed (int, optional): Random number generator seed value. 
                              Defaults to None, in which case a random value 
                              will be chosen.

    Returns:
        dict: a mapping from a small number of Networks to their parsimony 
        scores.
    """
    rng = np.random.default_rng(seed = seed)
    start_net = partition_gene_trees(subgenome_assign, rng = rng)
    gene_tree_list : list = NetworkParser(gene_tree_file).get_all_networks()
    
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



#################
#### TESTING ####
################# 
    
def test():
    
    # test_seed = random.randint(0,1000) #698
    # print(f"TESTER SEED : {test_seed}")
    scores = [] 
    for dummy in range(1):
        test_seed = random.randint(0,1000) #698 # 464 #32 913 #868
        
        #print(f"TESTER SEED : {test_seed}")
        try:
            run_dict = INFER_MP_ALLOP(
                    '/Users/mak17/Documents/PhyNetPy/src/J_pruned_v2.nex',
                    {'U': ['01uA', '01uB'], 'T': ['01tA', '01tB'], 
                     'B': ['01bA'], 'F': ['01fA'], 'C': ['01cA'], 'A': ['01aA'],
                     'D': ['01dA'], 'O': ['01oA']},
                    seed = test_seed)
            
            scores.append(
                run_dict
            )
            print(run_dict)
            
            # scores.append(
            # '/Users/mak17/Documents/PhyloGenPy/PhyNetPy/src/PhyNetPy/D10.nex', 
            #         {"B": ["01bA"], "A": ["01aA"], "X": ["01xA", "01xB"],
            #          "Y": ["01yA", "01yB"], "Z": ["01zA", "01zB"]},
            #         seed = test_seed
            #         ))
            
            #          '/Users/mak17/Documents/PhyNetPy/src/J_nex_n1.nex',
            #         {'F': ['01fA'], 'T': ['01tA', '01tB'], 
            #          'W': ['01wA', '01wB'], 'B': ['01bA'], 
            #          'V': ['01vA', '01vB'], 'A': ['01aA'], 
            #          'U': ['01uA', '01uB'], 'C': ['01cA'], 
            #          'E': ['01eA'], 'X': ['01xA', '01xB'], 
            #          'Y': ['01yA', '01yB'], 'O': ['01oA'], 
            #          'Z': ['01zB', '01zA'], 'D': ['01dA']},
            #         
        except:
            print(test_seed)
            raise Exception("HALT")


    
# cp = Profile()  
# cp.enable()

test()
#test_start()

# cp.disable()
# cp.dump_stats("statsrun.txt")


# stream = open('/Users/mak17/Documents/PhyloGenPy/statsrun(2).txt', 'w')
# stats = pstats.Stats('/Users/mak17/Documents/PhyloGenPy/statsrun.txt', 
#                       stream=stream)
# stats.sort_stats('cumtime')
# stats.print_stats(20)



#print(ALLOP_SCORE("/Users/mak17/Documents/PhyNetPy/src/bubble_J.nex",
# "/Users/mak17/Documents/PhyNetPy/src/J_pruned_v2.nex", 
# {'U': ['01uA', '01uB'], 'T': ['01tA', '01tB'], 'B': ['01bA'], 'F': ['01fA'], 
# C': ['01cA'], 'A': ['01aA'], 'D': ['01dA'], 'O': ['01oA']}))