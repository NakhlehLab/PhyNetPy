"""
Author: Mark Kessler

Description: This file contains a model for computing Maximum Parsimony over data that includes allopolyploidization.

Last Stable Edit: 1/29/24
Included in version : 0.1.0
Approved to Release Date : N/A

"""

from cProfile import Profile
import copy
import io
import math
from Node import Node
from Graph import DAG
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



##NOTES##

"""
Clusters are represented a few different ways in this file.

1) frozen set of strings 
2) frozen set of Nodes
3) set of Nodes 
4) set of strings

Where the strings are the result of a get_name() call on each Node in the cluster.



The ILP (integer linear programming) Algorithm used to compute an MDC tree can be found here: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000501

The rest of this file is based on the algorithms described here: https://academic.oup.com/sysbio/article/71/3/706/6380964


Further documentation, code flow diagrams, and tutorials for this method can be found at https://phylogenomics.rice.edu
"""

########################
## EXCEPTION  HANDLER ##
########################

class InferAllopError(Exception):
    def __init__(self, message="Something went wrong during the execution of MP-Sugar"):
        self.message = message
        super().__init__(self.message)

########################
### Helper Functions ###
########################

def random_object(mylist : list, rng):
    """
    Select a random object from a list using a default rng object from numpy

    Args:
        mylist (list): any list of objects
        rng (_type_): numpy default rng

    Returns:
        _type_: _description_
    """
    rand_index = rng.integers(0, len(mylist))
    return mylist[rand_index]
    
def process_clusters(clusters : set) -> set:
    """
    Convert a set of tuples to a set of frozensets
    """
    return set([frozenset(tup) for tup in clusters])

def cluster_as_nameset(cluster : set)-> set:
    """
    Convert from a set of nodes to a set of strings (names)
    """
    return frozenset([node.get_name() for node in cluster])
    
def clusters_contains(cluster : set, set_of_clusters : set)->bool:
    """
    Check if a cluster is in a set of clusters by checking names
    (the objects will be different, but two clusters are equal if their names are equal)
    
    Args:
        cluster (set): a cluster
        set_of_clusters (set): a set of clusters

    Returns:
        bool: True, if cluster is an element of set_of_clusters. False if not.
    """
    names = cluster_as_nameset(cluster)
    
    for item in set_of_clusters:
        names_item = cluster_as_nameset(item)
        if names == names_item:
            return True
    return False

def cluster_partition(cluster:frozenset, processed:dict)->frozenset:
    """
    Given a cluster such as ('A', 'B', 'C'), if a cluster such as ('A', 'B') has already been processed,
    split the original cluster into subsets-- {('A', 'B'), ('C')}.

    Args:
        cluster (frozenset): _description_
        processed (dict): _description_

    Returns:
        frozenset: the partitioned cluster
    """
    
    editable_cluster = set(cluster) #to allow mutations
    
    new_cluster = set() # build up partioned cluster from scratch
    
    #Search already processed clusters for a cluster that is a subset of the original cluster
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
          
def generate_tree_from_clusters(tree_clusters:set)->DAG:
    """
    Given a set of clusters (given by the taxa labels, ie {('A','B','C'), ('B','C'), ('D','E','F'), ('D','E')}), 
    reconstruct the tree it represents.

    Returns:
        DAG: the MDC tree
    """
    net:DAG = DAG()
    nodes = []
    edges = []
    
    i = 2 #start with clusters of size 2
    j = 1
    
    is_root = False
    
    processed_clusters = {}
    root_children = []
    
    #remove clusters as theyre processed
    while len(tree_clusters) != 0:
        
        #get all clusters of a certain length
        clusters_len_i = [c for c in tree_clusters if len(c) == i]
        
        #last round, so the 2 clusters need to point to the root instead of two different parents!
        if len(tree_clusters) == len(clusters_len_i):
            is_root = True
            
        #Process smallest clusters first
        for cluster in clusters_len_i:
            
            cluster_parent = Node(name=f"Internal_{j}")
            nodes.append(cluster_parent)
            j+=1
            
            if is_root:
                root_children.append(cluster_parent)
            
            #detect previously encountered clusters in current cluster
            partitioned_cluster = cluster_partition(cluster, processed_clusters)
            
            for subtree in partitioned_cluster:
                if type(subtree) == frozenset: # Already found the cluster before
                    #connect previous cluster to this current cluster parent
                    processed_clusters[subtree].set_parent([cluster_parent])
                    edges.append((processed_clusters[subtree], cluster_parent))
                else: 
                    #subtree is simply a taxa label (a string), so create a new node
                    taxaNode = Node(name=subtree, parent_nodes=[cluster_parent])
                    nodes.append(taxaNode)
                    edges.append((taxaNode, cluster_parent))
                
            processed_clusters[cluster] = cluster_parent
            tree_clusters.remove(cluster)
        
        i+=1 #process next set of clusters


    #connect the 2 disjoint clusters together with the root
    root = Node(name="ROOT")
    for root_child in root_children:
        root_child.set_parent([root])   
        edges.append((root_child, root))
    
    nodes.append(root)
    
    #add all the accumulated nodes and edges
    net.add_edges(edges, as_list=True)
    net.add_nodes(nodes)
        
    return net

def partition_gene_trees(gene_map : dict[str, list[str]], num_retic : int = 1, rng = None) -> DAG:
    """
    TODO: rework this
    ex: 
    {"B": ["01bA"], "A": ["01aA"], "C": ["01cA"], "X": ["01xA", "01xB"], "Y": ["01yA", "01yB"], "Z": ["01zA", "01zB"]}

    Args:
        gene_map (dict[str, list[str]]): 

    Returns:
        DAG: _description_
    """
    retic_ct = 0
    
    std_leaves = [leaf for leaf in gene_map.keys() if len(gene_map[leaf]) == 1]
    std_leaves.sort()
    
    ploidy_leaves = [leaf for leaf in gene_map.keys() if len(gene_map[leaf]) != 1]
    simple_network : DAG = Yule(.1, len(std_leaves), rng = rng).generate_tree()


    
    #Change the names of the leaves to match 
    for leaf_pair in zip(simple_network.get_leaves(), std_leaves):
        net_leaf : Node = leaf_pair[0]
        #net_leaf.set_name(leaf_pair[1])
        simple_network.update_node_name(net_leaf, leaf_pair[1])
    

    #Partition the ploidy samples
    partitions : list[set[str]] = []
    for dummy in range(num_retic):
        partitions.append(set())
    
    #partition randomly
    for ploidy_leaf in ploidy_leaves:
        rand_set : set[str] = random_object(partitions, rng= rng)
        rand_set.add(ploidy_leaf)
    
    #Make clades out of each partition and hook them up to the simple network via a reticulation node.
    for partition in partitions:
        if len(partition) > 0:
            clade : DAG = Yule(.1, len(partition), rng = rng).generate_tree()
            
            
            for node in clade.nodes:
                #node.set_name(node.get_name() + "_c" + str(retic_ct))
                clade.update_node_name(node, node.get_name() + "_c" + str(retic_ct))  
            #print("-------")
            #clade.print_graph()
            #print("-------")
            
            for leaf_pair in zip(clade.get_leaves(), partition):
                net_leaf : Node = leaf_pair[0]
                #net_leaf.set_name(leaf_pair[1])
                clade.update_node_name(net_leaf, leaf_pair[1])
                
            #Connect to graph
            
            #Make new internal nodes for proper attachment
            new_node1 = Node(name = "#H" + str(retic_ct), is_reticulation = True)
            new_node2 = Node(name = "H" + str(retic_ct) + "_par1")
            new_node3 = Node(name = "H" + str(retic_ct) + "_par2")
            clade_root : Node = clade.root()[0]
            clade_root.add_parent(new_node1)
            new_node1.set_parent([new_node2, new_node3])
            clade.add_edges([new_node1, clade_root])
            clade.add_edges([new_node2, new_node1])
            clade.add_edges([new_node3, new_node1])

            clade.add_nodes([new_node1, new_node2, new_node3])
            
        
            #Get the connection points
            connecting_edges = simple_network.diff_subtree_edges(rng = rng)
            a : Node = connecting_edges[0][1]
            b : Node = connecting_edges[0][0]
            c : Node = connecting_edges[1][1]
            d : Node = connecting_edges[1][0]
            
            #Add the clade's nodes/edges now that the connection points have been selected
            simple_network.add_nodes(clade.nodes)
            simple_network.add_edges(clade.edges, as_list=True)
            
            #remove and add back edges
            simple_network.remove_edge([b, a])
       
            simple_network.remove_edge([d, c])
           
            simple_network.add_edges([[new_node2, a], [b, new_node2], [new_node3, c], [d, new_node3]], as_list=True)
            
   
            #handle the parent bindings
            a.remove_parent(b)
            a.add_parent(new_node2)
            c.remove_parent(d)
            c.add_parent(new_node3)
            new_node2.add_parent(b)
            new_node3.add_parent(d)

            retic_ct += 1

    return simple_network 

def get_other_copies(gene_tree_leaf : Node, gene_map : dict)->list[str]:
    """
    Given a gene tree leaf, get all other gene copy names for the Taxon for which gene_tree_leaf is a value.

    Args:
        gene_tree_leaf (Node): Leaf node of a gene tree
        gene_map (dict): a taxon map 

    Raises:
        InferAllopError: raised if gene_tree_leaf is not listed in the taxon map as an item of any value

    Returns:
        list[str]: List of other gene copy names
    """
    for copy_names in gene_map.values():
        if gene_tree_leaf.get_name() in copy_names:
            return copy_names
    
    raise InferAllopError(f"Leaf name '{gene_tree_leaf.get_name()}' not found in gene copy mapping")
          
def allele_map_set(g:DAG, gene_map : dict[str, list[str]]) ->list: 
    """
    Let a MUL tree, T', have taxa labels drawn from the set X (keys of gene_map input dict).
    Calculate all possible mappings from taxa labels of g (values of gene_map input dict) into X.
    
   
    Args:
        g (DAG): A gene tree

    Returns:
        list: a list of functions that map labels of g to labels of a MUL tree.
    """
    
    
    funcs = []
    funcs.append(AlleleMap())
    
    #Map each gene tree leaf to a mul tree leaf
    for gleaf in g.get_leaves():
        
        new_funcs = []
        
        #Only consider options for gleaf that correspond to the gene_map.
        #other_copies = get_other_copies(gleaf, self.mul, self.gene_map)
        other_copies = get_other_copies(gleaf, gene_map)
        for mul_leaf in other_copies:
            
            copy_funcs = copy.deepcopy(funcs)
            for func in copy_funcs:
                status = func.put(gleaf, mul_leaf)
                if status == 1: #map was successful, function is valid
                    new_funcs.append(func)
                
        funcs = new_funcs  
            
    
    return funcs      


 
    
#######################
### MDC START TREE  ###
#######################

class MDC_Tree:
    
    def __init__(self, dag_list: list) -> None:
        """
        Given a set of gene trees, construct the species tree using the MDC criterion.

        Args:
            dag_list (list): a list of gene trees (DAGs)
        """
        
        self.gene_trees = dag_list
        self.validate_gene_trees()
        self.cg_edges = set() # Set of compatibility graph edges
        self.cluster2index = {} # Map clusters to unique integers
        self.index2cluster = {} # the reverse map
        self.wv : list = [] # the function wv(A) where A is a cluster
    
        #Compute compatibility graph
        self.compatibility_graph() # self.wv set in call to this function
        
        indices = [str(num) for num in range(len(self.wv))]
        
        #Each decision variable corresponds to a cluster. value = 0 if not included in MDC tree, 1 if it is included.
        self.x : np.array = np.array(p.LpVariable.matrix("X", indices, lowBound=0, upBound=1, cat="Integer"))
        
        #Now we can solve the maximization problem using the pulp library
        self.mdc : DAG = self.solve()
        
    
    def compute_wv(self, cluster_set : set)->list:
        """
        Compute the number of extra lineages (per gene tree) for each cluster found in the set of gene trees
        Calling this function means that the index2cluster mapping has been filled out properly.
        Args:
            cluster_set (set): A set of clusters 

        Returns:
            list: An implicit map from clusters (defined by the list index) to their wv values defined by equation 3 in https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000501
        """
        m = 0
        wv_unadjusted = {cluster_as_nameset(cluster) : 0 for cluster in cluster_set}
        
        # Each cluster has |{gene trees}| alpha(A, T) values to add up
        for A in cluster_set:
            for T in self.gene_trees:
                #compute extra lineages contributed by cluster A in gene tree T
                extra = self.eq2(T, A)
                
                #Compute m value. defined as the maximum extra lineages over all clusters/gene trees
                if extra > m:
                    m = extra
                    
                wv_unadjusted[cluster_as_nameset(A)] += extra
        
        #adjust the wv map per eq3 for ILP reasons
        wv_map = {key: m - wv_unadjusted[key] + 1 for key in wv_unadjusted.keys()}  
        
        #convert to correct list
        wv = []
        for index in range(len(list(wv_map.keys()))):
            wv.append(wv_map[self.index2cluster(index)])
        
        return wv
        
    def eq2(T : DAG, A : set) -> int:
        """
        For a cluster, A, and a gene tree T, compute the added lineages based on equation 2 of https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000501
        xl = k - 1, where k is the number of maximal clades in T such that leaf_descendants(C) is a subset of A
        
        Args:
            T (DAG): A gene tree.
            A (set): A cluster.

        Raises:
            Exception: if the computation fails and no maximal clades are found

        Returns:
            int: number of extra lineages for A and T.
        """
        
        num_maximal_clades = 0
        cluster = set(cluster_as_nameset(A)) # visited set. remove elements as maximal clades are found
        cur = T.root()[0]
        q = deque()
        
        q.append(cur)
        
        #bfs- If you start at the root, you start at the largest possible leaf set and work down
        while len(q) != 0 or len(cluster) > 0:
            leaf_desc = T.leaf_descendants(cur)
            break_subtree = False
            
            #Check if the current clade in T is a maximal one, ie contains only elements from cluster A.
            if cluster.issuperset(set([leaf.get_name() for leaf in leaf_desc])):
                num_maximal_clades+=1
                break_subtree = True
                for item in leaf_desc:
                    cluster.remove(item.get_name())
            
            q.pop()
            
            if not break_subtree: # Don't search subtrees after finding a maximal clade
                for child in T.get_parents(cur):
                    q.appendleft(child) 
        
        if num_maximal_clades == 0:
            raise Exception("Something went wrong computing xl for a cluster and a gene tree")
        
        return num_maximal_clades - 1 #eq2

    def compatibility_graph(self):
        """
        Computes the compatability graph described in 
        https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000501
        
        Also is in charge of populating the mapping wv: A -> N, where A is the set of encountered clusters, and N is the set of natural numbers
        """
        clusters = set() #keep track of encountered clusters
        
        index = 0 #map clusters to variable indeces for the integer linear programming step
        
        for T in self.gene_trees:
            
            clusters_T = utils.get_all_clusters(T, T.root()[0])
            
            #convert each tuple element to a frozenset
            clusters_T = set([frozenset(tup) for tup in clusters_T])
            
            #process each cluster in the gene tree
            for A in clusters_T:
                if clusters_contains(A, clusters):
                    #This cluster has been seen before in another gene tree
                    self.wv[self.cluster2index[cluster_as_nameset(A)]] += 1
                else:
                    #This cluster hasn't been seen before
                    self.wv.append(1)
                    self.cluster2index[cluster_as_nameset(A)] = index
                    self.index2cluster[index] = cluster_as_nameset(A)
                    clusters.add(A)
                    index += 1
            
            #Add edge c1--c2 to compatability graph. Each cluster in T gets connected to every other cluster in T.
            for c1 in clusters_T:
                for c2 in clusters_T:
                    if c1 != c2:
                        self.cg_edges.add(frozenset([cluster_as_nameset(c1), cluster_as_nameset(c2)]))
        
        self.wv = self.compute_wv(clusters)
        
    def solve(self):
        """
        Uses the PULP integer linear programming library to solve the optimization problem.
        Calculates the clique that minimizes extra lineages by solving the problem described by formulation 4 in 
        https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000501
        
        Then, using the selected clusters, the mdc tree is reconstructed.
        """
        
        #init the model
        find_tree_model = p.LpProblem("MDC-Tree-Construction", p.LpMaximize)
        
        #add the objective function to the model (4)
        find_tree_model += p.lpSum(self.x * np.array(self.wv))
        
        # Add the constraints. If two clusters are not connected in the compatability graph, then only one can appear in the mdc tree
        # and thus x_i + x_j must not exceed 1.
        for i in range(len(self.x)):
            for j in range(i, len(self.x)):
                if frozenset([self.index2cluster[i], self.index2cluster[j]]) not in self.cg_edges and i!=j:
                    find_tree_model += p.lpSum([self.x[i], self.x[j]]) <= 1 , "Compatibility Graph Constraints" + f'<{i},{j}>'

        #solve the problem
        find_tree_model.solve()
        
        #collect all clusters with value 1, they are in the mdc tree
        tree_clusters = set()
        for v in find_tree_model.variables():
            try:
                if v.value() == 1: #if cluster is included in the optimal solution
                    tree_clusters.add(self.index2cluster[int(v.name.split("_")[1])])
            except:
                print("error couldnt find value")
        
        #reconstruct the tree and return  
        return generate_tree_from_clusters(tree_clusters)
    
    def get(self):
        return self.mdc
        
    def validate_gene_trees(self):
        """
        Validates that each gene tree has the same number of taxa and same taxa labels
        """
        taxa : list = self.gene_trees[0].get_leaves()
        taxa_names = [t.get_name() for t in taxa]
        for T in self.gene_trees[1:]:
            leaves = T.get_leaves()
            assert(len(leaves) == len(taxa))
            leaf_names = [l.get_name() for l in leaves]
            for name in leaf_names:
                assert(name in taxa_names)
        
        
class AlleleMap:
    """
    Data structure that holds a mapping from gene tree leaf names to mul tree leaf names.
    Internally handles the mechanism for making sure gene copies are not mapped to the same subgenome
    
    TODO: Add in constraint flexibility
    """
    
    def __init__(self) -> None:
        self.map = dict()
        self.disallowed = set() 

    def put(self, g_leaf : Node, mul_leaf : str):
        """
        Map a gene tree leaf to a MUL tree leaf

        Args:
            g_leaf (Node): a gene tree leaf
            mul_leaf (str): a MUL tree leaf label

        Returns:
            int: 0 if the mapping was unsuccessful, 1 if it was.
        """
        
        #If a gene tree leaf has already been mapped to a mul tree leaf, disallow the mapping
        if mul_leaf in self.disallowed:
            return 0
        
        #Otherwise, set the mapping and avoid mapping a gene tree leaf to this mul leaf in the future
        self.map[g_leaf.get_name()] = mul_leaf
        self.disallowed.add(mul_leaf)
        return 1
    
################
### MUL TREE ###
################

class MUL(DAG):
    
    def __init__(self, gene_map : dict, rng):
        
        self.net = None
        self.mul = None
        self.gene_map = gene_map
        self.rng = rng
                        
    
    def to_mul(self, net : DAG) -> DAG:
        """
        Creates a MUlti-Labeled Species Tree from a network

        Args:
            net (DAG): A Network

        Raises:
            InferAllopError: If the network is malformed or without correct ploidyness

        Returns:
            DAG: a MUL tree in a DAG obj
        """
        #Number of network leaves must match the number of gene map keys
        if len(net.get_leaves()) != len(self.gene_map.keys()):
            # print([node.get_name() for node in net.get_leaves()])
            # print(net.newick())
            raise InferAllopError(f"Input network has incorrect amount of leaves. Given : {len(net.get_leaves())} Expected : { len(self.gene_map.keys())}")
       
        copy_gene_map = copy.deepcopy(self.gene_map)
        mul_tree = DAG()
        
        #Create copies of all the nodes in net and keep track of the conversion
        network_2_mul : dict[Node, Node] = {node : Node(name = node.get_name()) for node in net.nodes}
        
        #Add all nodes and edges from net into the mul tree
        mul_tree.add_nodes(list(network_2_mul.values()))
        for edge in net.edges:
            mul_tree.add_edges([network_2_mul[edge[0]], network_2_mul[edge[1]]])
        
        
        #Bottom-Up traversal starting at leaves. Algorithm from STEP 1 in : https://doi.org/10.1371/journal.pgen.1002660
        
        #start at leaves, push onto queue when all children have been moved to the processed set
        processed : set[Node] = set()
        traversal_queue = deque(mul_tree.get_leaves())
        
        while len(traversal_queue) != 0:
            cur = traversal_queue.pop()
            
            original_pars = [node for node in mul_tree.get_parents(cur)]
            
            if mul_tree.in_degree(cur) == 2:
                #reticulation node. make a copy of subgraph
                #subtree = self.subtree_copy(mul_tree, cur)
                subtree = mul_tree.subtree_copy(cur)
                retic_pars = mul_tree.get_parents(cur)
                a = retic_pars[0]
                b = retic_pars[1]
                mul_tree.remove_edge([b, cur])
                mul_tree.add_edges([b, subtree.root()[0]])
                mul_tree.add_nodes(subtree.nodes)
                mul_tree.add_edges(subtree.edges, as_list=True)
                processed.add(subtree.root()[0])
            
            
            processed.add(cur)
            
            for par in original_pars:
                cop = set(mul_tree.get_children(par))
                if cop.issubset(processed):
                    traversal_queue.append(par)
        
        #Get rid of excess connection nodes
        mul_tree.prune_excess_nodes()
        
        #Rename tips based on gene mapping
        for leaf in mul_tree.get_leaves():
            new_name = copy_gene_map[leaf.get_name().split("_")[0]].pop()
            mul_tree.update_node_name(leaf, new_name)

        self.mul = mul_tree  
        return mul_tree             
        
    def extra_lineages(self, coal_event_map : dict, f:dict)-> int:
        """
        Computes the number of extra lineages in a mapping from a gene tree T, into a MUL tree, T'.

        Args:
            coal_event_map (dict): A mapping from edges in T' to a list of nodes of T that have been mapped into that edge. All nodes are by name, not object
            f (dict): An allele map. Used to ensure the correct number of lineages for each leaf branch. Some leaves may not be included in the map.

        Returns:
            int: number of extra lineages
        """
        edge_2_xl = {}
        root = self.mul.root()[0]
        self.xl_helper(root, edge_2_xl, coal_event_map, f)
        
        
        #root of g will be mapped to this edge, but doesn't count as a coal event
        root_xl = edge_2_xl[(root.get_name(), None)][0] - len(coal_event_map[(root.get_name(), None)]) - 1 
    
        extra_lin_total = 0
        mul_leaves = self.mul.get_leaves()
        
        
        for edge in self.mul.edges:
            
            # Only deal with edges that are not the root branch
            # Code follows Lemma 1 from -- https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000501
            if edge[1] is not None:
                extra_lin_total += edge_2_xl[(edge[1].get_name(), edge[0].get_name())][0] 
                extra_lin_total -= 1
                
                if edge[1] in mul_leaves: # branches connected to leaves will have no coalescent events
                    extra_lin_total -= 0
                elif edge[1] is not None:
                    extra_lin_total -= len(coal_event_map[(edge[1].get_name(), edge[0].get_name())]) 
    
        
        extra_lin_total += root_xl # Add in any extra lineages in the root branch. Tends to be 0. Is this necessary?
        
        return extra_lin_total
    
    def xl_helper(self, start_node : Node, edge_xl_map : dict, coal_map : dict, f:dict):
        """
        Modifies the edge_xl_map parameter. This is a recursive function that processes an edge in T'.
        The number of extra lineages exiting a branch is equal to the sum of its child lineages entering the branch, minus the number of
        coalescent events, minus 1.

        Args:
            start_node (Node): Node to calculate the number of lineages exiting/entering it
            edge_xl_map (dict): Mapping from edges to the number of lineages exiting/entering it
            coal_map (dict): Mapping from edges to gene tree internal nodes that have been mapped to said edge.
            f (dict): allele mapping (gene tree leaf names to mul tree leaf names -- types are strings)
        """
        if start_node in self.mul.get_leaves():
            par : Node  = self.mul.get_parents(start_node)[0] #start_node.get_parent() # should definitely not be None
            
            if start_node.get_name() in f.values():
                edge_xl_map[(start_node.get_name(), par.get_name())] = [1, 1] #bottom, top of branch will each be 1
            else:
                edge_xl_map[(start_node.get_name(), par.get_name())] = [0, 0] #There is no gene tree leaf mapped to this mul tree leaf :(
        else:
            if start_node == self.mul.root()[0]:
                par : Node = None
            else:
                par : Node  = self.mul.get_parents(start_node)[0] #start_node.get_parent()
                
            sum_of_child_tops = 0
            
            #Get each child's top lineage value 
            for child in self.mul.get_children(start_node):
                self.xl_helper(child, edge_xl_map, coal_map, f)
                sum_of_child_tops += edge_xl_map[(child.get_name(), start_node.get_name())][1]
            
            # Special case that start_node == root is not important since the top value is never used
            # Each node that is mapped to the start->par edge is a coal event that combines 2 lineages into 1
            if par is None:
                edge_xl_map[(start_node.get_name(), None)] = [sum_of_child_tops, sum_of_child_tops - len(coal_map[(start_node.get_name(), None)])]
            else:
                edge_xl_map[(start_node.get_name(), par.get_name())] = [sum_of_child_tops, sum_of_child_tops - len(coal_map[(start_node.get_name(), par.get_name())])]
            
    def gene_tree_map(self, g : DAG, leaf_map : dict, mrca_cache: dict[frozenset[Node], Node]) -> dict:
        """
        Maps a gene tree (T) into a MUL tree (T'), where each have taxa from the set X 

        Args:
            g (DAG): The gene tree
            leaf_map (dict): A function f: string -> string from gene tree leaf names to MUL tree leaf names. Aka an allele map.
        """
        # Let v′ = MRCA_T′(C_T(v)), and let u′ be the parent node of v′. Then, v is mapped to any point pv, excluding node u′, in the branch (u′, v′) in T′
        tnode_2_edgeloc = {} 
        edgeloc_2_tnode = {(edge[1].get_name(), edge[0].get_name()) : [] for edge in self.mul.edges}
        
        #Map each internal node of the gene tree into an edge of the MUL tree
        gene_tree_leaves = g.get_leaves()
        mul_root = self.mul.root()[0]
        
        leaf_desc_map = g.get_item("leaf descendants") #leaf_descendants_all()
        
        for node in g.nodes:
            if node not in gene_tree_leaves: #only map internal nodes
                c_t_ofv = leaf_desc_map[node] #g.leaf_descendants(node)

                cluster = frozenset([leaf_map[leaf.get_name()] for leaf in c_t_ofv])
                try:
                    v_prime : Node = mrca_cache[cluster]
                except:
                    v_prime : Node = self.mul.mrca(cluster) 
                    mrca_cache[cluster] = v_prime 
                
                edge = [v_prime.get_name()]
                
                if v_prime == mul_root:
                    u_prime = None
                    tnode_2_edgeloc[node.get_name()] = (v_prime.get_name(), None)
                    edge.append(None)
                else:
                    u_prime : Node = self.mul.get_parents(v_prime)[0] # v_prime.get_parent()
                    tnode_2_edgeloc[node.get_name()] = (v_prime.get_name(), u_prime.get_name())
                    edge.append(u_prime.get_name())
                
                #Add node to edge mapping
                if tuple(edge) in edgeloc_2_tnode.keys():
                    edgeloc_2_tnode[tuple(edge)].append(node.get_name())
                else:
                    edgeloc_2_tnode[tuple(edge)] = [node.get_name()]
        
        
        return edgeloc_2_tnode           
            
    def XL(self, g : DAG, mrca_cache : dict[frozenset[Node], Node]) -> int:
        """
        Computes the number of extra lineages in the map from the gene tree g, into MUL tree T.
        EQ1 from https://academic.oup.com/sysbio/article/71/3/706/6380964
        
        Args:
            g (DAG): A gene tree

        Returns:
            int: the minimum number of extra lineages over all possible allele maps
        """
    
        return min([self.XL_Allele(g, allele_map.map, mrca_cache) for allele_map in g.get_item("allele maps")])

    def XL_Allele(self, g : DAG, f : dict, mrca_cache : dict[frozenset[Node], Node]) -> int:
        """
        Compute the extra lineages given a MUL tree T, a gene tree, g, and an allele mapping f.

        Args:
            g (DAG): A gene tree
            f (dict): An allele map from leaves of g to leaves of T

        Returns:
            int: number of extra lineages
        """
        #map the gene tree into the MUL tree
        edge_2_nodes = self.gene_tree_map(g, f, mrca_cache)
        
        #compute the extra lineages
        return self.extra_lineages(edge_2_nodes, f)

    def score(self, gt_list : list[DAG])->int:
        """
        Compute the total score of a MUL tree given a list of gene trees. Right side of EQ 2 from EQ1 from https://academic.oup.com/sysbio/article/71/3/706/6380964

        Args:
            gt_list (list): a list of gene trees

        Returns:
            int: The MUL tree score 
        """
        # Sum the XL values over all gene trees
        mrca_cache = {}    
        gt_scores = [self.XL(gt, mrca_cache) for gt in gt_list]
        #print(gt_scores)
        return sum(gt_scores)


class InferMPAllop:
    
    def __init__(self, network : DAG, gene_map : dict[str, str], gene_trees : list[DAG], iter : int, rng) -> None:
        for gene_tree in gene_trees:
            gene_tree.put_item("allele maps", allele_map_set(gene_tree, gene_map))
            gene_tree.put_item("leaf descendants", gene_tree.leaf_descendants_all())
        self.mp_allop_model : Model = ModelFactory(MPAllopComponent(network, gene_map, gene_trees, rng)).build()
        self.iter = iter
        self.results = None
        
    def run(self) -> float:
        """
        Computes the network with the lowest (highest likelihood) parsimony score over 
        a set of gene trees

        Returns:
            DAG: max likelihood parsimony network
        """
        hc = HillClimbing(MPAllopProposalKernel(), None, None, self.iter, self.mp_allop_model)
        end_state : State = hc.run()
        self.results = hc.nets_2_scores
        return end_state.likelihood()
    

class MPAllopComponent(ModelComponent):
    
    def __init__(self, network : DAG, gene_map : dict[str, str], gene_trees : list[DAG], rng) -> None:
        super().__init__(set())
        self.network = network
        self.gene_map = gene_map
        self.gene_trees = gene_trees
        self.rng = rng
        
    
    def build(self, model : Model):
        
        mul_node : MULNode = MULNode(self.gene_map, self.rng)
        net_node : NetworkContainer = NetworkContainer(self.network)
        gene_trees_node : GeneTreesNode = GeneTreesNode(self.gene_trees)
        score_root_node : ParsimonyScore = ParsimonyScore()
        
        model.nodetypes["root"] = [score_root_node]
        model.nodetypes["internal"] = [mul_node]
        model.nodetypes["leaf"] = [gene_trees_node, net_node]
        model.network = self.network
        model.network_container = net_node
        
        gene_trees_node.join(score_root_node)
        mul_node.join(score_root_node)
        net_node.join(mul_node)
        
        
        
class NetworkContainer(StateNode):
    def __init__(self, network : DAG):
        super().__init__()
        self.network : DAG = network
    
    def update(self, new_net : DAG):
        self.network = new_net
        model_parents : list[CalculationNode] = self.get_model_parents()
        for model_parent in model_parents:
            model_parent.upstream()
        
    def get(self) -> DAG:
        return self.network


class MULNode(CalculationNode):
    
    def __init__(self, gene_map : dict, rng):
        super().__init__()
        self.multree : MUL = MUL(gene_map, rng)
        
    def calc(self):
        
        model_children = self.get_model_children()
        if len(model_children) == 1:
            if type(model_children[0]) is NetworkContainer:
                self.multree.to_mul(model_children[0].get())
            else:
                raise InferAllopError("Malformed MP Allop Model Graph. Expected MUL Node to have Network Container Child")
        
        return self.cache(self.multree)
    
    def sim(self):
        pass
    
    def update(self):
        self.upstream()
    
    def get(self):
        if self.dirty:
            return self.calc()
        else:
            return self.cached
    


class GeneTreesNode(StateNode):
    
    def __init__(self, gene_tree_list : list[DAG]):
        super().__init__()
        self.gene_trees : list[DAG] = gene_tree_list
    
    def update(self, new_tree : DAG, index : int):
        self.gene_trees[index] = new_tree
        model_parents : list[CalculationNode] = self.get_model_parents()
        if len(model_parents) == 1:
            model_parents[0].upstream()
        else: 
            raise InferAllopError("Expected Scoring node as a singular parent. Your model is malformed.")
        
    def get(self) -> list[DAG]:
        return self.gene_trees
    
class ParsimonyScore(CalculationNode):
    
    def __init__(self):
        super().__init__()
        
    def calc(self):
        model_children = self.get_model_children()
        
        if len(model_children) == 2:
            g_trees : list[DAG] = [child for child in model_children if type(child) == GeneTreesNode][0].get()
            mul : MUL = [child for child in model_children if type(child) == MULNode][0].get()
            
            if mul.gene_map is None:
                ##Invalid Network, return score of -inf so that this model is rejected
                raise Exception("An invalid network has been proposed somehow")
            else:  
                return self.cache(-1 * mul.score(g_trees))
        else:
            raise InferAllopError("Malformed Model. Parsimony Score function for MP ALLOP should only have 2 feeder nodes")
    
    def sim(self):
        pass
    
    def update(self):
        self.dirty = True
    
    def get(self):
        if self.dirty:
            return self.calc()
        else:
            return self.cached
        


#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################



def INFER_MP_ALLOP_BOOTSTRAP(start_network_file: str, gene_tree_file : str, subgenome_assign : dict[str, str], iter_ct : int = 500, seed = None):
    rng = np.random.default_rng(seed=seed)
    gene_tree_list : list = NetworkParser(gene_tree_file).get_all_networks()
    start_net = NetworkParser(start_network_file).get_all_networks()[0]
    leaf_map = subgenome_assign
    mp_model = InferMPAllop(start_net, leaf_map, gene_tree_list, iter_ct, rng=rng)
    likelihood = mp_model.run()
    return mp_model.results

def INFER_MP_ALLOP(gene_tree_file : str, taxon_assign : dict[str, str], iter_ct : int = 500, seed = None):
    
    rng = np.random.default_rng(seed=seed)
    start_net = partition_gene_trees(taxon_assign, rng = rng)
    gene_tree_list : list = NetworkParser(gene_tree_file).get_all_networks()
    leaf_map = taxon_assign
    mp_model = InferMPAllop(start_net, leaf_map, gene_tree_list, iter_ct, rng = rng)
    likelihood = mp_model.run()
    return mp_model.results

# def test2():
#     mul = MUL({"B": ["B"], "A": ["A"], "C": ["C"], "D":["D"], "X": ["X1", "X2"], "Y": ["Y1", "Y2"], "Z": ["Z1", "Z2"]}, np.random.default_rng(913))
#     start_net = NetworkParser("/Users/mak17/Documents/PhyloGenPy/PhyNetPy/src/PhyNetPy/Bayesian/mp_allop_start_net.nex").get_all_networks()[0]
#     mul.to_mul(start_net)
#     gt_list = NetworkParser("/Users/mak17/Documents/PhyloGenPy/PhyNetPy/src/PhyNetPy/Bayesian/mp_allop_tester.nex").get_all_networks()
#     print(score(mul, gt_list))
        

# def test3():
#     mul = MUL({"B": ["01bA"], "A": ["01aA"], "X": ["01xA", "01xB"], "Y": ["01yA", "01yB"], "Z": ["01zA", "01zB"]}, np.random.default_rng(913))
#     start_net = NetworkParser("/Users/mak17/Documents/PhyloGenPy/PhyNetPy/src/PhyNetPy/scenarioD_ideal.nex").get_all_networks()[0]
#     mul.to_mul(start_net)
#     gt_list = NetworkParser("/Users/mak17/Documents/PhyloGenPy/PhyNetPy/src/PhyNetPy/D10.nex").get_all_networks()
#     print(score(mul, gt_list))      
           
def test():
    
    # test_seed = random.randint(0,1000) #698
    # print(f"TESTER SEED : {test_seed}")
    scores = []
    for dummy in range(1):
        test_seed = random.randint(0,1000) #698 # 464 #32 913 #
        
        print(f"TESTER SEED : {test_seed}")
        try:
            # scores.append(
            #     MP_ALLOP('/Users/mak17/Documents/PhyloGenPy/PhyNetPy/src/PhyNetPy/J_nex_n1.nex',
            #         {'F': ['01fA'], 'T': ['01tA', '01tB'], 'W': ['01wA', '01wB'], 'B': ['01bA'], 'V': ['01vA', '01vB'], 'A': ['01aA'], 'U': ['01uA', '01uB'], 'C': ['01cA'], 'E': ['01eA'], 'X': ['01xA', '01xB'], 'Y': ['01yA', '01yB'], 'O': ['01oA'], 'Z': ['01zB', '01zA'], 'D': ['01dA']},
            #         seed= test_seed)
            # )
            # scores.append(MP_ALLOP(
            #         '/Users/mak17/Documents/PhyloGenPy/PhyNetPy/src/PhyNetPy/D10.nex', 
            #         {"B": ["01bA"], "A": ["01aA"], "X": ["01xA", "01xB"], "Y": ["01yA", "01yB"], "Z": ["01zA", "01zB"]},
            #         seed = test_seed
            #         ))
            MP_SUGAR('/Users/mak17/Documents/PhyloGenPy/PhyNetPy/src/PhyNetPy/J_nex_n1.nex',
                    {'F': ['01fA'], 'T': ['01tA', '01tB'], 'W': ['01wA', '01wB'], 'B': ['01bA'], 'V': ['01vA', '01vB'], 'A': ['01aA'], 'U': ['01uA', '01uB'], 'C': ['01cA'], 'E': ['01eA'], 'X': ['01xA', '01xB'], 'Y': ['01yA', '01yB'], 'O': ['01oA'], 'Z': ['01zB', '01zA'], 'D': ['01dA']},
                    seed= random.randint(0, 1000))
        except:
            print(test_seed)
            raise Exception("HALT")

    print(scores)

def test_start():
    scores = []
    for dummy in range(15):
        test_seed = random.randint(0,1000) #698 # 464 #32 913 #
        
        print(f"TESTER SEED : {test_seed}")
        try:
            run_dict = MP_SUGAR_WITH_STARTNET("/Users/mak17/Documents/PhyNetPy/src/bubble_J_sad.nex", '/Users/mak17/Documents/PhyNetPy/src/J_pruned_v2.nex',
                    {'U': ['01uA', '01uB'], 'T': ['01tA', '01tB'], 'B': ['01bA'], 'F': ['01fA'], 'C': ['01cA'], 'A': ['01aA'], 'D': ['01dA'], 'O': ['01oA']},
                    seed = test_seed)
            scores.append(
                run_dict
            )
            print(run_dict)
            # scores.append(MP_ALLOP(
            #         '/Users/mak17/Documents/PhyloGenPy/PhyNetPy/src/PhyNetPy/D10.nex', 
            #         {"B": ["01bA"], "A": ["01aA"], "X": ["01xA", "01xB"], "Y": ["01yA", "01yB"], "Z": ["01zA", "01zB"]},
            #         seed = test_seed
            #         ))
            # scores.append(MP_SUGAR('/Users/mak17/Documents/PhyNetPy/src/J_nex_n1.nex',
            #         {'F': ['01fA'], 'T': ['01tA', '01tB'], 'W': ['01wA', '01wB'], 'B': ['01bA'], 'V': ['01vA', '01vB'], 'A': ['01aA'], 'U': ['01uA', '01uB'], 'C': ['01cA'], 'E': ['01eA'], 'X': ['01xA', '01xB'], 'Y': ['01yA', '01yB'], 'O': ['01oA'], 'Z': ['01zB', '01zA'], 'D': ['01dA']},
            #         seed= random.randint(0, 1000)))
        except:
            print(test_seed)
            raise Exception("HALT")

    for mapping in scores:
        for net, score in mapping.items():
            print(score)
            print(net.newick())
    
    
def test_p():
    
    # test_seed = random.randint(0,1000) #698
    # print(f"TESTER SEED : {test_seed}")
    scores = [] 
    for dummy in range(1):
        test_seed = random.randint(0,1000) #698 # 464 #32 913 #868
        
        #print(f"TESTER SEED : {test_seed}")
        try:
            run_dict = INFER_MP_ALLOP('/Users/mak17/Documents/PhyNetPy/src/J_pruned_v2.nex',
                    {'U': ['01uA', '01uB'], 'T': ['01tA', '01tB'], 'B': ['01bA'], 'F': ['01fA'], 'C': ['01cA'], 'A': ['01aA'], 'D': ['01dA'], 'O': ['01oA']},
                    seed = test_seed)
            scores.append(
                run_dict
            )
            print(run_dict)
            # scores.append(MP_ALLOP(
            #         '/Users/mak17/Documents/PhyloGenPy/PhyNetPy/src/PhyNetPy/D10.nex', 
            #         {"B": ["01bA"], "A": ["01aA"], "X": ["01xA", "01xB"], "Y": ["01yA", "01yB"], "Z": ["01zA", "01zB"]},
            #         seed = test_seed
            #         ))
            # scores.append(MP_SUGAR('/Users/mak17/Documents/PhyNetPy/src/J_nex_n1.nex',
            #         {'F': ['01fA'], 'T': ['01tA', '01tB'], 'W': ['01wA', '01wB'], 'B': ['01bA'], 'V': ['01vA', '01vB'], 'A': ['01aA'], 'U': ['01uA', '01uB'], 'C': ['01cA'], 'E': ['01eA'], 'X': ['01xA', '01xB'], 'Y': ['01yA', '01yB'], 'O': ['01oA'], 'Z': ['01zB', '01zA'], 'D': ['01dA']},
            #         seed= random.randint(0, 1000)))
        except:
            print(test_seed)
            raise Exception("HALT")

    # for mapping in scores:
    #     for net, score in mapping.items():
    #         print(score)
    #         print(net.newick())   
    
    # print(
    #     MP_ALLOP(
    #         '/Users/mak17/Documents/PhyloGenPy/PhyNetPy/src/PhyNetPy/J_trees(7).nex', 
    #         {'W': ['01wB', '03wA', '01wA', '02wB', '02wA', '03wB'], 'B': ['02bA', '01bA', '03bA'], 'X': ['03xA', '02xB', '02xA', '01xB', '03xB', '01xA'], 'E': ['01eA', '03eA', '02eA'], 'Y': ['02yB', '01yB', '01yA', '03yA', '02yA', '03yB'], 'D': ['03dA', '01dA', '02dA'], 'U': ['03uA', '03uB', '02uA', '01uB', '02uB', '01uA'], 'T': ['03tA', '02tA', '02tB', '03tB', '01tA', '01tB'], 'F': ['03fA', '01fA', '02fA'], 'Z': ['02zA', '01zA', '03zB', '03zA', '01zB', '02zB'], 'A': ['01aA', '02aA', '03aA'], 'V': ['03vA', '02vA', '01vA', '01vB', '02vB', '03vB'], 'C': ['02cA', '03cA', '01cA'], 'O': ['01oA']},
    #         seed = random.randint(0,1000)
    #         )
    #     )
    
    # print(
    #     MP_ALLOP('/Users/mak17/Documents/PhyloGenPy/PhyNetPy/src/PhyNetPy/J_nex_n1.nex',
    #              {'F': ['01fA'], 'T': ['01tA', '01tB'], 'W': ['01wA', '01wB'], 'B': ['01bA'], 'V': ['01vA', '01vB'], 'A': ['01aA'], 'U': ['01uA', '01uB'], 'C': ['01cA'], 'E': ['01eA'], 'X': ['01xA', '01xB'], 'Y': ['01yA', '01yB'], 'O': ['01oA'], 'Z': ['01zB', '01zA'], 'D': ['01dA']},
    #              seed= random.randint(0, 1000))
        
    # )

    
# cp = Profile()  
# cp.enable()

test_p()
#test_start()

# cp.disable()
# cp.dump_stats("statsrun.txt")


# stream = open('/Users/mak17/Documents/PhyloGenPy/statsrun(2).txt', 'w')
# stats = pstats.Stats('/Users/mak17/Documents/PhyloGenPy/statsrun.txt', stream=stream)
# stats.sort_stats('cumtime')
# stats.print_stats(20)

def single_network_score(start_net_filename : str, gene_trees_filename : str, taxon_map : dict[str, list[str]]):
    
    rng = np.random.default_rng()
    T = MUL(taxon_map, rng)
    T.to_mul(NetworkParser(start_net_filename).get_all_networks()[0])
    #print(T.mul.newick())
    gene_trees = NetworkParser(gene_trees_filename).get_all_networks()
    for gene_tree in gene_trees:
            gene_tree.put_item("allele maps", allele_map_set(gene_tree, taxon_map))
            gene_tree.put_item("leaf descendants", gene_tree.leaf_descendants_all())
    
    return T.score(gene_trees)

#print(single_network_score("/Users/mak17/Documents/PhyNetPy/src/bubble_J.nex", "/Users/mak17/Documents/PhyNetPy/src/J_pruned_v2.nex", {'U': ['01uA', '01uB'], 'T': ['01tA', '01tB'], 'B': ['01bA'], 'F': ['01fA'], 'C': ['01cA'], 'A': ['01aA'], 'D': ['01dA'], 'O': ['01oA']}))