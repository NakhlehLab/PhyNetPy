import copy
import math
from PhyNetPy.Bayesian import Node
from PhyNetPy.Bayesian.Graph import DAG
from collections import deque
import pulp as p
import numpy as np

from NetworkBuilder2 import NetworkBuilder2 
from Node import Node

########################
### Helper Functions ###
########################


def process_clusters(clusters):
    """
    Convert from a set of tuples to a set of frozensets
    """
    return set([frozenset(tup) for tup in clusters])

def cluster_as_nameset(cluster):
    """
    Convert from a set of nodes to a set of strings (names)
    """
    return frozenset([node.get_name() for node in cluster])
    
def clusters_contains(cluster, set_of_clusters)->bool:
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
        _type_: _description_
    """
    
    editable_cluster = set(cluster)
    
    new_cluster = set()
    
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
    net.addEdges(edges)
    net.addNodes(nodes)
        
    return net

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
        self.cg_edges = set()
        self.cluster2index = {}
        self.index2cluster = {}
        self.wv : list = []
    
        #Compute compatibility graph
        self.compatibility_graph() # self.wv set in call to this function
        
        indices = [str(num) for num in range(len(self.wv))]
        
        #Each decision variable corresponds to a cluster. value = 0 if not included in MDC tree, 1 if it is included.
        self.x : np.array = np.array(p.LpVariable.matrix("X", indices, lowBound=0, upBound=1, cat="Integer"))
        
        #Now we can solve the maximization problem using the pulp library
        self.mdc : DAG = self.solve()
        
        
    
    def compatibility_graph(self):
        """
        Computes the compatability graph described in 
        https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000501
        
        Also is in charge of populating the mapping wv: A -> N, where A is the set of encountered clusters, and N is the set of natural numbers
        """
        clusters = set() #keep track of encountered clusters
        
        index = 0 #map clusters to variable indeces for the integer linear programming step
        
        for T in self.gene_trees:
            clusters_T = T.get_all_clusters(T.findRoot()[0])
            
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
        taxa : list = self.gene_trees[0].getLeafs()
        taxa_names = [t.get_name() for t in taxa]
        for T in self.gene_trees[1:]:
            leaves = T.getLeafs()
            assert(len(leaves) == len(taxa))
            leaf_names = [l.get_name() for l in leaves]
            for name in leaf_names:
                assert(name in taxa_names)
         
################
### MUL TREE ###
################

class MUL(DAG):
    
    def __init__(self):
        self.net = None
        self.mul = None
        
    def toMulTree(self, network:DAG)->DAG:
        """
        From a Phylogenetic Network, create a Multilabeled Species Tree (MUL). In paper, this function is U(phi).
        This is accomplished with Depth First Search.
        
        Each time a node is searched, its parent is a different node: thus, making a copy and setting the unique parent accomplishes
        the subtree duplication involved in creating a mul tree.

        Args:
            network (DAG): Phylogenetic Network

        Returns:
            DAG: A TREE that is the MUL representation of the input network
        """
        
        visited : dict = dict()  # Map from nodes to the number of times they've been visited
    
        mul_tree = DAG()
        root : Node = network.findRoot()[0]
        
        # stack for dfs. appendleft / popleft for LIFO 
        q = deque()
        
        q.appendleft(root)
        
        visited.add(root)

        while len(q) != 0:
            cur = q.popleft()

            for neighbor in network.findDirectPredecessors(cur):
                q.append(neighbor)
                if neighbor not in visited.keys():
                    visited[neighbor] = 1
                    mul_tree.addNodes(neighbor)
                    mul_tree.addEdges((neighbor, cur))
                    
                else:
                    #node is in a subtree of a reticulation.
                    new_node = neighbor.make_copy(cur, visited[neighbor])
                    visited[neighbor] += 1
                    mul_tree.addNodes(new_node)
                    mul_tree.addEdges((new_node, cur))
        
        
        self.mul = mul_tree
        return mul_tree
        
    
    
    def mul_tree_score(self)->int:
        pass        
            
            
def XL(T:MUL, g:DAG, F: set):
    xls = []
    
    for allele_map in F:
        xls.append(XL_Allele(T, g, allele_map))
    
    return min(xls)

def XL_Allele(T:MUL, g:DAG, f:list)->int:
    pass



class InferMPAllop:
    
    def __init__(self, filename:str) -> None:
        #"src/PhyNetPy/Bayesian/mdc_tester.nex"
        self.gene_trees = NetworkBuilder2(filename).get_all_networks()
        self.cur_net : DAG = MDC_Tree(self.gene_trees).get()
        self.test_net : DAG = copy.deepcopy(self.cur_net)
        
        self.revert_info = None
        self.apply_info = None
        
    def modify(net : DAG) -> DAG:
        pass
    
    def revert(self) -> DAG:
        pass
    
    def apply(self, net : DAG) -> DAG:
        pass
    
    def infer(self, max_iter = 1000):
        
        i = 0
        cur_min = math.inf

        
        while i < max_iter:
            #Calculate XL score 
            score = XL(MUL().toMulTree(self.test_net))
            
            #If this network gives a lower score, save changes
            if score < cur_min:
                cur_min = score
                self.cur_net = self.apply(self.cur_net)
            
            #Try new changes
            self.test_net = self.modify(self.test_net)
                 
            i+=1
        
        
        return self.cur_net
    
    
    
        
        

    
    
    
    
    