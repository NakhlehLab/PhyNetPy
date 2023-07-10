from collections import deque
import copy
import random
from Node import Node
import numpy as np


def random_object(mylist, rng):
        
        rand_index = rng.integers(0, len(mylist))
        return mylist[rand_index]

def newickSubstring(children, node):
    """
        Returns the newick string for a subtree beginning at node

        ie a whole tree (A, (B, C)D)E with newickSubstring(D) returns (B, C)D

        node-- the root node (type Node) that starts the computation
        children-- a mapping from parent nodes to a list of their children
        
        """
    if node in children.keys():

        retStr = "("

        for child in children[node][0]:
            if not children[node][1]:
                retStr += newickSubstring(children, child)
            else:
                retStr += child.get_name()
                if child.length() is not None:
                    #TODO: NETWORKS?
                    retStr += ":" + str(list(child.length().values())[0])
            retStr += ", "

        retStr = retStr[:-2]  # Get rid of spurious comma

        retStr += ")" + node.get_name()
        if node.length() is not None:
            retStr += ":" + str(list(node.length().values())[0])
    else:
        retStr = node.get_name()
        if node.length() is not None:
            retStr += ":" + str(list(node.length().values())[0])

    return retStr


class GraphTopologyError(Exception):
    """
        This exception is raised when a graph is malformed in some way
        """

    def __init__(self, message="Error. Graph is malformed."):
        self.message = message
        super().__init__(self.message)


class Graph:
    """
        An "interface" level graph implementation. implements all common functionality
        between digraphs and undirected graphs
    """
    edges = []
    nodes = []
    edgeWeights = {}

    def __init__(self, edges, nodes, weights):
        self.edges = edges
        self.nodes = nodes
        self.edgeWeights = weights


    def addNodes(self, nodes):
        """
                if nodes is a list of data (doesn't matter what the data is), then add each data point to the list
                if nodes is simply a piece of data, then just add the data to the nodes list.
                """
        if type(nodes) == list:
            for node in nodes:
                if node not in self.nodes:
                    self.nodes.append(node)
        else:
            if nodes not in self.nodes:
                self.nodes.append(nodes)
        return

    def addEdges(self, edges, as_list = False):
        """
                if edges is a list of tuples, then add each tuple to the list of tuples
                if edges is simply a tuple, then just add the tuple to the edge list.
        """
        if as_list:
            for edge in edges:
                self.edges.append(edge)  
                edge[1].add_parent(edge[0])      
        else:
            self.edges.append(edges)
            edges[1].add_parent(edges[0])
        return

    def removeNode(self, node, remove_edges : bool = False):
        """
                Removes node from the list of nodes. If removeEdges is true/enabled,
                also prunes all edges from the graph that are connected to the node
                """
        
        if node in self.nodes:
            self.nodes.remove(node)
            if remove_edges:
                for edge in self.edges:
                    if node in edge:
                        self.edges.remove(edge)
                        
        

    def removeEdge(self, edge : list[Node]):
        """
                Removes edge from the list of edges. Does not delete nodes with no edges
                """
        #print(f"Requesting removal of : {edge[0].get_name(), edge[1].get_name()}")
            
        if edge in self.edges:
            #print("REMOVING EDGE")
            self.edges.remove(edge)
            edge[1].remove_parent(edge[0])
        else:
            raise GraphTopologyError("Edge should be deleted, was not")
        return

    def getNumberOfNodes(self):
        """
                returns the number of nodes in the graph
                """
        return len(self.nodes)

    def getNumberOfEdges(self):
        """
                returns the number of edges in the graph
                """
        return len(self.edges)

    def setEdgeWeights(self, edge_dict):
        for key, value in edge_dict.items():
            if key in self.edges:
                self.edgeWeights[key] = value
        return

    def getTotalWeight(self):
        tot = 0
        for edge in self.edges:
            tot += self.edgeWeights[edge]
        return tot

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


class DAG(Graph):
    """
    This class represents a directed graph containing nodes of any data type, and edges.
    An edge is a tuple (a,b) where a and b are nodes in the graph, and the direction of the edge
    is from a to b. (a,b) is not the same as (b,a).

    This particular graph instance must only have one root and must be connected. It must also be acyclic, but for efficiency, 
    cycles are allowed to be created. A checker method is provided, however.
    """

    def __init__(self, edges=None, nodes=None, weights=None):
        """
        Initialize a Directed Acyclic Graph (DAG) object.
        You may initialize with any combination of edges/nodes/weights, or provide none at all.

        Args:
            edges (list, optional): A list of Node 2-tuples. Defaults to None.
            nodes (list, optional): A list of Node objs. Defaults to None.
            weights (list, optional): _description_. Defaults to None.
        """
        
        self.reticulations = {}
        self.items : dict[str, object] = {}
        
        if edges is None:
            edges = []
        else:
            self.refresh_edge_data()
        if nodes is None:
            nodes = []
        if weights is None:
            weights = []

        self.UID = 0
        super().__init__(edges, nodes, weights)

        
    def get_item(self, key : str):
        return self.items[key]
    
    def put_item(self, key : str, item):
        if key not in self.items:
            self.items[key] = item

    def add_uid_node(self, node : Node):
        """
        Ensure an added node has a unique name that hasn't been used before/is not currently in use for this graph

        Args:
            node (Node): _description_
        """
        if node not in self.nodes:
            self.nodes.append(node)
            node.set_name("UID_" + str(self.UID))
            self.UID += 1
    
    def in_degree(self, node: Node):
        return len(self.inEdges(node))

    def out_degree(self, node: Node):
        return len(self.outEdges(node))

    def inEdges(self, node: Node):
        return [edge for edge in self.edges if edge[1] == node]

    def outEdges(self, node: Node):
        return [edge for edge in self.edges if edge[0] == node]
    
    def refresh_edge_data(self):
        new_retics = {}
        
        for edge in self.edges:
            if edge[0] in new_retics.keys():
                new_retics[edge[0]].add(edge)
            else:
                new_retics[edge[0]] = set(edge)
        
        #Gather only nodes with more than one out edge
        pruned = {node : new_retics[node] for node in new_retics.keys() if len(new_retics[node]) > 1}
        self.reticulations = pruned
    
    def get_reticulations(self):
        return self.reticulations
    
    def set_reticulations(self, retics):
        self.reticulations = retics
        
    def findRoot(self) -> list[Node]:
        """
        Finds the root of this DAG. It is an error if one does not exist
        or if there are more than one.
        
        TODO: Fix so that I return only a node
        """
        root = [node for node in self.nodes if self.in_degree(node) == 0 and self.out_degree(node) != 0]
        if len(root) != 1:
            raise GraphTopologyError("This graph does not have 1 and only 1 root node")
        return root

    def findDirectPredecessors(self, node: Node) -> list:
        """
        Returns a list of the children of node

        node-- A Node Object
        """
        return [edge[0] for edge in self.inEdges(node)]

    def findDirectSuccessors(self, node: Node) -> list:
        """
        Returns a list of the parent(s) of node. For a tree, this 
        list should be of length 1. For a network, a child may have more
        than one.

        node-- A Node Object
        """
        return [edge[1] for edge in self.outEdges(node)]

    def get_leaves(self) -> list:
        """
        returns the list of leaves in the graph, aka the set of nodes where there are no incoming edges
        """
        return [node for node in self.nodes if self.out_degree(node) == 0]

    def has_node_named(self, name : str) -> Node:
        """
        Check whether the graph has a node with a certain name.
        Strings must be exactly equal.

        Args:
            name (str): the name to search for

        Returns:
            Node: the node with the given name
        """
        for node in self.nodes:
            if node.get_name() == name:
                return node

        return False

    def print_graph(self):
        for node in self.nodes:
            if type(node) is Node:
                print(node.asString())

    def newick(self) -> str:
        """
        Build a map of parents to children using dfs, and then use that
        to call newickSubstring on the root. That will give the newick
        string for the network.

        Returns: a newick string.
        """

        root = self.findRoot()[0]
        children = {root: [set(), False]}
        visitedRetic = []

        # stack for dfs
        q = deque()
        q.append(root)

        while len(q) != 0:
            cur = q.pop()

            for neighbor in self.findDirectSuccessors(cur):

                # properly handle children mapping for parent "cur"
                if cur in children.keys():
                    children[cur][0].add(neighbor)
                else:
                    children[cur] = [{neighbor}, False]

                # tabulate whether node should be reprinted in any call
                # to newickSubstring. The subtree of a reticulation node
                # need only be printed once
                if neighbor.is_reticulation and neighbor in visitedRetic:
                    children[cur][1] = True
                elif neighbor.is_reticulation:
                    visitedRetic.append(neighbor)

                q.append(neighbor)

        # call newickSubstring on the root of the graph to get the entire string
        return newickSubstring(children, root) + ";"


    def pretty_print_edges(self)-> None:
        for edge in self.edges:
            print(f"<{edge[0].get_name()}, {edge[1].get_name()}>")
            print(edge)
            print("------")
        
        
        
    def prune_excess_nodes(self) -> None:
        
        root = self.findRoot()[0]
        
        q = deque()
        q.appendleft(root)
        

        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            for neighbor in self.findDirectSuccessors(cur):
                current_node : Node = neighbor
                previous_node : Node = cur
                node_removed = False
                
                #There could be a chain of nodes with in/out degree = 1. Resolve the whole chain before moving on to search more nodes
                while self.in_degree(current_node) == self.out_degree(current_node) == 1:
                    self.removeEdge([previous_node, current_node])
                    
                    previous_node = current_node
                    temp = self.findDirectSuccessors(current_node)[0]
                    self.removeNode(current_node, remove_edges=True)
                    current_node = temp
                    node_removed = True
                
                #We need to connect cur to its new successor
                if node_removed:
                    #self.removeEdge([previous_node, current_node])
                    self.addEdges([cur, current_node])    
                    current_node.set_parent([cur])
                
                #Resume search from the end of the chain if one existed, or this is neighbor if nothing was done
                q.append(current_node)
                    
    def generate_branch_lengths(self) -> None:
        """
        Assumes that each node in the graph does not yet have a branch length associated with it,
        but has a defined "t" attribute.
        
        """
        root = self.findRoot()[0]
        root.add_length(0, None)
        
        # stack for dfs
        q = deque()
        q.append(root)
        visited = set()

        while len(q) != 0:
            cur = q.pop()

            for neighbor in self.findDirectSuccessors(cur):
                if neighbor not in visited:
                    t_par = cur.attribute_value_if_exists("t")
                    t_nei = neighbor.attribute_value_if_exists("t")
                    
                    #Handle case that a "t" value doesn't exist
                    if t_par is None or t_nei is None:
                        raise Exception("Assumption that t attribute exists for all nodes has been violated")
                    neighbor.add_length(t_par - t_nei, cur)
                
                    q.append(neighbor)
                    visited.add(neighbor)

    def lca(self, set_of_nodes: set)-> Node:
        """
        Computes the Least Common Ancestor of a set of graph nodes

        Args:
            set_of_nodes (set): A set of Node objs that should be a subset of the set of leaves. 

        Returns:
            Node: The node that is the LCA of the set of leaves that was passed in.
        """
        format_set = set()
        for item in set_of_nodes:
            if type(item) is str:
                node_version = self.has_node_named(item)
                if node_version == False:
                    raise GraphTopologyError("Wrong name passed into LCA calculation")
                else:
                    format_set.add(node_version)
            elif type(item) is Node:
                format_set.add(item)
            else:
                raise GraphTopologyError("Wrong item type passed into LCA calculation")
        
        
        set_of_nodes = format_set
                
        leaf_2_parents = {} # mapping from each node in set_of_nodes to a mapping from ancestors to dist from node.
        
        for leaf in set_of_nodes:
            #Run bfs upward from each node 
            node_2_lvl : dict = {}
            
            # queue for bfs
            q = deque()
            q.append(leaf)
            visited = set()
            node_2_lvl[leaf] = 0

            while len(q) != 0:
                cur = q.popleft()

                #Seach cur's parents
                for neighbor in self.findDirectPredecessors(cur):
                    if neighbor not in visited:
                        node_2_lvl[neighbor] = node_2_lvl[cur] + 1
                        q.append(neighbor)
                        visited.add(neighbor)
            
            leaf_2_parents[leaf] = node_2_lvl
        
        
        #Compare each leaf's parents
        intersection = set(self.nodes)
        for leaf, par_level in leaf_2_parents.items():
            intersection = intersection.intersection(set(par_level.keys()))
        
        
    
        additive_level = {} # Map potential LCA's to cumulative distance from all the nodes
        
        for node in intersection: # A LCA has to be in each node's ancestor set
            lvl = 0
            for leaf in set_of_nodes:
                try:
                    lvl += leaf_2_parents[leaf][node]
                except KeyError:
                    continue
            
            additive_level[node] = lvl
        
        #The LCA is the node that minimizes cumulative distance
        return min(additive_level, key=additive_level.get)
                      
    def leaf_descendants(self, node : Node)->tuple:
        """
        Compute the set of all leaf nodes that are descendants of the parameter node. Uses DFS to find paths to leaves

        Args:
            node (Node): The node for which to compute leaf children

        Returns:
            tuple: The list of all leaves
        """
        root = node

        # stack for dfs
        q = deque()
        q.appendleft(root)
        leaves = set()

        while len(q) != 0:
            cur = q.popleft()
            
            if len(self.findDirectSuccessors(cur)) == 0: #cur is a leaf if out_degree = 0
                leaves.add(cur)
            
            for neighbor in self.findDirectSuccessors(cur): #Continue path to a leaf
                q.append(neighbor)
        
        return leaves    
        
    def get_all_clusters(self, node : Node)-> set:
        """
        Compile a list of non-trivial clusters (size > 1) that make up this graph.
        Ie: for a graph ((A, B)C, D); , set of all clusters is {(A,B), (A,B,C)}

        Args:
            node (Node): For any user call, this should be the root. For internal calls, it is the starting point for search.

        Returns:
            set: A set of all clusters in this graph. Each cluster is represented as a set.
        """
        
        cluster_set = set()
        graph_leaves = self.get_leaves()
        children = self.findDirectSuccessors(node)
        
        #Each leaf_descendant set of a child is a cluster, so long as it is not trivial
        for child in children:
            if child not in graph_leaves:
                #Get potential cluster
                leaf_descendant_set = self.leaf_descendants(child)
                #Check for size 
                if len(leaf_descendant_set) > 1: 
                    cluster_set.add(tuple(leaf_descendant_set))
                
                #Recurse over the next subtree
                cluster_set = cluster_set.union(self.get_all_clusters(child))
        
        return cluster_set
                
    def is_acyclic(self)->bool:
        """
        Checks via topological sort that this graph object is acyclic
        
        TODO: Untested

        Returns:
            bool: True, if graph is acyclic. Raises an error if there is a cycle.
        """
        
        graph_copy : DAG = copy.deepcopy(self)
        
        L = list() # Empty list where we put the sorted elements
        try:
            Q = {graph_copy.findRoot()[0]} # Set of all nodes with no incoming edges
        except GraphTopologyError:
            return False
        while len(Q) != 0:
            n = Q.pop()
            L.append(n)
            outgoing_n_edges = {edge[1]:edge for edge in graph_copy.edges if edge[0] == n}
            for m, e in outgoing_n_edges.items(): 
                graph_copy.removeEdge(e)
                if graph_copy.in_degree(m) == 0:
                    Q.add(m)
        
        if len(graph_copy.edges) != 0:
            return False
        else:
            return True
    
    
    
    def diff_subtree_edges(self, rng)-> list[list[Node]]:
        """
        Returns 2 edges such that neither edge is in the same "direct" subtree.
    
        Returns:
            list[list[Node]]: _description_
        """
        first_edge = random_object(self.edges, rng)
        first_edge_subtree = self.leaf_descendants(first_edge[1])
        second_edge = random_object([edge for edge in self.edges if len(set(self.leaf_descendants(edge[1])).intersection(set(first_edge_subtree))) == 0], rng)
        return [first_edge, second_edge]
    
    def subgenome_count(self, n : Node)->int:

        if self.findRoot()[0] == n:
            return 1
        else:
            parents = self.findDirectPredecessors(n)
            if len(parents) > 0:
                return sum([self.subgenome_count(parent) for parent in self.findDirectPredecessors(n)])
            else:
                return 0
    
    def edges_downstream_of_node(self, n : Node):
        q = deque()
        q.appendleft(n)
        
        edges : list[list[Node]] = list()
        
        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            for neighbor in self.findDirectSuccessors(cur):
                
                edges.append([cur, neighbor])
                
                #Resume search from the end of the chain if one existed, or this is neighbor if nothing was done
                q.append(neighbor)
        
        return edges
        
    def edges_to_subgenome_count(self, downstream_node : Node, delta : int, start_node):
        
        # print("----------")
        # print(f"FINDING VALID CONNECTIONS FOR {downstream_node.get_name()}")
        # print(f"STARTING BFS AT {start_node.get_name()}")
        
        
        q = deque()
        q.appendleft(start_node)
        
        edges_2_sub = {tuple(edge) : 0 for edge in self.edges}
        
        
        

        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            for neighbor in self.findDirectSuccessors(cur):
                
                edges_2_sub[(cur, neighbor)] += 1
                
                #Resume search from the end of the chain if one existed, or this is neighbor if nothing was done
                q.append(neighbor)
        
        partition : dict[int, list[list[Node]]] = {}
        for edge, value in edges_2_sub.items():
            if value not in partition.keys():
                partition[value] = [list(edge)]
            else:
                partition[value].append(list(edge))
        
        #Filter out invalid keys
        filter1 = {key : value for (key, value) in partition.items() if key <= delta}
        
        #Filter out edges that would create a cycle from param edge
        filter2 = {}
        for subct, edges in filter1.items():
            for target in edges:
                downstream_edges = self.edges_downstream_of_node(downstream_node)
                #print(f"DO NOT CONNECT TO THESE : {[(edge[0].get_name(), edge[1].get_name()) for edge in downstream_edges]}")
                if target not in downstream_edges:
                    if subct not in filter2.keys():
                        filter2[subct] = [target]
                    else:
                        filter2[subct].append(target)
       
        valid_connections = {subct : [[edge[0].get_name(), edge[1].get_name()] for edge in values] for (subct, values) in filter2.items()}
        # print(f"valid connections: {valid_connections}")
        # print("-------------")
        
        return filter2



  
