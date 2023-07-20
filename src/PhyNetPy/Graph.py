""" 
Author : Mark Kessler
Last Stable Edit : 7/16/23
First Included in Version : 0.1.0

"""

from collections import defaultdict, deque
import copy
import math
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



class DAG():
    """
    This class represents a directed graph containing nodes of any data type, and edges.
    An edge is a list [a,b] where a and b are nodes in the graph, and the direction of the edge
    is from a to b. [a,b] is NOT the same as [b,a], as this is a DIRECTED graph.

    Allowances:
    1) You may create cycles-- BUT we have provided a method to check if this graph object is acyclic
    2) You may have multiple roots. Be mindful of whether this graph is connected and what root you wish to operate on
    3) You may end up with floater nodes/edges, ie this may be an unconnected graph with multiple connected components. 
       We will provide in the next release a method to check for whether your graph object is one single connected component.
    
    """
    

    def __init__(self, edges=None, nodes=None, weights=None) -> None:
        """
        Initialize a Directed Acyclic Graph (DAG) object.
        You may initialize with any combination of edges/nodes/weights, or provide none at all.

        Args:
            edges (list, optional): A list of Node 2-tuples. Defaults to None.
            nodes (list, optional): A list of Node objs. Defaults to None.
            weights (list, optional): _description_. Defaults to None.
        """
        
        # Map nodes to the number of children they have
        self.out_degrees : dict[Node, int] = defaultdict(int)
        
        # Map nodes to the number of parents they have
        self.in_degrees : dict[Node, int] = defaultdict(int)
        
        # List of nodes in the graph with in degree of 0. Note that there could be floater nodes w/ out degree 0
        self.roots : list[Node] = []
        
        # List of nodes in the graph with out degree of 0. Note that there could be floater nodes w/ in degree 0
        self.leaves : list[Node] = []
        
        # Blob storage for anything that you want to associate with this network. Just give it a string key!
        self.items : dict[str, object] = {}
        
        # Map of nodes to their parents
        self.parent_map : dict[Node, list[Node]] = defaultdict(list) 
        
        # Map of nodes to their children
        self.child_map : dict[Node, list[Node]] = defaultdict(list)
        
        #Map of names of nodes to nodes
        self.node_names : dict[str, Node] = {}
        
        if edges is None:
            self.edges = []
        else:
            self.edges = edges
        
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes
            for node in self.nodes:
                self.node_names[node.get_name()] = node
            
        if weights is None:
            self.weights = []
        else:
            self.weights = weights

        # Initialize the unique id count
        self.UID = 0
        
        # If edges and nodes provided, then start bookkeeping!
        if nodes is not None and edges is not None:
            
            #Free floater nodes/edges are *technically* allowed
            for edge in self.edges:
                self.in_degrees[edge[1]] += 1
                self.out_degrees[edge[0]] += 1
                self.parent_map[edge[1]].append(edge[0])
                self.child_map[edge[0]].append(edge[1])
            
            self.leaves = [node for node in self.nodes if self.out_degrees[node] == 0]
            self.roots = [node for node in self.nodes if self.in_degrees[node] == 0]
            
            
    def print_adjacency(self):
        print("CHILD MAP:")
        print({par.get_name(): [child.get_name() for child in value] for (par, value) in self.child_map.items()})
        print("PARENT MAP:")
        print({child.get_name(): [par.get_name() for par in value] for (child, value) in self.parent_map.items()})
        
    def print_degrees(self):
        print("OUT")
        print({node.get_name() : value for (node, value) in self.out_degrees.items()})
        print("IN")
        print({node.get_name() : value for (node, value) in self.in_degrees.items()})
        
    def addNodes(self, nodes):
        """
        if nodes is a list of data (doesn't matter what the data is), then add each data point to the list
        if nodes is simply a piece of data, then just add the data to the nodes list.
        """
        if type(nodes) == list:
            for node in nodes:
                if node not in self.nodes:
                    self.nodes.append(node)
                    self.node_names[node.get_name()] = node
        else:
            if nodes not in self.nodes:
                self.nodes.append(nodes)
                self.node_names[nodes.get_name()] = nodes
        return

    def update_node_name(self, node, name):
        if node.get_name() is not None:
            del self.node_names[node.get_name()]
        node.set_name(name)
        self.node_names[name] = node
        
    def addEdges(self, edges, as_list = False):
        """
                if edges is a list of tuples, then add each tuple to the list of tuples
                if edges is simply a tuple, then just add the tuple to the edge list.
        """
        if as_list:
            for edge in edges:
                self.edges.append(edge) 
                self.out_degrees[edge[0]] += 1
                self.in_degrees[edge[1]] += 1
                self.child_map[edge[0]].append(edge[1])
                self.parent_map[edge[1]].append(edge[0])
                edge[1].add_parent(edge[0])  
                self.reclassify_node(edge[0], True, True)
                self.reclassify_node(edge[1], False, True)    
        else:
            self.edges.append(edges)
            edges[1].add_parent(edges[0])
            self.out_degrees[edges[0]] += 1
            self.in_degrees[edges[1]] += 1
            self.child_map[edges[0]].append(edges[1])
            self.parent_map[edges[1]].append(edges[0])
            self.reclassify_node(edges[0], True, True)
            self.reclassify_node(edges[1], False, True)   
        return

    def removeNode(self, node : Node):
        """
        Removes node from the list of nodes.
        Also prunes all edges from the graph that are connected to the node
        """
        
        if node in self.nodes:
            for edge in self.edges:
                if node in edge:
                    self.removeEdge(edge)
                        
            self.nodes.remove(node)
            del self.in_degrees[node]
            del self.out_degrees[node]
            del self.child_map[node]
            del self.parent_map[node]  
            del self.node_names[node.get_name()]     
    
    def removeEdge(self, edge : list[Node]):
        """
        Removes edge from the list of edges. Does not delete nodes with no edges
        """
        
        if edge in self.edges:
            self.edges.remove(edge)
            edge[1].remove_parent(edge[0])
            self.out_degrees[edge[0]] -= 1
            self.in_degrees[edge[1]] -= 1
            
            try:
                self.parent_map[edge[1]].remove(edge[0])
            except:
                pass
        
            try:
                self.child_map[edge[0]].remove(edge[1])
            except:
                pass
            
            self.reclassify_node(edge[0], True, False)
            self.reclassify_node(edge[1], False, False)
        
    def reclassify_node(self, node : Node, is_par : bool, is_addition : bool):
        if is_addition:
            if is_par:
                # If out degree now = 1, then the node was previously a leaf and is not anymore
                if self.out_degrees[node] == 1:
                    try:
                        self.leaves.remove(node)
                    except:
                        pass
                if self.in_degrees[node] == 0:
                    if node not in self.roots:
                        self.roots.append(node)
            else:
                # If in_degree now = 1, then the node was previously a root and is not anymore
                if self.in_degrees[node] == 1:
                    try:
                        self.roots.remove(node)
                    except:
                        pass
                if self.out_degrees[node] == 0:
                    if node not in self.leaves:
                        self.leaves.append(node)
        else:
            if is_par:
                # if out degree is now = 0, then the node is now a leaf
                if self.out_degrees[node] == 0:
                    self.leaves.append(node)
            else:
                # if in degree is now = 0, the node is now a root
                if self.in_degrees[node] == 0:
                    self.roots.append(node)
        
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

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges
        
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
            self.update_node_name(node, "UID_" + str(self.UID))
            #node.set_name("UID_" + str(self.UID))
            self.UID += 1
    
    def in_degree(self, node: Node):
        return self.in_degrees[node]

    def out_degree(self, node: Node):
        return self.out_degrees[node]

    def inEdges(self, node: Node):
        return [[par, node] for par in self.parent_map[node]]

    def outEdges(self, node: Node):
        return [[node, child] for child in self.child_map[node]]
    
    def root(self) -> list[Node]:
        """
        Finds the root of this DAG. It is an error if one does not exist
        or if there are more than one.
    
        """
        return [root for root in self.roots if self.out_degrees[root] != 0]

    def get_parents(self, node: Node) -> list[Node]:
        """
        Returns a list of the parents of node

        node-- A Node Object
        """
        return self.parent_map[node]
        

    def get_children(self, node: Node) -> list[Node]:
        """
        Returns a list of the children of a node. For a tree, this 
        list should be of length 2. For a network, this number may only be one

        node-- A Node Object
        """
        return self.child_map[node]

    def get_leaves(self) -> list[Node]:
        """
        returns the list of leaves in the graph, aka the set of nodes where there are no incoming edges
        """
        # print("LEAF NAMES:")
        # print([node.get_name() for node in self.leaves])
        # self.print_adjacency()
        # self.print_degrees()
        #Don't return floater leaves
        return [leaf for leaf in self.leaves if self.in_degrees[leaf] != 0]

    def has_node_named(self, name : str) -> Node:
        """
        Check whether the graph has a node with a certain name.
        Strings must be exactly equal.

        Args:
            name (str): the name to search for

        Returns:
            Node: the node with the given name
        """
        try:
            return self.node_names[name]
        except:
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

        root = self.root()[0]
        children = {root: [set(), False]}
        visitedRetic = []

        # stack for dfs
        q = deque()
        q.append(root)

        while len(q) != 0:
            cur = q.pop()

            for neighbor in self.get_children(cur):

                # properly handle children mapping for parent "cur"
                if cur in children.keys():
                    children[cur][0].add(neighbor)
                else:
                    children[cur] = [{neighbor}, False]

                # tabulate whether node should be reprinted in any call
                # to newickSubstring. The subtree of a reticulation node
                # need only be printed once
                if self.in_degrees[neighbor] == 2 and neighbor in visitedRetic:
                    children[cur][1] = True
                elif self.in_degrees[neighbor] == 2:
                    visitedRetic.append(neighbor)

                q.append(neighbor)

        # call newickSubstring on the root of the graph to get the entire string
        return newickSubstring(children, root) + ";"


    def pretty_print_edges(self)-> None:
        for edge in self.edges:
            print(f"<{edge[0].get_name()}, {edge[1].get_name()}>")
            print(edge)
            print("------")
        
    def remove_floaters(self) -> None:
        """
        Remove all nodes with in degree == out degree == 0
        """
        
        floaters = [node for node in self.nodes if self.in_degrees[node] == 0 and self.out_degrees[node] == 0]
        for floater in floaters:
            self.nodes.remove(floater)
            if floater in self.in_degrees.keys():
                del self.in_degrees[floater]
            if floater in self.out_degrees.keys():
                del self.out_degrees[floater]
            if floater in self.child_map.keys():
                del self.child_map[floater]
            if floater in self.parent_map.keys():
                del self.parent_map[floater]
       
    def remove_excess_branch(self):
        root = self.root()[0]
        if self.out_degrees[root] == 1:
            self.removeEdge([root, self.get_children(root)[0]])
        
        
    def prune_excess_nodes(self) -> None:
        
        root = self.root()[0]
        
        q = deque()
        q.appendleft(root)
        
        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            for neighbor in self.get_children(cur):
                current_node : Node = neighbor
                previous_node : Node = cur
                node_removed = False
                
                #There could be a chain of nodes with in/out degree = 1. Resolve the whole chain before moving on to search more nodes
                while self.in_degree(current_node) == self.out_degree(current_node) == 1:
                    self.removeEdge([previous_node, current_node])
                    
                    previous_node = current_node
                    temp = self.get_children(current_node)[0]
                    self.removeNode(current_node)
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
        root = self.root()[0]
        root.add_length(0, None)
        
        # stack for dfs
        q = deque()
        q.append(root)
        visited = set()

        while len(q) != 0:
            cur = q.pop()

            for neighbor in self.get_children(cur):
                if neighbor not in visited:
                    t_par = cur.attribute_value_if_exists("t")
                    t_nei = neighbor.attribute_value_if_exists("t")
                    
                    #Handle case that a "t" value doesn't exist
                    if t_par is None or t_nei is None:
                        raise Exception("Assumption that t attribute exists for all nodes has been violated")
                    neighbor.add_length(t_par - t_nei, cur)
                
                    q.append(neighbor)
                    visited.add(neighbor)

    def mrca(self, set_of_nodes: set)-> Node:
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
                for neighbor in self.get_parents(cur):
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
        
            if self.out_degrees[cur] == 0:
                leaves.add(cur)
                
            for neighbor in self.get_children(cur): #Continue path to a leaf
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
        children = self.get_children(node)
        
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
            Q = {graph_copy.root()[0]} # Set of all nodes with no incoming edges
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

        if self.root()[0] == n:
            return 1
        else:
            parents = self.get_parents(n)
            if len(parents) > 0:
                return sum([self.subgenome_count(parent) for parent in self.get_parents(n)])
            else:
                return 0
    
    def edges_downstream_of_node(self, n : Node):
        q = deque()
        q.appendleft(n)
        
        edges : list[list[Node]] = list()
        
        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            for neighbor in self.get_children(cur):
                
                edges.append([cur, neighbor])
                
                #Resume search from the end of the chain if one existed, or this is neighbor if nothing was done
                q.append(neighbor)
        
        return edges
    
    def edges_to_subgenome_count2(self):
        
        
        start_nodes = self.root()
        if len(self.root()) != 1:
            raise Exception("Please specify a start node for this network, there is more than one root (or none)")
        start_node = start_nodes[0]
            
        q = deque()
        q.appendleft(start_node)
        
        edges_2_sub = {tuple(edge) : 0 for edge in self.edges}
        
        
        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            for neighbor in self.get_children(cur):
                
                edges_2_sub[(cur, neighbor)] += 1
                
                #Resume search from the end of the chain if one existed, or this is neighbor if nothing was done
                q.append(neighbor)
        
        return edges_2_sub
    
    def edges_to_subgenome_count(self, downstream_node : Node = None, delta : float = math.inf, start_node = None):
        
        # print("----------")
        # print(f"FINDING VALID CONNECTIONS FOR {downstream_node.get_name()}")
        # print(f"STARTING BFS AT {start_node.get_name()}")
        if start_node is None:
            start_nodes = self.root()
            if len(self.root()) != 1:
                raise Exception("Please specify a start node for this network, there is more than one root (or none)")
            start_node = start_nodes[0]
            
        q = deque()
        q.appendleft(start_node)
        
        edges_2_sub = {tuple(edge) : 0 for edge in self.edges}
        
        
        while len(q) != 0:
            cur = q.pop() #pop right for bfs

            for neighbor in self.get_children(cur):
                
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
        if downstream_node is not None:
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
            return filter2
        else:
            return filter1

    def leaf_descendants_all(self):
        desc_map : dict[Node, set[Node]] = {}
        leaves = self.get_leaves()
        
        self.leaf_desc_help(self.root()[0], leaves, desc_map)
        return desc_map
        
    
    def leaf_desc_help(self, node : Node, leaves : list[Node], desc_map : dict[Node, set[Node]]):
        
        if node not in desc_map.keys():
            if node in leaves:
                desc_map[node] = {node}
            else:
                desc_map[node] = set()
                for child in self.get_children(node):
                    desc_map[node] = desc_map[node].union(self.leaf_desc_help(child, leaves, desc_map))
                       
        return desc_map[node]

    def to_newick(self):
        processed_retics = set()
        return self.newick_help(self.root()[0], processed_retics) + ";"
    
    def newick_help(self, node : Node, processed_retics : set[Node]):
        
        if node in self.leaves:
            return node.get_name()
        else:
            if self.in_degrees[node] == 2 and node in processed_retics:
                if node.get_name()[0] != "#":
                    return "#" + node.get_name()
                return node.get_name()
            else:
                if self.in_degrees[node] == 2:
                    processed_retics.add(node)
                    if node.get_name()[0] != "#":
                        node_name = "#" + node.get_name()
                    else:
                        node_name = node.get_name()
                else:
                    node_name = node.get_name()    
                    
                substr = "("
                for child in self.get_children(node):
                    substr += self.newick_help(child, processed_retics)
                    substr += ","
                substr = substr[0:-1]
                substr += ")"
                substr += node_name
                
                return substr
            
        

  
