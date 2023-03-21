from collections import deque
import copy
import random
from Node import Node
import numpy as np




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

    def addEdges(self, edges):
        """
                if edges is a list of tuples, then add each tuple to the list of tuples
                if edges is simply a tuple, then just add the tuple to the edge list.
        """
        if type(edges) == list:
            for edge in edges:
                self.edges.append(edge)        
        else:
            self.edges.append(edges)
        return

    def removeNode(self, node, remove_edges):
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
        return

    def removeEdge(self, edge):
        """
                Removes edge from the list of edges. Does not delete nodes with no edges
                """
        if edge in self.edges:
            self.edges.remove(edge)
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
        if edges is None:
            edges = []
        if nodes is None:
            nodes = []
        if weights is None:
            weights = []

        
        super().__init__(edges, nodes, weights)

    def inDegree(self, node: Node):
        return len(self.inEdges(node))

    def outDegree(self, node: Node):
        return len(self.outEdges(node))

    def inEdges(self, node: Node):
        return [edge for edge in self.edges if edge[1] == node]

    def outEdges(self, node: Node):
        return [edge for edge in self.edges if edge[0] == node]

    def findRoot(self) -> list:
        """
        Finds the root of this DAG. It is an error if one does not exist
        or if there are more than one.
        
        TODO: Fix so that I return only a node
        """
        root = [node for node in self.nodes if self.inDegree(node) == 0]
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

    def getLeafs(self) -> list:
        """
        returns the list of leaves in the graph, aka the set of nodes where there are no incoming edges
        """
        return [node for node in self.nodes if self.outDegree(node) == 0]

    def hasNodeWithName(self, name):
        for node in self.nodes:
            if node.get_name() == name:
                return node

        return False

    def printGraph(self):
        for node in self.nodes:
            if type(node) is Node:
                print(node.asString())

    def newickString(self):
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

    def generate_branch_lengths(self):
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
                    neighbor.add_length(t_par - t_nei, cur)
                
                    q.append(neighbor)
                    visited.add(neighbor)

    def LCA(self, set_of_nodes: set)-> Node:
        """
        Computes the Least Common Ancestor of a set of graph nodes

        Args:
            set_of_nodes (set): A set of Node objs that should be a subset of the set of leaves. 

        Returns:
            Node: The node that is the LCA of the set of leaves that was passed in.
        """
        
        leaf_2_parents = {}
        
        for leaf in set_of_nodes:
        
            node_2_lvl : dict = {}
            
            # queue for bfs
            q = deque()
            q.append(leaf)
            visited = set()
            node_2_lvl[leaf] = 0

            while len(q) != 0:
                cur = q.popleft()

                for neighbor in self.findDirectPredecessors(cur):
                    if neighbor not in visited:
                        node_2_lvl[neighbor] = node_2_lvl[cur] + 1
                        q.append(neighbor)
                        visited.add(neighbor)
            
            leaf_2_parents[leaf] = node_2_lvl
        
        #Compare each leaf's parents
        intersection = set()
        for leaf, par_level in leaf_2_parents.items():
            intersection.union(set(par_level.keys()).difference(intersection))
        
        additive_level = {}
        for node in intersection:
            lvl = 0
            for leaf in set_of_nodes:
                try:
                    lvl += leaf_2_parents[leaf][node]
                except KeyError:
                    continue
            
            additive_level[node] = lvl
        
        return min(additive_level, key=additive_level.get)
                      
    def leaf_descendants(self, node)->tuple:
        root = node

        # stack for dfs
        q = deque()
        q.appendleft(root)
        leaves = set()

        while len(q) != 0:
            cur = q.popleft()
            if len(self.findDirectSuccessors(cur)) == 0:
                leaves.add(cur)
                
            for neighbor in self.findDirectSuccessors(cur):
                q.append(neighbor)
        
        return leaves    
        
    def get_all_clusters(self, node)-> set:
        cluster_set = set()
        graph_leaves = self.getLeafs()
        children = self.findDirectSuccessors(node)
        #print(f'children names: {[n.get_name() for n in children]}')
        for child in children:
            if child not in graph_leaves:
                #print(f'child: {[n.get_name() for n in self.leaf_descendants(child)]}')
                leaf_descendant_set = self.leaf_descendants(child)
                if len(leaf_descendant_set) > 1: 
                    cluster_set.add(tuple(leaf_descendant_set))
                cluster_set = cluster_set.union(self.get_all_clusters(child))
        
        return cluster_set
                
    def is_acyclic(self):
        """
        Checks via topological sort that this graph object is acyclic
        
        TODO: Untested

        Returns:
            bool: True, if graph is acyclic. Raises an error if there is a cycle.
        """
        
        graph_copy : DAG = copy.deepcopy(self)
        
        L = list() # Empty list where we put the sorted elements
        Q = {graph_copy.findRoot()[0]} # Set of all nodes with no incoming edges
        while len(Q) != 0:
            n = Q.pop()
            L.append(n)
            outgoing_n_edges = {edge[1]:edge for edge in graph_copy.edges if edge[0] == n}
            for m, e in outgoing_n_edges.items(): 
                graph_copy.removeEdge(e)
                if graph_copy.inDegree(m) == 0:
                    Q.add(m)
        if len(graph_copy.edges) != 0:
            GraphTopologyError("Graph has cycles")
        else:
            return True



class DAG_Modifiers:
    
    def __init__(self, network : DAG) -> None:
        self.net = network
        
        
    def add_retic(self)->dict:
        pass
    
    def remove_retic(self)->dict:
        pass
    
    def edge_tail_relocation(self)->dict:
        pass
    
    def edge_head_relocation(self)->dict:
        pass
    
    def retic_relocation(self)->dict:
        pass
    
    def retic_direction_flip(self)->dict:
        pass
    
    
    def random_mod(self):
        random_selection = random.randint(1,6)
        if random_selection == 1:
            self.add_retic()
        elif random_selection == 2:
            self.remove_retic()
        elif random_selection == 3:
            self.edge_tail_relocation()
        elif random_selection == 4:
            self.edge_head_relocation()
        elif random_selection == 5:
            self.retic_relocation()
        else:
            self.retic_direction_flip()
        
    def get(self):
        return self.net
    

  
  
