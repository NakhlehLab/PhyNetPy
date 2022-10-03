from collections import deque
from queue import Queue
from Node import Node


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
                if child.length is not None:
                    retStr += ":" + str(child.length())
            retStr += ", "

        retStr = retStr[:-2]  # Get rid of spurious comma

        retStr += ")" + node.get_name()
        if node.length() is not None:
            retStr += ":" + str(node.length())
    else:
        retStr = node.get_name()
        if node.length() is not None:
            retStr += ":" + str(node.length())

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
                if edge not in self.edges:
                    self.edges.append(edge)
        else:
            if edges not in self.edges:
                self.edges.append(edges)
        return

    def removeNode(self, node, removeEdges):
        """
                Removes node from the list of nodes. If removeEdges is true/enabled,
                also prunes all edges from the graph that are connected to the node
                """
        if node in self.nodes:
            self.nodes.remove(node)
            if removeEdges:
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

    def findShortestPathLength(self, nodeStart, nodeEnd):
        """
                TODO
                """
        return 0

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

    def setEdgeWeights(self, edgeDict):
        for key, value in edgeDict.items():
            if key in self.edges:
                self.edgeWeights[key] = value
        return

    def getTotalWeight(self):
        sum = 0
        for edge in self.edges:
            sum += self.edgeWeights[edge]
        return sum

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


class DAG(Graph):
    """
        This class represents a directed graph containing nodes of any data type, and edges.
        An edge is a tuple (a,b) where a and b are nodes in the graph, and the direction of the edge
        is from a to b. (a,b) is not the same as (b,a).

        This particular graph instance must only have one root and must be connected.
    """

    connectedBool = True

    def __init__(self, edges=[], nodes=[], weights=[]):
        super().__init__(edges, nodes, weights)

    def inDegree(self, node):
        return len(self.inEdges(node))

    def outDegree(self, node, debug=False):
        if debug:
            print("OUT EDGES" + str(self.outEdges(node)))
        return len(self.outEdges(node))

    def inEdges(self, node):
        return [edge for edge in self.edges if edge[1] == node]

    def outEdges(self, node):
        return [edge for edge in self.edges if edge[0] == node]

    def findRoot(self):
        """
                Finds the root of this DAG. It is an error if one does not exist
                or if there are more than one.
                """
        root = [node for node in self.nodes if self.inDegree(node) == 0]
        if len(root) != 1:
            raise GraphTopologyError("This graph does not have 1 and only 1 root node")
        return root

    def findDirectPredecessors(self, node):
        """
                Returns a list of the children of node

                node-- A Node Object
                """
        return [edge[0] for edge in self.inEdges(node)]

    def findDirectSuccessors(self, node):
        """
                Returns a list of the parent(s) of node. For a tree, this 
                list should be of length 1. For a network, a child may have more
                than one.

                node-- A Node Object
                """
        return [edge[1] for edge in self.outEdges(node)]

    def getLeafs(self):
        """
                returns the list of leaves in the graph
                """
        return [node for node in self.nodes if self.outDegree(node) == 0]

    def top_sort(self):
        """
                Execute a topological sort.
                Returns a list of nodes
                """
        self.connectedBool = True
        sorted_nodes, visited = deque(), set()
        count = 0
        for node in self.findRoot():  # start dfs at roots for correctness
            if node not in visited:
                self.dfs_recursive(node, visited, sorted_nodes)
                # if the initial dfs call is run more than once,
                # then nodes were not reached initially in which case
                # the graph is not connected
                if count != 0:
                    self.connectedBool = False
                count += 1

        return list(sorted_nodes)

    def dfs_recursive(self, start_node, visited, sorted_nodes):
        """
                recursive dfs helper
                """
        visited.add(start_node)
        if start_node in self.nodes:
            neighbors = self.findDirectSuccessors(start_node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    self.dfs(neighbor, visited, sorted_nodes)
        sorted_nodes.appendleft(start_node)

    def bfs(self, startNode):
        """
                Vanilla bfs
                """

        dist = {startNode: 0}
        parent = {startNode: None}

        q = Queue(maxsize=0)  # unlimited length queue
        q.put(startNode)

        while not q.empty():
            cur = q.get()
            for neighbor in self.findDirectSuccessors(cur):
                if neighbor not in dist.keys():
                    dist[neighbor] = dist[cur] + 1
                    parent[neighbor] = cur
                    q.put(neighbor)

        return parent, dist

    def graphSize(self, node):
        parent, dist = self.bfs(node)
        return len(list(parent.keys()))

    def hasNodeWithName(self, name):
        for node in self.nodes:
            if node.get_name() == name:
                # print(node)
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

