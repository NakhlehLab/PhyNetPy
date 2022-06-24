from collections import deque

class Graph:
        """
        This class represents a directed graph containing nodes of any data type, and edges.
        An edge is a tuple (a,b) where a and b are nodes in the graph, and the direction of the edge
        is from a to b. (a,b) is not the same as (b,a).
        """
        edges = []
        nodes = []
        connectedBool = True
        
        def __init__(self, edges, nodes):
                self.edges = edges
                self.nodes = nodes
        
        def addNodes(self, nodes):
                """
                if nodes is a list of data (doesn't matter what the data is), then add each data point to the list
                if nodes is simply a piece of data, then just add the data to the nodes list.
                """
                if type(nodes) == list:
                        for node in nodes:
                                self.nodes.append(node)
                else:
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

        def inDegree(self, node):
                return len(self.inEdges(node))
                
        def outDegree(self, node):
                return len(self.outEdges(node))

        def inEdges(self, node):
                return [edge for edge in self.edges if edge[1] == node]
                
        def outEdges(self, node):
                return [edge for edge in self.edges if edge[0] == node]
                
        def findRoots(self):
                """
                if this directed graph has nodes with in-degree 0, then they will be returned in a list.
                if no such nodes exist, an empty list is returned
                """
                return [node for node in self.nodes if self.inDegree(node)==0]

        
        def findDirectPredecessors(self, node):
                return [edge[0] for edge in self.inEdges(node)]
        
        def findDirectSuccessors(self, node):
                return [edge[1] for edge in self.outEdges(node)]
        
        def findShortestPathLength(self, node):
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
        
        def getLeafs(self):
                """
                returns the list of leaves in the graph
                """
                return [node for node in self.nodes if self.outDegree(node)==0]

        def top_sort(self):
                """
                Execute a topological sort.
                Returns a list of nodes
                """
                self.connectedBool = True
                sorted_nodes, visited = deque(), set()
                count = 0
                for node in self.findRoots(): #start dfs at roots for correctness
                        if node not in visited:
                                self.dfs(node, visited, sorted_nodes)
                                #if the initial dfs call is run more than once,
                                #then nodes were not reached initially in which case
                                #the graph is not connected
                                if count != 0:
                                        self.connectedBool = False
                                count+=1
                                
                return list(sorted_nodes)
        
        def dfs(self, start_node, visited, sorted_nodes):
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

        def isConnected(self):
                """
                If dfs is run and the nodes it finds is not a complete list, then there are
                more than one connected component and the graph is therefore not connected
                """
                self.top_sort()
                if self.connectedBool:
                        return True
                return False


        def addReticulationEdge(self):
                """
                Unwritten
                """
                return 0

        


def graphTestSuite():
        g = Graph([("a","b"), ("a", "c"), ("b","d"), ("c","d")], ["a", "b", "c", "d"])
        
        print(g.findDirectSuccessors("a"))
        print(g.getLeafs())
        print(g.getNumberOfNodes())
        print(g.getNumberOfEdges())
        print(g.top_sort())
        print(g.isConnected())
        g.addNodes("e")
        print(g.getNumberOfNodes())
        print(g.top_sort())
        print(g.isConnected())
        g.addEdges(("e", "a"))
        print(g.isConnected())
        print(g.getNumberOfEdges())
        g.removeNode("c", True)
        print(g.top_sort())
        print(g.getNumberOfEdges())



graphTestSuite()