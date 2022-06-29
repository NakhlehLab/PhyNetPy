from collections import deque



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
                        if node not in self.nodes:
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

class diGraph(Graph):
        """
        This class represents a directed graph containing nodes of any data type, and edges.
        An edge is a tuple (a,b) where a and b are nodes in the graph, and the direction of the edge
        is from a to b. (a,b) is not the same as (b,a).
        """
        
        connectedBool = True
        
        def __init__(self, edges, nodes, weights):
                super().__init__(edges, nodes, weights)
        
        
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

        
class undiGraph(Graph):

        def __init__(self, edges, nodes, weights):
                super().__init__(edges, nodes, weights)
        
        def isConnected(self):
                return True
                
        def minimumSpanningTree(self):
                """
                implements kruskal's algorithm
                returns a graph instance representing the MST
                """

                if self.isConnected() == False:
                        return "Attempting to find MST of unconnected graph"
                
                mst = undiGraph([], [], {})
                edgeCandidates = self.edgeWeights.copy()

                while mst.getNumberOfEdges() < self.getNumberOfNodes() - 1:
                        edgeCandidates = self.findMinEdgeAndInsert(mst, edgeCandidates)
                
                return mst


        def findMinEdgeAndInsert(self, graph, potentialEdges):
                
                minKey = list(min(potentialEdges, key=potentialEdges.get))
                
                graph.addNodes([minKey[0], minKey[1]])
                graph.addEdges(frozenset(minKey))
                graph.setEdgeWeights({frozenset(minKey) : self.edgeWeights[frozenset(minKey)]})
                print("Adding nodes", minKey[0], minKey[1])

                if graph.containsCycle() == True:
                        graph.removeNode(minKey[0], False)
                        graph.removeNode(minKey[1], False)
                        graph.removeEdge(minKey)
                        del potentialEdges[frozenset(minKey)]
                        self.findMinEdgeAndInsert(graph, potentialEdges)
                else:
                        del potentialEdges[frozenset(minKey)]
                        return potentialEdges    


        def containsCycle(self):
                """
                need to implement this for correctness on graphs with cycles
                """
                return False


        


def graphTestSuite():
        g = undiGraph([frozenset(["a","b"]), frozenset(["a", "c"]), frozenset(["c","d"]), frozenset(["c","b"]), frozenset(["b", "e"])], ["a", "b", "c", "d", "e"], {frozenset(["a","b"]):1, frozenset(["c","d"]):1.2, frozenset(["c","b"]):2, frozenset(["b", "e"]):3})
        # print(g.findDirectSuccessors("a"))
        # print(g.getLeafs())
        # print(g.getNumberOfNodes())
        # print(g.getNumberOfEdges())
        print(g.minimumSpanningTree().getNumberOfEdges())
        print(g.minimumSpanningTree().getNumberOfNodes())
        print(g.minimumSpanningTree().getTotalWeight())
        # print(g.isConnected())
        # g.addNodes("e")
        # print(g.getNumberOfNodes())
        # print(g.top_sort())
        # print(g.isConnected())
        # g.addEdges(("e", "a"))
        # print(g.isConnected())
        # print(g.getNumberOfEdges())
        # g.removeNode("c", True)
        # print(g.top_sort())
        # print(g.getNumberOfEdges())



graphTestSuite()