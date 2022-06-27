from collections import deque
import Graph

class undirectedGraph(Graph):
        """
        This class represents a directed graph containing nodes of any data type, and edges.
        An edge is a tuple (a,b) where a and b are nodes in the graph, and the direction of the edge
        is from a to b. (a,b) is not the same as (b,a).
        """
        edges = []
        nodes = []
        edgeWeights = {}
        connectedBool = True
        
        def __init__(self, edges, nodes, weights):
                super.__init__(edges, nodes, weights)
        
        
        def addEdges(self, edges):
                """
                if edges is a list of sets, then add each set to the list of sets
                if edges is simply a set, then just add the set to the edge list.
                """
                if type(edges) == list:
                        for edge in edges:
                                if edge not in self.edges:
                                        self.edges.append(edge)
                else:
                        if edge not in self.edges:
                                self.edges.append(edges)
                return
        
                        
        

        def inDegree(self, node):
                return len(self.inEdges(node))
                
        def outDegree(self, node):
                return len(self.outEdges(node))

        def inEdges(self, node):
                return [edge for edge in self.edges if node in edge]
                
        def outEdges(self, node):
                return [edge for edge in self.edges if node in edge]
                
        def findRoots(self):
                """
                if this directed graph has nodes with in-degree 0, then they will be returned in a list.
                if no such nodes exist, an empty list is returned
                """
                return None

        
        def findNeighbors(self, node):
                neighbors = set()
                for edge in self.inEdges(node):
                        if edge[0] == node:
                                neighbors.add(edge[1])
                        else:
                                neighbors.add(edge[0])
                return neighbors
        
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
                returns the list of leaves in the graph. In this case, any nodes
                that do not have any edges
                """
                return [node for node in self.nodes if self.outDegree(node)==0]

        
        
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
                
                return False


        def addReticulationEdge(self):
                """
                Unwritten
                """
                return 0
        
        def setEdgeWeights(self, edgeDict):
                for key, value in edgeDict.items():
                        if key in self.edges:
                                self.edgeWeights[key] = value
                return 

                
        def minimumSpanningTree(self):
                """
                implements kruskal's algorithm
                returns a graph instance representing the MST
                """

                if self.isConnected() == False:
                        return "Attempting to find MST of unconnected graph"
                
                mst = Graph([],[], {})
                edgeCandidates = self.edgeWeights.copy()

                while mst.getNumberOfEdges() <= self.getNumberOfNodes() - 1:
                        edgeCandidates = self.findMinEdgeAndInsert(mst, edgeCandidates)
                
                return mst


        def findMinEdgeAndInsert(self, graph, potentialEdges):
                
                minKey = min(potentialEdges, key=potentialEdges.get)
                
                graph.addNodes([minKey[0], minKey[1]])
                graph.addEdges(minKey)
                print("Adding nodes", minKey[0], minKey[1])

                if graph.containsCycle() == True:
                        graph.removeNode(minKey[0], False)
                        graph.removeNode(minKey[1], False)
                        graph.removeEdge(minKey)
                        del potentialEdges[minKey]
                        self.findMinEdgeAndInsert(graph, potentialEdges)
                else:
                        del potentialEdges[minKey]
                        return potentialEdges
                


        def containsCycle(self):
                return False