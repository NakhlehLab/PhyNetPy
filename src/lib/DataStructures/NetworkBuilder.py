from nexus import NexusReader
from Bio import Phylo
from io import StringIO
from Graph import DAG
from Node import Node


class NetworkBuilder:

        def __init__(self, filename):
                self.reader = NexusReader.from_file(filename)
                self.networks = []
                self.build()

        def build(self):
                """
                Using the reader object, iterate through each of the trees 
                defined in the file and store them as Network objects into the 
                networks array
                """

                for t in self.reader.trees:
                        #grab the right hand side of the tree definition
                        handle = StringIO(str(t).split("=")[1])

                        #parse the string handle
                        tree = Phylo.read(handle, "newick")

                        #build the graph of the network
                        self.networks.append(self.buildFromTreeObj(tree))

                        return #just do 1 for now
        
        def buildFromTreeObj(self, tree):
                """
                Given a biopython Tree object (with nested clade objects)
                walk through the tree and build a network/ultrametric network
                from the nodes
                """

                #Build a parent dictionary from the biopython tree obj
                parents = {}
                for clade in tree.find_clades(order="level"):
                        for child in clade:
                                parents[child] = clade
                
                #create new directed acyclic graph 
                net = DAG()

                #populate said graph with nodes and their attributes
                for node, par in parents.items():
                        newNode = Node(node.branch_length, par, name = node.name)
                        net.addNodes(newNode)
                
                return net




        
        def printNetworks(self):
                for net in self.networks:
                        net.printGraph()
                
        


n = NetworkBuilder("src/io/testfile3.nex")

n.printNetworks()
        
