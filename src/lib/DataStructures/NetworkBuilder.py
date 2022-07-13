from nexus import NexusReader
from Bio import Phylo
from io import StringIO
from Graph import DAG
from Node import Node


def get_parent(tree, child_clade):
    node_path = tree.get_path(child_clade)
    print(node_path)
    if len(node_path) == 0:
        return None
    elif len(node_path) == 1:
        return node_path[0]
    return node_path[-2]




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
                        handle = StringIO(str(t).split("=")[1])
                        tree = Phylo.read(handle, "newick")
                        self.networks.append(self.buildFromTreeObj(tree))
                        return
        
        def buildFromTreeObj(self, tree):
                parents = {}
                for clade in tree.find_clades(order="level"):
                        for child in clade:
                                parents[child] = clade
                
                net = DAG()

                for node, par in parents.items():
                        newNode = Node(node.branch_length, par, name = node.name)
                        net.addNodes(newNode)
                
                return net




        
        def printNetworks(self):
                for net in self.networks:
                        net.printGraph()
                
        


n = NetworkBuilder("src/io/testfile3.nex")

n.printNetworks()
        
