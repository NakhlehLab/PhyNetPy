from nexus import NexusReader
from Bio import Phylo
from io import StringIO
from Graph import DAG
from Node import Node
import copy



def buildFromTreeObj(tree):
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
        
        #create new empty directed acyclic graph 
        net = copy.deepcopy(DAG())

        #populate said graph with nodes and their attributes
        
        
        for node, par in parents.items():
        
                if str(node.name)[0] == "#":
                        retValue = True
                else:
                        retValue = False
                
                oldNode = net.hasNodeWithName(node.name)
                if oldNode != False:
                        oldNode.addParent(par)
                else:
                        newNode = Node(node.branch_length, par, name = node.name, isReticulation=retValue)
                        net.addNodes(copy.deepcopy(newNode))

        return net

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
                        newNetwork = buildFromTreeObj(tree)
                        #build the graph of the network
                        self.networks.append(newNetwork)
        

                
        def printNetworks(self):
                i = 0
                for net in self.networks:
                        print("=========NETWORK #" + str(i) + "========")
                        net.printGraph()
                        i+=1
                
        


n = NetworkBuilder("src/io/testfile3.nex")

n.printNetworks()
        
