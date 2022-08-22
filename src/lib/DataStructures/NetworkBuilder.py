from operator import index
from nexus import NexusReader
from Bio import Phylo
from io import StringIO
from Graph import DAG
from Node import Node
import copy
from Node import NodeError


class NetworkBuilder:

    def __init__(self, filename):
        self.reader = NexusReader.from_file(filename)
        self.networks = []
        self.internalCount = 0
        self.build()

    def build(self):
        """
                Using the reader object, iterate through each of the trees 
                defined in the file and store them as Network objects into the 
                networks array
                """

        for t in self.reader.trees:
            # grab the right hand side of the tree definition
            handle = StringIO(str(t).split("=")[1])

            # parse the string handle
            tree = Phylo.read(handle, "newick")
            newNetwork = self.buildFromTreeObj(tree)
            # build the graph of the network
            self.networks.append(newNetwork)

    def buildFromTreeObj(self, tree):
        """
                Given a biopython Tree object (with nested clade objects)
                walk through the tree and build a network/ultrametric network
                from the nodes
                """

        print(tree)
        # Build a parent dictionary from the biopython tree obj
        parents = {}
        for clade in tree.find_clades(order="level"):
            for child in clade:
                parents[child] = clade

        # create new empty directed acyclic graph
        net = copy.deepcopy(DAG())

        # populate said graph with nodes and their attributes
        edges = []

        for node, par in parents.items():
            childNode = self.parseNode(node, net)
            parentNode = self.parseNode(par, net)

            childNode.addParent(parentNode)
            edges.append([parentNode, childNode])
            print("ADDING EDGE FROM " + parentNode.getName() + " TO " + childNode.getName())
        net.addEdges(edges)

        return net

    def parseAttributes(self, attrStr):
        """
                Takes the formatting string from the extended newick grammar and parses
                it into the event type and index.

                IE: #H1 returns "Hybridization", 1
                IE: #LGT21 returns "Lateral Gene Transfer", 21

                """
        if len(attrStr) < 2:
            raise NodeError("reticulation event label formatting incorrect")

        indexLookup = 0

        # decipher event type
        if attrStr[0] == "R":
            event = "Recombination"
            indexLookup = 1
        elif attrStr[0] == "H":
            event = "Hybridization"
            indexLookup = 1
        elif attrStr[0] == "L":
            try:
                if attrStr[1] == "G" and attrStr[2] == "T":
                    event = "Lateral Gene Transfer"
                    indexLookup = 3
            except:
                raise NodeError("Invalid label format string (event error)")
        else:
            raise NodeError("Invalid label format string (event error) ")

        # parse node index
        try:
            strnum = attrStr[indexLookup:]
            num = int(strnum)
            return event, num
        except:
            raise NodeError("Invalid label format string (number error)")

    def parseNode(self, node, network):
        """
                Takes in a clade, and outputs a Node object with the appropriate attributes
                """
        # if type(node) != Phylo.Clade:
        #         raise NodeError("attempting to parse a node that is not of class Clade")


        if node.name is None:
            newInternal = "Internal" + str(self.internalCount)
            self.internalCount += 1
            node.name = newInternal
            if node.branch_length is None:
                newNode = Node(name=newInternal)
            else:
                newNode = Node(branchLen=node.branch_length, name=newInternal)
            network.addNodes(newNode)
            return newNode

        extendedNewickParsedLabel = node.name.split("#")

        # if node already exists, just add its other parent
        oldNode = network.hasNodeWithName(extendedNewickParsedLabel[0])
        if oldNode != False:
            return oldNode

        # if its a reticulation node, grab the formatting information
        # only allow labels to have a singular #
        if len(extendedNewickParsedLabel) == 2:
            eventType, num = self.parseAttributes(extendedNewickParsedLabel[1])
            retValue = True
        elif len(extendedNewickParsedLabel) == 1:
            retValue = False
        else:
            raise NodeError("Node has a name that contains too many '#' characters. Must only contain 1")

        # create new node, with attributes if a reticulation node
        if (retValue):
            newNode = copy.deepcopy(
                Node(node.branch_length, name=extendedNewickParsedLabel[0], isReticulation=retValue))
            newNode.addAttribute("eventType", eventType)
            newNode.addAttribute("index", num)
        else:
            newNode = copy.deepcopy(
                Node(node.branch_length, name=extendedNewickParsedLabel[0], isReticulation=retValue))

        network.addNodes(newNode)
        return newNode

    def printNetworks(self):
        i = 0
        for net in self.networks:
            print("=========NETWORK #" + str(i) + "========")
            net.asciiGraph()
            i += 1

    def getNetwork(self, i):
        return self.networks[i]

##n = NetworkBuilder("src/io/testfile3.nex")
# n.printNetworks()

##test = n.getNetwork(1)
##print(test.newickString())
