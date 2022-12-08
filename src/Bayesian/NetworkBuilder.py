from operator import index
from nexus import NexusReader
from Bio import Phylo
from io import StringIO
from Graph import DAG
from Node import Node
import copy
from Node import NodeError

class NetworkBuilderError(Exception):
    def __init__(self, message = "Something went wrong with building the network") -> None:
        self.message = message
        super().__init__(self.message)


class NetworkBuilder:

    def __init__(self, filename):
        self.reader = NexusReader.from_file(filename)
        self.networks = []
        self.internalCount = 0
        self.name_2_net = {}
        self.inheritance_queue : set = set()
        self.build()

    def build(self):
        """
        Using the reader object, iterate through each of the trees 
        defined in the file and store them as Network objects into the 
        networks array
        """

    
        if self.reader.trees is None:
            raise NetworkBuilderError("There are no trees listed in the file")

        for t in self.reader.trees:
            # grab the right hand side of the tree definition for the tree, and the left for the name
            name = str(t).split("=")[0].split(" ")[1]
            handle = StringIO("=".join(str(t).split("=")[1:]))

            # parse the string handle
            tree = Phylo.read(handle, "newick")
            newNetwork = self.buildFromTreeObj(tree)
            # build the graph of the network
            self.networks.append(newNetwork)
            self.name_2_net[newNetwork] = name

    def buildFromTreeObj(self, tree):
        """
                Given a biopython Tree object (with nested clade objects)
                walk through the tree and build a network/ultrametric network
                from the nodes
                """

        # Build a parent dictionary from the biopython tree obj
        parents = {}
        for clade in tree.find_clades(order="level"):
            for child in clade:
                parents[child] = clade

        # create new empty directed acyclic graph
        net = DAG()

        # populate said graph with nodes and their attributes
        edges = []

        for node, par in parents.items():
            parentNode = self.parseNode(par, net, called_as_parent = True)
            childNode = self.parseNode(node, net, parent = parentNode)
    
            edges.append([parentNode, childNode])
        net.addEdges(edges)
        
        for node_pair in self.inheritance_queue:
            the_node = node_pair[0]
            par_node = node_pair[1]
            
            gamma = the_node.attribute_value_if_exists("gamma")
            if len(list(gamma.keys())) != 1:
                raise NetworkBuilderError("There is an incorrect amount of entries in the gamma attribute")
            complement = 1 - gamma[list(gamma.keys())[0]]
            the_node.add_attribute("gamma", {par_node: complement}, append=True)
            

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

    def parseNode(self, node, network, called_as_parent = False, parent : Node = None):
        
        if node.name is None:
            newInternal = "Internal" + str(self.internalCount)
            self.internalCount += 1
            node.name = newInternal
            if node.branch_length is None:
                newNode = Node(name=newInternal)
            else:
                newNode = Node(branch_len={parent: node.branch_length}, name=newInternal)
            network.addNodes(newNode)
            return newNode

        extendedNewickParsedLabel = node.name.split("#")

        # if node already exists, just add its other parent and any other info 
        oldNode = network.hasNodeWithName(extendedNewickParsedLabel[-1])
        if oldNode != False:
            if oldNode.is_reticulation() and not called_as_parent:
                oldNode.add_length(node.branch_length, parent)
                print(node.comment)
                if node.comment is not None:
                    contents = oldNode.comment.split("=")
                    if "&gamma" == contents[0]:
                        if oldNode.attribute_value_if_exists("gamma") is not None:
                            oldNode.add_attribute("gamma", {parent.get_name(): float(contents[1])}, append = True)
                else:
                    print(oldNode.attribute_value_if_exists("gamma"))
                    print(parent.get_name())
                    if oldNode.attribute_value_if_exists("gamma") is not None:
                        if len(oldNode.attribute_value_if_exists("gamma").values()) != 1:
                            raise NetworkBuilderError("Gamma attribute malformed")
                        complement = 1- list(oldNode.attribute_value_if_exists("gamma").values())[0]
                        oldNode.add_attribute("gamma", {parent.get_name(): complement}, append = True)
                        
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
        if retValue:
            newNode = Node({parent: node.branch_length}, name=extendedNewickParsedLabel[1], is_reticulation=retValue)
            newNode.add_attribute("eventType", eventType)
            newNode.add_attribute("index", num)
        else:
            newNode = Node({parent: node.branch_length}, name=extendedNewickParsedLabel[0])

        if node.comment is not None:
            contents = node.comment.split("=")
            if "&gamma" == contents[0]:
                print(parent.get_name())
                newNode.add_attribute("gamma", {parent.get_name(): float(contents[1])})
            else:
                newNode.add_attribute("comment", node.comment)
                print("adding to inheritance queue")
                self.inheritance_queue.add([newNode, parent])
            
        network.addNodes(newNode)
        return newNode

    def getNetwork(self, i):
        return self.networks[i]

    def get_all_networks(self):
        return self.networks

    def name_of_network(self, network):
        return self.name_2_net[network]


# nb = NetworkBuilder('src/test/NetworkBuilderTests/files/hybridization_with_gamma.nex')
# net = nb.getNetwork(0)
# net.printGraph()
