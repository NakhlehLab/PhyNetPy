from operator import index
import traceback
from nexus import NexusReader
from Bio import Phylo
from io import StringIO
from Graph import DAG
from Node import Node
import copy
from Node import NodeError

class NetworkBuilder2Error(Exception):
    def __init__(self, message = "Something went wrong with building the network") -> None:
        self.message = message
        super().__init__(self.message)


def merge_attributes(attr1 : dict, attr2 : dict) -> dict:
    """


    Args:
        attr1 (dict): _description_
        attr2 (dict): _description_

    Returns:
        dict: _description_
    """
    final_attr = dict()
    if "eventType" in attr1:
        final_attr["eventType"] = attr1["eventType"]
    if "index" in attr1:
        final_attr["index"] = attr1["index"]
        
    comment_set = set()
    if "comment" in attr1:
        comment_set.add(attr1["comment"])
    if "comment" in attr2:
        comment_set.add(attr2["comment"])

    if "gamma" in attr1:
        final_gamma = {}
        
        for key in attr1["gamma"]:
            if attr1["gamma"][key][0] == None:
                
                attr1["gamma"][key][0] = 1 - attr2["gamma"][list(attr2["gamma"].keys())[0]][0]
                
            if key in attr2["gamma"]:
                if attr2["gamma"][key][0] == None:
                    attr2["gamma"][key][0] = 1 - attr1["gamma"][list(attr1["gamma"].keys())[0]][0]
                gamma2 = attr2["gamma"][key]
                gamma1 = attr1["gamma"][key]
                
                value = [gamma1]
                value.extend([gamma2])
                final_gamma[key] = value
            else:
                final_gamma[key] = [attr1["gamma"][key]]
        for key in attr2["gamma"]:
            if attr2["gamma"][key][0] == None:
                attr2["gamma"][key][0] = 1 - attr1["gamma"][list(attr1["gamma"].keys())[0]][0]
            if key not in attr1["gamma"]:
                final_gamma[key] = [attr2["gamma"][key]]
        
        final_attr["gamma"] = final_gamma
    
    return final_attr
    
                
        
        
        
        
    
        
    

class NetworkBuilder2:

    def __init__(self, filename):
        try:
            self.reader = NexusReader.from_file(filename)
        except Exception as err:
            traceback.print_exc()
            raise NetworkBuilder2Error()
        
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
            raise NetworkBuilder2Error("There are no trees listed in the file")

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
            parentNode = self.parseParent(par, net)
            childNode = self.parseChild(node, net, parentNode)
            #
            childNode.add_parent(parentNode)
            #
            edges.append([parentNode, childNode])
        

        net.addEdges(edges, as_list=True)
        
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
        if attrStr[1][0] == "R":
            event = "Recombination"
            indexLookup = 1
        elif attrStr[1][0] == "H":
            event = "Hybridization"
            indexLookup = 1
        elif attrStr[1][0] == "L":
            try:
                if attrStr[1][1] == "G" and attrStr[1][2] == "T":
                    event = "Lateral Gene Transfer"
                    indexLookup = 3
            except:
                raise NodeError("Invalid label format string (event error)")
        else:
            raise NodeError("Invalid label format string (event error) ")

        # parse node index
        try:
            strnum = attrStr[1][indexLookup:]
            num = int(strnum)
            return event, num
        except:
            raise NodeError("Invalid label format string (number error)")

    def parseChild(self, node, network: DAG, parent : Node):

        if network.has_node_named(node.name):
            parsed_node : Node = network.has_node_named(node.name)
            if node.branch_length is not None:
                parsed_node.add_length(node.branch_length, parent)
            more_attr = self.parse_comment(node, parent)
            
            parsed_node.attributes = merge_attributes(more_attr, parsed_node.attributes)
            
        else:
            if node.name is None:
                newInternal = "Internal" + str(self.internalCount)
                self.internalCount += 1
                node.name = newInternal
                
            parsed_node = Node(name=node.name)
            if node.name[0] == "#":
                parsed_node.set_is_reticulation(True)
                
            parsed_node.add_length(node.branch_length, parent)
            parsed_node.attributes = self.parse_comment(node, parent)
            
        network.addNodes(parsed_node)
        return parsed_node
    
    def parseParent(self, node, network: DAG, parent : Node = None):
        
        if network.has_node_named(node.name):
            #No need to do anything
            return network.has_node_named(node.name)
        else:
            if node.name is None:
                newInternal = "Internal" + str(self.internalCount)
                self.internalCount += 1
                node.name = newInternal
                
            parsed_node = Node(name=node.name)
            if node.name[0] == "#":
                parsed_node.set_is_reticulation(True)
              
            if not(parent is None or node.branch_length is None):  
                parsed_node.add_length(node.branch_length, parent)
                
            parsed_node.attributes = self.parse_comment(node, parent)
            
        network.addNodes(parsed_node)
        return parsed_node

    def parse_comment(self, node, parent:Node):
        attr = {}
        if node.name[0] == "#":
            event, num = self.parseAttributes(node.name.split("#"))
            attr["eventType"] = event
            attr["index"] = num
            if parent is not None:
                attr["gamma"] = {parent.get_name(): [None, node.branch_length]}
        
        if node.comment is not None:
            if node.comment.split("=")[0] == "&gamma":
                if parent is not None:
                    attr["gamma"] = {parent.get_name(): [float(node.comment.split("=")[1]), node.branch_length]}
            else:
                attr["comment"] = node.comment
        
        return attr
            
        
    def getNetwork(self, i):
        return self.networks[i]

    def get_all_networks(self):
        return self.networks

    def name_of_network(self, network):
        return self.name_2_net[network]

# nb = NetworkBuilder2('src/PhyNetPy/test/files/paper_networks.nex')
# net:DAG = nb.getNetwork(0)
# net.printGraph()