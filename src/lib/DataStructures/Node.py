import copy
import math


class NodeError(Exception):

    def __init__(self, message="Unknown Error in Node Class"):
        super().__init__(message)


class Node:
    """
        Node wrapper class for storing data in a graph object

        propose/reject/accept structure not parallelizable at this time.
        """

    def __init__(self, branchLen=None, parNode=None, attr=None, isReticulation=False, name=None):
        self.branchLength = branchLen
        self.tempLen = None
        self.attributes = attr
        self.isReticulation = isReticulation
        self.parent = parNode
        self.label = name

    def addAttribute(self, key, value):
        """
                put a key and value pair into the node attribute dictionary
                """
        self.attributes[key] = value

    def propose(self, newValue):
        """
                Stores the current value in a temporary holder while the
                newValue gets tested for viability 
                """
        self.tempLen = self.branchLength
        self.branchLength = newValue

    def accept(self):
        """
                If the proposed change is good, accept it by flushing the data 
                out of the temp container, symbolically cementing .height as the 
                official height
                """
        self.tempLen = None

    def reject(self):
        """
                If the proposed change is bad, reset the official height to what it
                was (the contents of the temp container).

                Flush the temp container of all data.
                """

        self.branchLength = self.tempLen
        self.tempLen = None

    def branchLen(self):
        """
                Defines either the weight of the edge between this node and its parent, or
                simply the difference in time between two nodes.
                """

        return self.branchLength

    def asString(self):
        myStr = "Node " + str(self.label) + ": "
        if self.branchLength != None:
            myStr += str(round(self.branchLength, 2)) + " "
        if self.parent != None:
            myStr += " has parent(s) " + str([node.getName() for node in self.getParent(all=True)])

        myStr += " is a reticulation node? " + str(self.isReticulation)
        myStr += " has attributes: " + str(self.attributes)

        return myStr

    def getName(self):
        return self.label

    def setName(self, newName):
        self.label = newName

    def addParent(self, par):
        if self.parent is not None:
            newParent = copy.deepcopy(self.parent)
            newParent.append(par)
            self.parent = newParent
        else:
            self.parent = [par]

    def getParent(self, all=False):
        if all:
            return self.parent
        else:
            return self.parent[0]

    def setParent(self, newPar):
        self.parent = [newPar]

    def setBranchLength(self, length):
        self.branchLength = length

    def setIsReticulation(self, bool):
        self.isReticulation = bool

    def isReticulation(self):
        return self.isReticulation

    def attrLookup(self, attr):
        if attr in self.attributes:
            return self.attributes[attr]
        else:
            return None


class UltrametricNode(Node):

    def __init__(self, height=None, par=[], attributes={}, isRetNode=False, label=None):
        self.height = height
        super().__init__(parNode=par, attr=attributes, isReticulation=isRetNode, name=label)

    def branchLen(self, otherNode):
        if (type(otherNode) != UltrametricNode):
            raise NodeError(
                "Attempting to gather a branch length between an Ultrametric Node and a non-Ultrametric Node")

        return math.abs(self.height - otherNode.getHeight())

    def getHeight(self):
        return self.height

    def asString(self):
        myStr = "Node " + str(self.label) + ": "
        if self.height != None:
            myStr += str(self.height) + " "
        if self.parent != None:
            myStr += " has parent " + str(self.parent.name)

        myStr += " is a reticulation node? " + str(self.isReticulation)

        return myStr

    def propose(self, newValue):
        """
                Stores the current value in a temporary holder while the
                newValue gets tested for viability 
                """
        self.tempHeight = self.height
        self.height = newValue

    def accept(self):
        """
                If the proposed change is good, accept it by flushing the data 
                out of the temp container, symbolically cementing .height as the 
                official height
                """
        self.tempHeight = None

    def reject(self):
        """
                If the proposed change is bad, reset the official height to what it
                was (the contents of the temp container).

                Flush the temp container of all data.
                """

        self.height = self.tempHeight
        self.tempHeight = None
