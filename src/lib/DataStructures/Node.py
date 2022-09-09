import copy
import math


class NodeError(Exception):

    def __init__(self, message="Unknown Error in Node Class"):
        super().__init__(message)


class Node:
    """
        Node class for storing data in a graph object
    """

    def __init__(self, branch_len=None, parent_nodes=None, attr=None, is_reticulation=False, name=None):
        self.branch_length = branch_len
        self.tempLen = None
        self.attributes = attr
        self.is_reticulation = is_reticulation
        self.parent = parent_nodes
        self.label = name

    def length(self):
        """
            The length of a node is its branch length.

            Returns: The branch length, a float
        """

        return self.branch_length

    def asString(self):
        myStr = "Node " + str(self.label) + ": "
        if self.branch_length is not None:
            myStr += str(round(self.branch_length, 2)) + " "
        if self.parent is not None:
            myStr += " has parent(s) " + str([node.get_name() for node in self.get_parent(return_all=True)])

        myStr += " is a reticulation node? " + str(self.is_reticulation)
        myStr += " has attributes: " + str(self.attributes)

        return myStr

    def get_name(self):
        """
        Returns the name of the node
        """
        return self.label

    def set_name(self, new_name):
        """
        Sets the name of the node to new_name.
        """
        self.label = new_name

    def add_parent(self, par):
        """
        Add 'par' to the list of parent nodes for this node
        """

        #check for lousy input
        if type(par) is not Node:
            raise NodeError("Attempted to add a non node entity as a parent")

        if self.parent is not None:
            new_parent = copy.deepcopy(self.parent)
            new_parent.append(par)
            self.parent = new_parent
        else:
            self.parent = [par]

    def get_parent(self, return_all=False):
        """
        Retrieve either the one/first parent or the whole list of parents

        If return_all is set to True, then the method will return the whole array.
        The default behavior is just to return one.

        Returns: Either a node obj, or a list of node objs
        """
        if return_all:
            return self.parent
        else:
            return self.parent[0]

    def set_parent(self, new_parents):
        """
        Set the parent array to new_parents, a list of Node objs
        """
        self.parent = list(new_parents)

    def set_length(self, length):
        """
        Set the branch length of this Node to length
        """
        self.branch_length = length

    def set_is_reticulation(self, is_retic):
        """
        Sets whether a node is a reticulation Node (or not)
        """
        self.is_reticulation = is_retic

    def is_reticulation(self):
        """
        Retrieves whether a node is a reticulation Node (or not)
        """
        return self.is_reticulation

    def add_attribute(self, key, value):
        """
        Put a key and value pair into the node attribute dictionary.

        If the key is already present, it will overwrite the old value.
        """
        self.attributes[key] = value

    def attribute_value_if_exists(self, attr):
        """
        If attr is a key in the attributes mapping, then
        its value will be returned.

        Otherwise, returns None.
        """
        if attr in self.attributes:
            return self.attributes[attr]
        else:
            return None


class UltrametricNode(Node):

    def __init__(self, height=None, par=[], attributes={}, isRetNode=False, label=None):
        self.height = height
        super().__init__(parent_nodes=par, attr=attributes, is_reticulation=isRetNode, name=label)

    def length(self, otherNode):
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

        myStr += " is a reticulation node? " + str(self.is_reticulation)

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
