import copy
import math


class NodeError(Exception):

    def __init__(self, message="Unknown Error in Node Class"):
        super().__init__(message)


class Node:
    """
        Node class for storing data in a graph object
    """

    def __init__(self, branch_len:dict=None, parent_nodes=None, attr=None, is_reticulation=False, name=None):
        self.branch_lengths = branch_len
        if attr is None:
            self.attributes = {}
        else:
            self.attributes = attr
        self.is_retic = is_reticulation
        self.parent = parent_nodes
        self.label = name
        self.seq = None

    def length(self)->dict:
        """
            The length of a node is its branch length.

            Returns: A list of branch lengths
        """
        return self.branch_lengths

    def asString(self):
        myStr = "Node " + str(self.label) + ": "
        if self.branch_lengths is not None:
            for branch in self.branch_lengths.values():
                if branch is not None:
                    myStr += str(round(branch, 4)) + " "
        if self.parent is not None:
            myStr += " has parent(s) " + str([node.get_name() for node in self.get_parent(return_all=True)])

        myStr += " is a reticulation node? " + str(self.is_retic)
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

    def set_length(self, length:float, par):
        """
        Set the branch length of this Node to length
        """
        self.branch_lengths = {par: length}
    
    def add_length(self, new_len:float, new_par):
        """
        Add a branch length value to the node. 
        This node is a reticulation node.

        Args:
            new_len (float): a branch length value
        """
        if self.branch_lengths is None:
            self.branch_lengths = {new_par: new_len}
        else:
            self.branch_lengths[new_par] = new_len

    def set_is_reticulation(self, is_retic):
        """
        Sets whether a node is a reticulation Node (or not)
        """
        self.is_retic = is_retic

    def is_reticulation(self):
        """
        Retrieves whether a node is a reticulation Node (or not)
        """
        return self.is_retic

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
    
    def set_seq(self, sequence : str):
        self.seq = sequence
    
    def get_seq(self)->str:
        return self.seq


