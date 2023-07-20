""" 
Author : Mark Kessler
Last Stable Edit : 7/16/23
First Included in Version : 0.1.0

"""



    
def dict_merge(a : dict, b: dict):
    merged = {}
    for key in set(a.keys()).union(set(b.keys())):
        if key in a.keys() and key in b.keys():
            result = [a[key]]
            result.extend([b[key]])
            merged[key] = result
        elif key in a.keys():
            merged[key] = [a[key]]
        else:
            merged[key] = [b[key]]
    
    return merged

class NodeError(Exception):

    def __init__(self, message="Unknown Error in Node Class"):
        super().__init__(message)


class Node:
    """
    Node class that provides support for branch lengths, parent child relationships, and network constructs like reticulation 
    nodes and other phylogenetic attributes.
    """

    def __init__(self, branch_len:dict=None, parent_nodes=None, attr=None, is_reticulation=False, name=None):
        
        self.branch_lengths = branch_len
        
        if attr is None:
            self.attributes = {}
        else:
            self.attributes = attr
            
        self.is_retic = is_reticulation
        
        if parent_nodes is None:
            self.parent = []
        else:
            self.parent = parent_nodes
            
        self.label = name
        self.seq = None
        self.t = None
        self.is_dirty = False

    def length(self)->dict:
        """
        Returns:
            dict: The mapping from parent nodes to the branch lengths associated with them
        """
        return self.branch_lengths

    def asString(self):
        myStr = "Node " + str(self.label) + ": "
        if self.branch_lengths is not None:
            for branches in self.branch_lengths.values():
                if branches is not None:
                    for branch in branches:
                        if branch is not None:
                            myStr += str(round(branch, 4)) + " "
        if len(self.parent) > 0 :
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
        self.is_dirty = True

    def add_parent(self, par):
        """
        Add 'par' to the list of parent nodes for this node
        """

        #check for lousy input
        if type(par) is not Node:
            raise NodeError("Attempted to add a non node entity as a parent")

        if not par in self.parent:
            self.parent.append(par)
    
    def remove_parent(self, old_par):
        if old_par in self.parent:
            self.parent.remove(old_par)
        
            

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
        self.branch_lengths = {par: [length]}
    
    def add_length(self, new_len:float, new_par):
        """
        Add a branch length value to the node. 
        This node is a reticulation node.

        Args:
            new_len (float): a branch length value
        """
        if self.branch_lengths is None:
            self.branch_lengths = {new_par: [new_len]}
        else:
            if new_par in self.branch_lengths: # A BUBBLE
                self.branch_lengths[new_par].append(new_len)
            else:
                self.branch_lengths[new_par] = [new_len]

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

    def add_attribute(self, key, value, append = False, add = False):
        """
        Put a key and value pair into the node attribute dictionary.

        If the key is already present, it will overwrite the old value.
        """
        
        if append:
            if key in self.attributes.keys():
                content = self.attributes[key]
                # print(content)
                # print(value)
                
                if type(content) is dict:
                    if add:
                        content = dict_merge(content, value)
                    else:
                        content.update(value)
                    self.attributes[key] = content
                elif type(content) is list:
                    content.extend(value)
                    self.attributes[key] = content
        else:
            self.attributes[key] = value

    def attribute_value_if_exists(self, attr:str):
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
    

        


    