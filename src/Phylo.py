"""
Author : Mark Kessler
Last Edit : 5/12/26
First Included in Version : 0.3.0

Core phylogenetic data structures shared across PhyNetPy.

Provides the :class:`Branch` class which stores the length, inheritance
probability, and parent identity for a single branch (edge) in a
phylogenetic tree or network.
"""

import warnings 


class Branch:
    """
    A class that represents a branch in a phylogenetic network.
    """
    def __init__(self, length: float = None, inheritance_probability: float = None, parent_id : str = None) -> None:
        """
        Initialize a Branch object.
        """
        self.length = length
        self.inheritance_probability = inheritance_probability
        self.parent_id = parent_id
        
    
    def __len__(self) -> float:
        """
        Get the length of the branch.
        """
        return self.length

    @property
    def length(self) -> float:
        """
        Get the length of the branch.
        """
        return self._length
    
    @length.setter
    def length(self, value: float) -> None:
        """
        Set the length of the branch.
        """
        if value is not None and value < 0:
            warnings.warn("Branch length cannot be negative, the length will not be changed")
            return

        self._length = value
    
    @property
    def inheritance_probability(self) -> float:
        """
        Get the inheritance probability of the branch.
        """
        return self._inheritance_probability
    
    @inheritance_probability.setter
    def inheritance_probability(self, value: float) -> None:
        """
        Set the inheritance probability of the branch.
        """
        if value is not None and (value < 0 or value > 1):
            warnings.warn("Inheritance probability must be between 0 and 1, the inheritance probability will not be changed")
            return
        self._inheritance_probability = value
    
    @property
    def parent_id(self) -> str:
        """
        Get the parent id of the branch.
        """
        return self._parent_id
    
    @parent_id.setter
    def parent_id(self, value: str) -> None:
        """
        Set the parent id of the branch.
        """
        self._parent_id = value
    
    def __str__(self) -> str:
        """
        Return a string representation of the branch.
        """
        return f"Branch(length={self.length}, inheritance_probability={self.inheritance_probability})"
    
    def __repr__(self) -> str:
        """
        Return a string representation of the branch.
        """
        return self.__str__()