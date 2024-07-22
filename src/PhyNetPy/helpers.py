"""
General file for various functions that streamline actual code. This file is 
not for specific data structure manipulations, only for helpers that operate
on python structures like dicts, lists, sets, etc.

Returns:
    _type_: _description_
"""

import math
from typing import Union



def minmaxkey(mapping : dict[object, Union[int, float]], mini : bool = True) -> object:
    """
    Return the object in a mapping with the minimum or maximum value associated
    with it.

    Args:
        mapping (dict[object, int  |  float]): A mapping from objects to 
                                               numerical values

    Returns:
        object: The object with the minimimum or maximum value.
    """

    cur = math.inf
    cur_key = None
    if not mini:
        cur = cur * -1
        
    for key, value in mapping.items():
        if mini:
            if value < cur:
                cur = value
                cur_key = key
        else:
            if value > cur:
                cur = value
                cur_key = key
    
    return cur_key