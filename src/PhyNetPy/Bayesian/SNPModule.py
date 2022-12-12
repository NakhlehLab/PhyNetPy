from math import sqrt

def partials_index(n:int) -> int:
    """
    Computes the starting index in computing a linear index for an (n,r) pair.
    Returns the index, if r is 0.
    
    i.e n=1 returns 0, since (1,0) is index 0
    i.e n=3 returns 5 since (3,0) is preceded by (1,0), (1,1), (2,0), (2,1), and (2,2)

    Args:
        n (int): an n value (number of lineages) from an (n,r) pair

    Returns:
        int: starting index for that block of n values
    """
    return int(.5 * (n - 1) * (n + 2))

def undo_index(num: int)->list:
    """
    Takes an index from the linear vector and turns it into an (n,r) pair
    
    i.e 7 -> [3,2]

    Args:
        num (int): the index

    Returns:
        list: a 2-tuple (n,r)
    """
    a = 1
    b = 1
    c = -2 - 2 * num
    d = (b ** 2) - (4 * a * c)
    sol = (-b + sqrt(d)) / (2 * a)
    n = int(sol)
    r = num - partials_index(n)

    return [n, r]

def map_nr_to_index(n:int, r:int) -> int:
    """
    Takes an (n,r) pair and maps it to a 1d vector index

    (1,0) -> 0
    (1,1) -> 1
    (2,0) -> 2
    ...
    """
    starts = int(.5 * (n - 1) * (n + 2))
    return starts + r