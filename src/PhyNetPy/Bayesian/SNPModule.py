from math import sqrt, comb
import numpy as np

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


# class PartialSnappLikelihood:
    
#     def __init__(self) -> None:
#         self.top = 
#         self.bottom = 

def Rule1(F_b : dict, site_count : int, vector_len : int, m_y : int, Qt : np.ndarray):
    F_t = {}
            
    # Do this for each marker
    for site in range(site_count):
        for ft_index in range(0, vector_len):
            tot = 0
            actual_index = undo_index(ft_index)
            n_t = actual_index[0]
            r_t = actual_index[1]

            for n_b in range(n_t, m_y + 1):  # n_b always at least 1
                for r_b in range(0, n_b + 1):
                    index = map_nr_to_index(n_b, r_b)
                    exp_val = Qt[index][ft_index]  # Q(n,r);(n_t, r_t)

                    tot += exp_val * F_b[index][site]

            F_t[(n_t, r_t, site)] = tot
            
            
def Rule2(F_t_y : dict, F_t_z : dict, site_count : int, vector_len : int) -> dict:
    
    F_b = {}
    
    for site in range(site_count):
        for index in range(vector_len):
            actual_index = undo_index(index)
            n = actual_index[0]
            r = actual_index[1]
            tot = 0

            # EQUATION 19
            for n_y in range(1, n):
                for r_y in range(0, r + 1):
                    if r_y <= n_y and r - r_y <= n - n_y:  # Ensure that the combinatorics makes sense
                        # Compute the constant term
                        const = comb(n_y, r_y) * comb(n - n_y, r - r_y) / comb(n, r)

                        # Grab Ftz(n_y, r_y)
                        term1 = F_t_z[(n_y, r_y, site)]

                        # Grab Fty(n - n_y, r - r_y)
                        term2 = F_t_y[(n - n_y, r - r_y, site)]

                        tot += term1 * term2 * const

            F_b[(n, r, site)] = tot
    return F_b

def Rule3(F_t_x : dict, g_this : float, g_that : float, site_count : int):

    #Get the other branch
    # sibling_branches = node_par.get_branches()
    # if sibling_branches[0] == self:
    #     sibling_branch : BranchNode = sibling_branches[1]
    # else:
    #     sibling_branch : BranchNode = sibling_branches[0]

    # g_this = self.inheritance_probability()
    # g_that = sibling_branch.inheritance_probability()

    # if g_this + g_that != 1:
    #     raise ModelError("Set of inheritance probabilities do not sum to 1 for node<" + node_par.name + ">")

    for site in range(site_count):
        pass
        
def Rule4():
    pass
        



def SNAPPNET_TEST():
    