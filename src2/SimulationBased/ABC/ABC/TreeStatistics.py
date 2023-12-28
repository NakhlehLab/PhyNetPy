import statistics
import ete3
from Simulator import tree_nleaf, tree_height

def __tree_branch_lst(t, arr):
    """
    Returns an array of branch lengths. Private helper function used for calculating summary
    stats regarding branches.
    """
    if(t == None): # empty tree
        return []
    arr.append(t.dist) 
    num_c = len(t.children)  
    if(num_c == 1): # tree with 1 child
        __tree_branch_lst(t.children[0], arr) 
    elif(num_c == 2): # tree with 2 children
        __tree_branch_lst(t.children[0], arr) 
        __tree_branch_lst(t.children[1], arr) 
    return arr

def tree_branch_sum(t):
    """
    Returns the sum of the distances of all the branches in the tree. 
    """
    branch_arr = __tree_branch_lst(t, []) # get array of branch lengths
    if(branch_arr == []):
        return 0
    return sum(branch_arr)

def tree_branch_mean(t):
    """
    Returns the mean of the distances of all the branches in the tree. 
    """
    branch_arr = __tree_branch_lst(t,[ ]) # get array of branch lengths
    if(branch_arr == []):
        return 0
    return statistics.mean(branch_arr)

def tree_branch_median(t):
    """
    Returns the median of the distances of all the branches in the tree. 
    """
    branch_arr = __tree_branch_lst(t, []) # get array of branch lengths
    if(branch_arr == []):
        return 0
    return statistics.median(branch_arr)

def tree_branch_variance(t):
    """
    Returns the variance of the distances of all the branches in the tree. 
    """
    branch_arr = __tree_branch_lst(t, []) # get array of branch lengths
    if(branch_arr == [] or len(branch_arr) < 2):
        return 0
    return statistics.variance(branch_arr)



def __tree_root_dist(node):
    """
    Returns the distance from a node to the root. Private helper function used for calculating 
    summary stats regarding tree depth.
    """
    if node == None or node.up == None:
        return 0 
    return __tree_root_dist(node.up) + node.dist

def __tree_depth_lst(node, arr):
    """
    Returns an array of leaf depths. Private helper function used for calculating summary
    stats regarding tree depth.
    """
    if(node == None):
        return []
    if(node.is_leaf()): # if node is leaf, add it's depth to 'arr'
        arr.append(__tree_root_dist(node))
    else: # node is not leaf so recurse to find leaves
        num_c = len(node.children)  
        if(num_c == 1): # tree with 1 child
            __tree_depth_lst(node.children[0], arr) 
        elif(num_c == 2): # tree with 2 children
            __tree_depth_lst(node.children[0], arr) 
            __tree_depth_lst(node.children[1], arr) 
    return arr

def tree_depth_mean(t):
    """
    Returns the mean of leaf depths in the tree. 
    """
    depth_arr = __tree_depth_lst(t, []) # get array of leaf depths
    if(depth_arr == []):
        return 0
    return statistics.mean(depth_arr)

def tree_depth_median(t):
    """
    Returns the median of leaf depths in the tree. 
    """
    depth_arr = __tree_depth_lst(t, []) # get array of leaf depths
    if(depth_arr == []):
        return 0
    return statistics.median(depth_arr)

def tree_depth_variance(t):
    """
    Returns the variance of leaf depths in the tree. 
    """
    depth_arr = __tree_depth_lst(t, []) # get array of leaf depths
    if(depth_arr == [] or len(depth_arr) < 2):
        return 0
    return statistics.variance(depth_arr)

def __tree_internal_height_lst(node, arr):
    """
    Returns an array of the reciprocal of the heights (maximum depths) 
    of subtrees of t rooted at internal nodes of t (not including the root). 
    Private helper function used for calculating summary stats regarding 
    tree balance.
    """
    if(node == None):
        return []
    if(not(node.is_leaf()) and not(node.is_root())): # if node is internal (not root or leaf)
        height_subtree = tree_height(node)
        if(height_subtree != 0):
            arr.append(1/tree_height(node)) # add reciprocal of height of subtree rooted at 'node' to 'arr'
        else:
            arr.append(0)
    # must find all internal nodes to add reciprocal of heights to 'arr'
    num_c = len(node.children)  
    if(num_c == 1): # tree with 1 child
        __tree_internal_height_lst(node.children[0], arr) 
    elif(num_c == 2): # tree with 2 children
        __tree_internal_height_lst(node.children[0], arr) 
        __tree_internal_height_lst(node.children[1], arr) 
    return arr

def tree_balance(t):
    """
    Returns B1 balance index. This is the sum of the reciprocal of the 
    heights (maximum depths) of subtrees of t rooted at internal nodes 
    of t (not including the root).
    """
    height_arr = __tree_internal_height_lst(t, []) # get array of reciprocal subtree heights
    if(height_arr == []):
        return 0
    return sum(height_arr)

def __tree_leaf_diff(node):
    """
    Returns the absolute value of the difference between the number of leaves
    on the left side of the tree and the number of leaves on the right side.
    Private helper function used for calculating summary stats regarding the
    colless index.
    """
    if(node == None):
        return 0
    if(not(node.is_leaf())):
        nleaf_l = 0 # will hold number of leaves on left side of 'node'
        nleaf_r = 0 # will hold number of leaves on right side of 'node'
        num_c = len(node.children)  
        if(num_c == 1): # tree with 1 child
            nleaf_l = tree_nleaf(node.children[0]) 
        elif(num_c == 2): # tree with 2 children
            nleaf_l = tree_nleaf(node.children[0]) 
            nleaf_r = tree_nleaf(node.children[1])
        return abs(nleaf_l - nleaf_r)
    return 0

def __tree_leaf_diff_lst(node, arr):
    """
    Returns an array of absolute values of the differences between the number 
    of leaves on the left side of a node and the number of leaves on the right 
    side for all internal nodes including the root. Private helper function 
    used for calculating summary stats regarding the colless index.
    """
    if(node == None):
        return []
    if(not(node.is_leaf())): # node is internal (not a leaf)
        arr.append(__tree_leaf_diff(node)) # add leaf difference to array
    num_c = len(node.children)  
    if(num_c == 1): # tree with 1 child
        __tree_leaf_diff_lst(node.children[0], arr) 
    elif(num_c == 2): # tree with 2 children
        __tree_leaf_diff_lst(node.children[0], arr) 
        __tree_leaf_diff_lst(node.children[1], arr) 
    return arr

def tree_root_colless(t):
    """
    Returns the colless index for the root node of a tree. That is,
    returns the absolute value of the difference between the number of leaves
    on the left side of the tree and the number of leaves on the right side
    for the root node.
    """
    if(t == None):
        return 0
    return __tree_leaf_diff(t)

def tree_sum_colless(t):
    """
    Returns the sum of the colless indices for every internal node of a tree. 
    """
    if(t == None):
        return 0
    arr = __tree_leaf_diff_lst(t, []) # get array of colless indices
    if(arr == []):
        return 0
    return sum(arr)

def tree_mean_colless(t):
    """
    Returns the mean of the colless indices for every internal node of a tree. 
    """
    if(t == None):
        return 0
    arr = __tree_leaf_diff_lst(t, []) # get array of colless indices
    if(arr == []):
        return 0
    return statistics.mean(arr)

def tree_median_colless(t):
    """
    Returns the median of the colless indices for every internal node of a tree. 
    """
    if(t == None):
        return 0
    arr = __tree_leaf_diff_lst(t, []) # get array of colless indices
    if(arr == []):
        return 0
    return statistics.median(arr)

def tree_variance_colless(t):
    """
    Returns the variance of the colless indices for every internal node of a tree. 
    """
    if(t == None):
        return 0
    arr = __tree_leaf_diff_lst(t, []) # get array of colless indices
    if(arr == [] or len(arr) < 2):
        return 0
    return statistics.variance(arr)


def tree_stat(tree_arr, summ_fn):
    """
    Applies function 'summ_fn()' to every element of 'tree_arr' and returns
    the array of results. 'summ_fn()' is a summary function that takes in a single
    tree and returns a numerical value (e.g. tree height, mean branch length,
    colless index). 
    
    Statistics are weighted using 1/SD of the statistic (using the SD of the 
    simulated population). This weighting scheme was chosen based on Schalte et al.
    which describes that a weighted Minkowski distance (weight = 1/SD) is commonly 
    chosen to normalize statistics. 
    
    In ABC-SMC, when going beyond 1 generation, "the distribution of summary statistics 
    in later generations can differ considerably from prior samples" (Schalte et al.). 
    Because of this, adaptive distance functions (i.e. adaptive schemes) given in the 
    ELFI or pyABC packages may be used rather than a simple weighted Minkowski distance. 
    However, this is not necessary for our current implementation, given that we use only 
    1-2 generations in ABC-SMC and the distribution of summary statistics between generations 
    do not vary considerably. This is due to the fact that even from the first generation, 
    the considered simulations are conditioned on the number of goal_leaves, so within the 
    first generation, simulated trees look similar to the "true"/observed tree. In later 
    generations, the simulated trees' distributions of summary statistics may look slightly 
    different from the first generation's distributions, but due to this conditioning on 
    number of leaves, the simulated trees are generally much more homogenous in structure 
    (and thus in calculated statistics) than simulated trees not conditioned on number of leaves. 
    Thus having distributions of summary statistics in later generations that differ considerably 
    from the first generation is not an issue and using an adaptive distance scheme is not necessary.

    [Schalte et al.: https://www.biorxiv.org/content/10.1101/2021.07.29.454327v1.full.pdf]
    """
    obs_tree_stats = [] # holds (or will hold) the observed stats
    stat_index = 0 # keeps track of which stat is the current one

    res_arr = [] # array that will hold the summary statistic of the trees in 'tree_arr'
    for i in tree_arr: # for each tree in 'tree_arr'
        if(type(i) != ete3.coretype.tree.TreeNode): # if 'tree_arr' is an array of simulated trees
            calc_stat = summ_fn(i[0]) # calculate the summary statistic of current tree, 'i'
            """
            curr_obs_stat = obs_tree_stats[stat_index] # get statistic value for observed tree
            if(curr_obs_stat == 0): # if observed stat is 0, no need to normalize
                norm_stat = calc_stat 
            else: # normalize current statistic with observed statistic
                norm_stat = (calc_stat - curr_obs_stat) / curr_obs_stat 
            """
            res_arr.append(calc_stat) 
        else: # 'tree_arr' is a one element array containing only the observed tree ('obs')
            obs_stat = summ_fn(i.get_tree_root()) # calculate the summary statistic for 'obs' tree
            obs_tree_stats.append(obs_stat) # add observed statistic to array
            res_arr.append(0) 
            return res_arr
    stat_index = (stat_index + 1) % len(obs_tree_stats) # find new index of observed statistic
    
    sd = (statistics.pstdev(res_arr))
    if(sd == 0):
        sd = 1
        #print("FAIL: SD = 0")
        #print(res_arr)
    wres_arr = [x/sd for x in res_arr]
    return wres_arr # return array of summary statistics


"""
Below are the set of summary statistic functions that can be passed 
into 'tree_stat()'. Each function requires an array of trees ('tree_arr') 
and returns an array where the elements are the summary statistic calculated 
on each tree in 'tree_arr'.
"""
def branch_sum_stat(tree_arr):
    return tree_stat(tree_arr, tree_branch_sum)

def branch_mean_stat(tree_arr):
    return tree_stat(tree_arr, tree_branch_mean)

def branch_median_stat(tree_arr):
    return tree_stat(tree_arr, tree_branch_median)

def branch_variance_stat(tree_arr):
    return tree_stat(tree_arr, tree_branch_variance)

def height_stat(tree_arr):
    return tree_stat(tree_arr, tree_height)

def depth_mean_stat(tree_arr):
    return tree_stat(tree_arr, tree_depth_mean)

def depth_median_stat(tree_arr):
    return tree_stat(tree_arr, tree_depth_median)

def depth_variance_stat(tree_arr):
    return tree_stat(tree_arr, tree_depth_variance)

def balance_stat(tree_arr):
    return tree_stat(tree_arr, tree_balance)

def nleaves_stat(tree_arr):
   return tree_stat(tree_arr, tree_nleaf)

def root_colless_stat(tree_arr):
    return tree_stat(tree_arr, tree_root_colless)

def sum_colless_stat(tree_arr):
    return tree_stat(tree_arr, tree_sum_colless)

def mean_colless_stat(tree_arr):
    return tree_stat(tree_arr, tree_mean_colless)

def median_colless_stat(tree_arr):
    return tree_stat(tree_arr, tree_median_colless)

def variance_colless_stat(tree_arr):
    return tree_stat(tree_arr, tree_variance_colless)