import cProfile
from ete3 import Tree
import random
import numpy
import statistics
import math
import scipy

__seq_dict = {} # A global dictionary with 'sequence number : sequence' pairs. Populated by 'growtree()'.
__seq_counter = 0 # A global counter for sequence number. Each cell has a unique number. Incremented by 'growtree()'.
__lineage_dict = {} # A global dictionary keeping track every extant lineage (TreeNode objects) and their associated rates
#__goal_leaves = 0 # The number of leaves a simulated tree should reach before stopping
__curr_lineages = 1 # A global counter of the current number of extant lineages (leaves) in a growing simulated tree
__sub_silent_rate = .9
__sub_nonsilent_rate = .14
__sub_lethal_rate = .01

__sub_multiple_rate = .9

class SimulationError(Exception):
    def __init__(self, message = "Something went wrong with the simulation")->None:
        self.message = message
        super().__init__(self.message)
        

def gen_sequence(length: int, off_lim:str = None) -> numpy.ndarray:
    """
    Randomly generates a 'length' long genetic sequence of bases. 'off_lim' is by default 'None', but can be used
    to specify a base that is not allowed to appear in the sequence (e.g. upon a substitution, the base that was 
    previously in the substitution site is not a valid newly substituted base).
    """

    if(off_lim == None): # no restrictions on bases
        weights = [.25, .25, .25, .25]
    elif (off_lim == "A"): # sequence cannot include 'A'
        weights = [0, 1/3, 1/3, 1/3]        
    elif(off_lim == "C"): # sequence cannot include 'C'
        weights = [1/3, 0, 1/3, 1/3]       
    elif(off_lim == "G"): # sequence cannot include 'G'
        weights = [1/3, 1/3, 0, 1/3]      
    else: #sequence cannot include 'T'
        weights = [1/3, 1/3, 1/3, 0]
    
    seq = numpy.random.choice(["A", "C", "G", "T"] , length, weights)
    return seq

def gen_event(lin_dict:dict)->list:
    """
    Randomly selects a live lineage to perform a birth, death, or substitution event
    on, based on the weighted probabilities of each.

    Args:
        lin_dict (dict): should only be the __lineage_dict item.

    Returns:
        list: a 2-tuple, the first element is the lineage key, the second is a string
              that denotes the type of event (birth, death, or substitution)
    """
    lineage = random.choice(list(lin_dict.keys()))
    rates = lin_dict[lineage]
    
    selection = random.random()
    
    if sum(rates) == 0:
        raise SimulationError("Sum of the rates is 0")
    
    normalized_rates = [rate / sum(rates) for rate in rates]
    
    if normalized_rates[2] == 1:
        raise SimulationError("Sub rate blow up")
    
    debug = False
    if debug and normalized_rates[2] > .99:
        print("CURRENT SUB RATE: " + str(normalized_rates[2]))
    
    if selection < normalized_rates[0]:
        return [lineage, "birth"]
    elif selection < normalized_rates[0] + normalized_rates[1]:
        return [lineage, "death"]
    else:
        return [lineage, "sub"]
    
    
def gen_rate(mean, sd):
    """
    Samples a new rate based on a gamma distribution given the mean rate and the shape of the distribution. 
    """
    log_mean = math.log(mean)
    #print(log_mean)
    new_rate = numpy.random.lognormal(mean=log_mean, sigma=sd)
    #print(new_rate)
   
    return new_rate

def gen_sub_type(sub_arr):
    """
    silent_rate = sub_arr[0]
    non_silent_rate = sub_arr[1]
    lethal_rate = sub_arr[2]
    """
    
    rates = sub_arr[:3]
    multiple_sub_rate = sub_arr[3]
    seq_length = sub_arr[4]

    selection = random.random()
    
    if sum(rates) == 0:
        raise SimulationError("Sum of the rates is 0")
    
    normalized_rates = [rate / sum(rates) for rate in rates]

    if selection < normalized_rates[0]: # silent sub
        number_subs = 1
        if(multiple_sub_rate*100 >= random.choice(range(1, 101))):
            number_subs = random.choice(range(2, round(seq_length*.25)))
        return ["silent", number_subs]
    elif selection < normalized_rates[0] + normalized_rates[1]: # non-silent sub
        number_subs = 1
        if(multiple_sub_rate*100 >= random.choice(range(1, 101))):
            number_subs = random.choice(range(2, round(seq_length*.1)))
        return ["nonsilent", number_subs]
    else: # lethal sub
        return ["lethal", 1]

def tree_nleaf(t):
    # """
    # Returns number of leaves in a tree.
    # """
    if t is not None: # empty tree
        if(t.is_leaf()): # node is leaf so increment
            return 1
        num_leaves = 0
        num_c = len(t.children)  
        if(num_c == 1): # tree with 1 child
            num_leaves = tree_nleaf(t.children[0]) 
        elif(num_c == 2): # tree with 2 children
            num_leaves = tree_nleaf(t.children[0]) + tree_nleaf(t.children[1]) # add leaves of both children
        return num_leaves
    else:
        return 0

def tree_height(t): 
    """
    Returns the height (maximum depth) of the tree. 
    """
    if t == None:
        return 0 
    left_h = 0
    right_h = 0
    num_c = len(t.children)  
    if(num_c == 1): # tree with 1 child
        left_h = tree_height(t.children[0]) 
    elif(num_c == 2): # tree with 2 children
        left_h = tree_height(t.children[0]) 
        right_h = tree_height(t.children[1]) 
    return max(left_h, right_h) + t.dist

def growtree(seq, b, d, s, b_sd, d_sd, max_leaves, branch_info, sub_array):
    """
    Returns a birth-death tree. Used as a recursive helper function for 'gen_tree()' that produces
    the birth-death tree. Populates '__seq_dict' with 'sequence number : sequence' pairs. 
    """
    global __seq_counter 
    global __lineage_dict
    global __curr_lineages
    global __seq_dict
    global __sum_dict
    
    rng = random.Random()

    # initializing the tree and branch length
    t = Tree()
    key = "SEQUENCE_" + str(__seq_counter) # create sequence number key using the global counter
    t.name = key # TreeNode's name is the sequence number ID
    __seq_dict[key] = seq # set the 'sequence number : sequence' pair in the '__seq_dict' dictionary
    __lineage_dict[t] = [b, d, s] # add new TreeNode and its associated rates into the lineage dictionary
    t.dist = 0
    __sum_dict = sum(__lineage_dict[t])
    
    infinite_sub_checker = 0

    # finding the wait time to any event (b, d, or s) based on rates
    curr_time = 0
    total_iter = 0
    number_of_leaves = 1
    print(f"CUR LINEAGE CT: {__curr_lineages} --- MAX_LEAVES : {max_leaves}")
    while(__curr_lineages <= max_leaves):
        
        total_iter += 1
        if (__curr_lineages == 0):
            return t
        
        rate_any_event = __sum_dict # sum_dict(__lineage_dict) # sum of all the rates for all extant lineages
        wait_time = rng.expovariate(rate_any_event)
        curr_time += wait_time
        try:
            event_pair = gen_event(__lineage_dict)
        except:
            return t

        event_lineage_key = event_pair[0]
        # 'event' holds the event that just occurred as a string, 'curr_t' holds the TreeNode object (lineage) on which the event occurred
        event = event_pair[1]
        curr_t = event_lineage_key
        

        # extracting the attributes associated with the lineage of interest (where the event will occur)
        curr_seq = __seq_dict[curr_t.name] # getting the current lineage's sequence
        curr_rates_lst = __lineage_dict[curr_t] # getting the rates associated with the current lineage
        curr_b = curr_rates_lst[0]
        curr_d = curr_rates_lst[1]
        curr_s = curr_rates_lst[2]
        # if branch length is a variable of time, add 'wait_time' onto this lineage's branch length
        if(branch_info == 0): 
            curr_t.dist += wait_time
        # if branch length is a variable of expected number of substitutions, add this expected number onto this lineage's branch length
        if(branch_info == 2): 
            curr_t.dist += curr_s * wait_time
       
    
        if(event == "birth"): # recursively call fn for children, same rates but different max_time
            __curr_lineages += 1 # increase the number of extant lineages by 1 (1 extant linage turns into 2)
        
            # Below is creating the child tree 'c1' and setting its attributes
            __seq_counter += 1 # increment global counter so that each child has a unique sequence number
            c1 = Tree() # create a child tree
            key = "SEQUENCE_" + str(__seq_counter) # create sequence number key using the global counter
            c1.name = key
            __seq_dict[key] = curr_seq # set the 'sequence number : sequence' pair in the '__seq_dict' dictionary
            __lineage_dict[c1] = [curr_b, curr_d, curr_s] # set the child tree's associated rates (same as parent)
            c1.dist = 0

            # Below is creating the child tree 'c2' and setting its attributes
            __seq_counter += 1 # increment global counter so that each child has a unique sequence number
            c2 = Tree() # create a child tree
            key = "SEQUENCE_" + str(__seq_counter) # create sequence number key using the global counter
            c2.name = key
            __seq_dict[key] = curr_seq # set the 'sequence number : sequence' pair in the '__seq_dict' dictionary
            __lineage_dict[c2] = [curr_b, curr_d, curr_s] # set the child tree's associated rates (same as parent)
            c2.dist = 0
            
            # update sum of rates
            __sum_dict += sum(__lineage_dict[c2])
            __sum_dict += sum(__lineage_dict[c1])
            __sum_dict -= sum(__lineage_dict[curr_t])

            # Add children onto tree and delete parent lineage from the extant lineages dictionary (no longer a valid lineage)
            curr_t.add_child(c1)
            curr_t.add_child(c2)
            del __lineage_dict[curr_t]
            number_of_leaves += 1 # 1 lineage which was a leaf is now 2 leaves
            
            
        elif(event == "sub"): # change current rates based on sampling from a gamma distribution and continue to next event
            infinite_sub_checker += 1
            sub_event = gen_sub_type(sub_array)
            sub_event_type = sub_event[0]
            number_subs = sub_event[1]
            #print(sub_event_type)
            if(sub_event_type != "lethal"):
                
                if(sub_event_type == "nonsilent"):            
                    # mean of gamma distribution is current rate
                    curr_b = gen_rate(curr_b, b_sd) # generate new birth rate
                    curr_d = gen_rate(curr_d, d_sd) # generate new death rate
                    
                    
                    # update sum of rates
                    __sum_dict += sum([curr_b, curr_s, curr_d])
                    __sum_dict -= sum(__lineage_dict[event_lineage_key])
                    
                    __lineage_dict[event_lineage_key] = [curr_b, curr_d, curr_s]
                
                if(number_subs == 1):
                    sub_site = random.randint(0, len(curr_seq) - 1) # randomly pick a site to sub a base in the sequence
                    old_letter = curr_seq[sub_site : sub_site + 1] # find old base at this site so that the sub does not change the site to the same base
                    sub_letter = gen_sequence(1, off_lim = old_letter) # generate a new base for this site that is not the old base (not 'old_letter')
                    # generate the new sequence using the old sequence with one base changed from 'old_letter' to 'new_letter'
                    # at index 'sub_site' in the sequence
                    curr_seq[sub_site] = sub_letter[0]
                else:
                    sub_site = random.randint(0, len(curr_seq) - number_subs) # randomly pick a site to sub a base in the sequence
                    #old_letters = curr_seq[sub_site : sub_site + number_subs] # find old base at this site so that the sub does not change the site to the same base
                    sub_letters = gen_sequence(number_subs) # generate a new base for this site that is not the old base (not 'old_letter')
                    # generate the new sequence using the old sequence with one base changed from 'old_letter' to 'new_letter'
                    # at index 'sub_site' in the sequence
                    curr_seq[sub_site : sub_site + number_subs] = sub_letters
                
                #__seq_dict[event_lineage_key] = curr_seq # update the 'sequence number : sequence' pair in the '__seq_dict' dictionary
                
                # if branch length is a variable of number of substitutions, increase lineage's branch length by 1
                #print("sub")
                if(branch_info == 1):
                    #print(curr_t.dist)
                    curr_t.dist += number_subs
                    #print(curr_t.dist)
            else:
                #print("LETHAL")
                event = "death"
            
           
        if(event == "death"): # event is death so return None (lineage goes extinct)
            #print("death triggred")
            # update sum of rates
            __sum_dict -= sum(__lineage_dict[event_lineage_key])
            del __lineage_dict[event_lineage_key] # remove this lineage from the extant lineage dictionary
            __curr_lineages -= 1 # the number of extant lineages in the tree decreases by 1 (this one died)
            node_to_remove = curr_t
            curr_parent = curr_t.up
            if(curr_parent != None):
                while(len(curr_parent.children) < 2):
                    node_to_remove = curr_parent
                    curr_parent = curr_parent.up
                    if(curr_parent == None):
                        break

            node_to_remove.detach()


    #pr.disable()
    #pr.print_stats()
    #print(total_iter)
    #print(tree_nleaf(t))
        #if infinite_sub_checker > 50000:
            #raise SimulationError("INFINITE SUB LOOP")
    return t








        
def gen_tree(b, d, s, sd_b, sd_d, branch_info, seq_length, goal_leaves, sampling_rate):
    """
    Returns a birth-death tree. All rates (birth, death, and substitution) may change upon a substitution.
    'b', 'd', and 's' are the initial values of the birth, death, and substitution rates (respectively).
    'goal_leaves' is the amount of leaves the simulated tree should reach before terminating. Thus the tree 
    stops growing when either all lineages go extinct or when it has the same number of leaves as 'goal_leaves'.
    Only extant lineages are present in the final tree. If there are no extant lineages, 'None' will be returned 
    by the function. 'branch_info' is an argument to specify what information is attached to branches. If 
    'branch_info' is 0, the branch length is a variable of the time for that lineage. If 'branch_info' is 1, 
    the branch length is a variable of the number of substitutions that occurred in that lineage. If 'branch_info' 
    is 2, the branch length is a variable of the expected number of substitutions for that lineage. 'shape_b', 
    'shape_d', and 'shape_s' are the shapes of the respective gamma distributions from which each rate is sampled 
    from upon a substitution. 'seq_length' specifies the length of the genetic sequence for the cells 
    (the root will have a randomly generated sequence of length 'seq_length' and subsequent lineages will carry 
    on this sequence, with modifications upon a substitution). Translations from sequence number to sequence can 
    be found in '__seq_dict' (a dictionary populated with 'sequence number : sequence' pairs).
    """
    global __seq_counter 
    global __lineage_dict
    global __curr_lineages
    global __seq_dict
    global __sub_silent_rate
    global __sub_nonsilent_rate
    global __sub_lethal_rate
    global __sub_multiple_rate
    
    __seq_dict = {} 
    __seq_counter = 0
    __lineage_dict = {} 
    __curr_lineages = 1 

    b_sd = sd_b
    d_sd = sd_d

    sub_array = [__sub_silent_rate, __sub_nonsilent_rate, __sub_lethal_rate, __sub_multiple_rate, seq_length]
    
    seq = gen_sequence(seq_length) # generate random genetic sequence for root cell 
    #print("branch info", branch_info)
    t = growtree(seq, b, d, s, b_sd, d_sd, goal_leaves/sampling_rate, branch_info, sub_array) # generate the tree 
    # reset all global vars before constructing another tree
    
    print(tree_height(t))
    #print(tree_nleaf(t))
    #print(__curr_lineages)
    return t

def getNewick(t):
    """
    Returns a tree in Newick tree format.
    """
    if (t != None):
        return t.write()
    return ";"

def outputNewick(t, name):
    """
    Writes a tree, 't', in Newick tree format into a file. 'name' specifies the 
    file's name in which the tree is written into. If the tree is empty (i.e. if 
    't' is 'None') no output file is created.
    """
    if (t != None):
        t.write(outfile = name + ".nw")
    else:
        print("Empty tree, no output file created.")

def get_seq(seq_num = None):
    """
    Returns a genetic sequence for a cell in the simulated phylogeny (simulated using 'gen_tree'). 
    If 'seq_num' is None (which is the default), the whole 'sequence number : sequence' dictionary
    is returned. Otherwise, 'seq_num' can be initialized to an integer argument for the specific
    sequence corresponding to that sequence number to be returned.
    """
    if(seq_num == None): # 'seq_num' is not initialized so return whole dictionary
        return __seq_dict
    else:
        key = "SEQUENCE_" + str(seq_num) # find the specific key corresponding to 'seq_num'
        value = __seq_dict.get(key) # find the sequence in the dictionary corresponding to the key
        if(value == None): # no sequence exists corresponding to 'key'
            print(key, "does not exist.")
            return None
        return value # return this specific sequence

def print_seq(seq_num = None):
    """
    Prints a genetic sequence for a cell in the simulated phylogeny (simulated using 'gen_tree'). 
    If 'seq_num' is None (which is the default), the whole 'sequence number : sequence' dictionary
    is printed in FASTA format. Otherwise, 'seq_num' can be initialized to an integer argument 
    for the specific sequence corresponding to that sequence number to be printed in FASTA format
    (i.e. the single 'sequence number : sequence' pair is printed).
    """
    if(seq_num == None): # 'seq_num' is not initialized so print whole dictionary in FASTA format
        for key, value in __seq_dict.items():
            print('>', key, '\n', value)
    else:
        key = "SEQUENCE_" + str(seq_num) # find the specific key corresponding to 'seq_num'
        value = __seq_dict.get(key) # find the sequence in the dictionary corresponding to the key
        if(value == None): # no sequence exists corresponding to 'key'
            print(key, "does not exist.")
        else:
            print('>', key, '\n', value) # print this 'key' : 'value' pair in FASTA format

#################### TREE STATISTICS ########################################

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