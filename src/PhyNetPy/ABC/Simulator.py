import random
import elfi
from ete3 import Tree
import random
import numpy
import statistics
import math
from SimulatedTree import SimulatedTree
import numpy as np




class SimulationError(Exception):
    def __init__(self, message = "Something went wrong with the simulation")->None:
        self.message = message
        super().__init__(self.message)
        

def sample_leaves(tree, goal_leaves):
    sampling_rate_arr = [] #???
    curr_leaves = tree_nleaf(tree)

    sampling_rate = goal_leaves/curr_leaves
    sampling_rate_arr.append(sampling_rate)
    num_delete_goal = math.ceil(curr_leaves * (1-sampling_rate))
    leaf_lst = tree.get_leaves()

    deleted_leaf_lst = random.sample(leaf_lst,k=num_delete_goal)
    for leaf in deleted_leaf_lst:
        leaf.delete()

    return tree

def gen_tree_sims(d = 1, r = 0.5, sub_rate = 1, birth_shape = 1, death_shape = 1, sub_shape = 1, leaf_goal = 10, sampling_rate = 0.01, is_prior = False, random_state = None):
    """
    Returns a simulated phylogenetic tree (using growtree.gen_tree()) with the 
    initial diversification rate = 'd', initial turnover rate = 'r', initial 
    substitution rate = 1. Initial birth and death rates are calculated from
    the initial values for diversification and turnover (see 'gen_rates_bd()' 
    function above for the calculation). Initial shapes for the distributions 
    of rates for birth, death, and substitution are 'birth_shape', 'death_shape', 
    and 'sub_shape', respectively. The tree is returned in a one element array 
    in order to be compatible with the ELFI package. 'random_state' is not 
    currently used, but is included as a parameter since some ELFI functions 
    pass a value for 'random_state' into this function. Currently the value 
    '1' is being passed in for 'branch_info' (branch length is a variable of 
    the number of substitutions that occurred in that lineage) since this is
    the most descriptive for generating summary statistics that accurately 
    infer distribution shape parameters.
    """
    global d_dist
    global r_dist
    global sub_dist
    arr = []
    random_state = random_state or np.random # this value is not currently used
    if(is_prior): # using prior dist to simulate trees
        curr_nleaf = -9999999999
        while(curr_nleaf < leaf_goal):
            d_drawn = gen_param(d_dist)
            r_drawn = gen_param(r_dist)
            while(r_drawn>=1):
                r_drawn = gen_param(r_dist)
            s_drawn = gen_param(sub_dist)
            rate_arr = calc_rates_bd(d_drawn, r_drawn) # calculate the initial birth and death rates from 'd' and 'r'
            birth = rate_arr[0] # extract initial birth rate from result array
            death = rate_arr[1] # extract initial death rate from result array
            #TODO: there's hardcoded info here
            new_tree = ABCSimulator().growtree(100, sampling_rate, leaf_goal, birth, death, s_drawn, birth_shape, death_shape, sub_shape, 1).get_tree()
            curr_nleaf = tree_nleaf(new_tree)
            
        new_tree = sample_leaves(new_tree, leaf_goal)
        
    else: # use artificial true rates to simulate an observed tree
        rate_arr = calc_rates_bd(d, r) # calculate the initial birth and death rates from 'd' and 'r'
        birth = rate_arr[0] # extract initial birth rate from result array
        death = rate_arr[1] # extract initial death rate from result array
        new_tree = ABCSimulator().growtree(100, sampling_rate, leaf_goal, birth, death, sub_rate, birth_shape, death_shape, sub_shape, 1).get_tree()

    arr.append(new_tree) # simulate tree and place in 1 element array
    
    return arr
        

def calc_rates_bd(d, r):
    """
    Returns a two-element array containing the birth and death rate 
    calculated from the diversification rate, 'd', and the turnover 
    rate, 'r'. Note that (diversification = birth - death) and 
    (turnover = death / birth). Thus birth rate can be calculated by:
        birth = diversification / (1 - turnover)
    and death rate can be calculated by:
        death = turnover * birth    
    """
    birth_calc = d / (1 - r) # calculate birth rate from 'd' and 'r'
    death_calc = r * birth_calc # calculate death rate from calculated birth rate and 'r'
    return [birth_calc, death_calc] # return birth and death rates in an array

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
    
    debug = True
    if debug:
        print("CURRENT SUB RATE: " + str(normalized_rates[2]))
    
    if selection < normalized_rates[0]:
        return [lineage, "birth"]
    elif selection < normalized_rates[0] + normalized_rates[1]:
        return [lineage, "death"]
    else:
        return [lineage, "sub"]
    
    
def gen_rate(mean, shape):
    """
    Samples a new rate based on a gamma distribution given the mean rate and the shape of the distribution. 
    """
    scale_calc = mean / shape
    return numpy.random.gamma(shape, scale = scale_calc, size = None)

def gen_param(prior_dist):
    """
    Draws a single sample from 'prior_dist' and returns it (where 
    'prior_dist' is an object of the 'elfi.Prior' class). This 
    function is used for generating true parameters.
    """
    return (prior_dist.generate())[0] # draw sample and extract the value from a 1 element array


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

class ABCSimulator:
    
    
    def __init__(self) -> None:
        self.simulated_trees = []
    
    
    def growtree(self, seq_len: int, sampling_rate, goal_leaves, b, d, s, shape_b, shape_d, shape_s, branch_info) -> SimulatedTree:
        """
        Returns a birth-death tree. Used as a recursive helper function for 'gen_tree()' that produces
        the birth-death tree. Populates 'seq_dict' with 'sequence number : sequence' pairs. 
        """
        
        seq_counter = 0
        lineage_dict = {}
        curr_lineages = 1
        seq_dict = {}
        sum_rates = 0 
        max_leaves = math.ceil(goal_leaves / sampling_rate)
        seq = gen_sequence(seq_len)

        rng = random.Random()

        # initializing the tree and branch length
        t = Tree()
        key = "SEQUENCE_" + str(seq_counter) # create sequence number key using the global counter
        t.name = key # TreeNode's name is the sequence number ID
        seq_dict[key] = seq # set the 'sequence number : sequence' pair in the '__seq_dict' dictionary
        lineage_dict[t] = [b, d, s] # add new TreeNode and its associated rates into the lineage dictionary
        t.dist = 0
        sum_rates = sum(lineage_dict[t])
        
        infinite_sub_checker = 0

        # finding the wait time to any event (b, d, or s) based on rates
        curr_time = 0
        total_iter = 0
        number_of_leaves = 1
        
        while(curr_lineages <= max_leaves):

            total_iter += 1
            if (curr_lineages == 0):
                st = SimulatedTree(t, seq_dict, lineage_dict, sum_rates, curr_lineages)
                self.simulated_trees.append(st)
                return st
            
            rate_any_event = sum_rates # sum_dict(__lineage_dict) # sum of all the rates for all extant lineages
            wait_time = rng.expovariate(rate_any_event)
            curr_time += wait_time
            try:
                event_pair = gen_event(lineage_dict)
            except:
                st = SimulatedTree(t, seq_dict, lineage_dict, sum_rates, curr_lineages)
                self.simulated_trees.append(st)
                return st

            event_lineage_key = event_pair[0]
            # 'event' holds the event that just occurred as a string, 'curr_t' holds the TreeNode object (lineage) on which the event occurred
            event = event_pair[1]
            curr_t = event_lineage_key
            

            # extracting the attributes associated with the lineage of interest (where the event will occur)
            curr_seq = seq_dict[curr_t.name] # getting the current lineage's sequence
            curr_rates_lst = lineage_dict[curr_t] # getting the rates associated with the current lineage
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
                seq_counter += 1 # increment global counter so that each child has a unique sequence number
                c1 = Tree() # create a child tree
                key = "SEQUENCE_" + str(seq_counter) # create sequence number key using the global counter
                c1.name = key
                seq_dict[key] = curr_seq # set the 'sequence number : sequence' pair in the '__seq_dict' dictionary
                lineage_dict[c1] = [curr_b, curr_d, curr_s] # set the child tree's associated rates (same as parent)
                c1.dist = 0

                # Below is creating the child tree 'c2' and setting its attributes
                seq_counter += 1 # increment global counter so that each child has a unique sequence number
                c2 = Tree() # create a child tree
                key = "SEQUENCE_" + str(seq_counter) # create sequence number key using the global counter
                c2.name = key
                seq_dict[key] = curr_seq # set the 'sequence number : sequence' pair in the '__seq_dict' dictionary
                lineage_dict[c2] = [curr_b, curr_d, curr_s] # set the child tree's associated rates (same as parent)
                c2.dist = 0
                
                # update sum of rates
                sum_rates += sum(lineage_dict[c2])
                sum_rates += sum(lineage_dict[c1])
                sum_rates -= sum(lineage_dict[curr_t])

                # Add children onto tree and delete parent lineage from the extant lineages dictionary (no longer a valid lineage)
                curr_t.add_child(c1)
                curr_t.add_child(c2)
                del lineage_dict[curr_t]
                number_of_leaves += 1 # 1 lineage which was a leaf is now 2 leaves
                
                
            elif(event == "sub"): # change current rates based on sampling from a gamma distribution and continue to next event
                
                infinite_sub_checker += 1
                # mean of gamma distribution is current rate
                curr_b = gen_rate(curr_b, shape_b) # generate new birth rate
                curr_d = gen_rate(curr_d, shape_d) # generate new death rate
                curr_s = gen_rate(curr_s, shape_s) # generate new sub rate
                
                # update sum of rates
                sum_rates += sum([curr_b, curr_s, curr_d])
                sum_rates -= sum(lineage_dict[event_lineage_key])
                
                lineage_dict[event_lineage_key] = [curr_b, curr_d, curr_s]
                
                sub_site = random.randint(0, len(curr_seq) - 1) # randomly pick a site to sub a base in the sequence
                old_letter = curr_seq[sub_site] # find old base at this site so that the sub does not change the site to the same base
                sub_letter = gen_sequence(1, off_lim = old_letter) # generate a new base for this site that is not the old base (not 'old_letter')
                # generate the new sequence using the old sequence with one base changed from 'old_letter' to 'new_letter'
                # at index 'sub_site' in the sequence
                curr_seq[sub_site] = sub_letter[0]
                
                #__seq_dict[event_lineage_key] = curr_seq # update the 'sequence number : sequence' pair in the '__seq_dict' dictionary
                
                # if branch length is a variable of number of substitutions, increase lineage's branch length by 1
                if(branch_info == 1):
                    curr_t.dist += 1
            
            else: # event is death so return None (lineage goes extinct)
                # update sum of rates
                sum_rates -= sum(lineage_dict[event_lineage_key])
                del lineage_dict[event_lineage_key] # remove this lineage from the extant lineage dictionary
                curr_lineages -= 1 # the number of extant lineages in the tree decreases by 1 (this one died)
                node_to_remove = curr_t
                curr_parent = curr_t.up
                if(curr_parent != None):
                    while(len(curr_parent.children) < 2):
                        node_to_remove = curr_parent
                        curr_parent = curr_parent.up
                        if(curr_parent == None):
                            break

                node_to_remove.detach()

            if infinite_sub_checker > 50000:
                raise SimulationError("INFINITE SUB LOOP")
        
        st = SimulatedTree(t, seq_dict, lineage_dict, sum_rates, curr_lineages)
        self.simulated_trees.append(st)
        return st