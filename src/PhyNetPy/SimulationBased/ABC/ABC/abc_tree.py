from ast import Delete
import elfi
import growtree
import numpy as np
import scipy
import ete3
import matplotlib.pyplot as plt
import statistics
import random
import math
import cProfile


b_dist = elfi.Prior(scipy.stats.uniform, 0, 2)
de_dist = elfi.Prior(scipy.stats.uniform, 0, .5) 
b_sd_dist = elfi.Prior(scipy.stats.expon, 0, 1)
de_sd_dist = elfi.Prior(scipy.stats.expon, 0, 1)
sampling_rate_arr = []

def sample_leaves(tree, goal_leaves):
    global sampling_rate_arr
    curr_leaves = growtree.tree_nleaf(tree)
    sampling_rate = goal_leaves/curr_leaves
    sampling_rate_arr.append(sampling_rate)
    num_delete_goal = math.ceil(curr_leaves * (1-sampling_rate))
    leaf_lst = tree.get_leaves()
    deleted_leaf_lst = random.sample(leaf_lst,k=num_delete_goal)
    for leaf in deleted_leaf_lst:
        leaf.delete()
    return tree


def gen_tree_sims(b=1, de=.01, leaf_goal = 10, sampling_rate_p = 0.01, is_prior = False, random_state = None):
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
    global b_dist
    global de_dist
    arr = []
    random_state = random_state or np.random # this value is not currently used
    sd_b = gen_param(b_sd_dist)
    sd_d = gen_param(de_sd_dist)
    if(is_prior): # using prior dist to simulate trees
        curr_nleaf = -9999999999
        while(curr_nleaf < leaf_goal):
            print("in while loop 2")
            birth = gen_param(b_dist)
            death = gen_param(de_dist)
            s_drawn = 75
            print("check 1")
            new_tree = growtree.gen_tree(b = birth, d = death, s = s_drawn, sd_b = sd_b, sd_d = sd_d, branch_info = 1, seq_length = 100, goal_leaves=leaf_goal, sampling_rate=sampling_rate_p)
            print("check 2")
            curr_nleaf = growtree.tree_nleaf(new_tree)
            print("check 3")
            #print("nleaf: ", curr_nleaf )
        new_tree = sample_leaves(new_tree, leaf_goal)
        #print(new_tree)
        #print("satisfied nleaf condition", growtree.tree_nleaf(new_tree))
    else: # use artificial true rates to simulate an observed tree
        birth = b
        death = de
        sub_rate = 10
        curr_nleaf = -9999999999
        while(curr_nleaf < leaf_goal):
            new_tree = growtree.gen_tree(b = birth, d = death, s = sub_rate, sd_b = sd_b, sd_d = sd_d, branch_info = 1, seq_length = 100, goal_leaves=leaf_goal, sampling_rate=sampling_rate_p)
            curr_nleaf = growtree.tree_nleaf(new_tree)
            #print("nleaf: ", curr_nleaf )
        new_tree = sample_leaves(new_tree, leaf_goal)
        print(sd_b)
        print(sd_d)

    arr.append(new_tree) # simulate tree and place in 1 element array
    
    return arr

# array of statistic values for observed tree, used for normalization of statistics
obs_tree_stats = []
stat_index = 0

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
    global obs_tree_stats # holds (or will hold) the observed stats
    global stat_index # keeps track of which stat is the current one

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
    return tree_stat(tree_arr, growtree.tree_branch_sum)

def branch_mean_stat(tree_arr):
    return tree_stat(tree_arr, growtree.tree_branch_mean)

def branch_median_stat(tree_arr):
    return tree_stat(tree_arr, growtree.tree_branch_median)

def branch_variance_stat(tree_arr):
    return tree_stat(tree_arr, growtree.tree_branch_variance)

def height_stat(tree_arr):
    return tree_stat(tree_arr, growtree.tree_height)

def depth_mean_stat(tree_arr):
    return tree_stat(tree_arr, growtree.tree_depth_mean)

def depth_median_stat(tree_arr):
    return tree_stat(tree_arr, growtree.tree_depth_median)

def depth_variance_stat(tree_arr):
    return tree_stat(tree_arr, growtree.tree_depth_variance)

def balance_stat(tree_arr):
    return tree_stat(tree_arr, growtree.tree_balance)

def nleaves_stat(tree_arr):
   return tree_stat(tree_arr, growtree.tree_nleaf)

def root_colless_stat(tree_arr):
    return tree_stat(tree_arr, growtree.tree_root_colless)

def sum_colless_stat(tree_arr):
    return tree_stat(tree_arr, growtree.tree_sum_colless)

def mean_colless_stat(tree_arr):
    return tree_stat(tree_arr, growtree.tree_mean_colless)

def median_colless_stat(tree_arr):
    return tree_stat(tree_arr, growtree.tree_median_colless)

def variance_colless_stat(tree_arr):
    return tree_stat(tree_arr, growtree.tree_variance_colless)

def gen_param(prior_dist):
    """
    Draws a single sample from 'prior_dist' and returns it (where 
    'prior_dist' is an object of the 'elfi.Prior' class). This 
    function is used for generating true parameters.
    """
    return (prior_dist.generate())[0] # draw sample and extract the value from a 1 element array

def run_main(num_accept = 100, isreal_obs = True, is_rej = False, sampling_type = "q", is_summary = False, is_plot = False, is_print = False, sampling_rate = 0.01):
    """
    Runs sampling via ABC. Returns an array containing the inferred rates 
    (from the accepted samples) and the observed tree. 'num_accept' is the 
    number of accepted samples from which posterior distributions can be 
    created. If 'isreal_obs' is 'True' then real data is used in ABC, otherwise
    values for the artificial true rates are sampled from the prior distributions
    and an observed tree is created by simulating a tree with these true rates.
    If 'isreal_obs' is 'False', the array that is returned also contains the 
    artificial true values for the rates (as well as the inferred rates and the 
    observed tree). 'is_rej' will specify whether ABC SMC or rejection sampling
    is used (if 'is_rej' is 'True' then rejection sampling is used).    
    'sampling_type' is by default "q", which means a quantile 
    method is used, but if it is set to "t", a threshold method is used. If 
    'is_summary' is set to 'True', a brief summary of the inferred rates will 
    be printed to the terminal. If 'is_plot' is set to 'True', the distribution 
    of inferred rates will be plotted. If 'is_print' is set to 'True', a full
    description of the inferred rates will be printed to the terminal.
    """

    """
    Prior distributions of rate and shape parameters. Prior distributions
    for 'd', 'birth_s', 'death_s', and 'sub_s' are modeled with an 
    exponential distribution using a scale of 100. The prior distribution 
    for 'r' is modeled with a uniform distribution from 0 (inclusive) to 1
    (exclusive). 
    """

    global b_dist
    global de_dist
    global sampling_rate_arr
    
    """
    Below are the true parameters for diversification (d) 
    and turnover (r) rates. Diversification and turnover rates 
    are related to birth and death rates by the following equations:
        diversification = birth - death
        turnover = death / birth
    'd_true' must be at least 0 with no upper bound. 'r_true' must be 
    between 0 (inclusive) and 1 (exclusive). The values for 'd_true' 
    and 'r_true' are drawn from the prior distributions defined above.
    """
    birth_true = gen_param(b_dist)
    death_true = gen_param(de_dist)



    """
    Below are the true parameters for the distribution shape parameters.
    'birth_s_true' is the shape of the distribution of birth rates 
    which were involved in generating the multi-state birth-death (MSBD) tree. 
    'death_s_true' is the shape of the distribution of death rates and 
    'sub_s_true' is the shape of the distribution of substitution rates which 
    were involved in generating the MSBD tree. All distribution shape parameters 
    must be greater than or equal to 0 with no upper bound. The values for 
    'birth_s_true', 'death_s_true', and 'sub_s_true' are drawn from the prior 
    distributions defined above.
    """

    if isreal_obs: # use real data for observed tree
        # Below is the phylogeny from the real cancer data (given in Newick string format)
        tree_real_data = ete3.Tree("(n6:0,((((n4:0,n7:0)8bb16f00-dfee-4f81-ac96-767b993ca6e2:0,((((((n16:0,(h1:17,((((a2:0,a7:0)e7bc4028-414f-4bff-b699-8bd16b23ee7e:18,a3:1)2b854f8d-4936-436e-96c5-910be4aba19c:64,a5:2)816d774c-97d9-43e6-8b2e-f15671af8af2:62,(((h5:3,(a8:3,h8:0)09f1a603-d099-4870-9e79-d6e5bee8e2a3:8)0a18283f-9dc0-491c-9fea-1b8779299e72:30,(((h7:8,(h4:1,h6:0)6fb58ff1-8f8f-487e-893f-a7a5074f5230:22)ee5e30a1-dc03-43f1-9ad0-5a9811fb2360:25,h2:0)50a77d83-30fc-414d-b18b-96e2e4c9866c:10,((a4:3,a6:3)d9afdef5-aa08-4dcc-aa5b-a9b8649d4889:29,a1:0)2913b995-8502-4080-801f-42d014b5d58a:183)9f319495-041d-4361-86c3-785075bb1cf3:54)a6f16d8b-261e-4f09-83ec-99a518991759:82,h3:2)66771a30-0bee-44c5-8282-af5400c18959:149)bb4ffd6c-cf65-4ffb-bce2-cc434482b915:1155)6fffd3b9-b1f8-4bcf-bc47-c92f726cabfb:1325)389c6172-5350-4199-b387-29540a785b5a:42,n13:0)bf915916-25af-4cf4-b258-25422a0360e6:15,n11:0)15a37f26-1ebd-4a43-b3a5-a8962e5e0112:10,n15:0)5c9cf572-6674-45ed-8647-a96c30b2c098:7,n10:0)debcb1ee-4c8a-4db0-be04-25a7ad8e5aad:2,(n8:0,(n5:0,(n14:0,(n3:0,(n12:0,n9:0)ec159609-da23-43d0-814a-4981b51a1b72:0)c05653ec-9a44-4ab2-a5b0-8f3b3d10a41e:0)f7f52196-a2dc-443f-8e1d-54b5e4bb6dc2:0)3fc23945-5205-4af7-aaf8-3271f35280cf:0)ffe9b6b5-e105-4b3c-91e9-9482b268ff34:0)726af8d3-3dfe-49a3-bd23-0dfd36f568ff:0)27ece196-6123-4a35-bc65-db77090f1882:0,n1:0)fb7d141a-3eb9-4dd7-9bbd-8c079d65c0d3:0,n2:0)509e48c2-b79f-4dc2-afe1-67bedf7cc929:0);", format = 1)
        obs = tree_real_data
        print("obs height", growtree.tree_height(obs))
        obs_nleaf = growtree.tree_nleaf(obs)
    else: # simulate observed tree based on artificial true values sampled from the prior distributions 
        obs = (gen_tree_sims(b = birth_true, de = death_true, leaf_goal = 10, is_prior = False))[0] # observed tree (tree simulated with true rate and distribution shape parameters)
        while(growtree.tree_nleaf(obs) < 10): # artificial obs tree must have at least 10 leaves
            birth_true = gen_param(b_dist)
            death_true = gen_param(de_dist)        
            print("in while loop")
            obs = (gen_tree_sims(b = birth_true, de = death_true, leaf_goal = 10, is_prior = False))[0] # observed tree (tree simulated with true rate and distribution shape parameters)
        #print(birth_true)
        #print(death_true)
        obs_nleaf = growtree.tree_nleaf(obs)
    
    print("obs leaves: ", obs_nleaf)

    """
    'sim' is a simulator node with the 'gen_tree_sims()' function, the prior distributions of rate 
    and shape parameters, and the observed tree ('obs') passed to it as arguments.
    """
    sim = elfi.Simulator(elfi.tools.vectorize(gen_tree_sims), 0, 0, obs_nleaf, sampling_rate, True, observed = obs) 
    
    """
    Below are summary nodes with each node having a unique tree summary statistic function
    and all the simulated trees (includes the observed tree).
    """
    summ_branch_sum = elfi.Summary(branch_sum_stat, sim) 
    summ_branch_mean = elfi.Summary(branch_mean_stat, sim) 
    summ_branch_median = elfi.Summary(branch_median_stat, sim) 
    summ_branch_variance = elfi.Summary(branch_variance_stat, sim) 
    summ_height = elfi.Summary(height_stat, sim) 
    summ_depth_mean = elfi.Summary(depth_mean_stat, sim) 
    summ_depth_median = elfi.Summary(depth_median_stat, sim) 
    summ_depth_variance = elfi.Summary(depth_variance_stat, sim) 
    summ_balance = elfi.Summary(balance_stat, sim) 
    summ_nleaves = elfi.Summary(nleaves_stat, sim) 
    summ_root_colless = elfi.Summary(root_colless_stat, sim) 
    summ_colless_sum = elfi.Summary(sum_colless_stat, sim) 
    summ_colless_mean = elfi.Summary(mean_colless_stat, sim) 
    summ_colless_median = elfi.Summary(median_colless_stat, sim) 
    summ_colless_variance = elfi.Summary(variance_colless_stat, sim) 

    """
    Below are distance nodes that calculate the euclidean (squared distance): 
    ('summ_stat_1_sim' - 'summ_stat_1_obs') * 2 + ... + ('summ_stat_n_sim' - 'summ_stat_n_obs') * 2
    for 'n' summary statistics (where 'n' is the number of statistics provided in the creation
    of the distance node). 
    """

    # 'dist_all_summ' is a distance node containing all the available tree summary statistics
    dist_all_summ = elfi.Distance('euclidean', summ_branch_sum, summ_branch_mean, summ_branch_median, 
        summ_branch_variance, summ_height, summ_depth_mean, summ_depth_median, summ_depth_variance, 
        summ_balance, summ_nleaves, summ_root_colless, summ_colless_sum, summ_colless_mean, 
        summ_colless_median, summ_colless_variance)

    # 'dist_mmv' is a distance node containing all summary statistics, but excluding tree summary 
    # statistics which calculate sums (i.e. mean, median, and variance statistics, but not sum statistics)
    dist_mmv = elfi.Distance('euclidean', summ_branch_mean, summ_branch_median, summ_branch_variance, 
        summ_height, summ_depth_mean, summ_depth_median, summ_depth_variance, summ_balance, 
        summ_nleaves,summ_colless_mean, summ_colless_median, summ_colless_variance)

    # 'dist_mm' is a distance node containing all summary statistics, but excluding tree summary 
    # statistics which calculate sums and variances (i.e. mean and median statistics, but not 
    # sum and variance statistics)
    dist_mm = elfi.Distance('euclidean', summ_branch_mean, summ_branch_median, summ_height,
        summ_depth_mean, summ_depth_median, summ_balance, summ_nleaves, summ_colless_mean, 
        summ_colless_median)

    # 'dist_birth_all' is a distance node containing the tree summary statistics that showed some
    # change in birth rate in the correct direction when 'birth_true' rate was set at 1 and 10
    dist_birth_all = elfi.Distance('euclidean', summ_branch_mean, summ_branch_median, summ_branch_variance,
        summ_height, summ_depth_median, summ_depth_variance, summ_balance, summ_nleaves, summ_colless_sum,
        summ_colless_variance)

    # 'dist_birth_best' is a distance node containing the tree summary statistics that showed >= 2
    # change in birth rate in the correct direction when 'birth_true' rate was set at 1 and 10
    dist_birth_best = elfi.Distance('euclidean', summ_branch_mean, summ_branch_median, summ_branch_variance,
        summ_depth_median, summ_balance, summ_nleaves)

    # 'dist_death_all' is a distance node containing the tree summary statistics that showed some
    # change in death rate in the correct direction when 'death_true' rate was set at 1 and 10
    dist_death_all = elfi.Distance('euclidean', summ_branch_variance, summ_height, summ_depth_median, 
        summ_nleaves, summ_root_colless, summ_colless_median)

    # 'dist_death_best' is a distance node containing the tree summary statistics that showed >= 2
    # change in death rate in the correct direction when 'death_true' rate was set at 1 and 10
    dist_death_best = elfi.Distance('euclidean', summ_height, summ_depth_median, summ_root_colless, summ_colless_median)

    # 'dist_shared_all' is a distance node containing the tree summary statistics that showed some
    # change in the correct direction for both 'birth_true' and 'death_true' when each rate was set 
    # at 1 and 10
    dist_shared_all = elfi.Distance('euclidean', summ_branch_variance, summ_height, summ_depth_median, summ_nleaves)

    # 'dist_shared_best' is a distance node containing the tree summary statistics that showed >= 2
    # change in the correct direction for both 'birth_true' and 'death_true' when each rate was set 
    # at 1 and 10
    dist_shared_best = elfi.Distance('euclidean', summ_branch_variance, summ_height, summ_depth_median, summ_nleaves)

    # 'dist_score_okay' is a distance node containing the tree summary statistics that showed > 0.5 score of 
    # closeness in inference for the shape and rate parameters. 0.5 points were given for every inference deemed
    # as close to the true value and 1 point was given for every inference that contained the true value (true value
    # was in the range created by the mean and median inferred parameter). 
    dist_score_okay = elfi.Distance('euclidean', summ_branch_mean, summ_branch_median, summ_branch_variance, summ_height,
        summ_depth_mean, summ_depth_variance, summ_balance, summ_colless_sum, summ_colless_mean, summ_colless_median)

    # 'dist_score_good' is a distance node containing the tree summary statistics that showed > 1.5 score of 
    # closeness in inference for the shape and rate parameters. 0.5 points were given for every inference deemed
    # as close to the true value and 1 point was given for every inference that contained the true value (true value
    # was in the range created by the mean and median inferred parameter). 
    dist_score_good = elfi.Distance('euclidean', summ_branch_mean, summ_branch_median, summ_height, summ_depth_mean, 
        summ_depth_variance, summ_colless_mean, summ_colless_median)

    # 'dist_score_best' is a distance node containing the tree summary statistics that showed > 2 score of 
    # closeness in inference for the shape and rate parameters. 0.5 points were given for every inference deemed
    # as close to the true value and 1 point was given for every inference that contained the true value (true value
    # was in the range created by the mean and median inferred parameter). 
    dist_score_best = elfi.Distance('euclidean', summ_height, summ_depth_variance, summ_colless_mean)

    # 'dist_some_good' is a distance node containing the tree summary statistics that estimated some parameters well. 
    dist_some_good = elfi.Distance('euclidean', summ_branch_mean, summ_branch_median, summ_height, summ_depth_mean,
        summ_depth_variance, summ_colless_sum, summ_colless_mean, summ_colless_median, summ_colless_variance)

    # 'dist_overall_good' is a distance node containing the tree summary statistics that overall estimated parameters well. 
    dist_overall_good = elfi.Distance('euclidean', summ_depth_mean, summ_branch_median, summ_height, summ_depth_variance, 
        summ_colless_mean, summ_colless_median)

    # 'dist_overall_best' is a distance node containing the tree summary statistics that overall estimated parameters the closest. 
    dist_overall_best = elfi.Distance('euclidean', summ_depth_mean, summ_depth_variance)

    # 'dist_good_shape' is a distance node containing the tree summary statistics that best estimated shape parameters. 
    dist_good_shape = elfi.Distance('euclidean', summ_depth_mean, summ_depth_variance, summ_colless_median, 
        summ_colless_variance, summ_colless_mean, summ_colless_sum)

    # 'dist_good_bd' is a distance node containing the tree summary statistics that best estimated birth and death rate parameters. 
    dist_good_bd = elfi.Distance('euclidean', summ_branch_median, summ_depth_mean, summ_depth_variance, summ_height, 
        summ_colless_mean)

    dist_scatterplots = elfi.Distance('euclidean', summ_depth_variance, summ_branch_mean, summ_height, 
        summ_branch_variance, summ_branch_sum, summ_colless_mean, summ_colless_sum)

    dist_scatterplots2 = elfi.Distance('euclidean', summ_depth_variance, summ_branch_mean, summ_height, 
        summ_colless_mean, summ_root_colless)
    
    #dist = dist_scatterplots2 # choosing which distance node to use 
    
    dist1 = elfi.Distance('minkowski', summ_branch_sum, summ_height, summ_depth_mean, summ_colless_sum, summ_colless_variance, p=1)
    dist_all = elfi.Distance('minkowski', summ_branch_sum, summ_branch_mean, summ_branch_median, 
        summ_branch_variance, summ_height, summ_depth_mean, summ_depth_median, summ_depth_variance, 
        summ_balance, summ_nleaves, summ_root_colless, summ_colless_sum, summ_colless_mean, 
        summ_colless_median, summ_colless_variance, p=1)
    
    dist = dist_all
    batch_size = 20
    N = num_accept # number of accepted samples needed in 'result' in the sampling below
    result_type = None # will specify which type of sampling is used (threshold or quantile for rejection or smc for SMC ABC)

    if(is_rej): # use rejection sampling
        """
        'rej' is a rejection node used in inference with rejection sampling
        using 'dist' values in order to reject. 'batch_size' defines how many 
        simulations are performed in computation of 'dist'.
        """
        rej = elfi.Rejection(dist, batch_size = batch_size)
       
        """
        Note that it is not necessary to sample using both types of rejection described 
        above (threshold and quantiles). One or the other is sufficient and using both 
        will needlessly increase the runtime of the program. 
        """
        
        if sampling_type == "t": # use threshold method
            """
            Below is rejection using a threshold 'thresh'. All simulated trees generated
            from rates drawn from the priors that have a generated distance below 'thresh'
            will be accepted as samples. The simulator will generate as many trees as it 
            takes to accept the specified number of trees ('N' trees).
            """
            thresh = 0.1 # distance threshold
            result_thresh = rej.sample(N, threshold = thresh) # generate result
            result_type = result_thresh # setting method of rejection sampling
        else: # use quantile method
            """
            Below is rejection using quantiles. The quantile of trees size 'quant' 
            with the smallest generated distances are accepted. The simulator will 
            generate ('N' / 'quant') trees and accept 'N' of them.
            """
            quant = 0.1 # quantile of accepted trees

            result_quant = rej.sample(N, quantile = quant) # generate result
            result_type = result_quant # setting method of rejection sampling
    else: # use ABC SMC
        smc = elfi.SMC(dist, batch_size = batch_size)
        schedule = [2, 1.25] # schedule is a list of thresholds to use for each population
        long_schedule = [2, 1.25, .75]
        short_schedule = [2] # use short schedule for 1 round of ABC SMC
        result_smc = smc.sample(N, short_schedule)
        result_type = result_smc

    if is_summary: # printing a brief summary of the inferred rates and shapes
        if is_rej:
            result_type.summary() # summary statistics from the inference with rejection sampling
        else:
            result_type.summary(all = True)

    if is_plot: # plotting distribution of inferred rates and shapes
        result_type.plot_marginals() # plotting the marginal distributions of the birth and death rates for the accepted samples
        plt.ylabel('Rate frequency')
        plt.title('Distribution of rates for accepted samples')
        #result_type.plot_pairs() # plotting the pairwise relationships of the birth and death rates for the accepted samples
        plt.show() # display the plot of pairwise relationships

    # Finding the mean and median inferred rates and shapes below
    b_infer = result_type.samples['b_dist']
    de_infer = result_type.samples['de_dist']
    b_sd_infer = result_type.samples['b_sd_dist']
    de_sd_infer = result_type.samples['de_sd_dist']

    if is_print: # printing detailed summary of inferred rates and shapes


        # Displaying the true rates and shapes below to compare to the inferred rates and shapes
        if(not(isreal_obs)):
            print("true birth rate: " + str(birth_true))
            print("true death rate: " + str(death_true))
       

    res = [] # result array that will hold the inferred rates and the observed tree
    res.append(b_infer)
    res.append(de_infer)
    res.append(b_sd_infer)
    res.append(de_sd_infer)

    res.append(obs)

    if(not(isreal_obs)): # include the artificial true rates in the result array 
        res.append(birth_true)
        res.append(death_true)

    #print(sampling_rate_arr)
    print("sampling rate: ", sum(sampling_rate_arr)/len(sampling_rate_arr))
    # reset global var
    sampling_rate_arr = []
    return res

#pr = cProfile.Profile()   
#pr.enable()
run_main(is_summary = True, is_plot = True, num_accept = 100, isreal_obs=True) # uncomment to run abc directly by running this file
#run_main(is_summary = True, is_plot = True, num_accept = 100, isreal_obs=False)
#run_main(num_accept = 10, isreal_obs=True)
#pr.disable()
#pr.print_stats(25)