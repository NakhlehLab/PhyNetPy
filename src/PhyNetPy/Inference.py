
import sys
sys.path.insert(0, 'src/PhyNetPy/Bayesian')
sys.path.insert(1, 'src/PhyNetPy/ABC')

from Bayesian import NetworkBuilder as nb
from Bayesian import MSA
from Bayesian import MetropolisHastings as mh
from Bayesian import Alphabet as a
from Bayesian import BirthDeath as bd
from Bayesian import Matrix as m
from Bayesian import ModelGraph as mg
from Bayesian import GTR 
import elfi
import scipy
from ABC.SimulatedTree import SimulatedTree
from ABC.Simulator import *
from ABC.TreeStatistics import *
from matplotlib import pyplot as plt

    
    
def SNAPP_Likelihood(filename: str, u :float , v:float, coal:float, grouping:dict=None, auto_detect:bool = False, summary_path:str = None, network_path:str = None) -> float:
    """
    Given a filename that represents a path to a nexus file that defines and data, compute the maximum likelihood 
    network (trees for now, simulator/moves do not support inferring networks yet) and return its likelihood, P(network | data). 
    
    Makes use of the algorithm described in Bryant et al, and the network generalization in Rabier et al.
    Optimization of this algorithm is currently under development.
    
    Args:
        filename (str): Path to a nexus file. File must include a tree/network block, as well as a data block.
        u (float): Stationary transition probability from a green to red allele
        v (float): Stationary transition probability from a red to green allele
        coal (float): Coalesence rate, theta
        grouping (dict): A mapping from a general label (ie. "human") to any applicable taxa labels (ie "human1", "human2", etc.). 
                         Keys must be strings, and values must be a list of strings. By default, this is None. If auto_detect is also False, 
                         then each taxa has only one sample.
        auto_detect (bool): If enabled (default is to be disabled), then instead of providing a grouping map, this software will attempt
                            to find the most likely grouping of labels based on string similarity.
        summary_path (str): If set, then metropolis hastings log output will be written to this file. Must be txt file
        network_path (str): If set, then the final network will be converted to a newick string and written to this text file.                 
    
    Returns:
        a float: P(network | data) of the maximum likelihood network. 
    """

    aln = MSA(filename, grouping=grouping, grouping_auto_detect=auto_detect)
    #Only generates tree starting conditions
    network = bd.CBDP(1, .5, aln.num_groups()).generateTree()
    
    snp_params={"samples": len(aln.get_records()), "u": u, "v": v, "coal" : coal, "grouping":True}
    m = m.Matrix(aln, a.Alphabet("SNP"))
    snp_model = mg.Model(network, m, snp_params=snp_params)
    
    mh = mh.MetropolisHastings(mh.ProposalKernel(), GTR.JC(), m, 800, snp_model) #TODO: Submodel unnecessary for snp. make optional?
    result_state = mh.run()
    
    result_state.current_model.summary(network_path, summary_path)
        
    return result_state.likelihood()


def SNAPP_Likelihood(filename: str, u :float , v:float, coal:float, grouping:dict=None, auto_detect:bool = False, show_partials:bool = False) -> list:
    """
    Given a filename that represents a path to a nexus file that defines a network and data, compute the log-likelihood for the
    network given the data. 
    
    Makes use of the algorithm described in Bryant et al, and the network generalization in Rabier et al.
    Optimization of this algorithm is currently under development.
    
    If the nexus file contains multiple trees/networks in the tree block, then an array containing all likelihoods will be returned

    Args:
        filename (str): Path to a nexus file. File must include a tree/network block, as well as a data block.
        u (float): Stationary transition probability from a green to red allele
        v (float): Stationary transition probability from a red to green allele
        coal (float): Coalesence rate, theta
        grouping (dict): A mapping from a general label (ie. "human") to any applicable taxa labels (ie "human1", "human2", etc.). 
                         Keys must be strings, and values must be a list of strings. By default, this is None. If auto_detect is also False, 
                         then each taxa has only one sample.
        auto_detect (bool): If enabled (default is to be disabled), then instead of providing a grouping map, this software will attempt
                            to find the most likely grouping of labels based on string similarity.
        show_partials (bool, optional): Set to true for formatted visibility into partial likelihoods for each node. Defaults to False.

    Returns:
        list of floats: P(network | data) for each network passed in, in the order in which they appear in the nexus file.
    """
    
    #Generate a multiple sequence alignment from the nexus file data
    aln = MSA.MSA(filename, grouping=grouping, grouping_auto_detect = auto_detect)
    
    #Read and parse the network described 
    networks = nb.NetworkBuilder(filename).get_all_networks()
    
    likelihoods = []
    for network in networks:
        snp_params={"samples": len(aln.get_records()), "u": u, "v": v, "coal" : coal, "grouping":False}
        #Create model
        snp_model = mg.Model(network, m.Matrix(aln, a.Alphabet("SNP")), None, snp_params=snp_params, verbose = True)
        #Compute the likelihood
        likelihoods.append(snp_model.SNP_likelihood())
 
    return likelihoods


def ABC(num_accept = 100, isreal_obs = True, is_rej = False, sampling_type = "q", is_summary = False, is_plot = False, is_print = False, sampling_rate = 0.01):
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

    global d_dist
    global r_dist
    global sub_dist
    global sampling_rate_arr
    
    birth_s = elfi.Prior(scipy.stats.expon, 0, 25) # prior distribution for birth distribution shape
    death_s = elfi.Prior(scipy.stats.expon, 0, 25) # prior distribution for death distribution shape
    sub_s = elfi.Prior(scipy.stats.expon, 0, 25) # prior distribution for substitution distribution shape

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
    d_true = gen_param(d_dist)
    r_true = gen_param(r_dist)
    while(r_true>=1):
        r_true = gen_param(r_dist)
    sub_true = gen_param(sub_dist)

    """
    Below are the true parameters for birth and death rates. 
    Death rate must be greater than or equal to 0 and birth rate 
    must be greater than death rate.
    """
    rate_arr = calc_rates_bd(d_true, r_true) # calculating the true birth and death parameters 
    birth_true = rate_arr[0] # extracting birth rate
    death_true = rate_arr[1] # extracting death rate

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
    birth_s_true = gen_param(birth_s)
    death_s_true = gen_param(death_s)
    sub_s_true = gen_param(sub_s)
    if isreal_obs: # use real data for observed tree
        # Below is the phylogeny from the real cancer data (given in Newick string format)
        tree_real_data = ete3.Tree("(n6:0,((((n4:0,n7:0)8bb16f00-dfee-4f81-ac96-767b993ca6e2:0,((((((n16:0,(h1:17,((((a2:0,a7:0)e7bc4028-414f-4bff-b699-8bd16b23ee7e:18,a3:1)2b854f8d-4936-436e-96c5-910be4aba19c:64,a5:2)816d774c-97d9-43e6-8b2e-f15671af8af2:62,(((h5:3,(a8:3,h8:0)09f1a603-d099-4870-9e79-d6e5bee8e2a3:8)0a18283f-9dc0-491c-9fea-1b8779299e72:30,(((h7:8,(h4:1,h6:0)6fb58ff1-8f8f-487e-893f-a7a5074f5230:22)ee5e30a1-dc03-43f1-9ad0-5a9811fb2360:25,h2:0)50a77d83-30fc-414d-b18b-96e2e4c9866c:10,((a4:3,a6:3)d9afdef5-aa08-4dcc-aa5b-a9b8649d4889:29,a1:0)2913b995-8502-4080-801f-42d014b5d58a:183)9f319495-041d-4361-86c3-785075bb1cf3:54)a6f16d8b-261e-4f09-83ec-99a518991759:82,h3:2)66771a30-0bee-44c5-8282-af5400c18959:149)bb4ffd6c-cf65-4ffb-bce2-cc434482b915:1155)6fffd3b9-b1f8-4bcf-bc47-c92f726cabfb:1325)389c6172-5350-4199-b387-29540a785b5a:42,n13:0)bf915916-25af-4cf4-b258-25422a0360e6:15,n11:0)15a37f26-1ebd-4a43-b3a5-a8962e5e0112:10,n15:0)5c9cf572-6674-45ed-8647-a96c30b2c098:7,n10:0)debcb1ee-4c8a-4db0-be04-25a7ad8e5aad:2,(n8:0,(n5:0,(n14:0,(n3:0,(n12:0,n9:0)ec159609-da23-43d0-814a-4981b51a1b72:0)c05653ec-9a44-4ab2-a5b0-8f3b3d10a41e:0)f7f52196-a2dc-443f-8e1d-54b5e4bb6dc2:0)3fc23945-5205-4af7-aaf8-3271f35280cf:0)ffe9b6b5-e105-4b3c-91e9-9482b268ff34:0)726af8d3-3dfe-49a3-bd23-0dfd36f568ff:0)27ece196-6123-4a35-bc65-db77090f1882:0,n1:0)fb7d141a-3eb9-4dd7-9bbd-8c079d65c0d3:0,n2:0)509e48c2-b79f-4dc2-afe1-67bedf7cc929:0);", format = 1)
        obs = tree_real_data
        #print("obs height", growtree.tree_height(obs))
        obs_nleaf = tree_nleaf(obs)
    else: # simulate observed tree based on artificial true values sampled from the prior distributions 
        obs = (gen_tree_sims(d = d_true, r = r_true, sub_rate = sub_true, birth_shape = birth_s_true, death_shape = death_s_true, sub_shape = sub_s_true, leaf_goal = 10, sampling_rate = sampling_rate, is_prior = False))[0] # observed tree (tree simulated with true rate and distribution shape parameters)
        while(tree_nleaf() < 10): # artificial obs tree must have at least 10 leaves
            d_true = gen_param(d_dist)
            r_true = gen_param(r_dist)
            sub_true = gen_param(sub_dist)
            rate_arr = calc_rates_bd(d_true, r_true) 
            birth_true = rate_arr[0]
            death_true = rate_arr[1]
            birth_s_true = gen_param(birth_s)
            death_s_true = gen_param(death_s)
            sub_s_true = gen_param(sub_s)
            obs = (gen_tree_sims(d = d_true, r = r_true, sub_rate = sub_true, birth_shape = birth_s_true, death_shape = death_s_true, sub_shape = sub_s_true, leaf_goal = 10, sampling_rate = sampling_rate, is_prior = False))[0] # observed tree (tree simulated with true rate and distribution shape parameters)
        obs_nleaf = tree_nleaf(obs)
    
    #print("obs leaves: ", obs_nleaf)

    """
    'sim' is a simulator node with the 'gen_tree_sims()' function, the prior distributions of rate 
    and shape parameters, and the observed tree ('obs') passed to it as arguments.
    """
    sim = elfi.Simulator(elfi.tools.vectorize(gen_tree_sims), 0, 0, 0, birth_s, death_s, sub_s, obs_nleaf, sampling_rate, True, observed = obs) 
    
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

    dist = elfi.Distance('minkowski', summ_branch_sum, summ_height, summ_depth_mean, summ_colless_sum, summ_colless_variance, p=1)
    dist_all = elfi.Distance('minkowski', summ_branch_sum, summ_branch_mean, summ_branch_median, 
        summ_branch_variance, summ_height, summ_depth_mean, summ_depth_median, summ_depth_variance, 
        summ_balance, summ_nleaves, summ_root_colless, summ_colless_sum, summ_colless_mean, 
        summ_colless_median, summ_colless_variance, p=1)
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
    d_infer = result_type.samples['d_dist']
    d_infer_mean = np.mean(d_infer)
    d_infer_median = np.median(d_infer)
    r_infer = result_type.samples['r_dist']
    r_infer_mean = np.mean(r_infer)
    r_infer_median = np.median(r_infer)
    sub_infer = result_type.samples['sub_dist']
    sub_infer_mean = np.mean(sub_infer)
    sub_infer_median = np.median(sub_infer)
    bd_infer_mean = calc_rates_bd(d_infer_mean, r_infer_mean) # calculating the mean inferred birth and death rates
    bd_infer_median = calc_rates_bd(d_infer_median, r_infer_median) # calculating the median inferred birth and death rates
    birth_s_infer = result_type.samples['birth_s']
    death_s_infer = result_type.samples['death_s']
    sub_s_infer = result_type.samples['sub_s']

    if is_print: # printing detailed summary of inferred rates and shapes
        print("mean inferred diversification rate: " + str(d_infer_mean))
        print("median inferred diversification rate: " + str(d_infer_median))
        print("mean inferred turnover rate: " + str(r_infer_mean))
        print("median inferred turnover rate: " + str(r_infer_median))
        print("mean inferred sub rate: " + str(sub_infer_mean))
        print("median inferred sub rate: " + str(sub_infer_median))
        print("mean inferred birth rate: " + str(bd_infer_mean[0]))
        print("median inferred birth rate: " + str(bd_infer_median[0]))
        print("mean inferred death rate: " + str(bd_infer_mean[1]))
        print("median inferred death rate: " + str(bd_infer_median[1]))

        print("mean inferred birth distribution shape: " + str(np.mean(birth_s_infer)))
        print("median inferred birth distribution shape: " + str(np.median(birth_s_infer)))
        print("mean inferred death distribution shape: " + str(np.mean(death_s_infer)))
        print("median inferred death distribution shape: " + str(np.median(death_s_infer)))
        print("mean inferred substitution distribution shape: " + str(np.mean(sub_s_infer)))
        print("median inferred substitution distribution shape: " + str(np.median(sub_s_infer)))
        print()

        # Displaying the true rates and shapes below to compare to the inferred rates and shapes
        print("true diversification rate: " + str(d_true))
        print("true turnover rate: " + str(r_true))
        print("true sub rate: " + str(sub_true))
        print("true birth rate: " + str(birth_true))
        print("true death rate: " + str(death_true))
        print("true birth distribution shape: " + str(birth_s_true))
        print("true death distribution shape: " + str(death_s_true))
        print("true substitution distribution shape: " + str(sub_s_true))

    res = [] # result array that will hold the inferred rates and the observed tree
    res.append(d_infer)
    res.append(r_infer)
    res.append(sub_infer)
    res.append(birth_s_infer)
    res.append(death_s_infer)
    res.append(sub_s_infer)
    res.append(obs)

    if(not(isreal_obs)): # include the artificial true rates in the result array 
        res.append(d_true)
        res.append(r_true)
        res.append(sub_true)
        res.append(birth_s_true)
        res.append(death_s_true)
        res.append(sub_s_true)

    #print(sampling_rate_arr)
    print(sum(sampling_rate_arr)/len(sampling_rate_arr))
    # reset global var
    sampling_rate_arr = []
    return res