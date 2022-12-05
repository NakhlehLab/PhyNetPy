import growtree as gt
import abc_tree as abc
import matplotlib.pyplot as plt
import elfi
import scipy
import math

# Define prior distributions for parameters (same as in 'abc_tree.py')
d_dist = elfi.Prior(scipy.stats.expon, 0, 1) # prior distribution for diversification
r_dist = elfi.Prior(scipy.stats.uniform, 0, 1) # prior distribution for turnover
sub_dist = elfi.Prior(scipy.stats.uniform, 0, 1) # prior distribution for sub
birth_s = elfi.Prior(scipy.stats.expon, 0, 25) # prior distribution for birth distribution shape
death_s = elfi.Prior(scipy.stats.expon, 0, 25) # prior distribution for death distribution shape
sub_s = elfi.Prior(scipy.stats.expon, 0, 25) # prior distribution for substitution distribution shape

def zero_log(i):
    """
    Log10 function that is safe with zeros (makes log10(0) = 0).
    """
    if(i == 0):
        return 0
    return math.log10(i)

def div_rate_v_stats(use_prior = True, N = 100):
    """
    Plots diversification rate on the x-axis and the value for various
    tree statistics on the y-axis. 'use_prior' determines whether the
    diversification rate will grow incrementally (if 'use_prior' is false)
    or will be sampled from the prior distribution each time (if 'use_prior'
    is true). 'N' is the number of trees generated from different diversification
    rates from which statistics will be calculated.
    """
    d_rate = 0.5 # initial div rate (used if 'use_prior' is false)
    d_arr = []
    b_sum_arr = []
    b_mean_arr = []
    b_median_arr = []
    b_variance_arr = []
    height_arr = []
    d_mean_arr = []
    d_median_arr = []
    d_variance_arr = []
    balance_arr = []
    nleaf_arr = []
    root_colless_arr = []
    sum_colless_arr = []
    mean_colless_arr = []
    median_colless_arr = []
    variance_colless_arr = []

    # generate rates for non-div parameters (stay constant for all trees generated below)
    r_rate = abc.gen_param(r_dist)
    while(r_rate >= 1): 
        r_rate = abc.gen_param(r_dist)
    sub_rate = abc.gen_param(sub_dist)
    birth_shape = abc.gen_param(birth_s)
    death_shape = abc.gen_param(death_s)
    sub_shape = abc.gen_param(sub_s)
    i = 0

    while(i < N): # 'N' is the number of rates drawn and trees generates
        if(use_prior):
            d_rate = abc.gen_param(d_dist)
        else:
            d_rate += i * 0.005
        d_arr.append(d_rate)

        t = abc.gen_tree_sims(d = d_rate, r = r_rate, sub_rate = sub_rate, birth_shape = birth_shape, death_shape = death_shape, sub_shape = sub_shape, is_prior = True)[0]
        # calc tree stats
        b_sum_arr.append(math.log10(gt.tree_branch_sum(t)+1))
        b_mean_arr.append(math.log10(gt.tree_branch_mean(t)+1))
        b_median_arr.append(math.log10(gt.tree_branch_median(t)+1))
        b_variance_arr.append(math.log10(gt.tree_branch_variance(t)+1))
        height_arr.append(math.log10(gt.tree_height(t)+1))
        d_mean_arr.append(math.log10(gt.tree_depth_mean(t)+1))
        d_median_arr.append(math.log10(gt.tree_depth_median(t)+1))
        d_variance_arr.append(math.log10(gt.tree_depth_variance(t)+1))
        balance_arr.append(math.log10(gt.tree_balance(t)+1))
        nleaf_arr.append(math.log10(gt.tree_nleaf(t)+1))
        root_colless_arr.append(math.log10(gt.tree_root_colless(t)+1))
        sum_colless_arr.append(math.log10(gt.tree_sum_colless(t)+1))
        mean_colless_arr.append(math.log10(gt.tree_mean_colless(t)+1))
        median_colless_arr.append(math.log10(gt.tree_median_colless(t)+1))
        variance_colless_arr.append(math.log10(gt.tree_variance_colless(t)+1))
        i += 1
    
    # plot stats vs rates
    fig, axs = plt.subplots(3, 5)
    axs[0, 0].plot(d_arr, b_sum_arr, 'ro')
    axs[0, 0].set_title('Div v branch sum')
    axs[0, 1].plot(d_arr, b_mean_arr, 'ro')
    axs[0, 1].set_title('Branch mean')
    axs[0, 2].plot(d_arr, b_median_arr, 'ro')
    axs[0, 2].set_title('Branch median')
    axs[0, 3].plot(d_arr, b_variance_arr, 'ro')
    axs[0, 3].set_title('Branch variance')
    axs[0, 4].plot(d_arr, height_arr, 'ro')
    axs[0, 4].set_title('Height')
    axs[1, 0].plot(d_arr, d_mean_arr, 'ro')
    axs[1, 0].set_title('Depth mean')
    axs[1, 1].plot(d_arr, d_median_arr, 'ro')
    axs[1, 1].set_title('Depth median')
    axs[1, 2].plot(d_arr, d_variance_arr, 'ro')
    axs[1, 2].set_title('Depth variance')
    axs[1, 3].plot(d_arr, balance_arr, 'ro')
    axs[1, 3].set_title('Balance')
    axs[1, 4].plot(d_arr, nleaf_arr, 'ro')
    axs[1, 4].set_title('Nleaf')
    axs[2, 0].plot(d_arr, root_colless_arr, 'ro')
    axs[2, 0].set_title('Root colless')
    axs[2, 1].plot(d_arr, sum_colless_arr, 'ro')
    axs[2, 1].set_title('Sum colless')
    axs[2, 2].plot(d_arr, mean_colless_arr, 'ro')
    axs[2, 2].set_title('Mean colless')
    axs[2, 3].plot(d_arr, median_colless_arr, 'ro')
    axs[2, 3].set_title('Median colless')
    axs[2, 4].plot(d_arr, variance_colless_arr, 'ro')
    axs[2, 4].set_title('Variance colless')

    axs[0, 0].set_ylim(bottom = 0)
    axs[0, 1].set_ylim(bottom = 0)
    axs[0, 2].set_ylim(bottom = 0)
    axs[0, 3].set_ylim(bottom = 0)
    axs[0, 4].set_ylim(bottom = 0)
    axs[1, 0].set_ylim(bottom = 0)
    axs[1, 1].set_ylim(bottom = 0)
    axs[1, 2].set_ylim(bottom = 0)
    axs[1, 3].set_ylim(bottom = 0)
    axs[1, 4].set_ylim(bottom = 0)
    axs[2, 0].set_ylim(bottom = 0)
    axs[2, 1].set_ylim(bottom = 0)
    axs[2, 2].set_ylim(bottom = 0)
    axs[2, 3].set_ylim(bottom = 0)
    axs[2, 4].set_ylim(bottom = 0)

    plt.show()

def turn_rate_v_stats(use_prior = True, N = 100):
    """
    Plots turnover rate on the x-axis and the value for various
    tree statistics on the y-axis. 'use_prior' determines whether the
    turnover rate will grow incrementally (if 'use_prior' is false)
    or will be sampled from the prior distribution each time (if 'use_prior'
    is true). 'N' is the number of trees generated from different turnover
    rates from which statistics will be calculated. 'N' must be <= 20 if
    'use_prior' is false so that the turnover rate will be within [0,1].
    """
    r_rate = 0.0005 # initial turn rate (used if 'use_prior' is false)
    r_arr = []
    b_sum_arr = []
    b_mean_arr = []
    b_median_arr = []
    b_variance_arr = []
    height_arr = []
    d_mean_arr = []
    d_median_arr = []
    d_variance_arr = []
    balance_arr = []
    nleaf_arr = []
    root_colless_arr = []
    sum_colless_arr = []
    mean_colless_arr = []
    median_colless_arr = []
    variance_colless_arr = []

    # generate rates for non-turn parameters (stay constant for all trees generated below)
    d_rate = abc.gen_param(d_dist)
    sub_rate = abc.gen_param(sub_dist)
    birth_shape = abc.gen_param(birth_s)
    death_shape = abc.gen_param(death_s)
    sub_shape = abc.gen_param(sub_s)
    i = 0

    while(i < N): # 'N' is the number of rates drawn and trees generates
        if(use_prior):
            r_rate = abc.gen_param(r_dist)
            while(r_rate >= 1): 
                r_rate = abc.gen_param(r_dist)
        else: # if use_prior is false, N must be <= 20
            r_rate += i * 0.005
        r_arr.append(r_rate)

        t = abc.gen_tree_sims(d = d_rate, r = r_rate, sub_rate = sub_rate, birth_shape = birth_shape, death_shape = death_shape, sub_shape = sub_shape, is_prior = True)[0]
        #print(t)

        # calc tree stats
        b_sum_arr.append(math.log10(gt.tree_branch_sum(t)+1))
        b_mean_arr.append(math.log10(gt.tree_branch_mean(t)+1))
        b_median_arr.append(math.log10(gt.tree_branch_median(t)+1))
        b_variance_arr.append(math.log10(gt.tree_branch_variance(t)+1))
        height_arr.append(math.log10(gt.tree_height(t)+1))
        d_mean_arr.append(math.log10(gt.tree_depth_mean(t)+1))
        d_median_arr.append(math.log10(gt.tree_depth_median(t)+1))
        d_variance_arr.append(math.log10(gt.tree_depth_variance(t)+1))
        balance_arr.append(math.log10(gt.tree_balance(t)+1))
        nleaf_arr.append(math.log10(gt.tree_nleaf(t)+1))
        root_colless_arr.append(math.log10(gt.tree_root_colless(t)+1))
        sum_colless_arr.append(math.log10(gt.tree_sum_colless(t)+1))
        mean_colless_arr.append(math.log10(gt.tree_mean_colless(t)+1))
        median_colless_arr.append(math.log10(gt.tree_median_colless(t)+1))
        variance_colless_arr.append(math.log10(gt.tree_variance_colless(t)+1))
        i += 1

    # plot stats vs rates
    fig, axs = plt.subplots(3, 5)
    axs[0, 0].plot(r_arr, b_sum_arr, 'ro')
    axs[0, 0].set_title('Turn v branch sum')
    axs[0, 1].plot(r_arr, b_mean_arr, 'ro')
    axs[0, 1].set_title('Branch mean')
    axs[0, 2].plot(r_arr, b_median_arr, 'ro')
    axs[0, 2].set_title('Branch median')
    axs[0, 3].plot(r_arr, b_variance_arr, 'ro')
    axs[0, 3].set_title('Branch variance')
    axs[0, 4].plot(r_arr, height_arr, 'ro')
    axs[0, 4].set_title('Height')
    axs[1, 0].plot(r_arr, d_mean_arr, 'ro')
    axs[1, 0].set_title('Depth mean')
    axs[1, 1].plot(r_arr, d_median_arr, 'ro')
    axs[1, 1].set_title('Depth median')
    axs[1, 2].plot(r_arr, d_variance_arr, 'ro')
    axs[1, 2].set_title('Depth variance')
    axs[1, 3].plot(r_arr, balance_arr, 'ro')
    axs[1, 3].set_title('Balance')
    axs[1, 4].plot(r_arr, nleaf_arr, 'ro')
    axs[1, 4].set_title('Nleaf')
    axs[2, 0].plot(r_arr, root_colless_arr, 'ro')
    axs[2, 0].set_title('Root colless')
    axs[2, 1].plot(r_arr, sum_colless_arr, 'ro')
    axs[2, 1].set_title('Sum colless')
    axs[2, 2].plot(r_arr, mean_colless_arr, 'ro')
    axs[2, 2].set_title('Mean colless')
    axs[2, 3].plot(r_arr, median_colless_arr, 'ro')
    axs[2, 3].set_title('Median colless')
    axs[2, 4].plot(r_arr, variance_colless_arr, 'ro')
    axs[2, 4].set_title('Variance colless')
    
    axs[0, 0].set_ylim(bottom = 0)
    axs[0, 1].set_ylim(bottom = 0)
    axs[0, 2].set_ylim(bottom = 0)
    axs[0, 3].set_ylim(bottom = 0)
    axs[0, 4].set_ylim(bottom = 0)
    axs[1, 0].set_ylim(bottom = 0)
    axs[1, 1].set_ylim(bottom = 0)
    axs[1, 2].set_ylim(bottom = 0)
    axs[1, 3].set_ylim(bottom = 0)
    axs[1, 4].set_ylim(bottom = 0)
    axs[2, 0].set_ylim(bottom = 0)
    axs[2, 1].set_ylim(bottom = 0)
    axs[2, 2].set_ylim(bottom = 0)
    axs[2, 3].set_ylim(bottom = 0)
    axs[2, 4].set_ylim(bottom = 0)
    
    plt.show()

def sub_rate_v_stats(use_prior = True, N = 100):
    """
    Plots sub rate on the x-axis and the value for various
    tree statistics on the y-axis. 'use_prior' determines whether the
    sub rate will grow incrementally (if 'use_prior' is false)
    or will be sampled from the prior distribution each time (if 'use_prior'
    is true). 'N' is the number of trees generated from different sub
    rates from which statistics will be calculated.
    """
    sub_rate = 0.05 # initial sub rate (used if 'use_prior' is false)
    sub_arr = []
    b_sum_arr = []
    b_mean_arr = []
    b_median_arr = []
    b_variance_arr = []
    height_arr = []
    d_mean_arr = []
    d_median_arr = []
    d_variance_arr = []
    balance_arr = []
    nleaf_arr = []
    root_colless_arr = []
    sum_colless_arr = []
    mean_colless_arr = []
    median_colless_arr = []
    variance_colless_arr = []

    # generate rates for non-div parameters (stay constant for all trees generated below)
    d_rate = abc.gen_param(d_dist)
    r_rate = abc.gen_param(r_dist)
    while(r_rate >= 1): 
        r_rate = abc.gen_param(r_dist)
    birth_shape = abc.gen_param(birth_s)
    death_shape = abc.gen_param(death_s)
    sub_shape = abc.gen_param(sub_s)
    i = 0

    while(i < N): # 'N' is the number of rates drawn and trees generates
        if(use_prior):
            sub_rate = abc.gen_param(sub_dist)
        else:
            sub_rate += i * 0.005
        sub_arr.append(d_rate)

        t = abc.gen_tree_sims(d = d_rate, r = r_rate, sub_rate = sub_rate, birth_shape = birth_shape, death_shape = death_shape, sub_shape = sub_shape, is_prior = True)[0]
        # calc tree stats
        b_sum_arr.append(math.log10(gt.tree_branch_sum(t)+1))
        b_mean_arr.append(math.log10(gt.tree_branch_mean(t)+1))
        b_median_arr.append(math.log10(gt.tree_branch_median(t)+1))
        b_variance_arr.append(math.log10(gt.tree_branch_variance(t)+1))
        height_arr.append(math.log10(gt.tree_height(t)+1))
        d_mean_arr.append(math.log10(gt.tree_depth_mean(t)+1))
        d_median_arr.append(math.log10(gt.tree_depth_median(t)+1))
        d_variance_arr.append(math.log10(gt.tree_depth_variance(t)+1))
        balance_arr.append(math.log10(gt.tree_balance(t)+1))
        nleaf_arr.append(math.log10(gt.tree_nleaf(t)+1))
        root_colless_arr.append(math.log10(gt.tree_root_colless(t)+1))
        sum_colless_arr.append(math.log10(gt.tree_sum_colless(t)+1))
        mean_colless_arr.append(math.log10(gt.tree_mean_colless(t)+1))
        median_colless_arr.append(math.log10(gt.tree_median_colless(t)+1))
        variance_colless_arr.append(math.log10(gt.tree_variance_colless(t)+1))
        i += 1
    
    # plot stats vs rates
    fig, axs = plt.subplots(3, 5)
    axs[0, 0].plot(sub_arr, b_sum_arr, 'ro')
    axs[0, 0].set_title('Div v branch sum')
    axs[0, 1].plot(sub_arr, b_mean_arr, 'ro')
    axs[0, 1].set_title('Branch mean')
    axs[0, 2].plot(sub_arr, b_median_arr, 'ro')
    axs[0, 2].set_title('Branch median')
    axs[0, 3].plot(sub_arr, b_variance_arr, 'ro')
    axs[0, 3].set_title('Branch variance')
    axs[0, 4].plot(sub_arr, height_arr, 'ro')
    axs[0, 4].set_title('Height')
    axs[1, 0].plot(sub_arr, d_mean_arr, 'ro')
    axs[1, 0].set_title('Depth mean')
    axs[1, 1].plot(sub_arr, d_median_arr, 'ro')
    axs[1, 1].set_title('Depth median')
    axs[1, 2].plot(sub_arr, d_variance_arr, 'ro')
    axs[1, 2].set_title('Depth variance')
    axs[1, 3].plot(sub_arr, balance_arr, 'ro')
    axs[1, 3].set_title('Balance')
    axs[1, 4].plot(sub_arr, nleaf_arr, 'ro')
    axs[1, 4].set_title('Nleaf')
    axs[2, 0].plot(sub_arr, root_colless_arr, 'ro')
    axs[2, 0].set_title('Root colless')
    axs[2, 1].plot(sub_arr, sum_colless_arr, 'ro')
    axs[2, 1].set_title('Sum colless')
    axs[2, 2].plot(sub_arr, mean_colless_arr, 'ro')
    axs[2, 2].set_title('Mean colless')
    axs[2, 3].plot(sub_arr, median_colless_arr, 'ro')
    axs[2, 3].set_title('Median colless')
    axs[2, 4].plot(sub_arr, variance_colless_arr, 'ro')
    axs[2, 4].set_title('Variance colless')

    axs[0, 0].set_ylim(bottom = 0)
    axs[0, 1].set_ylim(bottom = 0)
    axs[0, 2].set_ylim(bottom = 0)
    axs[0, 3].set_ylim(bottom = 0)
    axs[0, 4].set_ylim(bottom = 0)
    axs[1, 0].set_ylim(bottom = 0)
    axs[1, 1].set_ylim(bottom = 0)
    axs[1, 2].set_ylim(bottom = 0)
    axs[1, 3].set_ylim(bottom = 0)
    axs[1, 4].set_ylim(bottom = 0)
    axs[2, 0].set_ylim(bottom = 0)
    axs[2, 1].set_ylim(bottom = 0)
    axs[2, 2].set_ylim(bottom = 0)
    axs[2, 3].set_ylim(bottom = 0)
    axs[2, 4].set_ylim(bottom = 0)

    plt.show()






def birth_shape_v_stats(use_prior = True, N = 100):
    """
    Plots birth shape on the x-axis and the value for various
    tree statistics on the y-axis. 'use_prior' determines whether the
    birth shape will grow incrementally (if 'use_prior' is false)
    or will be sampled from the prior distribution each time (if 'use_prior'
    is true). 'N' is the number of trees generated from different birth shapes
    from which statistics will be calculated.
    """ 
    birth_shape = 1 # initial birth shape (used if 'use_prior' is false)
    birth_s_arr = []
    b_sum_arr = []
    b_mean_arr = []
    b_median_arr = []
    b_variance_arr = []
    height_arr = []
    d_mean_arr = []
    d_median_arr = []
    d_variance_arr = []
    balance_arr = []
    nleaf_arr = []
    root_colless_arr = []
    sum_colless_arr = []
    mean_colless_arr = []
    median_colless_arr = []
    variance_colless_arr = []

    # generate rates for non-birth parameters (stay constant for all trees generated below)
    d_rate = abc.gen_param(d_dist)
    r_rate = abc.gen_param(r_dist)
    while(r_rate >= 1): 
        r_rate = abc.gen_param(r_dist)
    sub_rate = abc.gen_param(sub_dist)
    death_shape = abc.gen_param(death_s)
    sub_shape = abc.gen_param(sub_s)
    i = 0

    while(i < N): # 'N' is the number of rates drawn and trees generates
        if(use_prior):
            birth_shape = abc.gen_param(birth_s)
        else:
            birth_shape += i * .2
        birth_s_arr.append(birth_shape)
        t = abc.gen_tree_sims(d = d_rate, r = r_rate, sub_rate= sub_rate, birth_shape = birth_shape, death_shape = death_shape, sub_shape = sub_shape, is_prior = True)[0]
        #print(t)

        # calc tree stats
        b_sum_arr.append(math.log10(gt.tree_branch_sum(t)+1))
        b_mean_arr.append(math.log10(gt.tree_branch_mean(t)+1))
        b_median_arr.append(math.log10(gt.tree_branch_median(t)+1))
        b_variance_arr.append(math.log10(gt.tree_branch_variance(t)+1))
        height_arr.append(math.log10(gt.tree_height(t)+1))
        d_mean_arr.append(math.log10(gt.tree_depth_mean(t)+1))
        d_median_arr.append(math.log10(gt.tree_depth_median(t)+1))
        d_variance_arr.append(math.log10(gt.tree_depth_variance(t)+1))
        balance_arr.append(math.log10(gt.tree_balance(t)+1))
        nleaf_arr.append(math.log10(gt.tree_nleaf(t)+1))
        root_colless_arr.append(math.log10(gt.tree_root_colless(t)+1))
        sum_colless_arr.append(math.log10(gt.tree_sum_colless(t)+1))
        mean_colless_arr.append(math.log10(gt.tree_mean_colless(t)+1))
        median_colless_arr.append(math.log10(gt.tree_median_colless(t)+1))
        variance_colless_arr.append(math.log10(gt.tree_variance_colless(t)+1))
        i += 1

    # plot stats vs rates
    fig, axs = plt.subplots(3, 5)
    axs[0, 0].plot(birth_s_arr, b_sum_arr, 'ro')
    axs[0, 0].set_title('Birth shape v branch sum')
    axs[0, 1].plot(birth_s_arr, b_mean_arr, 'ro')
    axs[0, 1].set_title('Branch mean')
    axs[0, 2].plot(birth_s_arr, b_median_arr, 'ro')
    axs[0, 2].set_title('Branch median')
    axs[0, 3].plot(birth_s_arr, b_variance_arr, 'ro')
    axs[0, 3].set_title('Branch variance')
    axs[0, 4].plot(birth_s_arr, height_arr, 'ro')
    axs[0, 4].set_title('Height')
    axs[1, 0].plot(birth_s_arr, d_mean_arr, 'ro')
    axs[1, 0].set_title('Depth mean')
    axs[1, 1].plot(birth_s_arr, d_median_arr, 'ro')
    axs[1, 1].set_title('Depth median')
    axs[1, 2].plot(birth_s_arr, d_variance_arr, 'ro')
    axs[1, 2].set_title('Depth variance')
    axs[1, 3].plot(birth_s_arr, balance_arr, 'ro')
    axs[1, 3].set_title('Balance')
    axs[1, 4].plot(birth_s_arr, nleaf_arr, 'ro')
    axs[1, 4].set_title('Nleaf')
    axs[2, 0].plot(birth_s_arr, root_colless_arr, 'ro')
    axs[2, 0].set_title('Root colless')
    axs[2, 1].plot(birth_s_arr, sum_colless_arr, 'ro')
    axs[2, 1].set_title('Sum colless')
    axs[2, 2].plot(birth_s_arr, mean_colless_arr, 'ro')
    axs[2, 2].set_title('Mean colless')
    axs[2, 3].plot(birth_s_arr, median_colless_arr, 'ro')
    axs[2, 3].set_title('Median colless')
    axs[2, 4].plot(birth_s_arr, variance_colless_arr, 'ro')
    axs[2, 4].set_title('Variance colless')

    axs[0, 0].set_ylim(bottom = 0)
    axs[0, 1].set_ylim(bottom = 0)
    axs[0, 2].set_ylim(bottom = 0)
    axs[0, 3].set_ylim(bottom = 0)
    axs[0, 4].set_ylim(bottom = 0)
    axs[1, 0].set_ylim(bottom = 0)
    axs[1, 1].set_ylim(bottom = 0)
    axs[1, 2].set_ylim(bottom = 0)
    axs[1, 3].set_ylim(bottom = 0)
    axs[1, 4].set_ylim(bottom = 0)
    axs[2, 0].set_ylim(bottom = 0)
    axs[2, 1].set_ylim(bottom = 0)
    axs[2, 2].set_ylim(bottom = 0)
    axs[2, 3].set_ylim(bottom = 0)
    axs[2, 4].set_ylim(bottom = 0)

    plt.show()

def death_shape_v_stats(use_prior = True, N = 100):
    """
    Plots death shape on the x-axis and the value for various
    tree statistics on the y-axis. 'use_prior' determines whether the
    death shape will grow incrementally (if 'use_prior' is false)
    or will be sampled from the prior distribution each time (if 'use_prior'
    is true). 'N' is the number of trees generated from different death shapes
    from which statistics will be calculated.
    """ 
    death_shape = 1 # initial death shape (used if 'use_prior' is false)
    death_s_arr = []
    b_sum_arr = []
    b_mean_arr = []
    b_median_arr = []
    b_variance_arr = []
    height_arr = []
    d_mean_arr = []
    d_median_arr = []
    d_variance_arr = []
    balance_arr = []
    nleaf_arr = []
    root_colless_arr = []
    sum_colless_arr = []
    mean_colless_arr = []
    median_colless_arr = []
    variance_colless_arr = []

    # generate rates for non-death parameters (stay constant for all trees generated below)
    d_rate = abc.gen_param(d_dist)
    r_rate = abc.gen_param(r_dist)
    while(r_rate >= 1): 
        r_rate = abc.gen_param(r_dist)
    sub_rate = abc.gen_param(sub_dist)
    birth_shape = abc.gen_param(birth_s)
    sub_shape = abc.gen_param(sub_s)
    i = 0

    while(i < N): # 'N' is the number of rates drawn and trees generates
        if(use_prior):
            death_shape = abc.gen_param(death_s)
        else:
            death_shape += i * .2
        death_s_arr.append(death_shape)
        
        t = abc.gen_tree_sims(d = d_rate, r = r_rate, sub_rate = sub_rate,birth_shape = birth_shape, death_shape = death_shape, sub_shape = sub_shape, is_prior = True)[0]
        #print(t)

        # calc tree stats
        b_sum_arr.append(math.log10(gt.tree_branch_sum(t)+1))
        b_mean_arr.append(math.log10(gt.tree_branch_mean(t)+1))
        b_median_arr.append(math.log10(gt.tree_branch_median(t)+1))
        b_variance_arr.append(math.log10(gt.tree_branch_variance(t)+1))
        height_arr.append(math.log10(gt.tree_height(t)+1))
        d_mean_arr.append(math.log10(gt.tree_depth_mean(t)+1))
        d_median_arr.append(math.log10(gt.tree_depth_median(t)+1))
        d_variance_arr.append(math.log10(gt.tree_depth_variance(t)+1))
        balance_arr.append(math.log10(gt.tree_balance(t)+1))
        nleaf_arr.append(math.log10(gt.tree_nleaf(t)+1))
        root_colless_arr.append(math.log10(gt.tree_root_colless(t)+1))
        sum_colless_arr.append(math.log10(gt.tree_sum_colless(t)+1))
        mean_colless_arr.append(math.log10(gt.tree_mean_colless(t)+1))
        median_colless_arr.append(math.log10(gt.tree_median_colless(t)+1))
        variance_colless_arr.append(math.log10(gt.tree_variance_colless(t)+1))
        i += 1
    
    # plot stats vs rates
    fig, axs = plt.subplots(3, 5)
    axs[0, 0].plot(death_s_arr, b_sum_arr, 'ro')
    axs[0, 0].set_title('Death shape v branch sum')
    axs[0, 1].plot(death_s_arr, b_mean_arr, 'ro')
    axs[0, 1].set_title('Branch mean')
    axs[0, 2].plot(death_s_arr, b_median_arr, 'ro')
    axs[0, 2].set_title('Branch median')
    axs[0, 3].plot(death_s_arr, b_variance_arr, 'ro')
    axs[0, 3].set_title('Branch variance')
    axs[0, 4].plot(death_s_arr, height_arr, 'ro')
    axs[0, 4].set_title('Height')
    axs[1, 0].plot(death_s_arr, d_mean_arr, 'ro')
    axs[1, 0].set_title('Depth mean')
    axs[1, 1].plot(death_s_arr, d_median_arr, 'ro')
    axs[1, 1].set_title('Depth median')
    axs[1, 2].plot(death_s_arr, d_variance_arr, 'ro')
    axs[1, 2].set_title('Depth variance')
    axs[1, 3].plot(death_s_arr, balance_arr, 'ro')
    axs[1, 3].set_title('Balance')
    axs[1, 4].plot(death_s_arr, nleaf_arr, 'ro')
    axs[1, 4].set_title('Nleaf')
    axs[2, 0].plot(death_s_arr, root_colless_arr, 'ro')
    axs[2, 0].set_title('Root colless')
    axs[2, 1].plot(death_s_arr, sum_colless_arr, 'ro')
    axs[2, 1].set_title('Sum colless')
    axs[2, 2].plot(death_s_arr, mean_colless_arr, 'ro')
    axs[2, 2].set_title('Mean colless')
    axs[2, 3].plot(death_s_arr, median_colless_arr, 'ro')
    axs[2, 3].set_title('Median colless')
    axs[2, 4].plot(death_s_arr, variance_colless_arr, 'ro')
    axs[2, 4].set_title('Variance colless')
    
    axs[0, 0].set_ylim(bottom = 0)
    axs[0, 1].set_ylim(bottom = 0)
    axs[0, 2].set_ylim(bottom = 0)
    axs[0, 3].set_ylim(bottom = 0)
    axs[0, 4].set_ylim(bottom = 0)
    axs[1, 0].set_ylim(bottom = 0)
    axs[1, 1].set_ylim(bottom = 0)
    axs[1, 2].set_ylim(bottom = 0)
    axs[1, 3].set_ylim(bottom = 0)
    axs[1, 4].set_ylim(bottom = 0)
    axs[2, 0].set_ylim(bottom = 0)
    axs[2, 1].set_ylim(bottom = 0)
    axs[2, 2].set_ylim(bottom = 0)
    axs[2, 3].set_ylim(bottom = 0)
    axs[2, 4].set_ylim(bottom = 0)

    plt.show()

def sub_shape_v_stats(use_prior = True, N = 100):
    """
    Plots sub shape on the x-axis and the value for various
    tree statistics on the y-axis. 'use_prior' determines whether the
    sub shape will grow incrementally (if 'use_prior' is false)
    or will be sampled from the prior distribution each time (if 'use_prior'
    is true). 'N' is the number of trees generated from different sub shapes
    from which statistics will be calculated.
    """ 
    sub_shape = 1 # initial sub shape (used if 'use_prior' is false)
    sub_s_arr = []
    b_sum_arr = []
    b_mean_arr = []
    b_median_arr = []
    b_variance_arr = []
    height_arr = []
    d_mean_arr = []
    d_median_arr = []
    d_variance_arr = []
    balance_arr = []
    nleaf_arr = []
    root_colless_arr = []
    sum_colless_arr = []
    mean_colless_arr = []
    median_colless_arr = []
    variance_colless_arr = []

    # generate rates for non-sub parameters (stay constant for all trees generated below)
    d_rate = abc.gen_param(d_dist)
    r_rate = abc.gen_param(r_dist)
    while(r_rate >= 1): 
        r_rate = abc.gen_param(r_dist)
    sub_rate = abc.gen_param(sub_dist)
    birth_shape = abc.gen_param(birth_s)
    death_shape = abc.gen_param(death_s)
    i = 0

    while(i < N): # 'N' is the number of rates drawn and trees generates
        if(use_prior):
            sub_shape = abc.gen_param(sub_s)
        else:
            sub_shape += i * .2
        sub_s_arr.append(sub_shape)

        t = abc.gen_tree_sims(d = d_rate, r = r_rate, sub_rate = sub_rate, birth_shape = birth_shape, death_shape = death_shape, sub_shape = sub_shape, is_prior = True)[0]
        
        #print(t)

        # calc tree stats
        b_sum_arr.append(math.log10(gt.tree_branch_sum(t)+1))
        b_mean_arr.append(math.log10(gt.tree_branch_mean(t)+1))
        b_median_arr.append(math.log10(gt.tree_branch_median(t)+1))
        b_variance_arr.append(math.log10(gt.tree_branch_variance(t)+1))
        height_arr.append(math.log10(gt.tree_height(t)+1))
        d_mean_arr.append(math.log10(gt.tree_depth_mean(t)+1))
        d_median_arr.append(math.log10(gt.tree_depth_median(t)+1))
        d_variance_arr.append(math.log10(gt.tree_depth_variance(t)+1))
        balance_arr.append(math.log10(gt.tree_balance(t)+1))
        nleaf_arr.append(math.log10(gt.tree_nleaf(t)+1))
        root_colless_arr.append(math.log10(gt.tree_root_colless(t)+1))
        sum_colless_arr.append(math.log10(gt.tree_sum_colless(t)+1))
        mean_colless_arr.append(math.log10(gt.tree_mean_colless(t)+1))
        median_colless_arr.append(math.log10(gt.tree_median_colless(t)+1))
        variance_colless_arr.append(math.log10(gt.tree_variance_colless(t)+1))
        i += 1
    
    # plot stats vs rates
    fig, axs = plt.subplots(3, 5)
    axs[0, 0].plot(sub_s_arr, b_sum_arr, 'ro')
    axs[0, 0].set_title('Sub shape v branch sum')
    axs[0, 1].plot(sub_s_arr, b_mean_arr, 'ro')
    axs[0, 1].set_title('Branch mean')
    axs[0, 2].plot(sub_s_arr, b_median_arr, 'ro')
    axs[0, 2].set_title('Branch median')
    axs[0, 3].plot(sub_s_arr, b_variance_arr, 'ro')
    axs[0, 3].set_title('Branch variance')
    axs[0, 4].plot(sub_s_arr, height_arr, 'ro')
    axs[0, 4].set_title('Height')
    axs[1, 0].plot(sub_s_arr, d_mean_arr, 'ro')
    axs[1, 0].set_title('Depth mean')
    axs[1, 1].plot(sub_s_arr, d_median_arr, 'ro')
    axs[1, 1].set_title('Depth median')
    axs[1, 2].plot(sub_s_arr, d_variance_arr, 'ro')
    axs[1, 2].set_title('Depth variance')
    axs[1, 3].plot(sub_s_arr, balance_arr, 'ro')
    axs[1, 3].set_title('Balance')
    axs[1, 4].plot(sub_s_arr, nleaf_arr, 'ro')
    axs[1, 4].set_title('Nleaf')
    axs[2, 0].plot(sub_s_arr, root_colless_arr, 'ro')
    axs[2, 0].set_title('Root colless')
    axs[2, 1].plot(sub_s_arr, sum_colless_arr, 'ro')
    axs[2, 1].set_title('Sum colless')
    axs[2, 2].plot(sub_s_arr, mean_colless_arr, 'ro')
    axs[2, 2].set_title('Mean colless')
    axs[2, 3].plot(sub_s_arr, median_colless_arr, 'ro')
    axs[2, 3].set_title('Median colless')
    axs[2, 4].plot(sub_s_arr, variance_colless_arr, 'ro')
    axs[2, 4].set_title('Variance colless')

    axs[0, 0].set_ylim(bottom = 0)
    axs[0, 1].set_ylim(bottom = 0)
    axs[0, 2].set_ylim(bottom = 0)
    axs[0, 3].set_ylim(bottom = 0)
    axs[0, 4].set_ylim(bottom = 0)
    axs[1, 0].set_ylim(bottom = 0)
    axs[1, 1].set_ylim(bottom = 0)
    axs[1, 2].set_ylim(bottom = 0)
    axs[1, 3].set_ylim(bottom = 0)
    axs[1, 4].set_ylim(bottom = 0)
    axs[2, 0].set_ylim(bottom = 0)
    axs[2, 1].set_ylim(bottom = 0)
    axs[2, 2].set_ylim(bottom = 0)
    axs[2, 3].set_ylim(bottom = 0)
    axs[2, 4].set_ylim(bottom = 0)

    plt.show()

div_rate_v_stats()
turn_rate_v_stats()
sub_rate_v_stats()
birth_shape_v_stats()
death_shape_v_stats()
sub_shape_v_stats()