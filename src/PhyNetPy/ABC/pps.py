import abc_tree
import growtree as gt
import matplotlib.pyplot as plt

rates_arr = abc_tree.run_main(is_summary = True, is_plot = True, num_accept=100) # get array of inferred rates from ABC

# extract specific arrays of inferred rates
d_rate_arr = rates_arr[0]
r_rate_arr = rates_arr[1]
sub_rate_arr = rates_arr[2]
birth_s_arr = rates_arr[3]
death_s_arr = rates_arr[4]
sub_s_arr = rates_arr[5]
obs_tree = rates_arr[6]

n_sims = len(d_rate_arr) # number of trees that will be simulated from the rates that make up the posterior distributions
obs_nleaves = gt.tree_nleaf(obs_tree)

i = 0 
branch_sum_arr = [] # will hold the sum of branch lengths of the posterior simulated trees 
height_arr = [] # will hold the heights of the posterior simulated trees
depth_mean_arr = [] # will hold the mean of depths of the posterior simulated trees 
sum_colless_arr = [] # will hold the sum of the colless indices of the posterior simulated trees 

# calculating the observed statistics
branch_sum_obs = gt.tree_branch_sum(obs_tree)
height_obs = gt.tree_height(obs_tree)
depth_mean_obs = gt.tree_depth_mean(obs_tree)
sum_colless_obs = gt.tree_sum_colless(obs_tree)

while i < n_sims: # generating posterior simulated trees and calculating statistics 
        # generating the posterior simulated tree (using rates from the posterior distributions)
    sim_tree = abc_tree.gen_tree_sims(d = d_rate_arr[i], r = r_rate_arr[i], sub_rate = sub_rate_arr[i], birth_shape = birth_s_arr[i], death_shape = death_s_arr[i], sub_shape = sub_s_arr[i], is_prior = True, leaf_goal=obs_nleaves)[0]
    i += 1

    # calculating the summary statistics on the posterior simulated tree 
    branch_sum_arr.append(gt.tree_branch_sum(sim_tree))
    height_arr.append(gt.tree_height(sim_tree))
    depth_mean_arr.append(gt.tree_depth_mean(sim_tree))
    sum_colless_arr.append(gt.tree_sum_colless(sim_tree))

# plotting the distributions for the statistics calculated on the posterior simulated trees
# with the value of the statistic for the observed tree plotted as a point

fig, axs = plt.subplots(2, 3)
axs[0, 0].hist(branch_sum_arr, bins = 50)
axs[0, 0].plot(branch_sum_obs, 1, marker = "o", markersize = 5) # plot observed statistic point
axs[0, 0].set_title('Sum of branch lengths of posterior simulated trees')
#axs[0, 0].set_title('Distribution of mean branch lengths for posterior simulated trees compared to the observed tree')
axs[0, 1].hist(height_arr, bins = 50)
axs[0, 1].plot(height_obs, 1, marker = "o", markersize = 5) # plot observed statistic point
axs[0, 1].set_title('Height')
axs[0, 2].hist(depth_mean_arr, bins = 50)
axs[0, 2].plot(depth_mean_obs, 1, marker = "o", markersize = 5) # plot observed statistic point
axs[0, 2].set_title('Mean of depth')
axs[1, 0].hist(sum_colless_arr, bins = 50)
axs[1, 0].plot(sum_colless_obs, 1, marker = "o", markersize = 5) # plot observed statistic point
axs[1, 0].set_title('Sum of colless indices')

plt.show()