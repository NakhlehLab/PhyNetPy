import growtree
import elfi
import scipy


d_dist = elfi.Prior(scipy.stats.expon, 0, 10) # prior distribution for diversification
r_dist = elfi.Prior(scipy.stats.uniform, 0, 1) # prior distribution for turnover
sub_dist = elfi.Prior(scipy.stats.uniform, 5, 10) # prior distribution for sub
birth_s = elfi.Prior(scipy.stats.expon, 0, 25) # prior distribution for birth distribution shape
death_s = elfi.Prior(scipy.stats.expon, 0, 25) # prior distribution for death distribution shape
sub_s = elfi.Prior(scipy.stats.expon, 0, 25)

def calc_rates_bd(d, r):
    birth_calc = d / (1 - r) # calculate birth rate from 'd' and 'r'
    death_calc = r * birth_calc # calculate death rate from calculated birth rate and 'r'
    return [birth_calc, death_calc] # return birth and death rates in an array

def gen_param(prior_dist):
    return (prior_dist.generate())[0]

for i in range(0,100):
    d_drawn = gen_param(d_dist)  
    r_drawn = gen_param(r_dist)
    while(r_drawn>=1):
        r_drawn = gen_param(r_dist)
    s_drawn = gen_param(sub_dist)
    print("INITIAL SUB RATE: " + str(s_drawn))
    birth_shape = gen_param(birth_s)
    death_shape = gen_param(death_s)
    sub_shape = gen_param(sub_s)
    rate_arr = calc_rates_bd(d_drawn, r_drawn) 
    birth = rate_arr[0] 
    death = rate_arr[1]
    print("INITIAL SUB RATE: " + str(s_drawn / (s_drawn + birth + death)))
    t = growtree.gen_tree(b = birth, d = death, s = s_drawn, shape_b = birth_shape, shape_d = death_shape, shape_s = sub_shape, branch_info = 1, seq_length = 100, goal_leaves=32, sampling_rate=0.01)
    print(growtree.tree_height(t))
    #growtree.outputNewick(t, "NWtree")
