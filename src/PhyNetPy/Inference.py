
import sys
sys.path.insert(0, 'src/PhyNetPy')


from Bayesian import NetworkBuilder2 as nb
from Bayesian import MSA
from Bayesian import MetropolisHastings as mh
from Bayesian import Alphabet as a
from Bayesian import BirthDeath as bd
from Bayesian import Matrix as m
from Bayesian import ModelGraph as mg
from Bayesian import GTR 
from Bayesian import SNPModule as snpm
# import elfi
import scipy
# from ABC.SimulatedTree import SimulatedTree
# from ABC.Simulator import *
# from ABC.TreeStatistics import *
# from matplotlib import pyplot as plt
# import cProfile as p
    
    
def SNAPP_Likelihood1(filename: str, u :float , v:float, coal:float, grouping:dict=None, auto_detect:bool = False, summary_path:str = None, network_path:str = None) -> float:
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


def SNAPP_Likelihood(data_filename: str, u :float , v:float, coal:float, ploidy : list = None, grouping:dict=None, auto_detect:bool = False, show_partials:bool = False) -> list:
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
    aln = MSA.MSA(data_filename, sec_2_ploidy=ploidy, grouping=grouping, grouping_auto_detect = auto_detect, dtype = "SNP")
    
    #Read and parse the network described 
    networks = nb.NetworkBuilder2(data_filename).get_all_networks()
    
    likelihoods = []
    
    
    snp_params={"samples": aln.total_samples(), "u": u, "v": v, "coal" : coal, "grouping": grouping is not None}
    data_matrix = m.Matrix(aln, a.Alphabet("SNP", snp_ploidy=snp_params["samples"]))
    # profiler = p.Profile()
    # profiler.enable()
    for network in networks:

        #Create model
        snp_model = mg.Model(network, data_matrix, None, snp_params=snp_params, verbose = False)
        #Compute the likelihood
        
        likelihood = snp_model.SNP_likelihood()
        likelihoods.append(likelihood)
        
        
        
    # profiler.disable()
    # profiler.print_stats()
 
    return likelihoods





# cp = p.Profile()
# cp.enable()

#print(SNAPP_Likelihood('/Users/mak17/Documents/PhyloGenPy/PhyloGenPy/src/PhyNetPy/test/files/lvl2_network.nex', 1, 1, .005, ploidy = [3,3,3,3]))
print(SNAPP_Likelihood('/Users/mak17/Documents/PhyloGenPy/PhyloGenPy/src/PhyNetPy/test/files/lvl1_network.nex', 1, 1, .005, ploidy = [2,3,2]))
# cp.disable()
# cp.print_stats(sort="tottime")
#print(SNAPP_Likelihood('src/PhyNetPy/test/files/snptest_ez.nex', 1, 1, .2, ploidy = [1,1,1]))