from PhyNetPy.Data.Matrix import Matrix
from Bio import AlignIO
from PhyNetPy.Network.NetworkBuilder import NetworkBuilder
from PhyNetPy.Data.GTR import *
from PhyNetPy.Data.Alphabet import Alphabet
from PhyNetPy.Simulation.BirthDeath import CBDP
from PhyNetPy.Data.MSA import MSA
from MetropolisHastings import ProposalKernel, MetropolisHastings
from ModelGraph import Model



    
    
    
    
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
    network = CBDP(1, .5, aln.num_groups()).generateTree()
    
    snp_params={"samples": len(aln.get_records()), "u": u, "v": v, "coal" : coal, "grouping":True}
    m = Matrix(aln, Alphabet("SNP"))
    snp_model = Model(network, m, snp_params=snp_params)
    
    mh = MetropolisHastings(ProposalKernel(), JC(), m, 800, snp_model) #TODO: Submodel unnecessary for snp. make optional?
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
    aln = MSA(filename, grouping=grouping, grouping_auto_detect = auto_detect)
    
    #Read and parse the network described 
    networks = NetworkBuilder(filename).get_all_networks()
    
    likelihoods = []
    for network in networks:
        snp_params={"samples": len(aln.get_records()), "u": u, "v": v, "coal" : coal, "grouping":False}
        #Create model
        snp_model = Model(network, Matrix(aln, Alphabet("SNP")), None, snp_params=snp_params, verbose = show_partials)
        #Compute the likelihood
        likelihoods.append(snp_model.SNP_likelihood())
 
    return likelihoods


