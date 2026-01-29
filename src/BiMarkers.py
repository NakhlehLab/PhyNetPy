#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##
##  See "LICENSE.txt" for terms and conditions of usage.
##
##  If you use this work or any portion thereof in published work,
##  please cite it as:
##
##     Mark Kessler, Luay Nakhleh. 2025.
##
##############################################################################


""" 
Author : Mark Kessler
Last Edit : 1/23/26
First Included in Version : 1.0.0

CUDA-accelerated
Optimized for NVIDIA GPUs (tested with RTX 5070ti)

Docs   - [ ]
Tests  - [ ]
Design - [ ]
"""

from math import sqrt, comb, pow
from typing import Callable
from numba.core import base
import numpy as np
import scipy
from scipy.linalg import expm
import math


# CUDA imports - with graceful fallback
# Numba CUDA requires CUDA Toolkit 12.x (tested with CUDA 12.2)
# Both Numba and CuPy are used for GPU acceleration

CUPY_AVAILABLE = False
CUPY_RUNTIME_OK = False
NUMBA_CUDA_AVAILABLE = False

try:
    import cupy as cp
    # Check if CuPy can actually access a GPU
    CUPY_AVAILABLE = cp.cuda.is_available()
    if CUPY_AVAILABLE:
        # Verify CuPy can actually run operations (CUDA version compatibility check)
        # Use cp.random which requires kernel compilation - this catches CUDA DLL mismatches
        try:
            _test = cp.random.rand(10, dtype=cp.float64)
            _ = cp.sum(_test)
            del _test
            CUPY_RUNTIME_OK = True
        except (RuntimeError, Exception) as e:
            # CuPy may detect GPU but fail at runtime due to CUDA version mismatch
            # Common error: missing nvrtc64_XXX_0.dll for wrong CUDA toolkit version
            CUPY_RUNTIME_OK = False
except ImportError:
    cp = None

try:
    from numba import cuda, float64, int32, int64
    # Check if numba CUDA is actually available (requires compatible toolkit)
    NUMBA_CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    # Create mock decorators for when numba CUDA is not available
    class MockCuda:
        @staticmethod
        def is_available():
            return False
        @staticmethod
        def jit(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        @staticmethod
        def to_device(arr):
            return arr
        @staticmethod
        def device_array(*args, **kwargs):
            return np.zeros(args[0])
        @staticmethod
        def select_device(n):
            pass
        @staticmethod
        def grid(n):
            return 0
        class atomic:
            @staticmethod
            def add(*args):
                pass
    cuda = MockCuda()
    float64 = float
    int32 = int
    int64 = int

# Combined availability: CuPy for array ops, numba for custom kernels
# Use CUPY_RUNTIME_OK for actual operations, CUPY_AVAILABLE just means import worked
CUDA_IMPORTS_AVAILABLE = CUPY_RUNTIME_OK or NUMBA_CUDA_AVAILABLE

# Relative imports
from .MSA import MSA, DataSequence
from .BirthDeath import CBDP
from .NetworkParser import NetworkParser
from .Alphabet import Alphabet
from .Matrix import Matrix
# from .ModelGraph import (
#     Model, ModelNode, CalculationNode, Parameter, Accumulator, ExtantSpecies
# )
from .ModelFactory2 import *
from .Network import *
from .MetropolisHastings import MetropolisHastings, ProposalKernel
from .Visitor import *
from .Strategy import *
from .Executor import *
from .Traversal import *
from .ModelGraph2 import *


"""
SOURCES:

(1): 
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005932

CUDA Implementation Notes:
- Rules 0-4 are implemented as CUDA kernels for parallel execution
- Site-level parallelism is exploited (each site computed independently)
- Combinatorial loops are parallelized across GPU threads
- Matrix operations use CuPy for GPU-accelerated linear algebra
"""

# Global flag to toggle CUDA acceleration
global use_cuda
use_cuda = True

# CUDA device configuration
THREADS_PER_BLOCK = 256
MAX_BLOCKS = 65535

def n_to_index(n : int) -> int:
    """
    Computes the starting index in computing a linear index for an (n,r) pair.
    Returns the index, if r is 0.
    
    i.e n=1 returns 0, since (1,0) is index 0
    i.e n=3 returns 5 since (3,0) is preceded by 
        (1,0), (1,1), (2,0), (2,1), and (2,2)

    Args:
        n (int): an n value (number of lineages) from an (n,r) pair
    Returns:
        int: starting index for that block of n values
    """
    return int(.5 * (n - 1) * (n + 2))

def index_to_nr(index : int) -> list[int]:
    """
    Takes an index from the linear vector and turns it into an (n,r) pair
    
    i.e 7 -> [3,2]

    Args:
        index (int): the index

    Returns:
        list[int]: a 2-tuple (n,r)
    """
    a = 1
    b = 1
    c = -2 - 2 * index
    d = (b ** 2) - (4 * a * c)
    sol = (-b + sqrt(d)) / (2 * a)
    n = int(sol)
    r = index - n_to_index(n)

    return [n, r]

def nr_to_index(n : int, r : int) -> int:
    """
    Takes an (n,r) pair and maps it to a 1d vector index

    (1,0) -> 0
    (1,1) -> 1
    (2,0) -> 2
    ...
    
    Args:
        n (int): the number of lineages
        r (int): the number of red lineages (<= n)

    Returns:
        int: the index into the linear vector, that represents by (n, r)
    """
    
    return n_to_index(n) + r
 
def to_array(Fb_map : dict, 
             vector_len : int, 
             site_count : int) -> np.ndarray:
    """
    Takes a vpi/partial likelihood mapping, and translates it into a matrix
    such that the columns denote the site, and the row indeces correspond to 
    (n,r) pairs.
    
    This function serves the purpose of formatting the likelihoods at the root 
    for easy computation.

    Args:
        Fb_map (dict): vpi/partial likelihood mapping of a single dimension 
                       (ie, may not be downstream of a reticulation node, 
                       without it having been resolved)
        vector_len (int): rows of resulting matrix
        site_count (int): columns of resulting matrix

    Returns:
        np.ndarray: Matrix of dimension vector_len by site_count, 
                    containing floats
    """
    
    F_b = np.zeros((vector_len, site_count))  
    
    if not cs:
        for site in range(site_count):
            for nr_pair, prob in Fb_map[site].items():
                #nr_pair should be of the form ((n),(r))
                F_b[int(nr_to_index(nr_pair[0][0], nr_pair[1][0]))][site] = prob
    else:
        for site in range(site_count):
            for nr_key in Fb_map[site].Keys:
                n = nr_key.NX[0]
                r = nr_key.RX[0]
                prob = Fb_map[site][nr_key]
                F_b[nr_to_index(n, r)][site] = prob
    
    return F_b

def rn_to_rn_minus_dim(set_of_rns : dict[tuple[list[float]], float], 
                       dim : int) -> dict[tuple[list[float]], set[tuple]]:
    """
    This is a function defined as
    
    f: Rn;Rn -> Rn-dim;Rn-dim.
    
    This function takes set_of_rns and turns it into a mapping 
    {(nx[:-dim] , rx[:-dim]) : set((nx[-dim:] , rx[-dim:], probability))} 
    where the keys are vectors in Rn-dim, and their popped last elements and 
    the probability is stored as values.

    Args:
        set_of_rns (dict[tuple[list[float]], float]): a mapping in the form of 
                           {(nx , rx) -> probability in R} 
                           where nx and rx are both vectors in Rn
        dim (int): the number of dimensions to reduce from Rn.

    Returns:
        dict[tuple[list[float]], set[tuple]]: map in the form -- 
        {(nx[:-dim] , rx[:-dim]) : set((nx[-dim:] , rx[-dim:], probability))}
    """
    
    rn_minus_dim = {}
    
    for vectors, prob in set_of_rns.items():
        nx = vectors[0]
        rx = vectors[1]
        
        #keep track of the remaining elements and the probability
        new_value = (nx[-dim:], rx[-dim:], prob)
        
        #Reduce vectors dimension by grabbing the first n-dim elements
        new_key = (nx[:-dim], rx[:-dim])
        
        #Take care of duplicates
        if new_key in rn_minus_dim.keys():
            rn_minus_dim[new_key].add(new_value)
        else:
            init_value = set()
            init_value.add(new_value)
            rn_minus_dim[new_key] = init_value

    return rn_minus_dim

#####################
# Method Signatures #
#####################

def MCMC_BIMARKERS(filename: str, 
                   u : float = .5 ,
                   v : float = .5, 
                   coal : float = 1) -> dict[Network, float]:
    
    """
    Given a set of taxa with SNP data, perform a Markov Chain Monte Carlo
    chain to infer the most likely phylogenetic network that describes the
    taxa and data.

    Args:
        filename (str): string path destination of a nexus file that contains 
                        SNP data
        u (float, optional): Parameter for the probability of an
                             allele changing from red to green. Defaults to .5.
        v (float, optional): Parameter for the probability of an
                             allele changing from green to red. Defaults to .5.
        coal (float, optional): Parameter for the rate of coalescence. 
                                Defaults to 1.
        
    Returns:
        dict[Network, float]: The log likelihood (a negative number) of the most 
                              probable network, along with the network itself 
                              that achieved that score.
    """

    # Parse the data file into a sequence alignment
    aln = MSA(filename)
    
    # Generate starting network and place into model component
    start_net = CBDP(1, .5, aln.num_groups()).generate_network()
    
    snp_model = build_model(filename, 
                            start_net,
                            u, 
                            v, 
                            coal)
    
    mh = MetropolisHastings(ProposalKernel(),
                            data = Matrix(aln, Alphabet("SNP")), 
                            num_iter = 600,
                            model = snp_model) 
     
    result_state = mh.run()
    result_model = result_state.current_model

    return {result_model.network : result_model.likelihood()}

def SNP_LIKELIHOOD(filename : str,
                   u : float,
                   v : float, 
                   coal : float,
                   samples : dict[str, int],
                   max_workers: int = 8,
                   sequential: bool = True,
                   executor: Executor = None) -> float:
    """
    Given a set of taxa with SNP data and a phylogenetic network, calculate the 
    likelihood of the network given the data using the SNP likelihood algorithm.

    Args:
        filename (str): string path destination of a nexus file that 
                        contains SNP data and a network
        u (float, optional): Parameter for the probability of an
                             allele changing from red to green. Defaults to .5.
        v (float, optional): Parameter for the probability of an
                             allele changing from green to red. Defaults to .5.
        coal (float, optional): Parameter for the rate of coalescence. 
                                Defaults to 1.
        max_workers (int, optional): The number of workers to use for parallel 
                                     computation. Only used if sequential is False.
                                     Defaults to 8.
        sequential (bool, optional): Whether to use sequential computation. 
                                     Only used if sequential is False.
                                     Defaults to True.
        executor (Executor, optional): The executor to use for special 
                                    computations. Only used if sequential is False.
                                    Defaults to None.
    Returns:
        float: The log likelihood (a negative number) of the network.
    """
    
    net = NetworkParser(filename).get_network(0)
    
    aln = MSA(filename)
    
    snp_model = build_model(filename, 
                            net,
                            u, 
                            v, 
                            coal)
    
    q = BiMarkersTransition(sum(samples.values()), u, v, coal)
    
    strategy = SNPStrategy(q, u, v, coal, aln.dim()[1], sum(samples.values()))
    
    visitor = SNPModelVisitor(strategy)
    
    def likelihood_sequential(root: ModelNode) -> float:
        """
        The traversal for SNP likelihood is a level order traversal. 
        This is due to how VPI's are computed from prior VPI's. A new VPI 
        needs all VPI's that include incoming lineages to be computed first. 
        Computing by levels ensures that all VPI's are computed before the next level is computed.

        In this implementation, the levels are not computed in parallel.

        Args:
            root (ModelNode): The root node of the model.
        Returns:
            float: The log likelihood (a negative number) of the network.
        """

        for nodes, lvl in LevelParallelTraversal(root, bottom_up=True):
            print("PROCESSING LEVEL No. ", lvl)
            for node in nodes:
                print("Visiting node", node.get_name())
                visitor.visit(node)
            
        return snp_model.get_root().result

    
    def likelihood_parallel(root: ModelNode) -> float:
        """
        The traversal for SNP likelihood is a level order traversal. 
        This is due to how VPI's are computed from prior VPI's. A new VPI 
        needs all VPI's that include incoming lineages to be computed first. 
        Computing by levels ensures that all VPI's are computed before the next level is computed.

        In this implementation, the levels *ARE* computed in parallel.

        Args:
            root (ModelNode): The root node of the model.
        Returns:
            float: The log likelihood (a negative number) of the network.
        """
        from concurrent.futures import ThreadPoolExecutor

        # Traversal for LevelParallel yields a tuple of (level_number, nodes_at_level)
        # The level_num can be safely ignored.
        for level_num, nodes in LevelParallelTraversal(root, bottom_up=True):
            if len(nodes) == 1:
                visitor.visit(nodes[0])
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    pool.map(visitor.visit, nodes)
    
    if sequential:
        return likelihood_sequential(snp_model.get_root())
    else:
        return likelihood_parallel(snp_model.get_root())

    
       
def SNP_LIKELIHOOD_TEST(filename : str,
                         table : dict,
                         u : float = .5 ,
                         v : float = .5, 
                         coal : float = 1) -> dict:
    """
    THIS FUNCTION IS ONLY FOR TESTING PURPOSES
    Given a set of taxa with SNP data and a phylogenetic network, calculate the 
    likelihood of the network given the data using the SNP likelihood algorithm.

    Args:
        filename (str): string path destination of a nexus file that 
                        contains SNP data and a network
        table (dict): A dictionary of A/B/C or A/B/C/D lineage counts to probability values.
        u (float, optional): Parameter for the probability of an
                             allele changing from red to green. Defaults to .5.
        v (float, optional): Parameter for the probability of an
                             allele changing from green to red. Defaults to .5.
        coal (float, optional): Parameter for the rate of coalescence. 
                                Defaults to 1.
        
    Returns:
        dict: map from lineage counts (tuples of A/B/C or A/B/C/D) to non log probability values.
    """
    
    net = NetworkParser(filename).get_network(0)
        
    snp_model = build_model(filename, 
                            net,
                            u, 
                            v, 
                            coal)
    
    for taxa in snp_model.all_nodes[ExtantSpecies]:
        leaf : ExtantSpecies = taxa
        leaf.update([DataSequence(set_reds[leaf.label], leaf.label)])
    
    return snp_model.likelihood()

##################
# Model Building #
##################

def build_model(filename : str,
                net : Network,
                u : float = .5 ,
                v : float = .5, 
                coal : float = 1) -> Model:
    """
    Build a SNP model from a data file and network.
    """
    aln = MSA(filename)
    network = NetworkComponent(net = net)
    msa = MSAComponent({NetworkComponent}, aln)
    model = ModelFactory(network, msa).build()
    snp_root = RootAggregatorNode()
    model.root = snp_root
    net_root : RootNode = model.nodetypes["root"][0]
    net_root.join(snp_root)
    
    return model

#########################
### Transition Matrix ###
#########################

class BiMarkersTransition:
    """
    Class that encodes the probabilities of transitioning from one (n,r) pair 
    to another under a Biallelic model.

    Includes method for efficiently computing e^Qt

    Inputs:
    1) n-- the total number of samples in the species tree
    2) u-- the probability of going from the red allele to the green one
    3) v-- the probability of going from the green allele to the red one
    4) coal-- the coalescent rate constant, theta

    Assumption: Matrix indexes start with n=1, r=0, so Q[0][0] is Q(1,0);(1,0)

    Q Matrix is given by Equation 15 from:

    David Bryant, Remco Bouckaert, Joseph Felsenstein, Noah A. Rosenberg, 
    Arindam RoyChoudhury, Inferring Species Trees Directly from Biallelic 
    Genetic Markers: Bypassing Gene Trees in a Full Coalescent Analysis, 
    Molecular Biology and Evolution, Volume 29, Issue 8, August 2012, 
    Pages 1917–1932, https://doi.org/10.1093/molbev/mss086
    """

    def __init__(self, n : int, u : float, v : float, coal : float) -> None:
        """
        Initialize the Q matrix

        Args:
            n (int): sample count
            u (float): probability of a lineage going from red to green
            v (float): probability of a lineage going from green to red
            coal (float): coal rate, theta.
        Returns:
            N/A
        """

        # Build Q matrix
        self.n = n 
        self.u = u
        self.v = v
        self.coal = coal

        rows = int(.5 * self.n * (self.n + 3))
        self.Q : np.ndarray = np.zeros((rows, rows))
        
        # n ranges from 1 to individuals sampled (both inclusive)
        for n_prime in range(1, self.n + 1):  
            # r ranges from 0 to n (both inclusive)
            for r_prime in range(n_prime + 1):  
                
                # get indeces from n,r pair 
                n_r = nr_to_index(n_prime, r_prime)
                nm_rm = nr_to_index(n_prime - 1, r_prime - 1)
                n_rm = nr_to_index(n_prime, r_prime - 1)
                nm_r = nr_to_index(n_prime - 1, r_prime)
                n_rp = nr_to_index(n_prime, r_prime + 1)
                

                #### EQ 15 ####
                
                # THE DIAGONAL. always calculated
                self.Q[n_r][n_r] = - (n_prime * (n_prime - 1) / coal) \
                                       - (v * (n_prime - r_prime)) \
                                       - (r_prime * u)

                # These equations only make sense if r isn't 0 
                # (and the second, if n isn't 1).
                if 0 < r_prime <= n_prime:
                    if n_prime > 1:
                        self.Q[n_r][nm_rm] = (r_prime - 1) * n_prime / coal
                    self.Q[n_r][n_rm] = (n_prime - r_prime + 1) * v

                # These equations only make sense if r is strictly less than n 
                # (and the second, if n is not 1).
                if 0 <= r_prime < n_prime:
                    if n_prime > 1:
                        self.Q[n_r][nm_r] = (n_prime - 1 - r_prime) \
                                            * n_prime / coal
                    self.Q[n_r][n_rp] = (r_prime + 1) * u

    def expt(self, t : float = 1) -> np.ndarray:
        """
        Compute e^(Q*t) efficiently.
        
        Args:
            t (float): time, generally in coalescent units. Optional, defaults 
                       to 1, in which case e^Q is computed.
        
        Returns:
            np.ndarray: e^(Q*t).
        """
        return expm(self.Q * t)

    def cols(self) -> int:
        """
        return the dimension of the Q matrix
        
        Args:
            N/A
        Returns:
            int: the number of columns in the Q matrix
        """
        return self.Q.shape[1]

    def getQ(self) -> np.ndarray:
        """
        Retrieve the Q matrix.

        Args:
            N/A
        Returns:
            np.ndarray: The Q matrix.
        """
        return self.Q

class SNPStrategy(Strategy):
    """
    Visitor for the SNP model.
    """
    def __init__(self, q : BiMarkersTransition, u : float, v : float, coal : float, sites : int, max_samples : int) -> None:
        self.q : BiMarkersTransition = q
        self.u : float = u
        self.v : float = v
        self.coal : float = coal
        self.sites : int = sites
        print(max_samples)
        print(nr_to_index(max_samples, max_samples))
        self.vector_len : int = nr_to_index(max_samples, max_samples) + 1
    
    def _rule1(self, partial_likelihoods : np.ndarray, branch_len : float) -> np.array:
        return partial_likelihoods @ self.q.expt(branch_len)
        
    def compute_at_leaf(self, n: LeafNode) -> None:
        """
        Compute the partial likelihoods at a leaf node.

        The format for the partial likelihoods is a two dimensional array where the first dimension (rows) is the site index 
        and the second dimension (columns) is the number of samples for this leaf. 
        The position of the 1.0 probability is the number of red alleles at that site.
        """
        print("Computing at leaf", n.get_name())

        assert len(n.data) == 1, "Leaf node must have exactly one data sequence"
        reds : list[int] = n.data[0].get_numerical_seq()
        
        base_likelihoods : np.array = np.zeros((self.sites, self.vector_len), dtype=np.float64)
        
        for site in range(self.sites):
            base_likelihoods[site, nr_to_index(n.samples, reds[site])] = 1.0
        
        return self._rule1(base_likelihoods, n.branch().length)
         
    def compute_at_internal(self, n: InternalNode, samples : int) -> None:
        """
        Compute the partial likelihoods at an internal node.
        """
        rule2 = _disjoint_subnets(n)
        if rule2:
            # Use Rule 2
            pass
        else:
            # Use Rule 4
            pass
        return self.rule1(partial_likelihoods, len(n.branch))
        
    
    def compute_at_reticulation(self, n: ReticulationNode, samples : int) -> None:
        """
        Compute the partial likelihoods at a reticulation node.
        """
        # # Use Rule 3
        # branch1, branch2 = n.branches()

        # partials1 : np.array
        # partials2 : np.array
        # return self._rule1(partials1, branch1.length), self._rule1(partials2, branch2.length)
        print("Computing at reticulation", n.get_name())
        return 1

    def compute_at_root(self, n: RootNode, samples : int) -> None:
        """
        Compute the partial likelihoods at an internal node.
        """
        rule2 = _disjoint_subnets(n)
        if rule2:
            # Use Rule 2
            pass
        else:
            # Use Rule 4
            pass
        return 

    def compute_at_aggregator(self, n: RootAggregatorNode, root_partials : np.ndarray) -> None:
        """
        Compute the partial likelihoods at a root aggregator node.
        """
        #Normalize Q matrix
        q_null_space = scipy.linalg.null_space(self.q.getQ())
        x = q_null_space / (q_null_space[0] + q_null_space[1])

        #Compute log likelihood 
        L = np.log(np.dot(root_partials, x))
        return L

class SNPModelVisitor(Visitor):
    """
    Visitor for the SNP model.
    """
    def __init__(self, strategy: SNPStrategy) -> None:
        self.samples : dict[ModelNode, int] = {}
        self.strategy : SNPStrategy = strategy
    
    def visit_leaf(self, n: LeafNode) -> None:
        self.strategy.compute_at_leaf(n)
        self.samples[n] = n.samples
    
    def visit_internal(self, n: InternalNode) -> None:
        self.samples[n] = sum([self.samples[child] for child in n.get_model_children()])
        self.strategy.compute_at_internal(n, self.samples[n])
        
    def visit_reticulation(self, n: ReticulationNode) -> None:
        self.samples[n] = sum([self.samples[child] for child in n.get_model_children()])
        self.strategy.compute_at_reticulation(n, self.samples[n])
    
    def visit_root(self, n: RootNode) -> None:
        self.samples[n] = sum([self.samples[child] for child in n.get_model_children()])
        self.strategy.compute_at_root(n, self.samples[n])
    
    def visit_aggregator(self, n: RootAggregatorNode) -> None:
        self.strategy.compute_at_aggregator(n)
    
    def visit(self, n: ModelNode) -> None:
        """
        Visit a node.
        """
        """Dispatch to the correct visit method based on node type."""
        dispatch = {
            "leaf": self.visit_leaf,
            "internal": self.visit_internal,
            "root": self.visit_root,
            "reticulation": self.visit_reticulation,
            "root_aggregator": self.visit_aggregator,
        }
        return dispatch[n.get_node_type()](n)



class PartialLikelihoods:
    """
    Class that bookkeeps the vectors of population interfaces (vpis) and their
    associated likelihood values.
    
    Contains methods for evaluating and storing likelihoods based on the rules
    described by Rabier et al.
    """
    
    def __init__(self) -> None:
        """
        Initialize an empty PartialLikelihood obj.

        Args:
            N/A
        Returns:
            N/A
        """
        # A map from a vector of population interfaces (vpi)
        # -- represented as a tuple of strings-- 
        # to probability maps defined by rules 0-4.
        self.vpis : dict = {}
        
    def set_ploidy(self, ploidy : int) -> None:
        """
        Set the ploidy value for the partial likelihoods object.
        
        Args:
            ploidyness (int): ploidy value.
        Returns:
            N/A
        """
        self.ploidy = ploidy

    def Rule0():
        pass

        
    def Rule1(self,
              vpi_key_x : tuple, 
              branch_id_x : int,
              m_x : int, 
              Qt : np.ndarray) -> tuple:
        """
        Given a branch x, and partial likelihoods for the population interface 
        that includes x_bottom, we'd like to compute the partial likelihoods for 
        the population interface that includes x_top.
        
        This uses Rule 1 from (1)
        

        Args:
            vpi_key_x (tuple): the key to the vpi map, the value of which is a 
                             mapping containing mappings from vectors 
                             (nx, n_xbot; rx, r_xbot) to probability values 
                             for each site
            branch_id_x (int): the unique id of branch x
            m_x (int): number of possible lineages at the branch x
            Qt (np.ndarray): the transition rate matrix exponential

        Returns:
            tuple: vpi key that maps to the partial likelihoods at the 
                   population interface that now includes the top of this 
                   branch, x_top.
        """
        
        
        # Check if vectors are properly ordered
        if "branch_" + str(branch_id_x) + ": bottom" != vpi_key_x[-1]:
            vpi_key_temp = self.reorder_vpi(vpi_key_x,
                                            site_count, 
                                            branch_id_x, 
                                            False)
            del self.vpis[vpi_key_x]
            vpi_key_x = vpi_key_temp
            
            
        F_b = self.vpis[vpi_key_x]
        
        # Replace the instance of x_bot with x_top
        new_vpi_key = list(vpi_key_x)
        edit_index = vpi_key_x.index("branch_" + str(branch_id_x) + ": bottom")
        new_vpi_key[edit_index] = "branch_" + str(branch_id_x) + ": top"
        new_vpi_key = tuple(new_vpi_key)
        
        # Put the map back
        self.vpis[new_vpi_key] = F_b @ Qt
        del self.vpis[vpi_key_x]
        
        return new_vpi_key
                
    def Rule2(self, 
              vpi_key_x : tuple, 
              vpi_key_y : tuple,  
              branch_id_x : str,
              branch_id_y : str, 
              branch_id_z : str) -> tuple:
        """
        Given branches x and y that have no leaf descendents in common and a 
        parent branch z, and partial likelihood mappings for the population 
        interfaces that include x_top and y_top, we would like to calculate 
        the partial likelihood mapping for the population interface
        that includes z_bottom.
        
        This uses Rule 2 from (1)

        Args:
            vpi_key_x (tuple): The vpi that contains x_top
            vpi_key_y (tuple): The vpi that contains y_top
            branch_id_x (str): the unique id of branch x
            branch_id_y (str): the unique id of branch y
            branch_id_z (str): the unique id of branch z
        

        Returns:
            tuple: the vpi key that is the result of applying rule 2 to 
                   vpi_x and vpi_y. Should include z_bot.
        """
        
        #Reorder the vpis if necessary
        if "branch_" + str(branch_id_x) + ": top" != vpi_key_x[-1]:
            
            vpi_key_xtemp = self.reorder_vpi(vpi_key_x, 
                                             site_count, 
                                             branch_id_x,
                                             True)
            del self.vpis[vpi_key_x]
            vpi_key_x = vpi_key_xtemp
        
        if "branch_" + str(branch_id_y) + ": top" != vpi_key_y[-1]:
            
            vpi_key_ytemp = self.reorder_vpi(vpi_key_y, 
                                             site_count,
                                             branch_id_y, 
                                             True)
            del self.vpis[vpi_key_y]
            vpi_key_y = vpi_key_ytemp
            
        F_t_x = self.vpis[vpi_key_x]
        F_t_y = self.vpis[vpi_key_y]
        
        if not cs:
            F_b = {}
            #Compute F x,y,z_bot
            for site in range(site_count):
                nx_rx_map_y = rn_to_rn_minus_dim(F_t_y[site], 1)
                nx_rx_map_x = rn_to_rn_minus_dim(F_t_x[site], 1)
                F_b[site] = {}
                
                #Compute all combinations of (nx;rx) and (ny;ry)
                for vectors_x in nx_rx_map_x.keys():
                    for vectors_y in nx_rx_map_y.keys():
                        nx = list(vectors_x[0])
                        rx = list(vectors_x[1])
                        ny = list(vectors_y[0])
                        ry = list(vectors_y[1])
                        
                        #Iterate over all possible values of n_zbot, r_zbot
                        for index in range(vector_len):
                            actual_index = index_to_nr(index)
                            n_bot = actual_index[0]
                            r_bot = actual_index[1]
                            # Evaluate the formula given in rule 2, 
                            # and insert as an entry in F_b
                            entry = eval_Rule2(F_t_x[site], F_t_y[site],
                                               nx, ny, n_bot, rx, ry, r_bot)
                            F_b[site][entry[0]] = entry[1]
        else:
            F_b = self.evaluator.Rule2(F_t_x, F_t_y, site_count, vector_len)
        
        #Combine the vpis
        new_vpi_key_x= list(vpi_key_x)
        new_vpi_key_x.remove("branch_" + str(branch_id_x) + ": top")
        
        new_vpi_key_y= list(vpi_key_y)
        new_vpi_key_y.remove("branch_" + str(branch_id_y) + ": top")
        
        #Create new vpi key, (vpi_x, vpi_y, z_branch_bottom)
        z_name = "branch_" + str(branch_id_z) + ": bottom"
        vpi_y = np.append(new_vpi_key_y, z_name)
        new_vpi_key = tuple(np.append(new_vpi_key_x, vpi_y))
        
        
        #Update the vpi tracker
        self.vpis[new_vpi_key] = F_b
        del self.vpis[vpi_key_x]
        del self.vpis[vpi_key_y]
                         
        return new_vpi_key

    def Rule3(self, 
              vpi_key_x : tuple, 
              branch_id_x : str,
              branch_id_y : str, 
              branch_id_z : str,
              g_this : float,
              g_that : float,
              mx : int) -> tuple:
        """
        Given a branch x, its partial likelihood mapping at x_top, and parent 
        branches y and z, we would like to compute the partial likelihood 
        mapping for the population interface x, y_bottom, z_bottom.

        This uses Rule 3 from (1)
        
        Args:
            vpi_key_x (tuple): the vpi containing x_top
            branch_id_x (str): the unique id of branch x
            branch_id_y (str): the unique id of branch y
            branch_id_z (str): the unique id of branch z
            g_this (float): gamma inheritance probability for branch y
            g_that (float): gamma inheritance probability for branch z
            mx (int): number of possible lineages at x.

        Returns:
            tuple: the vpi key that now corresponds to F (x, y_bot, z_bot)
        """
        
        F_t_x = self.vpis[vpi_key_x]
        
        if not cs:
            F_b = {}
            for site in range(site_count):
                nx_rx_map = rn_to_rn_minus_dim(F_t_x[site], 1)
                F_b[site] = {}
                #Iterate over the possible (nx;rx) values
                for vector in nx_rx_map.keys():
                    nx = list(vector[0])
                    rx = list(vector[1])
                    #Iterate over the possible values for n_y, n_z, r_y, and r_z
                    for n_y in range(mx + 1):
                        for n_z in range(mx - n_y + 1):
                            if n_y + n_z >= 1:
                                for r_y in range(n_y + 1):
                                    for r_z in range(n_z + 1):
                                        # Evaluate the formula in rule 3 
                                        # and add the result to F_b
                                        entry = eval_Rule3(F_t_x[site],
                                                           nx,
                                                           rx, 
                                                           n_y, 
                                                           n_z, 
                                                           r_y, 
                                                           r_z, 
                                                           g_this, 
                                                           g_that)
                                        F_b[site][entry[0]] = entry[1]
        else:
            F_b = self.evaluator.Rule3(F_t_x, 
                                       site_count, 
                                       vector_len, 
                                       mx, 
                                       g_this, 
                                       g_that)
        
        #Create new vpi key                      
        new_vpi_key = list(vpi_key_x)
        new_vpi_key.remove("branch_" + str(branch_id_x) + ": top")
        new_vpi_key.append("branch_" + str(branch_id_y) + ": bottom")
        new_vpi_key.append("branch_" + str(branch_id_z) + ": bottom")
        new_vpi_key = tuple(new_vpi_key)
        
        # Update vpi tracker
        self.vpis[new_vpi_key] = F_b
        del self.vpis[vpi_key_x]
        
        return new_vpi_key               
            
    def Rule4(self, 
              vpi_key_xy : tuple, 
              branch_index_x : int, 
              branch_index_y : int, 
              branch_index_z : int) -> tuple:
        """
        Given a branches x and y that share common leaf descendants and that 
        have parent branch z, compute F z, z_bot via 
        Rule 4 described by (1)

        Args:
            vpi_key_x (tuple): vpi containing x_top and y_top.
            branch_index_x (int): the index of branch x
            branch_index_y (int): the index of branch y
            branch_index_z (int): the index of branch z

        Returns:
            tuple: vpi key for F(z,z_bot)
        """

        if "branch_" + str(branch_index_y) + ": top" != vpi_key_xy[-1]:
            vpi_key_temp = self.reorder_vpi(vpi_key_xy,
                                            site_count, 
                                            branch_index_y, 
                                            True)
            del self.vpis[vpi_key_xy]
            vpi_key_xy = vpi_key_temp
        
        F_t = self.vpis[vpi_key_xy]
        
        if not cs:
            F_b = {}
            for site in range(site_count):
                nx_rx_map = rn_to_rn_minus_dim(F_t[site], 2)
                
                F_b[site] = {}
                
                #Compute all combinations of (nx;rx) and (ny;ry)
                for vectors_x in nx_rx_map.keys():
                
                    nx = list(vectors_x[0])
                    rx = list(vectors_x[1])
                    
                    #Iterate over all possible values of n_zbot, r_zbot
                    for index in range(vector_len):
                        actual_index = index_to_nr(index)
                        n_bot = actual_index[0]
                        r_bot = actual_index[1]
                        # Evaluate the formula given in rule 2,
                        # and insert as an entry in F_b
                        entry = eval_Rule4(F_t[site], nx, rx, n_bot, r_bot)
                        F_b[site][entry[0]] = entry[1]
        else:      
            F_b = self.evaluator.Rule4(F_t, site_count, vector_len)
        
        #Create new vpi
        new_vpi_key = list(vpi_key_xy)
        new_vpi_key.remove("branch_" + str(branch_index_x) + ": top")
        new_vpi_key.remove("branch_" + str(branch_index_y) + ": top")
        new_vpi_key.append("branch_" + str(branch_index_z) + ": bottom")
        new_vpi_key = tuple(new_vpi_key)
    
        # Update vpi tracker
        self.vpis[new_vpi_key] = F_b
        del self.vpis[vpi_key_xy]
        
        return new_vpi_key
    
    def reorder_vpi(self, 
                    vpi_key: tuple, 
                    branch_index : int, 
                    for_top : bool) -> tuple:
        """
        For use when a rule requires a certain ordering of a vpi, and the
        current vpi does not satisfy it.
        
        I.E, For Rule1, have vpi (branch_1_bottom, branch_2_bottom) but 
        need to calculate for branch 1 top. 
        
        The vpi needs to be reordered to (branch_2_bottom, branch_1_bottom), 
        and the vectors in the partial likelihood mappings need to be reordered 
        to match.

        Args:
            vpi_key (tuple): a vpi tuple
            branch_index (int): branch index of the branch that needs 
                                to be in the front
            for_top (bool): bool indicating whether we are looking for 
                            branch_index_top or branch_index_bottom in the 
                            vpi key.

        Returns:
            tuple: the new, reordered vpi key.
        """
        #print("REORDER 1")
        if for_top:
            name = "branch_" + str(branch_index) + ": top"
            former_index = list(vpi_key).index(name)
        else:
            name = "branch_" + str(branch_index) + ": bottom"
            former_index = list(vpi_key).index(name)
            
        new_vpi_key = list(vpi_key)
        new_vpi_key.append(new_vpi_key.pop(former_index))
        
        F_map = self.vpis[vpi_key]
        
        if not cs:
            new_F = {}
            
            # Reorder all the vectors based on the location of the 
            # interface within the vpi
            for site in range(site_count):
                new_F[site] = {}
            
                for vectors, prob in F_map[site].items():
                    nx = list(vectors[0])
                    rx = list(vectors[1])
                    new_nx = list(nx)
                    new_rx = list(rx)
                    
                    #pop the element from the list and move to the front
                    new_nx.append(new_nx.pop(former_index))
                    new_rx.append(new_rx.pop(former_index))
            
                    new_F[site][(tuple(new_nx), tuple(new_rx))] = prob
            
            F_map = new_F
            
        else:
            for site in range(site_count):
                for tup in F_map[site].Keys:
                    tup.MoveToEnd(former_index)
        
        self.vpis[tuple(new_vpi_key)] = F_map
    
        return tuple(new_vpi_key)
    
    def get_key_with(self, branch_id : str) -> tuple:
        """
        From the set of vpis, grab the one (should only be one) that contains 
        the branch identified by branch_index

        Args:
            branch_index (int): unique branch identifier

        Returns:
            tuple: the vpi key corresponding to branch_index, or None if no 
                   such vpi currently exists
        """
    
        for vpi_key in self.vpis:
            top = "branch_" + branch_id + ": top"
            bottom = "branch_" + branch_id + ": bottom"
            
            #Return vpi if there's a match
            if top in vpi_key or bottom in vpi_key:
                return vpi_key
        
        #No vpis were found containing branch_index
        return None
            

def _disjoint_subnets(n: InternalNode) -> bool:
    """
    Determine if the left and right subnets of an internal node are disjoint.
    """
    
    lr = n.get_model_children()
    assert len(lr) == 2, "Internal node must have exactly two children"
    subnets : tuple[set[ModelNode], set[ModelNode]] = ({}, {})
    i = 0
    
    for child in lr:
        q = deque(child)
        while len(q) != 0:
            cur = q.popleft()
            for kin in cur.get_model_children():
                subnets[i].add(kin)
                q.append(kin)
        i += 1
    
    return subnets[0].isdisjoint(subnets[1])