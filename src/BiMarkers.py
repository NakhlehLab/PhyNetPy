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
from dataclasses import dataclass

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

def dim(n : int) -> int:
    return nr_to_index(n, n)

@dataclass
class NodeVPI:
    tensor: np.ndarray       # shape [S, d1, d2, ...]                  
    interfaces: list[str]    # ["α", "β", ...] length = tensor.ndim-1 
    max_lineages: list[int]  # [m_α, m_β, ...] per interface 


def state_dim(m: int) -> int:
    return nr_to_index(m, m) + 1

def build_split_tensor(m: int, gamma: float) -> np.ndarray:
    """
    Build the split coefficient tensor S for a reticulation node
    with max lineages m and inheritance probability gamma.

    S[i, j, k] = C(n, n_b) * C(r, r_b) * gamma^{n_b} * (1-gamma)^{n_d}

    where i ↔ (n, r), j ↔ (n_b, r_b), k ↔ (n_d, r_d)
    and n = n_b + n_d, r = r_b + r_d (zero otherwise).

    Shape: [dim(m), dim(m), dim(m)]
    """
    d = state_dim(m)
    S = np.zeros((d, d, d))

    for n in range(1, m + 1):
        for r in range(n + 1):
            i = nr_to_index(n, r)
            for nb in range(1, n + 1):
                nd = n - nb
                if nd < 1:
                    continue
                for rb in range(min(r, nb) + 1):
                    rd = r - rb
                    if rd < 0 or rd > nd:
                        continue

                    j = nr_to_index(nb, rb)
                    k = nr_to_index(nd, rd)

                    S[i, j, k] = (comb(n, nb) * comb(r, rb)
                                  * (gamma ** nb) * ((1 - gamma) ** nd))

    return S

def build_merge_tensor(mx: int, my: int) -> np.ndarray:
    """
    Build the merge coefficient tensor M for combining two interfaces
    with max lineages mx and my.

    M[i, j, k] = C(rz, rx) * C(nz-rz, nx-rx) / C(nz, nx)
    
    where i ↔ (nx, rx), j ↔ (ny, ry), k ↔ (nz, rz)
    and nz = nx + ny, rz = rx + ry (zero otherwise).

    Used by both Rule 2 (disjoint merge) and Rule 4 (overlapping merge).

    Shape: [dim(mx), dim(my), dim(mx + my)]
    """
    mz = mx + my
    M = np.zeros((state_dim(mx), state_dim(my), state_dim(mz)))

    for nx in range(1, mx + 1):
        for rx in range(nx + 1):
            i = nr_to_index(nx, rx)
            for ny in range(1, my + 1):
                for ry in range(ny + 1):
                    j = nr_to_index(ny, ry)
                    
                    nz = nx + ny
                    rz = rx + ry
                    k = nr_to_index(nz, rz)

                    M[i, j, k] = (comb(rz, rx) * comb(nz - rz, nx - rx) 
                                  / comb(nz, nx))

    return M

def _disjoint_subnets(n: InternalNode) -> bool:
    """
    Determine if the left and right subnets of an internal node are disjoint.
    """
    
    lr = n.get_model_children()
    assert len(lr) == 2, "Internal node must have exactly two children"
    subnets : tuple[set[ModelNode], set[ModelNode]] = (set(), set())
    i = 0
    
    for child in lr:
        q = deque([child])
        while len(q) != 0:
            cur = q.popleft()
            for kin in cur.get_model_children():
                subnets[i].add(kin)
                q.append(kin)
        i += 1
    
    return subnets[0].isdisjoint(subnets[1])


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
            
        return strategy.L

    
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
        
        return strategy.L
    
    if sequential:
        return likelihood_sequential(snp_model.get_root())
    else:
        return likelihood_parallel(snp_model.get_root())


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
        self.vector_len : int = state_dim(max_samples)
        self.L : np.ndarray 
    
    def _rule1(self, F : np.ndarray, branch_len : float) -> np.ndarray:
        """
        Given vpi tensor F, with interface α, and branch length t, and max lineages m_α, 
        compute the vpi tensor F_top at the top of the branch.

        Args:
            F (np.ndarray): vpi tensor F
            branch_len (float): branch length
        Returns:
            np.ndarray: vpi tensor F_top
        """
        return F @ self.q.expt(branch_len)
    
    def _rule2(self, F_x : np.ndarray, F_y : np.ndarray, m_x : int, m_y : int) -> np.ndarray:

        #Explanation of einsum:
        #s...i -> sum over the last dimension
        #s...j -> sum over the last dimension
        #ijk -> matrix multiplication of the last three dimensions
        #s...k -> result is shape [S, d1, d2, ...]
        return np.einsum("s...i, s...j, ijk->s...k", F_x, F_y, build_merge_tensor(m_x, m_y))
    
    def _rule3(self, F, mx, gammax) -> np.ndarray:
        #Explanation of einsum:
        
        return np.einsum('...i,ijk->...jk', F, build_split_tensor(mx, gammax))
    
    def _rule4(self, F, mx, my) -> np.ndarray:
        return np.einsum('...ij,ijk->...k', F, build_merge_tensor(mx, my))  
        
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
        
        F : np.ndarray = np.zeros((self.sites, state_dim(n.samples)), dtype=np.float64)
        
        for site in range(self.sites):
            F[site, nr_to_index(n.samples, reds[site])] = 1.0
        
        F = self._rule1(F, n.branch().length)
        n.vpi = NodeVPI(F, [f"{n.get_name()}_top"], [n.samples])

         
    def compute_at_internal(self, n: InternalNode, x : NodeVPI, y: NodeVPI) -> None:
        """
        Compute the partial likelihoods at an internal node.
        """
        rule2 = _disjoint_subnets(n)
        mx = x.max_lineages[-1]
        my = y.max_lineages[-1]
        
        if rule2:
            F = self._rule2(x.tensor, y.tensor, mx, my)
            interfaces = x.interfaces[:-1] + y.interfaces[:-1] + [f"{n.get_name()}_top"] 
            max_lin = x.max_lineages[:-1] + y.max_lineages[:-1] + [mx+my]
        else:
            F = self._rule4(x.tensor, x.max_lineages[-1], y.max_lineages[-1])
            interfaces = x.interfaces[:-2] + [f"{n.get_name()}_top"] 
            max_lin = x.max_lineages[:-2] + [mx + my]
        
        F = self._rule1(F, n.branch().length)  
           
        n.vpi = NodeVPI(F, interfaces, max_lin)
        
    def compute_at_reticulation(self, n : ReticulationNode, x : NodeVPI) -> None:
        """
        Compute the partial likelihoods at a reticulation node.
        """
        #TODO: Unsure of how to deal with this scenario. There are two branches, of course.
        # And because of that there would be two vpis?? I'm not sure. pls assist here!
        branches : tuple[Branch, Branch]= n.branch_info
        gamma = branches[0].inheritance_probability
        m = x.max_lineages[-1]
        
        F = self._rule3(x.tensor, x.max_lineages[-1], gamma)
        F = np.moveaxis(F, -2, -1)
        F = self._rule1(F, branches[0].length)
        F = np.moveaxis(F, -1, -2)
        F = self._rule1(F, branches[1].length)
        
        #Book keep the interfaces and lineages
        interfaces = x.interfaces[:-1] + [f"{n.get_name()}_left_bot", f"{n.get_name()}_right_bot"]  
        max_lin = x.max_lineages[:-1] + [m, m] 
        
        n.vpi = NodeVPI(F, interfaces, max_lin)
        

    def compute_at_root(self, n: RootNode, x : NodeVPI, y : NodeVPI) -> None:
        """
        Compute the partial likelihoods at an internal node.
        """
        rule2 = _disjoint_subnets(n)
        if rule2:
            F = self._rule2(x.tensor, y.tensor, x.max_lineages[-1], y.max_lineages[-1])
        else:
            F = self._rule4(x.tensor, x.max_lineages[-1], y.max_lineages[-1])
        
        n.vpi = NodeVPI(F, [f"{n.get_name()}_bottom"], [x.max_lineages[-1] + y.max_lineages[-1]])
        

    def compute_at_aggregator(self, n: RootAggregatorNode, root : NodeVPI) -> None:
        """
        Compute the partial likelihoods at a root aggregator node.
        """
        #Normalize Q matrix
        q_null_space = scipy.linalg.null_space(self.q.getQ())
        x = q_null_space / (q_null_space[0] + q_null_space[1])

        #Compute log likelihood 
        self.L = np.log(np.dot(root.tensor, x))
        
        

class SNPModelVisitor(Visitor):
    """
    Visitor for the SNP model.
    """
    def __init__(self, strategy: SNPStrategy) -> None:
        self.strategy : SNPStrategy = strategy
    
    def visit_leaf(self, n: LeafNode) -> None:
        self.strategy.compute_at_leaf(n)
        
    def visit_internal(self, n: InternalNode) -> None:
        child_vpis : list[NodeVPI]= [child.vpi for child in n.get_model_children()]
        self.strategy.compute_at_internal(n, child_vpis[0], child_vpis[1])
        
    def visit_reticulation(self, n: ReticulationNode) -> None:
        child_vpi : NodeVPI= [child.vpi for child in n.get_model_children()][0]
        self.strategy.compute_at_reticulation(n, child_vpi)
    
    def visit_root(self, n: RootNode) -> None:
        child_vpis : list[NodeVPI]= [child.vpi for child in n.get_model_children()]
        self.strategy.compute_at_root(n, child_vpis[0], child_vpis[1])
    
    def visit_aggregator(self, n: RootAggregatorNode) -> None:
        child_vpi = n.get_model_children()[0].vpi
        self.strategy.compute_at_aggregator(n, child_vpi)
    
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


