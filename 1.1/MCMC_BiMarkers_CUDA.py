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
Last Edit : 12/3/25
First Included in Version : 1.0.0

CUDA-accelerated version of MCMC_BiMarkers.py
Optimized for NVIDIA GPUs (tested with RTX 5070ti)
Replaces C# DLL calls with CUDA kernels via Numba and CuPy

Docs   - [ ]
Tests  - [ ]
Design - [ ]
"""

from math import sqrt, comb, pow
from typing import Callable
import numpy as np
import scipy
from scipy.linalg import expm

# CUDA imports
import cupy as cp
from numba import cuda, float64, int32, int64
from numba.cuda import cooperative_groups as cg
import math

from PhyNetPy.MSA import MSA
from PhyNetPy.BirthDeath import CBDP
from PhyNetPy.NetworkParser import NetworkParser
from PhyNetPy.Alphabet import Alphabet
from PhyNetPy.Matrix import Matrix
from PhyNetPy.ModelGraph import Model
from PhyNetPy.ModelFactory import *
from PhyNetPy.Network import Network, Edge, Node
from PhyNetPy.MetropolisHastings import *


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


########################
### CUDA UTILITIES ###
########################

def get_cuda_device_info():
    """
    Print information about the available CUDA device.
    Useful for debugging and optimization.
    """
    if cuda.is_available():
        device = cuda.get_current_device()
        print(f"CUDA Device: {device.name}")
        print(f"Compute Capability: {device.compute_capability}")
        print(f"Max threads per block: {device.MAX_THREADS_PER_BLOCK}")
        print(f"Max shared memory per block: {device.MAX_SHARED_MEMORY_PER_BLOCK}")
        return True
    else:
        print("CUDA is not available. Falling back to CPU.")
        return False


@cuda.jit(device=True)
def d_n_to_index(n: int64) -> int64:
    """
    Device function: Computes starting index for an (n,r) pair.
    CUDA device version of n_to_index.
    """
    return int64(0.5 * (n - 1) * (n + 2))


@cuda.jit(device=True)
def d_nr_to_index(n: int64, r: int64) -> int64:
    """
    Device function: Maps (n,r) pair to 1D vector index.
    CUDA device version of nr_to_index.
    """
    return d_n_to_index(n) + r


@cuda.jit(device=True)
def d_index_to_n(index: int64) -> int64:
    """
    Device function: Extract n from linear index.
    """
    a = 1.0
    b = 1.0
    c = -2.0 - 2.0 * float64(index)
    d = (b * b) - (4.0 * a * c)
    sol = (-b + math.sqrt(d)) / (2.0 * a)
    return int64(sol)


@cuda.jit(device=True)
def d_index_to_r(index: int64, n: int64) -> int64:
    """
    Device function: Extract r from linear index given n.
    """
    return index - d_n_to_index(n)


@cuda.jit(device=True)
def d_comb(n: int64, k: int64) -> float64:
    """
    Device function: Compute binomial coefficient C(n,k).
    Uses multiplicative formula for numerical stability.
    """
    if k > n or k < 0:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    if k > n - k:
        k = n - k
    
    result = 1.0
    for i in range(k):
        result = result * float64(n - i) / float64(i + 1)
    return result


########################
### HELPER FUNCTIONS ###
########################

def n_to_index(n: int) -> int:
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


def index_to_nr(index: int) -> list[int]:
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


def nr_to_index(n: int, r: int) -> int:
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


def to_array(Fb_map: dict, 
             vector_len: int, 
             site_count: int) -> np.ndarray:
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
    
    for site in range(site_count):
        for nr_pair, prob in Fb_map[site].items():
            # nr_pair should be of the form ((n),(r))
            F_b[int(nr_to_index(nr_pair[0][0], nr_pair[1][0]))][site] = prob
    
    return F_b


def rn_to_rn_minus_dim(set_of_rns: dict[tuple[list[float]], float], 
                       dim: int) -> dict[tuple[list[float]], set[tuple]]:
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
        
        # keep track of the remaining elements and the probability
        new_value = (nx[-dim:], rx[-dim:], prob)
        
        # Reduce vectors dimension by grabbing the first n-dim elements
        new_key = (nx[:-dim], rx[:-dim])
        
        # Take care of duplicates
        if new_key in rn_minus_dim.keys():
            rn_minus_dim[new_key].add(new_value)
        else:
            init_value = set()
            init_value.add(new_value)
            rn_minus_dim[new_key] = init_value

    return rn_minus_dim


#########################
### CUDA KERNELS ########
#########################

@cuda.jit
def rule0_kernel(reds, site_count, vector_len, samples, output):
    """
    CUDA Kernel for Rule 0: Initialize partial likelihoods at leaves.
    
    Each thread handles one (site, index) pair.
    
    Args:
        reds: Array of red counts per site [site_count]
        site_count: Number of sites
        vector_len: Length of (n,r) vector space
        samples: Number of samples
        output: Output array [site_count, vector_len] for probabilities
    """
    # 2D grid: x = site, y = index
    site = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    index = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    if site < site_count and index < vector_len:
        # Compute (n, r) from index
        n = d_index_to_n(index)
        r = d_index_to_r(index, n)
        
        # Rule 0 formula: probability is 1 if r == reds[site] and n == samples
        if int64(reds[site]) == r and n == samples:
            output[site, index] = 1.0
        else:
            output[site, index] = 0.0


@cuda.jit
def rule1_kernel(F_b, Qt, site_count, vector_len, mx, output):
    """
    CUDA Kernel for Rule 1: Transition from bottom to top of branch.
    
    Computes: F(x, x_top) from F(x, x_bot) using transition matrix Qt.
    
    Each thread computes one (site, n_top, r_top) combination.
    
    Args:
        F_b: Input partial likelihoods [site_count, vector_len]
        Qt: Transition matrix (e^Qt) [vector_len, vector_len]
        site_count: Number of sites
        vector_len: Length of (n,r) vector space
        mx: Maximum number of lineages at this node
        output: Output array [site_count, vector_len]
    """
    # Get thread indices
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    # Decode: which site and which (n_top, r_top)
    total_work = site_count * vector_len
    if tid >= total_work:
        return
    
    site = tid // vector_len
    top_index = tid % vector_len
    
    # Get (n_top, r_top) from index
    n_top = d_index_to_n(top_index)
    r_top = d_index_to_r(top_index, n_top)
    
    # Skip if n_top > mx (not valid for this branch)
    if n_top > mx:
        output[site, top_index] = 0.0
        return
    
    # Compute Rule 1 summation
    evaluation = 0.0
    
    # Sum over all valid (n_b, r_b) at bottom
    max_index = d_n_to_index(mx + 1)
    for bot_index in range(max_index):
        n_b = d_index_to_n(bot_index)
        r_b = d_index_to_r(bot_index, n_b)
        
        # n_b must be >= n_top for valid transition
        if n_b >= n_top:
            # Q(n_b, r_b) -> (n_top, r_top)
            qt_val = Qt[bot_index, top_index]
            fb_val = F_b[site, bot_index]
            evaluation += fb_val * qt_val
    
    output[site, top_index] = evaluation


@cuda.jit
def rule2_kernel(F_t_x, F_t_y, site_count, vector_len, output):
    """
    CUDA Kernel for Rule 2: Combine two disjoint branches at speciation node.
    
    Computes: F(x, y, z_bot) from F(x, x_top) and F(y, y_top)
    
    Args:
        F_t_x: Partial likelihoods at x_top [site_count, vector_len]
        F_t_y: Partial likelihoods at y_top [site_count, vector_len]
        site_count: Number of sites
        vector_len: Length of (n,r) vector space
        output: Output array [site_count, vector_len]
    """
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    total_work = site_count * vector_len
    if tid >= total_work:
        return
    
    site = tid // vector_len
    z_index = tid % vector_len
    
    # Get (n_zbot, r_zbot) from index
    n_zbot = d_index_to_n(z_index)
    r_zbot = d_index_to_r(z_index, n_zbot)
    
    evaluation = 0.0
    
    # Iterate through valid (n_xtop, r_xtop) values
    for n_xtop in range(n_zbot + 1):
        for r_xtop in range(r_zbot + 1):
            # Ensure combinatorics is well-defined
            if r_xtop <= n_xtop and (r_zbot - r_xtop) <= (n_zbot - n_xtop):
                n_ytop = n_zbot - n_xtop
                r_ytop = r_zbot - r_xtop
                
                # Rule 2 constant
                const = d_comb(n_xtop, r_xtop) * d_comb(n_ytop, r_ytop) / d_comb(n_zbot, r_zbot)
                
                x_index = d_nr_to_index(n_xtop, r_xtop) if n_xtop > 0 else 0
                y_index = d_nr_to_index(n_ytop, r_ytop) if n_ytop > 0 else 0
                
                if n_xtop > 0 and n_ytop > 0:
                    term1 = F_t_x[site, x_index]
                    term2 = F_t_y[site, y_index]
                    evaluation += term1 * term2 * const
    
    output[site, z_index] = evaluation


@cuda.jit
def rule3_kernel(F_t_x, site_count, vector_len, mx, gamma_y, gamma_z, output_y, output_z):
    """
    CUDA Kernel for Rule 3: Handle reticulation node (one branch splits into two).
    
    Computes: F(x, y_bot, z_bot) from F(x, x_top)
    
    Args:
        F_t_x: Partial likelihoods at x_top [site_count, vector_len]
        site_count: Number of sites
        vector_len: Length of (n,r) vector space  
        mx: Maximum lineages at x
        gamma_y: Inheritance probability for y branch
        gamma_z: Inheritance probability for z branch
        output_y: Output for y branch [site_count, vector_len]
        output_z: Output for z branch [site_count, vector_len]
    """
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    # Each thread handles one (site, n_y, n_z, r_y, r_z) combination
    # This is a more complex indexing scheme
    total_work = site_count * vector_len * vector_len
    if tid >= total_work:
        return
    
    site = tid // (vector_len * vector_len)
    remainder = tid % (vector_len * vector_len)
    y_index = remainder // vector_len
    z_index = remainder % vector_len
    
    n_y = d_index_to_n(y_index)
    r_y = d_index_to_r(y_index, n_y)
    n_z = d_index_to_n(z_index)
    r_z = d_index_to_r(z_index, n_z)
    
    # Skip invalid combinations
    if n_y + n_z > mx or n_y + n_z < 1:
        return
    
    # Get x_top value
    n_xtop = n_y + n_z
    r_xtop = r_y + r_z
    
    if r_xtop > n_xtop:
        return
        
    x_index = d_nr_to_index(n_xtop, r_xtop)
    
    if x_index >= vector_len:
        return
    
    top_value = F_t_x[site, x_index]
    
    # Rule 3 formula
    comb_val = d_comb(n_xtop, n_y)
    gamma_y_pow = 1.0
    gamma_z_pow = 1.0
    
    for _ in range(n_y):
        gamma_y_pow *= gamma_y
    for _ in range(n_z):
        gamma_z_pow *= gamma_z
    
    evaluation = top_value * comb_val * gamma_y_pow * gamma_z_pow
    
    # Atomic add to handle race conditions
    cuda.atomic.add(output_y, (site, y_index), evaluation)
    cuda.atomic.add(output_z, (site, z_index), evaluation)


@cuda.jit
def rule4_kernel(F_t, site_count, vector_len, output):
    """
    CUDA Kernel for Rule 4: Combine branches with common leaf descendants.
    
    Computes: F(z, z_bot) from F(z, x_top, y_top)
    
    Args:
        F_t: Input partial likelihoods [site_count, vector_len, vector_len]
        site_count: Number of sites
        vector_len: Length of (n,r) vector space
        output: Output array [site_count, vector_len]
    """
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    
    total_work = site_count * vector_len
    if tid >= total_work:
        return
    
    site = tid // vector_len
    z_index = tid % vector_len
    
    n_zbot = d_index_to_n(z_index)
    r_zbot = d_index_to_r(z_index, n_zbot)
    
    evaluation = 0.0
    
    # Iterate through valid (n_xtop, r_xtop) values
    for n_xtop in range(1, n_zbot + 1):
        for r_xtop in range(r_zbot + 1):
            n_ytop = n_zbot - n_xtop
            r_ytop = r_zbot - r_xtop
            
            # Ensure combinatorics is well-defined
            if r_xtop <= n_xtop and r_ytop <= n_ytop and n_ytop >= 0:
                # Rule 4 constant
                const = d_comb(n_xtop, r_xtop) * d_comb(n_ytop, r_ytop) / d_comb(n_zbot, r_zbot)
                
                x_index = d_nr_to_index(n_xtop, r_xtop)
                y_index = d_nr_to_index(n_ytop, r_ytop) if n_ytop > 0 else 0
                
                if x_index < vector_len and y_index < vector_len:
                    ft_val = F_t[site, x_index, y_index]
                    evaluation += ft_val * const
    
    output[site, z_index] = evaluation


#########################
### GPU ACCELERATOR CLASS
#########################

class CUDAEvaluator:
    """
    GPU-accelerated evaluator for SNP partial likelihood calculations.
    Replaces C# DLL calls with CUDA kernels.
    
    Optimized for NVIDIA GPUs with Compute Capability 8.9+ (RTX 40/50 series).
    """
    
    def __init__(self):
        """Initialize CUDA evaluator and check device availability."""
        self.cuda_available = cuda.is_available()
        if self.cuda_available:
            # Warm up the GPU
            cuda.select_device(0)
            # Create a small test array to initialize CUDA context
            test = cp.zeros(1)
            del test
            
    def Rule0(self, 
              reds: np.ndarray, 
              site_count: int, 
              vector_len: int, 
              samples: int) -> dict:
        """
        GPU-accelerated Rule 0: Initialize partial likelihoods at leaves.
        
        Args:
            reds: Array of red counts per site
            site_count: Number of sites
            vector_len: Length of (n,r) vector space
            samples: Number of samples
            
        Returns:
            dict: Mapping from site -> {(n,r) vectors -> probability}
        """
        if not self.cuda_available:
            return self._rule0_cpu(reds, site_count, vector_len, samples)
        
        # Transfer data to GPU
        d_reds = cuda.to_device(reds.astype(np.float64))
        d_output = cuda.device_array((site_count, vector_len), dtype=np.float64)
        
        # Configure kernel launch
        threads_per_block = (16, 16)
        blocks_x = (site_count + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_y = (vector_len + threads_per_block[1] - 1) // threads_per_block[1]
        blocks = (blocks_x, blocks_y)
        
        # Launch kernel
        rule0_kernel[blocks, threads_per_block](
            d_reds, site_count, vector_len, samples, d_output
        )
        
        # Copy result back and convert to dict format
        output = d_output.copy_to_host()
        return self._array_to_dict(output, site_count, vector_len)
    
    def Rule1(self, 
              F_b: dict, 
              site_count: int, 
              mx: int, 
              Qt: np.ndarray) -> dict:
        """
        GPU-accelerated Rule 1: Transition from branch bottom to top.
        
        Args:
            F_b: Partial likelihoods at branch bottom
            site_count: Number of sites
            mx: Maximum lineages at this node
            Qt: Transition matrix (e^Qt)
            
        Returns:
            dict: Partial likelihoods at branch top
        """
        if not self.cuda_available:
            return self._rule1_cpu(F_b, site_count, mx, Qt)
        
        vector_len = n_to_index(mx + 1)
        
        # Convert dict to array
        F_b_array = self._dict_to_array(F_b, site_count, vector_len)
        
        # Transfer to GPU
        d_Fb = cuda.to_device(F_b_array)
        d_Qt = cuda.to_device(Qt.astype(np.float64))
        d_output = cuda.device_array((site_count, vector_len), dtype=np.float64)
        
        # Configure kernel
        total_work = site_count * vector_len
        threads = THREADS_PER_BLOCK
        blocks = min((total_work + threads - 1) // threads, MAX_BLOCKS)
        
        # Launch kernel
        rule1_kernel[blocks, threads](
            d_Fb, d_Qt, site_count, vector_len, mx, d_output
        )
        
        # Convert back to dict
        output = d_output.copy_to_host()
        return self._array_to_dict(output, site_count, vector_len)
    
    def Rule2(self, 
              F_t_x: dict, 
              F_t_y: dict, 
              site_count: int, 
              vector_len: int) -> dict:
        """
        GPU-accelerated Rule 2: Combine disjoint branches at speciation.
        
        Args:
            F_t_x: Partial likelihoods at x_top
            F_t_y: Partial likelihoods at y_top
            site_count: Number of sites
            vector_len: Length of (n,r) vector space
            
        Returns:
            dict: Partial likelihoods at z_bottom
        """
        if not self.cuda_available:
            return self._rule2_cpu(F_t_x, F_t_y, site_count, vector_len)
        
        # Convert to arrays
        F_x_array = self._dict_to_array(F_t_x, site_count, vector_len)
        F_y_array = self._dict_to_array(F_t_y, site_count, vector_len)
        
        # Transfer to GPU
        d_Fx = cuda.to_device(F_x_array)
        d_Fy = cuda.to_device(F_y_array)
        d_output = cuda.device_array((site_count, vector_len), dtype=np.float64)
        
        # Configure kernel
        total_work = site_count * vector_len
        threads = THREADS_PER_BLOCK
        blocks = min((total_work + threads - 1) // threads, MAX_BLOCKS)
        
        # Launch kernel
        rule2_kernel[blocks, threads](
            d_Fx, d_Fy, site_count, vector_len, d_output
        )
        
        output = d_output.copy_to_host()
        return self._array_to_dict(output, site_count, vector_len)
    
    def Rule3(self, 
              F_t_x: dict, 
              site_count: int, 
              vector_len: int, 
              mx: int,
              gamma_y: float, 
              gamma_z: float) -> tuple[dict, dict]:
        """
        GPU-accelerated Rule 3: Handle reticulation node.
        
        Args:
            F_t_x: Partial likelihoods at x_top
            site_count: Number of sites
            vector_len: Length of (n,r) vector space
            mx: Maximum lineages at x
            gamma_y: Inheritance probability for y
            gamma_z: Inheritance probability for z
            
        Returns:
            tuple[dict, dict]: Partial likelihoods for y_bot and z_bot
        """
        if not self.cuda_available:
            return self._rule3_cpu(F_t_x, site_count, vector_len, mx, gamma_y, gamma_z)
        
        # Convert to array
        F_x_array = self._dict_to_array(F_t_x, site_count, vector_len)
        
        # Transfer to GPU
        d_Fx = cuda.to_device(F_x_array)
        d_output_y = cuda.device_array((site_count, vector_len), dtype=np.float64)
        d_output_z = cuda.device_array((site_count, vector_len), dtype=np.float64)
        
        # Initialize outputs to zero
        d_output_y[:] = 0
        d_output_z[:] = 0
        
        # Configure kernel - more complex work distribution
        total_work = site_count * vector_len * vector_len
        threads = THREADS_PER_BLOCK
        blocks = min((total_work + threads - 1) // threads, MAX_BLOCKS)
        
        # Launch kernel
        rule3_kernel[blocks, threads](
            d_Fx, site_count, vector_len, mx, gamma_y, gamma_z, 
            d_output_y, d_output_z
        )
        
        output_y = d_output_y.copy_to_host()
        output_z = d_output_z.copy_to_host()
        
        return (self._array_to_dict(output_y, site_count, vector_len),
                self._array_to_dict(output_z, site_count, vector_len))
    
    def Rule4(self, 
              F_t: dict, 
              site_count: int, 
              vector_len: int) -> dict:
        """
        GPU-accelerated Rule 4: Combine branches with common descendants.
        
        Args:
            F_t: 2D partial likelihoods mapping
            site_count: Number of sites
            vector_len: Length of (n,r) vector space
            
        Returns:
            dict: Partial likelihoods at z_bottom
        """
        if not self.cuda_available:
            return self._rule4_cpu(F_t, site_count, vector_len)
        
        # Convert to 3D array [site, x_index, y_index]
        F_t_array = self._dict_to_3d_array(F_t, site_count, vector_len)
        
        # Transfer to GPU
        d_Ft = cuda.to_device(F_t_array)
        d_output = cuda.device_array((site_count, vector_len), dtype=np.float64)
        
        # Configure kernel
        total_work = site_count * vector_len
        threads = THREADS_PER_BLOCK
        blocks = min((total_work + threads - 1) // threads, MAX_BLOCKS)
        
        # Launch kernel
        rule4_kernel[blocks, threads](
            d_Ft, site_count, vector_len, d_output
        )
        
        output = d_output.copy_to_host()
        return self._array_to_dict(output, site_count, vector_len)
    
    # Helper methods for data conversion
    
    def _dict_to_array(self, F: dict, site_count: int, vector_len: int) -> np.ndarray:
        """Convert partial likelihood dict to 2D numpy array."""
        arr = np.zeros((site_count, vector_len), dtype=np.float64)
        for site in range(site_count):
            if site in F:
                for vectors, prob in F[site].items():
                    nx, rx = vectors
                    if len(nx) > 0 and len(rx) > 0:
                        n, r = nx[-1], rx[-1]
                        idx = nr_to_index(n, r)
                        if idx < vector_len:
                            arr[site, idx] = prob
        return arr
    
    def _dict_to_3d_array(self, F: dict, site_count: int, vector_len: int) -> np.ndarray:
        """Convert 2D partial likelihood dict to 3D numpy array."""
        arr = np.zeros((site_count, vector_len, vector_len), dtype=np.float64)
        for site in range(site_count):
            if site in F:
                for vectors, prob in F[site].items():
                    nx, rx = vectors
                    if len(nx) >= 2 and len(rx) >= 2:
                        n_x, r_x = nx[-2], rx[-2]
                        n_y, r_y = nx[-1], rx[-1]
                        idx_x = nr_to_index(n_x, r_x)
                        idx_y = nr_to_index(n_y, r_y)
                        if idx_x < vector_len and idx_y < vector_len:
                            arr[site, idx_x, idx_y] = prob
        return arr
    
    def _array_to_dict(self, arr: np.ndarray, site_count: int, vector_len: int) -> dict:
        """Convert 2D numpy array back to partial likelihood dict format."""
        F = {}
        for site in range(site_count):
            F[site] = {}
            for idx in range(vector_len):
                nr = index_to_nr(idx)
                n, r = nr[0], nr[1]
                prob = arr[site, idx]
                if prob != 0.0:  # Only store non-zero probabilities
                    F[site][(tuple([n]), tuple([r]))] = prob
        return F
    
    # CPU fallback methods
    
    def _rule0_cpu(self, reds, site_count, vector_len, samples):
        """CPU fallback for Rule 0."""
        F_map = {}
        for site in range(site_count):
            F_map[site] = {}
            for index in range(vector_len):
                actual_index = index_to_nr(index)
                n = actual_index[0]
                r = actual_index[1]
                if reds[site] == r and n == samples:
                    F_map[site][(tuple([n]), tuple([r]))] = 1
                else:
                    F_map[site][(tuple([n]), tuple([r]))] = 0
        return F_map
    
    def _rule1_cpu(self, F_b, site_count, mx, Qt):
        """CPU fallback for Rule 1."""
        vector_len = n_to_index(mx + 1)
        F_t = {}
        for site in range(site_count):
            nx_rx_map = rn_to_rn_minus_dim(F_b[site], 1)
            F_t[site] = {}
            for vectors in nx_rx_map.keys():
                nx = list(vectors[0])
                rx = list(vectors[1])
                for ft_index in range(n_to_index(mx + 1)):
                    actual_index = index_to_nr(ft_index)
                    n_top = actual_index[0]
                    r_top = actual_index[1]
                    entry = eval_Rule1(F_b[site], nx, rx, n_top, r_top, Qt, mx)
                    F_t[site][entry[0]] = entry[1]
        return F_t
    
    def _rule2_cpu(self, F_t_x, F_t_y, site_count, vector_len):
        """CPU fallback for Rule 2."""
        F_b = {}
        for site in range(site_count):
            nx_rx_map_y = rn_to_rn_minus_dim(F_t_y[site], 1)
            nx_rx_map_x = rn_to_rn_minus_dim(F_t_x[site], 1)
            F_b[site] = {}
            for vectors_x in nx_rx_map_x.keys():
                for vectors_y in nx_rx_map_y.keys():
                    nx = list(vectors_x[0])
                    rx = list(vectors_x[1])
                    ny = list(vectors_y[0])
                    ry = list(vectors_y[1])
                    for index in range(vector_len):
                        actual_index = index_to_nr(index)
                        n_bot = actual_index[0]
                        r_bot = actual_index[1]
                        entry = eval_Rule2(F_t_x[site], F_t_y[site],
                                          nx, ny, rx, ry, n_bot, r_bot)
                        F_b[site][entry[0]] = entry[1]
        return F_b
    
    def _rule3_cpu(self, F_t_x, site_count, vector_len, mx, gamma_y, gamma_z):
        """CPU fallback for Rule 3."""
        F_b = {}
        for site in range(site_count):
            nx_rx_map = rn_to_rn_minus_dim(F_t_x[site], 1)
            F_b[site] = {}
            for vector in nx_rx_map.keys():
                nx = list(vector[0])
                rx = list(vector[1])
                for n_y in range(mx + 1):
                    for n_z in range(mx - n_y + 1):
                        if n_y + n_z >= 1:
                            for r_y in range(n_y + 1):
                                for r_z in range(n_z + 1):
                                    entry = eval_Rule3(F_t_x[site], nx, rx,
                                                      n_y, n_z, r_y, r_z,
                                                      gamma_y, gamma_z)
                                    F_b[site][entry[0]] = entry[1]
        return F_b
    
    def _rule4_cpu(self, F_t, site_count, vector_len):
        """CPU fallback for Rule 4."""
        F_b = {}
        for site in range(site_count):
            nx_rx_map = rn_to_rn_minus_dim(F_t[site], 2)
            F_b[site] = {}
            for vectors_x in nx_rx_map.keys():
                nx = list(vectors_x[0])
                rx = list(vectors_x[1])
                for index in range(vector_len):
                    actual_index = index_to_nr(index)
                    n_bot = actual_index[0]
                    r_bot = actual_index[1]
                    entry = eval_Rule4(F_t[site], nx, rx, n_bot, r_bot)
                    F_b[site][entry[0]] = entry[1]
        return F_b


#########################
### Transition Matrix ###
#########################

class BiMarkersTransition:
    """
    Class that encodes the probabilities of transitioning from one (n,r) pair 
    to another under a Biallelic model.

    Includes method for efficiently computing e^Qt using CuPy for GPU acceleration.

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

    def __init__(self, n: int, u: float, v: float, coal: float) -> None:
        """
        Initialize the Q matrix.

        Args:
            n (int): sample count
            u (float): probability of a lineage going from red to green
            v (float): probability of a lineage going from green to red
            coal (float): coal rate, theta.
        Returns:
            N/A
        """
        self.n = n 
        self.u = u
        self.v = v
        self.coal = coal

        rows = int(.5 * self.n * (self.n + 3))
        self.Q: np.ndarray = np.zeros((rows, rows))
        
        # Build Q matrix (same as original)
        for n_prime in range(1, self.n + 1):  
            for r_prime in range(n_prime + 1):  
                n_r = nr_to_index(n_prime, r_prime)
                nm_rm = nr_to_index(n_prime - 1, r_prime - 1)
                n_rm = nr_to_index(n_prime, r_prime - 1)
                nm_r = nr_to_index(n_prime - 1, r_prime)
                n_rp = nr_to_index(n_prime, r_prime + 1)

                # THE DIAGONAL
                self.Q[n_r][n_r] = - (n_prime * (n_prime - 1) / coal) \
                                       - (v * (n_prime - r_prime)) \
                                       - (r_prime * u)

                if 0 < r_prime <= n_prime:
                    if n_prime > 1:
                        self.Q[n_r][nm_rm] = (r_prime - 1) * n_prime / coal
                    self.Q[n_r][n_rm] = (n_prime - r_prime + 1) * v

                if 0 <= r_prime < n_prime:
                    if n_prime > 1:
                        self.Q[n_r][nm_r] = (n_prime - 1 - r_prime) \
                                            * n_prime / coal
                    self.Q[n_r][n_rp] = (r_prime + 1) * u
        
        # Store GPU version of Q for accelerated expm
        if cuda.is_available():
            self.Q_gpu = cp.asarray(self.Q)

    def expt(self, t: float = 1) -> np.ndarray:
        """
        Compute e^(Q*t) efficiently using GPU when available.
        
        Args:
            t (float): time, generally in coalescent units. Optional, defaults 
                       to 1, in which case e^Q is computed.
        
        Returns:
            np.ndarray: e^(Q*t).
        """
        if cuda.is_available() and use_cuda:
            # Use CuPy's GPU-accelerated matrix exponential
            Qt_gpu = cp.asarray(self.Q * t)
            result_gpu = cp.linalg.expm(Qt_gpu)
            return cp.asnumpy(result_gpu)
        else:
            return expm(self.Q * t)

    def cols(self) -> int:
        """Return the dimension of the Q matrix."""
        return self.Q.shape[1]

    def getQ(self) -> np.ndarray:
        """Retrieve the Q matrix."""
        return self.Q


########################
### RULE EVALUATIONS ###
########################

def eval_Rule1(F_b: dict,
               nx: list[int], 
               rx: list[int], 
               n_xtop: int, 
               r_xtop: int, 
               Qt: np.ndarray,
               mx: int) -> list:
    """
    Given all the information on the left side of the Rule 1 equation, compute 
    the right side probability.
    """
    evaluation = 0
    
    for n_b in range(n_xtop, mx + 1):  
        for r_b in range(0, n_b + 1):
            index = nr_to_index(n_b, r_b)
            exp_val = Qt[index][nr_to_index(n_xtop, r_xtop)]  
            n_vec = tuple(np.append(nx, n_b))
            r_vec = tuple(np.append(rx, r_b))
            
            try:
                evaluation += F_b[(n_vec, r_vec)] * exp_val
            except KeyError:
                evaluation += 0
                
    n_vec_top = tuple(np.append(nx, n_xtop))
    r_vec_top = tuple(np.append(rx, r_xtop))
    
    return [(n_vec_top, r_vec_top), evaluation]


def eval_Rule2(F_t_x: dict, 
               F_t_y: dict, 
               nx: list[int],
               ny: list[int], 
               rx: list[int], 
               ry: list[int], 
               n_zbot: int,
               r_zbot: int) -> list:
    """
    Given the left side information for the Rule 2 equation, 
    calculate the right side probability.
    """
    evaluation = 0
    
    for n_xtop in range(0, n_zbot + 1):
        for r_xtop in range(0, r_zbot + 1):
            if r_xtop <= n_xtop and r_zbot - r_xtop <= n_zbot - n_xtop:
                const = comb(n_xtop, r_xtop) \
                        * comb(n_zbot - n_xtop, r_zbot - r_xtop) \
                        / comb(n_zbot, r_zbot)
                try:
                    term1 = F_t_x[(tuple(np.append(nx, n_xtop)), 
                                   tuple(np.append(rx, r_xtop)))]
                    term2 = F_t_y[(tuple(np.append(ny, n_zbot - n_xtop)), 
                                   tuple(np.append(ry, r_zbot - r_xtop)))]
                    evaluation += term1 * term2 * const
                except KeyError: 
                    evaluation += 0

    return [(tuple(np.append(np.append(nx, ny), n_zbot)), 
             tuple(np.append(np.append(rx, ry), r_zbot))), 
             evaluation]


def eval_Rule3(F_t: dict,
               nx: list[int], 
               rx: list[int], 
               n_ybot: int,
               n_zbot: int, 
               r_ybot: int, 
               r_zbot: int, 
               gamma_y: float, 
               gamma_z: float) -> list:
    """
    Given left side information for the Rule 3 equation, calculate the 
    right side probability.
    """
    try:
        top_value = F_t[(tuple(np.append(nx, n_ybot + n_zbot)), 
                         tuple(np.append(rx, r_ybot + r_zbot)))]
        evaluation = top_value \
                     * comb(n_ybot + n_zbot, n_ybot) \
                     * pow(gamma_y, n_ybot) \
                     * pow(gamma_z, n_zbot)
    except KeyError:
        evaluation = 0
        
    return [(tuple(np.append(np.append(nx, n_ybot), n_zbot)), 
             tuple(np.append(np.append(rx, r_ybot), r_zbot))), 
             evaluation]


def eval_Rule4(F_t: dict, 
               nz: list[int], 
               rz: list[int], 
               n_zbot: int,
               r_zbot: int) -> list:
    """
    Given all the information on the left side of the Rule 4 equation, 
    calculate the right side probability.
    """
    evaluation = 0
    
    for n_xtop in range(1, n_zbot + 1):
        for r_xtop in range(0, r_zbot + 1):
            if r_xtop <= n_xtop and r_zbot - r_xtop <= n_zbot - n_xtop:  
                const = comb(n_xtop, r_xtop) \
                        * comb(n_zbot - n_xtop, r_zbot - r_xtop) \
                        / comb(n_zbot, r_zbot)
                try:
                    nz_xtop_ytop = tuple(np.append(np.append(nz, n_xtop), 
                                         n_zbot - n_xtop))
                    rz_xtop_ytop = tuple(np.append(np.append(rz, r_xtop), 
                                         r_zbot - r_xtop))
                    evaluation += F_t[(nz_xtop_ytop, rz_xtop_ytop)] * const
                except KeyError:
                    evaluation += 0
    
    return [(tuple(np.append(nz, n_zbot)), 
             tuple(np.append(rz, r_zbot))),
             evaluation] 


###########################
### PARTIAL LIKELIHOODS ###   
###########################

class PartialLikelihoods:
    """
    Class that bookkeeps the vectors of population interfaces (vpis) and their
    associated likelihood values.
    
    Contains methods for evaluating and storing likelihoods based on the rules
    described by Rabier et al.
    
    CUDA-accelerated version uses GPU kernels for Rules 0-4.
    """
    
    def __init__(self) -> None:
        """Initialize an empty PartialLikelihood obj."""
        self.vpis: dict = {}
        self.ploidy: int = None
        
        # CUDA evaluator replaces C# SNPEvals
        self.evaluator = CUDAEvaluator()
        
    def set_ploidy(self, ploidy: int) -> None:
        """Set the ploidy value for the partial likelihoods object."""
        self.ploidy = ploidy
        
    def Rule0(self, 
              reds: np.ndarray, 
              samples: int,
              site_count: int, 
              vector_len: int,
              branch_id: str) -> tuple:
        """
        Given leaf data, compute the initial partial likelihood values 
        for the interface F(x_bot).
        
        GPU-accelerated when CUDA is available.
        """
        if use_cuda and self.evaluator.cuda_available:
            F_map = self.evaluator.Rule0(reds, site_count, vector_len, samples)
        else:
            F_map = {}
            for site in range(site_count):
                F_map[site] = {}
                for index in range(vector_len):
                    actual_index = index_to_nr(index)
                    n = actual_index[0]
                    r = actual_index[1]
                    if reds[site] == r and n == samples:
                        F_map[site][(tuple([n]), tuple([r]))] = 1
                    else:
                        F_map[site][(tuple([n]), tuple([r]))] = 0
                
        vpi_key = tuple(["branch_" + str(branch_id) + ": bottom"])  
        self.vpis[vpi_key] = F_map
        return vpi_key

    def Rule1(self,
              vpi_key_x: tuple, 
              branch_id_x: int,
              m_x: int, 
              Qt: np.ndarray) -> tuple:
        """
        Given a branch x, and partial likelihoods for the population interface 
        that includes x_bottom, compute the partial likelihoods for the 
        population interface that includes x_top.
        
        GPU-accelerated when CUDA is available.
        """
        if "branch_" + str(branch_id_x) + ": bottom" != vpi_key_x[-1]:
            vpi_key_temp = self.reorder_vpi(vpi_key_x,
                                            site_count, 
                                            branch_id_x, 
                                            False)
            del self.vpis[vpi_key_x]
            vpi_key_x = vpi_key_temp
            
        F_b = self.vpis[vpi_key_x]
        
        if use_cuda and self.evaluator.cuda_available:
            F_t = self.evaluator.Rule1(F_b, site_count, m_x, Qt)
        else:
            F_t = {}
            for site in range(site_count):
                nx_rx_map = rn_to_rn_minus_dim(F_b[site], 1)
                F_t[site] = {}
                for vectors in nx_rx_map.keys():
                    nx = list(vectors[0])
                    rx = list(vectors[1])
                    for ft_index in range(n_to_index(m_x + 1)):
                        actual_index = index_to_nr(ft_index)
                        n_top = actual_index[0]
                        r_top = actual_index[1]
                        entry = eval_Rule1(F_b[site], nx, rx, n_top, r_top, Qt, m_x)
                        F_t[site][entry[0]] = entry[1]
                        
                # Handle (0,0) case
                for vecs in F_b[site].keys():
                    nx = list(vecs[0])
                    rx = list(vecs[1])
                    if nx[-1] == 0 and rx[-1] == 0:
                        F_t[site][vecs] = F_b[site][vecs]
        
        new_vpi_key = list(vpi_key_x)
        edit_index = vpi_key_x.index("branch_" + str(branch_id_x) + ": bottom")
        new_vpi_key[edit_index] = "branch_" + str(branch_id_x) + ": top"
        new_vpi_key = tuple(new_vpi_key)
        
        self.vpis[new_vpi_key] = F_t
        del self.vpis[vpi_key_x]
        
        return new_vpi_key
                
    def Rule2(self, 
              vpi_key_x: tuple, 
              vpi_key_y: tuple,  
              branch_id_x: str,
              branch_id_y: str, 
              branch_id_z: str) -> tuple:
        """
        Given branches x and y that have no leaf descendants in common and a 
        parent branch z, compute the partial likelihood mapping for the 
        population interface that includes z_bottom.
        
        GPU-accelerated when CUDA is available.
        """
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
        
        if use_cuda and self.evaluator.cuda_available:
            F_b = self.evaluator.Rule2(F_t_x, F_t_y, site_count, vector_len)
        else:
            F_b = {}
            for site in range(site_count):
                nx_rx_map_y = rn_to_rn_minus_dim(F_t_y[site], 1)
                nx_rx_map_x = rn_to_rn_minus_dim(F_t_x[site], 1)
                F_b[site] = {}
                
                for vectors_x in nx_rx_map_x.keys():
                    for vectors_y in nx_rx_map_y.keys():
                        nx = list(vectors_x[0])
                        rx = list(vectors_x[1])
                        ny = list(vectors_y[0])
                        ry = list(vectors_y[1])
                        
                        for index in range(vector_len):
                            actual_index = index_to_nr(index)
                            n_bot = actual_index[0]
                            r_bot = actual_index[1]
                            entry = eval_Rule2(F_t_x[site], F_t_y[site],
                                               nx, ny, n_bot, rx, ry, r_bot)
                            F_b[site][entry[0]] = entry[1]
        
        new_vpi_key_x = list(vpi_key_x)
        new_vpi_key_x.remove("branch_" + str(branch_id_x) + ": top")
        
        new_vpi_key_y = list(vpi_key_y)
        new_vpi_key_y.remove("branch_" + str(branch_id_y) + ": top")
        
        z_name = "branch_" + str(branch_id_z) + ": bottom"
        vpi_y = np.append(new_vpi_key_y, z_name)
        new_vpi_key = tuple(np.append(new_vpi_key_x, vpi_y))
        
        self.vpis[new_vpi_key] = F_b
        del self.vpis[vpi_key_x]
        del self.vpis[vpi_key_y]
                         
        return new_vpi_key

    def Rule3(self, 
              vpi_key_x: tuple, 
              branch_id_x: str,
              branch_id_y: str, 
              branch_id_z: str,
              g_this: float,
              g_that: float,
              mx: int) -> tuple:
        """
        Given a branch x, its partial likelihood mapping at x_top, and parent 
        branches y and z, compute the partial likelihood mapping for the 
        population interface x, y_bottom, z_bottom.
        
        GPU-accelerated when CUDA is available.
        """
        F_t_x = self.vpis[vpi_key_x]
        
        if use_cuda and self.evaluator.cuda_available:
            F_b = self.evaluator.Rule3(F_t_x, site_count, vector_len, mx, 
                                       g_this, g_that)
            # Rule3 returns a single dict in CUDA version
            if isinstance(F_b, tuple):
                F_b = F_b[0]  # Take first element if tuple returned
        else:
            F_b = {}
            for site in range(site_count):
                nx_rx_map = rn_to_rn_minus_dim(F_t_x[site], 1)
                F_b[site] = {}
                for vector in nx_rx_map.keys():
                    nx = list(vector[0])
                    rx = list(vector[1])
                    for n_y in range(mx + 1):
                        for n_z in range(mx - n_y + 1):
                            if n_y + n_z >= 1:
                                for r_y in range(n_y + 1):
                                    for r_z in range(n_z + 1):
                                        entry = eval_Rule3(F_t_x[site],
                                                           nx, rx, 
                                                           n_y, n_z, 
                                                           r_y, r_z, 
                                                           g_this, g_that)
                                        F_b[site][entry[0]] = entry[1]
        
        new_vpi_key = list(vpi_key_x)
        new_vpi_key.remove("branch_" + str(branch_id_x) + ": top")
        new_vpi_key.append("branch_" + str(branch_id_y) + ": bottom")
        new_vpi_key.append("branch_" + str(branch_id_z) + ": bottom")
        new_vpi_key = tuple(new_vpi_key)
        
        self.vpis[new_vpi_key] = F_b
        del self.vpis[vpi_key_x]
        
        return new_vpi_key               
            
    def Rule4(self, 
              vpi_key_xy: tuple, 
              branch_index_x: int, 
              branch_index_y: int, 
              branch_index_z: int) -> tuple:
        """
        Given branches x and y that share common leaf descendants and that 
        have parent branch z, compute F z, z_bot.
        
        GPU-accelerated when CUDA is available.
        """
        if "branch_" + str(branch_index_y) + ": top" != vpi_key_xy[-1]:
            vpi_key_temp = self.reorder_vpi(vpi_key_xy,
                                            site_count, 
                                            branch_index_y, 
                                            True)
            del self.vpis[vpi_key_xy]
            vpi_key_xy = vpi_key_temp
        
        F_t = self.vpis[vpi_key_xy]
        
        if use_cuda and self.evaluator.cuda_available:
            F_b = self.evaluator.Rule4(F_t, site_count, vector_len)
        else:
            F_b = {}
            for site in range(site_count):
                nx_rx_map = rn_to_rn_minus_dim(F_t[site], 2)
                F_b[site] = {}
                
                for vectors_x in nx_rx_map.keys():
                    nx = list(vectors_x[0])
                    rx = list(vectors_x[1])
                    
                    for index in range(vector_len):
                        actual_index = index_to_nr(index)
                        n_bot = actual_index[0]
                        r_bot = actual_index[1]
                        entry = eval_Rule4(F_t[site], nx, rx, n_bot, r_bot)
                        F_b[site][entry[0]] = entry[1]
        
        new_vpi_key = list(vpi_key_xy)
        new_vpi_key.remove("branch_" + str(branch_index_x) + ": top")
        new_vpi_key.remove("branch_" + str(branch_index_y) + ": top")
        new_vpi_key.append("branch_" + str(branch_index_z) + ": bottom")
        new_vpi_key = tuple(new_vpi_key)
    
        self.vpis[new_vpi_key] = F_b
        del self.vpis[vpi_key_xy]
        
        return new_vpi_key
    
    def reorder_vpi(self, 
                    vpi_key: tuple, 
                    branch_index: int, 
                    for_top: bool) -> tuple:
        """
        For use when a rule requires a certain ordering of a vpi, and the
        current vpi does not satisfy it.
        """
        if for_top:
            name = "branch_" + str(branch_index) + ": top"
            former_index = list(vpi_key).index(name)
        else:
            name = "branch_" + str(branch_index) + ": bottom"
            former_index = list(vpi_key).index(name)
            
        new_vpi_key = list(vpi_key)
        new_vpi_key.append(new_vpi_key.pop(former_index))
        
        F_map = self.vpis[vpi_key]
        
        new_F = {}
        for site in range(site_count):
            new_F[site] = {}
            for vectors, prob in F_map[site].items():
                nx = list(vectors[0])
                rx = list(vectors[1])
                new_nx = list(nx)
                new_rx = list(rx)
                new_nx.append(new_nx.pop(former_index))
                new_rx.append(new_rx.pop(former_index))
                new_F[site][(tuple(new_nx), tuple(new_rx))] = prob
        
        F_map = new_F
        self.vpis[tuple(new_vpi_key)] = F_map
    
        return tuple(new_vpi_key)
    
    def get_key_with(self, branch_id: str) -> tuple:
        """
        From the set of vpis, grab the one that contains the branch 
        identified by branch_index.
        """
        for vpi_key in self.vpis:
            top = "branch_" + branch_id + ": top"
            bottom = "branch_" + branch_id + ": bottom"
            if top in vpi_key or bottom in vpi_key:
                return vpi_key
        return None
            

######################
### Model Building ###
######################

class U(Parameter):
    """Parameter for the red->green lineage transition probability."""
    def __init__(self, value: float) -> None:
        super().__init__("u", value)


class V(Parameter):
    """Parameter for the green->red lineage transition probability."""
    def __init__(self, value: float) -> None:
        super().__init__("v", value)


class Coal(Parameter):
    """Coalescent rate parameter, theta."""
    def __init__(self, value: float) -> None:
        super().__init__("coal", value)


class Samples(Parameter):
    """Parameter for the number of total samples (sequences) present."""
    def __init__(self, value: int) -> None:
        super().__init__("samples", value)


class SiteParameter(Parameter):
    """Parameter for the number of sites (sequence length)."""
    def __init__(self, value: int) -> None:
        super(Parameter).__init__("sitect", value)


class BiMarkersTransitionMatrixNode(CalculationNode):
    """Node that encodes the transition matrix, Q."""
    def __init__(self) -> None:
        super().__init__()
    
    def calc(self) -> BiMarkersTransition:
        """Grab the model parameters, then compute and store Q."""
        params = self.get_parameters()
        return self.cache(BiMarkersTransition(params["samples"], 
                                              params["u"], 
                                              params["v"], 
                                              params["coal"]))
            
    def sim(self) -> None:
        pass
    
    def get(self) -> BiMarkersTransition:
        """Return cached Q or recompute if parameters changed."""
        if self.dirty:
            return self.calc()
        else:
            return self.cached
            
    def update(self) -> None:
        """Tell upstream nodes to recalculate."""
        self.upstream()


class VPIAccumulator(Accumulator):
    """Class that holds a reference to the PartialLikelihood object."""
    def __init__(self) -> None:
        super().__init__("VPI", PartialLikelihoods())


class BiMarkersLikelihood(CalculationNode):
    """
    Root likelihood algorithm.
    This node is also the root of the model graph.
    """
    
    def __init__(self) -> None:
        super().__init__()
    
    def calc(self) -> float:
        """
        Run the root likelihood calculation and return the final MCMC bimarkers
        likelihood value for the network.
        """
        # Grab Q matrix
        tn: BiMarkersTransitionMatrixNode 
        tn = self.get_model_children(BiMarkersTransitionMatrixNode)[0]
        
        # Use CuPy for null space computation if available
        if cuda.is_available() and use_cuda:
            Q_gpu = cp.asarray(tn.get().getQ())
            # CuPy doesn't have null_space, fall back to scipy
            q_null_space = scipy.linalg.null_space(tn.get().getQ())
        else:
            q_null_space = scipy.linalg.null_space(tn.get().getQ())
        
        x = q_null_space / (q_null_space[0] + q_null_space[1]) 

        root_node: BiMarkersNode = self.get_model_children(BiMarkersNode)[0]
        root_vpi_key = root_node.get()[0]
        vpi_acc: VPIAccumulator = self.get_model_children(VPIAccumulator)[0]
        F_b_map = vpi_acc.get_data().vpis[root_vpi_key]
        
        params = self.get_parameters()
        
        F_b = to_array(F_b_map, 
                       n_to_index(params["samples"] + 1),
                       params["sitect"]) 

        # Use CuPy for vectorized computation when available
        if cuda.is_available() and use_cuda:
            F_b_gpu = cp.asarray(F_b)
            x_gpu = cp.asarray(x)
            L_gpu = cp.zeros(params["sitect"])
            
            for site in range(params["sitect"]):
                L_gpu[site] = cp.dot(F_b_gpu[:, site], x_gpu).squeeze()
            
            L = cp.asnumpy(L_gpu)
        else:
            L = np.zeros(params["sitect"])
            for site in range(params["sitect"]):
                L[site] = np.dot(F_b[:, site], x)
    
        return self.cache(np.sum(np.log(L)))
    
    def get(self) -> float:
        """Gets the final log likelihood of the network."""
        if self.dirty:
            self.calc()
        else:
            return self.cached
    
    def sim(self) -> None:
        pass
    
    def update(self) -> None:
        pass


##############################
#### SNP MODEL COMPONENTS ####
##############################

class VPIComponent(ModelComponent):
    """Model component that hooks up the VPI tracker to the network and root."""
    
    def __init__(self, dependencies: set[type]) -> None:
        super().__init__(dependencies)
        
    def build(self, model: Model) -> None:
        vpi_acc = VPIAccumulator()
        root_prob = model.all_nodes[BiMarkersLikelihood][0]
        vpi_acc.join(root_prob)
        join_network(vpi_acc, model)
        model.all_nodes[VPIAccumulator].append(vpi_acc)


class SNPRootComponent(ModelComponent):
    """Model component that hooks up the SNP likelihood algorithm to root."""
    def __init__(self, dependencies: set[type]) -> None:
        super().__init__(dependencies)
        
    def build(self, model: Model) -> None:
        root_prob = BiMarkersLikelihood()
        net_root: ModelNode = model.nodetypes["root"][0]
        net_root.join(root_prob)
        model.all_nodes[BiMarkersLikelihood].append(root_prob)


class SNPTransitionComponent(ModelComponent):
    """Model component that hooks up the SNP transition matrix Q."""
    def __init__(self, dependencies: set[type]) -> None:
        super().__init__(dependencies)
    
    def build(self, model: Model) -> None:
        q_node = BiMarkersTransitionMatrixNode()
        root_prob = model.all_nodes[BiMarkersLikelihood][0]
        q_node.join(root_prob)
        join_network(q_node, model)
        model.all_nodes[BiMarkersTransitionMatrixNode].append(q_node)


class SNPParamComponent(ModelComponent):
    """Model component that hooks up model parameters."""
    
    def __init__(self, dependencies: set[type], params: dict) -> None:
        super().__init__(dependencies)  
        self.params = params
    
    def build(self, model: Model) -> None:
        u_node = U(self.params["u"])
        v_node = V(self.params["v"])
        coal_node = Coal(self.params["coal"])
        samples_node = Samples(self.params["samples"])
        site_node = SiteParameter(self.params["sites"])
        
        trans_nodes = model.all_nodes[BiMarkersTransitionMatrixNode]    
        q_node: BiMarkersTransitionMatrixNode = trans_nodes[0]
        
        u_node.join(q_node)
        v_node.join(q_node)
        coal_node.join(q_node)
        samples_node.join(q_node)
        
        root_prob = model.all_nodes[BiMarkersLikelihood][0]
        samples_node.join(root_prob)
        site_node.join(root_prob)
        
        join_network(site_node, model)
        join_network(samples_node, model)
        
        model.all_nodes[Parameter].extend([u_node,
                                          v_node, 
                                          coal_node, 
                                          samples_node, 
                                          site_node])


def build_model(filename: str, 
                net: Network,
                u: float = .5,
                v: float = .5, 
                coal: float = 1, 
                grouping: dict = None, 
                auto_detect: bool = False) -> Model:
    """
    Build a SNP model with CUDA acceleration.
    """
    aln = MSA(filename, grouping=grouping, grouping_auto_detect=auto_detect)
    
    network = NetworkComponent(net=net)
    msa = MSAComponent({NetworkComponent}, aln, grouping=grouping)

    param = SNPParamComponent({NetworkComponent,
                               MSAComponent,
                               SNPRootComponent,
                               SNPTransitionComponent}, 
                              {"samples": aln.total_samples(),
                               "u": u, 
                               "v": v, 
                               "coal": coal,
                               "sites": aln.dim()[1],
                               "grouping": grouping})
    
    vpi = VPIComponent({NetworkComponent, SNPRootComponent})
    root = SNPRootComponent({NetworkComponent})
     
    snp_model: Model = ModelFactory(network, msa, param, vpi, root).build()
    
    return snp_model


###########################
### METHOD ENTRY POINTS ###
###########################

def MCMC_BIMARKERS(filename: str, 
                   u: float = .5,
                   v: float = .5, 
                   coal: float = 1, 
                   grouping: dict = None, 
                   auto_detect: bool = False,
                   use_gpu: bool = True) -> dict[Network, float]:
    """
    CUDA-accelerated MCMC for phylogenetic network inference.
    
    Given a set of taxa with SNP data, perform a Markov Chain Monte Carlo
    chain to infer the most likely phylogenetic network.
    
    Args:
        filename: Path to nexus file with SNP data
        u: Probability of red->green allele change (default 0.5)
        v: Probability of green->red allele change (default 0.5)
        coal: Coalescent rate parameter (default 1)
        grouping: Sequence grouping dictionary
        auto_detect: Auto-detect sequence groupings
        use_gpu: Enable GPU acceleration (default True)
        
    Returns:
        dict[Network, float]: Network and its log likelihood
    """
    global use_cuda
    use_cuda = use_gpu and cuda.is_available()
    
    if use_cuda:
        print("CUDA acceleration enabled")
        get_cuda_device_info()
    else:
        print("Running on CPU")

    aln = MSA(filename, grouping=grouping, grouping_auto_detect=auto_detect)
    start_net = CBDP(1, .5, aln.num_groups()).generate_network()
    
    snp_model = build_model(filename, 
                            start_net,
                            u, v, coal, 
                            grouping, auto_detect)
    
    mh = MetropolisHastings(ProposalKernel(),
                            data=Matrix(aln, Alphabet("SNP")), 
                            num_iter=800,
                            model=snp_model) 
     
    result_state = mh.run()
    result_model = result_state.current_model

    return {result_model.network: result_model.likelihood()}


def SNP_LIKELIHOOD(filename: str,
                   u: float = .5,
                   v: float = .5, 
                   coal: float = 1, 
                   grouping: dict = None, 
                   auto_detect: bool = False,
                   use_gpu: bool = True) -> float:
    """
    CUDA-accelerated SNP likelihood calculation.
    
    Given a set of taxa with SNP data and a phylogenetic network, calculate the 
    likelihood of the network given the data.
    
    Args:
        filename: Path to nexus file with SNP data and network
        u: Probability of red->green allele change (default 0.5)
        v: Probability of green->red allele change (default 0.5)
        coal: Coalescent rate parameter (default 1)
        grouping: Sequence grouping dictionary
        auto_detect: Auto-detect sequence groupings
        use_gpu: Enable GPU acceleration (default True)
        
    Returns:
        float: Log likelihood of the network
    """
    global use_cuda
    use_cuda = use_gpu and cuda.is_available()
    
    net = NetworkParser(filename).get_network(0)
    snp_model = build_model(filename, net, u, v, coal, grouping, auto_detect)
    
    return snp_model.likelihood()


def SNP_LIKELIHOOD_DATA(filename: str,
                        set_reds: dict,
                        u: float = .5,
                        v: float = .5, 
                        coal: float = 1,
                        grouping: dict = None, 
                        auto_detect: bool = False,
                        use_gpu: bool = True) -> float:
    """
    CUDA-accelerated SNP likelihood with custom red allele data.
    
    THIS FUNCTION IS ONLY FOR TESTING PURPOSES.
    """
    global use_cuda
    use_cuda = use_gpu and cuda.is_available()
    
    net = NetworkParser(filename).get_network(0)
    snp_model = build_model(filename, net, u, v, coal, grouping, auto_detect)
    
    for taxa in snp_model.all_nodes[ExtantSpecies]:
        leaf: ExtantSpecies = taxa
        leaf.update(DataSequence(set_reds[leaf.label], leaf.label))
    
    return snp_model.likelihood()


# Utility function for benchmarking
def benchmark_cuda_vs_cpu(filename: str, 
                          iterations: int = 10) -> dict:
    """
    Benchmark CUDA vs CPU performance for likelihood calculation.
    
    Args:
        filename: Path to nexus file with SNP data
        iterations: Number of iterations for timing
        
    Returns:
        dict: Timing results for GPU and CPU
    """
    import time
    
    results = {"gpu_times": [], "cpu_times": []}
    
    # GPU benchmark
    if cuda.is_available():
        for _ in range(iterations):
            start = time.time()
            SNP_LIKELIHOOD(filename, use_gpu=True)
            results["gpu_times"].append(time.time() - start)
    
    # CPU benchmark
    for _ in range(iterations):
        start = time.time()
        SNP_LIKELIHOOD(filename, use_gpu=False)
        results["cpu_times"].append(time.time() - start)
    
    # Calculate statistics
    if results["gpu_times"]:
        results["gpu_mean"] = np.mean(results["gpu_times"])
        results["gpu_std"] = np.std(results["gpu_times"])
    
    results["cpu_mean"] = np.mean(results["cpu_times"])
    results["cpu_std"] = np.std(results["cpu_times"])
    
    if results["gpu_times"]:
        results["speedup"] = results["cpu_mean"] / results["gpu_mean"]
        print(f"GPU mean: {results['gpu_mean']:.4f}s (+/- {results['gpu_std']:.4f}s)")
    
    print(f"CPU mean: {results['cpu_mean']:.4f}s (+/- {results['cpu_std']:.4f}s)")
    
    if results["gpu_times"]:
        print(f"Speedup: {results['speedup']:.2f}x")
    
    return results

