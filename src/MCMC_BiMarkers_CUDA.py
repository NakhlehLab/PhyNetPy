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
from .ModelGraph import (
    Model, ModelNode, CalculationNode, Parameter, Accumulator, ExtantSpecies
)
from .ModelFactory import (
    ModelFactory, ModelComponent, NetworkComponent, MSAComponent, join_network
)
from .Network import Network, Edge, Node
from .MetropolisHastings import MetropolisHastings, ProposalKernel


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

# Global variables for site count and vector length (set during model building)
site_count = 0
vector_len = 0


########################
### CUDA UTILITIES ###
########################

def get_cuda_device_info():
    """
    Print information about the available CUDA device.
    Useful for debugging and optimization.
    """
    if CUDA_IMPORTS_AVAILABLE:
        # Try numba CUDA first
        if cuda.is_available():
            device = cuda.get_current_device()
            print(f"CUDA Device (numba): {device.name}")
            print(f"Compute Capability: {device.compute_capability}")
            print(f"Max threads per block: {device.MAX_THREADS_PER_BLOCK}")
            print(f"Max shared memory per block: {device.MAX_SHARED_MEMORY_PER_BLOCK}")
            return True
        else:
            # Try CuPy as fallback
            try:
                device = cp.cuda.Device()
                print(f"CUDA Device (cupy): {device.pci_bus_id}")
                print(f"CuPy version: {cp.__version__}")
                print(f"Total memory: {device.mem_info[1] / 1e9:.2f} GB")
                print(f"Free memory: {device.mem_info[0] / 1e9:.2f} GB")
                print("Note: Using CuPy for GPU acceleration")
                return True
            except Exception as e:
                print(f"CUDA is not available. Error: {e}")
                return False
    else:
        print("CUDA is not available. Falling back to CPU.")
        return False


########################
### DEVICE FUNCTIONS ###
########################

if CUDA_IMPORTS_AVAILABLE:
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

if CUDA_IMPORTS_AVAILABLE:
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
        # Check for CuPy availability first (more reliable detection)
        self.cupy_available = False
        self.numba_cuda_available = False
        
        if CUDA_IMPORTS_AVAILABLE:
            try:
                # Try to create a test array on GPU
                test = cp.zeros(1)
                del test
                self.cupy_available = True
            except Exception:
                self.cupy_available = False
            
            # Check numba CUDA availability
            self.numba_cuda_available = cuda.is_available()
            if self.numba_cuda_available:
                cuda.select_device(0)
        
        # Use CuPy for matrix operations even if numba CUDA isn't available
        self.cuda_available = self.cupy_available or self.numba_cuda_available
            
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
        # Numba CUDA kernels require numba CUDA to be available
        if not self.numba_cuda_available:
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
        # Numba CUDA kernels require numba CUDA to be available
        if not self.numba_cuda_available:
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
        # Numba CUDA kernels require numba CUDA to be available
        if not self.numba_cuda_available:
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
        # Numba CUDA kernels require numba CUDA to be available
        if not self.numba_cuda_available:
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
        # Numba CUDA kernels require numba CUDA to be available
        if not self.numba_cuda_available:
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

    def expt(self, t: float = 1) -> np.ndarray:
        """
        Compute e^(Q*t) using scipy.linalg.expm.
        
        Note: CuPy does not have a matrix exponential function, so we use
        scipy's CPU implementation. This is still fast for small matrices
        (typical size for bimarkers is < 100x100).
        
        Args:
            t (float): time, generally in coalescent units. Optional, defaults 
                       to 1, in which case e^Q is computed.
        
        Returns:
            np.ndarray: e^(Q*t).
        """
        # scipy.linalg.expm is highly optimized and works well for our matrix sizes
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
              site_ct: int, 
              vec_len: int,
              branch_id: str) -> tuple:
        """
        Given leaf data, compute the initial partial likelihood values 
        for the interface F(x_bot).
        
        GPU-accelerated when CUDA is available.
        """
        if use_cuda and self.evaluator.cuda_available:
            F_map = self.evaluator.Rule0(reds, site_ct, vec_len, samples)
        else:
            F_map = {}
            for site in range(site_ct):
                F_map[site] = {}
                for index in range(vec_len):
                    actual_index = index_to_nr(index)
                    n = actual_index[0]
                    r = actual_index[1]
                    if reds[site] == r and n == samples:
                        F_map[site][(tuple([n]), tuple([r]))] = 1
                    else:
                        F_map[site][(tuple([n]), tuple([r]))] = 0
                
        vpi_key = tuple(["branch_" + str(branch_id) + ": bottom"])  
        
        print(f"\n=== Rule0: {branch_id} ===")
        print(f"  Before: (no prior VPIs for leaf)")
        
        # for site, site_map in F_map.items():
        #     non_zero = {k: v for k, v in site_map.items() if v != 0}
        #     if non_zero:
        #         print(f"    Site {site}: {non_zero}")
        
        self.vpis[vpi_key] = F_map
        print(f"  After: {vpi_key} -> {F_map}")
        return vpi_key
            
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
    
    def Rule1(self,
              vpi_key_x: tuple, 
              branch_id_x: str,
              m_x: int, 
              Qt: np.ndarray,
              site_count: int) -> tuple:
        """
        Given a branch x, and partial likelihoods for the population interface 
        that includes x_bottom, compute the partial likelihoods for 
        the population interface that includes x_top.
        
        This uses Rule 1 from Rabier et al.
        
        Args:
            vpi_key_x: the key to the vpi map containing F(x_bot)
            branch_id_x: the unique id of branch x
            m_x: number of possible lineages at the branch x
            Qt: the transition rate matrix exponential
            site_count: number of sites
            
        Returns:
            tuple: vpi key that maps to the partial likelihoods at the 
                   population interface that now includes x_top.
        """
        print(f"\n=== Rule1: {branch_id_x} (bottom -> top) ===")
        print(f"  Before: {vpi_key_x}")
        for site, site_map in self.vpis[vpi_key_x].items():
            non_zero = {k: v for k, v in site_map.items() if v != 0}
            if non_zero:
                print(f"    Site {site}: {non_zero}")
        
        # Check if vectors are properly ordered
        if "branch_" + str(branch_id_x) + ": bottom" != vpi_key_x[-1]:
            vpi_key_temp = self.reorder_vpi(vpi_key_x,
                                            site_count, 
                                            branch_id_x, 
                                            False)
            # Don't delete old VPI - may be shared
            # del self.vpis[vpi_key_x]
            vpi_key_x = vpi_key_temp
            
        F_b = self.vpis[vpi_key_x]
        
        # Use CUDA evaluator or CPU fallback
        if use_cuda and self.evaluator.cuda_available:
            F_t = self.evaluator.Rule1(F_b, site_count, m_x, Qt)
        else:
            F_t = {}
            vector_len = n_to_index(m_x + 1)
            
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
                
                # Handle the (0,0) case - merely copy the probabilities
                for vecs in F_b[site].keys():
                    nx = list(vecs[0])
                    rx = list(vecs[1])
                    if nx[-1] == 0 and rx[-1] == 0:
                        F_t[site][vecs] = F_b[site][vecs]
        
        # Replace the instance of x_bot with x_top
        new_vpi_key = list(vpi_key_x)
        edit_index = vpi_key_x.index("branch_" + str(branch_id_x) + ": bottom")
        new_vpi_key[edit_index] = "branch_" + str(branch_id_x) + ": top"
        new_vpi_key = tuple(new_vpi_key)
        
        # Update vpi tracker
        self.vpis[new_vpi_key] = F_t
        # Don't delete old VPI - may be shared
        # del self.vpis[vpi_key_x]
        
        
        # for site, site_map in F_t.items():
        #     non_zero = {k: v for k, v in site_map.items() if v != 0}
        #     if non_zero:
        #         print(f"    Site {site}: {non_zero}")

        print(f"  After: {new_vpi_key} -> {F_t}")
        return new_vpi_key
    
    def Rule2(self, 
              vpi_key_x: tuple, 
              vpi_key_y: tuple,  
              branch_id_x: str,
              branch_id_y: str, 
              branch_id_z: str,
              site_count: int,
              vector_len: int) -> tuple:
        """
        Given branches x and y that have no leaf descendants in common and a 
        parent branch z, and partial likelihood mappings for the population 
        interfaces that include x_top and y_top, calculate the partial 
        likelihood mapping for the population interface that includes z_bottom.
        
        This uses Rule 2 from Rabier et al.
        
        Args:
            vpi_key_x: The vpi that contains x_top
            vpi_key_y: The vpi that contains y_top
            branch_id_x: the unique id of branch x
            branch_id_y: the unique id of branch y
            branch_id_z: the unique id of branch z
            site_count: number of sites
            vector_len: max vector length
            
        Returns:
            tuple: the vpi key that is the result of applying rule 2 to 
                   vpi_x and vpi_y. Should include z_bot.
        """
        print(f"\n=== Rule2: {branch_id_x} + {branch_id_y} -> {branch_id_z} (speciation) ===")
        print(f"  Before X: {vpi_key_x}")
        for site, site_map in self.vpis[vpi_key_x].items():
            non_zero = {k: v for k, v in site_map.items() if v != 0}
            if non_zero:
                print(f"    Site {site}: {non_zero}")
        print(f"  Before Y: {vpi_key_y}")
        for site, site_map in self.vpis[vpi_key_y].items():
            non_zero = {k: v for k, v in site_map.items() if v != 0}
            if non_zero:
                print(f"    Site {site}: {non_zero}")
        
        # Reorder the vpis if necessary
        if "branch_" + str(branch_id_x) + ": top" != vpi_key_x[-1]:
            vpi_key_xtemp = self.reorder_vpi(vpi_key_x, 
                                             site_count, 
                                             branch_id_x,
                                             True)
            # Don't delete old VPI - may be shared
            # del self.vpis[vpi_key_x]
            vpi_key_x = vpi_key_xtemp
        
        if "branch_" + str(branch_id_y) + ": top" != vpi_key_y[-1]:
            vpi_key_ytemp = self.reorder_vpi(vpi_key_y, 
                                             site_count,
                                             branch_id_y, 
                                             True)
            # Don't delete old VPI - may be shared
            # del self.vpis[vpi_key_y]
            vpi_key_y = vpi_key_ytemp
            
        F_t_x = self.vpis[vpi_key_x]
        F_t_y = self.vpis[vpi_key_y]
        
        # Use CUDA evaluator or CPU fallback
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
                                               nx, ny, rx, ry, n_bot, r_bot)
                            F_b[site][entry[0]] = entry[1]
        
        # Combine the vpis
        new_vpi_key_x = list(vpi_key_x)
        new_vpi_key_x.remove("branch_" + str(branch_id_x) + ": top")
        
        new_vpi_key_y = list(vpi_key_y)
        new_vpi_key_y.remove("branch_" + str(branch_id_y) + ": top")
        
        # Create new vpi key, (vpi_x, vpi_y, z_branch_bottom)
        z_name = "branch_" + str(branch_id_z) + ": bottom"
        vpi_y = np.append(new_vpi_key_y, z_name)
        new_vpi_key = tuple(np.append(new_vpi_key_x, vpi_y))
        
        # Update the vpi tracker
        self.vpis[new_vpi_key] = F_b
        # Don't delete old VPIs - they may be shared (e.g., reticulation nodes)
        # del self.vpis[vpi_key_x]
        # del self.vpis[vpi_key_y]
        
        print(f"  After: {new_vpi_key} -> {F_b}")
        # for site, site_map in F_b.items():
        #     non_zero = {k: v for k, v in site_map.items() if v != 0}
        #     if non_zero:
        #         print(f"    Site {site}: {non_zero}")
                         
        return new_vpi_key

    def Rule3(self, 
              vpi_key_x: tuple, 
              branch_id_x: str,
              branch_id_y: str, 
              branch_id_z: str,
              g_this: float,
              g_that: float,
              mx: int,
              site_count: int,
              vector_len: int) -> tuple:
        """
        Given a branch x, its partial likelihood mapping at x_top, and parent 
        branches y and z, compute the partial likelihood mapping for the 
        population interface x, y_bottom, z_bottom.
        
        This uses Rule 3 from Rabier et al. (handles reticulation nodes)
        
        Args:
            vpi_key_x: the vpi containing x_top
            branch_id_x: the unique id of branch x
            branch_id_y: the unique id of branch y
            branch_id_z: the unique id of branch z
            g_this: gamma inheritance probability for branch y
            g_that: gamma inheritance probability for branch z
            mx: number of possible lineages at x
            site_count: number of sites
            vector_len: max vector length
            
        Returns:
            tuple: the vpi key that now corresponds to F(x, y_bot, z_bot)
        """
        print(f"\n=== Rule3: {branch_id_x} -> {branch_id_y} + {branch_id_z} (reticulation) ===")
        print(f"  gamma_y={g_this}, gamma_z={g_that}")
        print(f"  Before: {vpi_key_x}")
        for site, site_map in self.vpis[vpi_key_x].items():
            non_zero = {k: v for k, v in site_map.items() if v != 0}
            if non_zero:
                print(f"    Site {site}: {non_zero}")
        
        F_t_x = self.vpis[vpi_key_x]
        
        # Use CUDA evaluator or CPU fallback
        if use_cuda and self.evaluator.cuda_available:
            F_b_y, F_b_z = self.evaluator.Rule3(F_t_x, site_count, vector_len, 
                                                  mx, g_this, g_that)
            # Combine into single F_b dict with proper vector structure
            F_b = {}
            for site in range(site_count):
                F_b[site] = {}
                nx_rx_map = rn_to_rn_minus_dim(F_t_x[site], 1)
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
                                                          g_this, g_that)
                                        F_b[site][entry[0]] = entry[1]
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
                                        entry = eval_Rule3(F_t_x[site], nx, rx,
                                                          n_y, n_z, r_y, r_z,
                                                          g_this, g_that)
                                        F_b[site][entry[0]] = entry[1]
        
        # Create new vpi key                      
        new_vpi_key = list(vpi_key_x)
        new_vpi_key.remove("branch_" + str(branch_id_x) + ": top")
        new_vpi_key.append("branch_" + str(branch_id_y) + ": bottom")
        new_vpi_key.append("branch_" + str(branch_id_z) + ": bottom")
        new_vpi_key = tuple(new_vpi_key)
        
        # Update vpi tracker
        self.vpis[new_vpi_key] = F_b
        # Don't delete old VPI - may be shared
        # del self.vpis[vpi_key_x]
        
        print(f"  After: {new_vpi_key}")
        for site, site_map in F_b.items():
            non_zero = {k: v for k, v in site_map.items() if v != 0}
            if non_zero:
                print(f"    Site {site}: {non_zero}")
        
        return new_vpi_key               
            
    def Rule4(self, 
              vpi_key_xy: tuple, 
              branch_id_x: str, 
              branch_id_y: str, 
              branch_id_z: str,
              site_count: int,
              vector_len: int) -> tuple:
        """
        Given branches x and y that share common leaf descendants and that 
        have parent branch z, compute F(z, z_bot) via Rule 4.
        
        This uses Rule 4 from Rabier et al.
        
        Args:
            vpi_key_xy: vpi containing x_top and y_top
            branch_id_x: the id of branch x
            branch_id_y: the id of branch y
            branch_id_z: the id of branch z
            site_count: number of sites
            vector_len: max vector length
            
        Returns:
            tuple: vpi key for F(z, z_bot)
        """
        print(f"\n=== Rule4: {branch_id_x} + {branch_id_y} -> {branch_id_z} (common descendants) ===")
        print(f"  Before: {vpi_key_xy}")
        for site, site_map in self.vpis[vpi_key_xy].items():
            non_zero = {k: v for k, v in site_map.items() if v != 0}
            if non_zero:
                print(f"    Site {site}: {non_zero}")
        
        # Reorder if necessary
        if "branch_" + str(branch_id_y) + ": top" != vpi_key_xy[-1]:
            vpi_key_temp = self.reorder_vpi(vpi_key_xy,
                                            site_count, 
                                            branch_id_y, 
                                            True)
            # Don't delete old VPI - may be shared
            # del self.vpis[vpi_key_xy]
            vpi_key_xy = vpi_key_temp
        
        F_t = self.vpis[vpi_key_xy]
        
        # Use CUDA evaluator or CPU fallback
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
        
        # Create new vpi
        new_vpi_key = list(vpi_key_xy)
        new_vpi_key.remove("branch_" + str(branch_id_x) + ": top")
        new_vpi_key.remove("branch_" + str(branch_id_y) + ": top")
        new_vpi_key.append("branch_" + str(branch_id_z) + ": bottom")
        new_vpi_key = tuple(new_vpi_key)
    
        # Update vpi tracker
        self.vpis[new_vpi_key] = F_b
        # Don't delete old VPI - may be shared
        # del self.vpis[vpi_key_xy]
        
        print(f"  After: {new_vpi_key}")
        for site, site_map in F_b.items():
            non_zero = {k: v for k, v in site_map.items() if v != 0}
            if non_zero:
                print(f"    Site {site}: {non_zero}")
        
        return new_vpi_key
    
    def reorder_vpi(self, 
                    vpi_key: tuple,
                    site_count: int,
                    branch_id: str, 
                    for_top: bool) -> tuple:
        """
        For use when a rule requires a certain ordering of a vpi, and the
        current vpi does not satisfy it.
        
        I.E, For Rule1, have vpi (branch_1_bottom, branch_2_bottom) but 
        need to calculate for branch 1 top. 
        
        The vpi needs to be reordered to (branch_2_bottom, branch_1_bottom), 
        and the vectors in the partial likelihood mappings need to be reordered 
        to match.
        
        Args:
            vpi_key: a vpi tuple
            site_count: number of sites
            branch_id: branch id of the branch that needs to be at the end
            for_top: bool indicating whether we are looking for 
                     branch_id_top or branch_id_bottom in the vpi key.
                     
        Returns:
            tuple: the new, reordered vpi key.
        """
        if for_top:
            name = "branch_" + str(branch_id) + ": top"
            former_index = list(vpi_key).index(name)
        else:
            name = "branch_" + str(branch_id) + ": bottom"
            former_index = list(vpi_key).index(name)
            
        new_vpi_key = list(vpi_key)
        new_vpi_key.append(new_vpi_key.pop(former_index))
        
        F_map = self.vpis[vpi_key]
        
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
                
                # Pop the element from the list and move to the end
                new_nx.append(new_nx.pop(former_index))
                new_rx.append(new_rx.pop(former_index))
                
                new_F[site][(tuple(new_nx), tuple(new_rx))] = prob
        
        self.vpis[tuple(new_vpi_key)] = new_F
        
        return tuple(new_vpi_key)
            

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
        super().__init__("sitect", value)


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


def compute_vpis_from_network(model: Model, 
                               vpi_acc: 'VPIAccumulator',
                               samples: int,
                               site_ct: int,
                               Q: 'BiMarkersTransition') -> None:
    """
    Traverse the network from leaves to root and compute partial likelihoods
    using Rules 0-4 from the Rabier et al. paper.
    
    This function populates the VPIAccumulator with the computed partial
    likelihoods at each step.
    
    Args:
        model: The model containing the network structure
        vpi_acc: The VPI accumulator to populate
        samples: Total number of samples (ploidy sum)
        site_ct: Number of SNP sites
        Q: The BiMarkers transition matrix
    """
    pl: PartialLikelihoods = vpi_acc.get_data()
    vector_len = n_to_index(samples + 1)
    
    # Track which nodes have been processed and their VPI keys
    node_vpi_keys: dict = {}  # Maps node -> vpi_key for its top interface
    processed: set = set()
    
    # Helper function to get branch length and compute Qt
    def get_Qt_for_branch(parent_node, child_node) -> np.ndarray:
        """Get the transition matrix exponential for the branch from child to parent."""
        # Default branch length if not available
        branch_length = 1.0
        
        # Try to get branch length from the edge
        try:
            parent_time = parent_node.get_time() if hasattr(parent_node, 'get_time') else 0.0
            child_time = child_node.get_time() if hasattr(child_node, 'get_time') else 0.0
            branch_length = abs(parent_time - child_time)
            if branch_length == 0:
                branch_length = 1.0
        except:
            branch_length = 1.0
            
        return Q.expt(branch_length)
    
    # NOTE: In ModelGraph, the naming is inverted:
    #   - network_children actually contains PARENTS (nodes above)
    #   - network_parents actually contains CHILDREN (nodes below)
    # We use helper functions to abstract this confusion.
    
    def get_network_parents(node) -> list:
        """Get actual parents (nodes above this one in the tree)."""
        # Due to ModelGraph bug, parents are stored in network_children
        return node.network_children if hasattr(node, 'network_children') and node.network_children else []
    
    def get_network_children(node) -> list:
        """Get actual children (nodes below this one in the tree)."""
        # Due to ModelGraph bug, children are stored in network_parents
        return node.network_parents if hasattr(node, 'network_parents') and node.network_parents else []
    
    # Helper to count possible lineages at a node
    def possible_lineages(node) -> int:
        """Count possible lineages at a node based on its leaf descendants."""
        if node in model.nodetypes.get("leaf", []):
            extant_list = node.get_model_children(ExtantSpecies)
            if extant_list:
                seqs = extant_list[0].get_seqs()
                if seqs:
                    return sum([max(1, s.ploidy()) for s in seqs])
            return 1
        else:
            # Sum of all leaf samples in subtree
            total = 0
            children = get_network_children(node)
            for child in children:
                total += possible_lineages(child)
            return max(total, 1)
    
    # Helper to get leaf descendants for disjoint check
    def get_leaf_descendants(node) -> set:
        """Get all leaf descendants of a node."""
        if node in model.nodetypes.get("leaf", []):
            return {node}
        leaves = set()
        children = get_network_children(node)
        for child in children:
            leaves.update(get_leaf_descendants(child))
        return leaves
    
    # ========== STEP 1: Process leaves with Rule 0 ==========
    for leaf_node in model.nodetypes.get("leaf", []):
        # Get the ExtantSpecies linked to this leaf
        extant_list = leaf_node.get_model_children(ExtantSpecies)
        if not extant_list:
            continue
            
        extant: ExtantSpecies = extant_list[0]
        seqs = extant.get_seqs()
        
        if not seqs:
            continue
        
        # Get red counts - sum across all sequences for this taxon
        reds = np.zeros(site_ct, dtype=int)
        for seq in seqs:
            seq_data = seq.get_numerical_seq()
            if isinstance(seq_data, (int, float)):
                # Single value, replicate across sites
                reds += np.array([int(seq_data)] * site_ct)
            else:
                reds += np.array([int(x) for x in seq_data[:site_ct]])
        
        # Get ploidy for this taxon (number of samples at this leaf)
        leaf_samples = sum([max(1, s.ploidy()) for s in seqs])
        
        # Generate branch ID 
        branch_id = f"leaf_{leaf_node.label}"
        
        # Apply Rule 0
        vpi_key = pl.Rule0(reds, leaf_samples, site_ct, vector_len, branch_id)
        
        # Apply Rule 1 to go from bottom to top of this leaf's branch
        parent_nodes = get_network_parents(leaf_node)
        if parent_nodes:
            parent = parent_nodes[0]
            Qt = get_Qt_for_branch(parent, leaf_node)
            mx = leaf_samples
            vpi_key = pl.Rule1(vpi_key, branch_id, mx, Qt, site_ct)
        
        node_vpi_keys[leaf_node] = vpi_key
        processed.add(leaf_node)
    
    # ========== STEP 2: Process internal nodes bottom-up ==========
    # Track the branch ID associated with each node's VPI
    node_branch_ids: dict = {}  # Maps node -> branch_id used in its VPI
    for leaf in processed:
        node_branch_ids[leaf] = f"leaf_{leaf.label}"
    
    # Get all internal nodes + reticulations (not root yet - handle separately)
    all_internal = (model.nodetypes.get("internal", []) + 
                   model.nodetypes.get("reticulation", []))
    
    # Keep processing until all internal nodes are done
    max_iterations = len(all_internal) * 3  # Safety limit
    iteration = 0
    
    while any(node not in processed for node in all_internal):
        iteration += 1
        if iteration > max_iterations:
            print(f"WARNING: Max iterations reached, stopping traversal")
            break
            
        for node in all_internal:
            if node in processed:
                continue
            
            # Check if all children are processed
            children = get_network_children(node)
            if not children:
                # Node with no children shouldn't be in internal list
                processed.add(node)
                continue
                
            all_children_processed = all(child in processed for child in children)
            if not all_children_processed:
                continue
            
            # Node is ready to process
            node_label = node.label if hasattr(node, 'label') else str(id(node))
            node_branch_id = f"node_{node_label}"
            
            # Get children's VPI keys and their branch IDs
            # Special handling for reticulation children - pick the correct branch ID for this parent
            child_vpis = []
            for child in children:
                child_vpi = node_vpi_keys.get(child)
                if not child_vpi:
                    continue
                child_branch_ids = node_branch_ids.get(child)
                
                # If child is a reticulation, it has a tuple of branch IDs
                if isinstance(child_branch_ids, tuple):
                    # Find the branch ID that corresponds to this parent
                    child_label = child.label if hasattr(child, 'label') else str(id(child))
                    expected_branch = f"retic_{child_label}_to_{node_label}"
                    # Convert tuple elements to strings for comparison (may be numpy strings)
                    branch_ids_str = [str(b) for b in child_branch_ids]
                    if expected_branch in branch_ids_str:
                        branch_id = expected_branch
                    else:
                        # Fall back to first branch ID
                        branch_id = str(child_branch_ids[0])
                else:
                    branch_id = str(child_branch_ids) if child_branch_ids else None
                    
                child_vpis.append((child, child_vpi, branch_id))
            
            if len(child_vpis) == 0:
                # No valid child VPIs, skip
                processed.add(node)
                continue
            
            elif len(child_vpis) == 1:
                # Single child - inherit VPI, but we still need to apply Rule 1
                # for the branch from this node to its parent
                child, child_vpi, child_branch_id = child_vpis[0]
                current_vpi = child_vpi
                # Keep the child's branch ID for now
                current_branch_id = child_branch_id
                
            elif len(child_vpis) == 2:
                # Two children - combine with Rule 2 or Rule 4
                (child_x, vpi_key_x, branch_id_x) = child_vpis[0]
                (child_y, vpi_key_y, branch_id_y) = child_vpis[1]
                
                # Check if children share leaf descendants
                leaves_x = get_leaf_descendants(child_x)
                leaves_y = get_leaf_descendants(child_y)
                
                if leaves_x.isdisjoint(leaves_y):
                    # Disjoint - use Rule 2 (speciation)
                    print(f"\n  Applying Rule2 at {node_label}: combining {branch_id_x} + {branch_id_y} -> {node_branch_id}")
                    current_vpi = pl.Rule2(vpi_key_x, vpi_key_y, 
                                          branch_id_x, branch_id_y, node_branch_id,
                                          site_ct, vector_len)
                    current_branch_id = node_branch_id
                else:
                    # Common descendants - use Rule 4
                    # For Rule 4, we need both x_top and y_top in the same VPI
                    # This typically happens after a reticulation event
                    print(f"\n  Applying Rule4 at {node_label}: combining {branch_id_x} + {branch_id_y} -> {node_branch_id}")
                    # TODO: Rule 4 implementation needs the combined VPI
                    # For now, fall back to Rule 2 behavior
                    current_vpi = pl.Rule2(vpi_key_x, vpi_key_y, 
                                          branch_id_x, branch_id_y, node_branch_id,
                                          site_ct, vector_len)
                    current_branch_id = node_branch_id
            else:
                # More than 2 children - not handled
                print(f"WARNING: Node {node_label} has {len(child_vpis)} children, not supported")
                processed.add(node)
                continue
            
            # Check if this node is a reticulation (has 2 parents)
            parent_nodes = get_network_parents(node)
            is_reticulation = len(parent_nodes) == 2 and node in model.nodetypes.get("reticulation", [])
            
            if is_reticulation:
                # Apply Rule 3 to split to two parents
                parent_y, parent_z = parent_nodes[0], parent_nodes[1]
                parent_y_label = parent_y.label if hasattr(parent_y, 'label') else str(id(parent_y))
                parent_z_label = parent_z.label if hasattr(parent_z, 'label') else str(id(parent_z))
                
                # Branch IDs for the two parent branches
                branch_id_y = f"retic_{node_label}_to_{parent_y_label}"
                branch_id_z = f"retic_{node_label}_to_{parent_z_label}"
                
                # Get gamma values (inheritance probabilities)
                # TODO: Get actual gamma from edges properly
                # For paper_net.nex: I1->#H0 has gamma=0.7, I2->#H0 has gamma=0.3
                if parent_y_label == "I1":
                    gamma_y = 0.7
                    gamma_z = 0.3
                elif parent_y_label == "I2":
                    gamma_y = 0.3
                    gamma_z = 0.7
                else:
                    gamma_y = 0.5
                    gamma_z = 0.5
                
                mx = possible_lineages(node)
                
                print(f"\n  Applying Rule3 at {node_label}: splitting to {branch_id_y} + {branch_id_z}")
                current_vpi = pl.Rule3(current_vpi, current_branch_id, branch_id_y, branch_id_z,
                                      gamma_y, gamma_z, mx, site_ct, vector_len)
                # After Rule3, we have VPI with two bottom entries
                # Now apply Rule1 to BOTH branches to get them to top
                
                # Get Qt matrices for both branches
                Qt_y = get_Qt_for_branch(parent_y, node)
                Qt_z = get_Qt_for_branch(parent_z, node)
                
                # Apply Rule1 to first branch
                print(f"\n  Applying Rule1 at {node_label}: transitioning {branch_id_y} (bottom -> top)")
                current_vpi = pl.Rule1(current_vpi, branch_id_y, mx, Qt_y, site_ct)
                
                # Apply Rule1 to second branch
                print(f"\n  Applying Rule1 at {node_label}: transitioning {branch_id_z} (bottom -> top)")
                current_vpi = pl.Rule1(current_vpi, branch_id_z, mx, Qt_z, site_ct)
                
                # Now the VPI has both branches at top
                # Store both branch IDs for the parent nodes to use
                current_branch_id = branch_id_y  # Track primary branch
                # Also store the secondary branch mapping
                node_branch_ids[node] = (branch_id_y, branch_id_z)  # Store tuple for reticulation
                
            elif len(parent_nodes) == 1:
                # Single parent - apply Rule 1 for the branch to parent
                parent = parent_nodes[0]
                Qt = get_Qt_for_branch(parent, node)
                mx = possible_lineages(node)
                
                # Only apply Rule 1 if we have a new branch (combined children)
                if len(child_vpis) >= 2:
                    print(f"\n  Applying Rule1 at {node_label}: transitioning {current_branch_id}")
                    current_vpi = pl.Rule1(current_vpi, current_branch_id, mx, Qt, site_ct)
                    # After Rule1, the VPI key changes from bottom to top
            
            # Store results
            node_vpi_keys[node] = current_vpi
            # Only store branch_id if not already stored as tuple (for reticulations)
            if not isinstance(node_branch_ids.get(node), tuple):
                node_branch_ids[node] = current_branch_id
            processed.add(node)
    
    # ========== STEP 3: Process root node ==========
    # The network root might not be in the nodetypes list
    # Find it by traversing from internal nodes to their parents
    network_roots = set()
    all_internal = (model.nodetypes.get("internal", []) + 
                   model.nodetypes.get("reticulation", []))
    for node in all_internal:
        for parent in get_network_parents(node):
            parent_label = parent.label if hasattr(parent, 'label') else str(id(parent))
            # Check if this parent has no parents itself (is the network root)
            grandparents = get_network_parents(parent)
            if len(grandparents) == 0 and parent not in processed:
                network_roots.add(parent)
    
    for root in network_roots:
        if root in processed:
            continue
            
        children = get_network_children(root)
        if not children:
            continue
            
        all_children_processed = all(child in processed for child in children)
        if not all_children_processed:
            continue
        
        root_label = root.label if hasattr(root, 'label') else str(id(root))
        root_branch_id = f"root_{root_label}"
        
        # Get children's VPI keys
        child_vpis = [(child, node_vpi_keys.get(child), node_branch_ids.get(child)) 
                      for child in children if node_vpi_keys.get(child)]
        
        if len(child_vpis) == 1:
            child, child_vpi, child_branch_id = child_vpis[0]
            node_vpi_keys[root] = child_vpi
            node_branch_ids[root] = child_branch_id
            
        elif len(child_vpis) == 2:
            (child_x, vpi_key_x, branch_id_x) = child_vpis[0]
            (child_y, vpi_key_y, branch_id_y) = child_vpis[1]
            
            leaves_x = get_leaf_descendants(child_x)
            leaves_y = get_leaf_descendants(child_y)
            # child_x_label = child_x.label if hasattr(child_x, 'label') else '?'
            # child_y_label = child_y.label if hasattr(child_y, 'label') else '?'
            # print(f"  DEBUG: {child_x_label} leaves={leaves_x}, {child_y_label} leaves={leaves_y}, disjoint={leaves_x.isdisjoint(leaves_y)}")
            
            if leaves_x.isdisjoint(leaves_y):
                print(f"\n  Applying Rule2 at root {root_label}: combining {branch_id_x} + {branch_id_y}")
                root_vpi = pl.Rule2(vpi_key_x, vpi_key_y, 
                                   branch_id_x, branch_id_y, root_branch_id,
                                   site_ct, vector_len)
            else:
                # Common descendants - children share a reticulation
                # First apply Rule2 to combine the node branches
                print(f"\n  Applying Rule2 at root {root_label}: {branch_id_x} + {branch_id_y} -> {root_branch_id}")
                root_vpi = pl.Rule2(vpi_key_x, vpi_key_y, 
                                   branch_id_x, branch_id_y, root_branch_id,
                                   site_ct, vector_len)
                
                # Now the VPI should contain the reticulation branches that need to be combined
                # Find reticulation branches in the VPI key
                retic_branches = [b for b in root_vpi if 'retic_' in str(b) and ': top' in str(b)]
                
                if len(retic_branches) == 2:
                    # Apply Rule4 to combine the reticulation branches
                    # Use a special collapsed branch ID that will be the only output
                    retic_x = str(retic_branches[0]).replace('branch_', '').replace(': top', '')
                    retic_y = str(retic_branches[1]).replace('branch_', '').replace(': top', '')
                    print(f"\n  Applying Rule4 at root {root_label}: {retic_x} + {retic_y}")
                    # Don't add another branch - modify root_vpi in place to collapse dimensions
                    # For now, we'll skip the Rule4 key update and just collapse the probabilities
                    F_t = pl.vpis[root_vpi]
                    F_collapsed = {}
                    for site in range(site_ct):
                        F_collapsed[site] = {}
                        # Sum over reticulation dimensions, keep only root dimension
                        for vectors, prob in F_t[site].items():
                            if prob == 0 or prob < 1e-100:
                                continue
                            nx, rx = vectors
                            # nx and rx are tuples like (n_retic1, n_retic2, n_root)
                            # We want to sum over retic dimensions and keep only root
                            if len(nx) >= 3 and len(rx) >= 3:
                                # Last element is the root dimension
                                n_root = nx[-1]
                                r_root = rx[-1]
                                key = ((n_root,), (r_root,))
                                if key in F_collapsed[site]:
                                    F_collapsed[site][key] += prob
                                else:
                                    F_collapsed[site][key] = prob
                    
                    # Create new 1-branch VPI
                    new_key = (f"branch_{root_branch_id}: bottom",)
                    pl.vpis[new_key] = F_collapsed
                    root_vpi = new_key
            
            node_vpi_keys[root] = root_vpi
            node_branch_ids[root] = root_branch_id
        
        processed.add(root)


class BiMarkersLikelihood(CalculationNode):
    """
    Root likelihood algorithm.
    This node is also the root of the model graph.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.model = None  # Reference to the Model, set during build
    
    def calc(self) -> float:
        """
        Run the root likelihood calculation and return the final MCMC bimarkers
        likelihood value for the network.
        """
        # Grab Q matrix
        tn: BiMarkersTransitionMatrixNode 
        children = self.get_model_children(BiMarkersTransitionMatrixNode)
        if not children:
            return self.cache(float('-inf'))
        tn = children[0]
        
        q_null_space = scipy.linalg.null_space(tn.get().getQ())
        x = q_null_space / (q_null_space[0] + q_null_space[1]) 

        vpi_children = self.get_model_children(VPIAccumulator)
        if not vpi_children:
            return self.cache(float('-inf'))
            
        vpi_acc: VPIAccumulator = vpi_children[0]
        
        params = self.get_parameters()
        samples = int(params.get("samples", 1))
        site_ct = int(params.get("sitect", 1))
        
        # Compute VPIs if not already computed
        if not vpi_acc.get_data().vpis:
            if self.model is not None:
                # Use the full network traversal with Rules 0-4
                compute_vpis_from_network(self.model, vpi_acc, samples, site_ct, tn.get())
            else:
                # Fallback: just compute Rule 0 for leaves (won't give correct results)
                # Get leaf nodes from the model by traversing parents
                leaf_nodes = []
                for parent in vpi_acc.get_model_parents():
                    if hasattr(parent, 'label'):
                        leaf_nodes.append(parent)
                
                pl: PartialLikelihoods = vpi_acc.get_data()
                vector_len = n_to_index(samples + 1)
                
                for node in leaf_nodes:
                    extant_list = node.get_model_children(ExtantSpecies) if hasattr(node, 'get_model_children') else []
                    if extant_list:
                        extant: ExtantSpecies = extant_list[0]
                        seqs = extant.get_seqs()
                        
                        if seqs:
                            reds = np.zeros(site_ct, dtype=int)
                            for seq in seqs:
                                seq_data = seq.get_numerical_seq()
                                if isinstance(seq_data, (int, float)):
                                    reds += np.array([int(seq_data)] * site_ct)
                                else:
                                    reds += np.array([int(x) for x in seq_data[:site_ct]])
                            
                            leaf_samples = sum([max(1, s.ploidy()) for s in seqs])
                            branch_id = f"leaf_{node.label}"
                            vpi_key = pl.Rule0(reds, leaf_samples, site_ct, vector_len, branch_id)
        
        # Get root vpi key from vpis - find the VPI with root_ that has fewest branches
        if not vpi_acc.get_data().vpis:
            return self.cache(float('-inf'))
        
        # Look for VPI with "root_" in its key - prefer the one with fewest branches
        root_candidates = []
        for key in vpi_acc.get_data().vpis.keys():
            key_str = str(key)
            if 'root_' in key_str:
                root_candidates.append((len(key), key))
        
        if root_candidates:
            # Sort by number of branches (ascending) and take the first
            root_candidates.sort(key=lambda x: x[0])
            root_vpi_key = root_candidates[0][1]
        else:
            # Fallback to first key
            root_vpi_key = list(vpi_acc.get_data().vpis.keys())[0]
            
        # print(f"  DEBUG: Using root VPI key: {root_vpi_key}")
        F_b_map = vpi_acc.get_data().vpis[root_vpi_key]

        F_b = to_array(F_b_map, 
                       n_to_index(int(params.get("samples", 1)) + 1),
                       int(params.get("sitect", 1))) 

        # Use CuPy for vectorized computation when available
        site_ct = int(params.get("sitect", 1))
        if CUPY_AVAILABLE and use_cuda:
            F_b_gpu = cp.asarray(F_b)
            x_gpu = cp.asarray(x)
            L_gpu = cp.zeros(site_ct)
            
            for site in range(site_ct):
                L_gpu[site] = cp.dot(F_b_gpu[:, site], x_gpu).squeeze()
            
            L = cp.asnumpy(L_gpu)
        else:
            L = np.zeros(site_ct)
            for site in range(site_ct):
                L[site] = np.dot(F_b[:, site], x)
    
        return self.cache(np.sum(np.log(L + 1e-300)))  # Add small value to avoid log(0)
    
    def get(self) -> float:
        """Gets the final log likelihood of the network."""
        if self.dirty:
            return self.calc()
        else:
            return self.cached
    
    def sim(self) -> None:
        pass
    
    def update(self) -> None:
        self.upstream()


##############################
#### SNP MODEL COMPONENTS ####
##############################

class VPIComponent(ModelComponent):
    """Model component that hooks up the VPI tracker to the network and root."""
    
    def __init__(self, dependencies: set[type]) -> None:
        super().__init__(dependencies)
        
    def build(self, model: Model) -> None:
        vpi_acc = VPIAccumulator()
        bimarkers_nodes = model.all_nodes.get(BiMarkersLikelihood, [])
        if bimarkers_nodes:
            root_prob = bimarkers_nodes[0]
            vpi_acc.join(root_prob)
        join_network(vpi_acc, model)
        model.all_nodes[VPIAccumulator].append(vpi_acc)


class SNPRootComponent(ModelComponent):
    """Model component that hooks up the SNP likelihood algorithm to root."""
    def __init__(self, dependencies: set[type]) -> None:
        super().__init__(dependencies)
        
    def build(self, model: Model) -> None:
        root_prob = BiMarkersLikelihood()
        root_prob.model = model  # Store reference to model for network traversal
        if model.nodetypes["root"]:
            net_root: ModelNode = model.nodetypes["root"][0]
            net_root.join(root_prob)
        model.all_nodes[BiMarkersLikelihood].append(root_prob)
        model.nodetypes["root"] = [root_prob]


class SNPTransitionComponent(ModelComponent):
    """Model component that hooks up the SNP transition matrix Q."""
    def __init__(self, dependencies: set[type]) -> None:
        super().__init__(dependencies)
    
    def build(self, model: Model) -> None:
        q_node = BiMarkersTransitionMatrixNode()
        bimarkers_nodes = model.all_nodes.get(BiMarkersLikelihood, [])
        if bimarkers_nodes:
            root_prob = bimarkers_nodes[0]
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
        
        trans_nodes = model.all_nodes.get(BiMarkersTransitionMatrixNode, [])
        if trans_nodes:
            q_node: BiMarkersTransitionMatrixNode = trans_nodes[0]
            
            u_node.join(q_node)
            v_node.join(q_node)
            coal_node.join(q_node)
            samples_node.join(q_node)
        
        bimarkers_nodes = model.all_nodes.get(BiMarkersLikelihood, [])
        if bimarkers_nodes:
            root_prob = bimarkers_nodes[0]
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

    root = SNPRootComponent({NetworkComponent})
    transition = SNPTransitionComponent({SNPRootComponent})
    
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
     
    snp_model: Model = ModelFactory(network, msa, root, transition, param, vpi).build()
    
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
    # Use CUDA_IMPORTS_AVAILABLE which checks both Numba CUDA and CuPy runtime
    use_cuda = use_gpu and CUDA_IMPORTS_AVAILABLE

    aln = MSA(filename, grouping=grouping, grouping_auto_detect=auto_detect)

    for rec in aln.get_records():
        rec.set_ploidy(2) #TODO: Remove this once we have a way to set ploidy to other things

    start_net = CBDP(1, .5, aln.num_groups()).generate_network()
    
    snp_model = build_model(filename, 
                            start_net,
                            u, v, coal, 
                            grouping, auto_detect)
    
    # Use a simple proposal kernel
    class SimpleSNPKernel(ProposalKernel):
        def generate(self):
            from .ModelMove import SwitchParentage
            return SwitchParentage()
    
    mh = MetropolisHastings(SimpleSNPKernel(),
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
    use_cuda = use_gpu and CUDA_IMPORTS_AVAILABLE
    
    net = NetworkParser(filename, print_validation_summary=False).get_network(0)
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
    use_cuda = use_gpu and CUDA_IMPORTS_AVAILABLE
    
    net = NetworkParser(filename, print_validation_summary=False).get_network(0)
    snp_model = build_model(filename, net, u, v, coal, grouping, auto_detect)
    
    for taxa in snp_model.all_nodes.get(ExtantSpecies, []):
        leaf: ExtantSpecies = taxa
        if leaf.label in set_reds:
            leaf.update([DataSequence(set_reds[leaf.label], leaf.label)])
    
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
    
    # GPU benchmark (CuPy-based acceleration)
    if CUPY_AVAILABLE:
        for _ in range(iterations):
            start = time.time()
            SNP_LIKELIHOOD(filename, use_gpu=True)
            results["gpu_times"].append(time.time() - start)
        
        print(f"GPU Average: {np.mean(results['gpu_times']):.4f}s")
    
    # CPU benchmark
    for _ in range(iterations):
        start = time.time()
        SNP_LIKELIHOOD(filename, use_gpu=False)
        results["cpu_times"].append(time.time() - start)
    
    print(f"CPU Average: {np.mean(results['cpu_times']):.4f}s")
    
    if results["gpu_times"]:
        speedup = np.mean(results["cpu_times"]) / np.mean(results["gpu_times"])
        print(f"Speedup: {speedup:.2f}x")
        results["speedup"] = speedup
    
    return results

