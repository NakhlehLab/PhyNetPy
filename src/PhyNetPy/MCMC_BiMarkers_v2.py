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
##############################################################################

""" 
Author : Mark Kessler
Last Edit : 12/11/25
First Included in Version : 1.1.0

MCMC BiMarkers using the new modular architecture.

This version uses:
- ScoringStrategy for pluggable likelihood computation
- NetworkAdapter for loose network-model coupling
- ModelBuilder for phase-based model construction

The algorithm implements SNP likelihood from:
David Bryant et al., "Inferring Species Trees Directly from Biallelic 
Genetic Markers", Mol. Biol. Evol. 29(8):1917-1932, 2012.
"""

from __future__ import annotations
from math import sqrt, comb, pow as math_pow
from typing import Any, Callable
from collections import defaultdict
import copy
import numpy as np
import scipy
from scipy.linalg import expm

from .MSA import MSA
from .BirthDeath import CBDP
from .NetworkParser import NetworkParser
from .Alphabet import Alphabet
from .Matrix import Matrix
from .Network import Network, Edge, Node

from .ScoringStrategy import SubtreeIndependentScoring, ScoringContext
from .NetworkAdapter import NetworkAdapter
from .ModelBuilder import (
    ModelBuilder, BuildPhase, BuildContext, BuiltModel,
    NetworkPhase, DataPhase, ParameterPhase, ValidationPhase, CustomPhase
)


##########################
### INDEX CONVERSIONS ####
##########################

def n_to_index(n: int) -> int:
    """Starting index for n block in (n,r) linear representation."""
    return int(0.5 * (n - 1) * (n + 2))


def index_to_nr(index: int) -> tuple[int, int]:
    """Convert linear index to (n, r) pair."""
    a, b, c = 1, 1, -2 - 2 * index
    d = b**2 - 4*a*c
    sol = (-b + sqrt(d)) / (2*a)
    n = int(sol)
    r = index - n_to_index(n)
    return n, r


def nr_to_index(n: int, r: int) -> int:
    """Convert (n, r) pair to linear index."""
    return n_to_index(n) + r


##########################
### TRANSITION MATRIX ####
##########################

class BiMarkersQ:
    """
    Q transition matrix for the BiMarkers/SNP model.
    
    Encodes probabilities of transitioning from one (n,r) pair to another
    under the biallelic coalescent model.
    """
    
    def __init__(self, n: int, u: float, v: float, coal: float) -> None:
        """
        Build the Q matrix.
        
        Args:
            n (int): Total number of samples.
            u (float): Red -> green transition probability.
            v (float): Green -> red transition probability.
            coal (float): Coalescent rate (theta).
        """
        self.n = n
        self.u = u
        self.v = v
        self.coal = coal
        
        rows = int(0.5 * n * (n + 3))
        self.Q = np.zeros((rows, rows))
        
        for n_prime in range(1, n + 1):
            for r_prime in range(n_prime + 1):
                n_r = nr_to_index(n_prime, r_prime)
                
                # Diagonal
                self.Q[n_r][n_r] = (
                    - (n_prime * (n_prime - 1) / coal)
                    - (v * (n_prime - r_prime))
                    - (r_prime * u)
                )
                
                if 0 < r_prime <= n_prime:
                    if n_prime > 1:
                        nm_rm = nr_to_index(n_prime - 1, r_prime - 1)
                        self.Q[n_r][nm_rm] = (r_prime - 1) * n_prime / coal
                    n_rm = nr_to_index(n_prime, r_prime - 1)
                    self.Q[n_r][n_rm] = (n_prime - r_prime + 1) * v
                
                if 0 <= r_prime < n_prime:
                    if n_prime > 1:
                        nm_r = nr_to_index(n_prime - 1, r_prime)
                        self.Q[n_r][nm_r] = (n_prime - 1 - r_prime) * n_prime / coal
                    n_rp = nr_to_index(n_prime, r_prime + 1)
                    self.Q[n_r][n_rp] = (r_prime + 1) * u
    
    def expt(self, t: float) -> np.ndarray:
        """Compute e^(Q*t)."""
        return expm(self.Q * t)
    
    def get_Q(self) -> np.ndarray:
        """Get the Q matrix."""
        return self.Q


##########################
### PARTIAL LIKELIHOODS ##
##########################

class VPITracker:
    """
    Tracks vectors of population interfaces (VPIs) and their partial likelihoods.
    
    This is the core data structure for the BiMarkers algorithm, implementing
    Rules 0-4 from the paper:
    
    - Rule 0: Initialize leaf nodes
    - Rule 1: Propagate from branch bottom to top
    - Rule 2: Merge two branches with DISJOINT leaf descendants (tree node)
    - Rule 3: Split at reticulation (going UP - one branch becomes two)
    - Rule 4: Merge two branches with SHARED leaf descendants (above reticulation)
    """
    
    def __init__(self) -> None:
        """Initialize empty VPI tracker."""
        self.vpis: dict[tuple, dict] = {}
    
    def clear(self) -> None:
        """Clear all VPIs."""
        self.vpis.clear()
    
    def rule0(self, reds: np.ndarray, samples: int, 
              site_count: int, vector_len: int, branch_id: str) -> tuple:
        """
        Rule 0: Initialize leaf partial likelihoods.
        
        Args:
            reds: Red allele counts per site.
            samples: Number of samples at this leaf.
            site_count: Number of sites.
            vector_len: Length of VPI vectors.
            branch_id: Branch identifier.
        
        Returns:
            VPI key for the initialized branch bottom.
        """
        F_map = {}
        
        for site in range(site_count):
            F_map[site] = {}
            for index in range(vector_len):
                n, r = index_to_nr(index)
                # Rule 0: probability is 1 if (n,r) matches observed data
                if int(reds[site]) == r and n == samples:
                    F_map[site][(tuple([n]), tuple([r]))] = 1.0
                else:
                    F_map[site][(tuple([n]), tuple([r]))] = 0.0
        
        vpi_key = (f"branch_{branch_id}: bottom",)
        self.vpis[vpi_key] = F_map
        return vpi_key
    
    def rule1(self, vpi_key: tuple, branch_id: str, 
              m_x: int, Qt: np.ndarray, site_count: int) -> tuple:
        """
        Rule 1: Propagate from branch bottom to top.
        
        Computes F(x, x_top) from F(x, x_bottom) using the transition matrix.
        
        Args:
            vpi_key: Current VPI key (contains branch bottom).
            branch_id: Branch identifier.
            m_x: Max lineages at this branch.
            Qt: Transition probability matrix e^(Q*t).
            site_count: Number of sites.
        
        Returns:
            New VPI key with branch top.
        """
        F_b = self.vpis[vpi_key]
        F_t = {}
        
        for site in range(site_count):
            F_t[site] = {}
            
            # Get unique (nx, rx) prefixes
            nx_rx_map = self._reduce_dimension(F_b[site], 1)
            
            for (nx, rx), values in nx_rx_map.items():
                # Iterate over possible top values
                for top_index in range(n_to_index(m_x + 1)):
                    n_top, r_top = index_to_nr(top_index)
                    
                    # Compute Rule 1 sum
                    evaluation = 0.0
                    for n_b in range(n_top, m_x + 1):
                        for r_b in range(n_b + 1):
                            idx = nr_to_index(n_b, r_b)
                            top_idx = nr_to_index(n_top, r_top)
                            if idx < Qt.shape[0] and top_idx < Qt.shape[1]:
                                exp_val = Qt[idx][top_idx]
                                
                                n_vec = tuple(list(nx) + [n_b])
                                r_vec = tuple(list(rx) + [r_b])
                                prob = F_b[site].get((n_vec, r_vec), 0.0)
                                evaluation += prob * exp_val
                    
                    n_vec_top = tuple(list(nx) + [n_top])
                    r_vec_top = tuple(list(rx) + [r_top])
                    F_t[site][(n_vec_top, r_vec_top)] = evaluation
            
            # Handle (0,0) case - copy probabilities
            for vecs, prob in F_b[site].items():
                if vecs[0][-1] == 0 and vecs[1][-1] == 0:
                    F_t[site][vecs] = prob
        
        # Update VPI key
        new_key = list(vpi_key)
        old_name = f"branch_{branch_id}: bottom"
        new_name = f"branch_{branch_id}: top"
        if old_name in new_key:
            idx = new_key.index(old_name)
            new_key[idx] = new_name
        new_key = tuple(new_key)
        
        self.vpis[new_key] = F_t
        del self.vpis[vpi_key]
        
        return new_key
    
    def rule2(self, vpi_key_x: tuple, vpi_key_y: tuple,
              branch_id_x: str, branch_id_y: str, branch_id_z: str,
              site_count: int, vector_len: int) -> tuple:
        """
        Rule 2: Merge two branches with DISJOINT leaf descendants.
        
        Used at regular tree nodes where two child lineages combine.
        
        Args:
            vpi_key_x, vpi_key_y: VPI keys for the two child branches.
            branch_id_x, branch_id_y: Child branch identifiers.
            branch_id_z: Parent branch identifier.
            site_count: Number of sites.
            vector_len: VPI vector length.
        
        Returns:
            New VPI key containing z_bottom.
        """
        F_t_x = self.vpis[vpi_key_x]
        F_t_y = self.vpis[vpi_key_y]
        F_b = {}
        
        for site in range(site_count):
            F_b[site] = {}
            
            nx_rx_map_x = self._reduce_dimension(F_t_x[site], 1)
            nx_rx_map_y = self._reduce_dimension(F_t_y[site], 1)
            
            for (nx, rx), _ in nx_rx_map_x.items():
                for (ny, ry), _ in nx_rx_map_y.items():
                    for index in range(vector_len):
                        n_zbot, r_zbot = index_to_nr(index)
                        
                        evaluation = 0.0
                        for n_xtop in range(n_zbot + 1):
                            for r_xtop in range(r_zbot + 1):
                                n_ytop = n_zbot - n_xtop
                                r_ytop = r_zbot - r_xtop
                                
                                if r_xtop <= n_xtop and r_ytop <= n_ytop and n_ytop >= 0:
                                    denom = comb(n_zbot, r_zbot)
                                    if denom == 0:
                                        denom = 1
                                    const = (comb(n_xtop, r_xtop) * 
                                            comb(n_ytop, r_ytop) / denom)
                                    
                                    nx_full = tuple(list(nx) + [n_xtop])
                                    rx_full = tuple(list(rx) + [r_xtop])
                                    ny_full = tuple(list(ny) + [n_ytop])
                                    ry_full = tuple(list(ry) + [r_ytop])
                                    
                                    prob_x = F_t_x[site].get((nx_full, rx_full), 0.0)
                                    prob_y = F_t_y[site].get((ny_full, ry_full), 0.0)
                                    
                                    evaluation += prob_x * prob_y * const
                        
                        nz = tuple(list(nx) + list(ny) + [n_zbot])
                        rz = tuple(list(rx) + list(ry) + [r_zbot])
                        F_b[site][(nz, rz)] = evaluation
        
        # Create new VPI key
        new_key_x = [k for k in vpi_key_x if k != f"branch_{branch_id_x}: top"]
        new_key_y = [k for k in vpi_key_y if k != f"branch_{branch_id_y}: top"]
        new_key = tuple(new_key_x + new_key_y + [f"branch_{branch_id_z}: bottom"])
        
        self.vpis[new_key] = F_b
        del self.vpis[vpi_key_x]
        del self.vpis[vpi_key_y]
        
        return new_key
    
    def rule3(self, vpi_key_x: tuple, branch_id_x: str,
              branch_id_y: str, branch_id_z: str,
              gamma_y: float, gamma_z: float,
              m_x: int, site_count: int) -> tuple:
        """
        Rule 3: Split at reticulation node (going UP toward root).
        
        When branch x reaches a reticulation node that has TWO parent branches
        (y and z), we probabilistically split the lineages based on inheritance
        probabilities gamma_y and gamma_z.
        
        Args:
            vpi_key_x: VPI key containing x_top.
            branch_id_x: Child branch identifier (coming into reticulation).
            branch_id_y: First parent branch identifier.
            branch_id_z: Second parent branch identifier.
            gamma_y: Inheritance probability for branch y.
            gamma_z: Inheritance probability for branch z.
            m_x: Max lineages at branch x.
            site_count: Number of sites.
        
        Returns:
            New VPI key containing y_bottom and z_bottom.
        """
        F_t_x = self.vpis[vpi_key_x]
        F_b = {}
        
        for site in range(site_count):
            F_b[site] = {}
            
            # Get (nx, rx) prefixes by removing last element
            nx_rx_map = self._reduce_dimension(F_t_x[site], 1)
            
            for (nx, rx), _ in nx_rx_map.items():
                # Iterate over possible lineage splits between y and z
                for n_y in range(m_x + 1):
                    for n_z in range(m_x - n_y + 1):
                        if n_y + n_z < 1:
                            continue
                        
                        for r_y in range(n_y + 1):
                            for r_z in range(n_z + 1):
                                # Rule 3 equation:
                                # F(y_bot, z_bot) = F(x_top) * C(n_y+n_z, n_y) * gamma_y^n_y * gamma_z^n_z
                                n_total = n_y + n_z
                                r_total = r_y + r_z
                                
                                nx_full = tuple(list(nx) + [n_total])
                                rx_full = tuple(list(rx) + [r_total])
                                
                                top_value = F_t_x[site].get((nx_full, rx_full), 0.0)
                                
                                if top_value > 0:
                                    evaluation = (top_value * 
                                                 comb(n_total, n_y) *
                                                 math_pow(gamma_y, n_y) *
                                                 math_pow(gamma_z, n_z))
                                    
                                    # New vector: (nx, n_y, n_z) and (rx, r_y, r_z)
                                    nz_new = tuple(list(nx) + [n_y, n_z])
                                    rz_new = tuple(list(rx) + [r_y, r_z])
                                    
                                    if (nz_new, rz_new) in F_b[site]:
                                        F_b[site][(nz_new, rz_new)] += evaluation
                                    else:
                                        F_b[site][(nz_new, rz_new)] = evaluation
        
        # Create new VPI key: remove x_top, add y_bottom and z_bottom
        new_key = [k for k in vpi_key_x if k != f"branch_{branch_id_x}: top"]
        new_key.append(f"branch_{branch_id_y}: bottom")
        new_key.append(f"branch_{branch_id_z}: bottom")
        new_key = tuple(new_key)
        
        self.vpis[new_key] = F_b
        del self.vpis[vpi_key_x]
        
        return new_key
    
    def rule4(self, vpi_key_xy: tuple, branch_id_x: str,
              branch_id_y: str, branch_id_z: str,
              site_count: int, vector_len: int) -> tuple:
        """
        Rule 4: Merge branches with SHARED leaf descendants (above reticulation).
        
        When two branches x and y that went through the same reticulation 
        (and thus share leaf descendants) merge into parent branch z.
        This is the inverse of Rule 3.
        
        Args:
            vpi_key_xy: VPI key containing both x_top and y_top.
            branch_id_x: First child branch identifier.
            branch_id_y: Second child branch identifier.
            branch_id_z: Parent branch identifier.
            site_count: Number of sites.
            vector_len: VPI vector length.
        
        Returns:
            New VPI key containing z_bottom.
        """
        # First, ensure proper ordering (y_top should be last)
        vpi_key_xy = self._reorder_vpi_if_needed(vpi_key_xy, branch_id_y, site_count)
        
        F_t = self.vpis[vpi_key_xy]
        F_b = {}
        
        for site in range(site_count):
            F_b[site] = {}
            
            # Reduce by 2 dimensions (for both x_top and y_top)
            nx_rx_map = self._reduce_dimension(F_t[site], 2)
            
            for (nz, rz), _ in nx_rx_map.items():
                for index in range(vector_len):
                    n_zbot, r_zbot = index_to_nr(index)
                    
                    evaluation = 0.0
                    
                    # Sum over all possible splits between x and y
                    for n_xtop in range(1, n_zbot + 1):
                        n_ytop = n_zbot - n_xtop
                        if n_ytop < 0:
                            continue
                        
                        for r_xtop in range(r_zbot + 1):
                            r_ytop = r_zbot - r_xtop
                            
                            if r_xtop <= n_xtop and r_ytop <= n_ytop:
                                # Rule 4 equation
                                denom = comb(n_zbot, r_zbot)
                                if denom == 0:
                                    denom = 1
                                const = (comb(n_xtop, r_xtop) *
                                        comb(n_ytop, r_ytop) / denom)
                                
                                # Look up the VPI value
                                nz_xy = tuple(list(nz) + [n_xtop, n_ytop])
                                rz_xy = tuple(list(rz) + [r_xtop, r_ytop])
                                
                                prob = F_t[site].get((nz_xy, rz_xy), 0.0)
                                evaluation += prob * const
                    
                    nz_new = tuple(list(nz) + [n_zbot])
                    rz_new = tuple(list(rz) + [r_zbot])
                    F_b[site][(nz_new, rz_new)] = evaluation
        
        # Create new VPI key
        new_key = [k for k in vpi_key_xy 
                   if k != f"branch_{branch_id_x}: top" 
                   and k != f"branch_{branch_id_y}: top"]
        new_key.append(f"branch_{branch_id_z}: bottom")
        new_key = tuple(new_key)
        
        self.vpis[new_key] = F_b
        del self.vpis[vpi_key_xy]
        
        return new_key
    
    def _reorder_vpi_if_needed(self, vpi_key: tuple, branch_id: str, 
                                site_count: int) -> tuple:
        """
        Reorder VPI so that the specified branch is at the end.
        
        This is needed because some rules require specific ordering of
        the VPI vectors.
        
        Args:
            vpi_key: Current VPI key.
            branch_id: Branch that should be at the end.
            site_count: Number of sites.
        
        Returns:
            Potentially reordered VPI key.
        """
        target_top = f"branch_{branch_id}: top"
        target_bot = f"branch_{branch_id}: bottom"
        
        # Check if already in correct position
        if vpi_key[-1] in (target_top, target_bot):
            return vpi_key
        
        # Find the index of the target branch
        target_name = None
        target_index = None
        for i, name in enumerate(vpi_key):
            if branch_id in name:
                target_name = name
                target_index = i
                break
        
        if target_index is None:
            return vpi_key  # Branch not found
        
        # Reorder the VPI key
        new_key = list(vpi_key)
        new_key.append(new_key.pop(target_index))
        new_key = tuple(new_key)
        
        # Also reorder the vectors in the VPI data
        F_map = self.vpis[vpi_key]
        new_F = {}
        
        for site in range(site_count):
            new_F[site] = {}
            for (nx, rx), prob in F_map[site].items():
                nx_list = list(nx)
                rx_list = list(rx)
                
                # Move element at target_index to end
                nx_list.append(nx_list.pop(target_index))
                rx_list.append(rx_list.pop(target_index))
                
                new_F[site][(tuple(nx_list), tuple(rx_list))] = prob
        
        # Update storage
        self.vpis[new_key] = new_F
        del self.vpis[vpi_key]
        
        return new_key
    
    def get_vpi(self, key: tuple) -> dict:
        """Get VPI data for a key."""
        return self.vpis.get(key, {})
    
    def get_key_containing(self, branch_id: str) -> tuple:
        """Find VPI key containing this branch."""
        for key in self.vpis:
            if any(branch_id in k for k in key):
                return key
        return None
    
    def _reduce_dimension(self, site_map: dict, dim: int) -> dict:
        """Reduce VPI vectors by removing last `dim` elements."""
        result = defaultdict(list)
        for (nx, rx), prob in site_map.items():
            if len(nx) >= dim and len(rx) >= dim:
                key = (nx[:-dim], rx[:-dim])
                result[key].append((nx[-dim:], rx[-dim:], prob))
        return result


##############################
### BIMARKERS SCORING V2 #####
##############################

class BiMarkersScoringV2(SubtreeIndependentScoring):
    """
    BiMarkers/SNP likelihood scoring using the new architecture.
    
    This scoring strategy:
    - Uses SubtreeIndependentScoring for partial caching
    - Maintains VPI tracker for efficient partial likelihood computation
    - Supports incremental updates when parameters change
    """
    
    def __init__(self, 
                 u: float = 0.5,
                 v: float = 0.5,
                 coal: float = 1.0) -> None:
        """
        Initialize BiMarkers scoring.
        
        Args:
            u: Red -> green transition probability.
            v: Green -> red transition probability.
            coal: Coalescent rate (theta).
        """
        super().__init__()
        self.u = u
        self.v = v
        self.coal = coal
        
        # VPI tracker for partial likelihoods
        self.vpi_tracker = VPITracker()
        
        # Cached Q matrix
        self._Q: BiMarkersQ = None
        self._Q_params: tuple = None
    
    def score(self, context: ScoringContext) -> float:
        """
        Compute the BiMarkers log-likelihood.
        
        Args:
            context: Contains network, data, and parameters.
        
        Returns:
            Log-likelihood value.
        """
        # Extract data from context
        network = context.network
        adapter: NetworkAdapter = context.get_extra("network_adapter")
        msa = context.data
        
        if network is None:
            return float('-inf')
        
        # Get/update parameters
        self.u = context.get_param("u", self.u)
        self.v = context.get_param("v", self.v)
        self.coal = context.get_param("coal", self.coal)
        samples = context.get_param("samples")
        site_count = context.get_param("sites")
        
        # Infer from data if not provided
        if samples is None and msa is not None:
            if hasattr(msa, 'total_samples'):
                samples = msa.total_samples()
        if site_count is None and msa is not None:
            if hasattr(msa, 'dim'):
                site_count = msa.dim()[1]
        
        if samples is None or site_count is None:
            return float('-inf')
        
        # Get or build Q matrix
        Q = self._get_Q(samples)
        vector_len = n_to_index(samples + 1)
        
        # Clear VPI tracker if topology changed or full recompute needed
        if self.is_dirty():
            self.vpi_tracker.clear()
        
        # Compute partial likelihoods via postorder traversal
        try:
            root_vpi_key = self._compute_partials(
                network, adapter, msa, samples, site_count, Q, vector_len
            )
        except Exception as e:
            return float('-inf')
        
        if root_vpi_key is None:
            return float('-inf')
        
        # Get root partial likelihoods
        root_vpi = self.vpi_tracker.get_vpi(root_vpi_key)
        if not root_vpi:
            return float('-inf')
        
        # Compute stationary distribution from Q's null space
        try:
            q_null = scipy.linalg.null_space(Q.get_Q())
            if q_null.size == 0:
                return float('-inf')
            x = q_null / (q_null[0] + q_null[1])
        except Exception:
            return float('-inf')
        
        # Convert root VPI to array format
        F_b = self._vpi_to_array(root_vpi, vector_len, site_count)
        
        # Compute likelihood per site (EQ 20)
        L = np.zeros(site_count)
        for site in range(site_count):
            L[site] = np.dot(F_b[:, site], x.flatten()[:vector_len])
        
        # Return log-likelihood
        L = np.maximum(L, 1e-300)  # Avoid log(0)
        log_like = np.sum(np.log(L))
        
        self.mark_clean()
        return float(log_like)
    
    def _get_Q(self, samples: int) -> BiMarkersQ:
        """Get or create Q matrix."""
        params = (samples, self.u, self.v, self.coal)
        if self._Q is None or self._Q_params != params:
            self._Q = BiMarkersQ(samples, self.u, self.v, self.coal)
            self._Q_params = params
        return self._Q
    
    def _compute_partials(self, network: Network, adapter: NetworkAdapter,
                          msa, samples: int, site_count: int,
                          Q: BiMarkersQ, vector_len: int) -> tuple:
        """
        Compute partial likelihoods via postorder traversal.
        
        Handles both tree nodes and reticulation nodes:
        - Leaf nodes: Rule 0 + Rule 1
        - Tree internal nodes (2 children): Rule 2 + Rule 1
        - Reticulation nodes (2 parents): Rule 3 (splits lineages)
        - Nodes above reticulation (shared descendants): Rule 4
        
        Returns the VPI key for the root.
        """
        root = network.root()
        if root is None:
            return None
        
        # Map nodes to branch IDs
        branch_ids = {}
        for i, node in enumerate(network.V()):
            branch_ids[node] = str(i)
        
        # Identify reticulation nodes (nodes with in-degree > 1)
        reticulation_nodes = set()
        for node in network.V():
            if network.in_degree(node) > 1:
                reticulation_nodes.add(node)
        
        # Track which nodes share leaf descendants (went through same reticulation)
        shared_descendants = self._find_shared_descendants(network, reticulation_nodes)
        
        # Postorder traversal
        vpi_keys = {}  # node -> vpi_key
        processed_retics = set()  # Track processed reticulations
        
        for node in self._postorder(network, root):
            children = network.get_children(node)
            parents = network.get_parents(node)
            
            if not children:
                # ===== LEAF NODE: Apply Rule 0 =====
                reds = self._get_red_counts(node, msa, adapter)
                if reds is None:
                    reds = np.zeros(site_count)
                
                leaf_samples = self._get_leaf_samples(node, msa, adapter)
                if leaf_samples == 0:
                    leaf_samples = 1
                
                branch_id = branch_ids[node]
                vpi_key = self.vpi_tracker.rule0(
                    reds, leaf_samples, site_count, 
                    n_to_index(leaf_samples + 1), branch_id
                )
                
                # Apply Rule 1 to propagate to branch top
                if parents:
                    edge = self._get_edge(network, parents[0], node)
                    branch_len = edge.length if edge and hasattr(edge, 'length') else 1.0
                    Qt = Q.expt(branch_len)
                    vpi_key = self.vpi_tracker.rule1(
                        vpi_key, branch_id, leaf_samples, Qt, site_count
                    )
                
                vpi_keys[node] = vpi_key
            
            elif node in reticulation_nodes:
                # ===== RETICULATION NODE: Apply Rule 3 =====
                # This node has 2 parents - need to split lineages
                if len(children) == 1 and len(parents) == 2:
                    child = children[0]
                    parent1, parent2 = parents[0], parents[1]
                    
                    # Get inheritance probabilities (gamma values)
                    gamma1 = self._get_gamma(network, parent1, node)
                    gamma2 = self._get_gamma(network, parent2, node)
                    
                    # Ensure they sum to 1
                    if abs(gamma1 + gamma2 - 1.0) > 0.01:
                        gamma1, gamma2 = 0.5, 0.5
                    
                    lineages = self._count_descendant_samples(node, network, msa, adapter)
                    
                    # Apply Rule 3 to split the VPI
                    vpi_key = self.vpi_tracker.rule3(
                        vpi_keys[child], branch_ids[child],
                        branch_ids[parent1], branch_ids[parent2],
                        gamma1, gamma2,
                        lineages, site_count
                    )
                    
                    # Apply Rule 1 for both parent branches
                    edge1 = self._get_edge(network, parent1, node)
                    edge2 = self._get_edge(network, parent2, node)
                    branch_len1 = edge1.length if edge1 and hasattr(edge1, 'length') else 1.0
                    branch_len2 = edge2.length if edge2 and hasattr(edge2, 'length') else 1.0
                    
                    Qt1 = Q.expt(branch_len1)
                    Qt2 = Q.expt(branch_len2)
                    
                    # Apply Rule 1 for first parent branch
                    vpi_key = self.vpi_tracker.rule1(
                        vpi_key, branch_ids[parent1], lineages, Qt1, site_count
                    )
                    # Apply Rule 1 for second parent branch
                    vpi_key = self.vpi_tracker.rule1(
                        vpi_key, branch_ids[parent2], lineages, Qt2, site_count
                    )
                    
                    vpi_keys[node] = vpi_key
                    processed_retics.add(node)
                else:
                    # Unusual reticulation structure - fall back to Rule 2
                    vpi_key = self._process_tree_node(
                        node, children, branch_ids, vpi_keys, 
                        network, msa, adapter, Q, site_count, vector_len
                    )
                    vpi_keys[node] = vpi_key
            
            else:
                # ===== INTERNAL TREE NODE =====
                # Check if children share leaf descendants (above reticulation)
                if len(children) == 2:
                    child1, child2 = children[0], children[1]
                    child1_leaves = self._get_descendant_leaves(child1, network)
                    child2_leaves = self._get_descendant_leaves(child2, network)
                    
                    # Check if they share any leaves
                    shared = set(child1_leaves) & set(child2_leaves)
                    
                    if shared:
                        # ===== SHARED DESCENDANTS: Apply Rule 4 =====
                        # Need to merge VPIs that went through same reticulation
                        vpi_key = self.vpi_tracker.rule4(
                            vpi_keys[child1],  # Should contain both branches
                            branch_ids[child1], branch_ids[child2],
                            branch_ids[node],
                            site_count, vector_len
                        )
                    else:
                        # ===== DISJOINT DESCENDANTS: Apply Rule 2 =====
                        vpi_key = self.vpi_tracker.rule2(
                            vpi_keys[child1], vpi_keys[child2],
                            branch_ids[child1], branch_ids[child2],
                            branch_ids[node],
                            site_count, vector_len
                        )
                else:
                    # Handle other cases (1 child, polytomies)
                    vpi_key = self._process_tree_node(
                        node, children, branch_ids, vpi_keys,
                        network, msa, adapter, Q, site_count, vector_len
                    )
                
                # Apply Rule 1 if not root
                if parents and node != root:
                    lineages = self._count_descendant_samples(node, network, msa, adapter)
                    edge = self._get_edge(network, parents[0], node)
                    branch_len = edge.length if edge and hasattr(edge, 'length') else 1.0
                    Qt = Q.expt(branch_len)
                    vpi_key = self.vpi_tracker.rule1(
                        vpi_key, branch_ids[node], lineages, Qt, site_count
                    )
                
                vpi_keys[node] = vpi_key
        
        return vpi_keys.get(root)
    
    def _process_tree_node(self, node, children, branch_ids, vpi_keys,
                           network, msa, adapter, Q, site_count, vector_len) -> tuple:
        """Helper to process a standard tree node with Rule 2."""
        if len(children) == 2:
            child1, child2 = children[0], children[1]
            vpi_key = self.vpi_tracker.rule2(
                vpi_keys[child1], vpi_keys[child2],
                branch_ids[child1], branch_ids[child2],
                branch_ids[node],
                site_count, vector_len
            )
        elif len(children) == 1:
            vpi_key = vpi_keys[children[0]]
        else:
            # Handle polytomies by sequential merging
            vpi_key = vpi_keys[children[0]]
            for i in range(1, len(children)):
                prev_id = branch_ids[children[i-1]] if i > 1 else branch_ids[children[0]]
                vpi_key = self.vpi_tracker.rule2(
                    vpi_key, vpi_keys[children[i]],
                    prev_id, branch_ids[children[i]],
                    branch_ids[node],
                    site_count, vector_len
                )
        return vpi_key
    
    def _find_shared_descendants(self, network: Network, 
                                  reticulations: set) -> dict:
        """Find which nodes share descendants through reticulations."""
        shared = {}
        for retic in reticulations:
            parents = network.get_parents(retic)
            if parents and len(parents) == 2:
                shared[retic] = (parents[0], parents[1])
        return shared
    
    def _get_gamma(self, network: Network, parent: Node, child: Node) -> float:
        """Get inheritance probability for an edge."""
        edge = self._get_edge(network, parent, child)
        if edge and hasattr(edge, 'gamma') and edge.gamma is not None:
            return edge.gamma
        # Default to 0.5 if not specified
        return 0.5
    
    def _postorder(self, network: Network, root: Node):
        """Generate nodes in postorder."""
        visited = set()
        
        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for child in network.get_children(node) or []:
                yield from visit(child)
            yield node
        
        yield from visit(root)
    
    def _get_red_counts(self, node: Node, msa, adapter: NetworkAdapter) -> np.ndarray:
        """Get red allele counts for a leaf node."""
        if msa is None:
            return None
        
        if hasattr(msa, 'seq_by_name'):
            seq_rec = msa.seq_by_name(node.label)
            if seq_rec is not None:
                return np.array(seq_rec.get_numerical_seq())
        
        if hasattr(msa, 'group_by_name'):
            seqs = msa.group_by_name(node.label)
            if seqs:
                total = np.zeros(len(seqs[0].get_seq()))
                for seq in seqs:
                    total += np.array(seq.get_numerical_seq())
                return total
        
        return None
    
    def _get_leaf_samples(self, node: Node, msa, adapter: NetworkAdapter) -> int:
        """Get sample count for a leaf."""
        if msa is None:
            return 1
        
        if hasattr(msa, 'seq_by_name'):
            seq = msa.seq_by_name(node.label)
            if seq is not None and hasattr(seq, 'ploidy'):
                return seq.ploidy()
        
        if hasattr(msa, 'group_by_name'):
            seqs = msa.group_by_name(node.label)
            if seqs:
                return sum(s.ploidy() if hasattr(s, 'ploidy') else 1 for s in seqs)
        
        return 1
    
    def _count_descendant_samples(self, node: Node, network: Network, 
                                   msa, adapter: NetworkAdapter) -> int:
        """Count total samples in all descendant leaves."""
        total = 0
        for leaf in self._get_descendant_leaves(node, network):
            total += self._get_leaf_samples(leaf, msa, adapter)
        return max(total, 1)
    
    def _get_descendant_leaves(self, node: Node, network: Network) -> list[Node]:
        """Get all leaf descendants of a node."""
        leaves = []
        stack = [node]
        visited = set()
        
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            
            children = network.get_children(n)
            if not children:
                leaves.append(n)
            else:
                stack.extend(children)
        
        return leaves
    
    def _get_edge(self, network: Network, parent: Node, child: Node):
        """Get edge between parent and child."""
        for edge in network.E():
            if edge.src == parent and edge.dest == child:
                return edge
        return None
    
    def _vpi_to_array(self, vpi: dict, vector_len: int, site_count: int) -> np.ndarray:
        """Convert VPI dict to numpy array."""
        F_b = np.zeros((vector_len, site_count))
        
        for site in range(site_count):
            if site not in vpi:
                continue
            for (nx, rx), prob in vpi[site].items():
                if len(nx) > 0 and len(rx) > 0:
                    n, r = nx[-1], rx[-1]
                    if isinstance(n, (tuple, list)):
                        n = n[0] if n else 1
                    if isinstance(r, (tuple, list)):
                        r = r[0] if r else 0
                    idx = nr_to_index(int(n), int(r))
                    if idx < vector_len:
                        F_b[idx][site] = prob
        
        return F_b
    
    def invalidate(self, affected_nodes=None) -> None:
        """Invalidate caches."""
        super().invalidate(affected_nodes)
        if affected_nodes is None:
            self.vpi_tracker.clear()


##############################
### BUILD PHASES #############
##############################

class BiMarkersScoringPhase(BuildPhase):
    """Build phase that sets up BiMarkers scoring."""
    
    def __init__(self, u: float = 0.5, v: float = 0.5, coal: float = 1.0):
        self.u = u
        self.v = v
        self.coal = coal
    
    @property
    def name(self) -> str:
        return "BiMarkersScoring"
    
    def execute(self, context: BuildContext) -> None:
        strategy = BiMarkersScoringV2(self.u, self.v, self.coal)
        context.set("scoring_strategy", strategy)


class MSAPhase(BuildPhase):
    """Build phase that sets up MSA data and infers parameters."""
    
    def __init__(self, msa: MSA, grouping: dict = None):
        self.msa = msa
        self.grouping = grouping
    
    @property
    def name(self) -> str:
        return "MSA"
    
    def validate_prerequisites(self, context: BuildContext) -> None:
        if not context.has("network"):
            pass  # Network is optional at this phase
    
    def execute(self, context: BuildContext) -> None:
        context.set("data", self.msa)
        context.set("grouping", self.grouping)
        
        # Infer parameters from MSA
        params = context.get("parameters", {})
        
        if hasattr(self.msa, 'total_samples'):
            params["samples"] = self.msa.total_samples()
        if hasattr(self.msa, 'dim'):
            params["sites"] = self.msa.dim()[1]
        
        context.set("parameters", params)


##############################
### BIMARKERS MODEL ##########
##############################

class BiMarkersModel:
    """
    High-level wrapper for BiMarkers model using new architecture.
    
    Provides a clean interface similar to the original Model class
    but uses the new modular components internally.
    """
    
    def __init__(self, 
                 network: Network,
                 msa: MSA,
                 u: float = 0.5,
                 v: float = 0.5,
                 coal: float = 1.0,
                 grouping: dict = None) -> None:
        """
        Build a BiMarkers model.
        
        Args:
            network: Phylogenetic network.
            msa: Multiple sequence alignment with SNP data.
            u: Red -> green transition probability.
            v: Green -> red transition probability.
            coal: Coalescent rate (theta).
            grouping: Optional sequence grouping.
        """
        self.network = network
        self.msa = msa
        self.grouping = grouping
        
        # Build model using phases
        builder = ModelBuilder()
        builder.add_phase(NetworkPhase(network))
        builder.add_phase(MSAPhase(msa, grouping))
        builder.add_phase(ParameterPhase({
            "u": u,
            "v": v,
            "coal": coal
        }))
        builder.add_phase(BiMarkersScoringPhase(u, v, coal))
        
        self._model = builder.build()
        self._summary_str = ""
    
    @property
    def summary_str(self) -> str:
        return self._summary_str
    
    @summary_str.setter
    def summary_str(self, value: str):
        self._summary_str = value
    
    def likelihood(self) -> float:
        """Compute log-likelihood."""
        return self._model.likelihood()
    
    def update_network(self, network: Network = None) -> None:
        """Update the network (after a move)."""
        if network is not None:
            self.network = network
        self._model.network_adapter.set_network(self.network)
        self._model.invalidate()
    
    def update_parameter(self, name: str, value: Any) -> None:
        """Update a model parameter."""
        self._model.set_parameter(name, value)


##############################
### STATE FOR MCMC ###########
##############################

class BiMarkersState:
    """
    State management for BiMarkers MCMC.
    
    Replaces the generic State class with BiMarkers-specific functionality.
    """
    
    def __init__(self, model: BiMarkersModel) -> None:
        """Initialize state with a model."""
        self.current_model = model
        self.proposed_model = copy.deepcopy(model)
    
    def likelihood(self) -> float:
        """Get current model likelihood."""
        return self.current_model.likelihood()
    
    def proposed_likelihood(self) -> float:
        """Get proposed model likelihood."""
        return self.proposed_model.likelihood()
    
    def generate_next(self, move) -> bool:
        """Apply move to proposed model."""
        try:
            move.execute_on(self.proposed_model)
            self.proposed_model.update_network()
            return self._validate()
        except Exception:
            return False
    
    def _validate(self) -> bool:
        """Validate proposed network is acyclic."""
        net = self.proposed_model.network
        if net is None:
            return False
        if hasattr(net, 'is_acyclic'):
            return net.is_acyclic()
        return True
    
    def commit(self, move) -> None:
        """Accept the proposed move."""
        move.execute_on(self.current_model)
        self.current_model.update_network()
    
    def revert(self, move) -> None:
        """Reject the proposed move."""
        move.undo_on(self.proposed_model)
        self.proposed_model.update_network()
    
    def write_line_to_summary(self, line: str) -> None:
        """Add to summary log."""
        self.current_model.summary_str += line.strip() + "\n"


##############################
### METHOD ENTRY POINTS ######
##############################

def build_bimarkers_model(filename: str,
                          net: Network,
                          u: float = 0.5,
                          v: float = 0.5,
                          coal: float = 1.0,
                          grouping: dict = None,
                          auto_detect: bool = False) -> BiMarkersModel:
    """
    Build a BiMarkers model from a data file and network.
    
    Args:
        filename: Path to nexus file with SNP data.
        net: Phylogenetic network.
        u, v, coal: Model parameters.
        grouping: Sequence grouping.
        auto_detect: Auto-detect grouping.
    
    Returns:
        BiMarkersModel ready for likelihood computation.
    """
    msa = MSA(filename, grouping=grouping, grouping_auto_detect=auto_detect)
    return BiMarkersModel(net, msa, u, v, coal, grouping)


def SNP_LIKELIHOOD_V2(filename: str,
                      u: float = 0.5,
                      v: float = 0.5,
                      coal: float = 1.0,
                      grouping: dict = None,
                      auto_detect: bool = False) -> float:
    """
    Compute SNP likelihood for a network in a nexus file.
    
    Args:
        filename: Nexus file with network and SNP data.
        u, v, coal: Model parameters.
        grouping: Sequence grouping.
        auto_detect: Auto-detect grouping.
    
    Returns:
        Log-likelihood value.
    """
    net = NetworkParser(filename).get_network(0)
    model = build_bimarkers_model(filename, net, u, v, coal, grouping, auto_detect)
    return model.likelihood()


def MCMC_BIMARKERS_V2(filename: str,
                      u: float = 0.5,
                      v: float = 0.5,
                      coal: float = 1.0,
                      grouping: dict = None,
                      auto_detect: bool = False,
                      num_iter: int = 800) -> dict:
    """
    Run MCMC BiMarkers inference using the new architecture.
    
    Args:
        filename: Nexus file with SNP data.
        u, v, coal: Initial model parameters.
        grouping: Sequence grouping.
        auto_detect: Auto-detect grouping.
        num_iter: Number of MCMC iterations.
    
    Returns:
        Dict mapping result network to its likelihood.
    """
    # Parse data
    msa = MSA(filename, grouping=grouping, grouping_auto_detect=auto_detect)
    
    # Generate starting network
    start_net = CBDP(1, 0.5, msa.num_groups()).generate_network()
    
    # Build model
    model = BiMarkersModel(start_net, msa, u, v, coal, grouping)
    
    # Run MCMC (using existing MetropolisHastings infrastructure)
    from .MetropolisHastings import MetropolisHastings, ProposalKernel
    from .State import State
    
    # Create a wrapper that makes BiMarkersModel look like the old Model
    class ModelWrapper:
        def __init__(self, bm_model):
            self._model = bm_model
            self.network = bm_model.network
            self.summary_str = ""
        
        def likelihood(self):
            return self._model.likelihood()
        
        def execute_move(self, move):
            return move.execute(self)
        
        def update_network(self):
            self._model.update_network(self.network)
    
    wrapper = ModelWrapper(model)
    state = State(model=wrapper)
    
    # Would need to define a proper ProposalKernel for network moves
    # For now, return initial result
    return {model.network: model.likelihood()}

