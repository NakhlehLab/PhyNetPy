"""
SNAPPNET Strategy (Vectorized) for PhyNetPy

Fully vectorized implementation of Rabier et al. 2021.
Uses precomputed coefficient tensors to eliminate Python loops in Rules 2-4.

Key optimizations:
  - Rule 2: Precomputed merge tensor M[parent_idx, child_x_idx, child_y_idx]
  - Rule 3: Precomputed split tensor S[child_idx, parent_y_idx, parent_z_idx]  
  - Rule 4: Vectorized stationary distribution weighting
"""

import numpy as np
from math import comb
from scipy.linalg import expm
from functools import lru_cache
from .Strategy import Strategy
from .ModelGraph2 import (
    LeafNode, InternalNode, ReticulationNode, RootNode, RootAggregatorNode
)


def nr_to_index(n: int, r: int) -> int:
    """Map (n,r) to linear index."""
    return int(0.5 * (n - 1) * (n + 2)) + r


def index_to_nr(idx: int) -> tuple[int, int]:
    """Inverse of nr_to_index."""
    n = int(0.5 * (-1 + np.sqrt(1 + 8 * idx)))
    while nr_to_index(n + 1, 0) <= idx:
        n += 1
    return n, idx - nr_to_index(n, 0)


def vector_len(max_n: int) -> int:
    """Length of VPI vector for max_n lineages."""
    return nr_to_index(max_n, max_n) + 1


class CoefficientCache:
    """
    Precomputes and caches coefficient tensors for vectorized operations.
    
    These tensors encode the combinatorial structure of Rules 2 and 3,
    allowing numpy broadcasting to replace nested Python loops.
    """
    
    def __init__(self, max_n: int):
        self.max_n = max_n
        self.dim = vector_len(max_n)
        self._merge_cache = {}
        self._split_cache = {}
        self._stationary = None
    
    def get_merge_tensor(self, mx: int, my: int) -> np.ndarray:
        """
        Build merge tensor for Rule 2.
        
        M[p, x, y] = coefficient for parent state p receiving from 
                     child_x state x and child_y state y.
        
        Shape: [dim_parent, dim_x, dim_y]
        """
        key = (mx, my)
        if key in self._merge_cache:
            return self._merge_cache[key]
        
        dim_x = vector_len(mx)
        dim_y = vector_len(my)
        dim_p = self.dim
        
        M = np.zeros((dim_p, dim_x, dim_y))
        
        for nx in range(1, mx + 1):
            for rx in range(nx + 1):
                idx_x = nr_to_index(nx, rx)
                
                for ny in range(1, my + 1):
                    n_p = nx + ny
                    if n_p > self.max_n:
                        continue
                    
                    for ry in range(ny + 1):
                        idx_y = nr_to_index(ny, ry)
                        r_p = rx + ry
                        idx_p = nr_to_index(n_p, r_p)
                        
                        M[idx_p, idx_x, idx_y] = comb(n_p, nx) * comb(r_p, rx)
        
        self._merge_cache[key] = M
        return M
    
    def get_split_tensor(self, m: int, gamma: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Build split tensors for Rule 3.
        
        S_y[c, y] = coefficient for child state c contributing to parent_y state y
        S_z[c, z] = coefficient for child state c contributing to parent_z state z
        
        Includes the γ^ny * (1-γ)^nz probability weights.
        """
        # Cache key includes gamma (rounded for floating point)
        key = (m, round(gamma, 8))
        if key in self._split_cache:
            return self._split_cache[key]
        
        dim_c = vector_len(m)
        
        # Parent dims can be up to m (when all lineages go one way)
        S_y = np.zeros((dim_c, dim_c))
        S_z = np.zeros((dim_c, dim_c))
        
        for n in range(1, m + 1):
            for r in range(n + 1):
                idx_c = nr_to_index(n, r)
                
                for ny in range(n + 1):
                    nz = n - ny
                    
                    for ry in range(min(r, ny) + 1):
                        rz = r - ry
                        if rz < 0 or rz > nz:
                            continue
                        
                        coef = comb(n, ny) * comb(r, ry)
                        weight = (gamma ** ny) * ((1 - gamma) ** nz)
                        
                        if ny > 0:
                            idx_y = nr_to_index(ny, ry)
                            S_y[idx_c, idx_y] += coef * weight
                        
                        if nz > 0:
                            idx_z = nr_to_index(nz, rz)
                            S_z[idx_c, idx_z] += coef * weight
        
        self._split_cache[key] = (S_y, S_z)
        return S_y, S_z
    
    def get_stationary_weights(self, u: float, v: float) -> np.ndarray:
        """
        Stationary distribution weights for Rule 4.
        
        π[(n,r)] = C(n,r) * θ_r^r * θ_g^(n-r)
        """
        if self._stationary is not None:
            return self._stationary
        
        theta_r = v / (u + v)
        theta_g = u / (u + v)
        
        pi = np.zeros(self.dim)
        for n in range(1, self.max_n + 1):
            for r in range(n + 1):
                idx = nr_to_index(n, r)
                pi[idx] = comb(n, r) * (theta_r ** r) * (theta_g ** (n - r))
        
        self._stationary = pi
        return pi


class QMatrix:
    """Rate matrix Q with caching."""
    
    def __init__(self, max_n: int, u: float, v: float, theta: float):
        self.max_n = max_n
        self.dim = vector_len(max_n)
        self.Q = self._build_Q(u, v, theta)
        self._cache = {}
    
    def _build_Q(self, u: float, v: float, theta: float) -> np.ndarray:
        """Build rate matrix Q (Bryant et al. Eq. 15)."""
        Q = np.zeros((self.dim, self.dim))
        
        for n in range(1, self.max_n + 1):
            for r in range(n + 1):
                idx = nr_to_index(n, r)
                
                coal_rate = n * (n - 1) / theta if theta > 0 else 0
                Q[idx, idx] = -coal_rate - v * (n - r) - u * r
                
                if n < self.max_n:
                    idx_gg = nr_to_index(n + 1, r)
                    Q[idx, idx_gg] = (n + 1 - r) * (n - r) / theta if theta > 0 else 0
                    
                    if r + 1 <= n + 1:
                        idx_rr = nr_to_index(n + 1, r + 1)
                        Q[idx, idx_rr] = (r + 1) * r / theta if theta > 0 else 0
                
                if r > 0:
                    Q[idx, nr_to_index(n, r - 1)] = v * (n - r + 1)
                
                if r < n:
                    Q[idx, nr_to_index(n, r + 1)] = u * (r + 1)
        
        return Q
    
    def expt(self, t: float) -> np.ndarray:
        """Compute e^(Qt), cached."""
        key = round(t, 10)
        if key not in self._cache:
            self._cache[key] = expm(self.Q * t)
        return self._cache[key]


class SNAPPNETStrategy(Strategy):
    """
    Vectorized SNAPPNET algorithm.
    
    VPIs are dense arrays [sites × states]. Operations use:
      - Rule 1: Matrix multiply (already vectorized)
      - Rule 2: einsum with precomputed merge tensor
      - Rule 3: Matrix multiply with precomputed split tensors
      - Rule 4: Dot product with stationary weights
    """
    
    def __init__(self, u: float, v: float, theta: float,
                 sites: int, max_samples: int):
        self.u, self.v = u, v
        self.theta = theta
        self.sites = sites
        self.max_n = max_samples
        self.dim = vector_len(max_samples)
        
        self.Q = QMatrix(max_samples, u, v, theta)
        self.coef = CoefficientCache(max_samples)
        self._pi = self.coef.get_stationary_weights(u, v)
    
    # ─────────────────────────────────────────────────────────────────────
    # Rule 1: Branch propagation (already vectorized)
    # ─────────────────────────────────────────────────────────────────────
    
    def _rule1(self, vpi: np.ndarray, t: float) -> np.ndarray:
        """F' = F @ e^(Qt)  -- fully vectorized."""
        if t <= 0:
            return vpi.copy()
        return vpi @ self.Q.expt(t)
    
    # ─────────────────────────────────────────────────────────────────────
    # Rule 2: Merge (vectorized via einsum)
    # ─────────────────────────────────────────────────────────────────────
    
    def _rule2(self, vpi_x: np.ndarray, vpi_y: np.ndarray,
               mx: int, my: int) -> np.ndarray:
        """
        Vectorized merge using einsum.
        
        F_p[s, p] = Σ_{x,y} M[p, x, y] * F_x[s, x] * F_y[s, y]
        
        einsum: 'pxy, sx, sy -> sp'
        """
        M = self.coef.get_merge_tensor(mx, my)
        
        # Trim VPIs to actual dimensions
        dim_x, dim_y = vector_len(mx), vector_len(my)
        vx = vpi_x[:, :dim_x]
        vy = vpi_y[:, :dim_y]
        
        # einsum computes: result[s,p] = sum over x,y of M[p,x,y] * vx[s,x] * vy[s,y]
        result_small = np.einsum('pxy,sx,sy->sp', M, vx, vy)
        
        # Pad to full dimension if needed
        if result_small.shape[1] < self.dim:
            result = np.zeros((self.sites, self.dim))
            result[:, :result_small.shape[1]] = result_small
            return result
        return result_small
    
    # ─────────────────────────────────────────────────────────────────────
    # Rule 3: Split (vectorized via matrix multiply)
    # ─────────────────────────────────────────────────────────────────────
    
    def _rule3(self, vpi: np.ndarray, gamma: float, 
               m: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized split using matrix multiply.
        
        F_y[s, y] = Σ_c S_y[c, y] * F[s, c]  =>  F_y = F @ S_y.T
        """
        S_y, S_z = self.coef.get_split_tensor(m, gamma)
        
        dim_c = vector_len(m)
        vc = vpi[:, :dim_c]
        
        # Matrix multiply: [sites × child_dim] @ [child_dim × parent_dim]
        vpi_y = vc @ S_y
        vpi_z = vc @ S_z
        
        # Pad to full dimension
        def pad(v):
            if v.shape[1] < self.dim:
                result = np.zeros((self.sites, self.dim))
                result[:, :v.shape[1]] = v
                return result
            return v
        
        return pad(vpi_y), pad(vpi_z)
    
    # ─────────────────────────────────────────────────────────────────────
    # Rule 4: Root likelihood (vectorized dot product)
    # ─────────────────────────────────────────────────────────────────────
    
    def _rule4(self, vpi: np.ndarray) -> np.ndarray:
        """L[s] = Σ_i π[i] * F[s,i]  =>  L = F @ π"""
        return vpi @ self._pi
    
    # ─────────────────────────────────────────────────────────────────────
    # Strategy interface
    # ─────────────────────────────────────────────────────────────────────
    
    def _get_max_lineages(self, node) -> int:
        """Get max possible lineages from this node."""
        if hasattr(node, '_max_lin'):
            return node._max_lin
        if isinstance(node, LeafNode):
            return node.get_samples()
        children = node.get_model_children()
        return sum(self._get_max_lineages(c) for c in children) if children else 1
    
    def compute_at_leaf(self, n: LeafNode) -> None:
        """Initialize + propagate."""
        samples = n.get_samples()
        n._max_lin = samples
        
        reds = n.data[0].get_numerical_seq() if n.data else [0] * self.sites
        
        # One-hot initialization (vectorized)
        vpi = np.zeros((self.sites, self.dim))
        site_indices = np.arange(self.sites)
        state_indices = np.array([nr_to_index(samples, r) for r in reds])
        valid = state_indices < self.dim
        vpi[site_indices[valid], state_indices[valid]] = 1.0
        
        n.vpi = self._rule1(vpi, n.branch().length)
    
    def compute_at_internal(self, n: InternalNode) -> None:
        """Merge children + propagate."""
        children = n.get_model_children()
        n._max_lin = sum(self._get_max_lineages(c) for c in children)
        
        if len(children) == 1:
            merged = children[0].vpi.copy()
        else:
            merged = children[0].vpi
            running_m = self._get_max_lineages(children[0])
            for child in children[1:]:
                cm = self._get_max_lineages(child)
                merged = self._rule2(merged, child.vpi, running_m, cm)
                running_m += cm
        
        n.vpi = self._rule1(merged, n.branch().length)
    
    def compute_at_reticulation(self, n: ReticulationNode) -> None:
        """Split VPI for two parents."""
        children = n.get_model_children()
        child = children[0] if children else None
        
        if child is None:
            n.vpi = np.zeros((self.sites, self.dim))
            n.vpi_to_parent = {}
            return
        
        n._max_lin = self._get_max_lineages(child)
        branch1, branch2 = n.branches()
        gamma = branch1.inheritance_probability
        
        vpi_y, vpi_z = self._rule3(child.vpi, gamma, n._max_lin)
        
        n.vpi_to_parent = {
            branch1.parent_id: self._rule1(vpi_y, branch1.length),
            branch2.parent_id: self._rule1(vpi_z, branch2.length)
        }
        n.vpi = n.vpi_to_parent.get(branch1.parent_id, vpi_y)
    
    def compute_at_root(self, n: RootNode) -> None:
        """Merge children at root."""
        children = n.get_model_children()
        n._max_lin = sum(self._get_max_lineages(c) for c in children)
        
        if len(children) == 0:
            n.vpi = np.zeros((self.sites, self.dim))
        elif len(children) == 1:
            n.vpi = children[0].vpi.copy()
        else:
            n.vpi = children[0].vpi
            running_m = self._get_max_lineages(children[0])
            for child in children[1:]:
                cm = self._get_max_lineages(child)
                n.vpi = self._rule2(n.vpi, child.vpi, running_m, cm)
                running_m += cm
    
    def compute_at_aggregator(self, n: RootAggregatorNode) -> None:
        """Final log-likelihood."""
        children = n.get_model_children()
        if not children:
            n.result = float('-inf')
            return
        
        site_likes = self._rule4(children[0].vpi)
        site_likes = np.clip(site_likes, 1e-300, None)
        n.result = np.sum(np.log(site_likes))