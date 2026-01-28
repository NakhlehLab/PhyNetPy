
"""
Executor Module for PhyNetPy

This module provides the Executor abstraction layer that supplies matrix/vector
computation backends to Strategies. Executors handle:
- Array operations (creation, manipulation, math)
- Matrix exponentials for transition matrices
- Batched/vectorized operations
- Device management (CPU/GPU)

The Strategy remains pure algorithm logic while the Executor handles HOW
computations are performed (and on what hardware).

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      EXECUTOR ARCHITECTURE                      │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   Visitor                                                       │
    │      │                                                          │
    │      │  "compute partials for these nodes"                      │
    │      ▼                                                          │
    │   Strategy                                                      │
    │      │                                                          │
    │      │  "I need P(t), einsum, element-wise product"             │
    │      ▼                                                          │
    │   Executor                                                      │
    │      │                                                          │
    │      ├──► CPUExecutor (NumPy)                                   │
    │      ├──► GPUExecutor (CuPy/JAX)                                │
    │      └──► MultiGPUExecutor (distributed)                        │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

Author: Mark Kessler
Version: 1.0.0
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional, Callable, Any, TypeVar
from enum import Enum, auto
import numpy as np

# Type alias for array types (numpy, cupy, jax arrays)
ArrayType = TypeVar('ArrayType')


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTOR CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class DeviceType(Enum):
    """Supported compute device types."""
    CPU = auto()
    CUDA = auto()      # NVIDIA GPU via CuPy
    JAX_CPU = auto()   # JAX on CPU
    JAX_GPU = auto()   # JAX on GPU
    JAX_TPU = auto()   # JAX on TPU


@dataclass
class ExecutorConfig:
    """
    Configuration for executor initialization.
    
    Attributes:
        device_type: Target compute device
        device_id: GPU device ID (for multi-GPU systems)
        dtype: Default data type for arrays
        enable_jit: Enable JIT compilation (JAX only)
        enable_autodiff: Enable automatic differentiation (JAX only)
        memory_pool_size: GPU memory pool size in bytes (optional)
        seed: Random seed for reproducibility
    """
    device_type: DeviceType = DeviceType.CPU
    device_id: int = 0
    dtype: str = 'float64'
    enable_jit: bool = True
    enable_autodiff: bool = False
    memory_pool_size: Optional[int] = None
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════════════
# ABSTRACT EXECUTOR BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class Executor(ABC):
    """
    Abstract base class for computation executors.
    
    An Executor provides a unified interface for array/matrix operations
    across different backends (NumPy, CuPy, JAX). Strategies use executors
    to perform computations without knowing the underlying implementation.
    
    Design Principles:
    - Executors are stateless computation providers
    - All array operations go through the executor
    - Strategies receive an executor at construction time
    - Executors handle device placement and memory management
    
    Example:
        >>> executor = GPUExecutor(ExecutorConfig(device_type=DeviceType.CUDA))
        >>> strategy = SNPStrategy(u=0.1, v=0.2, executor=executor)
        >>> # Strategy now uses GPU for all computations
    """
    
    def __init__(self, config: ExecutorConfig):
        """
        Initialize executor with configuration.
        
        Args:
            config: ExecutorConfig specifying device and options
        """
        self.config = config
        self._dtype = self._resolve_dtype(config.dtype)
        self._rng = None  # Lazy initialization
    
    # ───────────────────────────────────────────────────────────────────────────
    # Abstract Methods - Must be implemented by subclasses
    # ───────────────────────────────────────────────────────────────────────────
    
    @abstractmethod
    def array(self, data: Any, dtype: Optional[str] = None) -> ArrayType:
        """
        Create an array on the executor's device.
        
        Args:
            data: Input data (list, numpy array, etc.)
            dtype: Optional dtype override
        
        Returns:
            Array on the target device
        """
        pass
    
    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> ArrayType:
        """Create zero-filled array."""
        pass
    
    @abstractmethod
    def ones(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> ArrayType:
        """Create one-filled array."""
        pass
    
    @abstractmethod
    def empty(self, shape: Tuple[int, ...], dtype: Optional[str] = None) -> ArrayType:
        """Create uninitialized array (faster than zeros)."""
        pass
    
    @abstractmethod
    def eye(self, n: int, dtype: Optional[str] = None) -> ArrayType:
        """Create identity matrix."""
        pass
    
    @abstractmethod
    def diag(self, v: ArrayType) -> ArrayType:
        """Create diagonal matrix from vector, or extract diagonal."""
        pass
    
    @abstractmethod
    def arange(self, start: int, stop: int, step: int = 1) -> ArrayType:
        """Create array with evenly spaced values."""
        pass
    
    # ───────────────────────────────────────────────────────────────────────────
    # Mathematical Operations
    # ───────────────────────────────────────────────────────────────────────────
    
    @abstractmethod
    def exp(self, x: ArrayType) -> ArrayType:
        """Element-wise exponential."""
        pass
    
    @abstractmethod
    def log(self, x: ArrayType) -> ArrayType:
        """Element-wise natural logarithm."""
        pass
    
    @abstractmethod
    def sqrt(self, x: ArrayType) -> ArrayType:
        """Element-wise square root."""
        pass
    
    @abstractmethod
    def abs(self, x: ArrayType) -> ArrayType:
        """Element-wise absolute value."""
        pass
    
    @abstractmethod
    def sum(self, x: ArrayType, axis: Optional[int] = None) -> ArrayType:
        """Sum of array elements along axis."""
        pass
    
    @abstractmethod
    def prod(self, x: ArrayType, axis: Optional[int] = None) -> ArrayType:
        """Product of array elements along axis."""
        pass
    
    @abstractmethod
    def max(self, x: ArrayType, axis: Optional[int] = None) -> ArrayType:
        """Maximum along axis."""
        pass
    
    @abstractmethod
    def min(self, x: ArrayType, axis: Optional[int] = None) -> ArrayType:
        """Minimum along axis."""
        pass
    
    @abstractmethod
    def mean(self, x: ArrayType, axis: Optional[int] = None) -> ArrayType:
        """Mean along axis."""
        pass
    
    @abstractmethod
    def clip(self, x: ArrayType, min_val: float, max_val: float) -> ArrayType:
        """Clip values to range."""
        pass
    
    # ───────────────────────────────────────────────────────────────────────────
    # Linear Algebra Operations
    # ───────────────────────────────────────────────────────────────────────────
    
    @abstractmethod
    def matmul(self, a: ArrayType, b: ArrayType) -> ArrayType:
        """Matrix multiplication."""
        pass
    
    @abstractmethod
    def einsum(self, subscripts: str, *operands: ArrayType) -> ArrayType:
        """
        Einstein summation - critical for phylogenetic computations.
        
        Common patterns:
        - 'ij,sj->si': Transform partials through P matrix
        - 'si,i->s': Weight by frequencies
        - 'ij,jk->ik': Matrix multiply
        - 'bij,bsj->bsi': Batched transform
        """
        pass
    
    @abstractmethod
    def outer(self, a: ArrayType, b: ArrayType) -> ArrayType:
        """Outer product of two vectors."""
        pass
    
    @abstractmethod
    def dot(self, a: ArrayType, b: ArrayType) -> ArrayType:
        """Dot product."""
        pass
    
    @abstractmethod
    def transpose(self, x: ArrayType, axes: Optional[Tuple[int, ...]] = None) -> ArrayType:
        """Transpose array."""
        pass
    
    @abstractmethod
    def inv(self, x: ArrayType) -> ArrayType:
        """Matrix inverse."""
        pass
    
    @abstractmethod
    def eig(self, x: ArrayType) -> Tuple[ArrayType, ArrayType]:
        """
        Eigendecomposition.
        
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        pass
    
    @abstractmethod
    def solve(self, a: ArrayType, b: ArrayType) -> ArrayType:
        """Solve linear system Ax = b."""
        pass
    
    # ───────────────────────────────────────────────────────────────────────────
    # Phylogenetics-Specific Operations
    # ───────────────────────────────────────────────────────────────────────────
    
    @abstractmethod
    def expm(self, Q: ArrayType, t: float) -> ArrayType:
        """
        Matrix exponential: P(t) = exp(Qt)
        
        This is THE critical operation for phylogenetic likelihood.
        Implementations should optimize this heavily.
        
        Args:
            Q: Rate matrix (n_states, n_states)
            t: Branch length (time)
        
        Returns:
            Transition probability matrix P(t)
        """
        pass
    
    @abstractmethod
    def batch_expm(self, Q: ArrayType, t_values: ArrayType) -> ArrayType:
        """
        Batched matrix exponential for multiple branch lengths.
        
        Computes P(t) for all t values simultaneously - critical for GPU.
        
        Args:
            Q: Rate matrix (n_states, n_states)
            t_values: Array of branch lengths (n_branches,)
        
        Returns:
            Transition matrices (n_branches, n_states, n_states)
        """
        pass
    
    def pruning_step(self, 
                     child_partials: List[ArrayType],
                     transition_matrices: List[ArrayType]) -> ArrayType:
        """
        Standard Felsenstein pruning step.
        
        Computes: L_parent[s] = ∏_c ( ∑_t P_c[s,t] × L_c[t] )
        
        Default implementation using einsum. Subclasses may override
        for optimized versions.
        
        Args:
            child_partials: List of (n_sites, n_states) arrays
            transition_matrices: List of (n_states, n_states) P matrices
        
        Returns:
            Parent partial likelihoods (n_sites, n_states)
        """
        result = None
        
        for partial, P in zip(child_partials, transition_matrices):
            # P @ partial^T, but vectorized over sites
            transformed = self.einsum('ij,sj->si', P, partial)
            
            if result is None:
                result = transformed
            else:
                result = result * transformed
        
        return result
    
    def batch_pruning_step(self,
                           child_partials: ArrayType,
                           transition_matrices: ArrayType) -> ArrayType:
        """
        Batched pruning step for multiple nodes or trees.
        
        Args:
            child_partials: (n_children, n_sites, n_states)
            transition_matrices: (n_children, n_states, n_states)
        
        Returns:
            Combined partials (n_sites, n_states)
        """
        # Transform all children at once
        # (n_children, n_states, n_states) @ (n_children, n_sites, n_states)
        transformed = self.einsum('cij,csj->csi', transition_matrices, child_partials)
        
        # Product across children
        return self.prod(transformed, axis=0)
    
    def root_likelihood(self, 
                        root_partials: ArrayType, 
                        frequencies: ArrayType) -> float:
        """
        Compute log-likelihood at root.
        
        Args:
            root_partials: (n_sites, n_states)
            frequencies: Equilibrium frequencies (n_states,)
        
        Returns:
            Log-likelihood (scalar)
        """
        site_likes = self.einsum('si,i->s', root_partials, frequencies)
        log_like = self.sum(self.log(site_likes))
        return float(self.to_numpy(log_like))
    
    # ───────────────────────────────────────────────────────────────────────────
    # Array Manipulation
    # ───────────────────────────────────────────────────────────────────────────
    
    @abstractmethod
    def reshape(self, x: ArrayType, shape: Tuple[int, ...]) -> ArrayType:
        """Reshape array."""
        pass
    
    @abstractmethod
    def concatenate(self, arrays: List[ArrayType], axis: int = 0) -> ArrayType:
        """Concatenate arrays along axis."""
        pass
    
    @abstractmethod
    def stack(self, arrays: List[ArrayType], axis: int = 0) -> ArrayType:
        """Stack arrays along new axis."""
        pass
    
    @abstractmethod
    def split(self, x: ArrayType, indices: List[int], axis: int = 0) -> List[ArrayType]:
        """Split array at indices."""
        pass
    
    @abstractmethod
    def squeeze(self, x: ArrayType, axis: Optional[int] = None) -> ArrayType:
        """Remove single-dimensional entries."""
        pass
    
    @abstractmethod
    def expand_dims(self, x: ArrayType, axis: int) -> ArrayType:
        """Add dimension at axis."""
        pass
    
    @abstractmethod
    def broadcast_to(self, x: ArrayType, shape: Tuple[int, ...]) -> ArrayType:
        """Broadcast array to shape."""
        pass
    
    # ───────────────────────────────────────────────────────────────────────────
    # Indexing and Slicing
    # ───────────────────────────────────────────────────────────────────────────
    
    @abstractmethod
    def take(self, x: ArrayType, indices: ArrayType, axis: int = 0) -> ArrayType:
        """Take elements along axis."""
        pass
    
    @abstractmethod
    def where(self, condition: ArrayType, x: ArrayType, y: ArrayType) -> ArrayType:
        """Element-wise conditional selection."""
        pass
    
    # ───────────────────────────────────────────────────────────────────────────
    # Device/Memory Management
    # ───────────────────────────────────────────────────────────────────────────
    
    @abstractmethod
    def to_numpy(self, x: ArrayType) -> np.ndarray:
        """Convert array to numpy (CPU)."""
        pass
    
    @abstractmethod
    def from_numpy(self, x: np.ndarray) -> ArrayType:
        """Convert numpy array to executor's array type."""
        pass
    
    @abstractmethod
    def copy(self, x: ArrayType) -> ArrayType:
        """Create a copy of array."""
        pass
    
    def synchronize(self) -> None:
        """
        Synchronize device (wait for pending operations).
        
        No-op for CPU. GPU implementations should override.
        """
        pass
    
    # ───────────────────────────────────────────────────────────────────────────
    # Random Number Generation
    # ───────────────────────────────────────────────────────────────────────────
    
    @abstractmethod
    def random_uniform(self, shape: Tuple[int, ...], 
                       low: float = 0.0, high: float = 1.0) -> ArrayType:
        """Generate uniform random values."""
        pass
    
    @abstractmethod
    def random_normal(self, shape: Tuple[int, ...],
                      mean: float = 0.0, std: float = 1.0) -> ArrayType:
        """Generate normal random values."""
        pass
    
    @abstractmethod
    def random_choice(self, a: ArrayType, size: int, 
                      p: Optional[ArrayType] = None) -> ArrayType:
        """Random choice from array."""
        pass
    
    # ───────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ───────────────────────────────────────────────────────────────────────────
    
    def _resolve_dtype(self, dtype: str) -> Any:
        """Convert string dtype to backend-specific dtype."""
        return dtype  # Override in subclasses
    
    @property
    def device_name(self) -> str:
        """Human-readable device name."""
        return f"{self.config.device_type.name}:{self.config.device_id}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device_name})"


# ═══════════════════════════════════════════════════════════════════════════════
# CPU EXECUTOR (NumPy Reference Implementation)
# ═══════════════════════════════════════════════════════════════════════════════

class CPUExecutor(Executor):
    """
    CPU executor using NumPy.
    
    This serves as the reference implementation and fallback when
    GPU is not available.
    """
    
    def __init__(self, config: Optional[ExecutorConfig] = None):
        config = config or ExecutorConfig(device_type=DeviceType.CPU)
        super().__init__(config)
        self._rng = np.random.default_rng(config.seed)
    
    # ─────────────────────────────────────────────────────────────────────
    # Array Creation
    # ─────────────────────────────────────────────────────────────────────
    
    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype or self._dtype)
    
    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype or self._dtype)
    
    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype or self._dtype)
    
    def empty(self, shape, dtype=None):
        return np.empty(shape, dtype=dtype or self._dtype)
    
    def eye(self, n, dtype=None):
        return np.eye(n, dtype=dtype or self._dtype)
    
    def diag(self, v):
        return np.diag(v)
    
    def arange(self, start, stop, step=1):
        return np.arange(start, stop, step)
    
    # ─────────────────────────────────────────────────────────────────────
    # Math Operations
    # ─────────────────────────────────────────────────────────────────────
    
    def exp(self, x):
        return np.exp(x)
    
    def log(self, x):
        return np.log(x)
    
    def sqrt(self, x):
        return np.sqrt(x)
    
    def abs(self, x):
        return np.abs(x)
    
    def sum(self, x, axis=None):
        return np.sum(x, axis=axis)
    
    def prod(self, x, axis=None):
        return np.prod(x, axis=axis)
    
    def max(self, x, axis=None):
        return np.max(x, axis=axis)
    
    def min(self, x, axis=None):
        return np.min(x, axis=axis)
    
    def mean(self, x, axis=None):
        return np.mean(x, axis=axis)
    
    def clip(self, x, min_val, max_val):
        return np.clip(x, min_val, max_val)
    
    # ─────────────────────────────────────────────────────────────────────
    # Linear Algebra
    # ─────────────────────────────────────────────────────────────────────
    
    def matmul(self, a, b):
        return np.matmul(a, b)
    
    def einsum(self, subscripts, *operands):
        return np.einsum(subscripts, *operands)
    
    def outer(self, a, b):
        return np.outer(a, b)
    
    def dot(self, a, b):
        return np.dot(a, b)
    
    def transpose(self, x, axes=None):
        return np.transpose(x, axes)
    
    def inv(self, x):
        return np.linalg.inv(x)
    
    def eig(self, x):
        return np.linalg.eig(x)
    
    def solve(self, a, b):
        return np.linalg.solve(a, b)
    
    # ─────────────────────────────────────────────────────────────────────
    # Matrix Exponential
    # ─────────────────────────────────────────────────────────────────────
    
    def expm(self, Q, t):
        """
        Matrix exponential using eigendecomposition.
        
        P(t) = V @ diag(exp(λt)) @ V^{-1}
        """
        eigenvalues, V = np.linalg.eig(Q)
        V_inv = np.linalg.inv(V)
        exp_diag = np.diag(np.exp(eigenvalues.real * t))
        return (V @ exp_diag @ V_inv).real
    
    def batch_expm(self, Q, t_values):
        """Batch matrix exponential for multiple branch lengths."""
        # Eigendecomposition (done once)
        eigenvalues, V = np.linalg.eig(Q)
        V_inv = np.linalg.inv(V)
        
        # Vectorized exponential: (n_branches, n_states)
        exp_eigenvalues = np.exp(
            np.outer(t_values, eigenvalues.real)
        )
        
        # Construct P(t) for all branches
        # Using einsum: V @ diag(exp_eig) @ V_inv for each t
        n_branches = len(t_values)
        n_states = Q.shape[0]
        
        P_all = np.zeros((n_branches, n_states, n_states))
        for i in range(n_branches):
            exp_diag = np.diag(exp_eigenvalues[i])
            P_all[i] = (V @ exp_diag @ V_inv).real
        
        return P_all
    
    # ─────────────────────────────────────────────────────────────────────
    # Array Manipulation
    # ─────────────────────────────────────────────────────────────────────
    
    def reshape(self, x, shape):
        return np.reshape(x, shape)
    
    def concatenate(self, arrays, axis=0):
        return np.concatenate(arrays, axis=axis)
    
    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)
    
    def split(self, x, indices, axis=0):
        return np.split(x, indices, axis=axis)
    
    def squeeze(self, x, axis=None):
        return np.squeeze(x, axis=axis)
    
    def expand_dims(self, x, axis):
        return np.expand_dims(x, axis)
    
    def broadcast_to(self, x, shape):
        return np.broadcast_to(x, shape)
    
    # ─────────────────────────────────────────────────────────────────────
    # Indexing
    # ─────────────────────────────────────────────────────────────────────
    
    def take(self, x, indices, axis=0):
        return np.take(x, indices, axis=axis)
    
    def where(self, condition, x, y):
        return np.where(condition, x, y)
    
    # ─────────────────────────────────────────────────────────────────────
    # Device/Memory
    # ─────────────────────────────────────────────────────────────────────
    
    def to_numpy(self, x):
        return np.asarray(x)
    
    def from_numpy(self, x):
        return x  # Already numpy
    
    def copy(self, x):
        return np.copy(x)
    
    # ─────────────────────────────────────────────────────────────────────
    # Random
    # ─────────────────────────────────────────────────────────────────────
    
    def random_uniform(self, shape, low=0.0, high=1.0):
        return self._rng.uniform(low, high, shape)
    
    def random_normal(self, shape, mean=0.0, std=1.0):
        return self._rng.normal(mean, std, shape)
    
    def random_choice(self, a, size, p=None):
        return self._rng.choice(a, size=size, p=p)


# ═══════════════════════════════════════════════════════════════════════════════
# GPU EXECUTOR (CuPy Implementation)
# ═══════════════════════════════════════════════════════════════════════════════

class GPUExecutor(Executor):
    """
    GPU executor using CuPy (NVIDIA CUDA).
    
    Provides significant speedups for:
    - Large alignment matrices
    - Batched transition matrix exponentials
    - Vectorized partial likelihood updates
    
    Performance Notes:
    - Best for n_sites > 1000 (GPU overhead dominates for small data)
    - Memory transfers are expensive - keep data on GPU
    - Use batch operations whenever possible
    
    Example:
        >>> config = ExecutorConfig(device_type=DeviceType.CUDA, device_id=0)
        >>> executor = GPUExecutor(config)
        >>> strategy = SNPStrategy(u=0.1, v=0.2, executor=executor)
    """
    
    def __init__(self, config: Optional[ExecutorConfig] = None):
        config = config or ExecutorConfig(device_type=DeviceType.CUDA)
        super().__init__(config)
        
        # Import CuPy (fails gracefully if not installed)
        try:
            import cupy as cp
            self.cp = cp
        except ImportError:
            raise ImportError(
                "CuPy is required for GPUExecutor. "
                "Install with: pip install cupy-cuda11x (adjust for your CUDA version)"
            )
        
        # Set device
        self.device = cp.cuda.Device(config.device_id)
        self.device.use()
        
        # Optional: configure memory pool
        if config.memory_pool_size:
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(size=config.memory_pool_size)
        
        # Initialize RNG on GPU
        self._rng = cp.random.default_rng(config.seed)
        
        # Cache for eigendecompositions (avoid recomputation)
        self._eigen_cache = {}
    
    # ─────────────────────────────────────────────────────────────────────
    # Array Creation (on GPU)
    # ─────────────────────────────────────────────────────────────────────
    
    def array(self, data, dtype=None):
        with self.device:
            return self.cp.array(data, dtype=dtype or self._dtype)
    
    def zeros(self, shape, dtype=None):
        with self.device:
            return self.cp.zeros(shape, dtype=dtype or self._dtype)
    
    def ones(self, shape, dtype=None):
        with self.device:
            return self.cp.ones(shape, dtype=dtype or self._dtype)
    
    def empty(self, shape, dtype=None):
        with self.device:
            return self.cp.empty(shape, dtype=dtype or self._dtype)
    
    def eye(self, n, dtype=None):
        with self.device:
            return self.cp.eye(n, dtype=dtype or self._dtype)
    
    def diag(self, v):
        return self.cp.diag(v)
    
    def arange(self, start, stop, step=1):
        with self.device:
            return self.cp.arange(start, stop, step)
    
    # ─────────────────────────────────────────────────────────────────────
    # Math Operations (GPU-accelerated)
    # ─────────────────────────────────────────────────────────────────────
    
    def exp(self, x):
        return self.cp.exp(x)
    
    def log(self, x):
        return self.cp.log(x)
    
    def sqrt(self, x):
        return self.cp.sqrt(x)
    
    def abs(self, x):
        return self.cp.abs(x)
    
    def sum(self, x, axis=None):
        return self.cp.sum(x, axis=axis)
    
    def prod(self, x, axis=None):
        return self.cp.prod(x, axis=axis)
    
    def max(self, x, axis=None):
        return self.cp.max(x, axis=axis)
    
    def min(self, x, axis=None):
        return self.cp.min(x, axis=axis)
    
    def mean(self, x, axis=None):
        return self.cp.mean(x, axis=axis)
    
    def clip(self, x, min_val, max_val):
        return self.cp.clip(x, min_val, max_val)
    
    # ─────────────────────────────────────────────────────────────────────
    # Linear Algebra (cuBLAS/cuSOLVER accelerated)
    # ─────────────────────────────────────────────────────────────────────
    
    def matmul(self, a, b):
        return self.cp.matmul(a, b)
    
    def einsum(self, subscripts, *operands):
        return self.cp.einsum(subscripts, *operands)
    
    def outer(self, a, b):
        return self.cp.outer(a, b)
    
    def dot(self, a, b):
        return self.cp.dot(a, b)
    
    def transpose(self, x, axes=None):
        return self.cp.transpose(x, axes)
    
    def inv(self, x):
        return self.cp.linalg.inv(x)
    
    def eig(self, x):
        # Note: CuPy eig returns (eigenvalues, eigenvectors) like NumPy
        return self.cp.linalg.eig(x)
    
    def solve(self, a, b):
        return self.cp.linalg.solve(a, b)
    
    # ─────────────────────────────────────────────────────────────────────
    # Matrix Exponential (GPU-optimized)
    # ─────────────────────────────────────────────────────────────────────
    
    def expm(self, Q, t):
        """
        GPU-accelerated matrix exponential.
        
        Uses eigendecomposition with caching for repeated Q matrices.
        """
        # Check cache (using id of Q array)
        cache_key = id(Q)
        
        if cache_key not in self._eigen_cache:
            eigenvalues, V = self.eig(Q)
            V_inv = self.inv(V)
            self._eigen_cache[cache_key] = (eigenvalues, V, V_inv)
        
        eigenvalues, V, V_inv = self._eigen_cache[cache_key]
        
        # Compute exp(λt)
        exp_diag = self.diag(self.exp(eigenvalues.real * t))
        
        # P(t) = V @ diag(exp(λt)) @ V^{-1}
        return (self.matmul(self.matmul(V, exp_diag), V_inv)).real
    
    def batch_expm(self, Q, t_values):
        """
        Fully vectorized batch matrix exponential.
        
        This is where GPU really shines - computing many P(t) matrices
        simultaneously.
        
        ┌─────────────────────────────────────────────────────────────┐
        │  BATCH MATRIX EXPONENTIAL (GPU)                             │
        │                                                             │
        │  t_values: [t₁, t₂,..., tₙ]  (n_branches,)                   |
        │                │                                            │
        │                ▼                                            │
        │  exp(λ·t):  [[e^λ₁t₁, e^λ₂t₁, ...],   (n_branches, n_states)│
        │              [e^λ₁t₂, e^λ₂t₂, ...],                         │
        │              ...]                                           │
        │                │                                            │
        │                ▼  (all done in parallel on GPU)             │
        │  P(t) = V @ diag(exp(λt)) @ V⁻¹                             │
        │         (n_branches, n_states, n_states)                    │
        └─────────────────────────────────────────────────────────────┘
        """
        # Get or compute eigendecomposition
        cache_key = id(Q)
        
        if cache_key not in self._eigen_cache:
            eigenvalues, V = self.eig(Q)
            V_inv = self.inv(V)
            self._eigen_cache[cache_key] = (eigenvalues, V, V_inv)
        
        eigenvalues, V, V_inv = self._eigen_cache[cache_key]
        
        n_branches = len(t_values)
        n_states = Q.shape[0]
        
        # Vectorized exponential: (n_branches,) outer (n_states,) -> (n_branches, n_states)
        # exp(λ * t) for all branches and all eigenvalues
        exp_eigenvalues = self.exp(
            self.einsum('b,s->bs', t_values, eigenvalues.real)
        )
        
        # Construct all P(t) matrices using batched einsum
        # P[b] = V @ diag(exp_eigenvalues[b]) @ V_inv
        # 
        # Expanded: P[b,i,k] = sum_j V[i,j] * exp_eigenvalues[b,j] * V_inv[j,k]
        P_all = self.einsum('ij,bj,jk->bik', V, exp_eigenvalues, V_inv)
        
        return P_all.real
    
    # ─────────────────────────────────────────────────────────────────────
    # Phylogenetics Operations (GPU-optimized)
    # ─────────────────────────────────────────────────────────────────────
    
    def pruning_step(self, child_partials, transition_matrices):
        """
        GPU-optimized pruning step.
        
        For GPU efficiency, we stack inputs and use batched operations.
        """
        # Stack for batched processing
        stacked_partials = self.stack(child_partials, axis=0)  # (n_children, n_sites, n_states)
        stacked_P = self.stack(transition_matrices, axis=0)     # (n_children, n_states, n_states)
        
        # Batched transform: all children at once
        # (n_children, n_states, n_states) @ (n_children, n_sites, n_states)
        transformed = self.einsum('cij,csj->csi', stacked_P, stacked_partials)
        
        # Product across children
        return self.prod(transformed, axis=0)
    
    def batch_pruning_step(self, child_partials, transition_matrices):
        """
        Batched pruning across multiple trees.
        
        Args:
            child_partials: (n_trees, n_children, n_sites, n_states)
            transition_matrices: (n_trees, n_children, n_states, n_states)
        
        Returns:
            (n_trees, n_sites, n_states)
        """
        # Transform: batch over trees and children
        transformed = self.einsum('tcij,tcsj->tcsi', transition_matrices, child_partials)
        
        # Product across children (axis 1)
        return self.prod(transformed, axis=1)
    
    def initialize_leaf_partials(self, sequences, n_states):
        """
        GPU-vectorized leaf partial initialization.
        
        Args:
            sequences: (n_leaves, n_sites) integer sequences
            n_states: Number of states (2 for SNP, 4 for DNA)
        
        Returns:
            (n_leaves, n_sites, n_states) one-hot encoded partials
        """
        n_leaves, n_sites = sequences.shape
        
        # Create output array on GPU
        partials = self.zeros((n_leaves, n_sites, n_states))
        
        # Create index arrays for scatter operation
        leaf_idx = self.arange(0, n_leaves)[:, None]  # (n_leaves, 1)
        site_idx = self.arange(0, n_sites)[None, :]   # (1, n_sites)
        
        # Broadcast to full index grids
        leaf_grid = self.broadcast_to(leaf_idx, (n_leaves, n_sites))
        site_grid = self.broadcast_to(site_idx, (n_leaves, n_sites))
        
        # One-hot encoding using advanced indexing
        # partials[leaf, site, sequences[leaf, site]] = 1.0
        flat_leaves = self.reshape(leaf_grid, (-1,))
        flat_sites = self.reshape(site_grid, (-1,))
        flat_states = self.reshape(sequences, (-1,))
        
        # Handle ambiguous states (state >= n_states means all 1s)
        valid_mask = flat_states < n_states
        
        # Set valid states to 1
        # Note: CuPy scatter is done via indexing
        partials[flat_leaves[valid_mask], 
                 flat_sites[valid_mask], 
                 flat_states[valid_mask]] = 1.0
        
        # Set ambiguous states to all 1s
        ambig_leaves = flat_leaves[~valid_mask]
        ambig_sites = flat_sites[~valid_mask]
        for state in range(n_states):
            partials[ambig_leaves, ambig_sites, state] = 1.0
        
        return partials
    
    # ─────────────────────────────────────────────────────────────────────
    # Array Manipulation
    # ─────────────────────────────────────────────────────────────────────
    
    def reshape(self, x, shape):
        return self.cp.reshape(x, shape)
    
    def concatenate(self, arrays, axis=0):
        return self.cp.concatenate(arrays, axis=axis)
    
    def stack(self, arrays, axis=0):
        return self.cp.stack(arrays, axis=axis)
    
    def split(self, x, indices, axis=0):
        return self.cp.split(x, indices, axis=axis)
    
    def squeeze(self, x, axis=None):
        return self.cp.squeeze(x, axis=axis)
    
    def expand_dims(self, x, axis):
        return self.cp.expand_dims(x, axis)
    
    def broadcast_to(self, x, shape):
        return self.cp.broadcast_to(x, shape)
    
    # ─────────────────────────────────────────────────────────────────────
    # Indexing
    # ─────────────────────────────────────────────────────────────────────
    
    def take(self, x, indices, axis=0):
        return self.cp.take(x, indices, axis=axis)
    
    def where(self, condition, x, y):
        return self.cp.where(condition, x, y)
    
    # ─────────────────────────────────────────────────────────────────────
    # Device/Memory Management
    # ─────────────────────────────────────────────────────────────────────
    
    def to_numpy(self, x):
        """Transfer array from GPU to CPU."""
        return self.cp.asnumpy(x)
    
    def from_numpy(self, x):
        """Transfer array from CPU to GPU."""
        with self.device:
            return self.cp.asarray(x)
    
    def copy(self, x):
        return self.cp.copy(x)
    
    def synchronize(self):
        """Wait for all pending GPU operations to complete."""
        self.cp.cuda.Stream.null.synchronize()
    
    def clear_cache(self):
        """Clear eigendecomposition cache and free GPU memory."""
        self._eigen_cache.clear()
        mempool = self.cp.get_default_memory_pool()
        mempool.free_all_blocks()
    
    # ─────────────────────────────────────────────────────────────────────
    # Random Number Generation (GPU)
    # ─────────────────────────────────────────────────────────────────────
    
    def random_uniform(self, shape, low=0.0, high=1.0):
        with self.device:
            return self._rng.uniform(low, high, shape)
    
    def random_normal(self, shape, mean=0.0, std=1.0):
        with self.device:
            return self._rng.normal(mean, std, shape)
    
    def random_choice(self, a, size, p=None):
        with self.device:
            return self._rng.choice(a, size=size, p=p)


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTOR FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_executor(device: str = 'cpu', **kwargs) -> Executor:
    """
    Factory function to create an executor.
    
    Args:
        device: One of 'cpu', 'cuda', 'gpu', 'jax'
        **kwargs: Additional config options
    
    Returns:
        Appropriate Executor instance
    
    Example:
        >>> executor = create_executor('cuda', device_id=0)
        >>> executor = create_executor('cpu')
    """
    device_map = {
        'cpu': (CPUExecutor, DeviceType.CPU),
        'cuda': (GPUExecutor, DeviceType.CUDA),
        'gpu': (GPUExecutor, DeviceType.CUDA),
    }
    
    if device.lower() not in device_map:
        raise ValueError(f"Unknown device: {device}. Choose from {list(device_map.keys())}")
    
    executor_class, device_type = device_map[device.lower()]
    config = ExecutorConfig(device_type=device_type, **kwargs)
    
    return executor_class(config)


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example: Using executors with a strategy
    
    # Create CPU executor (always available)
    cpu_exec = create_executor('cpu')
    
    # Create test data
    Q = cpu_exec.array([[-0.3, 0.3], [0.2, -0.2]])  # SNP rate matrix
    t_values = cpu_exec.array([0.1, 0.2, 0.5, 1.0])
    
    # Batch matrix exponential
    P_matrices = cpu_exec.batch_expm(Q, t_values)
    print(f"Computed {len(t_values)} transition matrices")
    print(f"P(0.1) =\n{P_matrices[0]}")
    
    # Try GPU if available
    try:
        gpu_exec = create_executor('cuda', device_id=0)
        
        # Move data to GPU
        Q_gpu = gpu_exec.from_numpy(cpu_exec.to_numpy(Q))
        t_gpu = gpu_exec.from_numpy(cpu_exec.to_numpy(t_values))
        
        # Batch computation on GPU
        P_gpu = gpu_exec.batch_expm(Q_gpu, t_gpu)
        gpu_exec.synchronize()
        
        print(f"\nGPU computation successful!")
        print(f"P(0.1) on GPU =\n{gpu_exec.to_numpy(P_gpu[0])}")
        
    except ImportError as e:
        print(f"\nGPU not available: {e}")
