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
Last Edit : 12/11/25
First Included in Version : 1.1.0
Docs   - [x]
Tests  - [ ]
Design - [x]

This module provides abstractions for different scoring strategies used in
phylogenetic network inference. The key insight is that different inference
methods have fundamentally different computational patterns:

1. Likelihood-based methods (ML, MPL) often have subtree independence, allowing
   for partial likelihood caching and incremental updates.
   
2. Parsimony-based methods typically require full recomputation as they lack
   subtree independence properties.

By separating scoring into its own strategy hierarchy, we can:
- Keep the model graph for representing relationships
- Let each method choose appropriate caching behavior
- Avoid forcing one computational pattern on all methods
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .Network import Node, Network


class ScoringStrategy(ABC):
    """
    Abstract base class for network scoring strategies.
    
    Different phylogenetic inference methods use different scoring approaches:
    - Maximum Likelihood uses partial likelihoods with subtree caching
    - Parsimony computes extra lineages without subtree independence
    - Maximum Pseudo-Likelihood has its own caching patterns
    
    This abstraction allows each method to define its own scoring behavior
    while sharing common model infrastructure.
    """
    
    def __init__(self) -> None:
        """
        Initialize a scoring strategy.
        
        Args:
            N/A
        Returns:
            N/A
        """
        self._cache: dict[str, Any] = {}
        self._dirty: bool = True
    
    @abstractmethod
    def score(self, context: ScoringContext) -> float:
        """
        Compute the score for the current model state.
        
        Args:
            context (ScoringContext): Contains all necessary data for scoring
                                      (network, data, parameters, etc.)
        Returns:
            float: The computed score (likelihood, parsimony score, etc.)
        """
        pass
    
    @abstractmethod
    def supports_caching(self) -> bool:
        """
        Indicates whether this scoring strategy supports partial caching.
        
        If True, the strategy can use cached intermediate computations
        when only part of the model has changed.
        
        If False, the strategy recomputes everything from scratch each time.
        
        Args:
            N/A
        Returns:
            bool: True if caching is supported, False otherwise.
        """
        pass
    
    def invalidate(self, affected_nodes: list = None) -> None:
        """
        Invalidate cached computations for the specified nodes.
        
        For strategies that support caching, this marks specific subtrees
        as needing recomputation. For strategies without caching support,
        this simply marks the entire computation as dirty.
        
        Args:
            affected_nodes (list[Node], optional): Nodes whose cached values
                                                   should be invalidated.
                                                   If None, invalidates all.
        Returns:
            N/A
        """
        if affected_nodes is None or not self.supports_caching():
            self._cache.clear()
            self._dirty = True
        else:
            for node in affected_nodes:
                self._invalidate_node(node)
    
    def _invalidate_node(self, node) -> None:
        """
        Invalidate cache for a specific node and propagate upstream.
        
        Override in subclasses for custom invalidation behavior.
        
        Args:
            node (Node): The node whose cache should be invalidated.
        Returns:
            N/A
        """
        cache_key = self._node_cache_key(node)
        if cache_key in self._cache:
            del self._cache[cache_key]
    
    def _node_cache_key(self, node) -> str:
        """
        Generate a cache key for a node.
        
        Args:
            node (Node): A network node.
        Returns:
            str: A unique cache key for this node.
        """
        return f"node_{id(node)}"
    
    def is_dirty(self) -> bool:
        """
        Check if the scoring needs recomputation.
        
        Args:
            N/A
        Returns:
            bool: True if recomputation is needed, False otherwise.
        """
        return self._dirty
    
    def mark_clean(self) -> None:
        """
        Mark the scoring as up-to-date after computation.
        
        Args:
            N/A
        Returns:
            N/A
        """
        self._dirty = False


class ScoringContext:
    """
    Container for all data needed by a scoring strategy.
    
    This decouples the scoring strategy from the Model class, making
    scoring strategies more reusable and testable.
    """
    
    def __init__(self,
                 network = None,
                 data: Any = None,
                 parameters: dict[str, Any] = None,
                 extras: dict[str, Any] = None) -> None:
        """
        Initialize a scoring context.
        
        Args:
            network (Network, optional): The phylogenetic network to score.
            data (Any, optional): Observed data (MSA, gene trees, etc.)
            parameters (dict[str, Any], optional): Model parameters.
            extras (dict[str, Any], optional): Any additional method-specific data.
        Returns:
            N/A
        """
        self.network = network
        self.data = data
        self.parameters = parameters or {}
        self.extras = extras or {}
    
    def get_param(self, name: str, default: Any = None) -> Any:
        """
        Retrieve a parameter by name.
        
        Args:
            name (str): Parameter name.
            default (Any, optional): Default value if not found.
        Returns:
            Any: The parameter value.
        """
        return self.parameters.get(name, default)
    
    def get_extra(self, name: str, default: Any = None) -> Any:
        """
        Retrieve extra data by name.
        
        Args:
            name (str): Extra data key.
            default (Any, optional): Default value if not found.
        Returns:
            Any: The extra data value.
        """
        return self.extras.get(name, default)


class SubtreeIndependentScoring(ScoringStrategy):
    """
    Base class for scoring strategies with subtree independence.
    
    These strategies can cache partial computations at each node and only
    recompute affected subtrees when changes occur. This is typical for:
    - Felsenstein's pruning algorithm (ML on sequences)
    - BiMarkers/SNP likelihood
    - Any method where the likelihood at a node depends only on its descendants
    """
    
    def __init__(self) -> None:
        """
        Initialize a subtree-independent scoring strategy.
        
        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__()
        # Per-node partial computations
        self._node_partials: dict[str, Any] = {}
        # Track which nodes need recomputation
        self._dirty_nodes: set[str] = set()
    
    def supports_caching(self) -> bool:
        """
        Subtree-independent strategies support caching.
        
        Args:
            N/A
        Returns:
            bool: Always True.
        """
        return True
    
    def _invalidate_node(self, node) -> None:
        """
        Invalidate a node and all its ancestors (upstream propagation).
        
        Args:
            node (Node): The node to invalidate.
        Returns:
            N/A
        """
        key = self._node_cache_key(node)
        self._dirty_nodes.add(key)
        if key in self._node_partials:
            del self._node_partials[key]
    
    def get_partial(self, node) -> Any:
        """
        Get cached partial computation for a node.
        
        Args:
            node (Node): The node to look up.
        Returns:
            Any: The cached partial, or None if not cached.
        """
        key = self._node_cache_key(node)
        return self._node_partials.get(key)
    
    def set_partial(self, node, value: Any) -> None:
        """
        Cache a partial computation for a node.
        
        Args:
            node (Node): The node to cache for.
            value (Any): The partial computation result.
        Returns:
            N/A
        """
        key = self._node_cache_key(node)
        self._node_partials[key] = value
        self._dirty_nodes.discard(key)
    
    def is_node_dirty(self, node) -> bool:
        """
        Check if a specific node needs recomputation.
        
        Args:
            node (Node): The node to check.
        Returns:
            bool: True if the node needs recomputation.
        """
        key = self._node_cache_key(node)
        return key in self._dirty_nodes or key not in self._node_partials


class FullRecomputeScoring(ScoringStrategy):
    """
    Base class for scoring strategies that require full recomputation.
    
    These strategies cannot leverage subtree caching because their scoring
    functions don't have subtree independence. This is typical for:
    - Parsimony scoring (InferMPAllop)
    - Methods where global network structure affects local computations
    """
    
    def __init__(self) -> None:
        """
        Initialize a full-recompute scoring strategy.
        
        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__()
        self._last_score: float = None
    
    def supports_caching(self) -> bool:
        """
        Full-recompute strategies don't support partial caching.
        
        Args:
            N/A
        Returns:
            bool: Always False.
        """
        return False
    
    def invalidate(self, affected_nodes: list = None) -> None:
        """
        Any change invalidates the entire computation.
        
        Args:
            affected_nodes (list[Node], optional): Ignored - all changes
                                                   require full recompute.
        Returns:
            N/A
        """
        self._dirty = True
        self._last_score = None


class CompositeScoring(ScoringStrategy):
    """
    Combines multiple scoring strategies (e.g., likelihood + prior).
    
    Useful for Bayesian methods that combine data likelihood with priors.
    """
    
    def __init__(self, 
                 strategies: list[ScoringStrategy],
                 combiner: Callable[[list[float]], float] = sum) -> None:
        """
        Initialize a composite scoring strategy.
        
        Args:
            strategies (list[ScoringStrategy]): Component strategies.
            combiner (Callable): Function to combine component scores.
                                 Defaults to sum.
        Returns:
            N/A
        """
        super().__init__()
        self.strategies = strategies
        self.combiner = combiner
    
    def score(self, context: ScoringContext) -> float:
        """
        Compute combined score from all component strategies.
        
        Args:
            context (ScoringContext): The scoring context.
        Returns:
            float: The combined score.
        """
        component_scores = [s.score(context) for s in self.strategies]
        return self.combiner(component_scores)
    
    def supports_caching(self) -> bool:
        """
        Composite supports caching if all components do.
        
        Args:
            N/A
        Returns:
            bool: True if all components support caching.
        """
        return all(s.supports_caching() for s in self.strategies)
    
    def invalidate(self, affected_nodes: list = None) -> None:
        """
        Invalidate all component strategies.
        
        Args:
            affected_nodes (list[Node], optional): Nodes to invalidate.
        Returns:
            N/A
        """
        super().invalidate(affected_nodes)
        for strategy in self.strategies:
            strategy.invalidate(affected_nodes)

