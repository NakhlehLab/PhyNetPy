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

This module provides a simplified, phase-based model building system.

The original ModelFactory used a priority queue with type-based dependency
resolution, which had several issues:
- Didn't handle cycles
- Type-based dependencies don't capture instance relationships
- Complex and hard to debug

The new ModelBuilder uses explicit build phases:
1. Each phase is a well-defined step in model construction
2. Phases execute in a deterministic order
3. Each phase has access to previously built components
4. Much simpler to understand and debug

Common phases:
- NetworkPhase: Attach the phylogenetic network
- DataPhase: Attach observed data (MSA, gene trees)
- ParameterPhase: Set up model parameters
- ScoringPhase: Configure the scoring strategy
- ValidationPhase: Validate the complete model
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from collections import OrderedDict

if TYPE_CHECKING:
    from .Network import Network
    from .MSA import MSA

from .ScoringStrategy import ScoringStrategy, ScoringContext
from .NetworkAdapter import NetworkAdapter


class BuildContext:
    """
    Accumulates built components during the model construction process.
    
    Phases can read from and write to this context, allowing later phases
    to use components built by earlier phases.
    """
    
    def __init__(self) -> None:
        """
        Initialize an empty build context.
        
        Args:
            N/A
        Returns:
            N/A
        """
        self._components: OrderedDict[str, Any] = OrderedDict()
        self._metadata: dict[str, Any] = {}
    
    def set(self, key: str, value: Any) -> None:
        """
        Store a component in the context.
        
        Args:
            key (str): Component identifier.
            value (Any): The component.
        Returns:
            N/A
        """
        self._components[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a component from the context.
        
        Args:
            key (str): Component identifier.
            default (Any, optional): Default if not found.
        Returns:
            Any: The component, or default.
        """
        return self._components.get(key, default)
    
    def has(self, key: str) -> bool:
        """
        Check if a component exists in the context.
        
        Args:
            key (str): Component identifier.
        Returns:
            bool: True if the component exists.
        """
        return key in self._components
    
    def require(self, key: str) -> Any:
        """
        Get a component, raising an error if it doesn't exist.
        
        Args:
            key (str): Component identifier.
        Returns:
            Any: The component.
        Raises:
            BuildError: If the component doesn't exist.
        """
        if key not in self._components:
            raise BuildError(f"Required component '{key}' not found in context. "
                           f"Available: {list(self._components.keys())}")
        return self._components[key]
    
    def set_meta(self, key: str, value: Any) -> None:
        """
        Store metadata in the context.
        
        Args:
            key (str): Metadata key.
            value (Any): Metadata value.
        Returns:
            N/A
        """
        self._metadata[key] = value
    
    def get_meta(self, key: str, default: Any = None) -> Any:
        """
        Retrieve metadata from the context.
        
        Args:
            key (str): Metadata key.
            default (Any, optional): Default if not found.
        Returns:
            Any: The metadata value, or default.
        """
        return self._metadata.get(key, default)
    
    def all_components(self) -> dict[str, Any]:
        """
        Get all built components.
        
        Args:
            N/A
        Returns:
            dict[str, Any]: All components.
        """
        return dict(self._components)


class BuildPhase(ABC):
    """
    Abstract base class for model build phases.
    
    Each phase performs a specific step in model construction.
    Phases are executed in the order they are added to the builder.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name for this phase.
        
        Args:
            N/A
        Returns:
            str: The phase name.
        """
        pass
    
    @abstractmethod
    def execute(self, context: BuildContext) -> None:
        """
        Execute this build phase.
        
        Args:
            context (BuildContext): The build context with previously
                                    built components.
        Returns:
            N/A
        Raises:
            BuildError: If the phase fails.
        """
        pass
    
    def validate_prerequisites(self, context: BuildContext) -> None:
        """
        Check that required components exist before executing.
        
        Override in subclasses to specify dependencies.
        
        Args:
            context (BuildContext): The build context.
        Returns:
            N/A
        Raises:
            BuildError: If prerequisites are not met.
        """
        pass


class BuildError(Exception):
    """
    Exception raised during model building.
    """
    
    def __init__(self, message: str, phase: str = None) -> None:
        """
        Create a build error.
        
        Args:
            message (str): Error message.
            phase (str, optional): Phase where error occurred.
        Returns:
            N/A
        """
        if phase:
            message = f"[{phase}] {message}"
        super().__init__(message)
        self.phase = phase


class ModelBuilder:
    """
    Phase-based model builder.
    
    Usage:
        builder = ModelBuilder()
        builder.add_phase(NetworkPhase(my_network))
        builder.add_phase(DataPhase(my_msa))
        builder.add_phase(ScoringPhase(my_strategy))
        model = builder.build()
    """
    
    def __init__(self) -> None:
        """
        Initialize an empty model builder.
        
        Args:
            N/A
        Returns:
            N/A
        """
        self._phases: list[BuildPhase] = []
        self._built: bool = False
    
    def add_phase(self, phase: BuildPhase) -> 'ModelBuilder':
        """
        Add a build phase.
        
        Args:
            phase (BuildPhase): The phase to add.
        Returns:
            ModelBuilder: Self, for chaining.
        """
        self._phases.append(phase)
        return self
    
    def insert_phase(self, index: int, phase: BuildPhase) -> 'ModelBuilder':
        """
        Insert a build phase at a specific position.
        
        Args:
            index (int): Position to insert at.
            phase (BuildPhase): The phase to insert.
        Returns:
            ModelBuilder: Self, for chaining.
        """
        self._phases.insert(index, phase)
        return self
    
    def build(self) -> 'BuiltModel':
        """
        Execute all phases and build the model.
        
        Args:
            N/A
        Returns:
            BuiltModel: The constructed model.
        Raises:
            BuildError: If any phase fails.
        """
        context = BuildContext()
        
        for phase in self._phases:
            try:
                phase.validate_prerequisites(context)
                phase.execute(context)
            except BuildError:
                raise
            except Exception as e:
                raise BuildError(str(e), phase=phase.name) from e
        
        self._built = True
        return BuiltModel(context)


class BuiltModel:
    """
    The result of model building.
    
    Provides access to all built components and methods for scoring.
    """
    
    def __init__(self, context: BuildContext) -> None:
        """
        Create a built model from a build context.
        
        Args:
            context (BuildContext): The completed build context.
        Returns:
            N/A
        """
        self._context = context
        
        # Extract key components for quick access
        self._network_adapter: NetworkAdapter = context.get("network_adapter")
        self._scoring_strategy: ScoringStrategy = context.get("scoring_strategy")
        self._parameters: dict[str, Any] = context.get("parameters", {})
    
    @property
    def network(self):
        """
        Get the phylogenetic network.
        
        Args:
            N/A
        Returns:
            Network: The network, or None.
        """
        if self._network_adapter:
            return self._network_adapter.network
        return None
    
    @property
    def network_adapter(self) -> NetworkAdapter:
        """
        Get the network adapter.
        
        Args:
            N/A
        Returns:
            NetworkAdapter: The adapter.
        """
        return self._network_adapter
    
    @property
    def scoring_strategy(self) -> ScoringStrategy:
        """
        Get the scoring strategy.
        
        Args:
            N/A
        Returns:
            ScoringStrategy: The strategy.
        """
        return self._scoring_strategy
    
    def get_component(self, key: str, default: Any = None) -> Any:
        """
        Get a component by key.
        
        Args:
            key (str): Component key.
            default (Any, optional): Default if not found.
        Returns:
            Any: The component.
        """
        return self._context.get(key, default)
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """
        Get a parameter value.
        
        Args:
            name (str): Parameter name.
            default (Any, optional): Default if not found.
        Returns:
            Any: The parameter value.
        """
        return self._parameters.get(name, default)
    
    def set_parameter(self, name: str, value: Any) -> None:
        """
        Update a parameter value.
        
        This will invalidate cached computations if using a caching strategy.
        
        Args:
            name (str): Parameter name.
            value (Any): New value.
        Returns:
            N/A
        """
        self._parameters[name] = value
        if self._scoring_strategy:
            self._scoring_strategy.invalidate()
    
    def likelihood(self) -> float:
        """
        Compute the model likelihood/score.
        
        Args:
            N/A
        Returns:
            float: The likelihood or score.
        Raises:
            BuildError: If no scoring strategy is configured.
        """
        if self._scoring_strategy is None:
            raise BuildError("No scoring strategy configured")
        
        # Create scoring context
        context = ScoringContext(
            network=self.network,
            data=self._context.get("data"),
            parameters=self._parameters,
            extras=self._context.all_components()
        )
        
        return self._scoring_strategy.score(context)
    
    def invalidate(self, affected_nodes=None) -> None:
        """
        Invalidate cached computations.
        
        Args:
            affected_nodes: Nodes that have changed.
        Returns:
            N/A
        """
        if self._network_adapter:
            if affected_nodes:
                self._network_adapter.mark_dirty(affected_nodes)
            else:
                self._network_adapter.clear_all_data()
        
        if self._scoring_strategy:
            self._scoring_strategy.invalidate(affected_nodes)


# =====================
# Common Build Phases
# =====================

class NetworkPhase(BuildPhase):
    """
    Phase that sets up the phylogenetic network.
    """
    
    def __init__(self, network) -> None:
        """
        Create a network phase.
        
        Args:
            network: The phylogenetic network.
        Returns:
            N/A
        """
        self._network = network
    
    @property
    def name(self) -> str:
        return "Network"
    
    def execute(self, context: BuildContext) -> None:
        """
        Create a NetworkAdapter and store it in context.
        
        Args:
            context (BuildContext): The build context.
        Returns:
            N/A
        """
        adapter = NetworkAdapter(self._network)
        context.set("network", self._network)
        context.set("network_adapter", adapter)
        
        # Store leaf/root info for convenience
        context.set_meta("num_leaves", len(adapter.leaves()))
        if adapter.root():
            context.set_meta("root_label", adapter.root().label)


class DataPhase(BuildPhase):
    """
    Phase that attaches observed data to the model.
    """
    
    def __init__(self, data: Any, data_type: str = "msa") -> None:
        """
        Create a data phase.
        
        Args:
            data: The observed data (MSA, gene trees, etc.)
            data_type (str): Type identifier for the data.
        Returns:
            N/A
        """
        self._data = data
        self._data_type = data_type
    
    @property
    def name(self) -> str:
        return "Data"
    
    def execute(self, context: BuildContext) -> None:
        """
        Store data in context.
        
        Args:
            context (BuildContext): The build context.
        Returns:
            N/A
        """
        context.set("data", self._data)
        context.set_meta("data_type", self._data_type)


class ParameterPhase(BuildPhase):
    """
    Phase that sets up model parameters.
    """
    
    def __init__(self, parameters: dict[str, Any]) -> None:
        """
        Create a parameter phase.
        
        Args:
            parameters: Dictionary of parameter names to values.
        Returns:
            N/A
        """
        self._parameters = parameters
    
    @property
    def name(self) -> str:
        return "Parameters"
    
    def execute(self, context: BuildContext) -> None:
        """
        Store parameters in context.
        
        Args:
            context (BuildContext): The build context.
        Returns:
            N/A
        """
        # Merge with any existing parameters
        existing = context.get("parameters", {})
        existing.update(self._parameters)
        context.set("parameters", existing)


class ScoringPhase(BuildPhase):
    """
    Phase that configures the scoring strategy.
    """
    
    def __init__(self, strategy: ScoringStrategy) -> None:
        """
        Create a scoring phase.
        
        Args:
            strategy: The scoring strategy to use.
        Returns:
            N/A
        """
        self._strategy = strategy
    
    @property
    def name(self) -> str:
        return "Scoring"
    
    def execute(self, context: BuildContext) -> None:
        """
        Store scoring strategy in context.
        
        Args:
            context (BuildContext): The build context.
        Returns:
            N/A
        """
        context.set("scoring_strategy", self._strategy)


class ValidationPhase(BuildPhase):
    """
    Phase that validates the complete model.
    """
    
    def __init__(self, 
                 required_components: list[str] = None,
                 custom_validators: list = None) -> None:
        """
        Create a validation phase.
        
        Args:
            required_components: Component keys that must exist.
            custom_validators: List of (name, validator_func) tuples.
        Returns:
            N/A
        """
        self._required = required_components or []
        self._validators = custom_validators or []
    
    @property
    def name(self) -> str:
        return "Validation"
    
    def execute(self, context: BuildContext) -> None:
        """
        Validate the model.
        
        Args:
            context (BuildContext): The build context.
        Returns:
            N/A
        Raises:
            BuildError: If validation fails.
        """
        # Check required components
        for key in self._required:
            if not context.has(key):
                raise BuildError(f"Missing required component: {key}")
        
        # Run custom validators
        for name, validator in self._validators:
            if not validator(context):
                raise BuildError(f"Validation failed: {name}")


class CustomPhase(BuildPhase):
    """
    A flexible phase that executes a custom function.
    
    Useful for one-off build steps without creating a new class.
    """
    
    def __init__(self, 
                 name: str,
                 executor, 
                 prerequisites: list[str] = None) -> None:
        """
        Create a custom phase.
        
        Args:
            name: Phase name.
            executor: Function(context) to execute.
            prerequisites: Required component keys.
        Returns:
            N/A
        """
        self._name = name
        self._executor = executor
        self._prerequisites = prerequisites or []
    
    @property
    def name(self) -> str:
        return self._name
    
    def validate_prerequisites(self, context: BuildContext) -> None:
        """
        Check prerequisites exist.
        
        Args:
            context (BuildContext): The build context.
        Returns:
            N/A
        """
        for key in self._prerequisites:
            if not context.has(key):
                raise BuildError(
                    f"Prerequisite '{key}' not found for phase '{self._name}'"
                )
    
    def execute(self, context: BuildContext) -> None:
        """
        Execute the custom function.
        
        Args:
            context (BuildContext): The build context.
        Returns:
            N/A
        """
        self._executor(context)

