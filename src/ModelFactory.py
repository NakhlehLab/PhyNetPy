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
Last Edit : 3/11/25
First Included in Version : 1.0.0

V1 Architecture - Model Factory for component-based model building.
"""

from abc import ABC, abstractmethod
from queue import PriorityQueue
import math
from typing import Callable, Any

# Relative imports
from .ModelGraph import (
    Model, ModelNode, CalculationNode, NetworkNode, Parameter, 
    Accumulator, ExtantSpecies
)
from .Network import Network, Edge, Node
from .NetworkParser import NetworkParser
from .MSA import MSA


##########################
#### HELPER FUNCTIONS ####
##########################

def join_network(node: ModelNode, model: Model) -> None:
    """
    Make @node a model child of all network class model nodes.

    Args:
        node (ModelNode): A ModelNode.
        model (Model): The model, with a network attached.
    Returns:
        N/A
    """
    for internal_node in model.nodetypes["internal"]:
        node.join(internal_node)
    for leaf_node in model.nodetypes["leaf"]:
        node.join(leaf_node)
    for root_node in model.nodetypes["root"]:
        node.join(root_node)
    for retic in model.nodetypes["reticulation"]:
        node.join(retic)
    

##########################
#### MODEL COMPONENTS ####
##########################

class ModelComponent(ABC):
    """
    Abstract class (do not instantiate this), that represents the behavior for
    any phylogenetic network model component that gets fed into a ModelFactory.
    """
    
    # Class-level counter for tie-breaking in priority queue
    _counter = 0
    
    def __init__(self, dependencies: set[type] = None) -> None:
        """
        Initialize a model component that has a set of dependencies (component
        types) that must be added to the model before this component can be 
        built.

        Args:
            dependencies (set[type]): A set of model component types that need  
                                      to be built first before this component 
                                      can be added.
        Returns:
            N/A
        """
        self.component_dependencies = dependencies if dependencies else set()
        # Unique ID for tie-breaking in priority queue comparisons
        self._order = ModelComponent._counter
        ModelComponent._counter += 1
    
    def __lt__(self, other: 'ModelComponent') -> bool:
        """
        Comparison method for priority queue ordering.
        Uses insertion order as tie-breaker when priorities are equal.
        """
        return self._order < other._order
    
    @abstractmethod
    def build(self, model: Model) -> None:
        """
        Given a model, hook up this component to it.

        Args:
            model (Model): The model currently being built.
        Returns:
            N/A
        """
        pass


class NetworkComponent(ModelComponent):
    """
    Network Component Description:
    
    This component should be the first component built into the model, as most 
    other components will be connected to the network in some way.
    """
    def __init__(self, 
                 dependencies: set[type] = None, 
                 net: Network = None, 
                 node_constructor: Callable = None) -> None:
        """
        Builder for the network component of a phylogenetic model.

        Args:
            dependencies (set[type]): list of component dependencies
            net (Network): A network
            node_constructor (Callable): A construction method for custom 
                                         model network nodes.
        Returns:
            N/A
        """
        super().__init__(dependencies)
        self.network = net
        self.constructor = node_constructor if node_constructor else self._default_constructor
    
    def _default_constructor(self, name: str) -> ANetworkNode:
        """Default node constructor that creates ANetworkNode instances."""
        return ANetworkNode(name=name, node_type="network")
    
    def build(self, model: Model) -> None:
        """
        Attaches a network to the given Model

        Args:
            model (Model): A Model object, most likely completely empty.
        Returns:
            N/A
        """
        # set the model's reference to a network
        model.network = self.network
        
        if self.network is None:
            return
        
        # create map from network nodes to model nodes
        for node in self.network.V():
            new_node = self.constructor(node.label) 
            model.network_node_map[node] = new_node
            
            in_deg = model.network.in_degree(node)
            out_deg = model.network.out_degree(node)
            
            if out_deg == 0:  
                model.nodetypes["leaf"].append(new_node)
            elif in_deg == 1 and out_deg != 0:  
                model.nodetypes["internal"].append(new_node)
            elif in_deg == 2 and out_deg == 1:
                model.nodetypes["reticulation"].append(new_node)
            elif in_deg == 0:  
                model.nodetypes["root"].append(new_node)
    
        # attach model nodes with the proper edges
        for edge in model.network.E():
            modelnode1 = model.network_node_map[edge.src]
            modelnode2 = model.network_node_map[edge.dest]
            modelnode2.join(modelnode1)


class MSAComponent(ModelComponent):
    """
    Component that links network leaves with taxon data from a MSA object.
    """
    
    def __init__(self,
                 dependencies: set[type] = None, 
                 aln: MSA = None, 
                 grouping: dict[str, str] = None) -> None:
        """
        Initialize this MSA component.

        Args:
            dependencies (set[type]): list of component dependencies
            aln (MSA): MSA object that is associated with the network.
            grouping (dict[str, str], optional): A dictionary that maps
                                                sequence names to group names.
        Returns:
            N/A
        """
        super().__init__(dependencies)
        self.grouping: dict[str, str] = grouping
        self.aln: MSA = aln
        
    def build(self, model: Model) -> None:
        """
        Attach the MSA object to the model, and link the MSA object to the
        network leaves.

        Args:
            model (Model): The model that is being built.
        Returns:
            N/A
        """
        group_no = 0
        for network_node, model_node in model.network_node_map.items():
            if model_node in model.nodetypes["leaf"]:
                
                if self.grouping is not None:
                    sequences = self.aln.group_given_id(group_no)
                else:
                    seq_rec = self.aln.seq_by_name(network_node.label)
                    sequences = [seq_rec] if seq_rec else []
                
                new_ext_species = ExtantSpecies(network_node.label, sequences)
                model.all_nodes[ExtantSpecies].append(new_ext_species)
                new_ext_species.join(model_node)
                group_no += 1


class ANetworkNode(NetworkNode, CalculationNode):
    """
    This class provides a starting point for creating custom network nodes.
    """
    
    def __init__(self,
                 name: str = None, 
                 node_type: str = None,
                 likelihood: Callable = None,
                 simulation: Callable = None) -> None:
        """
        Initialize a network node.

        Args:
            name (str, optional): Node name. Defaults to None.
            node_type (str, optional): Type of network node. Defaults to None.
            likelihood (Callable, optional): Likelihood function. Defaults to None.
            simulation (Callable, optional): Simulation function. Defaults to None.
        """
        # Initialize NetworkNode (which sets up network_parents, network_children, branches)
        NetworkNode.__init__(self, branch=None)
        # Initialize CalculationNode
        CalculationNode.__init__(self)
        self.name = name
        if name is not None:
            self.set_name(name)  # Use set_name instead of setting label directly
        self.node_type = node_type
        self._likelihood_func = likelihood
        self._simulation_func = simulation
        self.updated = False
        self._time = 0.0
        
    def node_move_bounds(self) -> list[float]:
        """
        Return the bounds (time) for a node move.

        Args:
            N/A
        Returns:
            list[float]: [parental_bound, child_bound]
        """
        parents = self.get_parent(return_all=True)
        children = self.get_children()
        
        if parents is not None and len(parents) > 0:
            if len(parents) != 1:
                par_bound = self.get_time() - max([parent.get_time() for parent in parents]) 
            else:
                par_bound = self.get_time() - parents[0].get_time()
        else:
            par_bound = 0
            
        if children is not None and len(children) > 0:
            child_bound = min([child.get_time() for child in children]) - self.get_time()
        else:
            child_bound = float('inf')
            
        return [par_bound, child_bound]
    
    def get_time(self) -> float:
        """Get the time of this node."""
        return self._time
    
    def set_time(self, time: float) -> None:
        """Set the time of this node."""
        self._time = time
    
    def update(self, new_sequence: list = None, new_name: str = None) -> None:
        """
        Update the node. Accepts same signature as ExtantSpecies.update() for
        compatibility when propagating updates through the model graph.

        Args:
            new_sequence (list): Ignored for network nodes (only relevant for leaves).
            new_name (str): new name for the node.
        Returns:
            N/A
        """
        if new_name is not None:
            self.name = new_name
            self.set_name(new_name)
            self.label = new_name
        self.upstream()

    def get(self) -> float:
        """
        Get the partial model likelihood at this node.

        Args:
            N/A
        Returns:
            float: The partial model likelihood at this node.
        """
        if self.updated or self.dirty:
            return self.calc()
        else:
            return self.cached

    def calc(self) -> float:
        """
        Calculate the partial model likelihood at this node.

        Args:
            N/A
        Returns:
            float: The partial model likelihood at this node.
        """
        if self._likelihood_func is not None:
            children = self.get_children()
            if children is not None:
                self.cached = self._likelihood_func([child.get() for child in children], self)
            else:
                self.cached = self._likelihood_func()
        else:
            self.cached = 0.0
            
        self.updated = False
        self.dirty = False
        return self.cached

    def sim(self) -> Any:
        """
        Simulate model data at this node.

        Args:
            N/A
        Returns:
            Any: Any simulated data.
        """
        if self._simulation_func is not None:
            parents = self.get_parent(return_all=True)
            self.cached = self._simulation_func(parents)
        self.updated = False
        return self.cached

    def get_name(self) -> str:
        """
        Gets the name of the node.

        Args:
            N/A
        Returns:
            str: The name of the node.
        """
        return self.name


class ParameterComponent(ModelComponent):
    """
    A component that provides support for stand-alone model parameters.
    """
    
    def __init__(self, 
                 dependencies: set[type] = None, 
                 param_name: str = None, 
                 param_value: float = None,
                 attach_type: list[type] = None) -> None:
        """
        Initialize a parameter component.

        Args:
            dependencies (set[type]): List of component dependencies
            param_name (str): Name of the parameter
            param_value (float): Value of the parameter
            attach_type (list[type]): List of model node types to attach to.
        Returns:
            N/A
        """
        super().__init__(dependencies)
        self.name: str = param_name
        self.value: float = param_value
        self.attach_types: list[type] = attach_type if attach_type else []
        self.param = Parameter(self.name, self.value)
    
    def build(self, model: Model) -> None:
        """
        Attach the parameter to the model.

        Args:
            model (Model): The model that is being built.
        Returns:
            N/A
        """
        for node_type in self.attach_types:
            for model_node in model.all_nodes.get(node_type, []):
                model_node.join(self.param)
   

class AccumulatorComponent(ModelComponent):
    """
    Component that provides a way to accumulate data from the model.
    """
    
    def __init__(self, 
                 dependencies: set[type] = None, 
                 accumulator: Accumulator = None,
                 attach_types: list[type] = None) -> None:
        """
        Initialize an accumulator component.

        Args:
            dependencies (set[type]): List of component dependencies
            accumulator (Accumulator): An accumulator object.
            attach_types (list[type]): The types of model node to attach to.
        Returns:
            N/A
        """
        super().__init__(dependencies)
        self.accumulator = accumulator
        self.attach_types = attach_types if attach_types else []
    
    def build(self, model: Model) -> None:
        """
        Attach the accumulator to the model.

        Args:
            model (Model): The model that is being built.
        Returns:
            N/A
        """
        for node_type in self.attach_types:
            for model_node in model.all_nodes.get(node_type, []):
                model_node.join(self.accumulator)    


class InformationComponent(ModelComponent):
    """
    Component that provides a way to store relevant information in the model.
    """
    
    def __init__(self, 
                 dependencies: set[type] = None, 
                 info: Any = None,
                 attach_types: list[type] = None) -> None:
        """
        Initialize an information component.

        Args:
            dependencies (set[type]): List of component dependencies
            info (Any): Any relevant information.
            attach_types (list[type]): The types of model node to attach to.
        Returns:
            N/A
        """
        super().__init__(dependencies)
        self.info = info
        self.attach_types = attach_types if attach_types else []
    
    def build(self, model: Model) -> None:
        """
        Attach the information to the model.

        Args:
            model (Model): The model that is being built.
        Returns:
            N/A
        """
        for node_type in self.attach_types:
            for model_node in model.all_nodes.get(node_type, []):
                model_node.join(self.info)


#######################
#### MODEL FACTORY ####
#######################

class ModelFactory:
    """
    A factory class that builds a phylogenetic model from a set of model
    components.
    """
    
    def __init__(self, *items: ModelComponent) -> None:
        """
        Initialize the model factory with a set of model components.
        
        Args:
            *items (ModelComponent): A set of model components to be 
                                     built into a model.
        Returns:
            N/A
        """
        self.components = PriorityQueue()
        self.output_model: Model = Model()
        for item in items:
            self.put(item)
    
    def put(self, item: ModelComponent) -> None:
        """
        Submit a model component to the factory.

        Args:
            item (ModelComponent): A model component to be built.
        Returns:
            N/A
        """
        self.components.put([len(item.component_dependencies), item])
    
    def build(self) -> Model:
        """
        Build the model from the components.

        Args:
            N/A
        Returns:
            Model: The model that was built from the components.
        """
        while not self.components.empty():
            next_component: ModelComponent = self.components.get()[1]
            next_component.build(self.output_model)
            for component in self.components.queue:
                if type(next_component) in component[1].component_dependencies:
                    component[1].component_dependencies.remove(type(next_component))
                    component[0] -= 1
        
        return self.output_model


