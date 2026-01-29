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
from .ModelGraph2 import *
from .Network import Network, Edge, Node
from .NetworkParser import NetworkParser
from .MSA import MSA


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
            
            in_deg = model.network.in_degree(node)
            out_deg = model.network.out_degree(node)
            
            if out_deg == 0:  
                new_node = LeafNode(node.label, self.network.get_branches(node)["parent_branches"][0])
                model.nodetypes["leaf"].append(new_node)
            elif in_deg == 1 and out_deg != 0: 
                new_node = InternalNode(node.label, self.network.get_branches(node)["parent_branches"][0]) 
                model.nodetypes["internal"].append(new_node)
            elif in_deg == 2 and out_deg == 1:
                parent_branches = self.network.get_branches(node)["parent_branches"]
                new_node = ReticulationNode(node.label, parent_branches[0], parent_branches[1])
                model.nodetypes["reticulation"].append(new_node)
            elif in_deg == 0:  
                new_node = RootNode(node.label)
                model.nodetypes["root"].append(new_node)
                
            model.network_node_map[node] = new_node
            
    
        # attach model nodes with the proper edges
        edge : Edge
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
        Attach the MSA object to the model by setting the data field for each
        leaf node.

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
                
                model_node.set_data(sequences)

                group_no += 1




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


