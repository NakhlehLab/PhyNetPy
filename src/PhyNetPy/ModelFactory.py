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
Docs   - [x]
Tests  - [ ]
Design - [ ]
"""

from abc import ABC, abstractmethod
from queue import PriorityQueue

from ModelGraph import *
from Network import Network, Edge, Node
from NetworkParser import NetworkParser as np
from typing import Callable

from MSA import MSA

##########################
#### HELPER FUNCTIONS ####
##########################

def join_network(node : ModelNode, model : Model) -> None:
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
    
    def __init__(self, dependencies : set[type]) -> None:
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
        self.component_dependencies = dependencies
    
    @abstractmethod
    def build(self, model : Model) -> None:
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
    other components will be connected to the network in some way. When built, 
    the Model now contains a phylogenetic network where the root of that network 
    should be the root of the Model.
    
    """
    def __init__(self, 
                 dependencies: set[type], 
                 net : Network, 
                 node_constructor : Callable) -> None:
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
        self.constructor = node_constructor
    
    def build(self, model : Model) -> None:
        """
        Attaches a network to the given Model

        Args:
            model (Model): A Model object, most likely completely empty. This 
                           component should be the first to be added.
        Returns:
            N/A
        """
        
        #set the model's reference to a network
        model.network = self.network
        
        #create map from network nodes to model nodes, some bookkeeping
        for node in self.network.V():
            #ANetworkNode(name=node.label, node_type = "network")
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
            else:  
                model.nodetypes["root"].append(new_node)
    
        #attach model nodes with the proper edges
        for edge in model.network.E():
            # Handle network par-child relationships
            # Edge is from modelnode1 to modelnode2 in network, which means
            # modelnode2 is the parent
            modelnode1 = model.network_node_map[edge.src]
            modelnode2 = model.network_node_map[edge.dest]

            # Add modelnode1 as the child of modelnode2
            modelnode2.join(modelnode1)
    
# class SubsitutionModelComponent(ModelComponent):
#     pass

class MSAComponent(ModelComponent):
    """
    Component that links network leaves with taxon data from a MSA object.
    """
    
    def __init__(self,
                 dependencies: set[type], 
                 aln : MSA, 
                 grouping : dict[str, str] = None) -> None:
        """
        Initialize this MSA component.

        Args:
            dependencies (set[type]): list of component dependencies
            aln (MSA): MSA object that is associated with the network that 
                       should be incorporated into the model.
            grouping (dict[str, str], optional): A dictionary that maps
                                                sequence names to group names.
                                                Defaults to None.
        Returns:
            N/A
        """
        super().__init__(dependencies)
        self.grouping : dict[str, str] = grouping
        self.aln : MSA = aln
        
    def build(self, model : Model) -> None:
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
            # For each model node that is a leaf, attach the correct 
            # sequence record(s) to it.
            if model_node in model.nodetypes["leaf"]:
                
                if self.grouping is not None:
                    sequences = self.aln.group_given_id(group_no)
                else:
                    seq_rec = self.aln.seq_by_name(network_node.label)
                    sequences = [seq_rec]
                
                new_ext_species = ExtantSpecies(network_node.label,
                                                sequences)
                model.all_nodes[ExtantSpecies].append(new_ext_species)
                new_ext_species.join(model_node)
                    
class ANetworkNode(NetworkNode, CalculationNode):
    """
    This class provides a starting point for creating custom network nodes that
    can be used in a model. It is a combination of the NetworkNode and
    CalculationNode classes. It is recommended to use this class as a base for
    any custom network nodes that are created.
    """
    
    def __init__(self,
                 name : str = None, 
                 node_type : str = None,
                 likelihood : Callable = None,
                 simulation : Callable = None) -> None:
        """
        Initialize a network node.

        Args:
            name (str, optional): Node name. Defaults to None.
            node_type (str, optional): Any other subtype of 
                                       a network node, such as a leaf or 
                                       root or internal node. Defaults to None.
        """
        super(NetworkNode, self).__init__()
        super(CalculationNode).__init__()
        self.name = name
        self.node_type = node_type
        self.likelihood = likelihood
        self.simulation = simulation
        self.updated = False
        
    def node_move_bounds(self) -> list[int]:
        """
        Return the bounds (time) for a node move.

        Args:
            N/A
        Returns:
            list[int]: A list of two integers, the lower and upper bounds for
                       a node move.
        """
        return [0, 0]
    
    def update(self, new_name : str) -> None:
        """
        Update the name of the node.

        Args:
            new_name (str): new name for the node.
        Returns:
            N/A
        """
        self.name = new_name
        self.upstream()

    def get(self) -> float:
        """
        Get the partial model likelihood at this node.

        Args:
            N/A
        Returns:
            float: The partial model likelihood at this node.
        """
        if self.updated:
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
        if self.get_children() is not None:
            self.cached = self.likelihood([child.get() for child in self.get_children()], self)
        else:
            self.cached = self.likelihood()
            
        self.updated = False

        # return calculation
        return self.cached

    def sim(self) -> Any:
        """
        Simulate model data at this node.

        Args:
            N/A
        Returns:
            Any: Any simulated data.
        """
        self.cached = self.simulation(self.get_parents())
        self.updated = False

        # return calculation
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
    A component meta-type that provides support for various stand-alone model
    parameters that are not directly connected to the network or other 
    structures.
    """
    
    def __init__(self, 
                 dependencies: set[type], 
                 param_name : str, 
                 param_value : float,
                 attach_type: list[type]) -> None:
        """
        Initialize a parameter component.

        Args:
            dependencies (set[type]): List of component dependencies
            param_name (str): Name of the parameter
            param_value (float): Value of the parameter
            attach_type (list[type]): List of model node types that this 
                                      parameter should be attached to.
        Returns:
            N/A
        """
        self.name : str = param_name
        self.value : float = param_value
        self.attach_types : list[type] = attach_type
        self.param = Parameter(self.name, self.value)
        super().__init__(dependencies)
    
    def build(self, model : Model) -> None:
        """
        Attach the parameter to the model.

        Args:
            model (Model): The model that is being built.
        Returns:
            N/A
        """
        
        for node_type in self.attach_types:
            for model_node in model.all_nodes[node_type]:
                model_node.join(self.param)
        
#######################
#### MODEL FACTORY ####
#######################

class ModelFactory:
    """
    A factory class that builds a phylogenetic model from a set of model
    components.
    """
    
    def __init__(self, *items : ModelComponent) -> None:
        """
        Initialize the model factory with a set of model components.
        
        Args:
            *items (ModelComponent): A set of model components that are to be 
                                     built into a model.
        Returns:
            N/A
        """
        self.components = PriorityQueue()
        self.output_model : Model = Model()
        for item in items:
            self.put(item)
    
    def put(self, item : ModelComponent) -> None:
        """
        Submit a model component to the factory.

        Args:
            item (ModelComponent): A model component that is to be built into
                                   the model.
        Returns:
            N/A
        """
        self.components.put([len(item.component_dependencies), item])
    
    def build(self) -> Model:
        """
        Build the model from the components by delegating the building of each
        component to the component itself, and building the model in the order
        of the component dependencies.

        Args:
            N/A
        Returns:
            Model: The model that was built from the components.
        """
        
        while not self.components.empty():
            next_component : ModelComponent = self.components.get()[1]
            next_component.build(self.output_model)
            for component in self.components.queue:
                #Building the current component satisfied a dependency in another component
                if type(next_component) in component[1].component_dependencies:
                    component[1].component_dependencies.remove(type(next_component))
                    component[0] -= 1
        
        return self.output_model






