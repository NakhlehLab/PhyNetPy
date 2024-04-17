""" 
Author : Mark Kessler
Last Stable Edit : 4/10/24
First Included in Version : 1.0.0
Approved for Release : NO
"""

from abc import ABC, abstractmethod
from queue import PriorityQueue

from ModelGraph import *
from Network import Network, Edge, Node
from NetworkParser import NetworkParser as np
from typing import Callable

from MSA import MSA

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
        """
        self.component_dependencies = dependencies
    
    @abstractmethod
    def build(self, model : Model) -> None:
        """
        Given a model, hook up this component to it.

        Args:
            model (Model): The model currently being built.
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
        """
        
        #set the model's reference to a network
        model.network = self.network
        
        #create map from network nodes to model nodes, some bookkeeping
        for node in self.network.get_nodes():
            #ANetworkNode(name=node.get_name(), node_type = "network")
            new_node = self.constructor(node.get_name()) 
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
        for edge in model.network.get_edges():
            # Handle network par-child relationships
            # Edge is from modelnode1 to modelnode2 in network, which means
            # modelnode2 is the parent
            modelnode1 = model.network_node_map[edge.src]
            modelnode2 = model.network_node_map[edge.dest]

            # Add modelnode1 as the child of modelnode2
            modelnode2.join(modelnode1)
    
class SubsitutionModelComponent(ModelComponent):
    pass

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
            grouping (dict[str, str], optional): _description_. Defaults to None.
        """
        super().__init__(dependencies)
        self.grouping : dict[str, str] = grouping
        self.aln : MSA = aln
        
    def build(self, model : Model) -> None:
        group_no = 0
        for network_node, model_node in model.network_node_map.items():
            # For each model node that is a leaf, attach the correct 
            # sequence record(s) to it.
            if model_node in model.nodetypes["leaf"]:
                
                if self.grouping is not None:
                    sequences = self.aln.group_given_id(group_no)
                else:
                    seq_rec = self.aln.seq_by_name(network_node.get_name())
                    sequences = [seq_rec]
                
                new_ext_species = ExtantSpecies(network_node.get_name(),
                                                sequences)
                new_ext_species.join(model_node)
                    
class ANetworkNode(NetworkNode, CalculationNode):
    def __init__(self, name: str = None, node_type : str = None):
        super(NetworkNode, self).__init__()
        super(CalculationNode).__init__()
        self.name = name
        self.node_type = node_type

    def node_move_bounds(self):
        return [0, 0]
    
    def update(self, new_name):
        self.name = new_name
        self.upstream()

    def get(self):
        if self.updated:
            return self.calc()
        else:
            return self.cached

    def calc(self):
        if self.get_children() is not None:
            self.cached = self.likelihood([child.get() for child in self.get_children()], self)
        else:
            self.cached = self.likelihood()
            
        self.updated = False

        # return calculation
        return self.cached

    def sim(self):
        self.cached = self.simulation(self.get_parents())
        self.updated = False

        # return calculation
        return self.cached

    def get_name(self):
        return self.name

class ParameterComponent(ModelComponent):
    def __init__(self, dependencies: set[type], param_name : str, param_value : float) -> None:
        self.name : str = param_name
        self.value : float = param_value
        super().__init__(dependencies)
    
    def build(self, model: Model) -> None:
        root = model.nodetypes["root"]
        
        return super().build(model)
    
    
    
#######################
#### MODEL FACTORY ####
#######################


class ModelFactory:
    
    def __init__(self, *items : ModelComponent):
        self.components = PriorityQueue()
        self.output_model : Model = Model()
        for item in items:
            self.put(item)
    
    def put(self, item : ModelComponent):
        self.components.put([len(item.component_dependencies), item])
    
    def build(self) -> Model:
        
        while not self.components.empty():
            next_component : ModelComponent = self.components.get()[1]
            next_component.build(self.output_model)
            for component in self.components.queue:
                #Building the current component satisfied a dependency in another component
                if type(next_component) in component[1].component_dependencies:
                    component[1].component_dependencies.remove(type(next_component))
                    component[0] -= 1
        
        return self.output_model 
            






