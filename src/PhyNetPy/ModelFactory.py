""" 
Author : Mark Kessler
Last Stable Edit : 7/16/23
First Included in Version : 0.1.0
Approved to Release Date : N/A
"""

from abc import ABC, abstractmethod
from queue import PriorityQueue

from ModelGraph import *
from Graph import DAG
from NetworkParser import NetworkParser as np
from typing import Callable




class ModelComponent(ABC):
    
    def __init__(self, dependencies : set[type]) -> None:
        self.component_dependencies = dependencies
    
    @abstractmethod
    def build(self, model : Model) -> None:
        pass


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
            
class NetworkComponent(ModelComponent):
    """
    Network Component Description:
    
    This component should be the first component built into the model, as most other components will
    be connected to the network in some way. When built, the Model now contains a phylogenetic network 
    where the root of that network should be the root of the Model.
    
    """
    def __init__(self, dependencies: set, net : DAG, node_constructor : Callable) -> None:
        super().__init__(dependencies)
        self.network = net
        self.constructor = node_constructor
    
    def build(self, model : Model):
        """
        Attaches a network to the given Model

        Args:
            model (Model): A Model object, ideally empty. This component should be the first to be added.
        """
        
        #set the model's reference to a network
        model.network = self.network
        
        #create map from network nodes to model nodes, some bookkeeping
        for node in self.network.get_nodes():
            new_node = self.constructor(node.get_name()) #ANetworkNode(name=node.get_name(), node_type = "network")
            model.network_node_map[node] = new_node
            
            in_deg = model.network.in_degree(node)
            out_deg = model.network.out_degree(node)
            
            if out_deg == 0:  # This is a leaf
                model.nodetypes["leaf"].append(new_node)
            elif in_deg == 1 and out_deg != 0:  # An internal node that is not the root and is not a reticulation 
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
            modelnode1 = model.network_node_map[edge[0]]
            modelnode2 = model.network_node_map[edge[1]]

            # Add modelnode1 as the child of modelnode2
            modelnode2.join(modelnode1)
    
class SubsitutionModelComponent(ModelComponent):
    pass

class MSAComponent(ModelComponent):
    
    def __init__(self, dependencies: set, grouping : dict[str, str] = None) -> None:
        super().__init__(dependencies)
        self.grouping : dict[str, str] = grouping
        
    def build(self, model : Model) -> None:
        group_no = 0
        for network_node, model_node in model.network_node_map.items():
            if model_node in model.nodetypes["leaf"]:
                #sequence = self.data.get_number_seq(node.get_name()) 
                if self.grouping is not None:
                    sequences = model.data.aln.group_given_id(group_no)
                else:
                    sequences = model.data.aln.seq_by_name(network_node.get_name())
                
                new_ext_species : ExtantSpecies = ExtantSpecies(network_node.get_name(), sequences)
                new_ext_species.join(model_node)
        
class BranchLengthComponent(ModelComponent):
    
    def __init__(self, dependencies: set) -> None:
        super().__init__(dependencies)
    
    def build(self, model : Model):
        branch_index = 0
        node_heights = TreeHeights()
        node_heights_vec = []
        
        for network_node, model_node in model.network_node_map.items():
            
            branches = []
            gamma = network_node.attribute_value_if_exists("gamma")

            for branch_par, branch_lengths in network_node.length().items():
                
                if gamma is not None:
                    gammas = gamma[branch_par.get_name()]
                    
                for branch_len in branch_lengths:
                    #Create new branch
                    branch = BranchLengthNode(branch_index, branch_len)
                    node_heights_vec.append(branch_len)
                    branch_index += 1
                    branches.append(branch)
                    branch.set_net_parent(branch_par)
                    
                    if gamma is not None:
                        print("GAMMAS: " + str(gammas))
                        if branch_par.get_name() in gamma.keys():
                            
                            if len(gammas)==1:
                                branch.set_inheritance_probability(gammas[0][0])
                            else:
                                if gammas[0][1] == branch_len:
                                    branch.set_inheritance_probability(gammas[0][0])
                                    gammas = [gammas[1]]
                                else:
                                    branch.set_inheritance_probability(gammas[1][0])
                                    gammas = [gammas[0]]
                        
                    # Each branch has a link to the vector
                    node_heights.join(branch)
                   
        
            # Point the branch length node to the leaf node
            for branch in branches:
                branch.join(model_node)
            
        node_heights.update(node_heights_vec)
        
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





