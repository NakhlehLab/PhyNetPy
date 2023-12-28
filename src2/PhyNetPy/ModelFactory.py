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


# TODO: Probability multiplier component???   
            
class NetworkComponent(ModelComponent):
    def __init__(self, dependencies: set, net : DAG) -> None:
        super().__init__(dependencies)
        self.network = net
    
    def build(self, model : Model):
        model.network = self.network
        
        for node in self.network.get_nodes():
            new_node = ANetworkNode(name=node.get_name())
            model.network_node_map[node] = new_node
            
            if model.network.out_degree(node) == 0:  # This is a leaf
                model.nodetypes["leaf"].append(new_node)
            elif model.network.in_degree(node) != 0:  # An internal node that is not the root
                model.nodetypes["internal"].append(new_node)
            else:  
                model.nodetypes["root"].append(new_node)
    
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

class LikelihoodFunctionComponent(ModelComponent):
    
    def __init__(self, dependencies: set, root_func : Callable, internal_func : Callable, leaf_func : Callable) -> None:
        super().__init__(dependencies)
        self.root = root_func
        self.internal = internal_func
        self.leaf = leaf_func
    
    def build(self, model : Model) -> None:
        for network_node in model.nodetypes["leaf"]:
            network_node.set_likelihood_func(self.leaf)
        for network_node in model.nodetypes["internal"]:
            network_node.set_likelihood_func(self.internal)
        for network_node in model.nodetypes["root"]:
            network_node.set_likelihood_func(self.root)
        
class SimulationFunctionComponent(ModelComponent):
    
    def __init__(self, dependencies: set, root_func : Callable, internal_func : Callable, leaf_func : Callable) -> None:
        super().__init__(dependencies)
        self.root = root_func
        self.internal = internal_func
        self.leaf = leaf_func
    
    def build(self, model : Model) -> None:
        for network_node in model.nodetypes["leaf"]:
            network_node.set_simulation_func(self.leaf)
        for network_node in model.nodetypes["internal"]:
            network_node.set_simulation_func(self.internal)
        for network_node in model.nodetypes["root"]:
            network_node.set_simulation_func(self.root)

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
                    branch = SNPBranchNode(branch_index, branch_len, self.snp_Q, self.vpis)
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
    def __init__(self, name: str = None):
        super(NetworkNode, self).__init__()
        super(CalculationNode).__init__()
        self.name = name

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
        print(f"Calculating: {self.name}")
        
        if self.get_model_parents() is not None:
            self.cached = self.likelihood([child.get() for child in self.get_model_parents()])
        else:
            self.cached = self.likelihood()
            
        self.updated = False

        # return calculation
        return self.cached

    def calc_sim(self):
        
        self.cached = self.simulation(self.children)
        self.updated = False

        # return calculation
        return self.cached

    def get_name(self):
        return self.name

def sample_likelihood_func(num_list : list[int])->int:
    return 1 + sum(num_list)  

def leaf_likelihood_func()->int:
    return 0

def factory_tester():
    
    likelihood : LikelihoodFunctionComponent = LikelihoodFunctionComponent(set([NetworkComponent]), sample_likelihood_func, sample_likelihood_func, leaf_likelihood_func)
    network : NetworkComponent = NetworkComponent(set(), np('/Users/mak17/Documents/PhyloGenPy/PhyloGenPy/src/PhyNetPy/Bayesian/mp_allop_start_net.nex').get_all_networks()[0])
    
    my_model : Model = ModelFactory(likelihood, network).build()
    return my_model.likelihood()


