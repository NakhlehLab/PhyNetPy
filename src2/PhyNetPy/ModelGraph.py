""" 
Author : Mark Kessler
Last Stable Edit : 7/16/23
First Included in Version : 0.1.0
Approved to Release Date : N/A
"""

import math
import random
from abc import ABC, abstractmethod
from math import comb, pow
from GTR import *
from Graph import DAG
from Matrix import Matrix
from Move import Move
#from Bayesian.SNPTransition import SNPTransition
from scipy.special import binom
import scipy
#from Bayesian.SNPModule import *
from Node import Node
#from Bayesian.InferAllop import InferMPAllop
from typing import Callable


def vec_bin_array(arr, m):
    """
    Arguments:
    arr: Numpy array of positive integers
    m: Number of bits of each integer to retain (in the case of DNA, 4)

    Returns a copy of arr with every element replaced with a bit vector.
    Bits encoded as int8's, read from left to right. [1, 0, 1, 1] is 13.
    """
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[..., m - bit_ix - 1] = fetch_bit_func(strs).astype("int8")

    return ret


# def convert_to_heights(node, adj_dict):
#     """
#     This is a recursive function that is used to take a model that is initialized
#     with branch heights and turn it into a model based on node heights.

#     Usage: convert_to_heights(root_node, {})

#     The resulting heights are conditioned such that t=0 is at the root. Need to subtract dictionary value from
#     max(heights of all leaves) to get heights such that the root is the time furthest in the past

#     input: a ModelNode, to change the whole graph use the root.
#     output: a dictionary that maps each model node to a float height value

#     """
    
#     if node.get_parent() is None:  # Root
#         adj_dict[node] = 0  # Start root at t=0
#     else:
#         # For all other nodes, the height will be the branch length plus the node height of its parent
#         for branch in node.get_branches():
#             # Doesn't matter which parent is used to calculate node height, so just use the first one
#             if type(branch) is SNPBranchNode:
#                 if branch.net_parent in adj_dict.keys():
#                     adj_dict[node] = branch.get_length() + adj_dict[branch.net_parent]
#             else:
#                 adj_dict[node] = branch.get() + adj_dict[node.get_parent()]

#     # Done at the leaves
#     if type(node) is FelsensteinLeafNode or type(node) is SNPLeafNode:
#         return adj_dict

#     # Otherwise, recursively call on the children of this node
#     if node.get_children() is not None:
#         for child in node.get_children():
#             # combine maps of all children
#             adj_dict.update(convert_to_heights(child, adj_dict))

#     # Return the built-up mapping
#     return adj_dict


class ModelError(Exception):
    """
    Class to handle any errors related to building the model or running likelihoods computations
    on the model.
    """

    def __init__(self, message="Model is Malformed"):
        super().__init__(message)


        

class Model:
    """
    Class that describes a DAG structure that lazily computes a model likelihood.
    """

    def __init__(self, network: DAG = None, data: Matrix = None, submodel=JC(), snp_params:dict = None, mp_allop_genes :dict[str, str] = None, mp_allop_trees : list[DAG] = None, verbose = False):
        """

        Args:
            network (DAG): The network/tree
            data (Matrix): The sequence data
            submodel (GTR, optional): Any subclass of GTR. Defaults to JC().
            snp_params (dict, optional): dict that holds snp parameters u,v, 
                                         and theta (coalescent rate), as well as 
                                         the total number of samples. Defaults to None.
            verbose (bool, optional): Flag that, if enabled, prints out partial likelihoods for each node. Defaults to False.

        Raises:
            ModelError: If no SNP parameters have been passed in, but the data is of SNP or Binary type
            
            
        """
        
        ##-------ONLY USE THIS STUFF AS OF NOW-------##
        self.network = network
        self.network_container = None
        self.nodetypes = {"leaf":[], "internal": [], "root":[]}
        
        #384
        rand_seed = 992 #random.randint(0, 1000) #97 #508 #650 254 #
        self.seed = rand_seed
        print(f"MODEL SEED: {rand_seed}")
        self.rng = np.random.default_rng(rand_seed)
        
        ##-------------------------------------------##
        
        self.sub = submodel
        self.data = data
        
        self.verbose_out = verbose
        
        self.nodes = []
        self.netnodes_sans_root = []
        self.network_leaves = []
        
        self.tree_heights = None  # type TreeHeights
        self.submodel_node = None # type SubstitutionModel
       
        self.network_node_map : dict[Node, ModelNode] = {}
        self.snp_params = snp_params
        
        #Build the model graph
        """
        if self.data is None:
            if mp_allop_genes is not None and mp_allop_trees is not None:
                self.MP_black_box = InferMPAllop(mp_allop_genes, mp_allop_trees)
            else:
                raise ModelError("Must Provide MP Allop subgenome mapping")
        else:
            if self.data.get_type() == "DNA":
                self.felsenstein_root = None
                self.build_felsenstein()
            elif self.data.get_type() == "SNP":
                if snp_params is None:
                    raise ModelError("No parameters passed in for SNP model")
                else:
                    self.snp_root = None
                    self.snp_Q = None
                    # Need a data structure to keep track of VPI likelihoods
                    self.vpis = PartialLikelihoods()
                    self.build_SNP(snp_params)
            elif self.data.get_type() == "BINARY":
                if snp_params is None:
                    raise ModelError("No parameters passed in for Binary/SNP model")
                else:
                    self.snp_root = None
                    self.snp_Q = None
                    self.build_SNP(snp_params)
        """
            
                
        self.internal = [item for item in self.netnodes_sans_root if item not in self.network_leaves]
        self.summary_str = ""

    def change_branch(self, index: int, value: float):
        """
        Change a branch length in the model and update any nodes upstream from the changed node

        Inputs: index - index into the heights/lengths vector
                value - new height/length to replace the old one

        """
        self.tree_heights.singular_update(index, value)

        
    # def build_felsenstein(self) -> None:
    #     """
    #     Build the model graph for computing the Felsenstein's likelihood of the network,
    #     given the matrix data.
    #     """

    #     # Initialize branch length/height vector and save it for update usage
    #     tree_heights_node = TreeHeights()
    #     self.tree_heights = tree_heights_node
    #     tree_heights_vec = []

    #     # Initialize substitution model node
    #     submodelnode = SubstitutionModel(self.sub)
    #     self.submodel_node = submodelnode
    #     self.nodes.append(submodelnode)

    #     # Initialize substitution model parameter state node
    #     submodel_params = SubstitutionModelParams(self.sub.get_hyperparams()[0], self.sub.get_hyperparams()[1])
    #     self.nodes.append(submodel_params)

    #     # Join state node to its parent (the substitution model node)
    #     submodel_params.join(submodelnode)

    #     # Keep track of which branch maps to what index
    #     branch_index = 0

    #     # Add parsed phylogenetic network into the model
    #     for node in self.network.get_nodes():
    #         if self.network.out_degree(node) == 0:  # This is a leaf

    #             # Create branch for this leaf and add it to the height/length vector
    #             branch = BranchLengthNode(branch_index, list(node.length().values())[0])
    #             tree_heights_vec.append(list(node.length().values())[0])
    #             branch_index += 1

    #             # Each branch has a substitution model and a link to the vector
    #             tree_heights_node.join(branch)
    #             submodelnode.join(branch)

    #             # Calculate the leaf likelihoods
    #             sequence = self.data.get_number_seq(node.get_name())  # Get sequence from the matrix data
    #             new_leaf_node = FelsensteinLeafNode(partials=vec_bin_array(sequence, 4), branch=[branch],
    #                                                 name=node.get_name())
    #             new_ext_species = ExtantSpecies(node.get_name(), sequence)

    #             new_ext_species.join(new_leaf_node)
    #             self.nodes.append(new_ext_species)
    #             self.network_leaves.append(new_leaf_node)

    #             # Point the branch length node to the leaf node
    #             branch.join(new_leaf_node)

    #             # Add to list of model nodes
    #             self.nodes.append(new_leaf_node)
    #             self.nodes.append(branch)
    #             self.netnodes_sans_root.append(new_leaf_node)

    #             # Add to map
    #             self.network_node_map[node] = new_leaf_node

    #         elif self.network.in_degree(node) != 0:  # An internal node that is not the root

    #             # Create branch
    #             branch = BranchLengthNode(branch_index, list(node.length().values())[0])
    #             tree_heights_vec.append(list(node.length().values())[0])
    #             branch_index += 1

    #             # Link to the substitution model
    #             tree_heights_node.join(branch)
    #             submodelnode.join(branch)

    #             # Create internal node and link to branch
    #             new_internal_node = FelsensteinInternalNode(branch=[branch], name=node.get_name())
    #             branch.join(new_internal_node)

    #             # Add to nodes list
    #             self.nodes.append(new_internal_node)
    #             self.nodes.append(branch)
    #             self.netnodes_sans_root.append(new_internal_node)

    #             # Map node to the new internal node
    #             self.network_node_map[node] = new_internal_node
    #         else:  # The root. TODO: Add dependency on the base frequencies

    #             # Create root
    #             new_internal_node = FelsensteinInternalNode(name=node.get_name())
    #             self.felsenstein_root = new_internal_node

    #             # if not self.as_length:
    #             branch_height = BranchLengthNode(branch_index, 0)
    #             branch_index += 1
    #             tree_heights_vec.append(0)
    #             branch_height.join(new_internal_node)
    #             submodelnode.join(branch_height)
    #             tree_heights_node.join(branch_height)

    #             # Add to nodes list
    #             self.nodes.append(new_internal_node)

    #             # Add to node map
    #             self.network_node_map[node] = new_internal_node

    #     for edge in self.network.get_edges():
    #         # Handle network par-child relationships
    #         # Edge is from modelnode1 to modelnode2 in network, which means
    #         # modelnode2 is the parent
    #         modelnode1 = self.network_node_map[edge[0]]
    #         modelnode2 = self.network_node_map[edge[1]]

    #         # Add modelnode1 as the child of modelnode2
    #         modelnode2.join(modelnode1)

    #     # all the branches have been added, set the vector for the TreeHeight nodes
    #     # ADJUST BRANCH LENGTHS TO HEIGHTS
    #     tree_heights_adj = np.zeros(len(tree_heights_vec))
    #     adj_dict = convert_to_heights(self.felsenstein_root, {})

    #     # Keep track of the maximum leaf height, this is used to switch the node heights from root centric to
    #     # leaf centric
    #     max_height = 0

    #     # Set each node height
    #     for node, height in adj_dict.items():
    #         #Only one branch, since felsensteins is for trees only
    #         tree_heights_adj[node.get_branches()[0].get_index()] = height
    #         if height > max_height:
    #             max_height = height

    #     # Subtract dict height from max child height
    #     tree_heights_adj = np.ones(len(tree_heights_adj)) * max_height - tree_heights_adj

    #     # Update all the branch length nodes to be the proper calculated heights
    #     tree_heights_node.update(list(tree_heights_adj))
        

    # def build_SNP(self, snp_params: dict) -> None:
    #     """
    #     Build the model graph for SNAPP network likelihoods.

    #     Args:
    #         snp_params (dict): A mapping of parameter types to their values, to be used in
    #         making the snp transition matrix
    #     """
        
        

    #     # Initialize branch length/height vector and save it for update usage
    #     tree_heights_node = TreeHeights()
    #     self.tree_heights = tree_heights_node
    #     tree_heights_vec = []

    #     self.snp_Q = SNPTransition(snp_params["samples"], snp_params["u"], snp_params["v"], snp_params["coal"])

    #     # Keep track of which branch maps to what index
    #     branch_index = 0
    #     group_no = 0

    #     # Add parsed phylogenetic network into the model
            
    #     for node in self.network.get_nodes():
    #         if self.network.out_degree(node) == 0:  # This is a leaf


    #             # Add sequences for each group to a new SNP leaf node
    #             if snp_params["grouping"]:
    #                 sequences = self.data.aln.group_given_id(group_no)
    #             else:
    #                 sequences = self.data.aln.seq_by_name(node.get_name())
                
    #             print("SEQUENCES FOR " + node.get_name() + " ARE: ")
    #             for seq in sequences:
    #                 print(seq.seq)
                    
                    
    #             group_no += 1
                
    #             # Create branch for this leaf and add it to the height/length vector
    #             branches = []
    #             gamma = node.attribute_value_if_exists("gamma")
    #             print(gamma)
    #             for branch_par, branch_lengths in node.length().items():
                    
    #                 if gamma is not None:
    #                     gammas = gamma[branch_par.get_name()]
                        
    #                 for branch_len in branch_lengths:
    #                     #Create new branch
    #                     branch = SNPBranchNode(branch_index, branch_len, self.snp_Q, self.vpis)
    #                     tree_heights_vec.append(branch_len)
    #                     branch_index += 1
    #                     branches.append(branch)
    #                     branch.set_net_parent(branch_par)
                        
    #                     if gamma is not None:
    #                         print("GAMMAS: " + str(gammas))
    #                         if branch_par.get_name() in gamma.keys():
                                
    #                             if len(gammas)==1:
    #                                 branch.set_inheritance_probability(gammas[0][0])
    #                             else:
    #                                 if gammas[0][1] == branch_len:
    #                                     branch.set_inheritance_probability(gammas[0][0])
    #                                     gammas = [gammas[1]]
    #                                 else:
    #                                     branch.set_inheritance_probability(gammas[1][0])
    #                                     gammas = [gammas[0]]
                            
    #                     # Each branch has a link to the vector
    #                     tree_heights_node.join(branch)
    #                     # Add to list of nodes
    #                     self.nodes.append(branch)
    #                     print("NODE " + node.get_name() + " HAS BRANCH INDEX" + str(branch_index - 1))
                    

    #             new_leaf_node = SNPLeafNode(partials=sequences, branch=branches,
    #                                         name=node.get_name())
    #             new_ext_species = ExtantSpecies(node.get_name(), sequences)

    #             new_ext_species.join(new_leaf_node)
    #             self.nodes.append(new_ext_species)
    #             self.network_leaves.append(new_leaf_node)

    #             # Point the branch length node to the leaf node
    #             for branch in branches:
    #                 branch.join(new_leaf_node)

    #             # Add to list of model nodes
    #             self.nodes.append(new_leaf_node)
    #             self.netnodes_sans_root.append(new_leaf_node)

    #             # Add to map
    #             self.network_node_map[node] = new_leaf_node

    #         elif self.network.in_degree(node) != 0:  # An internal node that is not the root


    #             branches = []
    #             gamma = node.attribute_value_if_exists("gamma")
    #             print(gamma)
                
    #             for branch_par, branch_lengths in node.length().items():
                    
    #                 if gamma is not None:
    #                     gammas = gamma[branch_par.get_name()]
                    
    #                 for branch_len in branch_lengths:
    #                     #Create new branch
    #                     branch = SNPBranchNode(branch_index, branch_len, self.snp_Q, self.vpis)
    #                     branch.set_net_parent(branch_par)
    #                     #tree_heights_vec.append(node.length())
    #                     tree_heights_vec.append(branch_len)
    #                     branch_index += 1
    #                     branches.append(branch)
                        
                        
    #                     if gamma is not None:
    #                         print("GAMMAS: " + str(gammas))
    #                         if branch_par.get_name() in gamma.keys():
    #                             if len(gammas)==1:
    #                                 branch.set_inheritance_probability(gammas[0][0])
    #                             else:
    #                                 if gammas[0][1] == branch_len:
    #                                     branch.set_inheritance_probability(gammas[0][0])
    #                                     gammas = [gammas[1]]
    #                                 else:
    #                                     branch.set_inheritance_probability(gammas[1][0])
    #                                     gammas = [gammas[0]]
                        
    #                     # Each branch has a link to the vector
    #                     tree_heights_node.join(branch)
    #                     # Add to list of nodes
    #                     self.nodes.append(branch)
    #                     print("NODE " + node.get_name() + " HAS BRANCH INDEX" + str(branch_index - 1))
                    
               
    #             # Create internal node and link to branch
    #             new_internal_node = SNPInternalNode(self.data.siteCount(), branch=branches, name=node.get_name())
                
    #             for branch in branches:
    #                 branch.join(new_internal_node)

    #             # Add to nodes list
    #             self.nodes.append(new_internal_node)
    #             self.netnodes_sans_root.append(new_internal_node)

    #             # Map node to the new internal node
    #             self.network_node_map[node] = new_internal_node
    #         else:  # The root.
            
    #             branch_height = SNPBranchNode(branch_index, 0, self.snp_Q, self.vpis)
    #             branch_index += 1
    #             tree_heights_vec.append(0)
    #             tree_heights_node.join(branch_height)
    #             print("NODE " + node.get_name() + " HAS BRANCH INDEX" + str(branch_index - 1))
                
    #             # Create root
    #             new_internal_node = SNPInternalNode(self.data.siteCount(), name=node.get_name(), branch=[branch_height])
    #             branch_height.join(new_internal_node)
    #             self.snp_root = new_internal_node
                
    #             # Add to nodes list
    #             self.nodes.append(new_internal_node)
    #             self.nodes.append(branch_height)

    #             # Add to node map
    #             self.network_node_map[node] = new_internal_node

    #     for edge in self.network.get_edges():
    #         # Handle network par-child relationships
    #         # Edge is from modelnode1 to modelnode2 in network, which means
    #         # modelnode2 is the parent
            
    #         modelnode1 = self.network_node_map[edge[0]]
    #         modelnode2 = self.network_node_map[edge[1]] 

    #         # Add modelnode1 as the child of modelnode2
    #         modelnode2.join(modelnode1)

    #     #For each branch, set the node that it points to
    #     #Was not previously done, since Node objs hadn't been mapped to ModelNode objs yet!
    #     for node in self.nodes:
    #         if type(node) is SNPBranchNode:
    #             if node.net_parent: 
    #                 node.set_net_parent(self.network_node_map[node.net_parent])
                
    #     # Now adjust the model to be ultrametric
    #     tree_heights_adj = np.zeros(len(tree_heights_vec))
    #     adj_dict = convert_to_heights(self.snp_root, {})

    #     # Keep track of the maximum leaf height
    #     max_height = 0

    #     # Set each node height
    #     for node, height in adj_dict.items():
    #         for branch in node.get_branches():
    #             tree_heights_adj[branch.get_index()] = height
    #             if height > max_height:
    #                 max_height = height

    #     # Subtract dict height from max child height to set the max height leaf to be at t=0, and the root at the largest t value
    #     tree_heights_adj = np.ones(len(tree_heights_adj)) * max_height - tree_heights_adj

    #     # Update all the branch length nodes to be the proper calculated heights
    #     tree_heights_node.update(list(tree_heights_adj))
        
    #     #Calculate the leaf descendant set for each node
    #     self.snp_root.calc_leaf_descendants()
    
    
    def update_network(self):
       
        if self.network_container is not None:
            self.network_container.update(self.network)
                

        
            

    def likelihood(self):
        """
        Calculates the likelihood of the model graph lazily, by only
        calculating parts of the model that have been updated/state changed.
        
        Delegates which likelihood based on the type of model. This method is the only 
        likelihood method that should be called outside of this module!!!

        Inputs:
        Outputs: A numerical likelihood value, the dot product of all root vector likelihoods
        """
        
        # if self.data is None:
        #     return self.MP_Allop_Score()
        # elif self.data.get_type() == "DNA":
        #     return self.Felsenstein_likelihood()
        # elif self.data.get_type() == "SNP" or self.data.get_type() == "BINARY":
        #     return self.SNP_likelihood()
        return self.nodetypes["root"][0].get()

        
    
    
    # def Felsenstein_likelihood(self) -> float:
    #     # calculate the root partials or get the cached values
    #     partials = self.felsenstein_root.get()

    #     # Should be the only child of the substitution model node
    #     params_state = self.submodel_node.get_model_parents()[0]
    #     base_freqs = params_state.base_freqs.reshape((4,))
        
    #     if self.verbose_out: #Display all cached partials
    #         for node in self.netnodes_sans_root:
                
    #             print("------" + node.get_name() + "------")
    #             print(node.get())
    #             print("-------------------------------")
    #             print("       ")

    #     # tally up the logs of the dot products
    #     return np.sum(np.log(np.matmul(partials, base_freqs)))

    # def SNP_likelihood(self) -> float:
        
    #     q_null_space = scipy.linalg.null_space(self.snp_Q.Q)
    #     x = q_null_space / (q_null_space[0] + q_null_space[1]) # normalized so the first two values sum to one

    #     F_b_map = self.vpis.vpis[self.snp_root.get()[0]]
    #     F_b = to_array(F_b_map, partials_index(self.snp_params["samples"] + 1), self.data.siteCount()) 
    #     print(F_b) 
        
    #     L = np.zeros(self.data.siteCount())
       
    #     # EQ 20, Root probabilities
    #     for site in range(self.data.siteCount()):
    #         L[site] = np.dot(F_b[:, site], x)
            
    #     if self.verbose_out: #Display all cached partials
    #         for node in self.netnodes_sans_root:
    #             branches = node.get_branches()
    #             for branch in branches:
    #                 f_t = branch.get()[1]
    #                 f_b = branch.get()[0]
    #                 print("------" + node.get_name() + "------")
    #                 print("F_t :")
    #                 print(f_t)
    #                 print("F_b :")
    #                 print(f_b)
    #                 print("-------------------------------")
    #                 print("       ")
            
    #         # finally, print root probabilities
    #         print("------ROOT PROBS------")
    #         print("F_b :")
    #         print(F_b)
    #         print("-------------------------------")
    #         print("       ")
    #     print("NON-LOG PROBABILITY: " + str(np.sum(L)))
    #     return np.sum(np.log(L))

    def execute_move(self, move: Move):
        """
        The operator move has asked for permission to work on this model.
        Pass the move this model and get the model that is the result of the operation on this model. IT IS THE SAME OBJ

        Input: move, a Move obj or any subtype
        Output: the !same! obj that is the result of doing Move on this Model obj
        """
        return move.execute(self)

    def summary(self, tree_filename: str, summary_filename: str):
        """
        Writes summary of calculations to a file, and gets the current state of the model
        and creates a network obj so that the newick format can be output.

        Inputs:
        1) tree_filename : a string that is the name of the file to output a newick string to.
                           if the filename does not exist, a new file will be created in the directory in which
                           one is operating in.

        2) summary_filename : a string that is the name of the file to output logging information.
                              if the filename does not exist, a new file will be created in the current directory

        TODO: TEST FILE CREATION

        """
        # Step 1: create network obj
        net = DAG()

        network_nodes = []
        if self.data.get_type() == "SNP":
            network_nodes.extend([self.snp_root])
        else:
            network_nodes.extend([self.felsenstein_root])
            
        network_nodes.extend(self.network_leaves)
        network_nodes.extend(self.netnodes_sans_root)

        inv_map = {v: k for k, v in self.network_node_map.items()}
        net.add_nodes([inv_map[node] for node in network_nodes])

        for node in network_nodes:
            for branch in node.get_branches():
                branch_len = branch.get()
                if node.parents is not None:
                    inv_map[node].set_length(branch_len, None)
                else:
                    inv_map[node].set_length(branch_len, None)

            # Add edges
            if node.get_children() is not None:
                for child in node.get_children():
                    net.add_edges([inv_map[node], inv_map[child]]) 

        newick_str = net.newick()

        # Write newick string to output file
        if tree_filename is not None:
            text_file = open(tree_filename, "w")
            text_file.write(newick_str)
            text_file.close()

        # Step 2: write iter summary to a file
        if summary_filename is not None:
            text_file2 = open(summary_filename, "w")
            text_file2.write(self.summary_str)
            text_file2.close()

    # def get_tree_heights(self):
    #     return self.tree_heights

    # def get_network_leaves(self):
    #     return self.network_leaves

class ModelNode:
    """
    Class that defines the graphical structure and shared interactions between
    any node in the Model.
    """

    def __init__(self, successors=None, predecessors=None):
        self.successors = successors
        self.predecessors = predecessors

    def add_successor(self, model_node):
        """
        Adds a successor to this node.

        Input: model_node (type ModelNode)

        """
        if self.successors is None:
            self.successors = [model_node]
        else:
            self.successors.append(model_node)

    def add_predecessor(self, model_node):
        """
        Adds a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if self.predecessors is None:
            self.predecessors = [model_node]
        else:
            self.predecessors.append(model_node)

    def join(self, other_node):
        """
        Adds other_node as a parent, and adds this node as
        a child of other_node

        Input: other_node (type ModelNode)
        """
        self.add_successor(other_node)
        other_node.add_predecessor(self)

    def unjoin(self, other_node):
        self.remove_successor(other_node)
        other_node.remove_predecessor(self)

    def remove_successor(self, model_node):
        """
        Removes a successor to this node.

        Input: model_node (type ModelNode)
        """
        if model_node in self.successors:
            self.successors.remove(model_node)

    def remove_predecessor(self, model_node):
        """
        Removes a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if model_node in self.predecessors:
            self.predecessors.remove(model_node)

    def get_model_parents(self):
        """
        Returns: the list of child nodes to this node
        """
        return self.predecessors

    def get_model_children(self):
        """
        Returns: the list of parent nodes to this node
        """
        return self.successors

    def in_degree(self):
        """
        Calculates the in degree of the current node (ie number of children)

        If 0, this node is a leaf
        """
        if self.predecessors is None:
            return 0
        return len(self.predecessors)

    def out_degree(self):
        """
        Calculates the out degree of the current node (ie number of parents)

        If 0, this node is a root of the Model
        """
        if self.successors is None:
            return 0
        return len(self.successors)

    def find_root(self):
        """
        TODO: PLS MAKE MORE EFFICIENT THIS IS DUMB

        """
        if self.out_degree() == 0:
            return {self}
        else:
            roots = set()
            for neighbor in self.successors:
                roots.update(neighbor.find_root())  # set update

            return roots


class CalculationNode(ABC, ModelNode):
    """
    Subclass of a ModelNode that calculates a portion of the model likelihood.
    """

    def __init__(self, likelihood_func : Callable = None, simulation_func : Callable = None):
        super(CalculationNode, self).__init__()
        self.updated = True  # on initialization, we should do the calculation
        self.cached = None
        self.likelihood : Callable = likelihood_func
        self.simulation : Callable = simulation_func

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        This method should be implemented in each CalculationNode subclass.
        Updating internal data should be handled on an individual basis.

        When the model graph runs its calculate routine, this update method will have marked
        this calculation node and any calculation nodes upstream as needing recalculation.
        """
        pass

    @abstractmethod
    def get(self):
        """
        Either retrieves the cached calculation or redoes the calculation for this node
        This is an abstract method, due to the fact that the type of recalculation will vary.

        Returns: a vector of partial likelihoods
        """
        pass
    
    
    def set_likelihood_func(self, calc_func : Callable):
        """
        Set the likelihood function for this node
        """
        self.likelihood = calc_func
    
    def set_simulation_func(self, calc_func : Callable):
        """
        This method should be implemented in each CalculationNode subclass.
        Doing a calculation should be a unique operation depending on the type of node.

        Returns: A vector of partial likelihoods.
        """
        self.simulation = calc_func
        

    @abstractmethod
    def calc(self, *args, **kwargs):
        """
        This method should be implemented in each CalculationNode subclass.
        Doing a calculation should be a unique operation depending on the type of node.

        Returns: A vector of partial likelihoods.
        """
        pass
    
    @abstractmethod
    def calc_sim(self, *args, **kwargs):
        """
        This method should be implemented in each CalculationNode subclass.
        Doing a calculation should be a unique operation depending on the type of node.

        Returns: A vector of partial likelihoods.
        """
        pass

    def upstream(self):
        """
        Finds a path within the model graph from this node to the root, and marks each node along the way as updated
        using the switch_updated() method

        If all neighbors need to be recalculated, then so must every node upstream of it, and so we may stop updating
        """
        # First update self
        self.switch_updated()

        # Get parent nodes and check that this node is not the root (in which case we're done
        neighbors = self.get_model_children()
        if neighbors is None:
            return

        roots = self.find_root()

        # If all parent nodes are marked to be recalculated, then so must be each path from this node to the root,
        # so no further steps are required
        all_updated = True
        for neighbor in neighbors:
            if not neighbor.updated:
                all_updated = False

        # Otherwise, call upstream on each neighbor
        if not all_updated:
            for neighbor in neighbors:
                if neighbor in roots:
                    neighbor.upstream()
                    return
                neighbor.upstream()

    def switch_updated(self):
        """
        A model node is updated if any of its calculation nodes downstream have been changed.

        This method will be called when a downstream node calls its upstream() method, setting this node
        as a node that needs to be recalculated.
        """

        self.updated = True


class StateNode(ABC, ModelNode):
    """
    Model leaf nodes that hold some sort of data that calculation nodes use
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def update(self, *args, **kwargs):
        pass


class NetworkNode(ABC, ModelNode):
    """
    Class that handles common functionality of all network nodes
    and all the height/branch length hookups.
    """

    def __init__(self, branch=None):
        super(NetworkNode, self).__init__()
        self.branches = branch
        self.parents = None
        self.children = None

    def get_branches(self):
        if self.branches is None:
            self.branches = []
            for child in self.get_model_parents():
                if type(child) is BranchLengthNode: # or type(child) is SNPBranchNode:
                    self.branches.append(child)
        return self.branches

    def add_successor(self, model_node):
        """
        Adds a successor to this node.

        Input: model_node (type ModelNode)
        """
        if self.successors is None:
            self.successors = [model_node]
        else:
            self.successors.append(model_node)

        if type(model_node) is FelsensteinInternalNode or type(model_node) is SNPInternalNode:
            if self.parents is None:
                self.parents = [model_node]
            else:
                self.parents.append(model_node)

    def remove_successor(self, model_node):
        """
        Removes a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if model_node in self.successors:
            self.successors.remove(model_node)
            if self.parents is not None and model_node in self.parents:
                self.parents.remove(model_node)
                if len(self.parents) == 0:
                    self.parents = None

    def add_predecessor(self, model_node):
        """
        Adds a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if self.predecessors is None:
            self.predecessors = [model_node]
        else:
            self.predecessors.append(model_node)

        if type(model_node) is FelsensteinInternalNode or type(model_node) is FelsensteinLeafNode:
        #\
                #or type(model_node) is SNPInternalNode or type(model_node) is SNPLeafNode:
            if self.children is None:
                self.children = [model_node]
            else:
                self.children.append(model_node)

    def remove_predecessor(self, model_node):
        """
        Removes a predecessor to this node.

        Input: model_node (type ModelNode)
        """
        if model_node in self.predecessors:
            self.predecessors.remove(model_node)
            if self.children is not None:
                if model_node in self.children:
                    self.children.remove(model_node)

    @abstractmethod
    def node_move_bounds(self):
        pass

    def get_parent(self, return_all = False):
        if return_all:
            return self.parents
        else:
            if self.parents is not None:
                return self.parents[0]
            else:
                return None

    def get_children(self):
        return self.children

    
class BranchNode(ABC, ModelNode):
    def __init__(self, vector_index: int, branch_length: float) -> None:
        super().__init__()
        self.index : int = vector_index
        self.branch_length : float = branch_length
        self.net_parent : NetworkNode = None
        self.gamma : float = None

    def switch_index(self, new_index:int):
        """
        Change the lookup index of this branch in the TreeHeight node

        Args:
            new_index (int): a new index
        """
        self.index = new_index

    def get_index(self):
        """
        Returns:
            int: The index into the TreeHeight vector
        """
        return self.index

    def set_net_parent(self, parent: NetworkNode):
        """
        Set the network parent 

        Args:
            parent (NetworkNode): the node that this branch points to
        """
        self.net_parent = parent
        

    def inheritance_probability(self)->float:
        """
        Return the gamma rate/ inheritance probability for a branch stemming from a hybridization node

        Returns:
            float: A number from [0,1]
        """
        if self.gamma is None:
            raise ModelError("An inheritance probability is not available for this node")
        return self.gamma
    
    def set_inheritance_probability(self, new_gamma : float)->None:
        self.gamma = new_gamma
        print("SETTING INHERITANCE PROBABILITY: " + str(self.gamma) + " FOR BRANCH INDEX : " + str(self.index))
    
    


class BranchLengthNode(BranchNode, CalculationNode):
    """
    A calculation node that uses the substitution model to calculate the
    transition matrix Pij
    """

    def __init__(self, vector_index: int, branch_length: float):
        super().__init__(vector_index, branch_length)
        self.sub = None
        self.updated_sub = True
        self.as_height = True

    def update(self, new_bl: float):
        # update the branch length
        self.branch_length = new_bl

        # Mark this node and any nodes upstream as needing to be recalculated
        self.upstream()

    def update_sub(self, new_sub: GTR):
        self.sub = new_sub
        self.updated_sub = False
        self.upstream()

    def get(self):
        if self.updated:
            return self.calc()
        else:
            return self.cached

    def calc(self):
        # mark node as having been recalculated and cache the result
        self.cached = self.branch_length
        self.updated = False

        # return calculation
        return self.branch_length

    

    def transition(self):
        """
        Calculate the Pij matrix
        """
        if self.as_height:
            try:
                node = self.get_model_children()[0]
                if node.get_parent() is None:
                    branch_len = 0
                else:
                    #Only trees, grab [0]
                    parent_height = node.get_parent().get_branches()[0].get()
                    branch_len = parent_height - self.branch_length
            finally:
                pass
        else:
            branch_len = self.branch_length

        if self.updated_sub:
            # grab current substitution model
            for child in self.get_model_parents():
                if type(child) is SubstitutionModel:
                    self.sub = child.get_submodel()
                    self.updated_sub = False
                    return child.get().expt(branch_len)
        else:
            # TODO: cache this?
            return self.sub.expt(branch_len)


class TreeHeights(StateNode):
    """
    State node that holds the node heights/branch lengths
    """

    def __init__(self, node_height_vec=None):
        super().__init__()
        self.heights = node_height_vec

    def update(self, new_vector: list):

        # Only update the parts of the vector that have changed
        if self.heights is None:
            self.heights = new_vector
            for branch_node in self.get_model_children():
                branch_node.update(self.heights[branch_node.get_index()])
        else:
            for branch_node in self.get_model_children():
                if new_vector[branch_node.get_index()] != self.heights[branch_node.get_index()]:
                    branch_node.update(new_vector[branch_node.get_index()])

            self.heights = new_vector

    def singular_update(self, index: int, value: float):
        for branch_node in self.get_model_children():
            if branch_node.get_index() == index:
                branch_node.update(value)

    def get_heights(self):
        return self.heights


class SubstitutionModelParams(StateNode):

    def __init__(self, freq: np.array, trans: np.array) -> None:
        super().__init__()
        self.base_freqs = freq
        self.transitions = trans

    def update(self, new_freqs=None, new_trans=None):

        # should only have the one parent
        submodel_node = self.get_model_children()[0]

        if new_freqs is None and new_trans is None:
            raise ModelError("Nonsensical update")
        elif new_freqs is not None and new_trans is not None:
            submodel_node.update(self.new_submodel(new_freqs, new_trans))
        elif new_freqs is not None:
            submodel_node.update(self.new_submodel(new_freqs))
        else:
            submodel_node.update(self.new_submodel(new_trans=new_trans))

    def new_submodel(self, new_freqs: np.array = None, new_trans: np.array = None):
        """
        Given a change in transitions and/or base_frequencies, determines the proper subclass of GTR
        to return
        """
        if new_freqs is None:
            proposed_freqs = self.base_freqs
        else:
            proposed_freqs = new_freqs

        if new_trans is None:
            proposed_trans = self.transitions
        else:
            proposed_trans = new_trans

        # At least check if we can expedite the expt calculation
        if proposed_freqs == np.array([.25, .25, .25, .25]) and proposed_trans == np.ones(6):
            return JC()
        elif proposed_freqs == np.array([.25, .25, .25, .25]) \
                and (proposed_trans[1] == proposed_trans[4]) \
                and (proposed_trans[0] == proposed_trans[2] == proposed_trans[3] == proposed_trans[5]) \
                and (proposed_trans[0] + proposed_trans[1] == 1):
            return K2P(proposed_trans[0], proposed_trans[1])
        else:
            return GTR(proposed_freqs, proposed_trans)


class SubstitutionModel(CalculationNode):

    def __init__(self, submodel: GTR):
        super().__init__()
        self.sub = submodel

    def update(self, new_sub_model: GTR):
        # Set the new parameters
        self.sub = new_sub_model
        # Mark this node and any nodes upstream as needing to be recalculated
        self.upstream()

    def get(self):
        if self.updated:
            return self.calc()
        else:
            return self.cached

    def calc(self):
        self.updated = False
        self.cached = self.sub
        return self.sub

    def get_submodel(self):
        return self.sub


class ExtantSpecies(StateNode):

    def __init__(self, name: str, sequence: list):
        super().__init__()
        self.name = name
        #print("ADDING SEQ TO " + self.name + " : " + str(sequence))
        self.seq = sequence

    def update(self, new_sequence: list, new_name: str):
        # should only have a single leaf calc node as the parent
        self.seq = new_sequence
        self.name = new_name
        self.get_model_children()[0].update(new_sequence, new_name)

    def seq_len(self):
        if type(self.seq) is list:
            return len(self.seq[0].get_seq())
        else:
            return len(self.seq)

    def get_seq(self):
        return self.seq

    def get_name(self):
        return self.name


class FelsensteinLeafNode(NetworkNode, CalculationNode):

    def __init__(self, partials=None, branch=None, name: str = None):
        super(FelsensteinLeafNode, self).__init__(branch=branch)
        super(CalculationNode).__init__()
        self.matrix = partials
        self.name = name

    def node_move_bounds(self):
        """
        For a leaf node at a given height, it's height may be legally changed within a certain bounds.
        The height of the node may go as low as the closest parent's height.
        The height of the node may go as high as its current height

        Returns: interval (low, hi) that gives the parameters for a uniform selection to be made for a new node height.
        """
        return [0, self.parents[0].get_branches()[0].get()]

    def update(self, new_partials, new_name):
        self.matrix = vec_bin_array(new_partials, 4)
        self.name = new_name
        self.upstream()

    def get(self):
        if self.updated:
            return self.calc()
        else:
            return self.cached

    def calc(self):
        # mark node as having been recalculated and cache the result
        if self.matrix is None:
            for child in self.get_model_parents():
                if type(child) is ExtantSpecies:
                    self.matrix = vec_bin_array(child.get_seq(), 4)

        self.cached = self.matrix
        self.updated = False

        # return calculation
        return self.matrix

    def get_name(self):
        return self.name


class FelsensteinInternalNode(NetworkNode, CalculationNode):
    def __init__(self, branch=None, name: str = None):
        super(FelsensteinInternalNode, self).__init__(branch=branch)
        super(CalculationNode).__init__()
        self.partials = None
        self.name = name

    def node_move_bounds(self):
        """
        For an internal node at a given height, it's height may be legally changed within a certain bounds.
        The height of the node may go as low as the closest parent's height.
        The height of the node may go as high as the closest child's height

        Returns: interval (low, hi) that gives the parameters for a uniform selection to be made for a new node height.
        """
        # Node can go from its current height up towards parent
        if self.parents is None:
            # root node
            raise ModelError("NODE BOUNDS FUNCTION UNDEFINED FOR ROOT.")
        # Normal internal node
        # TODO: ADAPT FOR NETWORKS, MAX PARENT HEIGHTS
        lower_limit = self.parents[0].get_branches()[0].get()

        # Upper limit is defined by the closest (in height) child to the root, which is going to be max(child heights)
        upper_limit = max(0, max([child.get_branches()[0].get() for child in self.children]))
        return [lower_limit, upper_limit]

    def update(self):
        self.upstream()

    def get(self):
        if self.updated:
            return self.calc()
        else:
            return self.cached

    def calc(self):

        children = self.get_model_parents()

        matrices = []

        for child in children:
            # type check
            if type(child) != FelsensteinInternalNode and type(child) != FelsensteinLeafNode:
                continue

            # get the child partial likelihood. Could be another internal node, but could be a leaf
            matrix = child.get()

            # compute matrix * Pij transpose
            step1 = np.matmul(matrix, child.get_branches()[0].transition().transpose())

            # add to list of child matrices
            matrices.append(step1)

        # Element-wise multiply each matrix in the list
        result = np.ones(np.shape(matrices[0]))
        for matrix in matrices:
            result = np.multiply(result, matrix)
        self.partials = result

        # mark node as having been recalculated and cache the result
        self.cached = self.partials
        self.updated = False

        # return calculation
        return self.partials

    def get_name(self):
        return self.name


class SNPLeafNode(NetworkNode, CalculationNode):
    """
    SNPLeafNode is a node that holds all group member's SeqRecord objs,
    for reference when its branch tries to calculate partial likelihoods based on the number of red alleles

    self.sequences is a list of SeqRecord objs
    self.name is a string
    """

    def __init__(self, partials=None, branch=None, name: str = None):
        super(SNPLeafNode, self).__init__(branch=branch)
        self.sequences = partials
        self.name = name
        self.leaf_descendants:set = set()
        self.total_samples : int = None

    def node_move_bounds(self):
        branches = []
        for par in self.parents:
            for branch in par.get_branches():
                branches.append(branch)
        return [0, min([branch.get_length() for branch in branches])]

    def update(self, new_sequences: list, new_name: str):
        self.sequences = new_sequences
        self.name = new_name
        self.upstream()

    def get(self):
        if self.updated:
            return self.calc()
        else:
            return self.cached

    def samples(self):
        if self.total_samples is None:
            self.total_samples = sum([rec.ploidy() for rec in self.sequences]) 
        return self.total_samples

    def possible_lineages(self):
        return self.samples()

    def seq_len(self):
        for child in self.predecessors:
            if type(child) is ExtantSpecies:
                return child.seq_len()
        raise ModelError("SNP Leaf Node does not have an ExtantSpecies node as a child")

    def red_count(self):
        tot = np.zeros(len(self.sequences[0].get_seq()))
        for seq_rec in self.sequences:
            tot = np.add(tot, np.array(seq_rec.get_numerical_seq()))
        return tot

    def calc(self):
        self.cached = [branch.get() for branch in self.get_branches()]
        self.updated = False
        return self.cached

    def get_name(self):
        return self.name


class SNPInternalNode(NetworkNode, CalculationNode):

    def __init__(self, site_count: int, branch=None, name: str = None):
        super(SNPInternalNode, self).__init__(branch=branch)
        self.partials = None
        self.name = name
        self.site_count = site_count
        self.leaf_descendants : set = set()

    def node_move_bounds(self):
        if self.parents is None:
            # root node
            return None
        # Normal internal node
        par_branches = []
        for par in self.parents:
            for branch in par.get_branches():
                par_branches.append(branch)       
                
        child_branches = []
        for child in self.children:
            for branch in child.get_branches():
                child_branches.append(branch) 
             
        #TODO: this is wrong       
        upper_limit = min([branch.get() for branch in par_branches])
        lower_limit = max(0, max([branch.get() for branch in child_branches]))
        return [lower_limit, upper_limit]

    def update(self):
        self.upstream()
    
    def calc_leaf_descendants(self):
        """
        Calculate the leaves that are descendants of a lineage/node.
        
        Returns:
            leaf_descendants (set) : a set of node descendants
        """
        for child in self.get_children():
            if type(child) is SNPLeafNode:
                self.leaf_descendants.add(child)
            else:
                #The union of all its children's descendants
                self.leaf_descendants = self.leaf_descendants.union(child.calc_leaf_descendants())
        
        return self.leaf_descendants
        
        
    def get(self)->tuple:
        if self.updated:
            return self.calc()
        else:
            return self.cached

    def possible_lineages(self):
        """
        Calculate the number of lineages that flow through this node.
        For non-reticulation nodes, if branch x has children y,z:

        Returns:
            int: number of lineages
        """

        return sum([child.samples() for child in self.leaf_descendants])
    
    
    def get_branch_from_child(self, child, avoid_index=None):
        for branch in child.get_branches():
            if branch.net_parent == self:
                if avoid_index is None:
                    return branch
                else:
                    if branch.index == avoid_index:
                        continue
                    return branch
        raise ModelError("No branch found between input child and this node")
    

    def calc(self):
        """
        Return the likelihoods, or in the case of the root, return the model likelihood.
        """
        self.cached = [branch.get() for branch in self.get_branches()]
        self.updated = False
        return self.cached
    
    def is_reticulation(self):
        """
        Returns:
            bool: True if this node is a reticulation node 
                    (more than 1 parent), False otherwise
        """
        if self.parents is None:
            return False
        return len(self.get_parent(return_all = True)) > 1

    def get_name(self):
        return self.name


# class SNPBranchNode(BranchNode, CalculationNode):

#     def __init__(self, vector_index: int, branch_length: float, Q: SNPTransition, vpi_tracker: PartialLikelihoods, verbose = False):
#         #Note: the vector index also acts as a unique branch identifier
#         super().__init__(vector_index, branch_length)
#         self.Q = Q
#         self.Qt = None
#         self.verbose = verbose
#         self.branch_height = None
#         self.vpi_tracker = vpi_tracker
    

#     def update(self, new_bl: float)->None:
#         """
#         update the branch length of this branch
#         Args:
#             new_bl (float): the new branch length/height for this branch
#         """
#         # update the branch length
#         self.branch_height = new_bl
#         self.branch_length = new_bl

#         # Mark this node and any nodes upstream as needing to be recalculated
#         self.upstream()

#     def get(self)->tuple:
#         vpi_key = self.vpi_tracker.get_key_with(self.index)
#         if vpi_key is None:
#             return self.calc()
#         else:
#             return vpi_key

#     def get_length(self):
#         return self.branch_length

#     def calc(self) -> tuple:
#         """
#         Calculates both the top and bottom partial likelihoods, based on Eq 14 and 19.

#         Returns a list of length 2, element [0] is the bottom likelihoods, element [1] is the top likelihoods
        
#         Calculated using eqs 12,14,16,19 from David Bryant, Remco Bouckaert, Joseph Felsenstein, Noah A. Rosenberg, Arindam RoyChoudhury, 
#         Inferring Species Trees Directly from Biallelic Genetic Markers: Bypassing Gene Trees in a Full Coalescent Analysis, Molecular Biology and 
#         Evolution, Volume 29, Issue 8, August 2012, Pages 1917-1932, https://doi.org/10.1093/molbev/mss086
        
#         Also, Rule 3,4 for networks Rabier CE, Berry V, Stoltz M, Santos JD, Wang W, et al. 
#         (2021) On the inference of complex phylogenetic networks by Markov Chain Monte-Carlo. 
#         PLOS Computational Biology 17(9): e1008380. https://doi.org/10.1371/journal.pcbi.1008380
#         """
        
#         #Get the network node parent of this branch object
#         node_par = self.get_model_children()[0]
        
#         #Calculate Q^t before calculating likelihoods
#         self.transition()
        
#         if type(node_par) is SNPLeafNode:
#             site_count = node_par.seq_len()
#         elif type(node_par) is SNPInternalNode:
#             site_count = node_par.site_count
#         else:
#             raise ModelError("site count error")

#         vector_len = partials_index(node_par.possible_lineages() + 1)  

#         # BOTTOM: Case 1, the branch is an external branch, so bottom likelihood is just the red counts
#         if type(node_par) is SNPLeafNode:
#             F_key = self.vpi_tracker.Rule0(node_par.red_count(), node_par.samples(), site_count, vector_len, self.index)  
            
#         # BOTTOM: Case 2, the branch is for an internal node, so bottom likelihoods need to be computed based on child tops
#         else:
#             # EQ 19
#             # Get the top likelihoods of each of the child branches
#             net_children = node_par.get_children()
            
#             if node_par.is_reticulation():
#                 #RULE 3
#                 x_branch = node_par.get_branch_from_child(net_children[0])
#                 F_t_x_key = x_branch.get()
                
#                 possible_lineages = node_par.possible_lineages() 
                
#                 #Get the other branch
#                 sibling_branches = node_par.get_branches()
#                 if sibling_branches[0] == self:
#                     sibling_branch : BranchNode = sibling_branches[1]
#                 else:
#                     sibling_branch : BranchNode = sibling_branches[0]
                
#                 g_this = self.inheritance_probability()
#                 g_that = sibling_branch.inheritance_probability()
                
                
#                 if g_this + g_that != 1:
#                     raise ModelError("Set of inheritance probabilities do not sum to 1 for node<" + node_par.name + ">")
                
#                 F_b_key = self.vpi_tracker.Rule3(F_t_x_key, vector_len, g_this, g_that, site_count, possible_lineages, x_branch.index, self.index, sibling_branch.index)
                
#                 #Do the calculations for the sibling branch
                
#                 sibling_branch.transition()
#                 F_t_key_sibling = self.vpi_tracker.Rule1(F_b_key, site_count, vector_len, node_par.possible_lineages(), sibling_branch.Qt, sibling_branch.index)
                
#                 sibling_branch.updated = False
#                 F_key = F_t_key_sibling
             
#             elif len(node_par.children) == 2:
                
#                 y_branch : SNPBranchNode = node_par.get_branch_from_child(net_children[0])
#                 F_t_y_key = y_branch.get()
#                 y_branch_index = y_branch.index
                
#                 z_branch : SNPBranchNode = node_par.get_branch_from_child(net_children[1], avoid_index=y_branch_index)
#                 F_t_z_key = z_branch.get()
#                 z_branch_index = z_branch.index
                
#                 #Find out whether lineage y and z have leaves in common 
#                 if not net_children[1].leaf_descendants.isdisjoint(net_children[0].leaf_descendants): #If two sets are not disjoint
#                     print("Y BRANCH INDEX: " + str(y_branch_index))
#                     print("Z BRANCH INDEX: " + str(z_branch_index))
#                     F_b_key = self.vpi_tracker.Rule4(F_t_z_key, site_count, vector_len, y_branch_index, z_branch_index, self.index)
#                 else: # Then use Rule 2
#                     F_b_key = self.vpi_tracker.Rule2(F_t_y_key, F_t_z_key, site_count, vector_len, y_branch_index, z_branch_index, self.index)
#                     #raise ModelError("temp catch")
#                 F_key = F_b_key
#             else:
#                 #A node should only have one child if it is the root node. simply pass along the vpi
#                 F_key = node_par.get_branch_from_child(net_children[0]).get()
                    
#         # TOP: Compute the top likelihoods based on the bottom likelihoods w/ eq 14&16
#         if node_par.parents is not None:
#             F_key = self.vpi_tracker.Rule1(F_key, site_count, vector_len, node_par.possible_lineages(), self.Qt, self.index)
#             self.updated = False
#         else:
#             self.updated = False
    
#         # print("F_T (at site 0)")
#         # print(F_key)
#         # print(self.vpi_tracker.vpis[F_key][0])
        
#         return F_key

#     def transition(self):
#         """
#         Calculate exp(Q^branch_len)
        
#         This function may only be called after making the adjustment to treating
#         everything as a height!!
#         """
#         node_par = self.successors[0]
        
#         if node_par.parents is None or len(node_par.parents) == 0:
#             #Root branch.
#             return
#         else:
#             #any branch should return the correct parent height
#             parent_height = self.net_parent.get_branches()[0].get_length()
#             branch_len = parent_height - self.branch_height
#             self.Qt = self.Q.expt(branch_len)
        
        
class NetworkContainer(StateNode):
    def __init__(self, network : DAG):
        super().__init__()
        self.network : DAG = network
    
    def update(self, new_net : DAG):
        self.network = new_net
        model_parents : list[CalculationNode] = self.get_model_children()
        for model_parent in model_parents:
            model_parent.upstream()
        
    def get(self) -> DAG:
        return self.network

