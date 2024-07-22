from ModelGraph import Model
from ModelFactory import *
from Network import Network, Edge, Node
from MCMC_BiMarkers import *
from __future__ import annotations

            
    
    
class SNPBranch:
    
    def __init__(self, 
                 src : BiMarkersNode, 
                 dest : BiMarkersNode, 
                 branch_length : float = None, 
                 gamma : float = None) -> None:
        
        self.top : BiMarkersNode = src
        self.bot : BiMarkersNode = dest
        self.gamma : float = gamma
        self.length : float = branch_length
        self.id : str = "(" + self.top.get_name() + " -> " + self.bot.get_name() + ")"
    
    def apply(self):
        """
        Calculates both the top and bottom partial likelihoods, 
        based on Eq 14 and 19.

        Returns a list of length 2, element [0] is the bottom likelihoods, 
        element [1] is the top likelihoods.
        
        Calculated using eqs 12,14,16,19 from David Bryant, Remco Bouckaert, 
        Joseph Felsenstein, Noah A. Rosenberg, Arindam RoyChoudhury, 
        Inferring Species Trees Directly from Biallelic Genetic Markers: 
        Bypassing Gene Trees in a Full Coalescent Analysis, Molecular Biology 
        and Evolution, Volume 29, Issue 8, August 2012, Pages 1917-1932, 
        https://doi.org/10.1093/molbev/mss086
        
        Also, Rule 3,4 for networks Rabier CE, Berry V, Stoltz M, Santos JD, 
        Wang W, et al. (2021) On the inference of complex phylogenetic networks 
        by Markov Chain Monte-Carlo. PLOS Computational Biology 17(9): e1008380.
        https://doi.org/10.1371/journal.pcbi.1008380
        """
        ###############
        #### SETUP ####
        ###############
        
        # Grab vpi tracker
        vpi_acc : VPIAccumulator = self.bot.get_model_children(VPIAccumulator)[0]
        pl : PartialLikelihoods = vpi_acc.data
        
        # Grab site count
        sitect_param : SiteParameter = self.bot.get_model_children(SiteParameter)[0]
        site_count = sitect_param.get_value()
        
        # Grab vector length at this node
        vector_len = n_to_index(self.bot.possible_lineages() + 1)  
        
        # ID each branch
        # par_branches : dict[BiMarkersNode, list[str]] = {}
        # parents : list[BiMarkersNode] = []
        # for par in self.get_model_parents(BiMarkersNode):
        #     branch_id = "<" + str(par.__hash__) + ", " \
        #                     + str(self.__hash__) + ">"
        #     par_branches[par] = branch_id
        #     parents.append(par)
        
        # child_branches : dict[BiMarkersNode, list[str]] = {}
        # children : list[BiMarkersNode] = {}
        # for child in self.get_model_children(BiMarkersNode):
        #     branch_id = "<" + str(self.__hash__) + ", " \
        #                     + str(child.__hash__) + ">"
        #     child_branches[child] = branch_id
        #     children.append(child) 
        
        

        ###########################################################
        #### IF THIS NODE IS A LEAF, THERE IS ONLY ONE BRANCH, ####
        #### AND ME MUST APPLY RULE 0. ############################
        ###########################################################
        if len(self.bot.get_children()) == 0: #TODO: Potential issue here
            F_key = pl.Rule0(self.red_count(), 
                             self.samples(),
                             site_count, 
                             vector_len, 
                             self.id) 
                
        #### Case 2, the branch is for an internal node, so bottom 
        # likelihoods need to be computed based on child tops
        else:
            
            in_branches = self.bot.get_in_branches()
            out_branches = self.bot.get_out_branches()
            
            # EQ 19
            if len(in_branches) == 2 and len(out_branches) == 1:
                #RULE 3
                y : SNPBranch = in_branches[0]
                z : SNPBranch = in_branches[1]
                x : SNPBranch = out_branches[0]
                
                x_id = x.id
                y_id = y.id
                z_id = z.id
                F_t_x_key = pl.get_key_with(x_id)
                
                possible_lineages = self.bot.possible_lineages() #bot or top? pretty sure bot
        
                
                g_this = y.gamma
                g_that = z.gamma
                
                if g_this + g_that != 1:
                    raise ModelError("Set of inheritance probabilities do not \
                        sum to 1 for node<" + self.bot.get_name() + ">")
                
                F_b_key = pl.Rule3(F_t_x_key, 
                                   x_id,
                                   y_id,
                                   z_id,
                                   g_this,
                                   g_that,
                                   possible_lineages)
                                
                #Do the rule 1 calculations for the sibling branch
                q = self.bot.get_model_children(BiMarkersTransitionMatrixNode)
                Q : BiMarkersTransitionMatrixNode = q[0]
                QT = Q.get()
                z_qt = QT.expt(self.length)
                
                F_t_key_sibling = pl.Rule1(F_b_key, 
                                           z_id, 
                                           z.bot.possible_lineages(),
                                           z_qt)
                
                F_key = F_t_key_sibling
            elif len(in_branches) == 1 and len(out_branches) == 2:
                #Either Rule 2 or Rule 4
                
                y_branch : SNPBranch = out_branches[0]
                F_t_y_key = y_branch.bot.get()
                y_branch_index = y_branch.id
                
                z_branch : SNPBranch = out_branches[1]
                F_t_z_key = z_branch.bot.get()
                z_branch_index = z_branch.id
                
                #Find out whether lineage y and z have leaves in common 
                if not net_children[1].leaf_descendants.isdisjoint(net_children[0].leaf_descendants): #If two sets are not disjoint
                    print("Y BRANCH INDEX: " + str(y_branch_index))
                    print("Z BRANCH INDEX: " + str(z_branch_index))
                    F_b_key = self.vpi_tracker.Rule4(F_t_z_key, site_count, vector_len, y_branch_index, z_branch_index, self.index)
                else: # Then use Rule 2
                    F_b_key = self.vpi_tracker.Rule2(F_t_y_key, F_t_z_key, site_count, vector_len, y_branch_index, z_branch_index, self.index)
                    #raise ModelError("temp catch")
                F_key = F_b_key
            else:
                #A node should only have one child if it is the root node. simply pass along the vpi
                F_key = node_par.get_branch_from_child(net_children[0]).get()
                    
        # TOP: Compute the top likelihoods based on the bottom likelihoods w/ eq 14&16
        if node_par.parents is not None:
            F_key = self.vpi_tracker.Rule1(F_key, site_count, vector_len, node_par.possible_lineages(), self.Qt, self.index)
            self.updated = False
        else:
            self.updated = False
    
        # print("F_T (at site 0)")
        # print(F_key)
        # print(self.vpi_tracker.vpis[F_key][0])
        
        return F_key
    
    
class BiMarkersNode(ANetworkNode):
    def __init__(self, 
                 in_branches : list[SNPBranch],
                 out_branches : list[SNPBranch],
                 name : str = None, 
                 node_type : str = None) -> None:
       
        super().__init__(name, node_type)
        self.in_branches : list[SNPBranch] = in_branches #List of "in edges"
        self.out_branches : list[SNPBranch] = out_branches
    
    def get_in_branches(self):
        return self.in_branches
    
    def get_out_branches(self):
        return self.out_branches
        
    def red_count(self) -> np.ndarray:
        """
        Only defined for leaf nodes, this method returns the count of red 
        alleles for each site for the associated species. The dimension of the
        resulting array will be (sequence count, sequence length)
        where sequence count is the number of data sequences associated with
        the species, and the sequence length is simply the length of each 
        sequence (these will all be the same).

        Returns:
            np.ndarray: An array that describes the red allele counts per 
                        sequence and per site.
        """
        if len(self.get_children()) == 0:
            spec : ExtantSpecies = self.get_model_children(ExtantSpecies)[0]
            seqs : list[SeqRecord] = spec.get_seqs()
            
            tot = np.zeros(len(seqs[0].get_seq()))
            for seq_rec in seqs:
                tot = np.add(tot, np.array(seq_rec.get_numerical_seq()))
            
            return tot
        else:
            warnings.warn("Red count method called on a network node that has\
                           network children. Needs to be called on a leaf!")
        
    def calc(self):
        # Grab vpi tracker
        vpi_acc : VPIAccumulator = self.get_model_children(VPIAccumulator)[0]
        pl : PartialLikelihoods = vpi_acc.data
        
        # Grab site count
        sitect_param : SiteParameter = self.get_model_children(SiteParameter)[0]
        site_count = sitect_param.get_value()
        
        # Grab vector length at this node
        vector_len = n_to_index(self.possible_lineages() + 1)  
        
        vpi_keys : list[str] = []
        for branch in self.branches:
            vpi_keys.append(branch.apply(pl, site_count, vector_len))
        
        return self.cache(vpi_keys)
    
            
    def calc_leaf_descendants(self) -> set[Node]:
        """
        Calculate the leaves that are descendants of a lineage/node.
        
        Returns:
            leaf_descendants (set) : a set of node descendants
        """
        for child in self.get_children():
            if len(child.get_children()) == 0:
                self.leaf_descendants.add(child)
            else:
                #The union of all its children's descendants
                child_desc = child.calc_leaf_descendants()
                self.leaf_descendants = self.leaf_descendants.union(child_desc)
        
        return self.leaf_descendants
        
    def get(self) -> tuple:
        if self.dirty:
            return self.calc()
        else:
            return self.cached

    def possible_lineages(self) -> int:
        """
        Calculate the number of lineages that flow through this node.
        For non-reticulation nodes, if branch x has children y,z:

        Returns:
            int: number of lineages
        """
        if len(self.get_children()) == 0:
            return self.samples()
        else:
            return sum([child.samples() for child in self.leaf_descendants])
    
    def samples(self)->int:
        if len(self.get_children()) == 0:
            seqs = self.get_model_children(ExtantSpecies)[0].get_seqs()
            return sum([rec.ploidy() for rec in seqs]) 
        else:
            Warning("Calling samples method on a node that is not a leaf")
    