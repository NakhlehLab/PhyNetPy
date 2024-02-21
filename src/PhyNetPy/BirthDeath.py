""" 
Author : Mark Kessler
Last Stable Edit : 2/20/24
First Included in Version : 1.0.0
Approved for Release: YES
"""


import random
import numpy as np
import scipy
from Node import Node
from Graph import DAG
from math import log, exp


class BirthDeathSimError(Exception):
    """
    This exception is thrown whenever something irrecoverably wrong happens
    during the process of generating a network within the Yule or CBDP algorithm
    """

    def __init__(self, message = "Something went wrong simulating a network"):
        self.message = message
        super().__init__(self.message)


def random_species_selection(nodes : list[Node], 
                             rng : np.random.Generator) -> Node:
    """
    Returns a random live Node from an array/set. The Node returned
    will be operated on during a birth or death event

    Args:
        nodes (list[Node]): a list of live lineages 
                            (aka Nodes with an attribute "live" mapped to True)
        rng (np.random.Generator): the result of a np.random.default_rng call
                            (which should have been initialized 
                            with a random int seed) 

    Returns:
        Node: The randomly selected node to operate on
    """
    liveNodes = live_species(nodes)
    
    #use the rng object to select an index
    randomInt = rng.integers(0, len(liveNodes))
    
    return liveNodes[randomInt]


def live_species(nodes : list[Node]) -> list[Node]:
    """
    Returns a subset of Nodes that represent live lineages. 
    A Node represents a live lineage if it has an attribute "live" set to True.

    nodes (list[Node]) -- an array of Node objects
    """
    return [node for node in nodes if node.attribute_value("live") is True]


class Yule:
    """
    The Yule class represents a pure birth model for simulating
    networks of a fixed (n) amount of extant taxa.

    gamma -- the birth rate. A larger birth rate will result in shorter 
            branch lengths and a younger network. 
            Value should be a non-negative real number.
    
    n -- number of extant taxa at the end of simulation

    time -- if conditioning on time, the age of the tree to be simulated
    
    rng -- numpy random number generator for drawing speciation times
    """

    def __init__(self, gamma : float, n : int = None, time : float = None, 
                 rng : np.random.Generator = None) -> None:

        # set birth rate
        self.set_gamma(gamma)
        
        # goal number of taxa/time
        if n is not None:
            self.set_taxa(n)
        else:
            if time is None:
                raise BirthDeathSimError("If you do not provide a value for the\
                                         number of taxa, please provide a time \
                                         constraint for network simulation")
            self.set_time(time)

        # current number of live lineages, always starts at 2
        self.lin : int = 2

        # helper var for labeling internal nodes
        self.internal_count : int = 1

        # amount of time elapsed during the simulation of a tree
        self.elapsed_time : float = 0

        # a list of trees generated under this model
        self.generated_trees : list[DAG] = []
        
        self.rng : np.random.Generator = rng

    def set_time(self, value : float)->None:
        """
        Set simulated network age

        Args:
            value (float): The age of any future simulated trees
        """
        self.condition = "T"
        self.time = value
        
    def set_taxa(self, value : int)->None:  
        """
        Set simulated tree taxa count

        Args:
            value (int): an integer >= 2

        Raises:
            BirthDeathSimError: if value < 2
        """
        self.condition = "N"
        self.N = value
        if value < 2:
            raise BirthDeathSimError("Please use a value >= 2")
    
    def set_gamma(self, value : float)-> None:
        """
        Setter for the gamma (birth rate) parameter

        Args:
            value (float): the new birth rate
        """
        if value <= 0:
            raise BirthDeathSimError("Birth rate must be > 0")
        self.gamma = value
        
    def draw_waiting_time(self) -> float:
        """
        Draw a waiting time until the next speciation event from 
        a memory-less exponential distribution.

        Since each lineage is equally likely for each event 
        under the Yule Model, the waiting time is given by the parameter 
        numlineages * birthRate or .lin * .gamma
        """
        scale = 1 / (self.lin * self.gamma)
        random_float_01 = self.rng.random()
        return scipy.stats.expon.ppf(random_float_01, scale = scale) 

    def event(self, nodes : list[Node], edges : list[list[Node]])-> int:
        """
        A speciation event occurs. Selects a random living lineage.

        Then add an internal "dead" node with: 
            branch length := t_parent + drawnWaitTime
        
        Set the parent to the chosen node as that internal node.

        nodes-- an array of nodes 

        edges-- an array of 2-tuples (as arrays) 

        WARNING: MUTATES THE PARAMETERS nodes AND edges!
        
        Returns: 0 if success, -1 if no more events can happen.      
        """

        # select random live lineage to branch from
        spec_node = random_species_selection(nodes, self.rng)

        # keep track of the old parent, we need to disconnect edges
        old_parent = spec_node.get_parent()

        # calculate the branch length to the internal node
        next_time = self.draw_waiting_time()
        branch_len = 0
        parent_time = spec_node.get_parent().attribute_value("t")
        
        if self.condition == "N":
            branch_len = self.elapsed_time + next_time - parent_time
            self.elapsed_time += next_time
        elif self.condition == "T": 
            if self.elapsed_time + next_time <= self.time:
                branch_len = self.elapsed_time + next_time - parent_time
                self.elapsed_time += next_time
            else: 
                return -1

        # create the new internal node
        new_internal = Node({spec_node.get_parent() : [branch_len]}, 
                            parent_nodes=[spec_node.get_parent()], 
                            attr={"t": self.elapsed_time, "live": False},
                           name="internal" + str(self.internal_count))
        
        self.internal_count += 1

        # set the extant species parent to be its direct ancestor
        spec_node.set_parent([new_internal])

        # there's a new live lineage
        self.lin += 1
        new_label = "spec" + str(self.lin)

        # create the node for the new extent species
        new_spec_node = Node(parent_nodes=[new_internal], 
                             attr={"live": True}, 
                             name=new_label)

        # add the newly created nodes
        nodes.append(new_spec_node)
        nodes.append(new_internal)

        # add the newly created branches 
        edges.append([new_internal, new_spec_node])
        edges.append([new_internal, spec_node])
        edges.append([old_parent, new_internal])
        
        # and remove the old connection (oldParent)->(specNode)
        edges.remove([old_parent, spec_node])

        return 0

    def generate_tree(self) -> DAG:
        """
        Simulate one tree. Starts with a root and 2 living lineages
        and then continuously runs speciation (in this case birth only) 
        events until there are exactly N live species.

        After the nth event, draw one more time and fill out the remaining
        branch lengths.

        Returns:
            DAG: The simulated tree
        """

        # Set up the tree with 2 living lineages and an "internal" root node
        node1 = Node(attr={"t": 0, "label": "root", "live": False}, name="root")
        node2 = Node(parent_nodes=[node1], attr={"live": True}, name="spec1")
        node3 = Node(parent_nodes=[node1], attr={"live": True}, name="spec2")

        nodes = [node1, node2, node3]
        edges = [[node1, node2], [node1, node3]]
        
        if self.condition == "N" and self.N == 2:
            return DAG(nodes=nodes, edges=edges)
            

        # until the tree contains N extant taxa, keep having speciation events
        if self.condition == "N":
            if self.N == 2:
                return DAG(nodes=nodes, edges=edges)
            
            #create N lineages
            while len(live_species(nodes)) < self.N:
                self.event(nodes, edges)

            # populate remaining branches with branch lengths according to
            # Eq 5.1? Just taking sigma_n for now
            next_time = self.draw_waiting_time()

            for node in live_species(nodes):
                node.add_attribute("t", self.elapsed_time + next_time)
                if len(node.get_parent(True)) != 0:
                    parent_time = node.get_parent().attribute_value("t")
                    final_time = self.elapsed_time + next_time - parent_time
                    node.set_length(final_time, node.get_parent())

            # return the simulated tree
            tree = DAG(nodes = nodes, edges = edges)

            # reset the elapsed time to 0, and the number of live branches to 2
            # for correctness generating future trees
            self.elapsed_time = 0
            self.lin = 2

        else:
            while self.elapsed_time < self.time:
                status = self.event(nodes, edges, "T")
                if status == -1:
                    break

            for node in live_species(nodes):
                # the live lineages are all leaves, 
                # and are thus all at the goal time
                node.add_attribute("t", self.time)
                
                #calculate branch lengths
                #TODO: Not sure why this is a conditional. no live lineage should not have a parent...
                if len(node.get_parent(True)) != 0:
                    parent_time = node.get_parent().attribute_value("t")
                    node.set_length(self.time - parent_time, node.get_parent())
                else:
                    node.set_length(0, node.get_parent())

            tree = DAG(nodes = nodes, edges = edges)

            # reset the elapsed time to 0, and the number of live branches to 2
            # for correctness generating future trees
            self.elapsed_time = 0
            self.lin = 2
        

        return tree

    def clear_generated(self) -> None:
        """
        Empty out the generated tree array
        """
        self.generated_trees = []

    def generate_trees(self, num_trees : int) -> list[DAG]:
        """
        Generate a set number of trees.

        num_trees-- number of trees to generate

        Outputs: the array of generated trees. 
                 (includes all that have been previously generated)
        """
        for dummy in range(num_trees):
            self.generated_trees.append(self.generate_tree())

        return self.generated_trees


class CBDP:
    """
    Constant Rate Birth Death Process Network simulation
    """

    def __init__(self, gamma : float, mu : float, n : int, sample : float = 1):
        """
        Create a new Constant Rate Birth Death Simulator.
        
        Args:
            gamma (float): birth rate
            mu (float): death rate
            n (int): number of taxa for simulated trees
            sample (float, optional): Sampling rate from (0, 1]. Defaults to 1.
        """
        
        self.set_sample(sample)
        self.set_bd(gamma, mu)
        self.set_n(n)
        
        self.generated_trees = []

    def set_bd(self, gamma : float, mu : float) -> None:
        """
        Set the birth and death rates for the model

        Args:
            gamma (float): The birth rate. Must be a positive real number, and 
            strictly greater than mu. 
            mu (float): The death rate. Must be a non-negative real number, less 
            than gamma. 
        """
        if gamma <= mu:
            BirthDeathSimError("Death rate is greater than or equal to the \
                                birth rate!")
        if gamma <=0:
            raise BirthDeathSimError("Please input a positive birth rate")
        if mu < 0:
            raise BirthDeathSimError("Please input a non-negative death rate")
        
        # Eq 15 from https://www.sciencedirect.com/science/article/pii/S0022519309003300#bib24
        self.gamma : float = gamma / self.sample
        self.mu : float = mu - gamma * (1 - (1 / self.sample))
        
        # probabilities of speciation or extinction event
        self.pBirth = self.gamma / (self.gamma + self.mu)
        self.pDeath = self.mu / (self.gamma + self.mu)
    
  
    def set_n(self, value : int) -> None:
        """
        Set the number of taxa for the simulated trees.

        Args:
            value (int): An integer value >= 2
        """
        if value < 2:
            raise BirthDeathSimError("Generated trees must have 2 or more taxa")
        self.N = value
    
    def set_sample(self, value : float) -> None:
        """
        Set the sampling rate. Must be a value in the interval (0,1].

        Args:
            value (float): a float between 0 (exclusive), and 1 (inclusive)

        Raises:
            BirthDeathSimError: if the provided value is not in the right range
        """
        # Ensure that the sampling rate is in the correct interval
        if 0 < value <= 1:
            self.sample : float = value
        else:
            raise BirthDeathSimError("Sampling rate must be drawn from (0,1]")
    
    def qinv(self, r:float) -> float:
        """
        Draw a time from the Qinv distribution from
        https://academic.oup.com/sysbio/article/59/4/465/1661436#app2

        r-- r[0] from the n-1 samples from [0,1]

        Returns: the time t, which is the age of a new simulated tree
        """
        term1 = (1 / self.gamma - self.mu)
        term2 = 1 - ((self.mu / self.gamma) * math.pow(r, 1 / self.N))
        term3 = 1 - math.pow(r, 1 / self.N)
        return term1 * math.log(term2 / term3)

    def finv(self, r:float, t:float) -> float:
        """   
        Draw a sample speciation time from the Finv distribution from
        https://academic.oup.com/sysbio/article/59/4/465/1661436#app2

        r-- r_i, from the sampled values from [0,1]
        t-- the age of the tree determined by Qinv(r[0])

        Returns: s_i from r_i
        """
        rate_dif = self.gamma - self.mu
        
        term1 = (1 / rate_dif)
        term2 = self.gamma - (self.mu * exp(-1 * t * rate_dif))
        term3 = r * (1 - exp(-1 * t * rate_dif))
        
        #finv equation
        log_term = log((term2 - self.mu * term3) / (term2 - self.gamma * term3))
        
        return term1 * log_term

    def generate_tree(self) -> DAG:
        """
        Simulate a single tree under the Constant Rate Birth Death Model.
        
        Follows the algorithm laid out by: 
        https://academic.oup.com/sysbio/article/59/4/465/1661436#app2
        (Hartmann, Wong, Stadler)

        Returns: A tree with n taxa chosen from the proper distributions.
        """

        # step 1
        r = [random.random() for _ in range(self.N)]

        # step 2
        t = self.qinv(r[0])

        # step 3
        s = {self.finv(r[i], t): (i + .5) for i in range(1, self.N)}

        # step 4 setup

        sKeys = list(s.keys())

        nodes = []
        edges = []

        # set up leaf nodes and internal nodes in proper order (fig 5)
        for j in range(2 * self.N - 1):
            if j % 2 == 0:
                # leaf node
                leaf = Node(attr={"t": 0}, name="T" + str(int(j / 2) + 1))
                nodes.append(leaf)
            else:
                internal = Node(attr={"t": sKeys[int((j - 1) / 2)]}, 
                                name="internal" + str(int((j - 1) / 2)))
                nodes.append(internal)

        # step 4
        for i in range(2 * self.N - 1):
            # for each node, connect it to the correct parent
            new_edge = self.connect(i, nodes)
            if new_edge is not None:
                edges.append(new_edge)

        # add edges and nodes to a tree
        tree = DAG(nodes = nodes, edges = edges)
        tree.generate_branch_lengths()

        return tree

    @staticmethod
    def connect(index : int, nodes : list[Node]) -> list[Node]:
        """
        nodes-- a list of nodes (list[i] is the ith node along a horizontal
        axis that alternates between species and internal s_i nodes/speciation events)

        index-- the node to connect to its parent in the tree

        Given the nodes and a node to connect, create a new edge.

        The parent node is defined to be the closest to nodes[index] in terms
        of time and proximity in the list. There are two candidates, the left and right 
        candidate. Each candidate is the nearest element in the list such that the time 
        attribute is larger than nodes[index]. The parent is the minimum of the 
        two candidates.

        Returns: the edge from nodes[index] to its correct parent

        """

        # find right candidate
        copy_index = index + 1
        right_candidate = None

        while copy_index < len(nodes):
            # search in the list to the right (ie increase the index)
            if nodes[copy_index].attribute_value("t") > nodes[index].attribute_value("t"):
                right_candidate = nodes[copy_index]
                break
            copy_index += 1

        # find left candidate
        copy_index = index - 1
        left_candidate = None
        while copy_index >= 0:
            # search in the left part of the list
            if nodes[copy_index].attribute_value("t") > nodes[index].attribute_value("t"):
                left_candidate = nodes[copy_index]
                break
            copy_index -= 1

        # take the minimum time (leaves being at time 0, root being at max time)
        if left_candidate is None and right_candidate is None:
            # We're running this on the root
            return
        elif left_candidate is None:
            selection = right_candidate
        elif right_candidate is None:
            selection = left_candidate
        else:
            comp = right_candidate.attribute_value("t") - left_candidate.attribute_value("t")
            if comp >= 0:
                selection = left_candidate
            else:
                selection = right_candidate

        # create new edge
        nodeT = nodes[index].attribute_value("t")
        futureT = selection.attribute_value("t")
        new_edge = [selection, nodes[index]]

        # set the branch length of the current node
        nodes[index].set_length(futureT - nodeT, selection)

        return new_edge

    def sample_trees(self, m : int) -> list:
        """
        Generate m trees and add them to the list of generated trees

        Returns: the list of all generated trees from this run and any prior
                    uncleared runs.
        """
        for dummy in range(m):
            self.generated_trees.append(self.generate_tree())

        return self.generated_trees

   
