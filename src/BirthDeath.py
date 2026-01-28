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
Last Edit : 11/10/25
First Included in Version : 2.0.0

Docs   - [x]
Tests  - [x] Passed 5/5 tests, with 100% coverage
Design - [x]

SOURCES:

1) https://www.sciencedirect.com/science/article/pii/S0022519309003300#bib24

2) https://academic.oup.com/sysbio/article/59/4/465/1661436#app2


"""

import random
import numpy as np
import scipy
from .Network import *
from math import log, exp
from typing import Union

TIP_ERROR_THRESHOLD : float = 5 * 1e3

#########################
#### EXCEPTION CLASS ####
#########################

class BirthDeathSimulationError(Exception):
    """
    This exception is thrown whenever something irrecoverably wrong happens
    during the process of generating a network within the Yule or CBDP algorithm
    """

    def __init__(self, message : str = "Something went wrong generating a\
                                        network"):
        """
        Create a custom BirthDeathSimulationError by providing an error message.

        Args:
            message (str, optional): A custom error message. Defaults to 
                                     "Something went wrong generating a
                                     network".
        Returns:
            N/A
        """
        self.message = message
        super().__init__(self.message)


def estimate_expected_tips(gamma : float, time : float) -> float:
    """
    Estimate the expected number of taxa produced by a Yule process.

    Args:
        gamma (float): Birth rate parameter (must be > 0).
        time (float): Target time horizon (must be >= 0).
    Returns:
        float: Expected number of taxa at the provided time.
    """
    try:
        return 2 * exp(gamma * time)
    except OverflowError:
        return math.inf


##########################
#### HELPER FUNCTIONS ####
##########################

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
    live_nodes = live_species(nodes)
    
    #use the rng object to select an index
    randomInt : int = rng.integers(0, len(live_nodes)) # type: ignore
    
    return live_nodes[randomInt]

def live_species(nodes : list[Node]) -> list[Node]:
    """
    Returns a subset of Nodes that represent live lineages. 
    A Node represents a live lineage if it has an attribute "live" set to True.

    Args:
        nodes (list[Node]): a list of all nodes in a network

    Returns:
        list[Node]: the subset of nodes in a network that represent 
                    live lineages (have attr key value pair "live" : True).
    """
    return [node for node in nodes if node.attribute_value("live") is True]

############################
#### Network Generators ####
############################

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

    def __init__(self, 
                 gamma : float, 
                 n : Union[int, None] = None, 
                 time : Union[float, None] = None, 
                 rng : Union[np.random.Generator, None] = None) -> None:
        """
        Raises:
            BirthDeathSimulationError: If something goes wrong simulating 
                                       networks.
        Args:
            gamma (Union[float, None]): Birth rate. Should be strictly positive.
            n (Union[int, None], optional): Number of taxa in simulated network. 
                                            Defaults to None.
            time (Union[float, None], optional): Age of simulated network, 
                                                 alternative to 
                                                 simulating to a number of taxa. 
                                                 Defaults to None.
            rng (Union[np.random.Generator, None], optional): A random number 
                                                              generator. This input allows
                                                              for consistent generation if a seed is given, 
                                                              generally for debugging.
                                                              Defaults to None.
        Returns:
            N/A
        """
        # set birth rate
        self.set_gamma(gamma)
        
        # goal number of taxa/time
        if n is not None:
            self.set_taxa(n)
        else:
            if time is None:
                raise BirthDeathSimulationError("If you do not provide a value\
                                                 for the number of taxa, please\
                                                 provide a time constraint for\
                                                 network simulation")
            self.set_time(time)
            

        # current number of live lineages, always starts at 2
        self.lin : int = 2

        # helper var for labeling internal nodes
        self.internal_count : int = 1

        # amount of time elapsed during the simulation of a tree
        self.elapsed_time : float = 0

        # a list of trees generated under this model
        self.generated_networks : list[Network] = []
        
        if rng is not None: 
            self.rng : np.random.Generator = rng
        else:
            seed : int = random.randint(0,10000)
            self.rng : np.random.Generator = np.random.default_rng(seed)

    def set_time(self, value : float) -> None:
        """
        Set simulated network age

        Args:
            value (float): The age of any future simulated trees
        Returns:
            N/A
        Raises:
            BirthDeathSimulationError: if value <= 0
        """
        self.condition = "T"
        if value <= 0:
            raise BirthDeathSimulationError("Please use a time value > 0")
        self.time = value
        expected_tips : float = estimate_expected_tips(self.gamma, value)
        if expected_tips > TIP_ERROR_THRESHOLD:
            raise BirthDeathSimulationError(
                "Expected number of taxa "
                f"({expected_tips:.2e}) for gamma={self.gamma} and time={value} "
                "is impractical to simulate."
            )
        
    def set_taxa(self, value : int) -> None:  
        """
        Set simulated tree taxa count

        Raises:
            BirthDeathSimulationError: if value < 2
        Args:
            value (int): an integer >= 2
        Returns:
            N/A
        """
        self.condition = "N"
        self.N = value
        if value < 2:
            raise BirthDeathSimulationError("Please use a value >= 2")
    
    def set_gamma(self, new_gamma : float) -> None:
        """
        Setter for the gamma (birth rate) parameter

        Raises:
            BirthDeathSimulationError: if new_gamma <= 0
        Args:
            new_gamma (float): the new birth rate
        Returns:
            N/A
        """
        if new_gamma <= 0:
            raise BirthDeathSimulationError("Birth rate must be > 0")
        self.gamma = new_gamma
        
    def _draw_waiting_time(self) -> float:
        """
        Draw a waiting time until the next speciation event from 
        a memory-less exponential distribution.

        Since each lineage is equally likely for each event 
        under the Yule Model, the waiting time is given by the parameter 
        numlineages * birthRate or .lin * .gamma
        
        Args:
            N/A
        Returns:
            float: waiting time value until next speciation event.
        """
        scale = 1 / (self.lin * self.gamma)
        random_float_01 : float  = self.rng.random()
        return scipy.stats.expon.ppf(random_float_01, scale = scale)# type: ignore

    def _event(self, network : Network) -> int: 
        """
        A speciation event occurs. Selects a random living lineage.

        Then add an internal "dead" node with: 
            branch length := t_parent + drawnWaitTime
        
        Set the parent to the chosen node as that internal node.

        Args:
            network (Network): the network currently being built
        Returns: 
            int: 0 if success, -1 if no more events can happen.      
        """

        # select random live lineage to branch from
        spec_node = random_species_selection(network.V(), self.rng)

        # keep track of the old parent, we need to disconnect edges
        # This node is guaranteed to only have 1 parent, since the network
        # is really just a binary tree.
        old_parent = network.get_parents(spec_node)[0]

        # calculate the branch length to the internal node
        next_time = self._draw_waiting_time()
        branch_len = 0
        parent_time = old_parent.get_time() 
        
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
        new_internal = Node(attr={"t": self.elapsed_time, "live": False},
                           name="internal" + str(self.internal_count))
        new_internal.set_time(self.elapsed_time)
        
        self.internal_count += 1

        # there's a new live lineage
        self.lin += 1
        new_label = "spec" + str(self.lin)

        # create the node for the new extant species
        new_spec_node = Node(attr={"live": True}, name=new_label)

        # add the newly created nodes
        network.add_nodes([new_spec_node, new_internal])

        # add the newly created branches 
        edge1 = Edge(new_internal, new_spec_node)
        edge2 = Edge(new_internal, spec_node)
        edge3 = Edge(old_parent, new_internal)
        
        edge3.set_length(branch_len)
        
        # remove the old connection (oldParent)->(specNode)
        # add the new edges
        network.remove_edge([old_parent, spec_node])
        network.add_edges([edge1, edge2, edge3])
        
        return 0

    def generate_network(self) -> Network:
        """
        Simulate one Network. Starts with a root and 2 living lineages
        and then continuously runs speciation (in this case birth only) 
        events until there are exactly N live species.

        After the nth event, draw one more time and fill out the remaining
        branch lengths.

        Args:
            N/A
        Returns:
            Network: The simulated tree
        """
        net : Network = Network()
        
        # Set up the tree with 2 living lineages and an "internal" root node
        node1 = Node("root", 
                     attr={"t" : 0, "label" : "root", "live" : False},
                     t = 0)
        node2 = Node("spec1", attr={"live" : True})
        node3 = Node("spec2", attr={"live" : True})

        
        net.add_nodes(node1, node2, node3)
        net.add_edges([Edge(node1, node2), Edge(node1, node3)])
        
            
        # until the tree contains N extant taxa, keep having speciation events
        if self.condition == "N":
            if self.N == 2:
                node2.set_time(1)
                node3.set_time(1)
                return net
            
            #create N lineages
            while len(live_species(net.V())) < self.N:
                self._event(net)

            # populate remaining branches with branch lengths according to
            # Eq 5.1? Just taking sigma_n for now
            next_time = self._draw_waiting_time()

            for node in live_species(net.V()):
                node.add_attribute("t", self.elapsed_time + next_time)
                node.set_time(self.elapsed_time + next_time)
                parents = net.get_parents(node)
                if len(parents) != 0:
                    parent = parents[0]
                    parent_time = parent.get_time()
                    final_len= self.elapsed_time + next_time - parent_time
                    net.get_edge(parent, node).set_length(final_len)
                
            # reset the elapsed time to 0, and the number of live branches to 2
            # for correctness generating future trees
            self.elapsed_time = 0
            self.lin = 2

        else:
            while self.elapsed_time < self.time:
                status = self._event(net)
                if status == -1:
                    break

            for node in live_species(net.V()):
                # the live lineages are all leaves, 
                # and are thus all at the goal time
                node.add_attribute("t", self.time)
                node.set_time(self.time)
                
                #calculate branch lengths
                parent = net.get_parents(node)[0]
                parent_time = parent.get_time()
                
                #set branch lengths
                net.get_edge(parent, node).set_length(self.time - parent_time)
                
            # reset the elapsed time to 0, and the number of live branches to 2
            # for correctness generating future trees
            self.elapsed_time = 0
            self.lin = 2
        
        return net

    def clear_generated(self) -> None:
        """
        Empty out the generated network array
        
        Args:
            N/A
        Returns:
            N/A
        """
        self.generated_networks = []

    def generate_networks(self, num_networks : int) -> list[Network]:
        """
        Generate a set number of trees.

        Args:
            num_networks (int): The number of networks to generate
        Returns: 
            list[Network]: The array of generated networks. 
                           (includes all that have been previously generated)
        """
        for _ in range(num_networks):
            self.generated_networks.append(self.generate_network())

        return self.generated_networks

class CBDP:
    """
    Constant Rate Birth Death Process Network simulation
    """

    def __init__(self, 
                 gamma : float,
                 mu : float, 
                 n : int, 
                 sample : float = 1) -> None:
        """
        Create a new Constant Rate Birth Death Simulator.
        
        You need the following parameters:
        1) A birth rate (gamma, as always)
        2) A death rate (mu, and should always be less than the birth rate 
                         so that networks can actually be generated and reach 
                         the goal amount of species)
        3) A goal amount of !LIVE! species in the simulated network.
        4) A sampling rate.
        
        NOTE! When doing network operations on a CBDP simulated tree with n LIVE
        lineages, you will have more than n leaves in the the network. If you want
        leaves, great, but if you want the live lineages, be sure to use the 
        live_species counter function.
        
        Raises:
            BirthDeathSimError: if birth/death/sample/n parameters 
                                are non-sensical
        Args:
            gamma (float): birth rate
            mu (float): death rate
            n (int): number of !LIVE! taxa for simulated trees
            sample (float, optional): Sampling rate from (0, 1]. Defaults to 1.
        Returns:
            N/A
        """
        
        self.set_sample(sample)
        self.set_bd(gamma, mu)
        self.set_n(n)
        
        self.generated_networks : list[Network]= []

    def set_bd(self, gamma : float, mu : float) -> None:
        """
        Set the birth and death rates for the model

        Raises:
            BirthDeathSimError: if birth/death parameters are non-sensical
        Args:
            gamma (float): The birth rate. Must be a positive real number, and 
                           strictly greater than mu. 
            mu (float): The death rate. Must be a non-negative real number, less 
                        than gamma. 
        Returns:
            N/A
        """
        
        if gamma <= 0:
            raise BirthDeathSimulationError("Please input a positive birth rate")
        if mu < 0:
            raise BirthDeathSimulationError("Please input a non-negative death rate")
        if gamma <= mu:
            raise BirthDeathSimulationError("Death rate is greater than or equal to the birth rate!")
        
        # Eq 15 from (1)
        self.gamma : float = gamma / self.sample
        self.mu : float = mu - gamma * (1 - (1 / self.sample))
        
        # Note: Mathematical analysis shows that with valid initial parameters
        # (gamma > mu >= 0, sample in (0,1]), the adjusted rates will always
        # maintain mu >= 0 and gamma > mu, so additional validation is unnecessary.
        
        # probabilities of speciation or extinction event
        self.pBirth = self.gamma / (self.gamma + self.mu)
        self.pDeath = self.mu / (self.gamma + self.mu)
    
    def set_n(self, new_n : int) -> None:
        """
        Set the number of taxa for the simulated trees.

        Raises:
            BirthDeathSimError: if the n parameter is non-sensical
        Args:
            new_n (int): An integer value >= 2
        Returns:
            N/A
        """
        if new_n < 2:
            raise BirthDeathSimulationError("Generated trees must have 2 or \
                                             more taxa")
        self.N = new_n
    
    def set_sample(self, new_sampling : float) -> None:
        """
        Set the sampling rate. Must be a value in the interval (0,1].

        Raises:
            BirthDeathSimulationError: If the provided sampling rate is not 
                                       in the right range
        Args:
            new_sampling (float): A float between 0 (exclusive), 
                                  and 1 (inclusive)
        Returns:
            N/A
        """
        # Ensure that the sampling rate is in the correct interval
        if 0 < new_sampling <= 1:
            self.sample : float = new_sampling
        else:
            raise BirthDeathSimulationError("Sampling rate must be drawn from (0,1]")
    
    def _qinv(self, r : float) -> float:
        """
        Draw a time from the Qinv distribution from (2)

        Args:
            r (float): r[0] from the n-1 samples from [0,1]
        Returns: 
            float: The time t, which is the age of a new simulated tree
        """
        term1 = 1 / (self.gamma - self.mu)  # Fixed: added parentheses
        term2 = 1 - ((self.mu / self.gamma) * pow(r, 1 / self.N))
        term3 = 1 - pow(r, 1 / self.N)
        return term1 * log(term2 / term3)

    def _finv(self, r : float, t : float) -> float:
        """   
        Draw a sample speciation time from the Finv distribution from (2)

        Args:
            r (float): r_i, from the sampled values from [0,1]
            t (float): The age of the tree determined by Qinv(r[0])
        Returns: 
            float: s_i from r_i
        """
        rate_dif = self.gamma - self.mu
        
        term1 = (1 / rate_dif)
        term2 = self.gamma - (self.mu * exp(-1 * t * rate_dif))
        term3 = r * (1 - exp(-1 * t * rate_dif))
        
        #_finv equation
        log_term = log((term2 - self.mu * term3) / (term2 - self.gamma * term3))
        
        return term1 * log_term

    def generate_network(self) -> Network:
        """
        Simulate a single network under the Constant Rate Birth Death Model.
        
        Follows the algorithm laid out by (2)

        Args:
            N/A
        Returns: 
            (Network) A network with n taxa chosen from the distributions.
        """
        net = Network()

        # step 1
        r = [random.random() for _ in range(self.N)]

        # step 2
        t = self._qinv(r[0])

        # step 3
        s = {self._finv(r[i], t): (i + .5) for i in range(1, self.N)}

        # step 4 setup

        sKeys = list(s.keys())

        # set up leaf nodes and internal nodes in proper order (fig 5)
        for j in range(2 * self.N - 1):
            if j % 2 == 0:
                # leaf node - mark as living
                leaf = Node(attr={"t": t, "live": True}, name="T" + str(int(j / 2) + 1))
                net.add_nodes(leaf)
                leaf.set_time(t)
                
            else:
                # internal node - mark as not living
                internal = Node(attr={"t": sKeys[int((j - 1) / 2)], "live": False}, 
                                name="internal" + str(int((j - 1) / 2)))
                net.add_nodes(internal)
                internal.set_time(sKeys[int((j - 1) / 2)])

        # step 4
        for i in range(2 * self.N - 1):
            # for each node, connect it to the correct parent
            new_edge : Union[Edge, None] = self._connect(i, net.V())
            if new_edge is not None:
                net.add_edges(new_edge)

        return net

    @staticmethod
    def _connect(index : int, nodes : list[Node]) -> Union[Edge, None]:
        """
        Given the nodes and a node to connect, create a new edge.

        The parent node is defined to be the closest to nodes[index] in terms
        of time and proximity in the list. There are two candidates, 
        the left and right candidate. Each candidate is the nearest element in 
        the list such that the time attribute is less than nodes[index]. 
        The parent is the maximum of the two candidates.
        
        Args:
            index (int): The index for a node to connect to its parent in the 
                         tree
            nodes (list[Node]): A list of nodes (list[i] is the ith node along 
                                a horizontal axis that alternates between 
                                species and internal s_i 
                                nodes/speciation events)
        Returns: 
            Union[Edge, None]: The edge from nodes[index] to its correct parent,
                               unless nodes[index] is the root in which case 
                               None is returned.
        """
        node_t : float = nodes[index].get_time() #.attribute_value("t")

        # find right candidate
        right_index : int = index + 1
        right_candidate : Union[Node, None] = None

        while right_index < len(nodes):
            # search in the list to the right (ie increase the index)
            right_t = nodes[right_index].get_time() #.attribute_value("t")
            
            if right_t < node_t:
                right_candidate = nodes[right_index]
                break
            right_index += 1

        # find left candidate
        left_index = index - 1
        left_candidate : Union[Node, None] = None
        while left_index >= 0:
            # search in the left part of the list
            left_t : float = nodes[left_index].get_time() #.attribute_value("t")
            if left_t < node_t:
                left_candidate = nodes[left_index]
                break
            left_index -= 1

        # take the minimum time (leaves being at time 0, root being at max time)
        if left_candidate is None and right_candidate is None:
            # We're running this on the root
            return
        elif left_candidate is None:
            selection = right_candidate
        elif right_candidate is None:
            selection = left_candidate
        else:
            right_cand_t = right_candidate.get_time() 
            left_cand_t = left_candidate.get_time() 
            
            if right_cand_t - left_cand_t <= 0:
                selection = left_candidate
            else:
                selection = right_candidate

        # create new edge
        node_T = nodes[index].get_time() 
        assert(selection is not None)
        future_T = selection.get_time() 
        new_edge = Edge(selection, nodes[index])

        # set the branch length of the current node
        new_edge.set_length(node_T - future_T)

        return new_edge

    def generate_networks(self, m : int) -> list[Network]:
        """
        Generate 'm' number of trees and add them to the list of generated trees

        Args:
            m (int): number of networks to generate
        Returns: 
            list[Network]: The list of all generated trees from this run and any 
                           prior uncleared runs.
        """
        for _ in range(m):
            self.generated_networks.append(self.generate_network())

        return self.generated_networks

    def clear_generated(self) -> None:
        """
        Empty out the generated network array.
        
        Args:
            N/A
        Returns:
            N/A
        """
        self.generated_networks = []
