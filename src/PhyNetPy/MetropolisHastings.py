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
Last Stable Edit : 3/11/25
First Included in Version : 1.0.0
Docs   - [x]
Tests  - [ ]
Design - [x]
"""

from State import State
from MSA import MSA
from Matrix import Matrix
from ModelGraph import Model
import time

from ModelMove import *
from GTR import *
import random

###########################
#### EXCEPTION CLASSES ####
###########################

class HillClimbException(Exception):
    """
    This exception is raised when there is an error running the Hill Climbing
    algorithm.
    """

    def __init__(self, 
                 message : str = "Error during a Hill Climbing run") -> None:
        """
        Initialize the exception with an error message.

        Args:
            message (str, optional): A custom error message. Defaults to 
                                     "Error during a Hill Climbing run".
        Returns:
            N/A
        """
        self.message = message
        super().__init__(self.message)
        
class MetropolisHastingsException(Exception):
    """
    This exception is raised when there is an error running the Metropolis 
    Hastings algorithm.
    """

    def __init__(self, 
                 message : str = "Error running Metropolis-Hastings") -> None:
        """
        Initialize the exception with an error message.

        Args:
            message (str, optional): A custom error message. Defaults to 
                                     "Error running Metropolis-Hastings".
        Returns:
            N/A
        """
        self.message = message
        super().__init__(self.message)

##########################
#### PROPOSAL KERNELS ####
##########################

class ProposalKernel(ABC):
    """
    Abstract class that defines proposal kernel behavior.
    
    In general, simply must have a generate method that spits out a move.
    """
    
    def __init__(self) -> None:
        """
        Initialize a proposal kernel
        
        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__()
    
    @abstractmethod
    def generate(self) -> Move:
        """
        *ABSTRACT METHOD*
        
        Generate the next move for a model to apply to the network.
    
        Args:
            N/A
        Returns:
            Move: Any newly instantiated object that is a subclass of Move.
        """
        raise NotImplementedError("Calling abstract method from the \
                                  ProposalKernel superclass. This is an \
                                  *abstract* class. Please implement a \
                                  subclass with a generate method that returns \
                                  a subclass of type 'Move'")
    
class Infer_MP_Allop_Kernel(ProposalKernel):
    """
    Proposal kernel for the Infer_MP_Allop_2.0 method.
    """
    
    def __init__(self) -> None:
        """
        Initialize proposal kernel for the Infer_MP_Allop_2.0 method.
        
        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__()
        self.iter = 0
   
    def generate(self) -> SwitchParentage:
        """
        Simply return a new SwitchParentage object.

        Args:
            N/A
        Returns:
            SwitchParentage: A new switch parentage move. Randomness is 
                             priced-in.
        """
        new_move = SwitchParentage(self.iter)
        self.iter += 1
        return new_move
    
########################################
#### HILL CLIMB, MH, and SIM ANNEAL ####
########################################

class HillClimbing:
    """
    Class that implements the Hill Climbing search method.
    """

    def __init__(self, 
                 pkernel : ProposalKernel, 
                 submodel : GTR = JC(), 
                 data : Matrix | None = None, 
                 model : Model | None = None,
                 num_iter : int = 500,
                 stochastic : int = -1) -> None:
        """
        Initialize a Hill Climb search.

        Args:
            pkernel (ProposalKernel): Some proposal kernel
            
            submodel (GTR, optional): A substitution model, if applicable. 
                                      Defaults to the Jukes Cantor (JC) model.
                                      
            data (Matrix | None, optional): A data matrix, if applicable. 
                                     Defaults to None.
            
            model (Model | None, optional): A Model obj. Defaults to None.
                            
            num_iter (int, optional): A number of iterations to run the search.
                                      Defaults to 500.
                                      
            stochastic (int, optional): a random seed, if Stochastic Hill 
                                        Climbing is used instead of Standard 
                                        Hill Climbing. Defaults to -1.
        Returns:
            N/A
        """
        if model is None:
            self.current_state = State()
            self.current_state.bootstrap(data, submodel)
        else:
            self.current_state = State(model)
            
        self.data = data
        self.submodel = submodel
        self.kernel = pkernel
        self.num_iter = num_iter
        self.nets_2_scores = {}
        
        if stochastic is not None:
            self.rng = np.random.default_rng(stochastic)
        else:
            self.rng = None
        
    def run(self) -> State:
        """
        Run the hill climbing algorithm one time on a bootstrapped starting 
        state.

        Args:
            N/A
        Returns:
            State: The final end state of the model after the input number of 
                   iterations. If the search has converged within the set 
                   iterations, this state will be some local minimum.
        """
        
        # Begin logging info
        self.current_state.write_line_to_summary("--------------------------")
        self.current_state.write_line_to_summary("------Begin Hillclimb-----")

        # Bookkeeping
        iter_no = 0
        top_network_ct = 1
        
        #Map from networks to likelihood scores. Maintains the top scorers
        leaderboard : dict[Network, float] = {}
        
        #Start iterating
        while iter_no < self.num_iter:
            
            # propose a new state
            next_move = self.kernel.generate()
          
            is_valid : bool = self.current_state.generate_next(next_move)
            
            if is_valid:
                # Calculate the difference in score between the 
                # proposed state and the current state
                cur : float = self.current_state.likelihood()
                
                proposed : float = self.current_state.proposed().likelihood()
                
                delta : float = cur - proposed
                accepted : bool = True 
                
                    
                if delta <= 0:
                    self.current_state.commit(next_move)  
                else:
                    # if self.rng is not None:
                    #     if self.rng.random() < .188 * pow(.97 ,iter_no):
                    #         self.current_state.commit(next_move)
                    #     else:
                    #         accepted = False
                    #         self.current_state.revert(next_move)
                    # else:
                    accepted = False
                    self.current_state.revert(next_move)
                
                # Grab network in current state
                cur_net = self.current_state.current_model.network
        
                if cur_net not in leaderboard.keys():
                    if accepted:
                        #calc score needed to make the leaderboard
                        if leaderboard != {}:
                            cur_max_val = max(leaderboard.values())
                        
                        # If leaderboard is not full
                        if len(list(leaderboard.keys())) < top_network_ct:
                            leaderboard[cur_net] = proposed
                        # Leaderboard full. Check to see if the current network
                        # cracks it.
                        elif proposed > cur_max_val:
                            leaderboard[cur_net] = proposed
                            old_net = [net for net in leaderboard.keys()
                                       if leaderboard[net] == cur_max_val][0]
                            del leaderboard[old_net]
                    
                # Log current iteration
                self.current_state.write_line_to_summary("ITER #" 
                                                         + str(iter_no) 
                                                         + " LIKELIHOOD = " 
                                                         + str(cur))
            else:
                raise HillClimbException("Move has resulted in an invalid \
                                          network state. Aborting search.")
            
            iter_no += 1
            
        
        self.nets_2_scores = leaderboard
        
        self.current_state.write_line_to_summary("DONE. EXITED WITH 0 ERRORS")
        self.current_state.write_line_to_summary("--------------------------")

        return self.current_state

    def run_many(self, count : int) -> list[float]:
        """
        Runs the hill climbing algorithm 'count' times, with different starting 
        states.

        Args: 
            count (int): the number of times to run a hill climbing chain.
        Returns: 
            list[float]: The statistics over all the hill climb run scores. 
                         [mean, median, max, min]

        """
        
        assert(self.data is not None)
        
        all_end_states = []
        for _ in range(count):
            self.current_state = State()
            self.current_state.bootstrap(self.data, self.submodel)
            end_state = self.run()
            all_end_states.append(end_state.likelihood())

        all_end_states.sort()
        length = len(all_end_states)

        if length % 2 == 0:
            median = .5 * (all_end_states[int(length / 2)] 
                        + all_end_states[int(length / 2) - 1])
        else:
            median = all_end_states[int((length + 1) * .5) - 1]

        mean = sum(all_end_states) / length
        max_val = all_end_states[-1]
        min_val = all_end_states[0]

        print("===============================================")
        print("Hill Climbing ran " + str(count) + " times...")
        print("===============================================")
        print("Mean score: " + str(mean) + "\n"
              + "Median score: " + str(median) + "\n"
              + "Maximum score: " + str(max_val) + "\n"
              + "Minimum score: " + str(min_val) + "\n")
        print("===============================================")

        return [mean, median, max_val, min_val]

class MetropolisHastings:
    """
    A special case of Hill Climbing, in which moves are accepted even if the 
    score is not an improvement, based on the Hastings Ratio of the current 
    move.
    """
        
    def __init__(self, 
                 pkernel : ProposalKernel, 
                 submodel : GTR = JC(), 
                 data : Matrix | None = None, 
                 model : Model | None = None,
                 num_iter : int = 500
                 ) -> None:
        """
        Initialize a Metropolis Hastings search.

        Args:
            pkernel (ProposalKernel): A proposal kernel. Should be a concrete
                                      implementation of ProposalKernel, and 
                                      !not! the abstract class.
            
            submodel (GTR, optional): A substitution model. Defaults to the
                                      Jukes Cantor (JC) model, but will only be 
                                      utilized if using a model bootstrap. 
            
            data (Matrix | None, optional): The data associated with the model. 
                                     Defaults to None, utilize if using 
                                     bootstrapping for the starting state.
            
            model (Model | None, optional): A phylogenetic model. Defaults to None, 
                                     utilize if using bootstrapping for the 
                                     starting state.
            
            num_iter (int, optional): A number of times to generate network 
                                      moves. Defaults to 500.
        Returns:
            N/A  
        """
        
        self.current_state = State(model)
        
        if model is None:
            self.current_state.bootstrap(data, submodel)
        
        self.data = data
        self.submodel = submodel
        self.kernel = pkernel
        self.num_iter = num_iter

    def run(self) -> State:
        """
        Run the Metropolis-Hastings algorithm. Returns a 'State' object that 
        contains 
        
        Args:
            N/A
        Returns: 
            State: The end state that locally minimizes the score (if the 
                   algorithm has converged by the given iteration count).
        """

        self.current_state.write_line_to_summary("----------------------------")
        self.current_state.write_line_to_summary("----Begin Metro-Hastings----")

        iter_no = 0

        while iter_no < self.num_iter:

            # propose a new state
            next_move = self.kernel.generate()
            self.current_state.generate_next(next_move)
            
            cur = self.current_state.likelihood() 
            prop = self.current_state.proposed().likelihood()

            #(logP(B) - logP(A)) + (logP(A|B) - logP(B|A)) > r ~ log(Unif(0, 1))
            if prop - cur + next_move.hastings_ratio() > random.random():
                self.current_state.commit(next_move)
            else:
                self.current_state.revert(next_move)

            self.current_state.write_line_to_summary("ITER #" + str(iter_no) 
                                                     + " LIKELIHOOD = " 
                                                     + str(cur))
            iter_no += 1

        self.current_state.write_line_to_summary("DONE. EXITED WITH 0 ERRORS")
        self.current_state.write_line_to_summary("--------------------------")

        return self.current_state
    
    def run_many_different_start(self,
                                 count : int, 
                                 format_stats = True) -> list[float]:
        """
        Runs the MH algorithm 'count' times.

        Args: 
            count (int): The number of times to run the algorithm.
            format_stats (bool): Flag that, if true, will print out stats. If 
                                false, no stats will be printed.
        Returns: 
            list[float] : Prints out and returns the statistics for each chain. 
                          [mean, median, max, min]
        """
        assert(self.data is not None and self.submodel is not None)
        
        all_end_states : list[float] = []
        
        for _ in range(count):
            self.current_state = State()
            self.current_state.bootstrap(self.data, self.submodel)
            end_state : State = self.run()
            all_end_states.append(end_state.likelihood())

        all_end_states.sort()
        
        length = len(all_end_states)

        if length % 2 == 0:
            median = .5 * (all_end_states[int(length / 2)] 
                           + all_end_states[int(length / 2) - 1])
        else:
            median = all_end_states[int((length + 1) * .5) - 1]

        mean = sum(all_end_states) / length
        max_val = all_end_states[-1]
        min_val = all_end_states[0]

        if format_stats:
            print("===============================================")
            print("MH ran " + str(count) + " times...")
            print("===============================================")
            print("Mean score: " + str(mean) + "\n"
                  + "Median score: " + str(median) + "\n"
                  + "Maximum score: " + str(max_val) + "\n"
                  + "Minimum score: " + str(min_val) + "\n")
            print("===============================================")

        return [mean, median, max_val, min_val]

