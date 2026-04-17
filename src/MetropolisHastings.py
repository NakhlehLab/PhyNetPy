#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

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

V1 Architecture - Metropolis-Hastings and Hill Climbing search algorithms.
"""

import copy
import math
import time
import random
from abc import ABC, abstractmethod
from typing import Literal, Optional

import numpy as np

# Relative imports
from .State import State
from .MSA import MSA
from .Matrix import Matrix
from .ModelGraph import Model
from .ModelMove import Move, SwitchParentage
from .GTR import GTR, JC
from .Network import Network


###########################
#### EXCEPTION CLASSES ####
###########################

class HillClimbException(Exception):
    """
    This exception is raised when there is an error running the Hill Climbing
    algorithm.
    """

    def __init__(self, 
                 message: str = "Error during a Hill Climbing run") -> None:
        """
        Initialize the exception with an error message.

        Args:
            message (str, optional): A custom error message.
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
                 message: str = "Error running Metropolis-Hastings") -> None:
        """
        Initialize the exception with an error message.

        Args:
            message (str, optional): A custom error message.
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
        raise NotImplementedError("Calling abstract method from the "
                                  "ProposalKernel superclass.")

    def report_outcome(self, accepted: bool, delta: float = 0.0) -> None:
        """Report whether the last generated move was accepted or rejected.

        Subclasses may override this to implement adaptive weight tuning.

        Args:
            accepted: True if the move improved the score and was committed.
            delta: Score change (proposed - current).  Positive means
                   the proposal was better (for a maximisation problem).
        """
    

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
            SwitchParentage: A new switch parentage move.
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
                 pkernel: ProposalKernel, 
                 submodel: GTR = None, 
                 data: Matrix | None = None, 
                 model: Model | None = None,
                 num_iter: int = 500,
                 stochastic: int = -1,
                 enhanced_stop: bool = True) -> None:
        """
        Initialize a Hill Climb search.

        Args:
            pkernel (ProposalKernel): Some proposal kernel
            submodel (GTR, optional): A substitution model. Defaults to JC.
            data (Matrix | None, optional): A data matrix. Defaults to None.
            model (Model | None, optional): A Model obj. Defaults to None.
            num_iter (int, optional): Number of iterations. Defaults to 500.
            stochastic (int, optional): Random seed. Defaults to -1.
            enhanced_stop (bool, optional): Early stopping flag. Defaults to True.
        Returns:
            N/A
        """
        if submodel is None:
            submodel = JC()
            
        if model is None:
            self.current_state = State()
            if data is not None:
                self.current_state.bootstrap(data, submodel)
        else:
            self.current_state = State(model)
            
        self.data = data
        self.submodel = submodel
        self.kernel = pkernel
        self.num_iter = num_iter
        self.nets_2_scores = {}
        self.enhanced_stop = enhanced_stop
        
        if stochastic != -1:
            self.rng = np.random.default_rng(stochastic)
        else:
            self.rng = None
        
    def run(self) -> State:
        """
        Run the hill climbing algorithm.

        Args:
            N/A
        Returns:
            State: The final end state of the model.
        """
        # Begin logging info
        self.current_state.write_line_to_summary("--------------------------")
        self.current_state.write_line_to_summary("------Begin Hillclimb-----")

        iter_no = 0
        top_network_ct = 1
        
        leaderboard: dict[Network, float] = {}
        no_progress = 0
        cached_cur: float | None = None
        
        while iter_no < self.num_iter:
            if no_progress == 150:
                break
            
            next_move = self.kernel.generate()
            is_valid: bool = self.current_state.generate_next(next_move)
            
            if is_valid:
                try:
                    if cached_cur is None:
                        cached_cur = self.current_state.likelihood()
                    cur = cached_cur
                    proposed: float = self.current_state.proposed().likelihood()
                except Exception:
                    try:
                        self.current_state.revert(next_move)
                    except Exception:
                        pass
                    self.kernel.report_outcome(False, delta=0.0)
                    if self.enhanced_stop:
                        no_progress += 1
                    iter_no += 1
                    continue
                
                delta: float = cur - proposed
                accepted: bool = True 
                
                if delta < 0:
                    self.current_state.commit(next_move)  
                    cached_cur = proposed
                    no_progress = 0
                else:
                    accepted = False
                    self.current_state.revert(next_move)
                    if self.enhanced_stop:
                        no_progress += 1

                self.kernel.report_outcome(accepted, delta=proposed - cur)
                
                cur_net = self.current_state.current_model.network
        
                if cur_net not in leaderboard.keys():
                    if accepted:
                        if leaderboard:
                            cur_max_val = max(leaderboard.values())
                        else:
                            cur_max_val = float('-inf')
                        
                        if len(list(leaderboard.keys())) < top_network_ct:
                            leaderboard[cur_net] = proposed
                        elif proposed > cur_max_val:
                            leaderboard[cur_net] = proposed
                            old_nets = [net for net in leaderboard.keys()
                                       if leaderboard[net] == cur_max_val]
                            if old_nets:
                                del leaderboard[old_nets[0]]
                    
                self.current_state.write_line_to_summary(
                    f"ITER #{iter_no} LIKELIHOOD = {cur}"
                )
            else:
                self.kernel.report_outcome(False, delta=0.0)
                if self.enhanced_stop:
                    no_progress += 1
            
            iter_no += 1
        
        self.nets_2_scores = leaderboard
        
        self.current_state.write_line_to_summary("DONE. EXITED WITH 0 ERRORS")
        self.current_state.write_line_to_summary("--------------------------")

        return self.current_state

    def run_many(self, count: int) -> list[float]:
        """
        Runs the hill climbing algorithm 'count' times.

        Args: 
            count (int): the number of times to run.
        Returns: 
            list[float]: [mean, median, max, min]
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
            median = 0.5 * (all_end_states[int(length / 2)] 
                        + all_end_states[int(length / 2) - 1])
        else:
            median = all_end_states[int((length + 1) * 0.5) - 1]

        mean = sum(all_end_states) / length
        max_val = all_end_states[-1]
        min_val = all_end_states[0]

        print("===============================================")
        print(f"Hill Climbing ran {count} times...")
        print("===============================================")
        print(f"Mean score: {mean}\n"
              f"Median score: {median}\n"
              f"Maximum score: {max_val}\n"
              f"Minimum score: {min_val}\n")
        print("===============================================")

        return [mean, median, max_val, min_val]


class MetropolisHastings:
    """
    A special case of Hill Climbing, in which moves are accepted even if the 
    score is not an improvement, based on the Hastings Ratio.
    """
        
    def __init__(self, 
                 pkernel: ProposalKernel, 
                 submodel: GTR = None, 
                 data: Matrix | None = None, 
                 model: Model | None = None,
                 num_iter: int = 500) -> None:
        """
        Initialize a Metropolis Hastings search.

        Args:
            pkernel (ProposalKernel): A proposal kernel.
            submodel (GTR, optional): A substitution model. Defaults to JC.
            data (Matrix | None, optional): The data. Defaults to None.
            model (Model | None, optional): A phylogenetic model. Defaults to None.
            num_iter (int, optional): Number of iterations. Defaults to 500.
        Returns:
            N/A  
        """
        if submodel is None:
            submodel = JC()
            
        self.current_state = State(model)
        
        if model is None and data is not None:
            self.current_state.bootstrap(data, submodel)
        
        self.data = data
        self.submodel = submodel
        self.kernel = pkernel
        self.num_iter = num_iter

    def run(self) -> State:
        """
        Run the Metropolis-Hastings algorithm.
        
        Args:
            N/A
        Returns: 
            State: The end state.
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

            # (logP(B) - logP(A)) + (logP(A|B) - logP(B|A)) > r ~ log(Unif(0, 1))
            if prop - cur + next_move.hastings_ratio() > random.random():
                self.current_state.commit(next_move)
            else:
                self.current_state.revert(next_move)

            self.current_state.write_line_to_summary(
                f"ITER #{iter_no} LIKELIHOOD = {cur}"
            )
            iter_no += 1

        self.current_state.write_line_to_summary("DONE. EXITED WITH 0 ERRORS")
        self.current_state.write_line_to_summary("--------------------------")

        return self.current_state
    
    def run_many_different_start(self,
                                 count: int, 
                                 format_stats: bool = True) -> list[float]:
        """
        Runs the MH algorithm 'count' times.

        Args: 
            count (int): The number of times to run.
            format_stats (bool): Flag to print stats.
        Returns: 
            list[float]: [mean, median, max, min]
        """
        assert(self.data is not None and self.submodel is not None)
        
        all_end_states: list[float] = []
        
        for _ in range(count):
            self.current_state = State()
            self.current_state.bootstrap(self.data, self.submodel)
            end_state: State = self.run()
            all_end_states.append(end_state.likelihood())

        all_end_states.sort()
        
        length = len(all_end_states)

        if length % 2 == 0:
            median = 0.5 * (all_end_states[int(length / 2)] 
                           + all_end_states[int(length / 2) - 1])
        else:
            median = all_end_states[int((length + 1) * 0.5) - 1]

        mean = sum(all_end_states) / length
        max_val = all_end_states[-1]
        min_val = all_end_states[0]

        if format_stats:
            print("===============================================")
            print(f"MH ran {count} times...")
            print("===============================================")
            print(f"Mean score: {mean}\n"
                  f"Median score: {median}\n"
                  f"Maximum score: {max_val}\n"
                  f"Minimum score: {min_val}\n")
            print("===============================================")

        return [mean, median, max_val, min_val]


class SimulatedAnnealing:
    """
    Simulated annealing search for phylogenetic network optimization.

    Temperature follows an exponential schedule from ``t_start`` toward ``t_end``
    over (1 - plateau_frac) * num_iter adjustment steps.

    * ``schedule="cool"`` (default): ``t_start`` should exceed ``t_end`` so T
      drops (classic SA: explore early, exploit late).
    * ``schedule="heat"``: ``t_end`` should exceed ``t_start`` so T rises
      (greedy / hill-climbing when T is tiny, then more uphill acceptance later
      to escape local maxima).
    * ``schedule="geometric_reheat"``: Hold T for ``steps_per_temp`` iterations,
      then multiply by ``cooling_alpha`` (geometric cooling) down to ``t_min``.
      If the run-best score has not improved for ``reheat_threshold``
      iterations, multiply T by ``reheat_factor`` (capped at ``t_start``) to
      escape local maxima.

    At each step, a worse move is accepted with probability exp(-delta / T)
    where ``delta = cur - proposed`` for the current vs proposed likelihood
    (maximization: negative delta means improving).

    Tracks the best network seen across the entire run and restores
    it at the end, so the returned state always holds the global best.
    """

    def __init__(self,
                 pkernel: ProposalKernel,
                 model: Model,
                 num_iter: int = 500,
                 t_start: float = 10.0,
                 t_end: float = 0.01,
                 n_restarts: int = 1,
                 seed: int = 42,
                 plateau_frac: float = 0.0,
                 progress_every: int = 0,
                 schedule: Literal["cool", "heat", "geometric_reheat"] = "cool",
                 *,
                 t_min: Optional[float] = None,
                 cooling_alpha: float = 0.93,
                 steps_per_temp: int = 100,
                 reheat_threshold: int = 1000,
                 reheat_factor: float = 2.0) -> None:
        """
        Args:
            pkernel: Proposal kernel that generates moves.
            model: Initial Model object (network + scorer).
            num_iter: Iterations per annealing run.
            t_start: Initial temperature (T0). For ``geometric_reheat``, also
                the ceiling when reheating.
            t_end: For ``cool``/``heat``, target temperature. For
                ``geometric_reheat`` ignored unless ``t_min`` is omitted, then
                used as the floor temperature (same role as T_min).
            n_restarts: Number of independent restarts.
            seed: RNG seed for reproducibility.
            plateau_frac: Fraction of iterations to hold at ``t_start``
                before beginning exponential cool/heat (0.0-1.0). Ignored for
                ``geometric_reheat``.
            progress_every: If > 0, print a progress line every this many
                iterations (current log score, temperature, best-so-far in run).
            schedule: ``"cool"`` / ``"heat"`` as above, or ``"geometric_reheat"``.
            t_min: Floor temperature for ``geometric_reheat`` (default: ``t_end``).
            cooling_alpha: Geometric factor applied every ``steps_per_temp``
                iterations (0 < alpha < 1).
            steps_per_temp: Iterations at a given T before cooling by alpha.
            reheat_threshold: If run-best has not improved in this many
                iterations, multiply T by ``reheat_factor`` (capped at ``t_start``).
            reheat_factor: Factor applied to T on reheat (> 1).
        """
        self.kernel = pkernel
        self.init_model = model
        self.num_iter = num_iter
        self.t_start = t_start
        self.t_end = t_end
        self.n_restarts = n_restarts
        self.rng = np.random.default_rng(seed)
        self.plateau_frac = max(0.0, min(plateau_frac, 0.99))
        self.progress_every = max(0, int(progress_every))
        self.schedule = schedule

        self.t_min_floor: float = float(t_min if t_min is not None else t_end)
        self.cooling_alpha = float(cooling_alpha)
        self.steps_per_temp = max(1, int(steps_per_temp))
        self.reheat_threshold = max(1, int(reheat_threshold))
        self.reheat_factor = float(reheat_factor)

        cool_iters = max(1, int(num_iter * (1.0 - self.plateau_frac)))
        if schedule == "cool" and not (t_start > t_end):
            raise ValueError(
                'SimulatedAnnealing schedule="cool" requires t_start > t_end '
                f"(got t_start={t_start}, t_end={t_end}).",
            )
        if schedule == "heat" and not (t_end > t_start):
            raise ValueError(
                'SimulatedAnnealing schedule="heat" requires t_end > t_start '
                f"(got t_start={t_start}, t_end={t_end}).",
            )
        if schedule == "geometric_reheat":
            if not (t_start > self.t_min_floor):
                raise ValueError(
                    'SimulatedAnnealing schedule="geometric_reheat" requires '
                    f"t_start > t_min (got t_start={t_start}, t_min={self.t_min_floor}).",
                )
            if not (0.0 < self.cooling_alpha < 1.0):
                raise ValueError(
                    "cooling_alpha must be in (0, 1) for geometric_reheat, "
                    f"got {self.cooling_alpha}.",
                )
            if self.reheat_factor <= 1.0:
                raise ValueError(
                    f"reheat_factor must be > 1, got {self.reheat_factor}.",
                )
            self.alpha = 1.0  # unused; kept for any code reading .alpha
        elif cool_iters > 1:
            self.alpha = (t_end / t_start) ** (1.0 / (cool_iters - 1))
        else:
            self.alpha = 1.0

        self.best_score: float = float('-inf')
        self.best_network = None
        self.run_stats: list[dict] = []

    def _single_run(self, state: State) -> dict:
        """Execute one SA run (cooling, heating, or geometric+reheat), return stats."""
        if self.schedule == "geometric_reheat":
            return self._single_run_geometric_reheat(state)

        temp = self.t_start
        plateau_end = int(self.num_iter * self.plateau_frac)
        accepted = 0
        uphill_accepted = 0
        best_run_score = state.likelihood()
        best_run_network = copy.deepcopy(state.current_model.network)

        for i in range(self.num_iter):
            try:
                next_move = self.kernel.generate()
                is_valid = state.generate_next(next_move)

                if not is_valid:
                    self.kernel.report_outcome(False, delta=0.0)
                    if i >= plateau_end:
                        temp *= self.alpha
                    continue

                try:
                    cur = state.likelihood()
                    proposed = state.proposed().likelihood()
                except Exception:
                    state.revert(next_move)
                    self.kernel.report_outcome(False, delta=0.0)
                    if i >= plateau_end:
                        temp *= self.alpha
                    continue

                delta = cur - proposed

                was_accepted = False
                if delta < 0:
                    state.commit(next_move)
                    accepted += 1
                    was_accepted = True
                elif temp > 0 and self.rng.random() < math.exp(-delta / temp):
                    state.commit(next_move)
                    accepted += 1
                    uphill_accepted += 1
                    was_accepted = True
                else:
                    state.revert(next_move)

                self.kernel.report_outcome(was_accepted, delta=proposed - cur)

                score_now = state.likelihood()
                if score_now > best_run_score:
                    best_run_score = score_now
                    best_run_network = copy.deepcopy(state.current_model.network)

                if i >= plateau_end:
                    temp *= self.alpha
            finally:
                if self.progress_every and (i + 1) % self.progress_every == 0:
                    try:
                        cur_sc = state.likelihood()
                    except Exception:
                        cur_sc = float("nan")
                    print(
                        f"  SA {i + 1}/{self.num_iter}  log_PL={cur_sc:.6f}  "
                        f"T={temp:.6g}  best_run={best_run_score:.6f}",
                        flush=True,
                    )

        return {
            "accepted": accepted,
            "uphill": uphill_accepted,
            "final_score": state.likelihood(),
            "best_score": best_run_score,
            "best_network": best_run_network,
        }

    def _single_run_geometric_reheat(self, state: State) -> dict:
        """Geometric cooling with reheats when the run-best score stalls."""
        temp = self.t_start
        t_floor = self.t_min_floor

        accepted = 0
        uphill_accepted = 0
        best_run_score = state.likelihood()
        best_run_network = copy.deepcopy(state.current_model.network)

        steps_at_level = 0
        since_best_improve = 0
        reheat_count = 0

        for i in range(self.num_iter):
            try:
                next_move = self.kernel.generate()
                is_valid = state.generate_next(next_move)

                if not is_valid:
                    self.kernel.report_outcome(False, delta=0.0)
                    since_best_improve += 1
                    steps_at_level += 1
                    temp, since_best_improve, steps_at_level, reheat_count = (
                        self._tick_geometric_reheat(
                            temp, since_best_improve, steps_at_level,
                            reheat_count, t_floor,
                        )
                    )
                    continue

                try:
                    cur = state.likelihood()
                    proposed = state.proposed().likelihood()
                except Exception:
                    state.revert(next_move)
                    self.kernel.report_outcome(False, delta=0.0)
                    since_best_improve += 1
                    steps_at_level += 1
                    temp, since_best_improve, steps_at_level, reheat_count = (
                        self._tick_geometric_reheat(
                            temp, since_best_improve, steps_at_level,
                            reheat_count, t_floor,
                        )
                    )
                    continue

                delta = cur - proposed

                was_accepted = False
                if delta < 0:
                    state.commit(next_move)
                    accepted += 1
                    was_accepted = True
                elif temp > 0 and self.rng.random() < math.exp(-delta / temp):
                    state.commit(next_move)
                    accepted += 1
                    uphill_accepted += 1
                    was_accepted = True
                else:
                    state.revert(next_move)

                self.kernel.report_outcome(was_accepted, delta=proposed - cur)

                score_now = state.likelihood()
                if score_now > best_run_score:
                    best_run_score = score_now
                    best_run_network = copy.deepcopy(state.current_model.network)
                    since_best_improve = 0
                else:
                    since_best_improve += 1

                steps_at_level += 1
                temp, since_best_improve, steps_at_level, reheat_count = (
                    self._tick_geometric_reheat(
                        temp, since_best_improve, steps_at_level,
                        reheat_count, t_floor,
                    )
                )
            finally:
                if self.progress_every and (i + 1) % self.progress_every == 0:
                    try:
                        cur_sc = state.likelihood()
                    except Exception:
                        cur_sc = float("nan")
                    print(
                        f"  SA {i + 1}/{self.num_iter}  log_PL={cur_sc:.6f}  "
                        f"T={temp:.6g}  best_run={best_run_score:.6f}  "
                        f"reheats={reheat_count}",
                        flush=True,
                    )

        return {
            "accepted": accepted,
            "uphill": uphill_accepted,
            "final_score": state.likelihood(),
            "best_score": best_run_score,
            "best_network": best_run_network,
            "reheat_count": reheat_count,
        }

    def _tick_geometric_reheat(
        self,
        temp: float,
        since_best_improve: int,
        steps_at_level: int,
        reheat_count: int,
        t_floor: float,
    ) -> tuple[float, int, int, int]:
        """One iteration of schedule: reheat if stalled, else geometric cool."""
        if since_best_improve >= self.reheat_threshold:
            temp = min(temp * self.reheat_factor, self.t_start)
            since_best_improve = 0
            steps_at_level = 0
            reheat_count += 1
        elif steps_at_level >= self.steps_per_temp:
            temp = max(temp * self.cooling_alpha, t_floor)
            steps_at_level = 0
        return temp, since_best_improve, steps_at_level, reheat_count

    def run(self) -> State:
        """
        Run simulated annealing (with optional restarts).

        Returns:
            State holding the best network found across all restarts.
        """
        from .ModelFactory import ModelFactory

        for restart in range(self.n_restarts):
            model = copy.deepcopy(self.init_model)
            state = State(model)

            stats = self._single_run(state)
            self.run_stats.append(stats)

            if stats["best_score"] > self.best_score:
                self.best_score = stats["best_score"]
                self.best_network = stats["best_network"]

        final_model = copy.deepcopy(self.init_model)
        final_model.network = self.best_network
        final_model.update_network()
        final_state = State(final_model)
        return final_state




