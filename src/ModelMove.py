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
Last Stable Edit : 4/24/26
First Included in Version : 1.0.0

Network proposal moves for MCMC and simulated-annealing search.

This module defines the proposal kernel primitives used by both the
Bayesian MCMC driver and the MPL simulated-annealing driver
(:mod:`src.MPL`).  Each move subclasses :class:`Move` and implements
``execute`` / ``undo`` / ``same_move`` / ``hastings_ratio``.  The kernel
wraps these into a weighted selection scheme; only the move classes
themselves need to understand how to mutate and restore the network.

Module layout
-------------

The file is organised in the following sections (search the banner
headers to jump between them):

    1. RNG HELPER METHODS
         Reproducible random sampling utilities (``_stable_sorted``,
         ``_rng_pick``, ``_rng_pick_weighted``) that every move should
         use instead of the stdlib ``random`` module.

    2. EXCEPTION CLASS
         :class:`MoveError` for fatal proposal failures.

    3. HELPER FUNCTIONS
         Shared graph-surgery utilities (edge splitting / merging,
         random edge sampling) used by multiple move classes.

    4. MOVE CLASS
         Abstract base :class:`Move` describing the contract every
         proposal must satisfy.

    5. TOPOLOGICAL MOVES
         :class:`SPR` (subtree-prune-and-regraft with distance-weighted
         regraft) and small tree-topology moves.

    6. PARAMETER-LEVEL MOVES
         :class:`_suppress_deg2`, :class:`_prune_orphan_chain`,
         :class:`ChangeNodeHeight`, :class:`ChangeInheritanceProb`,
         :class:`ChangeReticSource`, :class:`RelocateReticulation`,
         :class:`ChangeReticDest`, :class:`FlipReticulation`,
         :class:`AddReticulation`, :class:`RemoveReticulation`.

Every move is expected to leave ``model.network`` in a structurally
valid state (single root, no cycles, reticulations with in-degree 2,
no orphan internal nodes).  When a proposal cannot be realised, the
move either returns silently or rolls back via its ``undo_info``
snapshot so the caller can treat the outcome as a no-op.
"""

from __future__ import annotations
from collections import deque
import copy
from abc import ABC, abstractmethod
import numpy as np
from typing import TYPE_CHECKING, Any, Sequence

# Relative imports
from .Network import Network, Edge, Node
from .Logger import Logger

if TYPE_CHECKING:
    from .ModelGraph import Model


############################
#### RNG HELPER METHODS ####
############################
#
# All network moves previously used the stdlib ``random`` module, which pulls
# from a process-wide RNG that is not seeded by our SA/HC drivers.  That made
# proposal content non-reproducible across runs even with identical seeds.
# The helpers below route every random choice through ``model.rng`` (a
# ``numpy.random.Generator``) that is seeded once per search by
# ``MPL.search()`` using ``numpy.random.SeedSequence``.

def _stable_key(item: Any) -> Any:
    """Return a deterministic sort key for a ``Node`` or ``Edge``.

    ``Network`` stores nodes and edges in raw ``set`` containers, whose
    iteration order is hash-driven (i.e. memory-address-driven for
    user-defined classes).  Callers that sample from ``net.V()`` /
    ``net.E()`` therefore see different orderings across processes even
    with an identical seed.  Sorting by label before sampling fixes
    that without changing the graph data structures themselves.

    Falls back to ``id(item)`` for unknown types so we never raise.
    """
    if hasattr(item, "src") and hasattr(item, "dest"):
        src_lbl = getattr(item.src, "label", "") or ""
        dst_lbl = getattr(item.dest, "label", "") or ""
        return (str(src_lbl), str(dst_lbl))
    lbl = getattr(item, "label", None)
    if lbl is not None:
        return str(lbl)
    return id(item)


def _stable_sorted(seq: Sequence[Any]) -> list[Any]:
    """Return ``seq`` sorted by ``_stable_key``; tolerant of empty input."""
    if not seq:
        return []
    return sorted(seq, key=_stable_key)


def _rng_pick(rng: np.random.Generator, seq: Sequence[Any]) -> Any:
    """Uniformly sample one element from ``seq`` using ``rng``.

    ``seq`` is first sorted by ``_stable_key`` so the selection is
    reproducible across processes regardless of how the caller built
    the list.  Returns ``None`` when ``seq`` is empty so callers can
    short-circuit degenerate topologies without extra branching.
    """
    if not seq:
        return None
    ordered = _stable_sorted(seq)
    return ordered[int(rng.integers(0, len(ordered)))]


def _rng_pick_weighted(
    rng: np.random.Generator,
    seq: Sequence[Any],
    weights: Sequence[float],
) -> Any:
    """Weighted sample of one element from ``seq`` using ``rng``.

    Pairs ``seq`` with ``weights`` and sorts the pairs by
    ``_stable_key`` of the element so the draw is reproducible across
    processes.  Weights must be non-negative; they are normalised
    internally.  Returns ``None`` when ``seq`` is empty or the weight
    sum is non-positive.
    """
    if not seq:
        return None
    pairs = sorted(
        zip(seq, weights),
        key=lambda pw: _stable_key(pw[0]),
    )
    ordered_seq = [p for p, _ in pairs]
    w = np.asarray([w for _, w in pairs], dtype=float)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        return ordered_seq[int(rng.integers(0, len(ordered_seq)))]
    probs = w / total
    idx = int(rng.choice(len(ordered_seq), p=probs))
    return ordered_seq[idx]


#########################
#### EXCEPTION CLASS ####
#########################

class MoveError(Exception):
    """
    This exception is raised whenever there is a fatal error in executing a 
    network move.
    """
    def __init__(self, message: str = "Error executing a move") -> None:
        """
        Initializes a MoveError object.

        Args:
            message (str, optional): Custom error message. Defaults to
                                     "Error executing a move".
        Returns:
            N/A
        """
        self.message = message
        super().__init__(self.message)

##########################
#### HELPER FUNCTIONS ####
##########################

def insert_node_in_edge(edge: Edge, node: Node, net: Network) -> None:
    """
    Given an edge, a -> b, place a node c, such that a -> c -> b.
    This requires the deletion of edge a -> b, then the addition of edges
    a -> c and c -> b.

    Gamma preservation.  When ``b`` is a reticulation, the original
    edge ``a -> b`` carries an inheritance probability ``gamma`` that
    represents one component of ``b``'s parental-mass split.  After
    the insertion the new in-edge to ``b`` is ``c -> b`` (``a -> c``
    is just a fresh tree edge feeding the new internal node), so we
    propagate the original gamma to ``c -> b``.  Without this, every
    ``insert_node_in_edge`` call on a retic in-edge silently breaks
    the retic gamma sum invariant -- which is the root cause of the
    historical 14% bad-gamma rate in ``AddReticulation``.
    The new ``a -> c`` edge has no gamma (its retic-ness, if any,
    is the caller's responsibility to set explicitly).

    Args:
        edge (Edge): An edge, a -> b
        node (Node): A node, c.
        net (Network): The network that contains nodes a, b, and c
    Returns:
        N/A
    """
    a: Node = edge.src
    b: Node = edge.dest
    saved_gamma = edge.get_gamma()

    # Rewire the edges
    net.remove_edge(edge)
    net.add_edges(Edge(a, node))
    if saved_gamma is not None and b.is_reticulation():
        net.add_edges(Edge(node, b, gamma=saved_gamma))
    else:
        net.add_edges(Edge(node, b))
    
def connect_nodes(src: Node, dest: Node, net: Network) -> None:
    """
    Given two nodes in a network, connect them and check whether or not a 
    reticulation is created.

    Args:
        src (Node): The parent of the new edge
        dest (Node): The child of the new edge
        net (Network): Network for which to add the edge
    Returns:
        N/A
    """
    # Add the edge to the network
    net.add_edges(Edge(src, dest))
  
    # Check if dest is now a reticulation
    if net.in_degree(dest) > 1:
        dest.set_is_reticulation(True)

###########################
#### MOVE PARENT CLASS ####
###########################

##################
#### MOVE API ####
##################

class Move(ABC):
    """Abstract base class for every network proposal move.

    A :class:`Move` is a stateful object whose ``execute`` method
    mutates the ``Network`` held by a :class:`Model` and whose
    ``undo`` method restores the pre-execute state.  The contract
    is:

      * ``execute(model)`` performs the proposal and records just
        enough information (either compact deltas in
        ``same_move_info`` or a full ``copy.deepcopy`` snapshot in
        ``undo_info``) to later roll back or replay the move.
      * ``undo(model)`` returns the network to its pre-execute
        state.  Must be cheap and side-effect-free on anything
        else.
      * ``same_move(model)`` re-plays the proposal on a clone of
        the model, used by the parallel MCMC driver.
      * ``hastings_ratio()`` returns ``q(x | x') / q(x' | x)`` for
        proper MCMC; see the method-level docstring below for
        why every concrete move in this module currently returns
        ``1.0``.

    Subclasses are expected to leave the network in a
    structurally valid state (single root, acyclic, reticulations
    with in-degree 2, no orphan internals).  When a proposal is
    infeasible (e.g. no reticulations present for a retic-flip),
    the subclass should return silently rather than raise.
    """

    def __init__(self) -> None:
        """Initialise per-instance bookkeeping fields."""
        self.model = None
        self.undo_info = None
        self.same_move_info = None

    @abstractmethod
    def execute(self, model: Model) -> Model:
        """Apply the proposal to ``model.network`` in place.

        Args:
            model: Model whose ``network`` will be mutated.  The
                move uses ``model.rng`` (a seeded
                ``numpy.random.Generator``) for every random
                decision.

        Returns:
            The same ``model`` for chaining convenience.  The
            underlying network is mutated in place; no copy is
            returned.
        """
        pass

    @abstractmethod
    def undo(self, model: Model) -> None:
        """Undo the most recent call to :meth:`execute`.

        Args:
            model: The same model that was passed to
                :meth:`execute`.  After this call, the network
                is restored to its pre-execute state.
        """
        pass

    @abstractmethod
    def same_move(self, model: Model) -> None:
        """Replay the last proposal on an identical (cloned) model.

        Used by the parallel MCMC driver, which runs multiple
        chains with shared move sequences.  Concrete moves that
        use deep-copy ``undo_info`` typically implement this as a
        no-op, since replay is not meaningful without the exact
        random draws.

        Args:
            model: A topologically-equivalent clone of the model
                passed to :meth:`execute`.
        """
        pass
    
    def touched_nodes(self, net: Network) -> "set[Node] | None":
        """Network nodes whose incident branches or gammas this move modified.

        Consumed by scorers that maintain per-network-edge likelihood
        caches (see :class:`phynetpy._mcmc_gt._GTLikelihoodEngine`):
        any node returned here -- plus its ancestor path to the root
        -- will have its cached partials invalidated, everything else
        is reused across iterations.

        The default implementation returns ``None`` (fully dirty),
        which is the safe fallback for moves whose impact on the
        network structure is hard to localise.  Topology-preserving
        moves (``ChangeNodeHeight``, ``ChangeInheritanceProb``) should
        override and return the single node whose incident edges
        changed.

        Topology-changing moves (``SPR``, ``AddReticulation``,
        ``RemoveReticulation``, ``RelocateReticulation``,
        ``ChangeReticSource``, ``ChangeReticDest``,
        ``FlipReticulation``) generally keep the default ``None``:
        they replace edges entirely, which the engine's per-edge
        cache has to rebuild anyway.

        Args:
            net: Current network (i.e. the network after
                :meth:`execute`).  Overrides may use it to resolve
                stored labels to live :class:`Node` references.

        Returns:
            ``None`` for "fully dirty" (the engine rebuilds every
            cached partial), or a possibly-empty ``set[Node]`` of
            touched network nodes.
        """
        return None

    @abstractmethod
    def hastings_ratio(self) -> float:
        """Proposal asymmetry correction ``q(x|x') / q(x'|x)``.

        A proper MCMC sampler uses the Hastings ratio to correct for
        asymmetric proposal distributions so that the chain converges to
        the target distribution.  **Every concrete move in this module
        currently returns 1.0**, which would be exact only for strictly
        symmetric proposals.  Three are not:

        * :class:`SPR` weights candidate regraft edges by
          ``1 / d ** distance_decay`` of their topological distance from
          the prune point.  The reverse regraft probability depends on
          the hop-distance distribution *after* the move, which is not
          cheap to compute on a network.
        * :class:`ChangeNodeHeight` draws a Gaussian delta whose sigma
          is a fraction of the per-node *feasible* half-range; the
          half-range changes after the move, so proposals are mildly
          asymmetric even though the draw itself is symmetric.
        * :class:`ChangeInheritanceProb` clamps its Gaussian draw to the
          open interval ``(eps, 1 - eps)`` (rather than renormalising a
          truncated Gaussian).  This piles proposal mass on the boundaries.

        For the MPL search this module is driving, the Metropolis-Hastings
        criterion is applied to a *log-likelihood surface* being maximised
        under simulated annealing -- not to a stationary distribution we
        care about matching.  The un-corrected acceptance rule is still
        monotone in the score, so the search is still guaranteed to
        climb in expectation; the bias affects dwell-time distribution,
        which matters for Bayesian posteriors but not for finding the
        optimum.  Returning 1.0 is therefore a deliberate SA-scoped
        shortcut, not an oversight.  Bayesian callers should override
        these ratios before using the moves in an MCMC sampler.

        Returns:
            float: Hastings Ratio. ``1.0`` for symmetric moves and for
            the SA-approximate path used by ``MPL.search``.
        """
        pass


####GRAPH MOVES####
r"""
ALL OF THE FOLLOWING NETWORK MOVES HAVE VARIABLE NAMES THAT ARE BASED OFF OF THIS BASIC NETWORK STRUCTURE:

                  a
                    \
                     \  
                      \
            x          z  
           / \        / \
          /   \      /   \
         /     \    /     \
        /        c         \
       /         |          \
      /          |           \
                 y            b

"""


class AddReticulation(Move):
    """
    A move that adds a reticulation to a network.

    Host edges are split at a random point so their original branch
    length is preserved across the two halves.  The new reticulation
    edge is given a short length drawn from Exp(mean=0.1*min_host_len)
    and both parent edges of the new reticulation node receive
    gamma = 0.5.

    Uses deep-copy for undo/same_move to guarantee correctness.
    """

    def __init__(self, max_reticulations: int | None = None) -> None:
        """Create a new ``AddReticulation`` proposal.

        Args:
            max_reticulations: Optional hard cap on the number of
                reticulations.  If the current network already has
                ``max_reticulations`` reticulation nodes, the
                proposal silently returns without modifying the
                network.  ``None`` disables the cap.
        """
        super().__init__()
        self._max_retics = max_reticulations

    def execute(self, model: Model) -> Model:
        """Insert a new reticulation into ``model.network``.

        Args:
            model: Model whose ``network`` will be mutated in place.

        Returns:
            The same ``model`` (mutated) for chaining convenience.
        """
        net: Network = model.network

        # Respect the per-move cap on reticulation count, if set.
        if self._max_retics is not None:
            cur_retics = sum(1 for n in net.V() if n.is_reticulation())
            if cur_retics >= self._max_retics:
                return model

        self.undo_info = copy.deepcopy(net)

        rng = model.rng
        all_edges = net.E()
        src_e = _rng_pick(rng, all_edges)
        if src_e is None:
            model.update_network()
            return model
        avoid_these_edges = net.edges_upstream_of_node(src_e.src)
        eligible = [e for e in all_edges if e not in avoid_these_edges]
        if not eligible:
            model.update_network()
            return model
        dest_e = _rng_pick(rng, eligible)

        a: Node = src_e.src
        b: Node = src_e.dest
        x: Node = dest_e.src
        y: Node = dest_e.dest
        len_ab: float = src_e.get_length() or 1.0
        len_xy: float = dest_e.get_length() or 1.0

        z: Node = net.add_uid_node()
        c: Node = net.add_uid_node()
        c.set_is_reticulation(True)

        split_ab = float(rng.random())
        split_xy = float(rng.random())
        retic_len = float(rng.exponential(max(0.1 * min(len_ab, len_xy), 1e-6)))

        if a == x and b == y:
            # Bubble
            insert_node_in_edge(net.get_edge(a, b), z, net)
            net.get_edge(a, z).set_length(len_ab * split_ab)
            remaining = len_ab * (1.0 - split_ab)
            net.get_edge(z, b).set_length(remaining)

            insert_node_in_edge(net.get_edge(z, b), c, net)
            tree_zc = net.get_edge(z, c)
            tree_zc.set_length(remaining * split_xy)
            tree_zc.set_gamma(0.5)
            net.get_edge(c, b).set_length(remaining * (1.0 - split_xy))

            net.add_edges(Edge(z, c, length=retic_len, gamma=0.5))
            c.set_is_reticulation(True)
        else:
            # Standard case
            insert_node_in_edge(net.get_edge(x, y), c, net)
            net.get_edge(x, c).set_length(len_xy * split_xy)
            net.get_edge(c, y).set_length(len_xy * (1.0 - split_xy))
            net.get_edge(x, c).set_gamma(0.5)

            insert_node_in_edge(net.get_edge(a, b), z, net)
            net.get_edge(a, z).set_length(len_ab * split_ab)
            net.get_edge(z, b).set_length(len_ab * (1.0 - split_ab))

            retic_edge = Edge(z, c, length=retic_len, gamma=0.5)
            net.add_edges(retic_edge)
            c.set_is_reticulation(True)

        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model: Model) -> None:
        pass

    def hastings_ratio(self) -> float:
        return 1.0


class RemoveReticulation(Move):
    """Remove a reticulation node from the network.

    Algorithm:
      1. Pick a random reticulation node c (in=2, out=1).
      2. Randomly choose one of its two parent edges to delete.
      3. Remove that parent edge.  c now has (in=1, out=1) -- suppress c.
      4. The source of the deleted edge may now be degree-2 -- suppress it.

    Uses deep-copy for undo/same_move to guarantee correctness.
    """

    def __init__(self) -> None:
        super().__init__()

    def execute(self, model: Model) -> Model:
        net: Network = model.network
        self.undo_info = copy.deepcopy(net)

        retic_nodes = [n for n in net.V()
                       if n.is_reticulation() and net.in_degree(n) == 2]
        if not retic_nodes:
            return model

        rng = model.rng
        c: Node = _rng_pick(rng, retic_nodes)
        parents = net.get_parents(c)
        if len(parents) != 2:
            return model

        drop_parent: Node = _rng_pick(rng, parents)
        drop_edge = net.get_edge(drop_parent, c)

        net.remove_edge(drop_edge)

        c.set_is_reticulation(False)
        _suppress_deg2(net, c)

        _suppress_deg2(net, drop_parent)

        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model: Model) -> None:
        pass

    def hastings_ratio(self) -> float:
        return 1.0


class FlipReticulation(Move):
    """Flip the direction of one reticulation edge.

    Picks a random reticulation edge ``z -> c`` and reverses it to
    ``c -> z``, provided the result is still a valid DAG.  After the
    flip, ``z`` becomes a reticulation (in-degree 2) and ``c`` may
    lose its reticulation status.

    Invariants enforced on the candidate edge before the flip:

      * ``c`` must be a reticulation (so it currently has in-degree
        2 -- otherwise the flip wouldn't produce a well-defined
        reticulation swap).
      * ``z`` must not be the root (the root cannot become a
        reticulation).
      * ``z`` must not already be a reticulation and must have
        in-degree <= 1 (otherwise the flip would push ``z``'s
        in-degree to 3, an invalid structure that the acyclicity
        check does not catch).

    If the post-flip network is cyclic (rare but possible through
    other paths), the move rolls itself back via deep-copy undo.
    """

    def __init__(self) -> None:
        """Initialise with no per-instance parameters (see :class:`Move`)."""
        super().__init__()

    def execute(self, model: Model) -> Model:
        """Propose a single reticulation-edge flip on ``model.network``.

        Args:
            model: Model whose ``network`` will be mutated in place.

        Returns:
            The same ``model`` (mutated) for chaining convenience.
        """
        net: Network = model.network
        self.undo_info = copy.deepcopy(net)

        root = net.root()
        # Enumerate edges that are safe to flip (see class docstring
        # for the invariants filtered here).
        retic_edges = [e for e in net.E()
                       if e.dest.is_reticulation()
                       and e.src is not root
                       and not e.src.is_reticulation()
                       and net.in_degree(e.src) <= 1]
        if not retic_edges:
            return model

        retic_edge: Edge = _rng_pick(model.rng, retic_edges)
        z: Node = retic_edge.src
        c: Node = retic_edge.dest
        saved_gamma = retic_edge.get_gamma()
        saved_length = retic_edge.get_length()

        # Reverse the edge and propagate reticulation flags.
        net.remove_edge(retic_edge)
        net.add_edges(Edge(c, z, length=saved_length, gamma=saved_gamma))

        if net.in_degree(c) <= 1:
            c.set_is_reticulation(False)
        if net.in_degree(z) > 1:
            z.set_is_reticulation(True)
            # Reticulation invariant: gammas on z's two in-edges
            # must sum to 1.0.  The flipped edge ``c -> z`` carries
            # ``saved_gamma``, but z's pre-existing in-edge was a
            # tree edge with no gamma set (or 0.0 by default).  Now
            # that z is a reticulation we have to assign the
            # complementary mass on that edge -- otherwise the AC DP
            # scores z with a one-sided split and every proposal
            # collapses to the log floor.  This was the root cause
            # of the historical 0% accept rate for this move
            # (verified by ``runs/diag_retic_fix_proof.py``).
            sg = saved_gamma if saved_gamma is not None else 0.5
            other_gamma = max(0.0, min(1.0, 1.0 - sg))
            for in_e in net.in_edges(z):
                if in_e.src is c:
                    continue
                g = in_e.get_gamma()
                if g is None or g == 0.0 or not (0.0 < g < 1.0):
                    in_e.set_gamma(other_gamma)
                    break

        # The above invariant filters catch the common invalid
        # cases, but a cycle can still arise in rare topologies;
        # fall back cleanly when it does.
        if not net.is_acyclic():
            model.network = self.undo_info
            self.undo_info = None
            model.update_network()
            return model

        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model: Model) -> None:
        pass

    def hastings_ratio(self) -> float:
        return 1.0
    

class SwitchParentage(Move):
    """
    PSPP (Parentage-Switching for Polyploid Phylogenetics) move for 
    Infer_MP_Allop. Alters the genetic parentage of a subnetwork while 
    maintaining the same ploidy values for each leaf.
    
    Uses deep-copy for undo/same_move to guarantee correctness.
    """
    def __init__(self, debug_id: int = 0) -> None:
        super().__init__()
        self.valid_attachment_edges: list[Edge] = []
        self.logger = Logger(str(debug_id))
        
    def _random_choice(self, mylist: list, rng: np.random.Generator) -> Any:
        """Select a random element from a list, or None if empty."""
        if not mylist:
            return None
        return mylist[int(rng.integers(0, len(mylist)))]
    
    def execute(self, model: Model) -> Model:
        """
        Executes the PSPP (Switch-Parentage) move.

        Args:
            model (Model): A model object with a populated network field.
        Returns:
            Model: The modified model with a newly proposed network topology.
        """
        net: Network = model.network
        self.undo_info = copy.deepcopy(net)
        self.valid_attachment_edges = []
        
        # STEP 1: Select random non-root node
        non_root_nodes = [node for node in net.V() if node != net.root()]
        if not non_root_nodes:
            model.update_network()
            return model
            
        node_2_change: Node = self._random_choice(non_root_nodes, model.rng)
        if node_2_change is None:
            model.update_network()
            return model
    
        # Skip pointless changes: a child of root whose sibling is a leaf
        node_pars = net.get_parents(node_2_change)
        if len(node_pars) == 1:
            root_node = net.root()
            if node_pars[0] == root_node:
                root_kids = net.get_children(root_node)
                other_kids = [n for n in root_kids if n != node_2_change]
                if other_kids and net.out_degree(other_kids[0]) == 0:
                    model.update_network()
                    return model
            
        # STEP 2: Record target subgenome count before modification
        target: int = net.subgenome_count(node_2_change)
        
        # STEP 3: Remove a parent edge
        in_edges = list(net.in_edges(node_2_change))
        if not in_edges:
            model.update_network()
            return model
            
        edge_2_remove: Edge = self._random_choice(in_edges, model.rng)
        if edge_2_remove is None:
            model.update_network()
            return model
            
        self._delete_parent_edge(net, edge_2_remove)
        
        # STEP 4: Reconnect to achieve target subgenome count
        original_node: Node = node_2_change
        cur_ct = net.subgenome_count(original_node) if net.in_degree(original_node) >= 1 else 0
        is_first_iter = True
        
        for _ in range(100):
            if cur_ct == target:
                break
                
            if not is_first_iter:
                if not self.valid_attachment_edges:
                    break
                branch = self._random_choice(self.valid_attachment_edges, model.rng)
                if branch is None:
                    break
                    
                intermediate_node = net.add_uid_node()
                net.remove_edge(branch)
                self.valid_attachment_edges.remove(branch)
                
                e1 = Edge(branch.src, intermediate_node)
                e2 = Edge(intermediate_node, branch.dest)
                net.add_edges(e1)
                net.add_edges(e2)
                self.valid_attachment_edges.extend([e1, e2])
                downstream_node = intermediate_node
            else:
                downstream_node = original_node
                
            # Find a root for BFS (handle disconnected components)
            bfs_starts = [n for n in net.V()
                          if net.in_degree(n) == 0 and net.out_degree(n) != 0]
            if len(bfs_starts) > 1 and original_node in bfs_starts:
                bfs_starts.remove(original_node)
            if not bfs_starts:
                break
                
            edges_to_ct = net.edges_to_subgenome_count(
                downstream_node, target - cur_ct, bfs_starts[0])
        
            if not edges_to_ct:
                break
                
            random_key = self._random_choice(list(edges_to_ct.keys()), model.rng)
            if random_key is None:
                break
            
            edge_list = list(edges_to_ct[random_key])
            if not edge_list:
                break
                
            new_edge = self._random_choice(edge_list, model.rng)
            if new_edge is None:
                break
            
            # Insert connector node and attach
            connector_node = net.add_uid_node()
            attach_target = downstream_node
            e_to_child = Edge(connector_node, new_edge.dest)
            e_from_parent = Edge(new_edge.src, connector_node)
            e_to_target = Edge(connector_node, attach_target)
            
            net.add_edges([e_to_child, e_from_parent, e_to_target])
            self.valid_attachment_edges.append(e_to_target)
            net.remove_edge(new_edge)
            
            if net.in_degree(attach_target) > 1:
                attach_target.set_is_reticulation(True)
            
            cur_ct = net.subgenome_count(original_node)
            is_first_iter = False
            
        net.clean()
        self._reconcile_reticulation_flags(net)
        model.update_network()
        self.same_move_info = copy.deepcopy(net)
        return model

    def undo(self, model: Model) -> None:
        """Restores the network to its pre-move state."""
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model: Model) -> None:
        """Replays the same topology change on a different model instance."""
        if self.same_move_info is not None:
            model.network = copy.deepcopy(self.same_move_info)
        model.update_network()
    
    def hastings_ratio(self) -> float:
        return 1.0
         
    def _reconcile_reticulation_flags(self, net: Network) -> None:
        """
        Set each node's reticulation flag to match its actual in-degree.
        This corrects any stale flags left after topology edits or clean().
        """
        for node in net.V():
            node.set_is_reticulation(net.in_degree(node) > 1)

    def _delete_parent_edge(self, net: Network, edge: Edge) -> None:
        """
        Removes a parent edge from the network and cleans up cascading
        degree-1 nodes and reticulation chains.

        Args:
            net (Network): The network object.
            edge (Edge): The edge to delete (must be an actual edge in net).
        """
        target_node = edge.dest
        parent_node = edge.src

        net.remove_edge(edge)
        
        if target_node.is_reticulation() and net.in_degree(target_node) == 1:
            target_node.set_is_reticulation(False)

        # Clean up parent_node if it became degree-1 internal
        self._cleanup_node(net, parent_node)
    
    def _cleanup_node(self, net: Network, node: Node) -> None:
        """
        Remove or bypass a node that has become topologically degenerate
        after an edge deletion.  Handles:

        - passthrough  (in=1, out=1)  → bypass
        - orphan root  (in=0, out=1)  → remove, disconnect child
        - isolated      (in=0, out=0)  → remove
        - dead-end      (in>=1, out=0, non-leaf)  → remove, recurse parents

        The dead-end case covers reticulation nodes whose only child edge
        was deleted, leaving them with parents but no children.
        """
        in_deg = net.in_degree(node)
        out_deg = net.out_degree(node)
        
        if node == net.root():
            return
        
        if in_deg == 1 and out_deg == 1:
            parent = net.get_parents(node)[0]
            child = net.get_children(node)[0]
            
            parent_edge = net.get_edge(parent, node)
            child_edge = net.get_edge(node, child)
            
            net.remove_edge(parent_edge)
            net.remove_edge(child_edge)
            net.remove_nodes(node)
            net.add_edges(Edge(parent, child))
            
            if parent.is_reticulation() and net.in_degree(parent) == 1:
                parent.set_is_reticulation(False)
                self._cleanup_node(net, parent)
        
        elif in_deg == 0 and out_deg == 1:
            child = net.get_children(node)[0]
            child_edge = net.get_edge(node, child)
            net.remove_edge(child_edge)
            net.remove_nodes(node)
        
        elif in_deg == 0 and out_deg == 0:
            net.remove_nodes(node)

        elif in_deg >= 1 and out_deg == 0:
            # Dead-end internal node: has parent(s) but no children.
            # _cleanup_node only recurses upward through parents, so it
            # never encounters a genuine leaf taxon here.
            parents = list(net.get_parents(node))
            for p in parents:
                net.remove_edge(net.get_edge(p, node))
            node.set_is_reticulation(False)
            net.remove_nodes(node)
            for p in parents:
                if p.is_reticulation() and net.in_degree(p) == 1:
                    p.set_is_reticulation(False)
                self._cleanup_node(net, p)


class SPR(Move):
    """Subtree Prune and Regraft on a phylogenetic network.

    Network-safe algorithm:
      1. Select a random edge (u -> v) to prune, subject to:
         - v is not a leaf
         - v is not a reticulation (we don't detach reticulation children)
         - u is not a reticulation with out-degree 1 (would leave it (2,0))
         - u is not root with out-degree 2 (would leave root with 1 child)
      2. Remove the edge (u -> v).
      3. Repair u: if u becomes (1,1), suppress u (merge incident edges).
      4. Collect subtree nodes rooted at v (used to prevent cycles).
      5. Select a target edge (a -> b) where neither endpoint is in the
         subtree (prevents cycles).  Edges are weighted by inverse
         topological distance from the prune point so that nearby
         regraft locations are strongly preferred.
      6. Subdivide (a -> b) by inserting a new node w:
         a -> w -> b, with split branch lengths.
      7. Attach: w -> v.  w is now (1,2) -- a valid internal node.

    Uses deep-copy for undo/same_move to guarantee correctness.
    """

    # Default decay matches the prior class-constant value.  ``MPLKernel``
    # now adapts this per-proposal: high decay == tight local regrafts,
    # low decay == broad search.  Bumping the decay means edges at
    # distance d get sampled with weight ``1 / d**decay``, so decay=2
    # strongly favors nearby regrafts, decay=0.5 is nearly flat.
    _DEFAULT_DISTANCE_DECAY = 2.0

    def __init__(self,
                 distance_decay: float | None = None,
                 debug_id: int = 0) -> None:
        super().__init__()
        self.undo_info = None
        self.same_move_info = None
        self._distance_decay = (
            self._DEFAULT_DISTANCE_DECAY
            if distance_decay is None
            else float(distance_decay)
        )

    @staticmethod
    def _hop_distances(net: Network, origin: Node,
                       forbidden: set[Node]) -> dict[Node, int]:
        """BFS hop-distance from *origin* ignoring forbidden nodes."""
        dist: dict[Node, int] = {origin: 0}
        q: deque[Node] = deque([origin])
        while q:
            cur = q.popleft()
            d = dist[cur] + 1
            for nbr in list(net.get_children(cur)) + list(net.get_parents(cur)):
                if nbr in forbidden or nbr in dist:
                    continue
                dist[nbr] = d
                q.append(nbr)
        return dist

    def execute(self, model: Model) -> Model:
        net: Network = model.network
        self.undo_info = copy.deepcopy(net)

        root = net.root()
        prunable = []
        for e in net.E():
            u, v = e.src, e.dest
            if v in net.get_leaves():
                continue
            if v.is_reticulation():
                continue
            if u == root and net.out_degree(root) <= 2:
                continue
            if u.is_reticulation() and net.out_degree(u) <= 1:
                continue
            prunable.append(e)

        if not prunable:
            return model

        rng = model.rng
        prune_edge: Edge = _rng_pick(rng, prunable)
        prune_parent: Node = prune_edge.src
        subtree_root: Node = prune_edge.dest
        prune_len: float = prune_edge.get_length() or 1.0

        subtree_nodes = net.get_subtree_at(subtree_root)

        net.remove_edge(prune_edge)

        _suppress_deg2(net, prune_parent)

        forbidden = subtree_nodes
        eligible = [e for e in net.E()
                    if e.src not in forbidden and e.dest not in forbidden]
        if not eligible:
            model.network = self.undo_info
            self.undo_info = None
            model.update_network()
            return model

        hop = self._hop_distances(net, prune_parent, forbidden)
        weights: list[float] = []
        for e in eligible:
            d_src = hop.get(e.src, len(hop))
            d_dst = hop.get(e.dest, len(hop))
            d = min(d_src, d_dst) + 1
            weights.append(1.0 / (d ** self._distance_decay))

        target_edge: Edge = _rng_pick_weighted(rng, eligible, weights)
        a: Node = target_edge.src
        b: Node = target_edge.dest
        target_len: float = target_edge.get_length() or 1.0

        new_node: Node = net.add_uid_node()
        split = float(rng.random())
        insert_node_in_edge(target_edge, new_node, net)
        net.get_edge(a, new_node).set_length(target_len * split)
        net.get_edge(new_node, b).set_length(target_len * (1.0 - split))

        net.add_edges(Edge(new_node, subtree_root, length=prune_len))

        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model: Model) -> None:
        pass

    def hastings_ratio(self) -> float:
        # Distance-weighted regraft is *not* symmetric: P(regraft | prune)
        # and P(prune | regraft) depend on different hop-distance
        # distributions on the pre/post-move topology.  See the base-
        # class docstring for why the SA path accepts ``1.0`` anyway.
        return 1.0


################################
#### PARAMETER-LEVEL MOVES #####
################################
#
# Several moves need small structural surgery helpers below:
#
#   _suppress_deg2:     collapse an in=1 / out=1 passthrough node
#                       by merging its incident edges.
#   _prune_orphan_chain: remove a dead-end internal node plus any
#                       ancestors that become dead-ends, suppressing
#                       deg-2 parents along the way.
#
# These helpers are used by RelocateReticulation, ChangeReticSource,
# ChangeReticDest, and friends to keep the network structurally valid
# after edges are detached.
#
################################

def _suppress_deg2(net: Network, node: Node) -> None:
    """Collapse a degree-2 passthrough node into a single edge.

    ``node`` must have in-degree 1 and out-degree 1; if not, this is
    a no-op.  The two incident edges are merged into one whose
    branch length is the sum of the originals.  Gamma is inherited
    from the child-side edge when present, falling back to the
    parent-side edge.

    Args:
        net: Network containing ``node``.
        node: Candidate passthrough node.
    """
    if net.in_degree(node) != 1 or net.out_degree(node) != 1:
        return
    parent = net.get_parents(node)[0]
    child = net.get_children(node)[0]
    e_up = net.get_edge(parent, node)
    e_dn = net.get_edge(node, child)
    len_up = e_up.get_length() or 0.0
    len_dn = e_dn.get_length() or 0.0
    new_gamma = e_dn.get_gamma() if e_dn.get_gamma() else e_up.get_gamma()
    net.remove_edge(e_up)
    net.remove_edge(e_dn)
    net.remove_nodes(node)
    net.add_edges(Edge(parent, child, length=len_up + len_dn, gamma=new_gamma))


def _prune_orphan_chain(net: Network, node: Node) -> None:
    """Iteratively remove non-root dead-end internal nodes.

    Several moves detach an edge whose source had out-degree 1.
    The source is then left with in-degree >= 1 and out-degree 0
    (an "orphan internal" node) -- MPL scoring typically NaNs on
    networks in this state, and even when it doesn't, the orphan
    chain wastes proposals.

    This helper walks up from ``node``, removing each dead-end and
    its incoming edges and collapsing any parents that become
    degree-2 passthroughs.  The root is never pruned; if the chain
    reaches an in-degree-0 node, we stop.  Safe against parent
    nodes that have already been removed by an earlier recursion
    (e.g. "bubble" reticulations where both parents coincide).

    Args:
        net: Network to repair.
        node: Starting dead-end node.
    """
    stack = [node]
    while stack:
        n = stack.pop()
        # The same parent may appear in multiple orphan chains and could
        # have been removed already; skip nodes that are no longer in V().
        if n not in net.V():
            continue
        if net.in_degree(n) == 0:
            continue
        if net.out_degree(n) != 0:
            continue
        parents = list(net.get_parents(n))
        for e in list(net.in_edges(n)):
            net.remove_edge(e)
        try:
            net.remove_nodes(n)
        except Exception:
            pass
        for p in parents:
            if p not in net.V():
                continue
            if net.in_degree(p) > 0 and net.out_degree(p) == 0:
                stack.append(p)
            elif net.in_degree(p) == 1 and net.out_degree(p) == 1:
                _suppress_deg2(net, p)


class ChangeNodeHeight(Move):
    """Slide an internal node up or down by adjusting incident branch lengths.

    A random internal (non-root, non-leaf) node is chosen and shifted by
    a Gaussian delta.  The step standard deviation is ``sigma_frac``
    times the feasible half-range (the maximum shift that doesn't push
    any incident branch below ``_EPSILON``), so proposals stay in
    bounds naturally while still concentrating mass around the current
    height.  ``sigma_frac`` is meant to be tuned by ``MPLKernel``'s
    adaptive scheduler so the acceptance rate tracks a target.

    The uniform proposal it replaced was equivalent to
    ``sigma_frac ≈ 0.58`` (sd of a uniform is ``half_range/sqrt(3)``)
    and always ignored acceptance feedback.  Starting at ``0.4`` gives
    a tighter, higher-acceptance default that the adaptive layer can
    still crank back up when useful.
    """

    _EPSILON = 1e-6
    _DEFAULT_SIGMA_FRAC = 0.4
    # Bound the truncation fallback.  If the random Gaussian draw lands
    # outside [lower, upper], resample up to this many times before
    # clamping.  Three is enough in practice for sigma_frac <= 1.
    _TRUNCATE_RETRIES = 3

    def __init__(self, sigma_frac: float | None = None) -> None:
        super().__init__()
        self._sigma_frac = (
            self._DEFAULT_SIGMA_FRAC if sigma_frac is None else float(sigma_frac)
        )

    def execute(self, model: Model) -> Model:
        net: Network = model.network

        candidates = [
            n for n in net.V()
            if net.out_degree(n) > 0 and net.in_degree(n) > 0
        ]
        if not candidates:
            model.update_network()
            return model

        rng = model.rng
        node = _rng_pick(rng, candidates)
        parent_edges = list(net.in_edges(node))
        child_edges = list(net.out_edges(node))

        if not parent_edges or not child_edges:
            model.update_network()
            return model

        min_parent_len = min(e.get_length() for e in parent_edges)
        min_child_len = min(e.get_length() for e in child_edges)

        lower = -(min_parent_len - self._EPSILON)
        upper = min_child_len - self._EPSILON

        if lower >= upper:
            model.update_network()
            return model

        half_range = 0.5 * (upper - lower)
        sigma = self._sigma_frac * half_range
        delta = float(rng.normal(0.0, sigma))
        if not (lower <= delta <= upper):
            for _ in range(self._TRUNCATE_RETRIES):
                delta = float(rng.normal(0.0, sigma))
                if lower <= delta <= upper:
                    break
            else:
                # Clip as a last resort.  Introduces a tiny bias toward
                # the bounds but keeps the move structurally valid.
                delta = max(lower, min(upper, delta))

        edge_changes: list[tuple[str, str, float, float]] = []
        for e in parent_edges:
            old_len = e.get_length()
            new_len = old_len + delta
            edge_changes.append((e.src.label, e.dest.label, old_len, new_len))
            e.set_length(new_len)
        for e in child_edges:
            old_len = e.get_length()
            new_len = old_len - delta
            edge_changes.append((e.src.label, e.dest.label, old_len, new_len))
            e.set_length(new_len)

        self.undo_info = edge_changes
        self.same_move_info = edge_changes
        self._touched_label = node.label
        model.mark_touched(self.touched_nodes(model.network))
        return model

    def undo(self, model: Model) -> None:
        net: Network = model.network
        if self.undo_info is not None:
            for src_lbl, dest_lbl, old_len, _new_len in self.undo_info:
                src = net.has_node_named(src_lbl)
                dest = net.has_node_named(dest_lbl)
                if src is not None and dest is not None:
                    net.get_edge(src, dest).set_length(old_len)
        model.mark_touched(self.touched_nodes(model.network))

    def same_move(self, model: Model) -> None:
        net: Network = model.network
        if self.same_move_info is not None:
            for src_lbl, dest_lbl, _old_len, new_len in self.same_move_info:
                src = net.has_node_named(src_lbl)
                dest = net.has_node_named(dest_lbl)
                if src is not None and dest is not None:
                    net.get_edge(src, dest).set_length(new_len)
        model.mark_touched(self.touched_nodes(model.network))

    def touched_nodes(self, net: Network) -> "set[Node] | None":
        """Return the single node whose height changed.

        Sliding one internal node by a Gaussian delta only modifies
        its incident parent / child branch lengths.  For the MCMC_GT
        engine this means only the DP partials on the touched node's
        ancestor-to-root path need recomputation.
        """
        label = getattr(self, "_touched_label", None)
        if label is None:
            return None
        node = net.has_node_named(label)
        if node is None:
            return None
        return {node}

    def hastings_ratio(self) -> float:
        # Symmetric Gaussian *draw*, but the sigma scales with the
        # feasible half-range, which differs pre/post-move.  The
        # residual asymmetry is small in practice.  See ``Move.hastings_ratio``
        # for the SA rationale.
        return 1.0


class ChangeInheritanceProb(Move):
    """Propose a new inheritance probability for a random reticulation node.

    The two parent edges of a reticulation always carry complementary
    gammas (gamma and 1-gamma).  The new value is drawn from a
    Gaussian centered on the current gamma, truncated to (epsilon,
    1-epsilon), so proposals stay close to the current value and
    accept more often than a flat Uniform(0, 1) draw.
    """

    _SIGMA = 0.1
    _EPS = 0.01

    def __init__(self, sigma: float | None = None) -> None:
        super().__init__()
        if sigma is not None:
            self._sigma = sigma
        else:
            self._sigma = self._SIGMA

    def execute(self, model: Model) -> Model:
        net: Network = model.network

        retic_nodes = [
            n for n in net.V() if net.in_degree(n) == 2 and n.is_reticulation()
        ]
        if not retic_nodes:
            model.update_network()
            return model

        rng = model.rng
        node = _rng_pick(rng, retic_nodes)
        parent_edges = list(net.in_edges(node))
        if len(parent_edges) != 2:
            model.update_network()
            return model

        e1, e2 = parent_edges
        old_g1 = e1.get_gamma()
        old_g2 = e2.get_gamma()

        current = old_g1 if old_g1 is not None else 0.5
        new_g1 = float(rng.normal(current, self._sigma))
        new_g1 = max(self._EPS, min(1.0 - self._EPS, new_g1))
        e1.set_gamma(new_g1)
        e2.set_gamma(1.0 - new_g1)

        self.undo_info = (e1.src.label, e1.dest.label, old_g1,
                          e2.src.label, e2.dest.label, old_g2)
        self.same_move_info = (e1.src.label, e1.dest.label, new_g1,
                               e2.src.label, e2.dest.label, 1.0 - new_g1)
        self._touched_label = node.label
        model.mark_touched(self.touched_nodes(model.network))
        return model

    def undo(self, model: Model) -> None:
        net: Network = model.network
        if self.undo_info is not None:
            s1, d1, g1, s2, d2, g2 = self.undo_info
            n_s1, n_d1 = net.has_node_named(s1), net.has_node_named(d1)
            n_s2, n_d2 = net.has_node_named(s2), net.has_node_named(d2)
            if n_s1 is not None and n_d1 is not None:
                net.get_edge(n_s1, n_d1).set_gamma(g1)
            if n_s2 is not None and n_d2 is not None:
                net.get_edge(n_s2, n_d2).set_gamma(g2)
        model.mark_touched(self.touched_nodes(model.network))

    def same_move(self, model: Model) -> None:
        net: Network = model.network
        if self.same_move_info is not None:
            s1, d1, g1, s2, d2, g2 = self.same_move_info
            n_s1, n_d1 = net.has_node_named(s1), net.has_node_named(d1)
            n_s2, n_d2 = net.has_node_named(s2), net.has_node_named(d2)
            if n_s1 is not None and n_d1 is not None:
                net.get_edge(n_s1, n_d1).set_gamma(g1)
            if n_s2 is not None and n_d2 is not None:
                net.get_edge(n_s2, n_d2).set_gamma(g2)
        model.mark_touched(self.touched_nodes(model.network))

    def touched_nodes(self, net: Network) -> "set[Node] | None":
        """Return the single reticulation whose inheritance prob changed.

        Updating the gamma on one reticulation's two parent edges
        only affects that node's incident branch weights -- the rest
        of the per-network-edge partial cache can be reused.
        """
        label = getattr(self, "_touched_label", None)
        if label is None:
            return None
        node = net.has_node_named(label)
        if node is None:
            return None
        return {node}

    def hastings_ratio(self) -> float:
        # Clamped-Gaussian proposal.  Proper H would be a truncated-
        # Gaussian normalisation ratio ``Z(old) / Z(new)`` where
        # ``Z(c) = Phi((1-eps-c)/sigma) - Phi((eps-c)/sigma)``.  The
        # current clamp-to-bounds behaviour additionally piles mass on
        # the interval endpoints; see ``Move.hastings_ratio`` for why the
        # SA path tolerates the resulting bias.
        return 1.0


class ChangeReticSource(Move):
    """Move the source (tail / parent) end of a reticulation edge.

    Picks a random reticulation edge ``z -> c``, detaches the source
    end, cleans up anything ``z`` leaves behind (deg-2 collapse or
    orphan-chain pruning), then reattaches ``c``'s parent from a
    freshly-inserted node split into a randomly chosen host edge.
    Cycle-safe: host edges downstream of ``c`` are filtered out.

    Uses deep-copy for undo/same_move to guarantee correctness.
    """

    def __init__(self) -> None:
        """Initialise with no per-instance parameters (see :class:`Move`)."""
        super().__init__()

    def execute(self, model: Model) -> Model:
        """Propose a source-relocation for one reticulation edge.

        Args:
            model: Model whose ``network`` will be mutated in place.

        Returns:
            The same ``model`` (mutated) for chaining convenience.
        """
        net: Network = model.network
        self.undo_info = copy.deepcopy(net)

        retic_edges = [e for e in net.E() if e.dest.is_reticulation()]
        if not retic_edges:
            model.update_network()
            return model

        rng = model.rng
        retic_edge = _rng_pick(rng, retic_edges)
        z: Node = retic_edge.src
        c: Node = retic_edge.dest
        saved_gamma = retic_edge.get_gamma()
        saved_length = retic_edge.get_length()

        net.remove_edge(retic_edge)

        # Detaching the source can leave ``z`` in an invalid state.
        # Two cases:
        #   - out_degree(z) == 0 (orphan internal): recursively prune
        #     up the chain so scoring never sees the dead end.
        #   - in=1 / out=1 (passthrough): collapse to a single edge.
        if net.in_degree(z) > 0 and net.out_degree(z) == 0:
            _prune_orphan_chain(net, z)
        elif net.in_degree(z) == 1 and net.out_degree(z) == 1:
            _suppress_deg2(net, z)

        forbidden = set(net.edges_downstream_of_node(c))
        forbidden.update(net.in_edges(c))
        forbidden.update(net.out_edges(c))
        eligible = [e for e in net.E() if e not in forbidden]

        if not eligible:
            model.network = self.undo_info
            model.update_network()
            self.undo_info = None
            return model

        host = _rng_pick(rng, eligible)
        a, b = host.src, host.dest
        z_new = net.add_uid_node()
        host_len = host.get_length()
        split = float(rng.random())

        net.remove_edge(host)
        net.add_edges(Edge(a, z_new, length=host_len * split))
        net.add_edges(Edge(z_new, b, length=host_len * (1.0 - split)))
        net.add_edges(Edge(z_new, c, length=saved_length, gamma=saved_gamma))

        if net.in_degree(c) > 1:
            c.set_is_reticulation(True)

        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model: Model) -> None:
        pass

    def hastings_ratio(self) -> float:
        return 1.0


class RelocateReticulation(Move):
    """Atomically relocate a reticulation node end-to-end.

    Picks a random reticulation node, detaches both incoming edges
    and the single outgoing edge, cleans up any passthroughs or
    orphan chains left behind, and then reattaches a fresh
    reticulation on two newly split host edges.

    This is equivalent to ``ChangeReticSource + ChangeReticDest``
    in a single atomic step, avoiding the deep likelihood valley
    that would arise from the intermediate partially-detached
    state.  Cycle-safe via a ``descendants_of_child`` filter on
    host edges.

    Uses deep-copy for undo to guarantee correctness.
    """

    def __init__(self) -> None:
        """Initialise with no per-instance parameters (see :class:`Move`)."""
        super().__init__()

    def execute(self, model: Model) -> Model:
        """Propose a whole-reticulation relocation.

        Args:
            model: Model whose ``network`` will be mutated in place.

        Returns:
            The same ``model`` (mutated) for chaining convenience.
        """
        net: Network = model.network
        self.undo_info = copy.deepcopy(net)

        retic_nodes = [n for n in net.V() if n.is_reticulation()]
        if not retic_nodes:
            model.update_network()
            return model

        rng = model.rng
        retic = _rng_pick(rng, retic_nodes)
        in_edges = list(net.in_edges(retic))
        if len(in_edges) != 2:
            model.update_network()
            return model

        children = net.get_children(retic)
        if not children:
            model.update_network()
            return model
        child = children[0]

        e_child = net.get_edge(retic, child)
        saved_child_len = (
            e_child.get_length() if e_child is not None else None
        )

        saved_gammas = [e.get_gamma() for e in in_edges]
        saved_lengths = [e.get_length() for e in in_edges]
        old_parents = [e.src for e in in_edges]

        net.remove_edge(e_child)
        for e in in_edges:
            net.remove_edge(e)

        retic.set_is_reticulation(False)
        net.remove_nodes(retic)

        # Parents can repeat here when the retic is a "bubble" (both
        # in-edges from the same node), and a recursive prune of the
        # first occurrence may remove the parent before the second pass.
        seen_parents: set = set()
        for p in old_parents:
            if p in seen_parents or p not in net.V():
                continue
            seen_parents.add(p)
            if net.in_degree(p) > 0 and net.out_degree(p) == 0:
                _prune_orphan_chain(net, p)
            elif net.in_degree(p) == 1 and net.out_degree(p) == 1:
                _suppress_deg2(net, p)

        if net.in_degree(child) == 1 and net.out_degree(child) == 1:
            if not child.is_reticulation():
                _suppress_deg2(net, child)

        # Cycle-safety filter.  After the move we insert z1 on host1 and
        # z2 on host2, and wire z1/z2 -> new_retic -> child.  A cycle
        # forms iff ``child`` can reach z1 or z2, which happens iff a1
        # or a2 is a descendant of ``child``.  Forbid any host edge
        # whose source is reachable from ``child`` -- this is the same
        # check ``ChangeReticSource`` uses and is the fix for the
        # previous ``downstream_of_b1`` filter, which was filtering
        # against the wrong anchor and let ~20% of proposals produce
        # cyclic networks in realistic topologies.
        forbidden = set(net.edges_downstream_of_node(child))
        eligible = [e for e in net.E() if e not in forbidden]
        if len(eligible) < 2:
            model.network = self.undo_info
            model.update_network()
            self.undo_info = None
            return model

        host1 = _rng_pick(rng, eligible)
        a1, b1 = host1.src, host1.dest

        eligible2 = [e for e in eligible if e is not host1]
        if not eligible2:
            model.network = self.undo_info
            model.update_network()
            self.undo_info = None
            return model

        host2 = _rng_pick(rng, eligible2)
        a2, b2 = host2.src, host2.dest

        new_retic = net.add_uid_node()
        new_retic.set_is_reticulation(True)

        z1 = net.add_uid_node()
        h1_len = host1.get_length()
        split1 = float(rng.random())
        net.remove_edge(host1)
        net.add_edges(Edge(a1, z1, length=(h1_len or 0) * split1))
        net.add_edges(Edge(z1, b1, length=(h1_len or 0) * (1 - split1)))

        z2 = net.add_uid_node()
        h2_len = host2.get_length()
        split2 = float(rng.random())
        try:
            net.remove_edge(host2)
        except Exception:
            # Defensive: if host2 was somehow invalidated by host1's
            # split, fall back cleanly rather than leaving a partial
            # edit.  With the ``descendants_of_child`` filter above this
            # path is not expected to fire, but we keep the fallback
            # for robustness.
            model.network = self.undo_info
            model.update_network()
            self.undo_info = None
            return model
        net.add_edges(Edge(a2, z2, length=(h2_len or 0) * split2))
        net.add_edges(Edge(z2, b2, length=(h2_len or 0) * (1 - split2)))

        net.add_edges(Edge(z1, new_retic,
                           length=saved_lengths[0], gamma=saved_gammas[0]))
        net.add_edges(Edge(z2, new_retic,
                           length=saved_lengths[1], gamma=saved_gammas[1]))

        net.add_edges(Edge(new_retic, child, length=saved_child_len))

        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model: Model) -> None:
        pass

    def hastings_ratio(self) -> float:
        return 1.0


class ChangeReticDest(Move):
    """Move the destination (head) end of a reticulation edge.

    Picks a random reticulation edge ``z -> c``, detaches the
    destination end, clears ``c``'s reticulation flag and collapses
    it if it became a deg-2 passthrough, then reattaches into a
    freshly-inserted reticulation node split onto a randomly chosen
    host edge.  Cycle-safe: edges upstream of ``z`` are filtered
    out.

    Uses deep-copy for undo/same_move to guarantee correctness.
    """

    def __init__(self) -> None:
        """Initialise with no per-instance parameters (see :class:`Move`)."""
        super().__init__()

    def execute(self, model: Model) -> Model:
        """Propose a destination-relocation for one reticulation edge.

        Args:
            model: Model whose ``network`` will be mutated in place.

        Returns:
            The same ``model`` (mutated) for chaining convenience.
        """
        net: Network = model.network
        self.undo_info = copy.deepcopy(net)

        retic_edges = [e for e in net.E() if e.dest.is_reticulation()]
        if not retic_edges:
            model.update_network()
            return model

        rng = model.rng
        retic_edge = _rng_pick(rng, retic_edges)
        z: Node = retic_edge.src
        c: Node = retic_edge.dest
        saved_gamma = retic_edge.get_gamma()
        saved_length = retic_edge.get_length()

        net.remove_edge(retic_edge)

        if net.in_degree(c) <= 1:
            c.set_is_reticulation(False)
        if net.in_degree(c) == 1 and net.out_degree(c) == 1:
            _suppress_deg2(net, c)

        forbidden = set(net.edges_upstream_of_node(z))
        forbidden.update(net.in_edges(z))
        forbidden.update(net.out_edges(z))
        eligible = [e for e in net.E() if e not in forbidden]

        if not eligible:
            model.network = self.undo_info
            model.update_network()
            self.undo_info = None
            return model

        host = _rng_pick(rng, eligible)
        a, b = host.src, host.dest
        c_new = net.add_uid_node()
        c_new.set_is_reticulation(True)
        host_len = host.get_length()
        split = float(rng.random())

        # Reticulation invariant: the two in-edges of ``c_new`` must
        # carry inheritance probabilities that sum to 1.0.  We
        # preserve ``saved_gamma`` on the redirected source edge
        # ``z -> c_new`` and assign the complementary mass to the
        # new tree-derived in-edge ``a -> c_new``.  Falling back to
        # 0.5 when ``saved_gamma`` is None handles malformed inputs
        # without crashing.  Without this, the AC DP scores ``c_new``
        # with a one-sided gamma split that sends every proposal to
        # the log floor (root-cause of the historical 0% accept rate
        # for this move; verified by ``runs/diag_retic_fix_proof.py``).
        sg = saved_gamma if saved_gamma is not None else 0.5
        gamma_a = max(0.0, min(1.0, 1.0 - sg))

        net.remove_edge(host)
        net.add_edges(Edge(a, c_new, length=host_len * split, gamma=gamma_a))
        net.add_edges(Edge(c_new, b, length=host_len * (1.0 - split)))
        net.add_edges(Edge(z, c_new, length=saved_length, gamma=sg))

        model.update_network()
        return model

    def undo(self, model: Model) -> None:
        if self.undo_info is not None:
            model.network = self.undo_info
        model.update_network()

    def same_move(self, model: Model) -> None:
        pass

    def hastings_ratio(self) -> float:
        return 1.0

