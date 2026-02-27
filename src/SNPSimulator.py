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
Last Edit : 2/12/26
First Included in Version : 1.1.0

SNP Data Simulator for Phylogenetic Networks.

Simulates biallelic (SNP) data over a phylogenetic network using a 
forward-in-time 2-state continuous-time Markov chain (CTMC) along branches.
Useful for generating test datasets of arbitrary size for likelihood 
computation validation, stress testing, and GPU executor benchmarking.

Usage:
    from PhyNetPy.SNPSimulator import simulate, random_network

    # Generate a random level-2 network with 50 taxa
    net = random_network(n=50, level=2, seed=42)

    # Simulate 10000 SNP sites over it
    sim = simulate(n=50, s=10000, net=net, seed=42)

    # Write to nexus file for use with SNP_LIKELIHOOD
    sim.write_nexus("stress_test_50taxa.nex")

Docs   - [x]
Tests  - [ ]
Design - [x]
"""

from __future__ import annotations

import copy
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Union

from .Network import Network, Node, Edge
from .BirthDeath import Yule


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulatedSNPData:
    """
    Container for simulated SNP data and the generating network.
    
    Attributes:
        network (Network): The phylogenetic network used for simulation.
        data (dict[str, list[int]]): Mapping from taxon name to a list of 
                                      red allele counts per site.
        n_taxa (int): Number of leaf taxa.
        n_sites (int): Number of simulated SNP sites.
        samples (dict[str, int]): Number of sampled individuals per taxon.
        u (float): Mutation rate from red to green allele.
        v (float): Mutation rate from green to red allele.
        coal (float): Coalescent rate parameter (theta).
        seed (int): Random seed used for simulation.
    """
    network: Network
    data: dict[str, list[int]]
    n_taxa: int
    n_sites: int
    samples: dict[str, int]
    u: float
    v: float
    coal: float
    seed: int

    def taxa_names(self) -> list[str]:
        """Return sorted list of taxon names."""
        return sorted(self.data.keys())

    def write_nexus(self, filepath: str) -> None:
        """
        Write the simulated data and network to a NEXUS file compatible
        with SNP_LIKELIHOOD.

        The output file contains:
            - TAXA block with taxon labels
            - DATA block with SNP site patterns (0/1 per site for samples=1)
            - TREES block with the network in extended newick format 
              (including branch lengths and gamma annotations)

        Args:
            filepath (str): Path where the nexus file will be written.
        Returns:
            None
        """
        taxa = self.taxa_names()
        nwk = _network_to_rich_newick(self.network)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("#NEXUS\n\n")

            # TAXA block
            f.write("BEGIN TAXA;\n")
            f.write(f"DIMENSIONS NTAX={len(taxa)};\n")
            f.write(f"TAXLABELS {' '.join(taxa)};\n")
            f.write("END;\n\n")

            # DATA block
            f.write("BEGIN DATA;\n")
            f.write(f"  Dimensions nchar={self.n_sites};\n")
            f.write("  Format datatype=snp missing=? gap=- matchchar=.;\n")
            f.write("  Matrix\n")
            for taxon in taxa:
                seq = ''.join(str(x) for x in self.data[taxon])
                f.write(f"    {taxon} {seq}\n")
            f.write("  ;\nEND;\n\n")

            # TREES block
            f.write("BEGIN TREES;\n")
            f.write(f"Tree net = {nwk}\n")
            f.write("END;\n")


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def random_network(
    n: int,
    level: int = 1,
    birth_rate: float = 1.0,
    gamma_range: tuple[float, float] = (0.2, 0.8),
    seed: int | None = None
) -> Network:
    """
    Generate a random phylogenetic network with n taxa and a given number 
    of reticulations.

    First generates a random tree using a Yule (pure-birth) process, then 
    rebuilds the topology with clean node names. Finally, grafts reticulation 
    edges to reach the desired reticulation count.

    Note: The ``level`` parameter specifies the number of reticulation nodes
    to add, not the network level in the strict sense (max reticulations per
    biconnected component). Depending on placement, the resulting network's
    true level may be less than this value.

    Args:
        n (int): Number of leaf taxa (must be >= 3 for level >= 1).
        level (int): Number of reticulation nodes to add. Defaults to 1.
        birth_rate (float): Birth rate for the Yule process. Larger values 
                            produce shorter branch lengths. Defaults to 1.0.
        gamma_range (tuple[float, float]): Range from which inheritance 
                                            probabilities (gamma) are 
                                            uniformly drawn. Defaults to 
                                            (0.2, 0.8).
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Raises:
        ValueError: If n < 2 or level < 0, or not enough edges to place 
                    all reticulations.

    Returns:
        Network: A phylogenetic network with n leaves and `level` 
                 reticulation nodes.
    """
    if n < 2:
        raise ValueError("Need at least 2 taxa.")
    if level < 0:
        raise ValueError("Network level must be non-negative.")

    rng = np.random.default_rng(seed)

    # Step 1: Generate a base tree via Yule process
    yule = Yule(gamma=birth_rate, n=n, rng=rng)
    yule_net = yule.generate_network()

    # Step 2: Rebuild the network from scratch with clean names.
    # This avoids hash-breaking issues from renaming nodes in-place
    # (Node.__hash__ is name-based).
    net = _rebuild_with_names(yule_net)

    # Step 3: Add reticulations
    for retic_id in range(level):
        gamma = rng.uniform(gamma_range[0], gamma_range[1])
        success = _add_reticulation(net, rng, gamma, retic_id)
        if not success:
            import warnings
            warnings.warn(
                f"Could not place reticulation #{retic_id}. "
                f"Network has {retic_id} reticulations instead of {level}."
            )
            break

    return net


def _rebuild_with_names(yule_net: Network) -> Network:
    """
    Rebuild a Yule-generated tree as a fresh Network with clean node names.
    
    Leaves are named T1, T2, ... (sorted by original name).
    Internal nodes are named I1, I2, ...
    The root is named Root.
    
    This avoids the hash-invalidation issue from renaming nodes in-place
    (since Node.__hash__ and Node.__eq__ are name-based).

    Args:
        yule_net (Network): A tree-like network from a Yule/CBDP simulator.

    Returns:
        Network: A new Network with the same topology and branch lengths 
                 but with clean node names.
    """
    net = Network()
    old_root = yule_net.root()
    
    # Build a name mapping
    old_to_new_name: dict[str, str] = {}
    
    # Rename leaves
    leaves = sorted(yule_net.get_leaves(), key=lambda nd: nd.label)
    for i, leaf in enumerate(leaves):
        old_to_new_name[leaf.label] = f"T{i + 1}"
    
    # Root
    old_to_new_name[old_root.label] = "Root"
    
    # Internal nodes (BFS order for consistency)
    internal_count = 1
    queue = deque([old_root])
    visited = {old_root.label}
    while queue:
        node = queue.popleft()
        if node.label not in old_to_new_name:
            old_to_new_name[node.label] = f"I{internal_count}"
            internal_count += 1
        for child in yule_net.get_children(node):
            if child.label not in visited:
                visited.add(child.label)
                queue.append(child)
    
    # Create new nodes
    new_nodes: dict[str, Node] = {}
    for old_node in yule_net.V():
        new_name = old_to_new_name[old_node.label]
        new_node = Node(name=new_name, t=old_node.get_time())
        new_node.set_time(old_node.get_time())
        new_nodes[old_node.label] = new_node
    
    # Add all new nodes to the network
    for new_node in new_nodes.values():
        net.add_nodes(new_node)
    
    # Add edges
    for edge in yule_net.E():
        src_new = new_nodes[edge.src.label]
        dest_new = new_nodes[edge.dest.label]
        new_edge = Edge(src_new, dest_new, length=edge.get_length())
        net.add_edges(new_edge)
    
    return net


def _get_internal_edges(net: Network) -> list[Edge]:
    """
    Get all edges that connect two non-leaf nodes (internal edges).
    These are the candidate edges for reticulation insertion.

    Returns the actual Edge objects (not node pairs) to avoid edge lookup 
    issues after node renaming.

    Excludes edges to/from leaves, and edges involving reticulation nodes 
    (to avoid creating nested reticulations, which complicates the topology).

    Args:
        net (Network): A phylogenetic network.

    Returns:
        list[Edge]: List of internal Edge objects suitable for reticulation 
                    insertion.
    """
    leaves = set(net.get_leaves())
    candidates = []
    for edge in net.E():
        src, dest = edge.src, edge.dest
        if src not in leaves and dest not in leaves:
            if not src.is_reticulation() and not dest.is_reticulation():
                candidates.append(edge)
    return candidates


def _get_descendants(net: Network, node: Node) -> set[Node]:
    """
    Get all descendants of a node via BFS.

    Args:
        net (Network): The network.
        node (Node): Starting node.

    Returns:
        set[Node]: All nodes reachable from node going down the network.
    """
    desc = set()
    queue = deque([node])
    while queue:
        cur = queue.popleft()
        for child in net.get_children(cur):
            if child not in desc:
                desc.add(child)
                queue.append(child)
    return desc


def _add_reticulation(
    net: Network,
    rng: np.random.Generator,
    gamma: float,
    retic_id: int
) -> bool:
    """
    Add one reticulation edge to an existing network.

    Strategy:
        1. Find all internal edges (between non-leaf, non-retic nodes).
        2. Pick two edges that are NOT in an ancestor-descendant relationship
           (to avoid creating cycles).
        3. Bisect each edge to create two new internal nodes.
        4. The first new node becomes a tree node; the second becomes the 
           hybrid (#H) node with two parents.

    Args:
        net (Network): The network to modify (in-place).
        rng (np.random.Generator): Random number generator.
        gamma (float): Inheritance probability for the reticulation.
        retic_id (int): Integer identifier for naming the hybrid node.

    Returns:
        bool: True if a reticulation was successfully added, False if 
              not enough candidate edges were found.
    """
    candidates = _get_internal_edges(net)

    if len(candidates) < 2:
        return False

    # Try a few times to find a valid (non-ancestor-descendant) pair
    max_attempts = 50
    for _ in range(max_attempts):
        idxs = rng.choice(len(candidates), size=2, replace=False)
        edge1 = candidates[idxs[0]]
        edge2 = candidates[idxs[1]]

        e1_src, e1_dest = edge1.src, edge1.dest
        e2_src, e2_dest = edge2.src, edge2.dest

        # Check they don't share nodes
        nodes_1 = {e1_src, e1_dest}
        nodes_2 = {e2_src, e2_dest}
        if nodes_1 & nodes_2:
            continue

        # Check ancestor-descendant: e2_dest should not be an ancestor 
        # of e1_src (and vice versa)
        desc_e1 = _get_descendants(net, e1_dest)
        if e2_src in desc_e1 or e2_dest in desc_e1:
            continue
        desc_e2 = _get_descendants(net, e2_dest)
        if e1_src in desc_e2 or e1_dest in desc_e2:
            continue

        break
    else:
        return False

    # Bisect edge 1: e1_src -> new_tree_node -> e1_dest
    # Use the Edge object directly (no lookup needed)
    e1_len = edge1.get_length()
    half1 = e1_len / 2.0

    new_tree_node = Node(name=f"I_r{retic_id}")
    t_tree = e1_src.get_time() + half1
    new_tree_node.set_time(t_tree)

    net.add_nodes(new_tree_node)
    net.remove_edge(edge1)

    top1 = Edge(e1_src, new_tree_node, length=half1)
    bot1 = Edge(new_tree_node, e1_dest, length=half1)
    net.add_edges([top1, bot1])

    # Bisect edge 2: e2_src -> hybrid_node -> e2_dest
    e2_len = edge2.get_length()
    half2 = e2_len / 2.0

    hybrid = Node(name=f"#H{retic_id}", is_reticulation=True)
    t_hybrid = e2_src.get_time() + half2
    hybrid.set_time(t_hybrid)

    net.add_nodes(hybrid)
    net.remove_edge(edge2)

    top2 = Edge(e2_src, hybrid, length=half2, gamma=gamma)
    bot2 = Edge(hybrid, e2_dest, length=half2)
    net.add_edges([top2, bot2])

    # Add reticulation edge: new_tree_node -> hybrid
    retic_len = abs(t_hybrid - t_tree)
    if retic_len < 1e-10:
        retic_len = 0.01  # minimum branch length to avoid numerical issues

    retic_edge = Edge(new_tree_node, hybrid, length=retic_len, gamma=1 - gamma)
    net.add_edges(retic_edge)

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SNP SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(
    n: int,
    s: int,
    net: Network,
    samples: dict[str, int] | None = None,
    u: float = 1.0,
    v: float = 1.0,
    coal: float = 0.005,
    seed: int | None = None
) -> SimulatedSNPData:
    """
    Simulate SNP (biallelic marker) data over a phylogenetic network.

    Uses a forward-in-time simulation: at the root, an allele state (red or 
    green) is drawn from the stationary distribution. The state is then 
    propagated down the network along each branch using the 2-state CTMC 
    mutation model. At reticulation nodes, the parent lineage is chosen 
    probabilistically based on the inheritance probability (gamma).

    For samples=1 per taxon, this is an exact simulation under the biallelic 
    mutation model. For samples > 1, the simulation draws each sample 
    independently conditional on the leaf's evolved frequency — this is an 
    approximation (a full coalescent simulation within each population branch 
    would be needed for exact multi-sample simulation).

    Args:
        n (int): Expected number of taxa — used only for validation. 
                 The actual taxa come from the network's leaves.
        s (int): Number of SNP sites to simulate.
        net (Network): The phylogenetic network to simulate data on.
        samples (dict[str, int] | None): Number of sampled individuals per 
                                          taxon. Keys must match leaf names.
                                          Defaults to 1 per taxon.
        u (float): Mutation rate from red allele to green. Defaults to 1.0.
        v (float): Mutation rate from green allele to red. Defaults to 1.0.
        coal (float): Coalescent rate parameter (theta). Stored in output 
                      but not used in the forward simulation. Defaults to 0.005.
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Raises:
        ValueError: If the number of leaves in the network doesn't match n,
                    or if sample keys don't match leaf names.

    Returns:
        SimulatedSNPData: Container with simulated data, network, and metadata.
    """
    rng = np.random.default_rng(seed)

    leaves = net.get_leaves()
    leaf_names = sorted([leaf.label for leaf in leaves])

    if len(leaf_names) != n:
        raise ValueError(
            f"Network has {len(leaf_names)} leaves but n={n} was specified."
        )

    if samples is None:
        samples = {name: 1 for name in leaf_names}
    else:
        if set(samples.keys()) != set(leaf_names):
            raise ValueError(
                f"Sample keys {set(samples.keys())} don't match "
                f"leaf names {set(leaf_names)}."
            )

    # Build node lookup for fast access
    node_lookup: dict[str, Node] = {nd.label: nd for nd in net.V()}

    # Precompute edge info for BFS traversal
    # For each node, store (parent_node, branch_length, gamma_or_none)
    root = net.root()

    # Stationary distribution
    p_red = v / (u + v)

    # Simulate all sites in a vectorized manner
    data: dict[str, list[int]] = {name: [] for name in leaf_names}

    # Batch simulation for efficiency
    for _ in range(s):
        site_states = _simulate_one_site(net, root, p_red, u, v, rng)

        for name in leaf_names:
            base_state = site_states.get(name, 0)

            if samples[name] == 1:
                data[name].append(base_state)
            else:
                # For multi-sample: each sample independently gets the 
                # evolved state. The red count is the sum.
                red_count = sum(
                    1 for _ in range(samples[name])
                    if rng.random() < (base_state * 0.95 + (1 - base_state) * 0.05)
                )
                data[name].append(red_count)

    return SimulatedSNPData(
        network=net,
        data=data,
        n_taxa=n,
        n_sites=s,
        samples=samples,
        u=u,
        v=v,
        coal=coal,
        seed=seed if seed is not None else -1
    )


def _simulate_one_site(
    net: Network,
    root: Node,
    p_red: float,
    u: float,
    v: float,
    rng: np.random.Generator
) -> dict[str, int]:
    """
    Forward-simulate one biallelic SNP site on the network.

    Algorithm:
        1. Draw root state from stationary distribution (P(red) = v/(u+v)).
        2. BFS from the root, evolving the allele state along each branch 
           using the exact 2-state CTMC transition probabilities.
        3. At reticulation nodes (in-degree >= 2), only one parent lineage 
           contributes — chosen with probability equal to the inheritance 
           probability (gamma) on the corresponding edge.

    Args:
        net (Network): The phylogenetic network.
        root (Node): The root node.
        p_red (float): Stationary probability of the red allele.
        u (float): Mutation rate red → green.
        v (float): Mutation rate green → red.
        rng (np.random.Generator): Random number generator.

    Returns:
        dict[str, int]: Mapping from node name to allele state (0=green, 1=red).
    """
    states: dict[str, int] = {}

    # Draw root state
    states[root.label] = 1 if rng.random() < p_red else 0

    # BFS traversal from root
    queue = deque([root])
    visited = {root.label}

    while queue:
        node = queue.popleft()
        parent_state = states[node.label]

        for child in net.get_children(node):
            edge = net.get_edge(node, child)
            branch_len = edge.get_length()

            if child.is_reticulation():
                # Reticulation: decide if THIS parent contributes
                gamma_val = edge.get_gamma()
                if gamma_val is None:
                    gamma_val = 0.5

                if child.label in states:
                    # Already resolved by another parent — skip
                    continue

                # All parents must be visited before we can resolve
                parents = net.get_parents(child)
                all_parents_visited = all(p.label in visited for p in parents)

                if not all_parents_visited:
                    # Defer: this child will be handled when the other 
                    # parent visits it
                    continue

                # Now resolve: choose which parent lineage to follow
                # Collect all parent edges with their gammas
                parent_edges = []
                for p in parents:
                    pe = net.get_edge(p, child)
                    g = pe.get_gamma() if pe.get_gamma() is not None else 0.5
                    parent_edges.append((p, pe, g))

                # Normalize gammas (in case they don't sum to 1)
                total_g = sum(g for _, _, g in parent_edges)
                if total_g > 0:
                    probs = [g / total_g for _, _, g in parent_edges]
                else:
                    probs = [1.0 / len(parent_edges)] * len(parent_edges)

                # Choose parent
                chosen_idx = rng.choice(len(parent_edges), p=probs)
                chosen_parent, chosen_edge, _ = parent_edges[chosen_idx]
                chosen_len = chosen_edge.get_length()

                chosen_state = states[chosen_parent.label]
                child_state = _mutate_state(chosen_state, chosen_len, u, v, rng)
                states[child.label] = child_state

                if child.label not in visited:
                    visited.add(child.label)
                    queue.append(child)

            else:
                # Normal (tree) node: evolve state along branch
                child_state = _mutate_state(parent_state, branch_len, u, v, rng)
                states[child.label] = child_state

                if child.label not in visited:
                    visited.add(child.label)
                    queue.append(child)

    return states


def _mutate_state(
    state: int,
    t: float,
    u: float,
    v: float,
    rng: np.random.Generator
) -> int:
    """
    Evolve a single biallelic allele state along a branch of length t using 
    the exact transition probabilities of the 2-state CTMC.

    The rate matrix is:
        Q = [[-v,  v],
             [ u, -u]]

    where state 0 = green, state 1 = red.

    The transition probabilities are:
        P(stay at red | red, t)   = v/(u+v) + u/(u+v) * exp(-(u+v)*t)
        P(become red | green, t)  = v/(u+v) * (1 - exp(-(u+v)*t))

    Args:
        state (int): Current allele state (0=green, 1=red).
        t (float): Branch length (time in coalescent units).
        u (float): Mutation rate red → green.
        v (float): Mutation rate green → red.
        rng (np.random.Generator): Random number generator.

    Returns:
        int: The evolved allele state (0 or 1).
    """
    total = u + v
    exp_term = np.exp(-total * t)

    if state == 1:  # red
        # P(stay red) = v/(u+v) + u/(u+v) * exp(-(u+v)*t)
        p_stay = v / total + (u / total) * exp_term
        return 1 if rng.random() < p_stay else 0
    else:  # green
        # P(become red) = v/(u+v) * (1 - exp(-(u+v)*t))
        p_red = (v / total) * (1 - exp_term)
        return 1 if rng.random() < p_red else 0


# ═══════════════════════════════════════════════════════════════════════════════
# NEWICK SERIALIZATION (with branch lengths and gamma)
# ═══════════════════════════════════════════════════════════════════════════════

def _network_to_rich_newick(net: Network) -> str:
    """
    Convert a Network to an extended newick string with branch lengths 
    and inheritance probability (gamma) annotations.

    Format follows the PhyNetPy convention:
        - First occurrence of a reticulation: (subtree)#H0[&gamma=0.7]:0.05
        - Subsequent occurrences: #H0:0.05

    Args:
        net (Network): A phylogenetic network.

    Returns:
        str: Extended newick string ending with ';'.
    """
    processed_retics: set[str] = set()
    result = _newick_helper(net, net.root(), processed_retics) + ";"
    return result


def _newick_helper(
    net: Network,
    node: Node,
    processed_retics: set[str]
) -> str:
    """
    Recursive helper for newick serialization.

    Args:
        net (Network): The network.
        node (Node): Current node being serialized.
        processed_retics (set[str]): Set of reticulation node names that 
                                      have already been fully serialized.

    Returns:
        str: Newick substring for the subtree rooted at node.
    """
    leaves = set(net.get_leaves())

    if node in leaves:
        return node.label
    
    # If this is a reticulation node we've already processed, 
    # just emit the label (second occurrence)
    is_retic = node.is_reticulation()
    if is_retic and node.label in processed_retics:
        # Name should already start with '#'
        name = node.label if node.label.startswith('#') else '#' + node.label
        return name
    
    # Mark retic as processed (first occurrence)
    if is_retic:
        processed_retics.add(node.label)

    # Build children substring
    children = net.get_children(node)
    child_strs = []
    for child in children:
        child_str = _newick_helper(net, child, processed_retics)
        
        # Append branch length
        edge = net.get_edge(node, child)
        branch_len = edge.get_length()
        
        # Add gamma annotation ONLY on the first occurrence of a retic node
        # (the one that includes the full subtree with parentheses).
        # The second occurrence is just the bare "#H0" label — no gamma.
        if child.is_reticulation() and "(" in child_str:
            gamma_val = edge.get_gamma()
            if gamma_val is not None and gamma_val > 0:
                child_str += f"[&gamma={gamma_val}]"
        
        child_str += f":{branch_len}"
        child_strs.append(child_str)

    substr = "(" + ",".join(child_strs) + ")"

    # Add node name
    if is_retic:
        name = node.label if node.label.startswith('#') else '#' + node.label
    else:
        name = node.label

    substr += name

    return substr

