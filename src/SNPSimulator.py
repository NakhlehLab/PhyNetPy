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

Simulates biallelic (SNP) data by drawing one exact MSNC genealogy per
unlinked site and evolving a stationary 2-state continuous-time Markov chain
down it. This includes ILS, reticulation routing, and multiple samples per
species under the same model used by the Bryant marker likelihood.

Usage:
    from PhyNetPy.SNPSimulator import simulate, random_network

    # Generate a random level-2 network with 50 taxa
    net = random_network(n=50, level=2, seed=42)

    # Simulate 10000 SNP sites over it
    sim = simulate(n=50, s=10000, net=net, seed=42)

    # Write to nexus file for use with SNP_LIKELIHOOD
    sim.write_nexus("stress_test_50taxa.nex")

Docs   - [x]
Tests  - [x]
Design - [x]
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import operator

import numpy as np

from .Network import Network, Node, Edge
from .BirthDeath import Yule
from ._sim_markers import simulate_biallelic_markers
from ._units import BranchLengthUnit, BranchThetaKey


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
        samples (dict[str, int]): Number of sampled gene copies per taxon.
        u (float): Mutation rate from red to green allele.
        v (float): Mutation rate from green to red allele.
        theta (float): Population mutation rate ``4*N*mu``.
        seed (int): Random seed used for simulation.
    """
    network: Network
    data: dict[str, list[int]]
    n_taxa: int
    n_sites: int
    samples: dict[str, int]
    u: float
    v: float
    theta: float
    seed: int | None

    def taxa_names(self) -> list[str]:
        """Return sorted list of taxon names."""
        return sorted(self.data.keys())

    def write_nexus(self, filepath: str) -> None:
        """
        Write the simulated data and network to a NEXUS file compatible
        with SNP_LIKELIHOOD.

        The output file contains:
            - TAXA block with taxon labels
            - DATA block with one hexadecimal red-allele count per site
            - TREES block with the network in extended newick format 
              (including branch lengths and gamma annotations)

        Args:
            filepath (str): Path where the nexus file will be written.
        Returns:
            None
        """
        taxa = self.taxa_names()
        nwk = _network_to_rich_newick(self.network)
        if any(len(counts) != self.n_sites for counts in self.data.values()):
            raise ValueError("every taxon must have exactly n_sites counts.")
        if any(
            count < 0 or count > 15
            for counts in self.data.values()
            for count in counts
        ):
            raise ValueError(
                "NEXUS SNP rows encode counts as one hexadecimal character, "
                "so every red count must be in 0..15."
            )

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
                counts = self.data[taxon]
                seq = "".join(format(count, "X") for count in counts)
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
    net = Network(
        branch_length_unit=BranchLengthUnit.SUBSTITUTIONS_PER_SITE
    )
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
    Get all strict descendants of a node.

    Args:
        net (Network): The network.
        node (Node): Starting node.

    Returns:
        set[Node]: All nodes reachable from node going down the network,
                   excluding node itself.
    """
    return net.get_subtree_at(node) - {node}


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
    theta: float = 0.02,
    branch_thetas: dict[BranchThetaKey, float] | None = None,
    seed: int | None = None
) -> SimulatedSNPData:
    """
    Simulate SNP (biallelic marker) data over a phylogenetic network.

    Draws an independent MSNC genealogy per site, evolves the exact two-state
    mutation process down it, and aggregates sampled gene copies into red
    allele counts. This matches the model consumed by ``BiMarkers`` for one
    or multiple samples per taxon.

    Args:
        n (int): Expected number of taxa — used only for validation. 
                 The actual taxa come from the network's leaves.
        s (int): Number of SNP sites to simulate.
        net (Network): The phylogenetic network to simulate data on.
        samples (dict[str, int] | None): Number of sampled gene copies per
                                          taxon. Keys must match leaf names.
                                          Defaults to 1 per taxon.
        u (float): Mutation rate from red allele to green. Defaults to 1.0.
        v (float): Mutation rate from green allele to red. Defaults to 1.0.
        theta (float): Population mutation rate ``4*N*mu``.
        branch_thetas (dict | None): Fixed per-population theta overrides.
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Raises:
        ValueError: If the number of leaves in the network doesn't match n,
                    or if sample keys don't match leaf names.

    Returns:
        SimulatedSNPData: Container with simulated data, network, and metadata.
    """
    leaves = net.get_leaves()
    leaf_names = sorted([leaf.label for leaf in leaves])

    if len(leaf_names) != n:
        raise ValueError(
            f"Network has {len(leaf_names)} leaves but n={n} was specified."
        )

    if samples is None:
        samples = {name: 1 for name in leaf_names}
    else:
        if set(samples) != set(leaf_names):
            raise ValueError(
                f"Sample keys {set(samples)} don't match "
                f"leaf names {set(leaf_names)}."
            )
        try:
            samples = {
                name: operator.index(samples[name]) for name in leaf_names
            }
        except TypeError as exc:
            raise ValueError("sample counts must be integers.") from exc
        if any(count <= 0 for count in samples.values()):
            raise ValueError("sample counts must be positive.")

    mapping = {
        name: [f"{name}_{i}" for i in range(samples[name])]
        for name in leaf_names
    }
    data = simulate_biallelic_markers(
        net,
        s,
        mapping,
        theta=theta,
        u=u,
        v=v,
        branch_thetas=branch_thetas,
        rng=np.random.default_rng(seed),
    )

    return SimulatedSNPData(
        network=net,
        data=data,
        n_taxa=n,
        n_sites=s,
        samples=samples,
        u=u,
        v=v,
        theta=theta,
        seed=seed,
    )


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

