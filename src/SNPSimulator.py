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

from dataclasses import dataclass
import operator
import warnings

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
        """
        List the taxa present in the simulated data, in sorted order.

        Args:
            N/A
        Returns:
            list[str]: Taxon names, sorted lexicographically.
        """
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
        Raises:
            ValueError: If any taxon's count vector is not exactly n_sites
                        long, or if any red count falls outside 0..15.
        """
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

        taxa = self.taxa_names()

        # annotate=True keeps branch lengths and gammas in the TREES block,
        # which SNP_LIKELIHOOD needs to rebuild the generating network.
        nwk = self.network.newick(annotate = True)

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
        n (int): Number of leaf taxa. A tree on n leaves has only n - 2 edges
                 joining two internal nodes, and a reticulation consumes two
                 non-nested ones, so n >= 5 in practice for level >= 1.
        level (int): Number of reticulation nodes to add. Defaults to 1.
        birth_rate (float): Birth rate for the Yule process. Larger values 
                            produce shorter branch lengths. Defaults to 1.0.
        gamma_range (tuple[float, float]): Range from which inheritance 
                                            probabilities (gamma) are 
                                            uniformly drawn. Defaults to 
                                            (0.2, 0.8).
        seed (int | None): Random seed for reproducibility. Defaults to None.

    Raises:
        ValueError: If n < 2 or level < 0.

    Warns:
        UserWarning: If the topology runs out of legal insertion sites before
                     `level` reticulations are placed. The network is returned
                     with however many were added.

    Returns:
        Network: A phylogenetic network with n leaves and at most `level` 
                 reticulation nodes.
    """
    if n < 2:
        raise ValueError("Need at least 2 taxa.")
    if level < 0:
        raise ValueError("Network level must be non-negative.")

    rng = np.random.default_rng(seed)

    # Step 1: Generate a base tree via Yule process
    yule = Yule(gamma=birth_rate, n=n, rng=rng)
    net = yule.generate_network()

    # Yule emits "spec*"/"internal*"/"root" labels, and reports branch lengths
    # in expected substitutions per site -- a unit Network cannot infer itself.
    _clean_node_names(net)
    net.set_branch_length_unit(BranchLengthUnit.SUBSTITUTIONS_PER_SITE)

    # Step 2: Add reticulations
    for retic_id in range(level):
        gamma = rng.uniform(gamma_range[0], gamma_range[1])
        if not _add_reticulation(net, rng, gamma, retic_id):
            warnings.warn(
                f"Could not place reticulation #{retic_id}. "
                f"Network has {retic_id} reticulations instead of {level}."
            )
            break

    return net


def _clean_node_names(net: Network) -> None:
    """
    Relabel a simulator-generated tree in place to PhyNetPy naming conventions:
    leaves become T1..Tn (ordered by their original label), the root becomes
    Root, and the remaining internal nodes become I1..Ik ordered by depth.

    Renaming goes through :meth:`Network.update_node_name`, which rehashes the
    node in every name-keyed collection (``Node.__hash__`` is name-based, so a
    bare ``set_name`` would strand the node in the node/edge sets).

    Args:
        net (Network): A tree-like network from a Yule/CBDP simulator. Modified
                       in place.
    Returns:
        None
    """
    root = net.root()
    ordered_leaves = sorted(net.get_leaves(), key = lambda nd : nd.label)
    leaf_rank = {leaf : i for i, leaf in enumerate(ordered_leaves)}

    renames = [(leaf, f"T{i + 1}") for leaf, i in leaf_rank.items()]
    renames.append((root, "Root"))

    # Internal numbering keys off depth, then the lowest-ranked leaf beneath the
    # node. A plain BFS discovery order would be simpler but not reproducible:
    # V(), get_children(), and hence traversal order are all backed by sets, and
    # Node.__hash__ is hash(name), which Python randomizes per process.
    depth = net.bfs_dfs()[0]
    below = net.leaf_descendants_all()
    named = set(leaf_rank) | {root}
    internal = sorted(
        (node for node in depth if node not in named),
        key = lambda nd : (depth[nd], min(leaf_rank[l] for l in below[nd]))
    )
    renames.extend((nd, f"I{i}") for i, nd in enumerate(internal, start = 1))

    # Names are resolved up front: renaming mutates each node's hash, so the
    # pairs cannot be looked up from a Node-keyed container mid-loop.
    for node, name in renames:
        net.update_node_name(node, name)


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

    This deliberately does not delegate to ``GraphUtils.add_hybrid``, which
    performs the same bisection but leaves the four new edges without lengths
    and takes no gamma. Both are required here: the simulator evolves a
    coalescent down these branches, and ``write_nexus`` has to serialize them.

    Args:
        net (Network): The network to modify (in-place).
        rng (np.random.Generator): Random number generator.
        gamma (float): Inheritance probability for the reticulation.
        retic_id (int): Integer identifier for naming the hybrid node.

    Returns:
        bool: True if a reticulation was successfully added, False if 
              not enough candidate edges were found.
    """
    # Candidates join two non-leaf, non-reticulation nodes: bisecting a pendant
    # edge would hang a leaf below the hybrid, and reusing a reticulation
    # endpoint would nest hybridizations. They are sorted because Network.E()
    # is backed by a set whose order varies with PYTHONHASHSEED, which would
    # otherwise leave the draw below irreproducible across processes.
    leaves = set(net.get_leaves())
    candidates = sorted(
        (
            edge for edge in net.E()
            if {edge.src, edge.dest}.isdisjoint(leaves)
            and not edge.src.is_reticulation()
            and not edge.dest.is_reticulation()
        ),
        key = lambda edge : (edge.src.label, edge.dest.label)
    )

    if len(candidates) < 2:
        return False

    # Resample until the two edges are node-disjoint and neither lies in the
    # other's subtree; the hybrid edge would otherwise close a directed cycle.
    for _ in range(50):
        idxs = rng.choice(len(candidates), size=2, replace=False)
        edge1, edge2 = candidates[idxs[0]], candidates[idxs[1]]

        ends_1 = {edge1.src, edge1.dest}
        ends_2 = {edge2.src, edge2.dest}
        if ends_1 & ends_2:
            continue
        if ends_2 & net.get_subtree_at(edge1.dest):
            continue
        if ends_1 & net.get_subtree_at(edge2.dest):
            continue

        break
    else:
        return False

    e1_src, e1_dest = edge1.src, edge1.dest
    e2_src, e2_dest = edge2.src, edge2.dest

    # Bisect edge 1: e1_src -> new_tree_node -> e1_dest
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
