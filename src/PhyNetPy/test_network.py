import pytest
from typing import Iterable, Optional, Sequence, Tuple
from PhyNetPy.Network import Network, Node, Edge
from PhyNetPy.GraphUtils import *


#######################
#### TEST NETWORKS ####
#######################

def build_simple_tree_network() -> Network:
    """
    Construct a strictly binary tree with 10 leaves and no reticulation nodes.
    """
    leaves = _leaf_labels("simple", 10)
    (
        leaf1,
        leaf2,
        leaf3,
        leaf4,
        leaf5,
        leaf6,
        leaf7,
        leaf8,
        leaf9,
        leaf10,
    ) = leaves

    internal_nodes = [
        "simple_root",
        "simple_left",
        "simple_right",
        "simple_left_inner_1",
        "simple_left_inner_2",
        "simple_right_inner_1",
        "simple_right_inner_2",
        "simple_right_inner_3",
        "simple_right_inner_4",
    ]

    edges = [
        _edge("simple_root", "simple_left"),
        _edge("simple_root", "simple_right"),
        _edge("simple_left", "simple_left_inner_1"),
        _edge("simple_left", "simple_left_inner_2"),
        _edge("simple_left_inner_1", leaf1),
        _edge("simple_left_inner_1", leaf2),
        _edge("simple_left_inner_2", leaf3),
        _edge("simple_left_inner_2", leaf4),
        _edge("simple_right", "simple_right_inner_1"),
        _edge("simple_right", "simple_right_inner_2"),
        _edge("simple_right_inner_1", leaf5),
        _edge("simple_right_inner_1", "simple_right_inner_3"),
        _edge("simple_right_inner_3", leaf6),
        _edge("simple_right_inner_3", leaf7),
        _edge("simple_right_inner_2", "simple_right_inner_4"),
        _edge("simple_right_inner_2", leaf10),
        _edge("simple_right_inner_4", leaf8),
        _edge("simple_right_inner_4", leaf9),
    ]

    node_labels = internal_nodes + leaves
    net = _build_network(node_labels, edges)
    assert len(net.get_leaves()) == 10
    assert is_tree(net)
    return net

def build_single_reticulation_network() -> Network:
    """
    Construct a binary network with exactly one reticulation node and ~10 leaves.
    """
    leaves = _leaf_labels("retic", 10)
    (
        leaf1,
        leaf2,
        leaf3,
        leaf4,
        leaf5,
        leaf6,
        leaf7,
        leaf8,
        leaf9,
        leaf10,
    ) = leaves

    internal_nodes = [
        "retic_root",
        "retic_left",
        "retic_right",
        "retic_left_split",
        "retic_left_branch",
        "retic_p1",
        "retic_right_split",
        "retic_p2",
        "retic_right_branch",
        "retic_right_inner",
    ]

    # retic_core serves as a shared naming anchor for parent tracking clarity
    retic_nodes = {"retic_h1"}

    edges = [
        _edge("retic_root", "retic_left"),
        _edge("retic_root", "retic_right"),
        _edge("retic_left", "retic_left_split"),
        _edge("retic_left", "retic_left_branch"),
        _edge("retic_left_split", leaf1),
        _edge("retic_left_split", leaf2),
        _edge("retic_left_branch", leaf3),
        _edge("retic_left_branch", "retic_p1"),
        _edge("retic_p1", leaf4),
        _edge("retic_p1", "retic_h1", gamma=0.5),
        _edge("retic_right", "retic_right_split"),
        _edge("retic_right", "retic_right_branch"),
        _edge("retic_right_split", leaf5),
        _edge("retic_right_split", "retic_p2"),
        _edge("retic_p2", leaf6),
        _edge("retic_p2", "retic_h1", gamma=0.5),
        _edge("retic_right_branch", "retic_right_inner"),
        _edge("retic_right_branch", leaf9),
        _edge("retic_right_inner", leaf7),
        _edge("retic_right_inner", leaf8),
        _edge("retic_h1", leaf10),
    ]

    node_labels = internal_nodes + ["retic_h1"] + leaves
    net = _build_network(node_labels, edges, retic_labels=retic_nodes)
    assert len([n for n in net.V() if n.is_reticulation()]) == 1
    assert len(net.get_leaves()) == 10
    return net

def build_level_two_network() -> Network:
    """
    Construct a binary level-2 network containing two reticulation nodes
    within the same biconnected component.
    """
    leaves = _leaf_labels("level2", 10)
    (
        leaf1,
        leaf2,
        leaf3,
        leaf4,
        leaf5,
        leaf6,
        leaf7,
        leaf8,
        leaf9,
        leaf10,
    ) = leaves

    internal_nodes = [
        "level2_root",
        "level2_left",
        "level2_right",
        "level2_left_a",
        "level2_left_b",
        "level2_right_a",
        "level2_right_b",
        "level2_e1",
        "level2_f1",
        "level2_g1",
        "level2_m1",
    ]
    retic_nodes = {"level2_h1", "level2_h2"}

    edges = [
        _edge("level2_root", "level2_left"),
        _edge("level2_root", "level2_right"),
        _edge("level2_left", "level2_left_a"),
        _edge("level2_left", "level2_left_b"),
        _edge("level2_left_a", leaf1),
        _edge("level2_left_a", leaf2),
        _edge("level2_left_b", leaf3),
        _edge("level2_left_b", leaf4),
        _edge("level2_right", "level2_right_a"),
        _edge("level2_right", "level2_right_b"),
        _edge("level2_right_a", leaf5),
        _edge("level2_right_a", "level2_e1"),
        _edge("level2_e1", leaf6),
        _edge("level2_e1", "level2_h1", gamma=0.4),
        _edge("level2_right_b", "level2_f1"),
        _edge("level2_right_b", "level2_g1"),
        _edge("level2_f1", leaf7),
        _edge("level2_f1", "level2_h2", gamma=0.6),
        _edge("level2_g1", leaf8),
        _edge("level2_g1", "level2_h1", gamma=0.6),
        _edge("level2_h1", "level2_m1"),
        _edge("level2_m1", leaf9),
        _edge("level2_m1", "level2_h2"),
        _edge("level2_h2", leaf10),
    ]

    node_labels = internal_nodes + list(retic_nodes) + leaves
    net = _build_network(node_labels, edges, retic_labels=retic_nodes)
    assert len([n for n in net.V() if n.is_reticulation()]) == 2
    assert len(net.get_leaves()) == 10
    assert level(net) == 2
    return net

def build_multi_root_network() -> Network:
    """
    Construct a network containing two disconnected rooted components.
    """
    leaves = _leaf_labels("multi", 10)
    leaf1, leaf2, leaf3, leaf4, leaf5, leaf6, leaf7, leaf8, leaf9, leaf10 = leaves

    node_labels = [
        "multi_root_a",
        "multi_a_left",
        "multi_a_right",
        "multi_a_bridge",
        "multi_a_mid",
        "multi_h1",
        "multi_root_b",
        "multi_b_left",
        "multi_b_right",
        "multi_b_mid",
    ] + leaves

    retic_nodes = {"multi_h1"}

    edges = [
        _edge("multi_root_a", "multi_a_left"),
        _edge("multi_root_a", "multi_a_right"),
        _edge("multi_a_left", leaf1),
        _edge("multi_a_left", "multi_a_bridge"),
        _edge("multi_a_bridge", leaf2),
        _edge("multi_a_bridge", "multi_h1", gamma=0.5),
        _edge("multi_a_right", leaf3),
        _edge("multi_a_right", "multi_a_mid"),
        _edge("multi_a_mid", leaf4),
        _edge("multi_a_mid", "multi_h1", gamma=0.5),
        _edge("multi_h1", leaf5),
        _edge("multi_root_b", "multi_b_left"),
        _edge("multi_root_b", "multi_b_right"),
        _edge("multi_b_left", leaf6),
        _edge("multi_b_left", leaf7),
        _edge("multi_b_right", "multi_b_mid"),
        _edge("multi_b_right", leaf8),
        _edge("multi_b_mid", leaf9),
        _edge("multi_b_mid", leaf10),
    ]

    net = _build_network(node_labels, edges, retic_labels=retic_nodes)
    roots = [node for node in net.V() if net.in_degree(node) == 0]
    assert len(roots) == 2
    assert len(net.get_leaves()) == 10
    return net

def build_network_with_floater() -> Network:
    """
    Construct a binary network that also contains a floating (isolated) node.
    """
    leaves = _leaf_labels("floater", 10)
    (
        leaf1,
        leaf2,
        leaf3,
        leaf4,
        leaf5,
        leaf6,
        leaf7,
        leaf8,
        leaf9,
        leaf10,
    ) = leaves

    internal_nodes = [
        "floater_root",
        "floater_left",
        "floater_right",
        "floater_left_split",
        "floater_left_branch",
        "floater_p1",
        "floater_right_split",
        "floater_p2",
        "floater_right_branch",
        "floater_right_inner",
        "floater_island",
    ]
    retic_nodes = {"floater_h1"}

    edges = [
        _edge("floater_root", "floater_left"),
        _edge("floater_root", "floater_right"),
        _edge("floater_left", "floater_left_split"),
        _edge("floater_left", "floater_left_branch"),
        _edge("floater_left_split", leaf1),
        _edge("floater_left_split", leaf2),
        _edge("floater_left_branch", leaf3),
        _edge("floater_left_branch", "floater_p1"),
        _edge("floater_p1", leaf4),
        _edge("floater_p1", "floater_h1", gamma=0.5),
        _edge("floater_right", "floater_right_split"),
        _edge("floater_right", "floater_right_branch"),
        _edge("floater_right_split", leaf5),
        _edge("floater_right_split", "floater_p2"),
        _edge("floater_p2", leaf6),
        _edge("floater_p2", "floater_h1", gamma=0.5),
        _edge("floater_right_branch", "floater_right_inner"),
        _edge("floater_right_branch", leaf9),
        _edge("floater_right_inner", leaf7),
        _edge("floater_right_inner", leaf8),
        _edge("floater_h1", leaf10),
    ]

    net = _build_network(
        internal_nodes + ["floater_h1"] + leaves,
        edges,
        retic_labels=retic_nodes,
    )
    # Add floating node explicitly (already in node_labels to preserve naming consistency)
    floater = next(node for node in net.V() if node.label == "floater_island")
    assert net.in_degree(floater) == 0 and net.out_degree(floater) == 0
    floater_leaves = [
        node for node in net.get_leaves() if node.label.startswith("floater_leaf_")
    ]
    assert len(floater_leaves) == 10
    return net

def build_network_with_cycle() -> Network:
    """
    Construct a binary network that contains a directed cycle.
    """
    leaves = _leaf_labels("cycle", 10)
    (
        leaf1,
        leaf2,
        leaf3,
        leaf4,
        leaf5,
        leaf6,
        leaf7,
        leaf8,
        leaf9,
        leaf10,
    ) = leaves

    internal_nodes = [
        "cycle_root",
        "cycle_left",
        "cycle_left_a",
        "cycle_left_b",
        "cycle_left_c",
        "cycle_entry",
        "cycle_mid",
        "cycle_mid_branch",
        "cycle_end",
        "cycle_end_branch",
        "cycle_tail",
    ]
    retic_nodes = {"cycle_entry"}

    edges = [
        _edge("cycle_root", "cycle_left"),
        _edge("cycle_root", "cycle_entry", gamma=0.5),
        _edge("cycle_left", "cycle_left_a"),
        _edge("cycle_left", "cycle_left_b"),
        _edge("cycle_left_a", leaf1),
        _edge("cycle_left_a", leaf2),
        _edge("cycle_left_b", "cycle_left_c"),
        _edge("cycle_left_b", leaf3),
        _edge("cycle_left_c", leaf4),
        _edge("cycle_left_c", leaf5),
        _edge("cycle_entry", "cycle_mid"),
        _edge("cycle_mid", "cycle_mid_branch"),
        _edge("cycle_mid", "cycle_end"),
        _edge("cycle_mid_branch", leaf6),
        _edge("cycle_mid_branch", leaf7),
        _edge("cycle_end", "cycle_end_branch"),
        _edge("cycle_end", "cycle_entry"),
        _edge("cycle_end_branch", leaf8),
        _edge("cycle_end_branch", "cycle_tail"),
        _edge("cycle_tail", leaf9),
        _edge("cycle_tail", leaf10),
    ]

    node_labels = internal_nodes + list(retic_nodes) + leaves
    net = _build_network(node_labels, edges, retic_labels=retic_nodes)

    cycle_entry = next(node for node in net.V() if node.label == "cycle_entry")
    assert cycle_entry.is_reticulation()
    assert len(net.get_leaves()) == 10
    return net

def build_network_with_degree_one_nodes() -> Network:
    """
    Construct a network containing multiple intermediate nodes with in-degree 1 and out-degree 1.
    """
    leaves = _leaf_labels("deg", 10)
    (
        leaf1,
        leaf2,
        leaf3,
        leaf4,
        leaf5,
        leaf6,
        leaf7,
        leaf8,
        leaf9,
        leaf10,
    ) = leaves

    internal_nodes = [
        "deg_root",
        "deg_left",
        "deg_left_a",
        "deg_left_b",
        "deg_left_c",
        "deg_left_d",
        "deg_left_e",
        "deg_left_f",
        "deg_chain_start",
        "deg_chain_mid1",
        "deg_chain_mid2",
        "deg_chain_mid3",
        "deg_chain_leaf_a",
    ]

    edges = [
        _edge("deg_root", "deg_left"),
        _edge("deg_root", "deg_chain_start"),
        _edge("deg_left", "deg_left_a"),
        _edge("deg_left", "deg_left_b"),
        _edge("deg_left_a", "deg_left_c"),
        _edge("deg_left_a", "deg_left_d"),
        _edge("deg_left_c", leaf1),
        _edge("deg_left_c", leaf2),
        _edge("deg_left_d", leaf3),
        _edge("deg_left_d", leaf4),
        _edge("deg_left_b", "deg_left_e"),
        _edge("deg_left_b", "deg_left_f"),
        _edge("deg_left_e", leaf5),
        _edge("deg_left_e", leaf6),
        _edge("deg_left_f", leaf7),
        _edge("deg_left_f", leaf8),
        _edge("deg_chain_start", "deg_chain_mid1"),
        _edge("deg_chain_mid1", "deg_chain_mid2"),
        _edge("deg_chain_mid2", "deg_chain_mid3"),
        _edge("deg_chain_mid3", "deg_chain_leaf_a"),
        _edge("deg_chain_leaf_a", leaf9),
        _edge("deg_chain_leaf_a", leaf10),
    ]

    net = _build_network(internal_nodes + leaves, edges)
    # Confirm the presence of multiple degree-(1,1) nodes
    degree_one_nodes = [
        node
        for node in net.V()
        if net.in_degree(node) == 1 and net.out_degree(node) == 1
    ]
    assert len(degree_one_nodes) >= 3
    return net

def build_network_with_high_degree_parent() -> Network:
    """
    Construct a network where one parent node has more than two children.
    """
    leaves = _leaf_labels("high", 10)
    leaf1, leaf2, leaf3, leaf4, leaf5, leaf6, leaf7, leaf8, leaf9, leaf10 = leaves

    internal_nodes = [
        "high_root",
        "high_parent",
        "high_child_a",
        "high_child_b",
        "high_child_c",
        "high_right",
        "high_right_a",
        "high_right_b",
    ]

    edges = [
        _edge("high_root", "high_parent"),
        _edge("high_root", "high_right"),
        _edge("high_parent", "high_child_a"),
        _edge("high_parent", "high_child_b"),
        _edge("high_parent", "high_child_c"),
        _edge("high_child_a", leaf1),
        _edge("high_child_a", leaf2),
        _edge("high_child_b", leaf3),
        _edge("high_child_b", leaf4),
        _edge("high_child_c", leaf5),
        _edge("high_child_c", leaf6),
        _edge("high_right", "high_right_a"),
        _edge("high_right", "high_right_b"),
        _edge("high_right_a", leaf7),
        _edge("high_right_a", leaf8),
        _edge("high_right_b", leaf9),
        _edge("high_right_b", leaf10),
    ]

    net = _build_network(internal_nodes + leaves, edges)
    high_parent = next(node for node in net.V() if node.label == "high_parent")
    assert net.out_degree(high_parent) == 3
    return net









#################
#### HELPERS ####
#################


EdgeSpec = Tuple[str, str, Optional[float]]


def _leaf_labels(prefix: str, count: int) -> list[str]:
    """
    Generate deterministic leaf labels with the provided prefix.
    """
    if count <= 0:
        raise ValueError("Leaf count must be positive.")
    return [f"{prefix}_leaf_{i}" for i in range(1, count + 1)]


def _edge(src: str, dest: str, gamma: Optional[float] = None) -> EdgeSpec:
    """
    Small helper for describing an edge specification.
    """
    return (src, dest, gamma)


def _build_network(node_labels: Sequence[str],
                   edge_specs: Sequence[EdgeSpec],
                   retic_labels: Optional[Iterable[str]] = None,
                  ) -> Network:
    """
    Materialize a Network from node labels and directed edge specifications.
    """
    net = Network()
    retic_set = set(retic_labels or [])

    ordered_labels: list[str] = []
    seen: set[str] = set()

    for label in list(node_labels) + list(retic_set):
        if label not in seen:
            ordered_labels.append(label)
            seen.add(label)

    nodes: dict[str, Node] = {}
    for label in ordered_labels:
        nodes[label] = Node(label, is_reticulation=label in retic_set)

    net.add_nodes(*nodes.values())

    for spec in edge_specs:
        src, dest, gamma = spec
        if src not in nodes or dest not in nodes:
            raise KeyError(f"Edge references undefined nodes: {spec}")
        net.add_edges(Edge(nodes[src], nodes[dest], gamma=gamma))

    return net


def test_to_newick():
    """
    This test is designed to test the conversion from a Network object
    to a newick string. 
    
    As part of this test, the following components are also tested:
    - add edges
    - add nodes
    - processing reticulation edges
    """
    sn : Network = build_simple_tree_network()
    lvl2 : Network = build_level_two_network()
    assert(Network.from_newick(sn.newick()).is_isomorphic(sn))
    assert(Network.from_newick(lvl2.newick()).is_isomorphic(lvl2))
    
    

test_to_newick()