import pytest
from typing import Optional, Sequence, Tuple

from phynetpy.Network import Network, Node, Edge
from phynetpy.GraphUtils import (
    blobs,
    tree_of_blobs,
    subnet_given_leaves,
    induced_subnetwork_by_taxa,
    is_tree,
    level,
)


#############################
#### NETWORK BUILDERS    ####
#############################

EdgeSpec = Tuple[str, str, Optional[float]]


def _edge(src: str, dest: str, gamma: Optional[float] = None) -> EdgeSpec:
    return (src, dest, gamma)


def _build_network(
    node_labels: Sequence[str],
    edge_specs: Sequence[EdgeSpec],
    retic_labels: set[str] | None = None,
) -> Network:
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
        net.add_edges(Edge(nodes[src], nodes[dest], gamma=gamma))

    return net


def _node_by_label(net: Network, label: str) -> Node:
    node = net.has_node_named(label)
    assert node is not None, f"Node '{label}' not found"
    return node


def _leaf_labels(net: Network) -> set[str]:
    return {n.label for n in net.get_leaves()}


# ── Small tree: ((A,B),(C,D)); ──
def build_small_tree() -> Network:
    labels = ["Root", "I1", "I2", "A", "B", "C", "D"]
    edges = [
        _edge("Root", "I1"),
        _edge("Root", "I2"),
        _edge("I1", "A"),
        _edge("I1", "B"),
        _edge("I2", "C"),
        _edge("I2", "D"),
    ]
    net = _build_network(labels, edges)
    assert is_tree(net)
    return net


# ── Level-1 network with one reticulation ──
#        Root
#       /    \
#     L1      R1
#    / \     / \
#   A   P1  P2  B
#        \  /
#        #H0
#         |
#         C
def build_level1_network() -> Network:
    labels = ["Root", "L1", "R1", "P1", "P2", "#H0", "A", "B", "C"]
    edges = [
        _edge("Root", "L1"),
        _edge("Root", "R1"),
        _edge("L1", "A"),
        _edge("L1", "P1"),
        _edge("R1", "P2"),
        _edge("R1", "B"),
        _edge("P1", "#H0", gamma=0.4),
        _edge("P2", "#H0", gamma=0.6),
        _edge("#H0", "C"),
    ]
    net = _build_network(labels, edges, retic_labels={"#H0"})
    assert level(net) == 1
    return net


# ── Level-2 network: two reticulations in same blob ──
#          Root
#         /    \
#        I1     I2
#       / \    / \
#      A   P1 P2  B
#           \ /
#           #H0
#           / \
#          P3  D
#          |
#         #H1
#          |
#          C
def build_level2_network() -> Network:
    labels = ["Root", "I1", "I2", "P1", "P2", "P3", "#H0", "#H1", "A", "B", "C", "D"]
    edges = [
        _edge("Root", "I1"),
        _edge("Root", "I2"),
        _edge("I1", "A"),
        _edge("I1", "P1"),
        _edge("I2", "P2"),
        _edge("I2", "B"),
        _edge("P1", "#H0", gamma=0.5),
        _edge("P2", "#H0", gamma=0.5),
        _edge("#H0", "P3"),
        _edge("#H0", "D"),
        _edge("P3", "#H1", gamma=0.3),
        _edge("I1", "#H1", gamma=0.7),
        _edge("#H1", "C"),
    ]
    net = _build_network(labels, edges, retic_labels={"#H0", "#H1"})
    assert level(net) == 2
    return net


# ── Disjoint reticulations (two level-1 blobs) ──
#            Root
#           /    \
#         L       R
#        / \     / \
#       A  P1   P2  B
#           \ /
#           #H0
#            |
#            M
#           / \
#          P3  D
#          |
#         P4
#        / \
#       E  #H1
#           |
#           C
#  where #H1 also gets an edge from R
def build_two_blob_network() -> Network:
    """
    Two disjoint reticulation cycles so the network is level-1.

    Blob 1 cycle (undirected): Root-L-P1-#H0-P2-R-Root
    Blob 2 cycle (undirected): N-P4-#H1-P5-N
    """
    labels = [
        "Root", "L", "R", "P1", "P2", "#H0", "M", "N", "P4", "P5", "#H1",
        "A", "B", "C", "D", "E",
    ]
    edges = [
        _edge("Root", "L"),
        _edge("Root", "R"),
        _edge("L", "A"),
        _edge("L", "P1"),
        _edge("R", "P2"),
        _edge("R", "B"),
        _edge("P1", "#H0", gamma=0.5),
        _edge("P2", "#H0", gamma=0.5),
        _edge("#H0", "M"),
        _edge("M", "N"),
        _edge("M", "D"),
        _edge("N", "P4"),
        _edge("N", "P5"),
        _edge("P4", "C"),
        _edge("P4", "#H1", gamma=0.6),
        _edge("P5", "#H1", gamma=0.4),
        _edge("#H1", "E"),
    ]
    net = _build_network(labels, edges, retic_labels={"#H0", "#H1"})
    assert level(net) == 1
    return net


############################
#### BLOB TESTS         ####
############################

class TestBlobs:

    def test_tree_blobs_are_all_bridges(self):
        """Every edge in a tree is a bridge, so each blob should have exactly 2 nodes."""
        net = build_small_tree()
        comps = blobs(net)
        for comp in comps:
            assert len(comp) == 2, (
                f"Tree blob should be a single bridge edge (2 nodes), got {len(comp)}"
            )
        assert len(comps) == len(list(net.E()))

    def test_level1_has_cycle_blob(self):
        """A level-1 network should have at least one blob with >2 nodes (the cycle)."""
        net = build_level1_network()
        comps = blobs(net)
        big_blobs = [c for c in comps if len(c) > 2]
        assert len(big_blobs) >= 1, "Expected at least one non-trivial blob"
        cycle_blob = big_blobs[0]
        retic_in_blob = [n for n in cycle_blob if n.is_reticulation()]
        assert len(retic_in_blob) == 1

    def test_level2_blob_has_two_retics(self):
        """The level-2 network should have a blob containing both reticulations."""
        net = build_level2_network()
        comps = blobs(net)
        max_retics = max(
            sum(1 for n in comp if n.is_reticulation()) for comp in comps
        )
        assert max_retics == 2

    def test_all_nodes_covered(self):
        """The union of all blob node sets should cover every node in the network."""
        for builder in [build_small_tree, build_level1_network, build_level2_network]:
            net = builder()
            comps = blobs(net)
            covered = set()
            for comp in comps:
                covered.update(comp)
            assert covered == set(net.V()), (
                f"Blob union doesn't cover all nodes for {builder.__name__}"
            )


############################
#### TREE OF BLOBS TESTS ####
############################

class TestTreeOfBlobs:

    def test_tree_decomposes_into_bridge_subnetworks(self):
        """On a tree, tree_of_blobs should return one subnetwork per edge."""
        net = build_small_tree()
        components = tree_of_blobs(net)
        assert len(components) == len(list(net.E()))
        for sub in components:
            assert len(list(sub.V())) == 2
            assert len(list(sub.E())) == 1

    def test_level1_blob_subnetwork(self):
        """On a level-1 network, at least one blob subnetwork should contain the reticulation."""
        net = build_level1_network()
        components = tree_of_blobs(net)
        retic_blobs = [
            sub for sub in components
            if any(n.is_reticulation() for n in sub.V())
        ]
        assert len(retic_blobs) >= 1
        retic_sub = retic_blobs[0]
        retic_nodes = [n for n in retic_sub.V() if n.is_reticulation()]
        assert len(retic_nodes) == 1

    def test_level2_blob_subnetwork_has_two_retics(self):
        """On a level-2 network, the largest blob subnetwork should have 2 reticulations."""
        net = build_level2_network()
        components = tree_of_blobs(net)
        max_retics = max(
            sum(1 for n in sub.V() if n.is_reticulation())
            for sub in components
        )
        assert max_retics == 2

    def test_total_edges_match(self):
        """Total edges across all blob subnetworks should equal the network edge count."""
        for builder in [build_small_tree, build_level1_network, build_level2_network]:
            net = builder()
            components = tree_of_blobs(net)
            total_blob_edges = sum(len(list(sub.E())) for sub in components)
            assert total_blob_edges == len(list(net.E())), (
                f"Edge count mismatch for {builder.__name__}: "
                f"blob total={total_blob_edges}, network={len(list(net.E()))}"
            )

    def test_node_labels_preserved(self):
        """Node labels should be preserved in blob subnetworks."""
        net = build_level1_network()
        components = tree_of_blobs(net)
        all_labels = set()
        for sub in components:
            for n in sub.V():
                all_labels.add(n.label)
        original_labels = {n.label for n in net.V()}
        assert all_labels == original_labels

    def test_blob_subnetworks_are_independent(self):
        """Each blob subnetwork should be a distinct Network with distinct Node objects."""
        net = build_level1_network()
        components = tree_of_blobs(net)
        original_ids = {id(n) for n in net.V()}
        for sub in components:
            for n in sub.V():
                assert id(n) not in original_ids, (
                    "Blob subnetwork should contain copies, not original node objects"
                )

    def test_gamma_values_preserved(self):
        """Edge gamma values should carry over to blob subnetworks."""
        net = build_level1_network()
        components = tree_of_blobs(net)
        for sub in components:
            for e in sub.E():
                if e.dest.is_reticulation():
                    assert e.get_gamma() is not None and e.get_gamma() > 0, (
                        "Gamma value missing on reticulation edge in blob subnetwork"
                    )


####################################
#### SUBNET GIVEN LEAVES TESTS  ####
####################################

class TestSubnetGivenLeaves:

    def test_all_leaves_returns_full_network(self):
        """Requesting all leaves should reproduce the full topology."""
        net = build_small_tree()
        all_leaves = list(net.get_leaves())
        sub = subnet_given_leaves(net, all_leaves)
        assert _leaf_labels(sub) == _leaf_labels(net)

    def test_subset_leaves_tree(self):
        """On a tree, requesting a subset of leaves should prune irrelevant branches."""
        net = build_small_tree()
        target_nodes = [
            _node_by_label(net, "A"),
            _node_by_label(net, "B"),
        ]
        sub = subnet_given_leaves(net, target_nodes)
        sub_leaves = _leaf_labels(sub)
        assert "A" in sub_leaves
        assert "B" in sub_leaves
        assert "C" not in sub_leaves
        assert "D" not in sub_leaves

    def test_subnet_is_smaller(self):
        """Subnetwork should have fewer or equal nodes compared to original."""
        net = build_small_tree()
        target_nodes = [
            _node_by_label(net, "A"),
            _node_by_label(net, "B"),
        ]
        sub = subnet_given_leaves(net, target_nodes)
        assert len(list(sub.V())) <= len(list(net.V()))

    def test_single_leaf(self):
        """Requesting a single leaf should return a subnet containing that node."""
        net = build_small_tree()
        target = [_node_by_label(net, "A")]
        sub = subnet_given_leaves(net, target)
        sub_node_labels = {n.label for n in sub.V()}
        assert "A" in sub_node_labels
        assert len(list(sub.V())) == 1

    def test_subnet_with_reticulation_included(self):
        """When target leaves require the reticulation path, it should be included."""
        net = build_level1_network()
        target_nodes = [
            _node_by_label(net, "A"),
            _node_by_label(net, "C"),
        ]
        sub = subnet_given_leaves(net, target_nodes)
        sub_leaves = _leaf_labels(sub)
        assert "A" in sub_leaves
        assert "C" in sub_leaves

    def test_subnet_preserves_reticulation_when_needed(self):
        """The reticulation node should appear in the subnet when its descendant leaf is requested."""
        net = build_level1_network()
        target_nodes = [
            _node_by_label(net, "A"),
            _node_by_label(net, "B"),
            _node_by_label(net, "C"),
        ]
        sub = subnet_given_leaves(net, target_nodes)
        retic_labels = {n.label for n in sub.V() if n.is_reticulation()}
        assert "#H0" in retic_labels

    def test_subnet_excludes_reticulation_when_not_needed(self):
        """If target leaves don't require the reticulation path, it should be excluded."""
        net = build_level1_network()
        target_nodes = [
            _node_by_label(net, "A"),
            _node_by_label(net, "B"),
        ]
        sub = subnet_given_leaves(net, target_nodes)
        sub_leaves = _leaf_labels(sub)
        assert "C" not in sub_leaves

    def test_subnet_returns_new_objects(self):
        """Subnet should return copies, not references to original nodes."""
        net = build_small_tree()
        all_leaves = list(net.get_leaves())
        sub = subnet_given_leaves(net, all_leaves)
        original_ids = {id(n) for n in net.V()}
        for n in sub.V():
            assert id(n) not in original_ids

    def test_subnet_cross_reticulation_leaves(self):
        """Request leaves from both sides of a reticulation to verify correct topology."""
        net = build_level1_network()
        target_nodes = list(net.get_leaves())
        sub = subnet_given_leaves(net, target_nodes)
        assert _leaf_labels(sub) == _leaf_labels(net)

    def test_subnet_level2_partial(self):
        """On a level-2 network, request leaves that require traversing through the reticulation."""
        net = build_level2_network()
        target_nodes = [
            _node_by_label(net, "C"),
            _node_by_label(net, "D"),
        ]
        sub = subnet_given_leaves(net, target_nodes)
        sub_leaves = _leaf_labels(sub)
        assert "C" in sub_leaves
        assert "D" in sub_leaves
        assert "A" not in sub_leaves
        assert "B" not in sub_leaves


####################################
#### INDUCED SUBNETWORK TESTS   ####
####################################

class TestInducedSubnetworkByTaxa:

    def test_all_taxa(self):
        """Inducing on all taxa should preserve all leaf labels."""
        net = build_small_tree()
        taxa = [n.label for n in net.get_leaves()]
        sub = induced_subnetwork_by_taxa(net, taxa)
        assert _leaf_labels(sub) == _leaf_labels(net)

    def test_subset_taxa_tree(self):
        """Inducing on a subset should only retain those leaves."""
        net = build_small_tree()
        sub = induced_subnetwork_by_taxa(net, ["A", "B"])
        sub_leaves = _leaf_labels(sub)
        assert "A" in sub_leaves
        assert "B" in sub_leaves
        assert "C" not in sub_leaves
        assert "D" not in sub_leaves

    def test_invalid_taxon_raises(self):
        """Requesting a non-existent taxon should raise an error."""
        net = build_small_tree()
        with pytest.raises(Exception):
            induced_subnetwork_by_taxa(net, ["A", "NONEXISTENT"])

    def test_induced_on_reticulation_network(self):
        """Induce on leaves that require the reticulation path."""
        net = build_level1_network()
        sub = induced_subnetwork_by_taxa(net, ["A", "C"])
        sub_leaves = _leaf_labels(sub)
        assert "A" in sub_leaves
        assert "C" in sub_leaves


####################################
#### INTEGRATION TESTS          ####
####################################

class TestIntegration:

    def test_blobs_and_tree_of_blobs_agree_on_count(self):
        """blobs() and tree_of_blobs() should return the same number of components."""
        for builder in [build_small_tree, build_level1_network, build_level2_network]:
            net = builder()
            blob_count = len(blobs(net))
            tob_count = len(tree_of_blobs(net))
            assert blob_count == tob_count, (
                f"Component count mismatch for {builder.__name__}: "
                f"blobs={blob_count}, tree_of_blobs={tob_count}"
            )

    def test_subnet_then_blobs(self):
        """Taking a subnet then computing blobs should still work correctly."""
        net = build_level1_network()
        target_nodes = list(net.get_leaves())
        sub = subnet_given_leaves(net, target_nodes)
        comps = blobs(sub)
        assert len(comps) > 0

    def test_two_blob_network_decomposition(self):
        """A network with two disjoint reticulations should have two non-trivial blobs."""
        net = build_two_blob_network()
        comps = blobs(net)
        non_trivial = [c for c in comps if len(c) > 2]
        retic_blobs = [
            c for c in non_trivial
            if any(n.is_reticulation() for n in c)
        ]
        assert len(retic_blobs) == 2, (
            f"Expected 2 reticulation-containing blobs, got {len(retic_blobs)}"
        )

    def test_subnet_on_two_blob_network(self):
        """subnet_given_leaves on leaves from one side should not include unrelated leaves."""
        net = build_two_blob_network()
        target_nodes = [
            _node_by_label(net, "A"),
            _node_by_label(net, "B"),
        ]
        sub = subnet_given_leaves(net, target_nodes)
        sub_leaves = _leaf_labels(sub)
        assert "A" in sub_leaves
        assert "B" in sub_leaves
        assert "C" not in sub_leaves
        assert "D" not in sub_leaves
        assert "E" not in sub_leaves
