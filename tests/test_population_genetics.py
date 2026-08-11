"""Tests for population allele counts and admixture-graph semantics."""

from __future__ import annotations

import numpy as np
import pytest

from phynetpy.MSA import DataSequence, MSA
from phynetpy.Network import Edge, Network, Node
from phynetpy.PopulationGenetics import (
    AdmixtureGraph,
    PopulationData,
    PopulationGeneticsError,
)
from phynetpy.data import BiallelicMarkers


def test_population_data_preserves_missing_denominators() -> None:
    data = PopulationData(
        [[1, 0, 3], [0, 2, 0]],
        [[2, 0, 4], [2, 2, 0]],
        ["North", "South"],
        sites=["s1", "s2", "s3"],
    )

    assert data.populations == ("North", "South")
    assert data.sites == ("s1", "s2", "s3")
    assert data.n_populations == 2
    assert data.n_sites == 3
    np.testing.assert_allclose(
        data.allele_frequencies(),
        [[0.5, np.nan, 0.75], [0.0, 1.0, np.nan]],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        data.allele_frequencies("North"),
        [0.5, np.nan, 0.75],
        equal_nan=True,
    )


def test_from_genotypes_aggregates_mixed_ploidy_and_missing_calls() -> None:
    data = PopulationData.from_genotypes(
        [
            [0, 1, np.nan],
            [2, 1, 0],
            [1, "?", 1],
        ],
        ["a1", "a2", "b1"],
        {"A": ["a1", "a2"], "B": ["b1"]},
        ploidy=[2, 2, 1],
    )

    np.testing.assert_array_equal(
        data.alternate_allele_counts,
        [[2, 2, 0], [1, 0, 1]],
    )
    np.testing.assert_array_equal(
        data.called_allele_counts,
        [[4, 4, 2], [1, 0, 1]],
    )
    np.testing.assert_allclose(
        data.allele_frequencies(),
        [[0.5, 0.5, 0.0], [1.0, np.nan, 1.0]],
        equal_nan=True,
    )


def test_from_biallelic_markers_uses_mapping_and_sample_counts() -> None:
    alignment = MSA(data=[
        DataSequence([0, 1, 0], "a1"),
        DataSequence([2, 0, 1], "a2"),
        DataSequence([1, 1, 1], "b1"),
    ])
    alignment.get_records()[0].seq[2] = "?"
    markers = BiallelicMarkers(
        alignment,
        mapping={"A": ["a1", "a2"], "B": ["b1"]},
        samples={"a1": 2, "a2": 2, "b1": 2},
    )

    data = PopulationData.from_biallelic_markers(
        markers, called_alleles=markers.samples
    )

    np.testing.assert_array_equal(
        data.alternate_allele_counts,
        [[2, 1, 1], [1, 1, 1]],
    )
    np.testing.assert_array_equal(
        data.called_allele_counts,
        [[4, 4, 2], [2, 2, 2]],
    )


def test_marker_conversion_requires_an_explicit_denominator() -> None:
    markers = BiallelicMarkers(
        MSA(data=[DataSequence([1], "diploid_sample")])
    )

    with pytest.raises(PopulationGeneticsError, match="called_alleles is required"):
        PopulationData.from_biallelic_markers(markers)


def test_from_genotypes_preserves_numpy_masks() -> None:
    genotypes = np.ma.array([[1, 2]], mask=[[True, False]])

    data = PopulationData.from_genotypes(
        genotypes,
        ["sample"],
        {"Population": ["sample"]},
        ploidy=2,
    )

    np.testing.assert_array_equal(data.alternate_allele_counts, [[0, 2]])
    np.testing.assert_array_equal(data.called_allele_counts, [[0, 2]])
    np.testing.assert_allclose(
        data.allele_frequencies(),
        [[np.nan, 1.0]],
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ("alternate", "called", "message"),
    [
        ([[2]], [[1]], "cannot exceed"),
        ([[0.5]], [[1]], "integer"),
        ([[0, 1]], [[1]], "same shape"),
    ],
)
def test_population_data_rejects_invalid_counts(
    alternate, called, message: str
) -> None:
    with pytest.raises(PopulationGeneticsError, match=message):
        PopulationData(alternate, called, ["A"])


def test_genotype_population_membership_must_be_a_partition() -> None:
    with pytest.raises(PopulationGeneticsError, match="more than one"):
        PopulationData.from_genotypes(
            [[0], [1]],
            ["a", "b"],
            {"P1": ["a", "b"], "P2": ["b"]},
        )

    with pytest.raises(PopulationGeneticsError, match="unassigned"):
        PopulationData.from_genotypes(
            [[0], [1]],
            ["a", "b"],
            {"P1": ["a"]},
        )


def _valid_nonbinary_admixture_network() -> Network:
    """Three-way admixture plus a unary population epoch."""
    nodes = {
        label: Node(label, is_reticulation=(label == "#H0"))
        for label in [
            "Root", "I1", "I2", "I3", "#H0", "Epoch", "A", "B", "C", "D"
        ]
    }
    network = Network(nodes=set(nodes.values()))
    edges = [
        Edge(nodes["Root"], nodes["I1"]),
        Edge(nodes["Root"], nodes["I2"]),
        Edge(nodes["Root"], nodes["I3"]),
        Edge(nodes["I1"], nodes["A"]),
        Edge(nodes["I2"], nodes["B"]),
        Edge(nodes["I3"], nodes["C"]),
        Edge(nodes["I1"], nodes["#H0"], gamma=0.2),
        Edge(nodes["I2"], nodes["#H0"], gamma=0.3),
        Edge(nodes["I3"], nodes["#H0"], gamma=0.5),
        Edge(nodes["#H0"], nodes["Epoch"]),
        Edge(nodes["Epoch"], nodes["D"]),
    ]
    network.add_edges(edges)
    return network


def test_admixture_graph_accepts_nonbinary_and_unary_events() -> None:
    graph = AdmixtureGraph(_valid_nonbinary_admixture_network())

    assert graph.root.label == "Root"
    assert [node.label for node in graph.admixture_nodes] == ["#H0"]
    assert [node.label for node in graph.terminal_populations] == [
        "A", "B", "C", "D"
    ]
    assert graph.admixture_proportions("#H0") == {
        "I1": 0.2,
        "I2": 0.3,
        "I3": 0.5,
    }


def test_admixture_graph_rejects_inconsistent_proportions() -> None:
    network = _valid_nonbinary_admixture_network()
    hybrid = network.has_node_named("#H0")
    assert hybrid is not None
    network.get_edge(network.has_node_named("I3"), hybrid).set_gamma(0.4)

    with pytest.raises(PopulationGeneticsError, match="sum to one"):
        AdmixtureGraph(network)


def test_admixture_graph_rejects_unmarked_reticulation() -> None:
    network = _valid_nonbinary_admixture_network()
    hybrid = network.has_node_named("#H0")
    assert hybrid is not None
    hybrid.set_is_reticulation(False)

    with pytest.raises(PopulationGeneticsError, match="must be marked"):
        AdmixtureGraph(network)


def test_admixture_graph_rejects_cycles_and_multiple_roots() -> None:
    root = Node("Root")
    a = Node("A", is_reticulation=True)
    b = Node("B")
    cyclic = Network(nodes={root, a, b})
    cyclic.add_edges([
        Edge(root, a, gamma=0.5),
        Edge(b, a, gamma=0.5),
        Edge(a, b),
    ])
    with pytest.raises(PopulationGeneticsError, match="acyclic"):
        AdmixtureGraph(cyclic)

    disconnected = Network(nodes={Node("A"), Node("B")})
    with pytest.raises(PopulationGeneticsError, match="exactly one root"):
        AdmixtureGraph(disconnected)
