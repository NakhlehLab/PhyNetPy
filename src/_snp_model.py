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

Node graph backing the biallelic-marker (SNP) likelihood.

:func:`build_snp_model` turns a :class:`~.Network.Network` and its alignment
into a DAG of typed nodes that :mod:`~.BiMarkers` walks bottom-up to accumulate
partial likelihoods. The graph is rebuilt from scratch on every likelihood
evaluation, so it is independent of the :class:`~.ModelGraph.Model` that the
search loop mutates.

This representation exists only for the Bryant et al. biallelic algorithm,
whose per-node math differs between leaves, internal nodes, reticulations, and
the root. No other inference method in PhyNetPy uses it.
"""

from __future__ import annotations

from abc import ABC
from typing import Generator, Optional

from .Network import Network, Edge, Branch
from .MSA import MSA, DataSequence


###################
#### CONTAINER ####
###################

class SNPModel:
    """A built SNP node graph.

    Attributes:
        network: The network this graph was built from, kept so the exact
            blob-based network level can be recomputed.
        root: The node a post-order traversal starts from, normally the
            :class:`RootAggregatorNode` sitting above the network root.
        nodetypes: Built nodes bucketed by kind, used to count taxa and
            reticulations and to bind per-leaf sample counts.
    """

    def __init__(self) -> None:
        self.network: Optional[Network] = None
        self.root: Optional[ModelNode] = None
        self.nodetypes: dict[str, list[ModelNode]] = {
            "leaf": [], "internal": [], "reticulation": [], "root": [],
        }


###############
#### NODES ####
###############

class ModelNode(ABC):
    """
    Base class for a node in the SNP model graph.

    Only child links are tracked: the likelihood traversal moves strictly
    bottom-up, so parent pointers would never be read.
    """

    def __init__(self) -> None:
        self.children: list[ModelNode] = []

    def add_child(self, model_node: ModelNode) -> None:
        """
        Adds a successor to this node.

        Args:
            model_node (ModelNode): A ModelNode to add as a child.
        Returns:
            N/A
        """
        self.children.append(model_node)

    def get_model_children(self) -> list[ModelNode]:
        """
        Get the children nodes to this node.

        Args:
            N/A
        Returns:
            list[ModelNode]: The list of child nodes to this node.
        """
        return self.children

    def get_node_type(self) -> str:
        """
        Get the type of this node.
        """
        return self.node_type


class LeafNode(ModelNode):
    """
    A leaf node in the SNP model graph.
    """

    def __init__(self,
                 name: str,
                 branch_length: Branch,
                 data: list[DataSequence] = None,
                 samples: int = 1) -> None:
        """
        Initialize a LeafNode object.

        Args:
            name (str): The name of this leaf node.
            branch_length (Branch): The branch to this leaf's parent.
            data (list[DataSequence], optional): Sequences for this taxon.
            samples (int, optional): Number of sampled lineages.
        Returns:
            N/A
        """
        super().__init__()
        self.name: str = name
        self.node_type: str = "leaf"
        self.branch_info: Branch = branch_length
        self.data: list[DataSequence] = data
        self.samples: int = samples

    def get_name(self) -> str:
        """
        Returns the name of this leaf node.
        """
        return self.name

    def branch(self) -> Branch:
        """
        Returns the branch associated with this leaf node. A leaf has exactly
        one network parent and thus only one branch.

        Args:
            N/A
        Returns:
            Branch: A Branch object containing information about the branch.
        """
        return self.branch_info

    def set_data(self, data: list[DataSequence]) -> None:
        """
        Set the data for this leaf node.
        """
        self.data = data


class InternalNode(ModelNode):
    """
    An internal node in the SNP model graph.
    """

    def __init__(self, name: str, branch_length: Branch) -> None:
        """
        Initialize an InternalNode object.

        Args:
            name (str): The name of this internal node.
            branch_length (Branch): The branch to this node's parent.
        Returns:
            N/A
        """
        super().__init__()
        self.node_type = "internal"
        self.name: str = name
        self.branch_info: Branch = branch_length

    def branch(self) -> Branch:
        """
        Returns the branch associated with this internal node. An internal node
        has exactly one network parent and thus only one branch.

        Args:
            N/A
        Returns:
            Branch: A Branch object containing information about the branch.
        """
        return self.branch_info

    def get_name(self) -> str:
        """
        Returns the name of this internal node.
        """
        return self.name


class ReticulationNode(ModelNode):
    """
    A reticulation node in the SNP model graph.
    """

    def __init__(self, name: str, branch_1: Branch, branch_2: Branch) -> None:
        """
        Initialize a ReticulationNode object.

        Args:
            name (str): The name of this reticulation node.
            branch_1 (Branch): The branch to the first parent.
            branch_2 (Branch): The branch to the second parent.
        Returns:
            N/A
        """
        super().__init__()
        self.node_type = "reticulation"
        self.name: str = name
        self.branch_info: tuple[Branch, Branch] = (branch_1, branch_2)

    def get_name(self) -> str:
        """
        Returns the name of this reticulation node.
        """
        return self.name


class RootNode(ModelNode):
    """
    The network root in the SNP model graph.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize a RootNode object.

        Args:
            name (str): The name of this root node.
        Returns:
            N/A
        """
        super().__init__()
        self.node_type = "root"
        self.name: str = name

    def get_name(self) -> str:
        """
        Returns the name of this root node.
        """
        return self.name


class RootAggregatorNode(ModelNode):
    """
    Terminal node above the network root that holds the whole-network result.
    """

    def __init__(self) -> None:
        """
        Initialize a RootAggregatorNode object.

        Args:
            N/A
        Returns:
            N/A
        """
        super().__init__()
        self.name: str = "root_aggregator"
        self.node_type = "root_aggregator"

    def get_name(self) -> str:
        """
        Returns the name of this aggregator node.
        """
        return self.name


###################
#### TRAVERSAL ####
###################

def postorder(root: ModelNode) -> Generator[ModelNode, None, None]:
    """
    Yield every node reachable from ``root`` after all of its children.

    Children must be finished before their parent for a likelihood-style
    bottom-up accumulation. Model graphs are DAGs (a reticulation is reachable
    from both of its parents), so nodes are deduped on identity to be yielded
    exactly once.

    Args:
        root (ModelNode): Node to start from, normally the model root.
    Returns:
        Generator[ModelNode, None, None]: Nodes in post-order.
    """
    visited: set[int] = set()

    def _visit(node: ModelNode) -> Generator[ModelNode, None, None]:
        if id(node) in visited:
            return
        visited.add(id(node))
        for child in node.get_model_children():
            yield from _visit(child)
        yield node

    yield from _visit(root)


###############
#### BUILD ####
###############

def build_snp_model(net: Network, aln: MSA) -> SNPModel:
    """
    Build a SNP model graph from an in-memory network and alignment.

    Each network node becomes a typed model node classified by its degree, the
    network's edges are mirrored as child links, and every leaf is bound to its
    taxon's sequences. A :class:`RootAggregatorNode` is placed above the network
    root to collect the final whole-network likelihood.

    Args:
        net (Network): The species network.
        aln (MSA): The parsed biallelic alignment.
    Returns:
        SNPModel: The built graph, ready for :func:`postorder`.
    """
    model = SNPModel()
    model.network = net
    node_map: dict = {}

    for node in net.V():
        in_deg = net.in_degree(node)
        out_deg = net.out_degree(node)

        if out_deg == 0:
            branches = net.get_branches(node)["parent_branches"]
            new_node = LeafNode(node.label, branches[0])
            model.nodetypes["leaf"].append(new_node)
        elif in_deg == 1:
            branches = net.get_branches(node)["parent_branches"]
            new_node = InternalNode(node.label, branches[0])
            model.nodetypes["internal"].append(new_node)
        elif in_deg == 2 and out_deg == 1:
            branches = net.get_branches(node)["parent_branches"]
            new_node = ReticulationNode(node.label, branches[0], branches[1])
            model.nodetypes["reticulation"].append(new_node)
        elif in_deg == 0:
            new_node = RootNode(node.label)
            model.nodetypes["root"].append(new_node)

        node_map[node] = new_node

    edge: Edge
    for edge in net.E():
        node_map[edge.src].add_child(node_map[edge.dest])

    for network_node, model_node in node_map.items():
        if isinstance(model_node, LeafNode):
            seq_rec = aln.seq_by_name(network_node.label)
            model_node.set_data([seq_rec] if seq_rec else [])

    snp_root = RootAggregatorNode()
    snp_root.add_child(model.nodetypes["root"][0])
    model.root = snp_root

    return model
