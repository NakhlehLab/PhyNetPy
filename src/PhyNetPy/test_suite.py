

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
Last Edit : 3/11/25
First Included in Version : 1.0.0
Approved for Release: No
"""

import unittest
import numpy as np
from Alphabet import Alphabet, AlphabetError, DNA, RNA, PROTEIN, CODON, snp_alphabet
from BirthDeath import Yule, CBDP, BirthDeathSimulationError
from Network import Network, Node, Edge, UEdge, NodeError, EdgeError, NetworkError, __random_object, __dict_merge
from GeneTrees import GeneTrees, GeneTreeError, phynetpy_naming
from Infer_MP_Allop import INFER_MP_ALLOP, INFER_MP_ALLOP_BOOTSTRAP, ALLOP_SCORE
from State import State, acyclic_routine
from NJ import NJ, NJException

class TestAlphabet(unittest.TestCase):

    def test_dna_mapping(self):
        dna_alphabet = Alphabet(DNA)
        self.assertEqual(dna_alphabet.map('A'), 1)
        self.assertEqual(dna_alphabet.map('C'), 2)
        self.assertEqual(dna_alphabet.map('G'), 4)
        self.assertEqual(dna_alphabet.map('T'), 8)
        self.assertEqual(dna_alphabet.map('N'), 15)
        self.assertEqual(dna_alphabet.map('-'), 0)

    def test_rna_mapping(self):
        rna_alphabet = Alphabet(RNA)
        self.assertEqual(rna_alphabet.map('A'), 1)
        self.assertEqual(rna_alphabet.map('C'), 2)
        self.assertEqual(rna_alphabet.map('G'), 4)
        self.assertEqual(rna_alphabet.map('U'), 8)
        self.assertEqual(rna_alphabet.map('N'), 15)
        self.assertEqual(rna_alphabet.map('-'), 0)

    def test_protein_mapping(self):
        protein_alphabet = Alphabet(PROTEIN)
        self.assertEqual(protein_alphabet.map('A'), 1)
        self.assertEqual(protein_alphabet.map('C'), 3)
        self.assertEqual(protein_alphabet.map('G'), 7)
        self.assertEqual(protein_alphabet.map('T'), 19)
        self.assertEqual(protein_alphabet.map('X'), 22)
        self.assertEqual(protein_alphabet.map('-'), 0)

    def test_codon_mapping(self):
        codon_alphabet = Alphabet(CODON)
        self.assertEqual(codon_alphabet.map('A'), 1)
        self.assertEqual(codon_alphabet.map('C'), 2)
        self.assertEqual(codon_alphabet.map('G'), 4)
        self.assertEqual(codon_alphabet.map('U'), 8)
        self.assertEqual(codon_alphabet.map('N'), 14)
        self.assertEqual(codon_alphabet.map('-'), 0)

    def test_snp_alphabet(self):
        snp_alphabet_map = snp_alphabet(2)
        snp_alphabet_instance = Alphabet(snp_alphabet_map)
        self.assertEqual(snp_alphabet_instance.map('0'), 0)
        self.assertEqual(snp_alphabet_instance.map('1'), 1)
        self.assertEqual(snp_alphabet_instance.map('2'), 2)
        self.assertEqual(snp_alphabet_instance.map('N'), 3)
        self.assertEqual(snp_alphabet_instance.map('-'), 3)

    def test_invalid_mapping(self):
        dna_alphabet = Alphabet(DNA)
        with self.assertRaises(AlphabetError):
            dna_alphabet.map('Z')

    def test_get_type(self):
        dna_alphabet = Alphabet(DNA)
        self.assertEqual(dna_alphabet.get_type(), "DNA")
        custom_alphabet = Alphabet({'A': 1, 'B': 2})
        self.assertEqual(custom_alphabet.get_type(), "USER")

    def test_reverse_map(self):
        dna_alphabet = Alphabet(DNA)
        self.assertEqual(dna_alphabet.reverse_map(1), 'A')
        self.assertEqual(dna_alphabet.reverse_map(2), 'C')
        with self.assertRaises(AlphabetError):
            dna_alphabet.reverse_map(99)

class TestBirthDeath(unittest.TestCase):

    def test_yule_initialization(self):
        yule = Yule(gamma=0.5, n=10)
        self.assertEqual(yule.gamma, 0.5)
        self.assertEqual(yule.N, 10)
        self.assertEqual(yule.condition, "N")

    def test_yule_generate_network(self):
        yule = Yule(gamma=0.5, n=10)
        network = yule.generate_network()
        self.assertIsInstance(network, Network)
        self.assertEqual(len(network.V()), 19)  # 10 leaves + 9 internal nodes

    def test_yule_invalid_gamma(self):
        with self.assertRaises(BirthDeathSimulationError):
            Yule(gamma=-0.5, n=10)

    def test_yule_invalid_taxa(self):
        with self.assertRaises(BirthDeathSimulationError):
            Yule(gamma=0.5, n=1)

    def test_cbdp_initialization(self):
        cbdp = CBDP(gamma=0.5, mu=0.3, n=10)
        self.assertEqual(cbdp.gamma, 0.5 / 1)
        self.assertEqual(cbdp.mu, 0.3 - 0.5 * (1 - (1 / 1)))
        self.assertEqual(cbdp.N, 10)

    def test_cbdp_generate_network(self):
        cbdp = CBDP(gamma=0.5, mu=0.3, n=10)
        network = cbdp.generate_network()
        self.assertIsInstance(network, Network)
        self.assertEqual(len(network.V()), 19)  # 10 leaves + 9 internal nodes

    def test_cbdp_invalid_gamma_mu(self):
        with self.assertRaises(BirthDeathSimulationError):
            CBDP(gamma=0.3, mu=0.5, n=10)

    def test_cbdp_invalid_taxa(self):
        with self.assertRaises(BirthDeathSimulationError):
            CBDP(gamma=0.5, mu=0.3, n=1)

class TestNetwork(unittest.TestCase):

    def test_node_initialization(self):
        node = Node(name="A")
        self.assertEqual(node.label, "A")
        self.assertFalse(node.is_reticulation())
        self.assertEqual(node.get_attributes(), {})

    def test_node_attributes(self):
        node = Node(name="A", attr={"key": "value"})
        self.assertEqual(node.attribute_value("key"), "value")
        node.add_attribute("key2", "value2")
        self.assertEqual(node.attribute_value("key2"), "value2")

    def test_node_time(self):
        node = Node(name="A")
        node.set_time(5.0)
        self.assertEqual(node.get_time(), 5.0)
        with self.assertRaises(NodeError):
            node.get_time()

    def test_edge_initialization(self):
        node1 = Node(name="A")
        node2 = Node(name="B")
        edge = Edge(source=node1, destination=node2, length=1.0, gamma=0.5)
        self.assertEqual(edge.src, node1)
        self.assertEqual(edge.dest, node2)
        self.assertEqual(edge.get_length(), 1.0)
        self.assertEqual(edge.get_gamma(), 0.5)

    def test_uedge_initialization(self):
        node1 = Node(name="A")
        node2 = Node(name="B")
        uedge = UEdge(n1=node1, n2=node2, length=1.0)
        self.assertEqual(uedge.n1, node1)
        self.assertEqual(uedge.n2, node2)
        self.assertEqual(uedge.get_length(), 1.0)

    def test_network_initialization(self):
        network = Network()
        self.assertEqual(len(network.V()), 0)
        self.assertEqual(len(network.E()), 0)

    def test_network_add_nodes(self):
        network = Network()
        node = Node(name="A")
        network.add_nodes(node)
        self.assertIn(node, network.V())

    def test_network_add_edges(self):
        network = Network()
        node1 = Node(name="A")
        node2 = Node(name="B")
        network.add_nodes(node1, node2)
        edge = Edge(source=node1, destination=node2)
        network.add_edges(edge)
        self.assertIn(edge, network.E())

    def test_network_remove_nodes(self):
        network = Network()
        node = Node(name="A")
        network.add_nodes(node)
        network.remove_nodes(node)
        self.assertNotIn(node, network.V())

    def test_network_remove_edges(self):
        network = Network()
        node1 = Node(name="A")
        node2 = Node(name="B")
        network.add_nodes(node1, node2)
        edge = Edge(source=node1, destination=node2)
        network.add_edges(edge)
        network.remove_edge(edge)
        self.assertNotIn(edge, network.E())

    def test_random_object(self):
        rng = np.random.default_rng(42)
        mylist = [1, 2, 3, 4, 5]
        result = __random_object(mylist, rng)
        self.assertIn(result, mylist)

    def test_dict_merge(self):
        dict1 = {"a": 1, "b": 2}
        dict2 = {"b": 3, "c": 4}
        result = __dict_merge(dict1, dict2)
        self.assertEqual(result, {"a": [1], "b": [2, 3], "c": [4]})

class TestGeneTrees(unittest.TestCase):

    def test_phynetpy_naming(self):
        self.assertEqual(phynetpy_naming("01A"), "A")
        self.assertEqual(phynetpy_naming("99B"), "B")
        with self.assertRaises(GeneTreeError):
            phynetpy_naming("A01")
        with self.assertRaises(GeneTreeError):
            phynetpy_naming("012")

    def test_gene_trees_initialization(self):
        gene_trees = GeneTrees()
        self.assertEqual(len(gene_trees.trees), 0)
        self.assertEqual(len(gene_trees.taxa_names), 0)
        self.assertEqual(gene_trees.naming_rule, phynetpy_naming)

    def test_add_gene_tree(self):
        network = Network()
        node1 = Node(name="01A")
        node2 = Node(name="02B")
        network.add_nodes(node1, node2)
        network.add_edges(Edge(source=node1, destination=node2))
        gene_trees = GeneTrees()
        gene_trees.add(network)
        self.assertIn(network, gene_trees.trees)
        self.assertIn("01A", gene_trees.taxa_names)
        self.assertIn("02B", gene_trees.taxa_names)

    def test_mp_allop_map(self):
        network = Network()
        node1 = Node(name="01A")
        node2 = Node(name="02B")
        network.add_nodes(node1, node2)
        network.add_edges(Edge(source=node1, destination=node2))
        gene_trees = GeneTrees([network])
        subgenome_map = gene_trees.mp_allop_map()
        self.assertEqual(subgenome_map, {"A": ["01A"], "B": ["02B"]})

class TestInferMPAllop(unittest.TestCase):

    def setUp(self):
        self.gene_tree_file = "/path/to/gene_tree_file.nex"
        self.start_network_file = "/path/to/start_network_file.nex"
        self.subgenome_assign = {
            'A': ['01aA', '01aB'],
            'B': ['01bA', '01bB']
        }
        self.iter_ct = 100
        self.seed = 42

    def test_infer_mp_allop(self):
        results = INFER_MP_ALLOP(
            self.gene_tree_file,
            self.subgenome_assign,
            self.iter_ct,
            self.seed
        )
        self.assertIsInstance(results, dict)
        for network, score in results.items():
            self.assertIsInstance(network, Network)
            self.assertIsInstance(score, float)

    def test_infer_mp_allop_bootstrap(self):
        results = INFER_MP_ALLOP_BOOTSTRAP(
            self.start_network_file,
            self.gene_tree_file,
            self.subgenome_assign,
            self.iter_ct,
            self.seed
        )
        self.assertIsInstance(results, dict)
        for network, score in results.items():
            self.assertIsInstance(network, Network)
            self.assertIsInstance(score, float)

    def test_allop_score(self):
        score = ALLOP_SCORE(
            self.start_network_file,
            self.gene_tree_file,
            self.subgenome_assign
        )
        self.assertIsInstance(score, int)

class TestState(unittest.TestCase):

    def setUp(self):
        self.network = Network()
        self.node1 = Node(name="A")
        self.node2 = Node(name="B")
        self.edge = Edge(source=self.node1, destination=self.node2)
        self.network.add_nodes(self.node1, self.node2)
        self.network.add_edges(self.edge)
        self.state = State(model=None)

    def test_likelihood(self):
        self.state.current_model = self.network
        self.assertEqual(self.state.likelihood(), self.network.likelihood())

    def test_generate_next(self):
        move = Move()
        self.assertTrue(self.state.generate_next(move))

    def test_revert(self):
        move = Move()
        self.state.generate_next(move)
        self.state.revert(move)
        self.assertEqual(self.state.proposed_model, self.state.current_model)

    def test_commit(self):
        move = Move()
        self.state.generate_next(move)
        self.state.commit(move)
        self.assertEqual(self.state.current_model, self.state.proposed_model)

    def test_validate_proposed_network(self):
        move = Move()
        self.state.generate_next(move)
        self.assertTrue(self.state.validate_proposed_network(move))

class TestNJ(unittest.TestCase):

    def setUp(self):
        self.network = Network()
        self.node1 = Node(name="A")
        self.node2 = Node(name="B")
        self.node3 = Node(name="C")
        self.network.add_nodes(self.node1, self.node2, self.node3)
        self.edge1 = Edge(source=self.node1, destination=self.node2, length=1.0)
        self.edge2 = Edge(source=self.node2, destination=self.node3, length=1.0)
        self.network.add_edges(self.edge1, self.edge2)
        self.distance_matrix = {
            (self.node1, self.node2): 1.0,
            (self.node1, self.node3): 2.0,
            (self.node2, self.node3): 1.0
        }

    def test_nj(self):
        result = NJ(self.network, d=self.distance_matrix)
        self.assertIsInstance(result, Network)

    def test_nj_exception(self):
        with self.assertRaises(NJException):
            NJ(self.network)

if __name__ == '__main__':
    unittest.main()
