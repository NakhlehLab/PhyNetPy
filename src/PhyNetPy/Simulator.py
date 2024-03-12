import collections
import copy
from enum import Enum, unique, auto
from abc import ABC, abstractmethod
from Graph import DAG
from Node import Node
from NetworkParser import NetworkParser
import random
import math
import numpy as np
import utils
import dendropy
from dendropy.simulate import treesim


class NetworkConverter:
    def __init__(self, net: DAG):
        self.network = net
        self.mul_tree = None
        self.network_to_mul_map = {}

    def convert_to_multree(self) -> DAG:                           # I already have a MUL tree class, in Infer_MP_Allop.py (that I'll move to Graph.py), if you'd like to use that!
        """Creates a multilabeled tree from a network"""
        
        self.validate_network() 

        self.mul_tree = DAG()
        self._copy_network_nodes()
        self._add_network_edges()
        self._process_reticulations()

        # remove excess connection nodes
        utils.remove_binary_nodes(self.mul_tree)

        return self.mul_tree

    def validate_network(self):
        """Checks network validity for correct ploidyness and structure"""
        pass

    def make_clade_copy(self, network: DAG, node: Node) -> DAG: # dag.subtree_copy does this if you'd like to use that as well
        q = collections.deque([node])
        nodes, edges = [], []

        new_node = Node(name=node.get_name() + "_copy")
        nodes.append(new_node)
        netnode_to_mulnode = {node: new_node}

        while q:
            cur = q.pop()
            for neighbor in network.get_children(cur):
                new_node = Node(name=neighbor.get_name() + "_copy")
                nodes.append(new_node)
                netnode_to_mulnode[neighbor] = new_node
                edges.append((netnode_to_mulnode[cur], new_node))

                # resume search from the end of the chain if one exists, or this is neighbor if nothing was done
                q.append(neighbor)

        return DAG(edges=edges, nodes=nodes)

    def _copy_network_nodes(self):
        self.network_to_mul_map = {
            node: Node(name=node.get_name()) for node in self.network.nodes
        }
        self.mul_tree.add_nodes(list(self.network_to_mul_map.values()))

    def _add_network_edges(self):
        for edge in self.network.edges:
            u, v = self.network_to_mul_map[edge[0]], self.network_to_mul_map[edge[1]]
            self.mul_tree.add_edges([u, v])

    def _process_reticulations(self):
        processed: set[Node] = set()
        traversal_queue = collections.deque(self.mul_tree.get_leaves())

        while traversal_queue:
            cur = traversal_queue.pop()
            parents = [p for p in self.mul_tree.get_parents(cur)]

            if self.mul_tree.in_degree(cur) == 2:
                clade_copy_root = self._handle_reticulation(cur, parents)
                processed.add(clade_copy_root)

            processed.add(cur)

            for parent in parents:
                if all(child in processed for child in self.mul_tree.get_children(parent)):
                    traversal_queue.append(parent)

    def _handle_reticulation(self, reticulation_node, parents):
        a, b = parents
        subtree = self.make_clade_copy(self.mul_tree, reticulation_node)
        self.mul_tree.remove_edge([b, reticulation_node])
        self.mul_tree.add_edges([b, subtree.root()[0]])
        self.mul_tree.add_nodes(subtree.nodes)
        self.mul_tree.add_edges(subtree.edges)
        return subtree.root()[0]


@unique
class Event(Enum):
    ALLO_RECIPIENT = auto()
    AUTO_RECIPIENT = auto()
    LEAF = auto()
    SPECIATION = auto()
    HYBRID_DONATION = auto()
    HYBRID_DONATION_FROM_EXTINCT_DONOR = auto()
    ROOT = auto()
    UNSAMPLED_LEAF = auto()
    LOSS = auto()
    DUPLICATION = auto()


@unique
class NodeType(Enum):
    ROOT = "ROOT"
    SPECIATION = "SPECIATION"
    LEAF = "LEAF"
    HYBRID_DONOR = "HYBRID_DONOR"
    EXTINCT_HYBRID_DONOR = "EXTINCT_HYBRID_DONOR"
    ALLOPOLYPLOID = "ALLOPOLYPLOID"
    AUTOPOLYPLOID = "AUTOPOLYPLOID"


"""
Adapted from 
https://github.com/arvestad/jprime/blob/master/src/main/java/se/cbb/jprime/apps/genphylodata/GuestTreeInHybridGraphCreator.java
"""
class TreeSimulator:
    def __init__(self, loss_rate: float, extant_sampling_rate: float,
                 post_poly_timespan: float, post_poly_loss_rate: float, seed: int = None):
        self.seed = seed if seed else random.randint(0, 10000000)
        self.rng = random.Random(self.seed)
        self.post_poly_timespan = post_poly_timespan
        # self.birthrate = dup_rate
        self.deathrate = loss_rate
        # self.post_poly_birthrate = post_poly_dup_rate
        self.post_poly_deathrate = post_poly_loss_rate
        self.rho = extant_sampling_rate  # sampling probability of extant exon; rho = 1 indicates complete sampling

    def prune_tree(self, unpruned_tree: dendropy.Tree):
        tree = copy.deepcopy(unpruned_tree)
        tree.is_rooted = True
        taxa_to_remove = [node.taxon for node in tree.leaf_nodes() if node.event in [Event.LOSS, Event.UNSAMPLED_LEAF]]
        tree.prune_taxa(taxa_to_remove)

        # processed = set()
        # for node in nodes_to_remove:
        #     if node in processed:
        #         continue
        #     processed.add(node)
        #     try:
        #         nodes_to_remove.remove(node)
        #     except ValueError:
        #         pass
        #     assert not node._child_nodes
        #
        #     while (node.parent_node is not None) and (len(node.parent_node._child_nodes) == 1):
        #         node = node.parent_node
        #         processed.add(node)
        #     tree.prune_subtree(node, suppress_unifurcations=False)

        # tree.suppress_unifurcations()
        return tree


    def generate_gene_trees(self, network: DAG, n_individuals_per_species, n_gene_trees, save_dir=None):
        tns = dendropy.TaxonNamespace()
        unpruned_containing_tree = self.generate_unpruned_containing_tree(network, tns)
        if save_dir is not None:
            unpruned_containing_tree.write(path=f"{save_dir}/unpruned_containing_tree.nexus", schema="nexus",
                                  suppress_edge_lengths=False,
                                  unquoted_underscores=True)

        print("Unpruned containing tree:")
        print(unpruned_containing_tree.as_string(schema="nexus", suppress_edge_lengths=False))
        containing_tree = self.prune_tree(unpruned_containing_tree)
        print("Pruned containing tree:")
        print(containing_tree.as_string(schema="nexus", suppress_edge_lengths=False))
        assert len(containing_tree.leaf_nodes()) > 1
        gene_trees = self.simulate_coalescence(containing_tree, n_individuals_per_species, n_gene_trees)
        if save_dir is not None:
            gene_trees.write(path=f"{save_dir}/gene_trees.newick", schema="newick", suppress_edge_lengths=False, unquoted_underscores=True)

        return containing_tree, gene_trees

    def generate_unpruned_containing_tree(self, network, taxon_namespace: dendropy.TaxonNamespace):
        host_node = network.root()[0]

        if taxon_namespace is None:
            taxon_namespace = dendropy.TaxonNamespace()

        tree = dendropy.Tree(taxon_namespace=taxon_namespace)
        tree.is_rooted = True
        tree.seed_node.edge.length = 0.0
        self._add_node_info(tree.seed_node, host_node, Event.ROOT, host_node.attribute_value('t'))
        alive_nodes = collections.deque([tree.seed_node])

        while alive_nodes:
            lineage = alive_nodes.pop()
            event = lineage.event
            if event in (Event.LEAF, Event.LOSS, Event.UNSAMPLED_LEAF):
                continue

            host_node = lineage.netnode
            print(host_node.get_name())
            if event in (Event.SPECIATION, Event.ROOT):
                host_children = network.get_children(host_node)
                for child in host_children:
                    child_lineage = self.create_node(host_node, child, lineage.age)
                    lineage.add_child(child_lineage)
                    alive_nodes.append(child_lineage)

            # elif event == Event.DUPLICATION:
            #     child_lineages = [self.create_node(host_node, lineage.age) for _ in range(2)]
            #     for child_lineage in child_lineages:
            #         lineage.add_child(child_lineage)
            #         alive_nodes.append(child_lineage)

            elif event == Event.HYBRID_DONATION:
                hyb_child, tree_child = network.get_children(host_node)
                if hyb_child.attribute_value('type') not in [NodeType.AUTOPOLYPLOID, NodeType.ALLOPOLYPLOID]:
                    hyb_child, tree_child = tree_child, hyb_child

                donor_age = host_node.attribute_value('t')
                hyb_type = hyb_child.attribute_value('type')
                hyb_age = hyb_child.attribute_value('t')

                if hyb_type == NodeType.ALLOPOLYPLOID:
                    node1 = dendropy.Node(edge_length=donor_age - hyb_age)
                    self._add_node_info(node1, hyb_child, Event.ALLO_RECIPIENT,
                                        hyb_age)  ###TODO: CHECK hyb_age or donor age
                elif hyb_type == NodeType.AUTOPOLYPLOID:
                    node1 = dendropy.Node(edge_length=donor_age - hyb_age)
                    self._add_node_info(node1, hyb_child, Event.AUTO_RECIPIENT,
                                        hyb_age)  #### ###TODO: CHECK hyb_age or donor age
                else:
                    raise ValueError("Unknown node type!")

                node2 = self.create_node(host_node, tree_child, lineage.age)
                lineage.set_child_nodes([node1, node2])
                alive_nodes.extend([node1, node2])

            elif event == Event.HYBRID_DONATION_FROM_EXTINCT_DONOR:
                hyb_child, tree_child = network.get_children(host_node)[0]
                donor_age = host_node.attribute_value('t')
                hyb_type = hyb_child.attribute_value('type')
                hyb_child_age = hyb_child.attribute_value('t')

                if hyb_type == NodeType.ALLOPOLYPLOID:
                    node1 = dendropy.Node(edge_length=donor_age - hyb_child_age)
                    self._add_node_info(node1, hyb_child, Event.ALLO_RECIPIENT, donor_age)
                elif hyb_type == NodeType.AUTOPOLYPLOID:
                    node1 = dendropy.Node(edge_length=donor_age - hyb_child_age)
                    self._add_node_info(node1, hyb_child, Event.AUTO_RECIPIENT, donor_age)
                else:
                    raise ValueError("Unknown node type!")

                lineage.add_child(node1)
                alive_nodes.append(node1)

            elif event == Event.ALLO_RECIPIENT:
                c = self.create_node(host_node, network.get_children(host_node)[0], lineage.age)
                lineage.add_child(c)
                alive_nodes.append(c)

            elif event == Event.AUTO_RECIPIENT:
                tree_child = network.get_children(host_node)
                child_lineages = [self.create_node(host_node, tree_child, lineage.age) for _ in range(2)]
                lineage.set_children(child_lineages)
                alive_nodes.extend(child_lineages)

        self.add_node_names(tree, taxon_namespace)
        return tree

    def simulate_coalescence(self, containing_tree: dendropy.Tree, num_individuals_per_species=1, num_gene_trees=1) -> dendropy.TreeList:
        gene_to_species_map = dendropy.TaxonNamespaceMapping.create_contained_taxon_mapping(
            containing_taxon_namespace=containing_tree.taxon_namespace, num_contained=num_individuals_per_species,
            contained_taxon_label_separator="_"
        )

        gene_trees = dendropy.TreeList()
        for i in range(num_gene_trees):
            gene_trees.append(treesim.contained_coalescent_tree(containing_tree=containing_tree,
                                                                gene_to_containing_taxon_map=gene_to_species_map,
                                                                rng=self.rng))
        return gene_trees

    def add_node_names(self, tree, taxon_namespace=None):
        if not taxon_namespace:
            taxon_namespace = tree.taxon_namespace

        species_to_gene_count = dict()
        for leaf in tree.leaf_nodes():
            species = leaf.netnode.get_name()
            if species not in species_to_gene_count:
                species_to_gene_count[species] = 1
            else:
                species_to_gene_count[species] += 1

            leaf.taxon = taxon_namespace.require_taxon(label=f"{species}_{species_to_gene_count[species]}")

    def _add_node_info(self, node: dendropy.Node, host_node: Node, event: Event, age: float):
        node.netnode = None
        node.event = None
        node.annotations.add_bound_attribute("netnode")
        node.annotations.add_bound_attribute("event")
        node.netnode = host_node
        node.event = event
        node.age = age

    def create_node(self, branch_top, branch_bottom: Node, start_time: float):
        # branch_top = branch_bottom.get_parent()
        bottom_time, bottom_type = branch_bottom.attribute_value(
            "t"), branch_bottom.attribute_value("type")
        top_time, top_type = branch_top.attribute_value("t"), branch_top.attribute_value("type")

        # Check if the current node is within the post-polyploidization timespan
        if top_type in (
                NodeType.ALLOPOLYPLOID, NodeType.AUTOPOLYPLOID) and start_time >= top_time - self.post_poly_timespan:
            # increasing loss rate just after a WGD event
            # pd = np.random.exponential(max(self.post_poly_birthrate + self.post_poly_deathrate, 1e-48))
            pd = np.random.exponential(max(self.post_poly_deathrate, 1e-48))
            time = pd
            if start_time - time < top_time - self.post_poly_timespan:
                # pd = np.random.exponential(max(self.birthrate + self.deathrate, 1e-48))
                # sample again according to normal rates
                pd = np.random.exponential(max(self.deathrate, 1e-48))
                time += pd
                within_post_poly = False
            else:
                within_post_poly = True
        else:
            # pd = np.random.exponential(max(self.birthrate + self.deathrate, 1e-48))
            pd = np.random.exponential(max(self.deathrate, 1e-48))
            time = pd
            within_post_poly = False

        # Determine Event
        event_time = start_time - time
        event = None

        if event_time > bottom_time:
            rnd = self.rng.random()

            if within_post_poly:
                event = Event.LOSS if rnd < self.post_poly_deathrate else None
            else:
                event = Event.LOSS if rnd < self.deathrate else None

            if event is None:
                event_time = bottom_time

        if event_time <= bottom_time:
            event_time = bottom_time
            time = round(start_time - event_time, 8)
            rnd = self.rng.random()

            if bottom_type == NodeType.LEAF:
                event = Event.LEAF if rnd < self.rho else Event.UNSAMPLED_LEAF
            elif bottom_type == NodeType.SPECIATION:
                event = Event.SPECIATION
            elif bottom_type == NodeType.EXTINCT_HYBRID_DONOR:
                event = Event.HYBRID_DONATION_FROM_EXTINCT_DONOR
            elif bottom_type == NodeType.HYBRID_DONOR:
                event = Event.HYBRID_DONATION
            else:
                raise NotImplementedError(f"Lineage evolved to an unexpected event type: {bottom_type}")

        new_node = dendropy.Node(edge_length=time)
        self._add_node_info(new_node, branch_bottom, event, event_time)
        return new_node


def allonet_test():
    network = \
        NetworkParser("/Users/zhiyan/Documents/research/auto-allo/phynetpy_dev/src/testsim.nex").get_all_networks()[0]
    network.root()[0].add_attribute('type', NodeType.ROOT)
    for tip in network.get_leaves():
        tip.add_attribute('type', NodeType.LEAF)
        tip.add_attribute('t', 0)

    for nd in network.get_nodes():
        if nd.is_reticulation():
            nd.add_attribute('type', NodeType.ALLOPOLYPLOID)
            for par in nd.get_parent(return_all=True):
                par.add_attribute('type', NodeType.HYBRID_DONOR)

        if not nd.attribute_value('type'):
            nd.add_attribute('type', NodeType.SPECIATION)

    utils.init_node_heights(network)
    print("Network: " + network.to_newick())
    simulator = TreeSimulator(loss_rate=0.1, extant_sampling_rate=0.5,
                              post_poly_timespan=0.4, post_poly_loss_rate=0.3,
                              seed=1234234)

    containing_tree, gene_trees = simulator.generate_gene_trees(network, 1, 1)
    # print("Containing tree: " + containing_tree.as_string(schema="nexus", suppress_edge_lengths=False))
    print("Gene trees:\t")
    for gt in gene_trees:
        print(gt.as_string(schema="nexus", suppress_edge_lengths=False))



if __name__ == "__main__":
    allonet_test()