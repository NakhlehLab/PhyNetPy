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
Last Edit : 9/16/25
First Included in Version : 2.0.0

Docs   - [ ]
Tests  - [ ] 
Design - [x]
"""

import os
import tempfile
import subprocess
import warnings
from io import StringIO
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union

from Bio import Phylo

from .Network import *
from .GraphUtils import get_all_clusters

#########################
#### EXCEPTION CLASS #### 
#########################   
    
class GeneTreeError(Exception):
    """
    Error class for all errors relating to gene trees.
    """
    def __init__(self, message : str = "Gene Tree Module Error") -> None:
        """
        Initialize a GeneTreeError with a message.

        Args:
            message (str, optional): Custom error message. Defaults to 
                                     "Gene Tree Module Error".
        Returns:
            N/A
        """
        super().__init__(message)
        self.message = message
        
##########################
#### HELPER FUNCTIONS #### 
##########################
  
def phynetpy_naming(taxa_name: str) -> str:
    """
    The default method for sorting taxa labels into groups

    Args:
        taxa_name (str): a taxa label from a nexus file
    Raises:
        GeneTreeError: if there is a problem applying the naming rule

    Returns:
        str: a string that is the key for this label
    """
    if not taxa_name[0:2].isnumeric():
        raise GeneTreeError("Error Applying PhyNetPy Naming Rule: \
                             first 2 digits is not numerical")
    
    if taxa_name[2].isalpha():
        return taxa_name[2].upper()
    else:
        raise GeneTreeError("Error Applying PhyNetPy Naming Rule: \
                             3rd position is not an a-z character")

def external_naming(taxa_name: str) -> str:
    """
    TODO: Examine the need for this function and remove if not needed.
    """
    return taxa_name.split("_")[0]
    


####################
#### GENE TREES ####
####################

class GeneTrees:
    """
    A container for a set of networks that are binary and represent a 
    gene tree.
    
    Supports two strategies for grouping gene labels into species:
      1. An explicit ``species_gene_mapping`` dict (species -> gene labels).
      2. A ``naming_rule`` callable that derives the species key from a
         gene label string.
    
    When neither is provided every gene label is treated as its own species
    (identity mapping).
    """
    
    def __init__(
        self,
        gene_tree_list: Optional[List[Network]] = None,
        naming_rule: Optional[Callable[..., Any]] = None,
        species_gene_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Wrapper class for a set of networks that represent gene trees.

        Args:
            gene_tree_list (list[Network], optional): A list of networks, 
                of the binary tree variety. Defaults to None.
            naming_rule (Callable[..., Any], optional): A function 
                f : str -> str that maps a gene label to its species key.
                Ignored when *species_gene_mapping* is provided.
                Defaults to None (identity mapping).
            species_gene_mapping (dict[str, list[str]], optional): Explicit
                mapping from species name to a list of gene/allele labels.
                Takes priority over *naming_rule* when both are given.
        """
        
        self.trees: Set[Network] = set[Network]()
        self.taxa_names: Set[str] = set[str]()
        self._species_gene_mapping: Optional[Dict[str, List[str]]] = (
            species_gene_mapping
        )
        self.naming_rule: Optional[Callable[..., Any]] = naming_rule
        
        if gene_tree_list is not None:
            for tree in gene_tree_list:
                self.add(tree)
        
    def add(self, tree: Network) -> None:
        """
        Add a gene tree to the collection. Any new gene labels that belong to
        this tree will also be added to the collection of all 
        gene tree leaf labels.

        Args:
            tree (Network): A network that is a tree, must be binary.
        """

        self.trees.add(tree)
        
        for leaf in tree.get_leaves():
            self.taxa_names.add(leaf.label)
        
    @property
    def species_gene_mapping(self) -> Optional[Dict[str, List[str]]]:
        """Return the explicit species-to-gene mapping, if set."""
        return self._species_gene_mapping

    @species_gene_mapping.setter
    def species_gene_mapping(self, value: Optional[Dict[str, List[str]]]) -> None:
        self._species_gene_mapping = value

    def _resolve_mapping(self) -> Dict[str, List[str]]:
        """
        Build the species -> gene-labels mapping using the best available
        strategy.

        Priority:
          1. Explicit ``species_gene_mapping`` (filtered to present taxa).
          2. ``naming_rule`` callable applied to each taxon.
          3. Identity mapping (each label is its own species).

        Returns:
            dict[str, list[str]]: species -> gene label mapping.
        """
        if self._species_gene_mapping is not None:
            present = self.taxa_names
            return {
                sp: [g for g in genes if g in present]
                for sp, genes in self._species_gene_mapping.items()
                if any(g in present for g in genes)
            }

        if self.naming_rule is not None:
            mapping: Dict[str, List[str]] = {}
            for name in self.taxa_names:
                key = self.naming_rule(name)
                mapping.setdefault(key, []).append(name)
            return mapping

        return {name: [name] for name in self.taxa_names}

    def mp_allop_map(self) -> Dict[str, List[str]]:
        """
        Create a subgenome mapping from the stored set of gene trees.

        Uses the explicit species_gene_mapping if available, otherwise
        falls back to the naming_rule, and finally to an identity mapping.

        Returns:
            dict[str, list[str]]: subgenome mapping
        """
        return self._resolve_mapping()

    def validate_mapping(self) -> List[str]:
        """
        Check the explicit species_gene_mapping against the actual taxa
        present in the gene trees.

        Returns:
            list[str]: A list of warning/error messages. Empty means valid.
        """
        issues: List[str] = []
        if self._species_gene_mapping is None:
            return issues

        mapped_genes: Set[str] = set()
        for sp, genes in self._species_gene_mapping.items():
            for g in genes:
                if g in mapped_genes:
                    issues.append(
                        f"Gene '{g}' appears under multiple species entries"
                    )
                mapped_genes.add(g)
                if g not in self.taxa_names:
                    issues.append(
                        f"Mapped gene '{g}' (species '{sp}') not found "
                        f"in any gene tree"
                    )

        unmapped = self.taxa_names - mapped_genes
        if unmapped:
            issues.append(
                f"Taxa not covered by mapping: {sorted(unmapped)}"
            )
            warnings.warn(
                f"GeneTrees: {len(unmapped)} taxa not covered by the "
                f"species-gene mapping: {sorted(unmapped)}"
            )

        return issues

    def cluster_support(self,
                        include_trivial: bool = False,
                        normalize: bool = True) -> Dict[FrozenSet[str], float]:
        """
        Aggregate support for all rooted clusters across the gene tree set.

        Args:
            include_trivial (bool): include size-1 clusters. Defaults to False.
            normalize (bool): return frequencies in [0,1] instead of counts.

        Returns:
            dict[FrozenSet[str], float]: map cluster (as frozenset of leaf labels)
                                         to count or frequency.
        """
        counts: Dict[FrozenSet[str], int] = {}
        num_trees = len(self.trees)
        if num_trees == 0:
            return {}

        for tree in self.trees:
            clusters = get_all_clusters(tree, include_trivial=include_trivial)
            # Convert to label-level clusters
            label_clusters = [frozenset(n.label for n in clus) for clus in clusters]
            for lc in label_clusters:
                counts[lc] = counts.get(lc, 0) + 1

        if not normalize:
            return {k: float(v) for k, v in counts.items()}
        return {k: v / num_trees for k, v in counts.items()}

    def split_support(self, normalize: bool = True) -> Dict[FrozenSet[str], float]:
        """
        Aggregate support for unrooted splits (bipartitions), canonicalized
        to the smaller side of the split.

        Note: Only non-trivial splits (both parts size >= 2) are counted.

        Args:
            normalize (bool): return frequencies in [0,1] instead of counts.

        Returns:
            dict[FrozenSet[str], float]: map smaller-side cluster to count/freq.
        """
        counts: Dict[FrozenSet[str], int] = {}
        num_trees = len(self.trees)
        if num_trees == 0:
            return {}

        for tree in self.trees:
            taxa: Set[str] = set(leaf.label for leaf in tree.get_leaves())
            clusters = get_all_clusters(tree, include_trivial=False)
            for clus in clusters:
                a = set(n.label for n in clus)
                b = taxa.difference(a)
                if len(a) >= 2 and len(b) >= 2:
                    key = frozenset(a) if len(a) <= len(b) else frozenset(b)
                    counts[key] = counts.get(key, 0) + 1

        if not normalize:
            return {k: float(v) for k, v in counts.items()}
        return {k: v / num_trees for k, v in counts.items()}

    def support_on_reference(self,
                             ref_tree: Network,
                             include_trivial: bool = False,
                             normalize: bool = True) -> Dict[FrozenSet[str], float]:
        """
        Compute support of each rooted cluster present in a reference tree.

        Args:
            ref_tree (Network): reference binary tree.
            include_trivial (bool): include size-1 clusters. Defaults to False.
            normalize (bool): return frequencies in [0,1].

        Returns:
            dict[FrozenSet[str], float]: map of ref clusters to support.
        """
        # Precompute overall cluster support across the set
        all_support = self.cluster_support(include_trivial=include_trivial,
                                           normalize=normalize)

        sup_map: Dict[FrozenSet[str], float] = {}
        clusters = get_all_clusters(ref_tree, include_trivial=include_trivial)
        for clus in clusters:
            key = frozenset(n.label for n in clus)
            sup_map[key] = all_support.get(key, 0.0)
        return sup_map

    def annotate_reference_support(self,
                                   ref_tree: Network,
                                   include_trivial: bool = False,
                                   normalize: bool = True) -> None:
        """
        Annotate the reference tree's internal edges with support values stored
        in the edge weight field.

        For edge (u->v), the cluster is the set of leaf descendants of v.

        Args:
            ref_tree (Network): reference binary tree to annotate in place.
            include_trivial (bool): include size-1 clusters. Defaults to False.
            normalize (bool): frequencies vs counts.
        """
        # Build per-cluster support for ref
        ref_support = self.support_on_reference(ref_tree,
                                                include_trivial=include_trivial,
                                                normalize=normalize)

        # Total taxa for trivial filtering
        total_taxa = set(leaf.label for leaf in ref_tree.get_leaves())

        for edge in ref_tree.E():
            if not isinstance(edge, Edge):
                continue
            # Cluster is leaves under child
            clus = ref_tree.leaf_descendants(edge.dest)
            labels = {n.label for n in clus}
            # Skip trivial by default
            if not include_trivial and (len(labels) <= 1 or len(labels) == len(total_taxa)):
                continue
            val = ref_support.get(frozenset(labels), 0.0)
            edge.set_weight(float(val))

    def consensus_clusters(self,
                           threshold: float = 0.5,
                           include_trivial: bool = False) -> List[Set[str]]:
        """
        Return the set of rooted clusters whose support >= threshold.

        Note: This does not resolve incompatibilities; it is a simple filter.

        Args:
            threshold (float): minimum frequency in [0,1].
            include_trivial (bool): include size-1 clusters.

        Returns:
            list[Set[str]]: clusters passing threshold.
        """
        support = self.cluster_support(include_trivial=include_trivial,
                                       normalize=True)
        return [set(c) for c, v in support.items() if v >= threshold]

    def rf_distance(self, ref_tree: Network, normalize: bool = False) -> float:
        """
        Compute the average Robinson-Foulds distance (rooted, using clusters)
        between each gene tree and the reference.

        Args:
            ref_tree (Network): reference tree.
            normalize (bool): if True, divide by the maximum possible RF for the
                              shared taxon set of each comparison.

        Returns:
            float: average RF distance across the gene trees.
        """
        if len(self.trees) == 0:
            return 0.0

        ref_taxa = set(leaf.label for leaf in ref_tree.get_leaves())
        ref_clusters = set(frozenset(n.label for n in c)
                           for c in get_all_clusters(ref_tree, include_trivial=False))

        total = 0.0
        for tree in self.trees:
            taxa = set(leaf.label for leaf in tree.get_leaves())
            common = ref_taxa.intersection(taxa)
            if len(common) < 3:
                # Too small to define internal clusters
                continue
            # Project clusters to common taxa by intersecting
            def project_clusters(net: Network) -> Set[FrozenSet[str]]:
                return set(
                    frozenset(n.label for n in c if n.label in common)
                    for c in get_all_clusters(net, include_trivial=False)
                )

            a = ref_clusters
            b = project_clusters(tree)
            # Remove any empty projections
            a = set(clus for clus in a if len(clus) >= 2 and len(clus) < len(common))
            b = set(clus for clus in b if len(clus) >= 2 and len(clus) < len(common))

            rf = len(a.difference(b)) + len(b.difference(a))
            if normalize:
                # Max RF equals sum of cluster counts for both trees over common taxa
                max_rf = len(a) + len(b)
                total += (rf / max_rf) if max_rf > 0 else 0.0
            else:
                total += float(rf)

        return total / len(self.trees)

    def build_majority_rule_consensus_tree(self,
                                           threshold: float = 0.5) -> Network:
        """
        Construct a majority-rule (or threshold) consensus tree from the gene trees.

        Greedy compatibility: sort clusters by support descending; add if
        compatible with current set; then realize the set into a (possibly
        multifurcating) tree over the union of taxa.
        """
        taxa: Set[str] = set(self.taxa_names)
        if len(taxa) == 0:
            return Network()

        sup = self.cluster_support(include_trivial=False, normalize=True)
        # Filter by threshold and sort by support then size (large first)
        cand = [(c, v) for c, v in sup.items() if v >= threshold]
        cand.sort(key=lambda x: (x[1], len(x[0])), reverse=True)

        selected: List[FrozenSet[str]] = []
        def compatible(a: FrozenSet[str], b: FrozenSet[str]) -> bool:
            # Clusters a,b are compatible if one subset of the other or disjoint
            return a.issubset(b) or b.issubset(a) or a.isdisjoint(b)

        for c, _ in cand:
            if all(compatible(c, s) for s in selected):
                selected.append(c)

        return self._clusters_to_tree(taxa, set(selected))

    def _clusters_to_tree(self, taxa: Set[str], clusters: Set[FrozenSet[str]]) -> Network:
        """
        Realize a compatible set of clusters as a (possibly multifurcating) tree.

        Strategy: recursively partition taxa by maximal proper clusters within
        the current subset. Singleton taxa become leaves.
        """
        net = Network()
        name_to_node: Dict[str, Any] = {}

        def build(subtaxa: Set[str]) -> Any:
            if len(subtaxa) == 1:
                label = next(iter(subtaxa))
                if label not in name_to_node:
                    from Network import Node  # local import to avoid cycles
                    n = Node(label)
                    net.add_nodes(n)
                    name_to_node[label] = n
                return name_to_node[label]

            # Maximal proper clusters under this subtaxa
            proper = [c for c in clusters if len(c) < len(subtaxa) and c.issubset(subtaxa)]
            # Keep only maximal ones
            maximal: List[Set[str]] = []
            for c in sorted(proper, key=lambda x: len(x), reverse=True):
                if not any(c.issubset(m) for m in maximal):
                    maximal.append(set(c))

            # Fill remainder as singletons
            covered: Set[str] = set().union(*maximal) if maximal else set()
            for x in subtaxa.difference(covered):
                maximal.append({x})

            from Network import Node
            parent = Node(f"Internal_{len(net.V())}")
            net.add_nodes(parent)
            for block in maximal:
                child = build(block)
                net.add_edges(Edge(parent, child))
            return parent

        root = build(set(taxa))
        # Clean spurious degree-1 chains
        net.clean([False, False, True])
        return net

    def gene_concordance_factors(self, ref_tree: Network) -> Dict[Tuple[str, str], float]:
        """
        Compute a split-based concordance factor per internal edge of the reference.

        For each edge (u->v) with split (A | B), count across gene trees the
        fraction of informative trees (with at least one taxon from A and B)
        that contain the induced split on their leaf set. Returns a map keyed by
        (min(child_label), max(child_label)) of the edge child cluster's name
        representative to gCF. If the child is an internal node, the key uses a
        canonical name derived from its cluster (string-joined).
        """
        # Precompute clusters for all gene trees
        tree_clusters: List[Set[FrozenSet[str]]] = []
        for tree in self.trees:
            clus = get_all_clusters(tree, include_trivial=False)
            label_clus = set(frozenset(n.label for n in c) for c in clus)
            tree_clusters.append(label_clus)

        all_taxa = set(leaf.label for leaf in ref_tree.get_leaves())

        results: Dict[Tuple[str, str], float] = {}

        def cluster_key(labels: Set[str]) -> Tuple[str, str]:
            # produce a short key from the smallest and largest label
            if len(labels) == 0:
                return ("", "")
            s = sorted(labels)
            return (s[0], s[-1])

        # Iterate edges of reference
        for e in ref_tree.E():
            if not isinstance(e, Edge):
                continue
            A = set(n.label for n in ref_tree.leaf_descendants(e.dest))
            B = all_taxa.difference(A)
            if len(A) < 2 or len(B) < 2:
                continue

            informative = 0
            support = 0
            for i, tree in enumerate(self.trees):
                taxa_t = set(leaf.label for leaf in tree.get_leaves())
                A_t = frozenset(A.intersection(taxa_t))
                B_t = frozenset(B.intersection(taxa_t))
                if len(A_t) == 0 or len(B_t) == 0:
                    continue
                # Require both sides at least 2 to be informative for an internal edge
                if len(A_t) < 2 and len(B_t) < 2:
                    continue
                informative += 1
                if A_t in tree_clusters[i] or B_t in tree_clusters[i]:
                    support += 1
            if informative > 0:
                results[cluster_key(A)] = support / informative

        return results

    def astral(self,
                astral_jar_path: str,
                mapping_rule: Optional[Callable[[str], str]] = None,
                extra_args: Optional[List[str]] = None) -> Network:
        """
        Infer a species tree using ASTRAL from the stored gene trees.

        Requires a path to the ASTRAL .jar. We write trees to a temp file and a
        multi-allele mapping, then call ASTRAL and parse the result.

        The mapping is resolved in this order:
          1. The instance-level ``species_gene_mapping`` (if set).
          2. The *mapping_rule* argument passed here.
          3. The instance-level ``naming_rule``.
          4. Identity mapping (each gene = its own species).

        Args:
            astral_jar_path (str): Path to the ASTRAL jar file.
            mapping_rule (Callable, optional): Override callable for
                deriving species from gene labels.  Ignored when an
                explicit species_gene_mapping is available.
            extra_args (list[str], optional): Additional CLI args for ASTRAL.
        """
        if extra_args is None:
            extra_args = []

        if mapping_rule is not None and self._species_gene_mapping is None:
            old_rule = self.naming_rule
            self.naming_rule = mapping_rule
            species_map = self._resolve_mapping()
            self.naming_rule = old_rule
        else:
            species_map = self._resolve_mapping()

        with tempfile.TemporaryDirectory() as tmpdir:
            trees_path = os.path.join(tmpdir, "genes.tre")
            map_path = os.path.join(tmpdir, "mapping.txt")
            out_path = os.path.join(tmpdir, "astral_out.tre")

            with open(trees_path, "w") as f:
                for t in self.trees:
                    f.write(t.newick())
                    if not t.newick().strip().endswith(";"):
                        f.write(";")
                    f.write("\n")

            with open(map_path, "w") as f:
                for sp, genes in species_map.items():
                    f.write(f"{sp}: {' '.join(sorted(genes))}\n")

            cmd = [
                "java", "-jar", astral_jar_path,
                "-a", map_path, "-i", trees_path, "-o", out_path,
            ]
            cmd.extend(extra_args)
            subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            with open(out_path, "r") as f:
                newick = f.read().strip()

        handle = StringIO(newick)
        bio_tree = Phylo.read(handle, "newick")
        return self._biopy_tree_to_network(bio_tree)

    def _biopy_tree_to_network(self, tree: Any) -> Network:
        """
        Minimal Biopython tree -> PhyNetPy Network conversion for plain trees.
        """
        # Build parent mapping
        parents = {}
        for clade in tree.find_clades(order="level"):
            for child in clade:
                parents[child] = clade

        net = Network()

        # Ensure nodes exist and set times/edges
        created: Dict[Any, Any] = {}

        def ensure_node(clade, parent_node: Optional[Any]) -> Any:
            if clade in created:
                return created[clade]
            name = clade.name if clade.name is not None else f"Internal_{len(created)}"
            node = Node(name)
            # time
            if parent_node is None:
                node.set_time(0)
            else:
                bl = clade.branch_length if clade.branch_length is not None else 1.0
                node.set_time(parent_node.get_time() + bl)
            net.add_nodes(node)
            created[clade] = node
            return node

        # Create nodes top-down
        roots = [tree.root]
        for cl in roots:
            parent_node = ensure_node(cl, None)
            # BFS children
            q = [cl]
            while q:
                cur = q.pop(0)
                src_node = created[cur]
                for child in cur:
                    child_node = ensure_node(child, src_node)
                    e = Edge(src_node, child_node)
                    bl = child.branch_length if child.branch_length is not None else 1.0
                    e.set_length(bl)
                    net.add_edges(e)
                    q.append(child)
        return net

    def duplication_loss_summary(
        self,
        species_tree: Network,
        naming_rule: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        """
        Reconcile each gene tree against a species tree using LCA mapping and
        report total duplications and losses (parsimony-based estimate).

        The gene-to-species mapping is resolved in the same priority order
        as :meth:`_resolve_mapping`:
          1. Explicit ``species_gene_mapping``.
          2. *naming_rule* argument (if provided and no explicit mapping).
          3. Instance ``naming_rule``.
          4. Identity mapping.

        Returns:
            dict with ``"totals"`` and ``"per_tree"`` breakdowns.
        """
        if naming_rule is not None and self._species_gene_mapping is None:
            old_rule = self.naming_rule
            self.naming_rule = naming_rule
            species_map = self._resolve_mapping()
            self.naming_rule = old_rule
        else:
            species_map = self._resolve_mapping()

        gene_to_species: Dict[str, str] = {}
        for sp, genes in species_map.items():
            for g in genes:
                gene_to_species[g] = sp

        sp_name_to_node: Dict[str, Any] = {
            leaf.label: leaf for leaf in species_tree.get_leaves()
        }

        def lca_of_species(names: Set[str]) -> Any:
            return species_tree.mrca(set(names))

        def is_descendant(u: Any, v: Any) -> bool:
            return v in species_tree.get_subtree_at(u)

        def distance_down(anc: Any, desc: Any) -> int:
            if anc == desc:
                return 0
            from collections import deque
            q = deque([(anc, 0)])
            seen = {anc}
            while q:
                cur, d = q.popleft()
                for ch in species_tree.get_children(cur):
                    if ch in seen:
                        continue
                    if ch == desc:
                        return d + 1
                    seen.add(ch)
                    q.append((ch, d + 1))
            return 0

        totals = {"duplications": 0, "losses": 0}
        details: List[dict[str, int]] = []

        for tree in self.trees:
            leaf_to_species: Dict[Any, str] = {}
            for leaf in tree.get_leaves():
                sp = gene_to_species.get(leaf.label)
                if sp is None or sp not in sp_name_to_node:
                    continue
                leaf_to_species[leaf] = sp

            nodes = tree.V()
            children_map: Dict[Any, List[Any]] = {
                n: tree.get_children(n) for n in nodes
            }
            mapped: Dict[Any, Any] = {}

            def map_node(n: Any) -> Any:
                if n in mapped:
                    return mapped[n]
                if n in leaf_to_species:
                    sp_node = sp_name_to_node[leaf_to_species[n]]
                    mapped[n] = sp_node
                    return sp_node
                desc_species: Set[str] = set()
                def collect_species(x: Any) -> None:
                    if x in leaf_to_species:
                        desc_species.add(leaf_to_species[x])
                    else:
                        for ch in children_map[x]:
                            collect_species(ch)
                collect_species(n)
                if len(desc_species) == 0:
                    sp_node = species_tree.root()
                else:
                    sp_node = lca_of_species(desc_species)
                mapped[n] = sp_node
                return sp_node

            dups = 0
            losses = 0

            for n in nodes:
                ch = children_map[n]
                if len(ch) < 2:
                    continue
                m_n = map_node(n)
                m_c1 = map_node(ch[0])
                m_c2 = map_node(ch[1])
                if m_n == m_c1 or m_n == m_c2:
                    dups += 1
                if is_descendant(m_n, m_c1):
                    d1 = distance_down(m_n, m_c1)
                    losses += max(0, d1 - 1)
                if is_descendant(m_n, m_c2):
                    d2 = distance_down(m_n, m_c2)
                    losses += max(0, d2 - 1)

            totals["duplications"] += dups
            totals["losses"] += losses
            details.append({"duplications": dups, "losses": losses})

        return {"totals": totals, "per_tree": details}