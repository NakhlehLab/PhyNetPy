#!/usr/bin/env python3
"""
Shared helpers for the DEFJ benchmark scripts: dataset paths, ground-truth
networks, gene maps, and a gene-map builder that derives the species ->
subgenome-label mapping directly from gene-tree leaf labels.

DEFJ taxon labels follow the pattern ``<individual><species><homeolog>``,
e.g. ``01aA`` (individual 01, species ``a``, homeolog ``A``) or ``05yB``.
A diploid species has a single homeolog (``A``); a tetraploid has two
(``A``/``B``). The MP-Allop subgenome map sends each species to its list of
subgenome (MUL-tree tip) labels; its length is the species ploidy.

The current MP-Allop / ``AlleleMap`` implementation maps each gene-tree leaf to
a *distinct* subgenome tip, i.e. it assumes **one individual per subgenome**
(n = 1). For n > 1 (multiple individuals) the gene trees carry several alleles
per subgenome, which the bijective allele map cannot place. ``build_gene_map``
therefore builds the canonical one-individual-per-subgenome map (selecting the
lexicographically smallest individual), and exposes the full individual set so
callers can detect / handle multi-individual conditions explicitly.

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import re
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def defj_root(genes: int) -> Path:
    """Root of the DEFJ tree for a given gene-count tier (10 or 100)."""
    return project_root() / "DEFJ" / f"{genes}Genes" / "withOG"


# Ground-truth species networks (extended Newick), species-level labels.
TRUE_NETWORKS = {
    "D": "(((b:0.009,((x:0.006,(y:0.003,z:0.003):0.003):0.003)#H1:0):0.003,(#H1:0,a:0.009):0.003):0.04366667,o:0.10233333);",
    "E": "(o:0.10283333,(((a:0.006,((y:0.003,z:0.003):0.003)#H1:0):0.003,(x:0.009)#H2:0):0.003,(#H2:0,(#H1:0,b:0.006):0.003):0.003):0.04316667);",
    "F": "(o:0.10383333,((((a:0.003,(z:0.003)#H1:0):0.003,(y:0.006)#H2:0):0.003,(x:0.009)#H3:0):0.003,((#H2:0,(#H1:0,b:0.003):0.003):0.003,#H3:0):0.003):0.04216667);",
    # J has THREE reticulations: v (=#H1), the (w,x,y,z) clade (=#H2), and the
    # (t,u) clade (=#H3). An earlier transcription of this network omitted the
    # third reticulation and left t,u as diploids, which contradicts the data:
    # the J gene trees sample t,u,v,w,x,y,z all as tetraploids (homeologs A/B),
    # and the dataset's own true MUL tree (DEFJ/.../J/.../multree.newick) shows
    # all seven species duplicated. This topology was reconstructed from that
    # true MUL tree and verified to reproduce its optimal parsimony score.
    "J": "(o,(((a,b),((c,(d,(v)#H1)),(((z,y),x),w)#H2)),(((#H2,((e,#H1),(t,u)#H3)),#H3),f)));",
}

# Reticulation count of each ground-truth network (used for PhyloNet -maxRetic).
TRUE_RETICULATIONS = {"D": 1, "E": 2, "F": 3, "J": 3}

# Hardcoded n=1 gene maps (kept for cross-validation of build_gene_map).
GENE_MAPS = {
    "D": {
        "a": ["01aA"], "b": ["01bA"], "o": ["01oA"],
        "x": ["01xA", "01xB"], "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
    },
    "E": {
        "a": ["01aA"], "b": ["01bA"], "o": ["01oA"],
        "x": ["01xA", "01xB"], "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
    },
    "F": {
        "a": ["01aA"], "b": ["01bA"], "o": ["01oA"],
        "x": ["01xA", "01xB"], "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
    },
    "J": {
        "o": ["01oA"],
        "a": ["01aA"], "b": ["01bA"], "c": ["01cA"],
        "d": ["01dA"], "e": ["01eA"], "f": ["01fA"],
        "t": ["01tA", "01tB"], "u": ["01uA", "01uB"], "v": ["01vA", "01vB"],
        "w": ["01wA", "01wB"], "x": ["01xA", "01xB"],
        "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
    },
}

# DEFJ parameter grids.
SCENARIOS = ("D", "E", "F", "J")
GENE_COUNTS_10 = (1, 3, 10)
GENE_COUNTS_100 = (100,)
INDIVIDUALS = (1, 3, 9)
ILS_DEF = (4, 20, 100)   # D, E, F
ILS_J = (20,)            # J only has t=20
REPLICATES = tuple(range(1, 11))

_LABEL_RE = re.compile(r"^(?P<indiv>\d+)(?P<species>[a-z]+)(?P<homeolog>[A-Z]+)$")


def parse_label(label: str) -> tuple[str, str, str] | None:
    """Parse a DEFJ leaf label into (individual, species, homeolog)."""
    m = _LABEL_RE.match(label)
    if m is None:
        return None
    return m.group("indiv"), m.group("species"), m.group("homeolog")


def gene_tree_files(scenario: str, genes: int, g: int, n: int, t: int, r: int) -> Path:
    """Path to a DEFJ gene-tree Newick file."""
    fname = f"{scenario}2GTg{g}n{n}t{t}r{r}-g_trees.newick"
    return defj_root(genes) / scenario / f"g{g}" / f"n{n}" / f"t{t}" / f"r{r}" / fname


def read_leaf_labels(path: Path) -> set[str]:
    """Collect all leaf labels appearing in a Newick gene-tree file."""
    labels: set[str] = set()
    token = re.compile(r"[0-9A-Za-z_]+")
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("("):
                continue
            for tok in token.findall(line):
                if _LABEL_RE.match(tok):
                    labels.add(tok)
    return labels


def build_gene_map(labels: set[str]) -> tuple[dict[str, list[str]], dict]:
    """
    Build the canonical (one-individual-per-subgenome) MP-Allop gene map from a
    set of leaf labels.

    Returns:
        (gene_map, info) where
          gene_map: species -> sorted list of "<minIndiv><species><homeolog>"
                    labels (length == ploidy of that species).
          info: {
            "individuals": sorted set of individual prefixes seen,
            "homeologs_by_species": {species: sorted homeologs},
            "multi_individual": bool (True if >1 individual prefix present),
            "unparsed": sorted list of labels that did not match the scheme,
          }
    """
    species_homeologs: dict[str, set[str]] = {}
    individuals: set[str] = set()
    unparsed: list[str] = []

    for label in labels:
        parsed = parse_label(label)
        if parsed is None:
            unparsed.append(label)
            continue
        indiv, species, homeolog = parsed
        individuals.add(indiv)
        species_homeologs.setdefault(species, set()).add(homeolog)

    min_indiv = min(individuals) if individuals else "01"

    gene_map: dict[str, list[str]] = {}
    for species, homeologs in species_homeologs.items():
        gene_map[species] = [
            f"{min_indiv}{species}{h}" for h in sorted(homeologs)
        ]

    info = {
        "individuals": sorted(individuals),
        "homeologs_by_species": {s: sorted(h) for s, h in species_homeologs.items()},
        "multi_individual": len(individuals) > 1,
        "unparsed": sorted(unparsed),
    }
    return gene_map, info


def canonical_label_set(gene_map: dict[str, list[str]]) -> set[str]:
    """All subgenome (MUL-tip) labels in a gene map."""
    return {lab for labels in gene_map.values() for lab in labels}


def collapse_to_canonical(net, keep_labels: set[str]):
    """
    Prune a gene tree to the canonical one-individual-per-subgenome leaf set.

    Removes every leaf whose label is not in ``keep_labels`` and suppresses the
    resulting degree-2 internal nodes via ``Network.clean``. Mutates and returns
    ``net``.
    """
    root = net.root()
    # Iteratively strip non-kept nodes that have no children. Removing a
    # subgenome's extra individuals can leave an internal node with all
    # children gone (a dead-end), which must itself be removed; repeat until
    # the only childless nodes are kept leaves.
    changed = True
    while changed:
        changed = False
        dead = [v for v in list(net.V())
                if v is not root and net.out_degree(v) == 0
                and v.label not in keep_labels]
        for v in dead:
            for edge in list(net.in_edges(v)):
                net.remove_edge(edge)
            net.remove_nodes(v)
            changed = True
    # Suppress degree-2 chains and floaters left behind.
    net.clean([True, True, True])
    return net


def load_gene_trees(scenario: str, genes: int, g: int, n: int, t: int, r: int,
                    gene_map: dict[str, list[str]] | None = None,
                    collapse: bool = True):
    """
    Read DEFJ gene trees for a condition. When ``collapse`` is True and the
    condition has multiple individuals (n > 1), each tree is pruned to one
    individual per subgenome (the canonical gene-map labels), giving inputs
    that MP-Allop's bijective allele map can score. Requires phynetpy on the
    path (import is local to keep this module import-cheap).
    """
    from phynetpy.IO import read_newick

    path = gene_tree_files(scenario, genes, g, n, t, r)
    if not path.exists():
        raise FileNotFoundError(f"Gene tree file not found: {path}")

    keep = canonical_label_set(gene_map) if (collapse and gene_map) else None
    trees = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("("):
                continue
            net = read_newick(line)
            if keep is not None:
                collapse_to_canonical(net, keep)
            trees.append(net)
    return trees


def _self_test() -> int:
    """Validate build_gene_map against the hardcoded n=1 GENE_MAPS."""
    failures = 0
    for genes in (10,):
        for scenario in SCENARIOS:
            path = gene_tree_files(scenario, genes, g=1, n=1, t=20 if scenario == "J" else 4, r=1)
            if not path.exists():
                print(f"  SKIP {scenario}: {path} missing")
                continue
            labels = read_leaf_labels(path)
            built, info = build_gene_map(labels)
            expected = GENE_MAPS[scenario]
            built_norm = {k: sorted(v) for k, v in built.items()}
            exp_norm = {k: sorted(v) for k, v in expected.items()}
            ok = built_norm == exp_norm and not info["multi_individual"]
            print(f"  {scenario} n=1: {'OK' if ok else 'MISMATCH'} "
                  f"(species={len(built)}, indivs={info['individuals']})")
            if not ok:
                failures += 1
                print(f"     built={built_norm}")
                print(f"     expected={exp_norm}")

    # Spot-check an n=3 condition: same species/homeolog structure, 3 individuals.
    p3 = gene_tree_files("D", 10, g=1, n=3, t=4, r=1)
    if p3.exists():
        labels = read_leaf_labels(p3)
        built, info = build_gene_map(labels)
        same_struct = (
            {k: sorted(v) for k, v in built.items()}
            == {k: sorted(v) for k, v in GENE_MAPS["D"].items()}
        )
        print(f"  D n=3: individuals={info['individuals']} "
              f"multi={info['multi_individual']} "
              f"canonical-map-matches-n1={same_struct}")
        if not info["multi_individual"]:
            failures += 1
            print("     ERROR: n=3 not detected as multi-individual")

        # Collapsing n=3 trees must reproduce the n=1 leaf set exactly.
        try:
            gmap, _ = build_gene_map(read_leaf_labels(p3))
            collapsed = load_gene_trees("D", 10, g=1, n=3, t=4, r=1,
                                        gene_map=gmap, collapse=True)
            n1_labels = canonical_label_set(GENE_MAPS["D"])
            ok = all(
                {lf.label for lf in tr.get_leaves()} == n1_labels
                for tr in collapsed
            )
            print(f"  D n=3 collapse -> canonical leaf set: "
                  f"{'OK' if ok else 'MISMATCH'} "
                  f"({len(collapsed)} trees, "
                  f"{len(collapsed[0].get_leaves())} leaves each)")
            if not ok:
                failures += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  D n=3 collapse: ERROR {exc!r}")
            failures += 1

    return failures


if __name__ == "__main__":
    import sys
    print("Validating build_gene_map against hardcoded n=1 GENE_MAPS:")
    sys.exit(1 if _self_test() else 0)
