r"""#3 -- Measure the reticulation signal strength of the weekend data set.

Two independent signal layers determine whether MCMC_SEQ can recover the
reticulation:

1. Coalescent signal (loci level): the hybrid H (parent of the (L8,L9) cherry)
   inherits from the G1/major side (leaves L0-L3, gamma=0.70) or the G2/minor
   side (leaves L4-L7, gamma=0.30).  A locus "shows" the minor history when its
   gene tree joins (L8,L9) to the G2 side.  Expected minor fraction ~ 0.30.
   This is the fundamental information about the reticulation; more loci ->
   stronger, more resolvable signal.

2. Sequence signal (site level): whether 1000 sites at theta=0.02 carry enough
   variation to *recover* each gene tree at all.  If sequences are near-
   invariant the coalescent signal is unobservable regardless of loci count.

We measure both on the TRUE gene trees (layer 1, an oracle) and on UPGMA gene
trees rebuilt from the sequences (layer 1 as actually seen through layer 2),
across n_loci in {25, 50, 128}.  The gap between them is how much signal the
sequences lose.
"""
from __future__ import annotations

import os
import sys
import math
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from phynetpy.infer import JC69, simulate_multilocus
from phynetpy.Network import Network, Node, Edge
from _run_weekend import build_true_network, MAPPING, _TRUE_HEIGHTS


def build_major_tree() -> Network:
    """No-reticulation NULL: the major displayed tree (drop the minor edge).

    H descends only from P1, so (L8,L9) is sister to the G1 side {L0..L3}.
    Any 'minor' classification under this tree is pure ILS -- the null the
    reticulation signal must beat to be detectable.
    """
    edges = [
        ("R", "P1", None), ("R", "P2", None),
        ("P1", "G1", None), ("P1", "H", None),  # H now a plain child of P1
        ("P2", "G2", None),
        ("G1", "AB", None), ("G1", "CD", None),
        ("G2", "EF", None), ("G2", "GH", None),
        ("H", "IJ", None),
        ("AB", "L0", None), ("AB", "L1", None),
        ("CD", "L2", None), ("CD", "L3", None),
        ("EF", "L4", None), ("EF", "L5", None),
        ("GH", "L6", None), ("GH", "L7", None),
        ("IJ", "L8", None), ("IJ", "L9", None),
    ]
    nodes = {n: Node(n) for n in _TRUE_HEIGHTS}
    net = Network()
    net.add_nodes(*nodes.values())
    net.add_edges([Edge(nodes[p], nodes[c], length=_TRUE_HEIGHTS[p] - _TRUE_HEIGHTS[c])
                   for p, c, _ in edges])
    return net

G1_SIDE = {"L0", "L1", "L2", "L3"}   # major parent (gamma 0.70)
G2_SIDE = {"L4", "L5", "L6", "L7"}   # minor parent (gamma 0.30)
HYBRID = {"L8", "L9"}                 # the (L8,L9) cherry below H


# ---- gene-tree geometry helpers (work on ultrametric rooted binary trees) ----

def _root(net):
    return net.root() if not isinstance(net.root(), list) else net.root()[0]


def _leaf_labels(net, v):
    """Leaf-descendant label set of node v (memo-free, small trees)."""
    kids = net.get_children(v)
    if not kids:
        return {v.label}
    out = set()
    for c in kids:
        out |= _leaf_labels(net, c)
    return out


def _node_heights(net):
    """height(leaf)=0; internal = child-edge-length + child height (ultrametric)."""
    h = {}

    def rec(v):
        kids = net.get_children(v)
        if not kids:
            h[v] = 0.0
            return 0.0
        hv = None
        for c in kids:  # recurse into ALL children so every node gets a height
            e = net.get_edge(v, c)
            e = e[0] if isinstance(e, list) else e
            ch = float(e.get_length()) + rec(c)
            if hv is None:
                hv = ch
        h[v] = hv
        return hv

    r = _root(net)
    r = r[0] if isinstance(r, list) else r
    rec(r)
    return h


def _mrca_height(net, labels, heights, leafcache):
    """Height of the MRCA of ``labels`` = smallest clade node covering them."""
    best = None
    best_h = math.inf
    for v, ls in leafcache.items():
        if labels <= ls and heights[v] < best_h:
            best_h = heights[v]
            best = v
    return best_h


def _classify(net):
    """Return 'minor' if (L8,L9) coalesces with the G2 side before the G1 side."""
    r = _root(net)
    r = r[0] if isinstance(r, list) else r
    heights = _node_heights(net)
    leafcache = {v: _leaf_labels(net, v) for v in net.V() if net.get_children(v)}
    h_major = _mrca_height(net, HYBRID | G1_SIDE, heights, leafcache)
    h_minor = _mrca_height(net, HYBRID | G2_SIDE, heights, leafcache)
    return "minor" if h_minor < h_major else "major"


# ---- sequence signal --------------------------------------------------------

def _site_stats(aln):
    """Return (%variable, %parsimony-informative) over the alignment columns."""
    labels = list(aln)
    seqs = [aln[l] for l in labels]
    n = len(seqs[0])
    var = pi = 0
    for j in range(n):
        col = [s[j] for s in seqs]
        counts = Counter(c for c in col if c in "ACGT")
        if len(counts) > 1:
            var += 1
        if sum(1 for c in counts.values() if c >= 2) >= 2:
            pi += 1
    return 100.0 * var / n, 100.0 * pi / n


def main() -> None:
    from phynetpy._mcmc_seq import build_upgma_gene_tree

    true_net = build_true_network()
    null_net = build_major_tree()
    print("Reticulation-signal measurement (weekend 10-taxon net, 1000 sites)\n")
    print("  minor-history expected fraction (1 - gamma_major) = 0.30")
    print("  NULL = major displayed tree (no reticulation); its minor rate is\n"
          "         pure ILS -- the reticulation signal must beat it.\n")

    for n_loci in (25, 50, 128, 256):
        data = simulate_multilocus(true_net, MAPPING, n_loci=n_loci,
                                   seq_length=1000, theta=0.02, model=JC69(),
                                   seed=2024)
        null = simulate_multilocus(null_net, MAPPING, n_loci=n_loci,
                                   seq_length=1000, theta=0.02, model=JC69(),
                                   seed=99)
        tn_minor = Counter(_classify(gt) for gt in data.gene_trees)["minor"]
        null_minor = Counter(_classify(gt) for gt in null.gene_trees)["minor"]
        upgma = [build_upgma_gene_tree(aln) for aln in data.loci]
        sn_minor = Counter(_classify(gt) for gt in upgma)["minor"]
        stats = [_site_stats(aln) for aln in data.loci]
        var = np.array([s[0] for s in stats])
        pi = np.array([s[1] for s in stats])

        p_true = tn_minor / n_loci
        p_null = null_minor / n_loci
        # z for difference of two proportions (reticulation vs ILS-only null)
        pool = (tn_minor + null_minor) / (2 * n_loci)
        se_diff = math.sqrt(max(pool * (1 - pool) * 2 / n_loci, 1e-12))
        z = (p_true - p_null) / se_diff

        print(f"n_loci = {n_loci}")
        print(f"  sequence signal   : %var mean={var.mean():5.2f}  "
              f"%PI mean={pi.mean():5.2f}")
        print(f"  reticulation data : minor={tn_minor:3d}/{n_loci}  "
              f"frac={p_true:.3f}")
        print(f"  NULL (ILS only)   : minor={null_minor:3d}/{n_loci}  "
              f"frac={p_null:.3f}")
        print(f"  signal vs null    : diff={p_true - p_null:+.3f}  z={z:+.2f}  "
              f"{'DETECTABLE' if z > 2 else 'not distinguishable'}")
        print(f"  UPGMA(seq) minor  : {sn_minor:3d}/{n_loci} "
              f"(seq preserves call {sum(1 for a, b in zip(data.gene_trees, upgma) if _classify(a) == _classify(b))}/{n_loci})\n")


if __name__ == "__main__":
    main()
