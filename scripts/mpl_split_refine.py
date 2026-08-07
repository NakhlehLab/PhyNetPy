#!/usr/bin/env python3
"""
Split an inferred species network at the root into left/right leaf sets,
run MPL+SA on each subproblem, stitch the refined halves, and score on full data.

Taxa partition uses the **dominant tree** (majority reticulation edges) so the
root bipartition is disjoint (reticulate networks can otherwise overlap sides).

**Limitation:** MPL sums triplets over *all* species triples; optimizing left and
right halves separately ignores cross-partition triplets, so the stitched
network often scores poorly on full data unless followed by a **global** refine
phase (e.g. SA on the stitched network).

Usage (from repo root):
  python3 scripts/mpl_split_refine.py [--inferred runs/mpl_50k_hot_linear_best.nwk]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

# Repo root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phynetpy.GeneTrees import GeneTrees
from phynetpy.GraphUtils import dominant_tree, induced_subnetwork_by_taxa
from phynetpy.IO import convert_newick
from phynetpy._mpl import MPL
from phynetpy.infer import format_mpl_reference_comparison
from phynetpy.Network import Network


def _bipartition_leaves_at_root(net: Network) -> tuple[list[str], list[str]]:
    """Disjoint leaf label lists from the two principal sides below the root.

    Reticulate networks can assign the same leaf to multiple root sides; we
    therefore take the **dominant tree** (highest-gamma edge per retic) first
    so the root split is a standard tree bipartition.
    """
    try:
        net = dominant_tree(net)
    except Exception:
        pass
    r = net.root()
    while net.out_degree(r) == 1:
        kids = net.get_children(r)
        if not kids:
            raise ValueError("Degenerate network at root")
        r = kids[0]
    ch = net.get_children(r)
    if len(ch) < 2:
        raise ValueError("Need at least two child subtrees below root to split")

    if len(ch) == 2:
        a, b = ch[0], ch[1]
        left = sorted(x.label for x in net.leaf_descendants(a))
        right = sorted(x.label for x in net.leaf_descendants(b))
    else:
        left = sorted(x.label for x in net.leaf_descendants(ch[0]))
        rest: set = set()
        for c in ch[1:]:
            rest |= net.leaf_descendants(c)
        right = sorted(x.label for x in rest)

    if set(left) & set(right):
        raise ValueError("Left/right leaf sets overlap; cannot use simple root split")
    return left, right


def _prune_gene_trees_to_taxa(trees: list[Network], taxa: set[str]) -> list[Network]:
    """Return induced trees on taxa present in each gene tree (min 3 leaves for triplets)."""
    out: list[Network] = []
    for t in trees:
        present = [x for x in taxa if t.has_node_named(x) is not None]
        if len(present) < 3:
            continue
        out.append(induced_subnetwork_by_taxa(t, sorted(present)))
    return out


def _mapping_for_taxa(taxa: list[str]) -> dict[str, list[str]]:
    return {t: [t] for t in taxa}


def _stitch(left: Network, right: Network) -> Network:
    l_nwk = left.newick().rstrip(";").strip()
    r_nwk = right.newick().rstrip(";").strip()
    return Network.from_newick(f"({l_nwk},{r_nwk})Root;")


def main() -> None:
    here = os.path.dirname(__file__)
    root = os.path.join(here, "..")
    testfiles = os.path.join(root, "tests", "testfiles")

    ap = argparse.ArgumentParser(description="Split-refine-stitch MPL experiment")
    ap.add_argument(
        "--inferred",
        default=os.path.join(root, "runs", "mpl_50k_hot_linear_best.nwk"),
        help="Path to inferred network Newick (default: runs/mpl_50k_hot_linear_best.nwk)",
    )
    ap.add_argument("--iters", type=int, default=10000, help="SA iterations per side")
    ap.add_argument("--t-start", type=float, default=50000.0)
    ap.add_argument("--t-end", type=float, default=1.0)
    ap.add_argument("--seed-left", type=int, default=42)
    ap.add_argument("--seed-right", type=int, default=43)
    ap.add_argument(
        "--max-reticulations",
        type=int,
        default=1,
        help="Per subproblem; stitched network has at most sum of both sides (default 1)",
    )
    args = ap.parse_args()

    species_taxa = sorted(
        [
            "t1",
            "t4",
            "t15",
            "t36",
            "t38",
            "t43",
            "t49",
            "t52",
            "t74",
            "t83",
            "t84",
            "t85",
            "t94",
            "t109",
            "t111",
            "t123",
            "t124",
            "t130",
        ]
    )
    full_mapping = _mapping_for_taxa(species_taxa)

    gt_path = os.path.join(testfiles, "mpl_20taxa_gt.txt")
    true_path = os.path.join(testfiles, "mpl_20taxa.txt")

    trees: list[Network] = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if line:
                trees.append(Network.from_newick(line))

    with open(true_path) as f:
        raw = f.readline().strip()
    true_net = Network.from_newick(convert_newick(raw, standard="PhyNetPy"))

    gts_full = GeneTrees(gene_tree_list=list(trees))
    gts_full.species_gene_mapping = full_mapping

    inferred_path = args.inferred
    if not os.path.isfile(inferred_path):
        print(f"Inferred network not found: {inferred_path}", flush=True)
        print("Run a full MPL search first or pass --inferred PATH", flush=True)
        sys.exit(1)

    with open(inferred_path) as f:
        inferred = Network.from_newick(f.read().strip())

    left_taxa, right_taxa = _bipartition_leaves_at_root(inferred)
    print(f"Root split: |left|={len(left_taxa)}, |right|={len(right_taxa)}", flush=True)
    print(f"  Left:  {left_taxa}", flush=True)
    print(f"  Right: {right_taxa}", flush=True)

    init_left = induced_subnetwork_by_taxa(inferred, left_taxa)
    init_right = induced_subnetwork_by_taxa(inferred, right_taxa)

    gt_left = _prune_gene_trees_to_taxa(trees, set(left_taxa))
    gt_right = _prune_gene_trees_to_taxa(trees, set(right_taxa))
    print(f"Gene trees: left {len(gt_left)} / right {len(gt_right)} (with ≥3 taxa)", flush=True)

    map_l = _mapping_for_taxa(left_taxa)
    map_r = _mapping_for_taxa(right_taxa)

    gts_l = GeneTrees(gene_tree_list=gt_left)
    gts_l.species_gene_mapping = map_l
    gts_r = GeneTrees(gene_tree_list=gt_right)
    gts_r.species_gene_mapping = map_r

    mpl_l = MPL(copy.deepcopy(init_left), gts_l, map_l)
    mpl_r = MPL(copy.deepcopy(init_right), gts_r, map_r)

    print(f"\n--- Left SA {args.iters} iters ---", flush=True)
    best_l = mpl_l.search(
        method="sa",
        num_iter=args.iters,
        max_reticulations=args.max_reticulations,
        t_start=args.t_start,
        t_end=args.t_end,
        seed=args.seed_left,
        plateau_frac=0.0,
    )
    net_left = mpl_l.net
    print(f"Left best score: {best_l:.4f}", flush=True)

    print(f"\n--- Right SA {args.iters} iters ---", flush=True)
    best_r = mpl_r.search(
        method="sa",
        num_iter=args.iters,
        max_reticulations=args.max_reticulations,
        t_start=args.t_start,
        t_end=args.t_end,
        seed=args.seed_right,
        plateau_frac=0.0,
    )
    net_right = mpl_r.net
    print(f"Right best score: {best_r:.4f}", flush=True)

    stitched = _stitch(net_left, net_right)
    out_nwk = os.path.join(root, "runs", "mpl_split_refine_stitched.nwk")
    os.makedirs(os.path.dirname(out_nwk), exist_ok=True)
    with open(out_nwk, "w", encoding="utf-8") as f:
        f.write(stitched.newick() + "\n")
    print(f"\nWrote stitched network: {out_nwk}", flush=True)

    mpl_full = MPL(stitched, gts_full, full_mapping)
    full_score = mpl_full.score()
    true_mpl = MPL(true_net, gts_full, full_mapping)
    true_score = true_mpl.score()

    print(f"\n=== Full-data scores (same rho as full 18-taxa run) ===", flush=True)
    print(f"Stitched network: {full_score:.4f}", flush=True)
    print(f"True network:     {true_score:.4f}", flush=True)
    print(f"Gap (true - stitched): {true_score - full_score:.4f}", flush=True)

    rep = os.path.join(root, "runs", "mpl_split_refine_vs_true.txt")
    report = format_mpl_reference_comparison(
        stitched,
        true_net,
        mpl_full._rho,
        mpl_full._active_triplets,
    )
    with open(rep, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"Comparison report: {rep}", flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main()
