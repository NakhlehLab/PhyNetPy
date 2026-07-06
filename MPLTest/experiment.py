"""
Apples-to-apples MPL experiment, in the shape of mpl-score-camus.pdf, on the
CAMUS Wolbachia example dataset (21 strains + 1 outgroup, 1123 gene trees;
shipped in the CAMUS repo at internal/infer/testdata).

Mirrors the PDF's two halves, with PhyNetPy's MPL on BOTH sides:

  (1) SCORING  -- analog of PhyloNet's `CalGTProb ... -pseudo -o`:
      score each CAMUS output network's log-pseudo-likelihood after optimising
      its branch lengths + gammas on the fixed topology (CAMUS networks ship
      as topology-only extended Newick).

  (2) SEARCH   -- analog of `InferNetwork_MPL`:
      run PhyNetPy's own MPL hill-climb from the SAME constraint tree, both
      with the backbone fixed (fix_st=True == PhyloNet's -fs / "FT") and free,
      at 1 and 2 reticulations, and time it.

Then compare, at matched reticulation counts, the MPL score our search finds
vs the MPL score of the CAMUS network -- the question the PDF asks of
PhyloNet-MPL(FT) vs CAMUS.

Branch-length/gamma optimisation is bounded (max_rounds/branch_iters) so the
whole run is ~10 min; a single raw score is ~20 ms, the optimiser is the cost.

NOTE: gene trees + constraint tree carry numeric bootstrap-support labels on
internal nodes; PhyNetPy's from_newick dedupes nodes by label and silently
MERGES distinct internal nodes sharing a support value (corrupting topology,
hanging the LCA index).  We strip numeric internal labels before parsing.
"""
from __future__ import annotations

import argparse
import contextlib
import io as _io
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from phynetpy.Network import Network
from phynetpy.GeneTrees import GeneTrees
from phynetpy.MPL import MPLScorer, compute_gene_tree_triplets, _HAS_CYTHON_MPL
from phynetpy.ModelGraph import Model
from phynetpy._optimize import optimize_network_parameters
from phynetpy._mcmc_gt import _populate_default_branch_lengths
from phynetpy.infer import InferNetwork_ML

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "camus_testdata")

_SUPPORT = re.compile(r"\)(\d+(?:\.\d+)?)(?=[:,);])")
clean = lambda s: _SUPPORT.sub(")", s)

# Bounded optimisation (keeps each scored topology ~30s instead of ~6 min).
OPT = dict(max_rounds=8, branch_iters=8)


def read_lines(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(_io.StringIO()):
        yield


def n_retic(net):
    return sum(1 for v in net.V() if v.is_reticulation())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Cython MPL backend active: {_HAS_CYTHON_MPL}", flush=True)

    gt_lines = read_lines(os.path.join(DATA, "gene-trees.nwk"))
    trees = [Network.from_newick(clean(ln)) for ln in gt_lines]
    taxa = sorted(n.label for n in trees[0].get_leaves())
    mapping = {t: [t] for t in taxa}
    gts = GeneTrees(gene_tree_list=trees)
    gts.species_gene_mapping = mapping
    print(f"taxa: {len(taxa)}  gene trees: {len(trees)}", flush=True)

    t0 = time.time()
    pre = compute_gene_tree_triplets(
        gene_trees=gts, mapping=mapping, species_labels=taxa)
    rho = pre.rho_by_triplet
    active = [t for t in pre.triplets if any(rho[t][i] > 0.0 for i in range(3))]
    print(f"rho precompute: {time.time()-t0:.2f}s "
          f"({len(active)} active triplets)", flush=True)

    def score_optimized(net):
        _populate_default_branch_lengths(net)
        sc = MPLScorer(rho, active)
        m = Model(rng=np.random.default_rng(0))
        m.network = net
        m.set_likelihood_calculator(sc)
        return optimize_network_parameters(m, sc, mapping, scope="all", **OPT)

    def fresh_constraint():
        return Network.from_newick(
            clean(read_lines(os.path.join(DATA, "constraint.nwk"))[0]))

    # ── (0) Constraint tree (r=0) baseline ────────────────────────────
    t0 = time.time()
    s0 = score_optimized(fresh_constraint())
    print(f"\n[baseline] constraint tree r=0: optimised logPL = "
          f"{s0:.2f}  ({time.time()-t0:.1f}s)", flush=True)

    # ── (1) SCORE the CAMUS networks ──────────────────────────────────
    print("\n=== (1) Scoring CAMUS networks (optimised log-PL) ===", flush=True)
    camus_files = {
        "pipeline": ("network.nwk", 3),         # score r=1,2,3
        "q2_t05_max": ("net_q2_t05_max.nwk", 2),
        "q2_t05_norm": ("net_q2_t05_norm.nwk", 2),
        "q2_t05_sym_a01": ("net_q2_t05_sym_a01.nwk", 2),
    }
    camus_scores = {}
    for tag, (fn, rmax) in camus_files.items():
        for ln in read_lines(os.path.join(DATA, fn)):
            net = Network.from_newick(ln)
            r = n_retic(net)
            if r > rmax:
                continue
            t0 = time.time()
            s = score_optimized(net)
            dt = time.time() - t0
            camus_scores[(tag, r)] = (s, dt)
            print(f"  CAMUS[{tag}] r={r}: logPL={s:.2f}  ({dt:.1f}s)", flush=True)

    # ── (2) SEARCH with our MPL (InferNetwork_ML pseudo path) ─────────
    print("\n=== (2) PhyNetPy MPL search from the constraint tree ===", flush=True)
    search_results = {}
    for fix_st in (True, False):
        mode = "FT" if fix_st else "free"
        for maxret in (1, 2):
            inf = InferNetwork_ML(fresh_constraint(), gts, mapping,
                                  max_reticulations=maxret)
            t0 = time.time()
            with quiet():
                res = inf.search(
                    num_runs=1, num_iter=args.iters, max_failures=60,
                    pseudo=True, fix_st=fix_st,
                    optimize_params=True, optimize_scope="gamma",
                    final_optimize=True, seed=args.seed, **OPT)
            dt = time.time() - t0
            search_results[(mode, maxret)] = (
                res.best_log_likelihood, res.num_reticulations,
                res.num_networks_examined, dt)
            print(f"  search {mode}, maxret={maxret}: "
                  f"logPL={res.best_log_likelihood:.2f}  "
                  f"retic={res.num_reticulations}  "
                  f"examined={res.num_networks_examined}  ({dt:.1f}s)",
                  flush=True)

    # ── (3) Head-to-head summary ──────────────────────────────────────
    print("\n=== (3) Head-to-head: PhyNetPy MPL search vs CAMUS net (matched r) ===")
    print(f"{'r':>3} {'CAMUS(pipeline)':>18} {'ours FT':>14} {'ours free':>14}")
    for r in (1, 2):
        c = camus_scores.get(("pipeline", r), (float('nan'),))[0]
        ft = search_results.get(("FT", r), (float('nan'),))[0]
        fr = search_results.get(("free", r), (float('nan'),))[0]
        print(f"{r:>3} {c:>18.2f} {ft:>14.2f} {fr:>14.2f}")
    best_camus = max(camus_scores.items(), key=lambda kv: kv[1][0])
    print(f"\nConstraint (r=0) baseline : logPL={s0:.2f}")
    print(f"Best CAMUS network overall: {best_camus[0][0]} r={best_camus[0][1]} "
          f"logPL={best_camus[1][0]:.2f}")
    print("(higher / closer to 0 = better)")


if __name__ == "__main__":
    main()
