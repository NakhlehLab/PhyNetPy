"""
Benchmark harness for the MPL per-topology optimisation work (levers 1-3).

Compares the legacy raw-score Hill-Climbing search against the new
``optimize_params`` mode (per-topology gamma/branch optimisation with lazy
gating) on the r=1 dataset shipped with the test suite, and prints
log-pseudo-likelihood, topological distance to a reference network, and
wall-clock time for each configuration.

Usage
-----
    python scripts/bench_mpl_optimize.py [--iters N] [--reps R] [--seed S]

The harness is intentionally a plain script (not a pytest test) so it can be
run repeatedly while tuning the optimiser; it makes no assertions, it just
reports the accuracy/runtime trade-off so we can watch each lever land.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phynetpy.Network import Network
from phynetpy.GeneTrees import GeneTrees
from phynetpy._mpl import MPL
from phynetpy.IO import convert_newick
from phynetpy.GraphUtils import mu_distance, hardwired_cluster_distance


TESTFILES = os.path.join(os.path.dirname(__file__), "..", "tests", "testfiles")
TAXA = ["t14", "t15", "t49", "t68", "t69", "t72", "t75", "t91", "t114", "t133"]
MAPPING = {t: [t] for t in TAXA}

# A balanced starting species tree over the 10-taxon set (same seed topology
# the search proof-of-concept uses).
START_NEWICK = (
    "((((t14:1,t15:1):1,(t49:1,t68:1):1):1,"
    "((t69:1,t72:1):1,(t75:1,t91:1):1):1):1,"
    "(t114:1,t133:1):1);"
)

# Highest-scoring known r=1 network for this dataset (from test_mpl_5nets.py);
# used as the reference topology for distance-to-truth.
REFERENCE_R1_NEWICK = (
    "(((((t14:1.0,t75:1.0):0.8767426254184691,(t69:1.0,t114:1.0):2.0170864999696456)"
    ":5.213391915869064,(t91:1.0)#H1:1.0::0.3011071643186969):0.8237808968694593,"
    "t133:1.0):2.054746072134383,(((t15:1.0,#H1:1.0::0.6988928356813031):1.481149764930965,"
    "(t72:1.0,t49:1.0):3.0976006419526203):1.2243080352496307,t68:1.0):0.12981149034576273);"
)


def _load_gene_trees() -> GeneTrees:
    gt_path = os.path.join(TESTFILES, "subgeneset_3_ret1.txt")
    trees = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if line:
                trees.append(Network.from_newick(line))
    gts = GeneTrees(gene_tree_list=trees)
    gts.species_gene_mapping = MAPPING
    return gts


def _parse_phylonet(nwk: str) -> Network:
    return Network.from_newick(convert_newick(nwk, standard="PhyNetPy"))


def _distance(found: Network, reference: Network) -> tuple[float, float]:
    """Return (mu_distance, hardwired_cluster_distance), guarding failures."""
    try:
        mu = float(mu_distance(found, reference))
    except Exception as exc:  # pragma: no cover - diagnostic only
        mu = float("nan")
        print(f"    [mu_distance error: {type(exc).__name__}: {exc}]")
    try:
        hw = float(hardwired_cluster_distance(found, reference))
    except Exception as exc:  # pragma: no cover - diagnostic only
        hw = float("nan")
        print(f"    [hardwired_cluster_distance error: {type(exc).__name__}: {exc}]")
    return mu, hw


def _run_config(
    label: str,
    gts: GeneTrees,
    reference: Network,
    *,
    num_iter: int,
    max_reticulations: int,
    preset: str,
    optimize_band: float,
    seed: int,
) -> dict:
    start_net = Network.from_newick(START_NEWICK)
    mpl = MPL(start_net, gts, MAPPING)

    t0 = time.time()
    score = mpl.search(
        method="hc",
        num_iter=num_iter,
        max_reticulations=max_reticulations,
        preset=preset,
        optimize_band=optimize_band,
        seed=seed,
        print_comparison=False,
    )
    elapsed = time.time() - t0

    found = mpl.net
    n_ret = sum(1 for v in found.V() if v.is_reticulation())
    mu, hw = _distance(found, reference)
    return {
        "label": label,
        "score": score,
        "elapsed": elapsed,
        "n_ret": n_ret,
        "mu": mu,
        "hw": hw,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=400, help="HC iterations per run")
    ap.add_argument("--reps", type=int, default=3, help="seeded repetitions per config")
    ap.add_argument("--seed", type=int, default=0, help="base RNG seed")
    ap.add_argument("--band", type=float, default=5.0, help="lazy-gating band")
    ap.add_argument("--retic-scope", action="store_true",
                    help="also run the optimize-gamma+retic config (slower)")
    ap.add_argument("--scope-all", action="store_true",
                    help="also run the (slowest) optimize-all-params config")
    args = ap.parse_args()

    print("Loading gene trees (r=1 dataset) ...")
    t0 = time.time()
    gts = _load_gene_trees()
    print(f"  {len(gts.trees)} gene trees in {time.time() - t0:.1f}s")
    reference = _parse_phylonet(REFERENCE_R1_NEWICK)

    configs = [
        dict(label="preset=fast", max_reticulations=1, preset="fast"),
        dict(label="preset=default", max_reticulations=1, preset="default"),
    ]
    if args.retic_scope:
        configs.append(
            dict(label="preset=accurate", max_reticulations=1,
                 preset="accurate"),
        )
    if args.scope_all:
        configs.append(
            dict(label="preset=phylonet", max_reticulations=1,
                 preset="phylonet"),
        )

    results: dict[str, list[dict]] = {}
    for cfg in configs:
        runs = []
        for rep in range(args.reps):
            seed = args.seed + rep
            print(
                f"\n=== {cfg['label']} | rep {rep + 1}/{args.reps} "
                f"(seed={seed}, iters={args.iters}) ==="
            )
            res = _run_config(
                cfg["label"], gts, reference,
                num_iter=args.iters,
                max_reticulations=cfg["max_reticulations"],
                preset=cfg["preset"],
                optimize_band=args.band,
                seed=seed,
            )
            runs.append(res)
            print(
                f"    logPL={res['score']:.4f}  retics={res['n_ret']}  "
                f"mu_d={res['mu']:.3g}  hw_d={res['hw']:.3g}  "
                f"time={res['elapsed']:.1f}s"
            )
        results[cfg["label"]] = runs

    # Summary table.
    print("\n" + "=" * 92)
    print(f"{'config':<24} {'logPL(best)':>14} {'logPL(med)':>14} "
          f"{'mu_d(med)':>10} {'hw_d(med)':>10} {'time(med,s)':>12}")
    print("-" * 92)
    for label, runs in results.items():
        best_ll = max(r["score"] for r in runs)
        med_ll = statistics.median(r["score"] for r in runs)
        mu_med = statistics.median(r["mu"] for r in runs)
        hw_med = statistics.median(r["hw"] for r in runs)
        t_med = statistics.median(r["elapsed"] for r in runs)
        print(f"{label:<24} {best_ll:>14.1f} {med_ll:>14.1f} "
              f"{mu_med:>10.3g} {hw_med:>10.3g} {t_med:>12.1f}")
    print("=" * 92)


if __name__ == "__main__":
    main()
