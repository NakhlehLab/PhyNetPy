#!/usr/bin/env python3
"""
Full DEFJ benchmark sweep for MP-Allop-2.

Enumerates every DEFJ model condition that exists on disk (scenarios D/E/F/J x
gene counts {1,3,10,100} x individuals {1,3,9} x ILS {4,20,100} x 10 reps),
runs Simulated Annealing (x3 restarts) using the SwitchParentage (SRPP) move,
and records, per run:

  - initial / final parsimony (extra lineages),
  - accepted / uphill move counts,
  - mu-distance and hardwired-cluster distance to the ground-truth species
    network,
  - reticulation count, leaf count, gene-tree count,
  - wall-clock seconds,
  - the inferred network in (extended) Newick.

Multi-individual conditions (n > 1) are collapsed to one individual per
subgenome (see ``defj_common.collapse_to_canonical``) so the bijective allele
map can score them and so MP-Allop and PhyloNet receive identical inputs.

Results are appended to a CSV incrementally and the run is **resumable**: rows
already present (keyed by tier/scenario/g/n/t/r/method) are skipped.

Examples::

    # Full sweep (long; run in the background)
    .venv/Scripts/python.exe scripts/benchmark_defj.py

    # Smoke test: only 10-gene, scenarios D and E, replicate 1, few iters
    .venv/Scripts/python.exe scripts/benchmark_defj.py --tiers 10 \
        --scenarios D,E --reps 1 --hc-iters 50 --sa-iters 50 --limit 4

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import defj_common as dc  # noqa: E402

from phynetpy._infer_mp_allop import (  # noqa: E402
    MPAllopComponent,
    allele_map_set_ilp,
    partition_gene_trees,
)
from phynetpy.IO import read_newick  # noqa: E402
from phynetpy.ModelFactory import ModelFactory  # noqa: E402
from phynetpy.ModelMove import SwitchParentage  # noqa: E402
from phynetpy.MetropolisHastings import (  # noqa: E402
    SimulatedAnnealing, Infer_MP_Allop_Kernel,
)
from phynetpy.State import State  # noqa: E402
from phynetpy.GraphUtils import mu_distance, hardwired_cluster_distance  # noqa: E402


CSV_FIELDS = [
    "tier", "scenario", "g", "n", "t", "r", "method",
    "init_pars", "final_pars", "accepted", "uphill",
    "mu_d", "hw_d", "n_retics", "n_leaves", "n_genes",
    "seconds", "newick", "error",
]


def enumerate_conditions(tiers, scenarios, reps):
    """Yield (tier, scenario, g, n, t, r) for every condition that exists."""
    for tier in tiers:
        gene_counts = dc.GENE_COUNTS_100 if tier == 100 else dc.GENE_COUNTS_10
        for scenario in scenarios:
            ils = dc.ILS_J if scenario == "J" else dc.ILS_DEF
            for g in gene_counts:
                for n in dc.INDIVIDUALS:
                    for t in ils:
                        for r in reps:
                            path = dc.gene_tree_files(scenario, tier, g, n, t, r)
                            if path.exists():
                                yield (tier, scenario, g, n, t, r)


def build_model(scenario, tier, g, n, t, r, seed):
    """Build an MP-Allop model (collapsed to one individual per subgenome)."""
    rng = np.random.default_rng(seed)
    labels = dc.read_leaf_labels(dc.gene_tree_files(scenario, tier, g, n, t, r))
    gene_map, _ = dc.build_gene_map(labels)
    gts = dc.load_gene_trees(scenario, tier, g, n, t, r,
                             gene_map=gene_map, collapse=True)
    for gt in gts:
        gt.put_item("allele maps", allele_map_set_ilp(gt, gene_map))
        gt.put_item("leaf descendants", gt.leaf_descendants_all())
    start_net = partition_gene_trees(gene_map, rng=rng)
    model = ModelFactory(MPAllopComponent(start_net, gene_map, gts, rng)).build()
    model.rng = rng
    return model, len(gts)


def _net_stats(net, true_net):
    n_retics = sum(1 for v in net.V() if v.is_reticulation())
    n_leaves = len(list(net.get_leaves()))
    try:
        mu = mu_distance(net, true_net)
    except Exception as exc:  # leaf-set mismatch etc.
        mu = f"err:{exc}"
    try:
        hw = hardwired_cluster_distance(net, true_net)
    except Exception as exc:
        hw = f"err:{exc}"
    return n_retics, n_leaves, mu, hw


def run_hc(model, n_iters, true_net):
    state = State(copy.deepcopy(model))
    init_pars = -state.likelihood()
    t0 = time.perf_counter()
    accepted = 0
    for i in range(n_iters):
        move = SwitchParentage(i)
        if state.generate_next(move):
            cur = state.likelihood()
            proposed = state.proposed().likelihood()
            if cur - proposed < 0:
                state.commit(move)
                accepted += 1
            else:
                state.revert(move)
    seconds = time.perf_counter() - t0
    net = state.current_model.network
    n_retics, n_leaves, mu, hw = _net_stats(net, true_net)
    return {
        "init_pars": init_pars, "final_pars": -state.likelihood(),
        "accepted": accepted, "uphill": 0,
        "mu_d": mu, "hw_d": hw, "n_retics": n_retics, "n_leaves": n_leaves,
        "seconds": seconds, "newick": net.newick(), "error": "",
    }


def run_sa(model, n_iters, true_net, seed, n_restarts=3):
    init_pars = -State(copy.deepcopy(model)).likelihood()
    t0 = time.perf_counter()
    sa = SimulatedAnnealing(
        pkernel=Infer_MP_Allop_Kernel(), model=copy.deepcopy(model),
        num_iter=n_iters, t_start=5.0, t_end=0.01,
        n_restarts=n_restarts, seed=seed,
    )
    final_state = sa.run()
    seconds = time.perf_counter() - t0
    net = final_state.current_model.network
    accepted = sum(s.get("accepted", 0) for s in sa.run_stats)
    uphill = sum(s.get("uphill", 0) for s in sa.run_stats)
    n_retics, n_leaves, mu, hw = _net_stats(net, true_net)
    return {
        "init_pars": init_pars, "final_pars": -sa.best_score,
        "accepted": accepted, "uphill": uphill,
        "mu_d": mu, "hw_d": hw, "n_retics": n_retics, "n_leaves": n_leaves,
        "seconds": seconds, "newick": net.newick(), "error": "",
    }


def load_done_keys(csv_path: Path) -> set[tuple]:
    """Set of (tier,scenario,g,n,t,r,method) already recorded."""
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            done.add((row["tier"], row["scenario"], row["g"], row["n"],
                      row["t"], row["r"], row["method"]))
    return done


def append_row(csv_path: Path, row: dict) -> None:
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=dc.project_root() / "runs" / "defj" / "mp_allop_results.csv")
    parser.add_argument("--tiers", default="10,100",
                        help="comma-separated gene tiers (10,100)")
    parser.add_argument("--scenarios", default="D,E,F,J")
    parser.add_argument("--reps", default="1-10",
                        help="replicates, e.g. '1-10' or '1,2,3'")
    parser.add_argument("--methods", default="SA3",
                        help="which searches to run: HC and/or SA3 "
                             "(default SA3 only; HC is no longer run by default)")
    parser.add_argument("--hc-iters", type=int, default=0,
                        help="HC iterations (0 = auto by scenario size)")
    parser.add_argument("--sa-iters", type=int, default=0,
                        help="SA iterations per restart (0 = auto)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many runs (0 = no limit)")
    args = parser.parse_args()

    tiers = [int(x) for x in args.tiers.split(",") if x.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    methods = [m.strip().upper() for m in args.methods.split(",") if m.strip()]
    if "-" in args.reps:
        lo, hi = args.reps.split("-")
        reps = list(range(int(lo), int(hi) + 1))
    else:
        reps = [int(x) for x in args.reps.split(",") if x.strip()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(args.out)
    true_nets = {s: read_newick(nwk) for s, nwk in dc.TRUE_NETWORKS.items()}

    conditions = list(enumerate_conditions(tiers, scenarios, reps))
    print(f"DEFJ MP-Allop sweep: {len(conditions)} conditions x {len(methods)} "
          f"methods -> {len(conditions) * len(methods)} runs", flush=True)
    print(f"Output: {args.out}  (already done: {len(done)} rows)", flush=True)

    n_runs = 0
    for (tier, scenario, g, n, t, r) in conditions:
        # SA iteration budget per chain (num_iter per restart; SA3 = 3 restarts).
        # J gets the largest budget because it has the largest search space
        # (14 taxa, 3 reticulations) and benefits most from a longer anneal;
        # every other scenario shares a common budget. MP-Allop runs well ahead
        # of PhyloNet on runtime, so we spend that headroom on more iterations.
        sa_iters = args.sa_iters or (5000 if scenario == "J" else 1500)
        hc_iters = args.hc_iters or sa_iters

        pending = [m for m in methods
                   if (str(tier), scenario, str(g), str(n), str(t), str(r), m)
                   not in done]
        if not pending:
            continue

        label = f"{tier}G {scenario}-g{g}-n{n}-t{t}-r{r}"
        try:
            model, n_genes = build_model(scenario, tier, g, n, t, r, args.seed)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label}] BUILD FAILED: {exc!r}", flush=True)
            for m in pending:
                append_row(args.out, {
                    "tier": tier, "scenario": scenario, "g": g, "n": n,
                    "t": t, "r": r, "method": m, "init_pars": "", "final_pars": "",
                    "accepted": "", "uphill": "", "mu_d": "", "hw_d": "",
                    "n_retics": "", "n_leaves": "", "n_genes": "", "seconds": "",
                    "newick": "", "error": f"build:{exc}",
                })
            continue

        true_net = true_nets[scenario]
        base = {"tier": tier, "scenario": scenario, "g": g, "n": n,
                "t": t, "r": r, "n_genes": n_genes}

        for m in pending:
            t_run = time.perf_counter()
            try:
                if m == "HC":
                    res = run_hc(model, hc_iters, true_net)
                elif m == "SA3":
                    res = run_sa(model, sa_iters, true_net, args.seed, n_restarts=3)
                else:
                    print(f"  unknown method {m}, skipping", flush=True)
                    continue
            except Exception as exc:  # noqa: BLE001
                res = {k: "" for k in CSV_FIELDS}
                res["error"] = f"run:{exc}"
                res["seconds"] = time.perf_counter() - t_run
            row = {**base, "method": m, **{k: res.get(k, "") for k in CSV_FIELDS
                                           if k not in base and k != "method"}}
            append_row(args.out, row)
            n_runs += 1
            print(f"  [{label}] {m}: pars={res.get('final_pars')} "
                  f"mu_d={res.get('mu_d')} hw_d={res.get('hw_d')} "
                  f"{res.get('seconds', 0):.1f}s", flush=True)

            if args.limit and n_runs >= args.limit:
                print(f"Hit --limit {args.limit}; stopping.", flush=True)
                return 0

    print(f"Done. {n_runs} new runs written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
