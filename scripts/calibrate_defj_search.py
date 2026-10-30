#!/usr/bin/env python3
"""
Calibrate MP-Allop's simulated-annealing search knobs on DEFJ.

The trace in ``trace_defj_convergence.py`` showed two things about the
D/E/F conditions:

  * a mean ~80% of every chain runs *after* its last run-best improvement,
    so a stall-based stop reclaims most of the budget, and
  * the overwhelming majority of accepted moves are score-*neutral* plateau
    steps (ties are accepted unconditionally because ``exp(0) == 1``), and
    each acceptance is the expensive branch of an iteration.

It also showed that |delta| over non-tied proposals has median ~10 and p90
~39, so the shipped ``t_start=5.0`` is *cold* relative to real score
differences rather than hot.

This script measures runtime and accuracy for a handful of configurations so
the defaults can be chosen from data instead of theory.

Run::

    PHYNETPY_DEFJ_ROOT=... .venv/Scripts/python.exe scripts/calibrate_defj_search.py

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import argparse
import copy
import statistics as st
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import defj_common as dc  # noqa: E402
from benchmark_defj import build_model  # noqa: E402

from phynetpy.IO import read_newick  # noqa: E402
from phynetpy.GraphUtils import mu_distance, hardwired_cluster_distance  # noqa: E402
from phynetpy.MetropolisHastings import (  # noqa: E402
    Infer_MP_Allop_Kernel,
    SimulatedAnnealing,
)

# (scenario, tier, g, n, t, r)
CASES = [
    ("D", 10, 1, 1, 4, 1),
    ("D", 10, 1, 1, 100, 1),
    ("D", 10, 10, 1, 4, 1),
    ("D", 10, 10, 1, 100, 1),
    ("E", 10, 10, 1, 4, 1),
    ("E", 10, 10, 1, 100, 1),
    ("F", 10, 10, 1, 4, 1),
    ("F", 10, 10, 1, 100, 1),
]

# label -> kwargs overriding the shipped SA defaults
CONFIGS = {
    "baseline (shipped)":        dict(),
    "stall=200":                 dict(stall_limit=200),
    "neutral=0.0":               dict(neutral_accept_prob=0.0),
    "stall=200 neutral=0.0":     dict(stall_limit=200, neutral_accept_prob=0.0),
    "stall=200 neutral=0.25":    dict(stall_limit=200, neutral_accept_prob=0.25),
    "stall=200 nb=0.0 T=15":     dict(stall_limit=200, neutral_accept_prob=0.0,
                                      t_start=15.0),
}


def run_config(model, true_net, iters, seed, restarts, kwargs):
    base = dict(num_iter=iters, t_start=5.0, t_end=0.01,
                n_restarts=restarts, seed=seed)
    base.update(kwargs)

    t0 = time.perf_counter()
    sa = SimulatedAnnealing(pkernel=Infer_MP_Allop_Kernel(),
                            model=copy.deepcopy(model), **base)
    sa.run()
    seconds = time.perf_counter() - t0

    net = sa.best_network
    try:
        mu = mu_distance(net, true_net)
    except Exception:
        mu = float("nan")
    try:
        hw = hardwired_cluster_distance(net, true_net)
    except Exception:
        hw = float("nan")

    iters_run = sum(s["iters_run"] for s in sa.run_stats)
    accepted = sum(s["accepted"] for s in sa.run_stats)
    neutral = sum(s.get("neutral", 0) for s in sa.run_stats)
    return {
        "pars": -sa.best_score, "mu": mu, "hw": hw, "seconds": seconds,
        "iters_run": iters_run, "accepted": accepted, "neutral": neutral,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reps", type=int, default=2,
                        help="repeat each (case, config) this many times; "
                             "start-network construction is not reproducible "
                             "across runs, so averaging matters")
    parser.add_argument("--configs", default="",
                        help="comma-separated subset of config labels to run "
                             "(default: all)")
    args = parser.parse_args()

    if args.configs:
        wanted = [c.strip() for c in args.configs.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in CONFIGS]
        if unknown:
            print(f"Unknown config(s): {unknown}\nAvailable: {list(CONFIGS)}")
            return 2
        for label in list(CONFIGS):
            if label not in wanted:
                del CONFIGS[label]

    true_nets = {s: read_newick(nwk) for s, nwk in dc.TRUE_NETWORKS.items()}

    models = {}
    for case in CASES:
        scenario, tier, g, n, t, r = case
        try:
            models[case] = build_model(scenario, tier, g, n, t, r, args.seed)[0]
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {case}: {exc}")

    results = {label: [] for label in CONFIGS}

    for label, kwargs in CONFIGS.items():
        for case, model in models.items():
            scenario = case[0]
            for rep in range(args.reps):
                res = run_config(model, true_nets[scenario], args.iters,
                                 args.seed + rep, args.restarts, kwargs)
                res["case"] = case
                results[label].append(res)
        agg = results[label]
        print("%-26s pars %6.1f   mu_d %5.2f   hw_d %5.2f   "
              "%7.1fs total   iters %6.0f   accepted %5.0f (neutral %5.0f)" % (
                  label,
                  st.mean(r["pars"] for r in agg),
                  st.mean(r["mu"] for r in agg),
                  st.mean(r["hw"] for r in agg),
                  sum(r["seconds"] for r in agg),
                  st.mean(r["iters_run"] for r in agg),
                  st.mean(r["accepted"] for r in agg),
                  st.mean(r["neutral"] for r in agg),
              ), flush=True)

    print()
    base = results["baseline (shipped)"]
    base_s = sum(r["seconds"] for r in base)
    print("Speedup vs baseline, and accuracy delta (negative pars/mu is better):")
    for label, agg in results.items():
        if label == "baseline (shipped)":
            continue
        print("  %-26s %5.2fx faster   d_pars %+6.2f   d_mu %+5.2f   d_hw %+5.2f" % (
            label,
            base_s / sum(r["seconds"] for r in agg),
            st.mean(r["pars"] for r in agg) - st.mean(r["pars"] for r in base),
            st.mean(r["mu"] for r in agg) - st.mean(r["mu"] for r in base),
            st.mean(r["hw"] for r in agg) - st.mean(r["hw"] for r in base),
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
