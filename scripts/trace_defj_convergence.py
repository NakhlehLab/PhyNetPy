#!/usr/bin/env python3
"""
Measure when MP-Allop's simulated annealing actually converges on DEFJ, and
what it spends the rest of its iteration budget doing.

The DEFJ sweep gives every D/E/F condition a fixed budget of 3 restarts x
1500 iterations and never stops early. This script turns on the SA trace and
reports, per condition:

  - the iteration at which each chain last improved its run-best
    (``best_iter``), i.e. everything after it is wasted work,
  - the split of accepted moves into improving / score-neutral / genuinely
    uphill,
  - the distribution of |delta| over scored proposals, which is what the
    temperature schedule has to be calibrated against.

Run::

    PHYNETPY_DEFJ_ROOT=... .venv/Scripts/python.exe scripts/trace_defj_convergence.py

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import argparse
import copy
import statistics as st
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_defj import build_model  # noqa: E402

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
    ("F", 10, 10, 1, 4, 1),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=1500)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t-start", type=float, default=5.0)
    parser.add_argument("--t-end", type=float, default=0.01)
    args = parser.parse_args()

    print(f"SA budget: {args.restarts} restarts x {args.iters} iters "
          f"= {args.restarts * args.iters} scored iterations")
    print(f"Temperature: {args.t_start} -> {args.t_end}")
    print()

    all_deltas: list[float] = []
    waste_fracs: list[float] = []

    for (scenario, tier, g, n, t, r) in CASES:
        label = f"{scenario}-g{g}-t{t}"
        try:
            model, n_genes, _ = build_model(scenario, tier, g, n, t, r, args.seed)
        except Exception as exc:  # noqa: BLE001
            print(f"{label}: SKIP ({exc})")
            continue

        sa = SimulatedAnnealing(
            pkernel=Infer_MP_Allop_Kernel(), model=copy.deepcopy(model),
            num_iter=args.iters, t_start=args.t_start, t_end=args.t_end,
            n_restarts=args.restarts, seed=args.seed, trace=True,
        )
        sa.run()

        print(f"{label}  ({n_genes} gene trees)   best score = {-sa.best_score:.0f}")
        for ci, stats in enumerate(sa.run_stats):
            outcomes = Counter(row[5] for row in stats["trace"])
            deltas = [abs(row[4]) for row in stats["trace"]
                      if row[5] in ("improve", "neutral", "uphill", "reject")]
            all_deltas.extend(d for d in deltas if d > 0)

            best_iter = stats["best_iter"]
            waste = 1.0 - (best_iter + 1) / args.iters
            waste_fracs.append(waste)

            print(f"    chain {ci}: last improvement at iter {best_iter:>5} "
                  f"of {args.iters}  ->  {100 * waste:4.1f}% of the chain "
                  f"ran after the final improvement")
            print(f"             accepted={stats['accepted']:>5} "
                  f"(improve={outcomes['improve']:>4} "
                  f"neutral={outcomes['neutral']:>5} "
                  f"uphill={outcomes['uphill']:>5})  "
                  f"reject={outcomes['reject']:>5} "
                  f"invalid={outcomes['invalid']:>4}")
        print()

    if all_deltas:
        ds = sorted(all_deltas)
        print("Distribution of |delta| over non-tied scored proposals "
              f"(n={len(ds)}):")
        for q, name in ((0.5, "median"), (0.9, "p90"), (0.99, "p99")):
            print(f"    {name:>6}: {ds[int(q * (len(ds) - 1))]:.1f}")
        print(f"    {'max':>6}: {ds[-1]:.1f}")
        print()
        print(f"Mean fraction of each chain spent after its final "
              f"improvement: {100 * st.mean(waste_fracs):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
