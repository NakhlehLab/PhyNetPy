#!/usr/bin/env python3
"""
Profile MP-Allop-2 hill-climbing on a heavy DEFJ case (default: J, 10 genes,
moderate ILS) to locate runtime bottlenecks.

Reports cProfile output sorted by cumulative and total time, plus targeted
call counts for the prime suspects (network deep-copies, ``subgenome_count``,
MUL expansion, and the parsimony scorer).

Run::

    .venv/Scripts/python.exe scripts/profile_defj.py --iters 200
    .venv/Scripts/python.exe scripts/profile_defj.py --scenario J --g 10 --t 20 \
        --profile-out runs/defj/j_g10.prof

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

# Reuse the model builders / data loaders from the verification harness.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_mp_allop import build_model  # noqa: E402

from phynetpy.ModelMove import SwitchParentage  # noqa: E402
from phynetpy.State import State  # noqa: E402


def run_hc(model, n_iters: int) -> dict:
    """A plain hill-climb loop (mirrors the benchmark harness)."""
    state = State(model)
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
    return {"accepted": accepted, "final_pars": -state.likelihood()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="J")
    parser.add_argument("--g", type=int, default=10)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--t", type=int, default=20)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--profile-out", type=Path, default=None)
    args = parser.parse_args()

    print(f"Building model {args.scenario}-g{args.g}-n{args.n}-t{args.t}-r{args.r} ...",
          flush=True)
    t0 = time.perf_counter()
    model = build_model(args.scenario, args.g, args.n, args.t, args.r, args.seed)
    build_time = time.perf_counter() - t0
    n_gts = len(model._likelihood_calculator.gene_trees)
    print(f"  build: {build_time:.2f}s, {n_gts} gene trees", flush=True)

    print(f"Profiling {args.iters} hill-climb iterations ...", flush=True)
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    result = run_hc(model, args.iters)
    prof.disable()
    elapsed = time.perf_counter() - t0

    print(f"\nWall: {elapsed:.2f}s  ({1000 * elapsed / args.iters:.1f} ms/iter)  "
          f"accepted={result['accepted']}  final_pars={result['final_pars']:.0f}\n",
          flush=True)

    stats = pstats.Stats(prof)

    print("=" * 78)
    print(f"  TOP {args.top} BY CUMULATIVE TIME")
    print("=" * 78)
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(args.top)
    print(s.getvalue())

    print("=" * 78)
    print(f"  TOP {args.top} BY TOTAL (SELF) TIME")
    print("=" * 78)
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats(pstats.SortKey.TIME).print_stats(args.top)
    print(s.getvalue())

    # Targeted call counts for the prime suspects.
    print("=" * 78)
    print("  PRIME-SUSPECT CALL COUNTS")
    print("=" * 78)
    suspects = ("deepcopy", "subgenome_count", "to_mul", "edges_to_subgenome_count",
                "score", "XL", "newick", "is_acyclic", "__hash__", "clean")
    rows = []
    for func, stat in stats.stats.items():
        fname = func[2]
        if any(sus in fname for sus in suspects):
            cc, nc, tt, ct, _ = stat
            rows.append((ct, tt, nc, f"{Path(func[0]).name}:{func[1]}:{fname}"))
    rows.sort(reverse=True)
    print(f"  {'cumtime':>9} {'tottime':>9} {'ncalls':>9}  function")
    for ct, tt, nc, label in rows[:25]:
        print(f"  {ct:>9.3f} {tt:>9.3f} {nc:>9}  {label}")

    if args.profile_out:
        args.profile_out.parent.mkdir(parents=True, exist_ok=True)
        prof.dump_stats(str(args.profile_out))
        print(f"\nWrote profile to {args.profile_out}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
