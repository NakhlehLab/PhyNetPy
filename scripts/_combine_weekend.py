"""Combine the per-chain trace logs from the parallel weekend run into
cross-chain convergence diagnostics.

Reads ``runs/weekend_10t/chain*/trace.log`` (BEAST/Tracer format written by
``MCMCSeqResult.write_log``), computes the Gelman-Rubin R-hat for every numeric
column across chains, and reports the pooled reticulation-count posterior.  Run
this after (or during) the run; chains that are still mid-flight just contribute
their latest checkpointed samples.

    py scripts/_combine_weekend.py
    py scripts/_combine_weekend.py --root runs/weekend_10t
"""
from __future__ import annotations

import os
import sys
import glob
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phynetpy._chain_analysis import read_tracer_log, gelman_rubin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("runs", "weekend_10t"))
    a = ap.parse_args()

    logs = sorted(glob.glob(os.path.join(a.root, "chain*", "trace.log")))
    if not logs:
        print(f"No trace logs found under {a.root}/chain*/trace.log")
        return

    chains = []          # list of (label, traces_dict)
    for path in logs:
        try:
            _states, traces = read_tracer_log(path)
        except Exception as e:
            print(f"  (skip {path}: {e})")
            continue
        n = len(next(iter(traces.values()))) if traces else 0
        chains.append((os.path.basename(os.path.dirname(path)), traces))
        print(f"  {os.path.dirname(path)}: {n} samples, "
              f"cols={list(traces.keys())}")

    if len(chains) < 2:
        print("\nNeed >= 2 chains with samples for R-hat.")
        return

    common = set(chains[0][1].keys())
    for _, tr in chains[1:]:
        common &= set(tr.keys())

    print("\n" + "=" * 60)
    print(f"Cross-chain Gelman-Rubin R-hat ({len(chains)} chains)")
    print("  (values <= 1.05 indicate the chains have mixed)")
    print("-" * 60)
    for name in sorted(common):
        series = [tr[name] for _, tr in chains
                  if len(tr[name]) >= 2]
        if len(series) < 2:
            continue
        try:
            rh = gelman_rubin(series)
        except Exception as e:
            rh = float("nan")
        flag = "" if (rh != rh) else ("  OK" if rh <= 1.05 else "  <-- not mixed")
        print(f"    {name:<20} R-hat={rh:.4f}{flag}")

    # Pooled reticulation-count posterior.
    if "reticulationCount" in common:
        counts: Counter = Counter()
        for _, tr in chains:
            for v in tr["reticulationCount"]:
                counts[int(round(v))] += 1
        total = sum(counts.values())
        print("\nPooled reticulation-count posterior:")
        for r in sorted(counts):
            print(f"    r={r}: {counts[r] / total:.3f}  ({counts[r]}/{total})")


if __name__ == "__main__":
    main()
