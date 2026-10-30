"""
Check the numeric claims made in the prose against the benchmark JSON.

The tables and figures are generated, so they cannot drift. Sentences in
``sections/*.tex`` can. This script asserts the specific quantities the prose
states, and reports the measured value next to the claimed range so a
disagreement is obvious rather than subtle.

Every entry names the section and the sentence it guards. When a benchmark is
re-run and a claim moves outside its range, edit the prose, then widen or
correct the range here.

Run: python paper/phynetpy-1.0.0/benchmarks/verify_claims.py
Exits non-zero if any claim no longer holds.
"""

from __future__ import annotations

import sys

from common import load

FAILURES: list[str] = []
CHECKS = 0


def check(label: str, value: float, low: float, high: float, where: str) -> None:
    global CHECKS
    CHECKS += 1
    ok = low <= value <= high
    status = "ok  " if ok else "FAIL"
    print(f"  {status} {label:<52}{value:>10.3g}  (claim {low:g}-{high:g})")
    if not ok:
        FAILURES.append(f"{label}: measured {value:.4g}, prose says {low:g}-{high:g} ({where})")


def main() -> int:
    core = load("core")
    kernels = load("kernels")
    snp = load("snp")
    tests = load("testsuite")
    phylo = load("phylonet")

    print("Substitution models  [validation.tex: 'between 8.5x and 19x faster']")
    speedups = [e["speedup"] for e in core["substitution_models"].values()]
    check("min speedup vs expm", min(speedups), 6.0, 12.0, "sec:validation")
    check("max speedup vs expm", max(speedups), 14.0, 24.0, "sec:validation")
    worst_diff = max(
        e["max_abs_diff"] for e in core["substitution_models"].values()
    )
    check("max |diff| vs expm", worst_diff, 0.0, 1e-14, "sec:validation")

    print("\nGraph core  [validation.tex: '11x', '3.8x', '0.7x', '1.9x memory']")
    big = core["graph_core"]["1000_leaves"]
    check(
        "in_edges ratio vs NetworkX",
        big["ops"]["in_edges lookup"]["ratio_nx_over_phy"],
        8.0, 15.0, "sec:validation",
    )
    check(
        "get_leaves ratio vs NetworkX",
        big["ops"]["get_leaves"]["ratio_nx_over_phy"],
        2.2, 4.6, "sec:validation",
    )
    check(
        "topological_order ratio vs NetworkX",
        big["ops"]["topological_order"]["ratio_nx_over_phy"],
        0.45, 0.85, "sec:validation",
    )
    check(
        "PhyNetPy peak memory / NetworkX",
        1.0 / big["memory_ratio_nx_over_phy"],
        1.6, 2.2, "sec:validation",
    )

    print("\nCompiled kernel  [validation.tex: 'between 1.3x and 1.7x']")
    ratios = [e["speedup"] for e in kernels["mpl_engine"].values()]
    check("min compiled speedup", min(ratios), 1.15, 1.55, "sec:validation")
    check("max compiled speedup", max(ratios), 1.45, 2.1, "sec:validation")
    check(
        "max |log-PL difference| between paths",
        max(e["abs_diff"] for e in kernels["mpl_engine"].values()),
        0.0, 1e-10, "sec:validation",
    )

    print("\nMarker likelihood  [validation.tex: 'about 1.1 ms, 3 taxa/1000 sites']")
    cfg = snp["configurations"]["3taxa_1000sites"]
    check("warm ms per score", cfg["warm_per_call_s"] * 1000, 0.6, 2.2,
          "sec:validation")

    print("\nTest suite  [abstract, validation.tex, tables/testing]")
    check("tests collected", tests["suite"]["collected_total"], 1130, 1130,
          "abstract + sec:validation")
    check("tests passing", tests["suite"]["default_run"]["passed"], 1125, 1125,
          "sec:validation")
    check("tests failing", tests["suite"]["default_run"]["failed"], 0, 0,
          "sec:validation")
    check(
        "PhyloNet cross-check cases",
        tests["areas"]["phylonet_crosscheck"]["collected"],
        23, 23, "sec:validation",
    )
    check(
        "substitution-model tests",
        tests["areas"]["substitution_models"]["collected"],
        432, 432, "sec:validation",
    )
    check(
        "network-comparison tests",
        tests["areas"]["network_comparison"]["collected"],
        153, 153, "sec:validation",
    )

    print("\nHead-to-head  [validation.tex 5.3]")
    conds = phylo["conditions"]
    reps = phylo["meta"]["comparison"]["replicates_per_condition"]
    exact = {"default": 0, "phylonet": 0, "pn": 0}
    for entry in conds.values():
        agg = entry["aggregate"]
        exact["default"] += agg["phynetpy"]["default"]["exact_recoveries"]
        exact["phylonet"] += agg["phynetpy"]["phylonet"]["exact_recoveries"]
        exact["pn"] += agg["phylonet"]["exact_recoveries"]
    trials = len(conds) * reps

    check("conditions", len(conds), 10, 10, "sec:headtohead")
    check("replicates per condition", reps, 3, 3, "sec:headtohead")
    check("total trials", trials, 30, 30, "sec:headtohead")
    check("exact recoveries, PhyNetPy default", exact["default"], 6, 11,
          "sec:headtohead")
    check("exact recoveries, PhyNetPy phylonet preset", exact["phylonet"],
          5, 10, "sec:headtohead")
    check("exact recoveries, PhyloNet", exact["pn"], 15, 22, "sec:headtohead")
    check(
        "PhyloNet advantage in exact recoveries",
        exact["pn"] - max(exact["default"], exact["phylonet"]),
        5, 16, "sec:headtohead 'recovered it about twice as often'",
    )

    # Runtime: the prose says "faster on all but one condition".
    slower = [
        key
        for key, e in conds.items()
        if (e["aggregate"]["phylonet"]["runtime_s"]["mean"] or 0)
        < e["aggregate"]["phynetpy"]["default"]["runtime_s"]["mean"]
    ]
    check("conditions where PhyNetPy is slower on average", len(slower), 0, 2,
          "sec:headtohead 'faster on all but one'")

    ratios = [
        (e["aggregate"]["phylonet"]["runtime_s"]["mean"] or 0)
        / e["aggregate"]["phynetpy"]["default"]["runtime_s"]["mean"]
        for e in conds.values()
    ]
    check("largest runtime ratio", max(ratios), 3.0, 9.0,
          "sec:headtohead 'about 6x'")

    concs = [e["aggregate"]["concordance_mean"] for e in conds.values()]
    check("max mean gene-tree concordance", max(concs), 0.35, 0.60,
          "sec:headtohead concordance range")
    check("min mean gene-tree concordance", min(concs), 0.0, 0.20,
          "sec:headtohead concordance range")

    print(f"\n{CHECKS} claims checked.")
    if FAILURES:
        print(f"\n{len(FAILURES)} claim(s) no longer hold:", file=sys.stderr)
        for failure in FAILURES:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("All prose claims match the current benchmark output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
