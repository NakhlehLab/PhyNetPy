"""
Biallelic-marker (SNP) likelihood timing.

Measures ``score(network, BiallelicMarkers, criterion=Likelihood())`` -- the
Bryant et al. (2012) / Zhu et al. (2018) exact marker likelihood -- across
taxon counts and site counts.

Two timings are reported per configuration because they answer different
questions:

* ``cold_s`` -- one call on a freshly constructed engine.  This is what a user
  pays for a single one-off score, and it includes building the likelihood
  model graph.
* ``warm_per_call_s`` -- the median of repeated calls.  This is what an
  inference loop pays per candidate network, and it is the number that governs
  whether a search is tractable.

Quoting only the warm figure would overstate single-score performance; quoting
only the cold figure would overstate the cost inside a search.  Both go in the
JSON, and the manuscript says which is which.

Run: python paper/phynetpy-1.0.0/benchmarks/bench_snp.py
"""

from __future__ import annotations

import time
from typing import Any

from common import banner, environment, timed, write

RESULTS: dict[str, Any] = {"meta": {}, "configurations": {}}

# (label, newick, taxon count) -- ultrametric, in substitutions per site.
TOPOLOGIES = {
    3: "((A:0.01,B:0.01):0.01,C:0.02);",
    4: "((A:0.01,B:0.01):0.01,(C:0.01,D:0.01):0.01);",
    5: "(((A:0.01,B:0.01):0.005,C:0.015):0.005,(D:0.01,E:0.01):0.01);",
}

SITE_COUNTS = (100, 1000, 5000)
SEED = 1
THETA = 0.02


def main() -> None:
    from phynetpy import BranchLengthUnit, read_newick
    from phynetpy.criteria import Likelihood
    from phynetpy.infer import score, simulate
    from phynetpy.models import MSC

    RESULTS["meta"] = environment()
    RESULTS["meta"]["measurement"] = {
        "quantity": "exact biallelic-marker log-likelihood of a fixed network",
        "criterion": "Likelihood",
        "model": f"MSC(theta={THETA})",
        "branch_length_unit": "substitutions_per_site",
        "seed": SEED,
        "device": "CPU",
        "note": (
            "cold_s is a single call including likelihood-model construction; "
            "warm_per_call_s is the median of repeated calls, which is what an "
            "inference loop pays per candidate network."
        ),
    }

    banner("Biallelic-marker likelihood: score() timing")
    print(f"  {'taxa':>5} {'sites':>7} {'cold (s)':>12} {'warm (s)':>12}"
          f" {'log-lik':>16}")
    print(f"  {'-' * 5} {'-' * 7} {'-' * 12} {'-' * 12} {'-' * 16}")

    for n_taxa, newick in TOPOLOGIES.items():
        for n_sites in SITE_COUNTS:
            net = read_newick(newick)
            net.set_branch_length_unit(BranchLengthUnit.SUBSTITUTIONS_PER_SITE)
            markers = simulate(
                MSC(theta=THETA), net, n=n_sites, data="markers", seed=SEED
            )

            model = MSC(theta=THETA)
            crit = Likelihood()

            t0 = time.perf_counter()
            value = score(net, markers, model=model, criterion=crit)
            cold = time.perf_counter() - t0

            stats = timed(
                lambda: score(net, markers, model=model, criterion=crit),
                loops=3,
                repeats=5,
            )
            warm = stats["per_call_us"] / 1e6

            key = f"{n_taxa}taxa_{n_sites}sites"
            RESULTS["configurations"][key] = {
                "taxa": n_taxa,
                "sites": n_sites,
                "newick": newick,
                "cold_s": cold,
                "warm_per_call_s": warm,
                "log_likelihood": value,
                "protocol": stats,
            }
            print(
                f"  {n_taxa:>5} {n_sites:>7} {cold:>12.4f} {warm:>12.4f}"
                f" {value:>16.4f}"
            )

    write(RESULTS, "snp")


if __name__ == "__main__":
    main()
