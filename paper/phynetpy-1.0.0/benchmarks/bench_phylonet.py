"""
Head-to-head comparison of PhyNetPy and PhyloNet on the same inference method.

PhyNetPy's ``InferNetwork_MPL`` cell is a reimplementation of PhyloNet's
``InferNetwork_MPL`` command (Yu & Nakhleh 2015).  This script runs both on
byte-identical gene-tree input and reports runtime and topological accuracy, so
the reimplementation claim is tested rather than asserted.

Data are simulated by PhyNetPy itself under the multispecies network coalescent
from a known species tree, which makes the whole comparison reproducible from
this repository -- no external benchmark archive is required.  The trade-off is
that the generating process is PhyNetPy's own simulator, so this measures
agreement and speed on data from a model both implementations assume, not
robustness to model misspecification.

Timing methodology
------------------
Two numbers are recorded for PhyloNet and both are reported:

* ``wall_s`` -- total wall time of the ``java -jar`` invocation.  This includes
  JVM startup and NEXUS parsing, and is what a user actually waits for.
* ``self_reported_s`` -- PhyloNet's own "Running Time (min)" line, which
  excludes JVM startup.

PhyNetPy's ``wall_s`` is the in-process ``perf_counter`` span around ``infer``,
with the interpreter and all extension modules already loaded, plus a separately
recorded ``import_s`` for the one-time import cost.  Comparing PhyNetPy's
in-process time against PhyloNet's wall time would flatter PhyNetPy by exactly
the JVM startup cost, so the self-reported column is the like-for-like one.

Environment
-----------
``PHYLONET_JAR`` must point at a PhyloNet jar.  Set it, or pass ``--jar``.

Run::

    set PHYLONET_JAR=C:\\Users\\Marky\\Documents\\PhyloNetJar\\phylonet.jar
    python paper/phynetpy-1.0.0/benchmarks/bench_phylonet.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from common import banner, environment, write

# (taxa, gene trees, reticulations in the generating network) grid, ordered so
# runtime grows down the table.  The r=0 rows test species-tree recovery, where
# both implementations should agree exactly; the r=1 rows test the reticulate
# search, which is where the two searches can diverge.
GRID = (
    (4, 25, 0),
    (4, 100, 0),
    (5, 50, 0),
    (5, 200, 0),
    (6, 100, 0),
    (6, 400, 0),
    (8, 200, 0),
    (5, 100, 1),
    (6, 200, 1),
    (8, 200, 1),
)
SEED = 20260925
# Both searches are stochastic, and PhyloNet's is not seeded through the
# command-line interface, so a single run per condition measures noise as much
# as method. Each condition is repeated on independently simulated replicates
# and the aggregate is what the paper reports.
REPLICATES = 3
# theta sets the amount of incomplete lineage sorting, and therefore how hard
# the inference problem is. At theta = 0.02 the simulated gene trees are all
# identical to the species topology, so both tools recover it trivially and the
# comparison measures nothing. theta = 0.5 puts roughly half the gene trees
# discordant with the species tree, which is the regime these methods exist
# for. Each condition records its own concordance so the difficulty is visible
# in the results rather than implied.
THETA = 0.5

_NET_RE = re.compile(r"^(-?[\d.E+-]+):\s*(\(.+;?)\s*$", re.MULTILINE)
_TIME_RE = re.compile(r"Running Time \(min\):\s*([\d.E+-]+)")


def phylonet_nexus(newicks: list[str], max_retic: int, runs: int = 1) -> str:
    """A PhyloNet NEXUS block running InferNetwork_MPL on these gene trees."""
    lines = ["#NEXUS", "BEGIN TREES;"]
    names = []
    for i, nwk in enumerate(newicks, 1):
        name = f"gt{i}"
        names.append(name)
        text = nwk.strip()
        if not text.endswith(";"):
            text += ";"
        lines.append(f"Tree {name} = {text}")
    lines.append("END;")
    lines.append("")
    lines.append("BEGIN PHYLONET;")
    lines.append(
        f"InferNetwork_MPL ({','.join(names)}) {max_retic} -x {runs} -n 1 -pl 1;"
    )
    lines.append("END;")
    return "\n".join(lines) + "\n"


def run_phylonet(jar: Path, nexus_text: str) -> dict[str, Any]:
    """Invoke PhyloNet on a NEXUS block; return timings and the best network."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".nex", delete=False, encoding="ascii"
    ) as handle:
        handle.write(nexus_text)
        path = Path(handle.name)
    try:
        start = time.perf_counter()
        proc = subprocess.run(
            ["java", "-jar", str(jar), str(path)],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        wall = time.perf_counter() - start
    finally:
        path.unlink(missing_ok=True)

    matches = _NET_RE.findall(proc.stdout)
    best_score, best_net = (None, None)
    if matches:
        # PhyloNet prints candidates best-first within each run block.
        best_score, best_net = float(matches[0][0]), matches[0][1].strip()

    self_times = [float(v) for v in _TIME_RE.findall(proc.stdout)]
    return {
        "wall_s": wall,
        "self_reported_s": sum(self_times) * 60 if self_times else None,
        "log_pseudolikelihood": best_score,
        "newick": best_net,
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-400:] if proc.stderr else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jar",
        default=os.environ.get("PHYLONET_JAR"),
        help="Path to the PhyloNet jar (default: $PHYLONET_JAR).",
    )
    args = parser.parse_args()

    import_start = time.perf_counter()
    from phynetpy import random_network, read_newick
    from phynetpy.criteria import PseudoLikelihood
    from phynetpy.GraphUtils import (
        count_reticulations,
        mu_distance,
        network_clusters,
        robinson_foulds_distance,
    )
    from phynetpy.infer import infer, simulate
    from phynetpy.models import MSC

    import_s = time.perf_counter() - import_start

    def cluster_key(net: Any) -> frozenset:
        return frozenset(frozenset(c) for c in network_clusters(net))

    def concordance(gene_trees: Any, truth: Any) -> dict[str, int]:
        """How much gene-tree discordance the simulation actually produced.

        Reported per condition so a reader can see the difficulty of the
        problem rather than inferring it from theta.
        """
        target = cluster_key(truth)
        keys = [cluster_key(t) for t in gene_trees.trees]
        return {
            "gene_trees": len(keys),
            "concordant_with_truth": sum(1 for k in keys if k == target),
            "distinct_topologies": len(set(keys)),
        }

    def network_with_reticulations(
        n_taxa: int, wanted: int, base_seed: int
    ) -> tuple[Any, int]:
        """Draw a network that actually has ``wanted`` reticulations.

        ``random_network`` grafts reticulations onto a Yule tree and warns
        rather than raises when it cannot place one, so the returned network
        may have fewer than requested.  Accepting that silently would compare
        both tools against a "truth" nobody asked for, so retry over seeds and
        report the count that was actually achieved.
        """
        for offset in range(50):
            candidate = random_network(
                n_taxa, level=wanted, seed=base_seed + offset * 101
            )
            if count_reticulations(candidate) == wanted:
                return candidate, base_seed + offset * 101
        raise RuntimeError(
            f"could not draw a {n_taxa}-taxon network with {wanted} "
            f"reticulations in 50 attempts"
        )

    results: dict[str, Any] = {"meta": environment(), "conditions": {}}
    results["meta"]["phynetpy_import_s"] = import_s

    if not args.jar or not Path(args.jar).exists():
        print(
            "PhyloNet jar not found. Set PHYLONET_JAR or pass --jar.\n"
            f"  got: {args.jar!r}",
            file=sys.stderr,
        )
        results["error"] = f"jar not found: {args.jar!r}"
        write(results, "phylonet")
        return

    jar = Path(args.jar)
    java_version = subprocess.run(
        ["java", "-version"], capture_output=True, text=True
    ).stderr.splitlines()
    results["meta"]["comparison"] = {
        "method": "InferNetwork_MPL",
        "phylonet_jar": str(jar),
        "phylonet_jar_bytes": jar.stat().st_size,
        "java_version": java_version[0].strip() if java_version else "unknown",
        "max_reticulations": "matched to the generating network per condition",
        "phylonet_search_runs_per_invocation": 1,
        "replicates_per_condition": REPLICATES,
        "data_generator": f"phynetpy simulate under MSC(theta={THETA})",
        "base_seed": SEED,
        "seed_scheme": "replicate i uses seed SEED + i for simulation and search",
        "stochasticity_note": (
            "Both searches are stochastic. PhyNetPy is seeded per replicate; "
            "PhyloNet's command-line interface is not seeded here, so its "
            "variation across replicates is uncontrolled. Aggregates over "
            f"{REPLICATES} replicates are what the paper reports; per-replicate "
            "values are retained under conditions[*].replicates."
        ),
        "timing_note": (
            "PhyloNet wall_s includes JVM startup; self_reported_s does not. "
            "PhyNetPy wall_s is in-process with modules already imported; the "
            "one-time import cost is meta.phynetpy_import_s. Compare "
            "self_reported_s against PhyNetPy wall_s for like-for-like."
        ),
    }

    banner(f"Head-to-head: InferNetwork_MPL  (jar {jar.name})")
    print(f"  PhyNetPy import cost: {import_s:.2f} s (one time, not per run)")
    print()
    print(f"  {REPLICATES} replicates per condition; means and exact-recovery "
          f"counts shown")
    print()
    print(
        f"  {'taxa':>5} {'genes':>6} {'r':>2} {'conc':>5} |"
        f" {'PNP dflt':>8} {'PNP phnt':>9} {'PhyloNet':>9} |"
        f" {'exact: dflt':>11} {'phnt':>5} {'PN':>5}"
    )
    print(f"  {'':>21}  {'--- mean runtime (s) ---':>30}"
          f"   {'--- recovered truth ---':>25}")
    print("  " + "-" * 90)

    def fmt(value: float | None, width: int, places: int = 2) -> str:
        return (
            f"{value:>{width}.{places}f}"
            if isinstance(value, (int, float))
            else f"{'n/a':>{width}}"
        )

    PRESETS = ("default", "phylonet")

    for n_taxa, n_genes, n_retic in GRID:
        replicates: list[dict[str, Any]] = []

        for rep in range(REPLICATES):
            seed = SEED + rep
            network_seed = seed
            if n_retic == 0:
                gts = simulate(
                    MSC(theta=THETA),
                    taxa=n_taxa,
                    n=n_genes,
                    data="gene_trees",
                    seed=seed,
                )
                truth = gts.true_network
            else:
                truth, network_seed = network_with_reticulations(
                    n_taxa, n_retic, seed
                )
                gts = simulate(
                    MSC(theta=THETA), truth, n=n_genes, data="gene_trees",
                    seed=seed,
                )

            # Both tools get exactly the reticulation budget the generating
            # network uses, so neither is handicapped by a mismatched space.
            true_retics = count_reticulations(truth)
            newicks = [tree.newick() for tree in gts.trees]

            def distances(net: Any, target: Any = truth) -> dict[str, Any]:
                if net is None:
                    return {"mu": None, "rf": None}
                try:
                    return {
                        "mu": float(mu_distance(target, net)),
                        "rf": float(robinson_foulds_distance(target, net)),
                    }
                except Exception as exc:  # returned topology unusable
                    return {"mu": None, "rf": None, "error": repr(exc)}

            # PhyNetPy under two search configurations. "default" is what a user
            # gets without asking; "phylonet" reproduces PhyloNet's
            # optimise-every-parameter-per-topology behaviour and is the
            # like-for-like setting for accuracy. Reporting only one of them
            # would either understate accuracy or misrepresent the default.
            pnp: dict[str, Any] = {}
            for preset in PRESETS:
                sink = io.StringIO()
                start = time.perf_counter()
                with contextlib.redirect_stdout(sink):
                    result = infer(
                        gts,
                        criterion=PseudoLikelihood(),
                        max_reticulations=true_retics,
                        preset=preset,
                        seed=seed,
                    )
                pnp[preset] = {
                    "wall_s": time.perf_counter() - start,
                    "score": result.score,
                    "newick": result.best.newick(),
                    "reticulations_inferred": count_reticulations(result.best),
                    "distances": distances(result.best),
                }

            pn = run_phylonet(jar, phylonet_nexus(newicks, true_retics))
            pn_net = None
            if pn["newick"]:
                try:
                    pn_net = read_newick(pn["newick"])
                except Exception:
                    pn_net = None

            replicates.append({
                "replicate": rep,
                "seed": seed,
                "network_seed": network_seed,
                "reticulations_in_truth": true_retics,
                "truth_newick": truth.newick(),
                "ils": concordance(gts, truth),
                "phynetpy": pnp,
                "phylonet": {
                    **pn,
                    "reticulations_inferred": (
                        count_reticulations(pn_net)
                        if pn_net is not None
                        else None
                    ),
                    "distances": distances(pn_net),
                },
            })

        def collect(getter: Any) -> list[float]:
            values = [getter(r) for r in replicates]
            return [v for v in values if isinstance(v, (int, float))]

        def summarize(values: list[float]) -> dict[str, float | None]:
            if not values:
                return {"mean": None, "min": None, "max": None, "n": 0}
            return {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "n": len(values),
            }

        aggregate = {
            "phynetpy": {
                preset: {
                    "runtime_s": summarize(
                        collect(lambda r, p=preset: r["phynetpy"][p]["wall_s"])
                    ),
                    "mu": summarize(
                        collect(
                            lambda r, p=preset:
                            r["phynetpy"][p]["distances"]["mu"]
                        )
                    ),
                    "exact_recoveries": sum(
                        1
                        for r in replicates
                        if r["phynetpy"][preset]["distances"]["mu"] == 0
                    ),
                }
                for preset in PRESETS
            },
            "phylonet": {
                "runtime_s": summarize(
                    collect(lambda r: r["phylonet"]["self_reported_s"])
                ),
                "wall_s": summarize(
                    collect(lambda r: r["phylonet"]["wall_s"])
                ),
                "mu": summarize(
                    collect(lambda r: r["phylonet"]["distances"]["mu"])
                ),
                "exact_recoveries": sum(
                    1
                    for r in replicates
                    if r["phylonet"]["distances"]["mu"] == 0
                ),
            },
            "concordance_mean": (
                sum(
                    r["ils"]["concordant_with_truth"] / r["ils"]["gene_trees"]
                    for r in replicates
                )
                / len(replicates)
            ),
        }

        true_retics = replicates[0]["reticulations_in_truth"]
        key = f"{n_taxa}taxa_{n_genes}genes_r{true_retics}"
        results["conditions"][key] = {
            "taxa": n_taxa,
            "gene_trees": n_genes,
            "reticulations_requested": n_retic,
            "reticulations_in_truth": true_retics,
            "replicates": replicates,
            "aggregate": aggregate,
        }

        agg = aggregate
        print(
            f"  {n_taxa:>5} {n_genes:>6} {true_retics:>2}"
            f" {agg['concordance_mean']:>5.0%} |"
            f" {fmt(agg['phynetpy']['default']['runtime_s']['mean'], 8)}"
            f" {fmt(agg['phynetpy']['phylonet']['runtime_s']['mean'], 9)}"
            f" {fmt(agg['phylonet']['runtime_s']['mean'], 9)} |"
            f" {agg['phynetpy']['default']['exact_recoveries']}/{REPLICATES}"
            f"      {agg['phynetpy']['phylonet']['exact_recoveries']}/{REPLICATES}"
            f"       {agg['phylonet']['exact_recoveries']}/{REPLICATES}"
        )

    write(results, "phylonet")


if __name__ == "__main__":
    main()
