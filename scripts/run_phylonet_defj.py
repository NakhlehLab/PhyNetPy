#!/usr/bin/env python3
"""
Run PhyloNet's ``InferNetwork_MP_Allopp`` (Yan et al.) on the DEFJ dataset for a
head-to-head comparison with MP-Allop-2.

For each DEFJ condition we build a NEXUS file:

    #NEXUS
    BEGIN TREES;
      Tree gt1 = <newick>;
      ...
    END;
    BEGIN PHYLONET;
      InferNetwork_MP_Allopp (gt1,gt2,...) <maxRetic> -a <species:alleles map> \
          -pl <procs> -n 1;
    END;

invoke ``java -jar <PhyloNet.jar> <nexus>``, capture wall-clock and stdout,
parse the inferred species network (extended Newick) and its extra-lineage
score, and compute mu-distance / hardwired-cluster distance to the ground-truth
species network.

Gene trees are collapsed to one individual per subgenome (same as the MP-Allop
harness) so both methods receive identical inputs.

The number of reticulations PhyloNet may add is capped at the ground-truth
count per scenario (D=1, E=2, F=3, J=2), matching how MP-Allop fixes ploidy.

Results are appended to a resumable CSV.

Examples::

    # Verify the command + output parsing on one small case
    .venv/Scripts/python.exe scripts/run_phylonet_defj.py --probe \
        --scenario D --g 1 --t 4 --r 1

    # Full sweep (long; background)
    .venv/Scripts/python.exe scripts/run_phylonet_defj.py --pl 4

Copyright 2025 Mark Kessler, Luay Nakhleh. All rights reserved.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import defj_common as dc  # noqa: E402

from phynetpy.IO import read_newick  # noqa: E402
from phynetpy.GraphUtils import mu_distance, hardwired_cluster_distance  # noqa: E402

DEFAULT_JAR = Path.home() / "Desktop" / "PhyloNetv3_8_2.jar"

CSV_FIELDS = [
    "tier", "scenario", "g", "n", "t", "r",
    "max_retic", "xl_score", "final_pars",
    "mu_d", "hw_d", "n_retics", "n_leaves", "n_genes",
    "seconds", "newick", "error",
]

# "Inferred Network #1:" then a newick line; "Total number of extra lineages: N"
_NET_RE = re.compile(r"Inferred Network #\d+:\s*\n(.+)", re.MULTILINE)
_XL_RE = re.compile(r"Total number of extra lineages:\s*([0-9.]+)")


def build_allele_map_str(gene_map: dict[str, list[str]]) -> str:
    """PhyloNet taxa-map syntax: <sp1:a1,a2; sp2:b1>."""
    parts = [f"{sp}:{','.join(alleles)}" for sp, alleles in gene_map.items()]
    return "<" + "; ".join(parts) + ">"


def build_nexus(gene_tree_newicks: list[str], gene_map: dict[str, list[str]],
                max_retic: int, procs: int, n_return: int = 1) -> str:
    lines = ["#NEXUS", "BEGIN TREES;"]
    names = []
    for i, nwk in enumerate(gene_tree_newicks, 1):
        name = f"gt{i}"
        names.append(name)
        nwk = nwk.strip()
        if not nwk.endswith(";"):
            nwk += ";"
        lines.append(f"Tree {name} = {nwk}")
    lines.append("END;")
    lines.append("")
    lines.append("BEGIN PHYLONET;")
    gt_list = "(" + ",".join(names) + ")"
    amap = build_allele_map_str(gene_map)
    lines.append(
        f"InferNetwork_MP_Allopp {gt_list} {max_retic} "
        f"-a {amap} -pl {procs} -n {n_return};"
    )
    lines.append("END;")
    return "\n".join(lines) + "\n"


def parse_phylonet_output(stdout: str) -> tuple[str | None, float | None]:
    net_match = _NET_RE.search(stdout)
    xl_match = _XL_RE.search(stdout)
    net = net_match.group(1).strip() if net_match else None
    xl = float(xl_match.group(1)) if xl_match else None
    return net, xl


def run_one(jar: Path, scenario: str, tier: int, g: int, n: int, t: int, r: int,
            procs: int, true_net, keep_nexus: Path | None = None) -> dict:
    labels = dc.read_leaf_labels(dc.gene_tree_files(scenario, tier, g, n, t, r))
    gene_map, _ = dc.build_gene_map(labels)
    gts = dc.load_gene_trees(scenario, tier, g, n, t, r,
                             gene_map=gene_map, collapse=True)
    newicks = [gt.newick() for gt in gts]
    max_retic = dc.TRUE_RETICULATIONS[scenario]
    nexus = build_nexus(newicks, gene_map, max_retic, procs)

    with tempfile.NamedTemporaryFile("w", suffix=".nex", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(nexus)
        nexus_path = Path(fh.name)
    if keep_nexus is not None:
        keep_nexus.write_text(nexus, encoding="utf-8")

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            ["java", "-jar", str(jar), str(nexus_path)],
            capture_output=True, text=True, timeout=7200,
        )
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "seconds": time.perf_counter() - t0,
                "n_genes": len(gts)}
    finally:
        try:
            nexus_path.unlink()
        except OSError:
            pass
    seconds = time.perf_counter() - t0

    net_str, xl = parse_phylonet_output(stdout)
    if net_str is None:
        msg = (stderr or stdout).strip().replace("\n", " ")[:300]
        return {"error": f"parse:{msg}", "seconds": seconds,
                "n_genes": len(gts), "stdout": stdout, "stderr": stderr}

    res = {"xl_score": xl, "final_pars": xl, "seconds": seconds,
           "n_genes": len(gts), "newick": net_str, "error": "",
           "stdout": stdout, "stderr": stderr}
    try:
        inferred = read_newick(net_str)
        # PhyloNet emits redundant parentheses (e.g. "(((a,..)))#H1"), which
        # introduce degree-2 nodes. mu-distance is a metric on *reduced*
        # networks, so suppress them for a fair, like-for-like comparison
        # (MP-Allop already returns reduced networks).
        try:
            inferred.clean([True, True, True])
        except Exception:
            pass
        res["n_retics"] = sum(1 for v in inferred.V() if v.is_reticulation())
        res["n_leaves"] = len(list(inferred.get_leaves()))
        try:
            res["mu_d"] = mu_distance(inferred, true_net)
        except Exception as exc:
            res["mu_d"] = f"err:{exc}"
        try:
            res["hw_d"] = hardwired_cluster_distance(inferred, true_net)
        except Exception as exc:
            res["hw_d"] = f"err:{exc}"
    except Exception as exc:  # noqa: BLE001
        res["error"] = f"netparse:{exc}"
    return res


def load_done_keys(csv_path: Path) -> set[tuple]:
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            done.add((row["tier"], row["scenario"], row["g"], row["n"],
                      row["t"], row["r"]))
    return done


def append_row(csv_path: Path, row: dict) -> None:
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jar", type=Path, default=DEFAULT_JAR)
    parser.add_argument("--out", type=Path,
                        default=dc.project_root() / "runs" / "defj" / "phylonet_results.csv")
    parser.add_argument("--tiers", default="10,100")
    parser.add_argument("--scenarios", default="D,E,F,J")
    parser.add_argument("--reps", default="1-10")
    parser.add_argument("--pl", type=int, default=4, help="PhyloNet processors")
    parser.add_argument("--limit", type=int, default=0)
    # probe mode
    parser.add_argument("--probe", action="store_true",
                        help="run a single case and dump raw PhyloNet output")
    parser.add_argument("--scenario", default="D")
    parser.add_argument("--g", type=int, default=1)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--t", type=int, default=4)
    parser.add_argument("--r", type=int, default=1)
    parser.add_argument("--tier", type=int, default=10)
    args = parser.parse_args()

    if not args.jar.exists():
        print(f"ERROR: PhyloNet jar not found: {args.jar}", flush=True)
        return 2

    true_nets = {s: read_newick(nwk) for s, nwk in dc.TRUE_NETWORKS.items()}

    if args.probe:
        nexus_out = args.out.parent / "probe.nex"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        print(f"PROBE {args.tier}G {args.scenario}-g{args.g}-n{args.n}-"
              f"t{args.t}-r{args.r}", flush=True)
        res = run_one(args.jar, args.scenario, args.tier, args.g, args.n,
                      args.t, args.r, args.pl, true_nets[args.scenario],
                      keep_nexus=nexus_out)
        print(f"--- nexus written to {nexus_out} ---", flush=True)
        print("--- STDOUT (tail) ---", flush=True)
        print((res.get("stdout") or "")[-2000:], flush=True)
        if res.get("stderr"):
            print("--- STDERR (tail) ---", flush=True)
            print(res["stderr"][-1000:], flush=True)
        print("--- PARSED ---", flush=True)
        for k in ("xl_score", "mu_d", "hw_d", "n_retics", "n_leaves",
                  "seconds", "newick", "error"):
            print(f"  {k}: {res.get(k)}", flush=True)
        return 0

    tiers = [int(x) for x in args.tiers.split(",") if x.strip()]
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    if "-" in args.reps:
        lo, hi = args.reps.split("-")
        reps = list(range(int(lo), int(hi) + 1))
    else:
        reps = [int(x) for x in args.reps.split(",") if x.strip()]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(args.out)

    # Reuse the MP-Allop harness enumeration for an identical condition set.
    from benchmark_defj import enumerate_conditions
    conditions = list(enumerate_conditions(tiers, scenarios, reps))
    print(f"PhyloNet DEFJ sweep: {len(conditions)} conditions "
          f"(already done: {len(done)})", flush=True)

    n_runs = 0
    for (tier, scenario, g, n, t, r) in conditions:
        key = (str(tier), scenario, str(g), str(n), str(t), str(r))
        if key in done:
            continue
        label = f"{tier}G {scenario}-g{g}-n{n}-t{t}-r{r}"
        try:
            res = run_one(args.jar, scenario, tier, g, n, t, r, args.pl,
                          true_nets[scenario])
        except Exception as exc:  # noqa: BLE001
            res = {"error": f"run:{exc}", "seconds": 0, "n_genes": ""}
        row = {"tier": tier, "scenario": scenario, "g": g, "n": n, "t": t,
               "r": r, "max_retic": dc.TRUE_RETICULATIONS[scenario],
               **{k: res.get(k, "") for k in CSV_FIELDS
                  if k not in ("tier", "scenario", "g", "n", "t", "r", "max_retic")}}
        append_row(args.out, row)
        n_runs += 1
        print(f"  [{label}] xl={res.get('xl_score')} mu_d={res.get('mu_d')} "
              f"hw_d={res.get('hw_d')} retics={res.get('n_retics')} "
              f"{res.get('seconds', 0):.1f}s {res.get('error', '')}", flush=True)
        if args.limit and n_runs >= args.limit:
            print(f"Hit --limit {args.limit}; stopping.", flush=True)
            return 0

    print(f"Done. {n_runs} new PhyloNet runs written to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
