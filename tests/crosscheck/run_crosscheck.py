"""Numerical cross-check: PhyNetPy MCMC_SEQ engine vs. PhyloNet (Java + BEAGLE).

For each fixed (gene tree, species network, theta) state we compute, on BOTH
sides, the two factors that make up the MCMC_SEQ likelihood:

  * MSNC branch-length density   log P(g | Psi)
  * Felsenstein log-likelihood   log P(S | g)

PhyloNet's numbers come from its *own* classes
(``GeneTreeBrSpeciesNetDistribution`` and BEAGLE via ``UltrametricTree``),
invoked by the compiled ``CrossCheck`` Java harness in this directory.  Our
numbers come from :mod:`phynetpy._seq_likelihood`.  Identical inputs are fed to
both through a single shared spec file, so any disagreement is a real
discrepancy, not a difference in test setup.

The case list below is deliberately adversarial -- it targets the ways
coalescent / phylogenetic-likelihood engines actually break:

    * multiple alleles per species (intra-species coalescence, u > 1)
    * GTR with skewed base frequencies and asymmetric exchange rates
    * stacked reticulations (configuration enumeration / gamma bookkeeping)
    * boundary population sizes (very large -> heavy ILS, very small)
    * embeddings with zero probability (must agree on -inf)
    * IUPAC ambiguity codes and gaps in the alignment (tip partials)
    * branch-length saturation (P(t) -> stationary, underflow guards)
    * fully constant alignments
    * a larger 5-taxon caterpillar (cluster/configuration scaling)

Run from anywhere:

    py tests/crosscheck/run_crosscheck.py
"""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from pathlib import Path

from phynetpy.Network import Network
from phynetpy.models import BranchLengthUnit
from phynetpy._seq_likelihood import (
    JC69,
    GTR,
    FelsensteinCalculator,
    gene_tree_msnc_log_density,
)

# --------------------------------------------------------------------------
# Environment.  Defaults are the user's verified-working setup, overridable
# via env vars so the cross-check is portable to other machines / CI.
# --------------------------------------------------------------------------
JAR = os.environ.get("PHYLONET_JAR", r"C:\Users\Marky\Desktop\PhyloNetv3_8_2.jar")
BEAGLE_DIR = os.environ.get(
    "BEAGLE_DIR", r"C:\Program Files\Common Files\libhmsbeagle"
)
HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Test cases.  Networks use PhyloNet rich-newick (reticulation inheritance as
# ``:len::gamma``); branch lengths are expected substitutions/site.  Gene
# trees are ultrametric (heights encoded as branch lengths).  Optional keys:
#   seqs:  {label: dna}      -> triggers the Felsenstein comparison
#   model: "JC" | "GTR"      -> substitution model (default JC)
#   freqs: [piA,piC,piG,piT]  rates: [AC,AG,AT,CG,CT,GT]  (GTR only)
#   map:   {species: [alleles...]} -> multi-allele species mapping
# --------------------------------------------------------------------------
CASES: list[dict] = [
    # ---- baselines ----------------------------------------------------
    {
        "name": "tree2taxa",
        "net": "(A:0.05,B:0.05)R;",
        "gt": "(A:0.08,B:0.08)g0;",
        "theta": 0.02,
        "seqs": {
            "A": "ACGTACGTACGTACGTAACGTTGCA",
            "B": "ACGTACGAACGTAGGTAACGTTGCA",
        },
    },
    {
        "name": "tree3taxa_concordant",
        "net": "((A:0.01,B:0.01)I1:0.02,C:0.03)R;",
        "gt": "((A:0.02,B:0.02)g1:0.03,C:0.05)g0;",
        "theta": 0.02,
        "seqs": {
            "A": "ACGTACGTACGTACGTAACGTTGCAGGTACC",
            "B": "ACGTACGAACGTAGGTAACGTTGCAGGAACC",
            "C": "ACGTTCGAACGAAGGTAACCTTGCAGGTACT",
        },
    },
    {
        "name": "tree3taxa_deepILS",
        "net": "((A:0.01,B:0.01)I1:0.02,C:0.03)R;",
        "gt": "((B:0.04,C:0.04)g1:0.02,A:0.06)g0;",
        "theta": 0.02,
        "seqs": {
            "A": "ACGTACGTACGTACGTAACGTTGCAGGTACC",
            "B": "ACGTACGAACGTAGGTAACGTTGCAGGAACC",
            "C": "ACGTTCGAACGAAGGTAACCTTGCAGGTACT",
        },
    },
    {
        "name": "network1retic",
        "net": "((A:0.03,(B:0.01)#H1:0.02::0.3)X:0.03,"
               "(#H1:0.03::0.7,C:0.04)Y:0.02)R;",
        "gt": "((A:0.04,B:0.04)g1:0.04,C:0.08)g0;",
        "theta": 0.02,
    },
    # ---- substitution-model stress: GTR, skewed pi, asymmetric rates --
    {
        "name": "gtr_skewed",
        "net": "((A:0.01,B:0.01)I1:0.02,C:0.03)R;",
        "gt": "((A:0.02,B:0.02)g1:0.03,C:0.05)g0;",
        "theta": 0.02,
        "model": "GTR",
        "freqs": [0.1, 0.2, 0.3, 0.4],
        "rates": [1.0, 2.5, 0.7, 1.3, 3.1, 0.9],
        "seqs": {
            "A": "ACGTACGTACGTACGTAACGTTGCAGGTACCTTAGC",
            "B": "ACGTACGAACGTAGGTAACGTTGCAGGAACCTTAGG",
            "C": "ACGTTCGAACGAAGGTAACCTTGCAGGTACTTTAAC",
        },
    },
    # ---- multiple alleles per species (intra-species coalescence) -----
    {
        "name": "multiallele",
        "net": "((A:0.01,B:0.01)I1:0.02,C:0.03)R;",
        "gt": "(((a1:0.005,a2:0.005)n1:0.015,b1:0.02)n2:0.02,c1:0.04)g0;",
        "theta": 0.02,
        "map": {"A": ["a1", "a2"], "B": ["b1"], "C": ["c1"]},
        "seqs": {
            "a1": "ACGTACGTACGTACGTAACGTTGCAGGTACC",
            "a2": "ACGTACGTACGTACCTAACGTTGCAGGTACC",
            "b1": "ACGTACGAACGTAGGTAACGTTGCAGGAACC",
            "c1": "ACGTTCGAACGAAGGTAACCTTGCAGGTACT",
        },
    },
    # ---- stacked (two) reticulations ----------------------------------
    {
        "name": "net2retic",
        "net": "((A:0.03,(B:0.01)#H1:0.02::0.6)P1:0.03,"
               "(#H2:0.03::0.5,(#H1:0.03::0.4,(C:0.02)#H2:0.02::0.5)P2:0.01)P3:0.01)R;",
        "gt": "((A:0.04,B:0.04)g1:0.04,C:0.08)g0;",
        "theta": 0.02,
    },
    # ---- boundary population sizes ------------------------------------
    {
        "name": "bigtheta_heavyILS",
        "net": "((A:0.01,B:0.01)I1:0.02,C:0.03)R;",
        "gt": "((A:0.02,B:0.02)g1:0.03,C:0.05)g0;",
        "theta": 1.0,
    },
    {
        "name": "tinytheta",
        "net": "((A:0.01,B:0.01)I1:0.02,C:0.03)R;",
        "gt": "((A:0.02,B:0.02)g1:0.03,C:0.05)g0;",
        "theta": 0.002,
    },
    # ---- zero-probability embedding (must agree on -inf) --------------
    {
        "name": "invalid_embedding",
        "net": "((A:0.01,B:0.01)I1:0.02,C:0.03)R;",
        # A,C coalesce (0.02) below their only shared population (root, 0.03).
        "gt": "((A:0.02,C:0.02)g1:0.04,B:0.06)g0;",
        "theta": 0.02,
    },
    # ---- IUPAC ambiguity codes and gaps -------------------------------
    {
        "name": "ambiguity_gaps",
        "net": "((A:0.01,B:0.01)I1:0.02,C:0.03)R;",
        "gt": "((A:0.02,B:0.02)g1:0.03,C:0.05)g0;",
        "theta": 0.02,
        "seqs": {
            "A": "ACGTRYSWKMBDHVN-ACGTACGT",
            "B": "ACGT-NACGTRYKMACGTAC-GTA",
            "C": "NNGTACGTACG--CGTRYACGTAC",
        },
    },
    # ---- branch-length saturation (long branches) --------------------
    {
        "name": "saturation",
        "net": "((A:0.5,B:0.5)I1:1.0,C:1.5)R;",
        "gt": "((A:1.0,B:1.0)g1:1.5,C:2.5)g0;",
        "theta": 1.0,
        "seqs": {
            "A": "ACGTACGTACGTACGTAACGTTGCAGGTACC",
            "B": "TGCATGCATGCATGCATTGCAACGTCCATGG",
            "C": "GGCCAATTGGCCAATTCCGGAATTGGCCAAT",
        },
    },
    # ---- fully constant alignment ------------------------------------
    {
        "name": "constant_sites",
        "net": "((A:0.01,B:0.01)I1:0.02,C:0.03)R;",
        "gt": "((A:0.02,B:0.02)g1:0.03,C:0.05)g0;",
        "theta": 0.02,
        "seqs": {
            "A": "AAAAAAAAAACCCCCCCCCC",
            "B": "AAAAAAAAAACCCCCCCCCC",
            "C": "AAAAAAAAAACCCCCCCCCC",
        },
    },
    # ---- larger 5-taxon caterpillar -----------------------------------
    {
        "name": "tree5taxa",
        "net": "((((A:0.01,B:0.01)I1:0.01,C:0.02)I2:0.01,D:0.03)I3:0.01,E:0.04)R;",
        "gt": "((((A:0.02,B:0.02)g1:0.01,C:0.03)g2:0.02,D:0.05)g3:0.02,E:0.07)g0;",
        "theta": 0.02,
        "seqs": {
            "A": "ACGTACGTACGTACGTAACGTTGCAGGTACC",
            "B": "ACGTACGAACGTAGGTAACGTTGCAGGAACC",
            "C": "ACGTTCGAACGAAGGTAACCTTGCAGGTACT",
            "D": "ACGAACGTAGGTACGTAACGTAGCAGCTACC",
            "E": "ACGTACGTTCGTACGTATCGTTGGAGGTACA",
        },
    },
]


def phylonet_to_phynetpy(newick: str) -> str:
    """Convert PhyloNet rich-newick reticulations to PhyNetPy's gamma syntax.

    ``...#H1:<len>::<gamma>`` -> ``...#H1:<len>[&gamma=<gamma>]``.
    For plain trees (no ``::``) this is a no-op.
    """
    return re.sub(r":([-\d.eE+]+)::([-\d.eE+]+)", r":\1[&gamma=\2]", newick)


def write_spec(path: Path) -> None:
    lines: list[str] = []
    for c in CASES:
        lines.append(f"CASE {c['name']}")
        lines.append(f"NET {c['net']}")
        lines.append(f"GT {c['gt']}")
        lines.append(f"THETA {c['theta']}")
        if c.get("model"):
            lines.append(f"MODEL {c['model']}")
        if c.get("freqs"):
            lines.append("FREQS " + " ".join(str(x) for x in c["freqs"]))
        if c.get("rates"):
            lines.append("RATES " + " ".join(str(x) for x in c["rates"]))
        if c.get("map"):
            grp = " ".join(f"{sp}:{','.join(al)}" for sp, al in c["map"].items())
            lines.append(f"MAP {grp}")
        for label, seq in c.get("seqs", {}).items():
            lines.append(f"SEQ {label} {seq}")
        lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_java(spec: Path) -> dict[tuple[str, str], float | str]:
    """Run the Java harness; return {(case, factor): value-or-error-string}."""
    env = dict(os.environ)
    env["PATH"] = BEAGLE_DIR + os.pathsep + env.get("PATH", "")
    cmd = [
        "java",
        f"-Djava.library.path={BEAGLE_DIR}",
        "-cp",
        f"{HERE}{os.pathsep}{JAR}",
        "CrossCheck",
        str(spec),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        print("Java harness failed:\n", proc.stdout, proc.stderr, file=sys.stderr)
        raise SystemExit(1)
    out: dict[tuple[str, str], float | str] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "RESULT":
            case, factor = parts[1], parts[2]
            if parts[3] == "ERROR":
                out[(case, factor)] = "ERROR " + " ".join(parts[4:])
            else:
                out[(case, factor)] = float(parts[3])
    return out


def _model_for(case: dict):
    if case.get("model") == "GTR":
        return GTR(case["freqs"], case["rates"])
    return JC69()


def python_values(case: dict) -> dict[str, float]:
    """Compute MSNC density (+ Felsenstein if seqs) with the PhyNetPy engine."""
    vals: dict[str, float] = {}
    sp_net = Network.from_newick(phylonet_to_phynetpy(case["net"]))
    gt = Network.from_newick(case["gt"])
    sp_net.set_branch_length_unit(BranchLengthUnit.SUBSTITUTIONS_PER_SITE)
    gt.set_branch_length_unit(BranchLengthUnit.SUBSTITUTIONS_PER_SITE)
    if case.get("map"):
        species_of = {al: sp for sp, alleles in case["map"].items() for al in alleles}
    else:
        species_of = {n.label: n.label for n in gt.get_leaves()}
    vals["MSNC"] = gene_tree_msnc_log_density(
        gt, sp_net, species_of, theta=case["theta"]
    )
    if case.get("seqs"):
        gt_f = Network.from_newick(case["gt"])
        gt_f.set_branch_length_unit(BranchLengthUnit.SUBSTITUTIONS_PER_SITE)
        calc = FelsensteinCalculator(case["seqs"])
        vals["FELSEN"] = calc.log_likelihood(gt_f, _model_for(case))
    return vals


def fmt(v: float | str) -> str:
    if isinstance(v, str):
        return v
    if v == float("-inf"):
        return "-inf"
    return f"{v:.10f}"


def compare(case: dict, java: dict, tol: float = 1e-5):
    """Yield (factor, java_val, py_val, diff, ok) tuples for a case."""
    py = python_values(case)
    for factor in ("MSNC", "FELSEN"):
        if factor not in py:
            continue
        jv = java.get((case["name"], factor), "MISSING")
        pv = py[factor]
        ok = False
        diff: float | str = "n/a"
        if isinstance(jv, (int, float)):
            if math.isinf(jv) or math.isinf(pv):
                ok = (jv == pv)
                diff = 0.0 if ok else float("inf")
            else:
                diff = abs(jv - pv)
                ok = diff < tol
        yield factor, jv, pv, diff, ok


def main() -> int:
    spec = HERE / "cases.spec"
    write_spec(spec)
    java = run_java(spec)

    print(f"{'case':<24}{'factor':<8}{'PhyloNet':>18}{'PhyNetPy':>18}{'|diff|':>13}  ok")
    print("-" * 90)
    all_ok = True
    for c in CASES:
        for factor, jv, pv, diff, ok in compare(c, java):
            all_ok = all_ok and ok
            ds = diff if isinstance(diff, str) else f"{diff:.2e}"
            print(
                f"{c['name']:<24}{factor:<8}{fmt(jv):>18}{fmt(pv):>18}{ds:>13}  "
                f"{'YES' if ok else 'NO'}"
            )
    print("-" * 90)
    print("ALL MATCH" if all_ok else "MISMATCHES FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
