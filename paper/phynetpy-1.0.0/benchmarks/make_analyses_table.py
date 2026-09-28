"""
Generate the table of analyses available in PhyNetPy from the live registry.

The rows come from ``phynetpy.infer.registered_cells()``, so the table cannot
drift from the code: an engine that is added, removed, or renamed changes the
table on the next run.  Everything the registry does not know -- the scientific
problem each method addresses, its primary citation, the PhyloNet command it
corresponds to, and whether the implementation is a reimplementation or new --
lives in the annotation table below and is keyed on the method name, so a
registry entry with no annotation is reported as an error rather than silently
omitted.

Run: python paper/phynetpy-1.0.0/benchmarks/make_analyses_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
TABLES = PAPER / "tables"

# Keyed on Engine.method. Every field is verifiable from the cited source or
# from src/_engines.py; see INVENTORY.md sections 2.15 and 13.
ANNOTATIONS: dict[str, dict[str, str]] = {
    "InferNetwork_ML": {
        "problem": "Network topology and parameters by maximum likelihood",
        "output": "Network, log likelihood",
        "cite": "Yu2014ML",
        "phylonet": "InferNetwork\\_ML",
        "status": "Reimpl.",
    },
    "InferNetwork_MPL": {
        "problem": "Network topology from gene-tree triplet frequencies",
        "output": "Network, log pseudo-likelihood",
        "cite": "Yu2015MPL",
        "phylonet": "InferNetwork\\_MPL",
        "status": "Reimpl.",
    },
    "MCMC_GT": {
        "problem": "Posterior over networks given gene trees",
        "output": "Posterior sample, MAP network",
        "cite": "Wen2016MCMCGT",
        "phylonet": "MCMC\\_GT",
        "status": "Reimpl.",
    },
    "MCMC_SEQ": {
        "problem": "Joint posterior over networks and gene trees from sequences",
        "output": "Posterior sample, MAP network",
        "cite": "Wen2018MCMCSEQ",
        "phylonet": "MCMC\\_SEQ",
        "status": "Reimpl.",
    },
    "MLE_BiMarkers": {
        "problem": "Exact marker likelihood of a given network",
        "output": "Log likelihood (scoring only)",
        "cite": "Bryant2012SNAPP,Zhu2018BiMarkers",
        "phylonet": "MLE\\_BiMarkers",
        "status": "Reimpl., scoring only",
    },
    "MCMC_BiMarkers": {
        "problem": "Posterior over networks from biallelic markers",
        "output": "Posterior sample, MAP network",
        "cite": "Zhu2018BiMarkers",
        "phylonet": "MCMC\\_BiMarkers",
        "status": "Reimpl.",
    },
    "MP_Allop": {
        "problem": "Parsimonious network under polyploidy, minimising deep "
                   "coalescences",
        "output": "Network, extra-lineage count",
        "cite": "Yan2022Allop,Than2009MDC",
        "phylonet": "InferNetwork\\_MP\\_Allopp",
        "status": "New impl.",
    },
}

# Human-readable names for the axis classes.
DATA_NAMES = {
    "GeneTrees": "Gene trees",
    "Alignment": "Alignment",
    "BiallelicMarkers": "Markers",
}
CRITERION_NAMES = {
    "MDC": "Parsimony",
    "Likelihood": "Likelihood",
    "PseudoLikelihood": "Pseudo-lik.",
    "Bayesian": "Bayesian",
}

# Cells that are legal but carry no engine, so ``infer`` raises
# NotImplementedError. Verified against src/_engines.py lines 34-40 and the
# validity matrix; see INVENTORY.md 2.15.
UNBUILT = [
    ("Gene trees", "MSC", "Parsimony", "InferNetwork\\_MP", "Sched. 1.0.0"),
    (
        "Markers",
        "MSC",
        "Pseudo-lik.",
        "MLE\\_BiMarkers -pseudo",
        "Sched. 1.0.0",
    ),
    ("Alignment", "MSC", "Likelihood", "---", "Not planned"),
]


def main() -> int:
    from phynetpy.infer import registered_cells

    cells = registered_cells()
    missing = [c[3] for c in cells if c[3] not in ANNOTATIONS]
    if missing:
        print(
            f"error: registered cells with no annotation: {missing}.\n"
            "Add them to ANNOTATIONS so the table stays complete.",
            file=sys.stderr,
        )
        return 1

    rows = []
    # Group MSC rows before Allopolyploid rows, then order by data type.
    order = {"MSC": 0, "Allopolyploid": 1}
    data_order = {"GeneTrees": 0, "Alignment": 1, "BiallelicMarkers": 2}
    for data, model, criterion, method in sorted(
        cells,
        key=lambda c: (order.get(c[1], 9), data_order.get(c[0], 9), c[2]),
    ):
        note = ANNOTATIONS[method]
        cite = "\\cite{" + note["cite"] + "}"
        rows.append(
            f"\\texttt{{{method.replace('_', chr(92) + '_')}}} & "
            f"{DATA_NAMES.get(data, data)} & "
            f"{model} & "
            f"{CRITERION_NAMES.get(criterion, criterion)} & "
            f"{note['problem']} {cite} & "
            f"\\texttt{{{note['phylonet']}}} & "
            f"{note['status']} \\\\"
        )

    unbuilt_rows = [
        f"--- & {data} & {model} & {criterion} & "
        f"\\emph{{raises}} \\texttt{{NotImplementedError}} & "
        f"\\texttt{{{pn}}} & {status} \\\\"
        for data, model, criterion, pn, status in UNBUILT
    ]

    # Explicit column widths, because the natural widths of the engine names
    # and the PhyloNet command names overrun \textwidth. \tabcolsep is reduced
    # locally rather than globally so the rest of the paper is unaffected.
    body = (
        "% Generated by benchmarks/make_analyses_table.py from "
        "phynetpy.infer.registered_cells().\n"
        "% Do not edit by hand.\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{@{}"
        "l"                 # engine
        ">{\\raggedright\\arraybackslash}p{15mm}"   # data
        ">{\\raggedright\\arraybackslash}p{18mm}"   # model ("Allopolyploid")
        ">{\\raggedright\\arraybackslash}p{16mm}"   # criterion
        ">{\\raggedright\\arraybackslash}p{37mm}"   # problem
        "l"                 # phylonet command
        ">{\\raggedright\\arraybackslash}p{17mm}"   # status
        "@{}}\n\\toprule\n"
        "Engine & Data & Model & Criterion & Problem addressed & "
        "PhyloNet command & Status \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\midrule\n\\multicolumn{7}{@{}l}{\\emph{Cells that are "
        "well defined but carry no engine}} \\\\\n"
        + "\n".join(unbuilt_rows)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )

    out = TABLES / "analyses.tex"
    out.write_text(body, encoding="utf-8")
    print(f"  wrote {out.relative_to(PAPER)} ({len(rows)} engines, "
          f"{len(unbuilt_rows)} unbuilt cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
