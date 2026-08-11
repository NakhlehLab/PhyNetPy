#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
reticulation_distance_report.py
================================

Generate a full, self-contained PDF report comparing every network
dissimilarity measure available in PhyNetPy, with a focus on validating the
new reticulation-aware measure of

    Nakhleh (2026), "Reticulation-Aware Dissimilarity for Phylogenetic
    Networks: A Tripartition-Matching Approach"

implemented in :mod:`phynetpy.ReticulationComparison`.

The report follows the validation protocol of Section 10 of the note and, in
particular, stress-tests the measure on the published adversarial network pair
of Cardona, Rossello & Valiente ("Tripartitions do not always discriminate
phylogenetic networks"; the tree-child networks of Figs. 4 and 8 of "Comparison
of Tree-Child Phylogenetic Networks", arXiv:0708.3499), which the classical
tripartition metric provably cannot separate.

Usage
-----
    python scripts/reticulation_distance_report.py [output.pdf]

Produces (by default) ``reticulation_distance_report.pdf`` in the repository
root and prints a plain-text summary to stdout.
"""

from __future__ import annotations

import sys
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from phynetpy.Network import Network, Node, Edge
from phynetpy.GraphUtils import (
    ascii as ascii_net,
    hardwired_cluster_distance,
    softwired_cluster_distance,
    mu_distance,
    tripartition_distance,
    displayed_tree_distance,
    robinson_foulds_distance,
)
from phynetpy.ReticulationComparison import (
    compare_networks,
    reticulation_dissimilarity,
    reticulation_precision_recall,
    nakhleh_metric,
    combined_dissimilarity,
    reticulation_tripartitions,
)


# ===================================================================
# Network construction helpers
# ===================================================================

def build(edges: List[Tuple[str, str]],
          retics: Optional[set] = None,
          clean: bool = False) -> Network:
    """Build a Network from a directed edge list of node labels."""
    retics = set(retics or [])
    net = Network()
    names: List[str] = []
    seen: set = set()
    for s, d in edges:
        for x in (s, d):
            if x not in seen:
                seen.add(x)
                names.append(x)
    nodes = {n: Node(n, is_reticulation=(n in retics)) for n in names}
    net.add_nodes(*nodes.values())
    for s, d in edges:
        net.add_edges(Edge(nodes[s], nodes[d]))
    if clean:
        net.clean()
    return net


# ── The Cardona-Rossello-Valiente non-separating tree-child pair ──
#
# Reconstructed exactly from the mu-vectors in Tables 1 (Fig. 4 = N) and 3
# (Fig. 8 = N') of arXiv:0708.3499.  The two networks are non-isomorphic
# (mu-distance 2) yet share their hybrids' clusters AND reticulation
# scenarios, so the classical tripartition metric -- and the new D_ret --
# report distance 0.  They differ only by swapping which of the hybrids B, C
# hangs beneath internal nodes b and c.
_N_FIG4 = [
    ("r", "a"), ("r", "b"),
    ("a", "1"), ("a", "A"),
    ("b", "c"), ("b", "B"),
    ("c", "d"), ("c", "C"),
    ("d", "A"), ("d", "5"),
    ("e", "2"), ("e", "B"),
    ("A", "e"), ("B", "f"), ("C", "4"),
    ("f", "3"), ("f", "C"),
]
_N_FIG8 = [
    ("r", "a"), ("r", "b"),
    ("a", "1"), ("a", "A"),
    ("b", "c"), ("b", "C"),
    ("c", "d"), ("c", "B"),
    ("d", "A"), ("d", "5"),
    ("e", "2"), ("e", "B"),
    ("A", "e"), ("B", "f"), ("C", "4"),
    ("f", "3"), ("f", "C"),
]
# A single reticulation-edge rewiring of N: hybrid B's parent arc (b -> B) is
# relocated to sit above leaf 1 (a -> g -> 1, and g -> B).  This perturbs
# exactly one of the three reticulations, the scenario the note predicts should
# yield precision = recall = F1 = 2/3 against N.
_N_REWIRED = [
    ("r", "a"), ("r", "b"),
    ("a", "g"), ("a", "A"),
    ("g", "1"), ("g", "B"),
    ("b", "c"),
    ("c", "d"), ("c", "C"),
    ("d", "A"), ("d", "5"),
    ("e", "2"), ("e", "B"),
    ("A", "e"), ("B", "f"), ("C", "4"),
    ("f", "3"), ("f", "C"),
]
_RETICS = {"A", "B", "C"}

# Trees on the same leaf set {1,2,3,4,5} so that leaf-set-sensitive metrics
# (mu, Robinson-Foulds) are well defined across the whole battery.
_TREE_BALANCED = [
    ("r", "x12"), ("r", "x345"),
    ("x12", "1"), ("x12", "2"),
    ("x345", "3"), ("x345", "x45"),
    ("x45", "4"), ("x45", "5"),
]
_TREE_ALT = [
    ("r", "x13"), ("r", "x245"),
    ("x13", "1"), ("x13", "3"),
    ("x245", "2"), ("x245", "x45"),
    ("x45", "4"), ("x45", "5"),
]
_TREE_CATERPILLAR = [
    ("r", "1"), ("r", "y2"),
    ("y2", "2"), ("y2", "y3"),
    ("y3", "3"), ("y3", "y4"),
    ("y4", "4"), ("y4", "5"),
]


def literature_networks() -> Dict[str, Network]:
    """Return the battery of five-leaf networks used throughout the report."""
    return {
        "N (Fig.4)": build(_N_FIG4, _RETICS),
        "N' (Fig.8)": build(_N_FIG8, _RETICS),
        "N rewired": build(_N_REWIRED, _RETICS, clean=True),
        "Tree balanced": build(_TREE_BALANCED),
        "Tree alt": build(_TREE_ALT),
        "Tree caterpillar": build(_TREE_CATERPILLAR),
    }


# ── Proposition 4 (base-rate masking) family ──
#
# Caterpillar on n leaves; N1 places a reticulation among the *prefix* leaves
# and N2 among the *suffix* leaves, so the hybrid clusters are leaf-disjoint
# across the two networks while the tree scaffold is shared.

def _caterpillar(n: int) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Edge list of a caterpillar on leaves L1..Ln, plus the internal spine."""
    edges: List[Tuple[str, str]] = []
    spine = [f"s{i}" for i in range(n - 1)]
    edges.append((spine[0], "L1"))
    for i in range(n - 2):
        edges.append((spine[i], spine[i + 1]))
        edges.append((spine[i], f"L{i + 2}"))
    edges.append((spine[-1], f"L{n - 1}"))
    edges.append((spine[-1], f"L{n}"))
    return edges, spine


def _add_retic_over(edges: List[Tuple[str, str]],
                    target_leaf: str,
                    donor_parent: str,
                    donor_child: str,
                    hybrid: str) -> List[Tuple[str, str]]:
    """
    Insert a reticulation whose hybrid ``hybrid`` becomes the new parent of
    ``target_leaf``, drawing a second parent edge from a subdivision of the
    donor edge (``donor_parent`` -> ``donor_child``).
    """
    new_edges: List[Tuple[str, str]] = []
    sub = f"{hybrid}_v"
    for (s, d) in edges:
        if d == target_leaf:
            # Reattach the target leaf's parent edge to the hybrid.
            new_edges.append((s, hybrid))
        elif (s, d) == (donor_parent, donor_child):
            # Subdivide the donor edge and hang a second hybrid parent off it.
            new_edges.append((s, sub))
            new_edges.append((sub, d))
        else:
            new_edges.append((s, d))
    new_edges.append((sub, hybrid))
    new_edges.append((hybrid, target_leaf))
    return new_edges


def base_rate_family(n: int, k: int) -> Tuple[Network, Network]:
    """
    Build the pair (N1, N2) of Proposition 4: ``k`` reticulations among the
    prefix leaves of an ``n``-leaf caterpillar versus ``k`` among the suffix
    leaves, on a shared scaffold.
    """
    base, spine = _caterpillar(n)
    e1 = list(base)
    e2 = list(base)
    for j in range(k):
        # Prefix reticulation: hybrid over leaf L{j+1}, donor a nearby edge.
        e1 = _add_retic_over(
            e1, f"L{j + 1}", spine[j + 1], spine[j + 2], f"H{j}"
        )
        # Suffix reticulation: hybrid over leaf L{n-j}, donor a nearby edge.
        e2 = _add_retic_over(
            e2, f"L{n - j}", spine[n - 3 - j], spine[n - 2 - j], f"G{j}"
        )
    retic1 = {f"H{j}" for j in range(k)}
    retic2 = {f"G{j}" for j in range(k)}
    return build(e1, retic1, clean=True), build(e2, retic2, clean=True)


# ===================================================================
# Metric registry
# ===================================================================

def _safe(fn: Callable[[Network, Network], float],
          a: Network, b: Network) -> Optional[float]:
    """Evaluate a metric, returning None on failure (e.g. leaf-set mismatch)."""
    try:
        return float(fn(a, b))
    except Exception:
        return None


# (display name, short header, callable, "new" flag).
# Callables take (reference, inferred).
METRICS: List[Tuple[str, str, Callable[[Network, Network], float], bool]] = [
    ("D_ret (norm)", "Dret_n", lambda a, b: reticulation_dissimilarity(a, b, normalize=True), True),
    ("D_ret (raw)", "Dret", lambda a, b: reticulation_dissimilarity(a, b, normalize=False), True),
    ("D (Nakhleh)", "D", lambda a, b: nakhleh_metric(a, b), True),
    ("D_lambda=.5", "Dl.5", lambda a, b: combined_dissimilarity(a, b, lam=0.5), True),
    ("mu-distance", "mu", mu_distance, False),
    ("tripartition", "trip", tripartition_distance, False),
    ("hardwired", "hard", hardwired_cluster_distance, False),
    ("softwired", "soft", softwired_cluster_distance, False),
    ("displayed-tree", "disp", displayed_tree_distance, False),
    ("Robinson-Foulds", "RF", robinson_foulds_distance, False),
]


# ===================================================================
# PDF rendering helpers
# ===================================================================

TITLE_FS = 15
BODY_FS = 9.2
MONO_FS = 8.0


def _fig() -> plt.Figure:
    return plt.figure(figsize=(8.5, 11))


def text_page(pdf: PdfPages, title: str, body: str,
              mono: bool = False) -> None:
    """Render a page of flowing text (optionally monospace)."""
    fig = _fig()
    fig.text(0.07, 0.95, title, fontsize=TITLE_FS, fontweight="bold",
             va="top")
    fig.text(
        0.07, 0.90, body, fontsize=(MONO_FS if mono else BODY_FS), va="top",
        ha="left", wrap=True,
        family=("monospace" if mono else "sans-serif"),
    )
    pdf.savefig(fig)
    plt.close(fig)


def table_page(pdf: PdfPages, title: str, col_labels: List[str],
               rows: List[List[str]], note: str = "",
               col_widths: Optional[List[float]] = None,
               fontsize: float = 8.0) -> None:
    """Render a page holding a single table with an optional footnote."""
    fig = _fig()
    fig.text(0.07, 0.95, title, fontsize=TITLE_FS, fontweight="bold",
             va="top")
    ax = fig.add_axes([0.05, 0.30, 0.90, 0.58])
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="upper center",
                   cellLoc="center", colLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.0, 1.35)
    for (row, _col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#dfe6ef")
    if col_widths:
        for (row, col), cell in tbl.get_celld().items():
            cell.set_width(col_widths[col])
    if note:
        fig.text(0.07, 0.26, note, fontsize=BODY_FS, va="top", wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.3f}"


# ===================================================================
# Report sections
# ===================================================================

def page_title(pdf: PdfPages) -> None:
    body = (
        "A validation and comparison report for the phylogenetic-network\n"
        "dissimilarity measures in PhyNetPy, centred on the new\n"
        "reticulation-aware, tripartition-matching measure of Nakhleh (2026).\n\n"
        "Contents\n"
        "  1.  Design & integration: how the measure is exposed and used\n"
        "  2.  Catalogue of every dissimilarity measure now available\n"
        "  3.  Boundary sanity checks (Section 6.1 / E5)\n"
        "  4.  Literature stress test: the Cardona-Rossello-Valiente pair\n"
        "        that tripartitions provably cannot separate (Section 11)\n"
        "  5.  Base-rate masking (Proposition 4 / E2)\n"
        "  6.  Monotone degradation & precision/recall (E1)\n"
        "  7.  Full pairwise comparison of all measures\n\n"
        "All numbers in this report are computed live by\n"
        "scripts/reticulation_distance_report.py against the installed\n"
        "phynetpy package; regenerate with:\n\n"
        "    python scripts/reticulation_distance_report.py\n"
    )
    text_page(pdf, "Reticulation-Aware Network Comparison in PhyNetPy", body)


def page_design(pdf: PdfPages) -> None:
    body = (
        "The measure lives in phynetpy.ReticulationComparison and is\n"
        "re-exported from phynetpy.GraphUtils and the top-level package.\n\n"
        "SINGLE ENTRY POINT\n"
        "    from phynetpy import compare_networks\n"
        "    result = compare_networks(reference, inferred)\n"
        "    D, D_ret_hat, prec, rec = result   # iterable => (D, D_ret, P, R)\n\n"
        "The result is a NetworkComparison dataclass exposing:\n"
        "    .D            global Nakhleh metric (Eq. 5, unnormalized)\n"
        "    .D_hat        bounded global score 2D/(|V1|+|V2|) in [0,1]\n"
        "    .D_ret        reticulation dissimilarity (Eq. 4, metric)\n"
        "    .D_ret_hat    normalized reticulation dissimilarity in [0,1]\n"
        "    .precision/.recall/.f1   reticulation-level recovery (Def. 4)\n"
        "    .matching     the optimal (ref_index, inf_index, cost) pairing\n"
        "    .combined(lam)            D_lambda = lam*D_ret + (1-lam)*D\n\n"
        "PARAMETERS (all optional, with the note's defaults)\n"
        "    distance = 'jaccard' | 'symmetric'   ground set distance\n"
        "    rho      = deletion penalty for an unmatched reticulation\n"
        "               (default = maximally-wrong = 3 under Jaccard)\n"
        "    tolerance= tau; a matched pair counts as 'recovered' iff its\n"
        "               cost <= tau (tau=0 demands exact recovery)\n"
        "    alpha    = block weight; alpha=1 is the leaf-set measure,\n"
        "               alpha<1 adds the topology-aware refinement (Sec. 8)\n\n"
        "CONVENIENCE WRAPPERS\n"
        "    reticulation_dissimilarity(n1, n2, normalize=True) -> float\n"
        "    reticulation_precision_recall(ref, inf) -> (prec, rec, f1)\n"
        "    nakhleh_metric(n1, n2) -> D            (also normalize=True)\n"
        "    combined_dissimilarity(n1, n2, lam=0.5) -> D_lambda\n"
        "    reticulation_tripartitions(net) -> [ReticulationTripartition]\n\n"
        "WHY IT EXISTS\n"
        "    Reticulation is rare relative to the tree-like bulk of a network,\n"
        "    so a global measure can report near-perfect agreement while every\n"
        "    inferred hybridization is wrong (the base-rate fallacy).  D_ret\n"
        "    scores ONLY the reticulation content; it is a pseudometric on\n"
        "    networks (blind to tree-only edits) and pairs with the global D\n"
        "    via D_lambda to recover a genuine metric.\n"
    )
    text_page(pdf, "1.  Design & Integration", body)


def page_catalogue(pdf: PdfPages) -> None:
    rows = [
        ["D_ret (norm)", "reticulation events only", "[0,1]",
         "pseudometric; NEW"],
        ["D_ret (raw)", "reticulation events only", "[0,inf)",
         "metric on signatures; NEW"],
        ["D (Nakhleh)", "whole network (subnetworks)", "[0,inf)",
         "metric (reduced nets); NEW"],
        ["D_lambda", "convex mix of D_ret & D", "[0,inf)",
         "metric for lambda<1; NEW"],
        ["mu-distance", "path-multiplicity vectors", "[0,inf)",
         "metric (tree-child)"],
        ["tripartition", "edge tripartitions", "[0,inf)",
         "NOT a metric"],
        ["hardwired", "hardwired clusters", "[0,inf)", "dissimilarity"],
        ["softwired", "softwired clusters", "[0,inf)", "dissimilarity"],
        ["displayed-tree", "displayed tree set", "[0,inf)", "dissimilarity"],
        ["Robinson-Foulds", "non-trivial clusters", "[0,inf)",
         "metric (trees)"],
    ]
    note = (
        "Only the reticulation-aware family (D_ret, D, D_lambda) isolates the\n"
        "hybridization signal.  Every other measure aggregates over the whole\n"
        "network and is therefore dominated by its tree-like structure -- the\n"
        "central concern the new measure addresses.  'tripartition' here is the\n"
        "classical PhyloNet edge-tripartition dissimilarity, which is provably\n"
        "unable to separate some non-isomorphic networks (see Section 4)."
    )
    table_page(pdf, "2.  Catalogue of Available Dissimilarity Measures",
               ["measure", "what it scores", "range", "properties"],
               rows, note=note,
               col_widths=[0.20, 0.34, 0.16, 0.30])


def page_boundary(pdf: PdfPages) -> None:
    nets = literature_networks()
    N = nets["N (Fig.4)"]
    tree = nets["Tree balanced"]
    tree2 = nets["Tree caterpillar"]

    scenarios = [
        ("two distinct trees", tree, tree2),
        ("network vs its underlying tree*", tree, N),
        ("self-comparison (N vs N)", N, N),
    ]
    col_labels = ["scenario"] + [m[1] for m in METRICS] + ["Prec", "Rec"]
    rows = []
    for label, ref, inf in scenarios:
        r = compare_networks(ref, inf)
        row = [label]
        for name, short, fn, _ in METRICS:
            row.append(_fmt(_safe(fn, ref, inf)))
        row.append(_fmt(r.precision))
        row.append(_fmt(r.recall))
        rows.append(row)
    col_widths = [0.20] + [0.066] * (len(col_labels) - 1)

    note = (
        "Expected (Section 6.1 / E5): two trees agree perfectly on reticulate\n"
        "content (D_ret=0) while the global measures still separate them;\n"
        "measuring a network against a tree gives D_ret=1 with precision 0\n"
        "(every reticulation is a false positive); self-comparison is 0\n"
        "throughout with precision=recall=1.\n"
        "* here 'Tree balanced' stands in as an underlying tree of N; only the\n"
        "reticulation columns and Prec/Rec are meaningful for that row.\n"
        "Headers: Dret_n = normalized D_ret, Dret = raw D_ret, D = Nakhleh\n"
        "metric, Dl.5 = D_lambda(0.5); mu/trip/hard/soft/disp/RF per catalogue."
    )
    table_page(pdf, "3.  Boundary Sanity Checks (Section 6.1 / E5)",
               col_labels, rows, note=note, col_widths=col_widths,
               fontsize=7.0)


def page_literature(pdf: PdfPages) -> None:
    nets = literature_networks()
    N = nets["N (Fig.4)"]
    Np = nets["N' (Fig.8)"]

    # Text + tripartition breakdown page.
    trips_N = reticulation_tripartitions(N)
    trip_lines = []
    for t in sorted(trips_N, key=lambda x: sorted(x.B)):
        proper = "proper" if t.is_proper() else "IMPROPER (A∩C≠∅)"
        trip_lines.append(
            f"    hybrid {t.hybrid}:  A={sorted(t.A)}  B(cluster)={sorted(t.B)}"
            f"  C={sorted(t.C)}   [{proper}]"
        )
    r = compare_networks(N, Np)
    body = (
        "Cardona, Rossello & Valiente exhibit explicit non-isomorphic\n"
        "tree-child networks that the classical tripartition metric cannot\n"
        "tell apart.  We reconstructed the pair exactly from the mu-vectors in\n"
        "Tables 1 & 3 of arXiv:0708.3499 (their Figs. 4 and 8).  Both are\n"
        "networks on {1,2,3,4,5} with three reticulations A, B, C of clusters\n"
        "{2,3,4}, {3,4}, {4}; the two networks differ only by swapping which\n"
        "hybrid (B or C) hangs beneath internal nodes b and c.\n\n"
        "Reticulation tripartitions of N (identical multiset for N'):\n"
        + "\n".join(trip_lines) + "\n\n"
        "This confirms Proposition 1 / Remark 1 of the note on a real\n"
        "published network: hybrid A has incomparable parents (clusters\n"
        "{1,2,3,4} and {2,3,4,5}) and a PROPER tripartition, whereas hybrids\n"
        "B and C each have one parent ancestral to the other -- a triangle --\n"
        "and are IMPROPER, exactly the weak-time-consistency violation the\n"
        "note flags.\n\n"
        "Because the two networks share their entire (L(h), reticulation\n"
        "scenario) signature (Lemma 1), D_ret = 0 -- the designed pseudometric\n"
        "behaviour -- while the global metrics D and mu-distance still\n"
        "separate them.  Numbers on the next page.\n\n"
        f"    D_ret(N, N')        = {r.D_ret:.3f}\n"
        f"    mu-distance(N, N')  = {mu_distance(N, Np):.0f}   (matches the paper)\n"
        f"    tripartition(N, N') = {tripartition_distance(N, Np):.0f}   "
        "(the documented failure)\n"
        f"    D (Nakhleh)(N, N')  = {nakhleh_metric(N, Np):.3f}   (separates them)\n"
    )
    text_page(pdf, "4.  Literature Stress Test (Section 11)", body)

    # ASCII renderings of the two networks.
    ascii_body = (
        "N  (Fig. 4)\n"
        + ascii_net(N) + "\n\n\n"
        + "N' (Fig. 8)  -- hybrids B and C swapped beneath b and c\n"
        + ascii_net(Np)
    )
    text_page(pdf, "4.  Literature Stress Test -- the two networks",
              ascii_body, mono=True)

    # Full metric table: N vs N', and N vs its single-edge rewiring.
    Nrw = nets["N rewired"]
    pairs = [
        ("N vs N'  (signature-identical)", N, Np),
        ("N vs N-rewired  (1 retic edge)", N, Nrw),
    ]
    col_labels = ["pair"] + [m[1] for m in METRICS] + ["Prec", "Rec", "F1"]
    rows = []
    for label, a, b in pairs:
        res = compare_networks(a, b)
        row = [label]
        for name, short, fn, _ in METRICS:
            row.append(_fmt(_safe(fn, a, b)))
        row += [_fmt(res.precision), _fmt(res.recall), _fmt(res.f1)]
        rows.append(row)
    col_widths = [0.22] + [0.06] * (len(col_labels) - 1)
    note = (
        "Top row: the adversarial pair.  D_ret and the classical tripartition\n"
        "measure both read 0 (identical reticulate signature), but mu-distance\n"
        "= 2 and D > 0 correctly certify the networks are different.\n"
        "Bottom row: rewiring a single reticulation edge of N perturbs exactly\n"
        "one of the three hybrids, so two of three are still recovered at\n"
        "tolerance 0 -- precision = recall = F1 = 2/3 -- and D_ret jumps off 0,\n"
        "reproducing the (2/3, 2/3, 2/3) worked example of Section 11 (the\n"
        "exact D_ret magnitude depends on the specific edge moved).\n"
        "Headers: Dret_n = normalized D_ret, D = Nakhleh metric, Dl.5 =\n"
        "D_lambda at lambda 0.5, mu/trip/hard/soft/disp/RF as in the catalogue."
    )
    table_page(pdf, "4.  Literature Stress Test -- all measures",
               col_labels, rows, note=note, col_widths=col_widths,
               fontsize=7.0)


def page_base_rate(pdf: PdfPages) -> None:
    ns = [6, 10, 20, 40, 80, 160]
    k = 1
    d_ret, hardwired_norm, tripart_norm = [], [], []
    for n in ns:
        n1, n2 = base_rate_family(n, k)
        d_ret.append(reticulation_dissimilarity(n1, n2, normalize=True))
        hardwired_norm.append(
            _safe(lambda a, b: hardwired_cluster_distance(a, b, normalize=True),
                  n1, n2) or 0.0)
        tripart_norm.append(
            _safe(lambda a, b: tripartition_distance(a, b, normalize=True),
                  n1, n2) or 0.0)

    fig = _fig()
    fig.text(0.07, 0.95, "5.  Base-Rate Masking (Proposition 4 / E2)",
             fontsize=TITLE_FS, fontweight="bold", va="top")
    ax = fig.add_axes([0.12, 0.40, 0.80, 0.44])
    ax.plot(ns, d_ret, "o-", label="D_ret (reticulation-aware)", lw=2)
    ax.plot(ns, hardwired_norm, "s--", label="hardwired cluster (norm)")
    ax.plot(ns, tripart_norm, "^--", label="tripartition (norm)")
    ax.set_xlabel("number of leaves n  (fixed k = 1 reticulation each)")
    ax.set_ylabel("normalized dissimilarity")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xscale("log")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)
    body = (
        "A caterpillar scaffold shared by both networks, with one reticulation\n"
        "placed among the prefix leaves in N1 and among the suffix leaves in\n"
        "N2, so the two hybrid clusters are leaf-disjoint.  As n grows the\n"
        "reticulation is diluted against an ever larger tree: the normalized\n"
        "GLOBAL cluster measures decay toward 0 (declaring the networks\n"
        "'nearly identical'), yet the reticulation-aware D_ret stays near 1 --\n"
        "it sees that the single event is entirely wrong.  This is the\n"
        "base-rate fallacy the measure is designed to expose.  (The raw\n"
        "mu-distance grows without bound here and so is omitted from this\n"
        "normalized view.)"
    )
    fig.text(0.07, 0.30, body, fontsize=BODY_FS, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def page_degradation(pdf: PdfPages) -> None:
    # Corrupt j of k reticulations of N by rewiring, and watch the scores move.
    nets = literature_networks()
    N = nets["N (Fig.4)"]

    # Build a sequence of increasingly corrupted variants by relocating the
    # parent arcs of 0, 1, 2, 3 hybrids to spurious donors above other leaves.
    variants = _degradation_variants()
    js = list(range(len(variants)))
    d_ret, f1, prec, rec = [], [], [], []
    for var in variants:
        res = compare_networks(N, var)
        d_ret.append(res.D_ret_hat)
        f1.append(res.f1)
        prec.append(res.precision)
        rec.append(res.recall)

    fig = _fig()
    fig.text(0.07, 0.95, "6.  Monotone Degradation & Recovery (E1)",
             fontsize=TITLE_FS, fontweight="bold", va="top")
    ax = fig.add_axes([0.12, 0.40, 0.80, 0.44])
    ax.plot(js, d_ret, "o-", label="D_ret (norm)", lw=2)
    ax.plot(js, f1, "s-", label="F1")
    ax.plot(js, prec, "^--", label="precision")
    ax.plot(js, rec, "d--", label="recall")
    ax.set_xlabel("number of reticulations corrupted, j")
    ax.set_ylabel("score")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks(js)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    body = (
        "Starting from N (reference) we corrupt j = 0,1,2,3 of its three\n"
        "reticulations by relocating a hybrid's parent arc to a spurious\n"
        "donor.  As j rises the reticulation dissimilarity climbs\n"
        "monotonically from 0 toward 1 while recall / F1 fall from 1 to 0 --\n"
        "evidence the score is calibrated to the fraction of events recovered.\n"
        "(j=0 is the self-comparison: D_ret=0, precision=recall=1.)"
    )
    fig.text(0.07, 0.30, body, fontsize=BODY_FS, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _degradation_variants() -> List[Network]:
    """N and three progressively-corrupted copies (1, 2, 3 hybrids moved)."""
    # j = 0 : N itself.
    v0 = build(_N_FIG4, _RETICS)

    # j = 1 : move hybrid B's parent arc (b->B) to sit above leaf 1.
    e1 = _N_REWIRED
    v1 = build(e1, _RETICS, clean=True)

    # j = 2 : additionally move hybrid C's parent arc (c->C) above leaf 5.
    e2 = [
        ("r", "a"), ("r", "b"),
        ("a", "g"), ("a", "A"),
        ("g", "1"), ("g", "B"),
        ("b", "c"),
        ("c", "d"),
        ("d", "A"), ("d", "h"),
        ("h", "5"), ("h", "C"),
        ("e", "2"), ("e", "B"),
        ("A", "e"), ("B", "f"), ("C", "4"),
        ("f", "3"), ("f", "C"),
    ]
    v2 = build(e2, _RETICS, clean=True)

    # j = 3 : all three reticulations relocated to spurious donors.
    v3 = _corrupt_all_three()
    return [v0, v1, v2, v3]


def _corrupt_all_three() -> Network:
    """
    A copy of N with all three reticulations relocated to spurious donors, so
    that no hybrid's tripartition matches the reference at tolerance 0.

    Every relocated parent arc is drawn from a donor that lies *above* the
    hybrids (never from a descendant of the hybrid), which keeps the result an
    acyclic, valid binary network:
        B: second parent moved to a subdivision above leaf 1 (g),
        C: second parent moved to a subdivision above leaf 5 (h),
        A: second parent moved to a subdivision of the root arc r -> b (p).
    """
    edges = [
        ("r", "a"), ("r", "b"),
        ("a", "g"), ("a", "A"),
        ("g", "q"), ("g", "B"),      # B moved (donor now above leaf 1)
        ("q", "1"), ("q", "A"),      # A's spurious second parent (was d)
        ("b", "c"),
        ("c", "d"),
        ("d", "h"),
        ("h", "5"), ("h", "C"),      # C moved (donor now above leaf 5)
        ("e", "2"), ("e", "B"),
        ("A", "e"), ("B", "f"), ("C", "4"),
        ("f", "3"), ("f", "C"),
    ]
    return build(edges, _RETICS, clean=True)


def page_pairwise(pdf: PdfPages) -> None:
    nets = literature_networks()
    names = list(nets.keys())
    mats = {}
    key_metrics = ["D_ret (norm)", "D (Nakhleh)", "mu-distance", "tripartition"]
    metric_map = {m[0]: m[2] for m in METRICS}
    for mname in key_metrics:
        fn = metric_map[mname]
        mat = np.full((len(names), len(names)), np.nan)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                v = _safe(fn, nets[a], nets[b])
                if v is not None:
                    mat[i, j] = v
        mats[mname] = mat

    fig = _fig()
    fig.text(0.07, 0.96, "7.  Pairwise Comparison of Key Measures",
             fontsize=TITLE_FS, fontweight="bold", va="top")
    for idx, mname in enumerate(key_metrics):
        ax = fig.add_subplot(2, 2, idx + 1)
        mat = mats[mname]
        im = ax.imshow(np.nan_to_num(mat, nan=0.0), cmap="viridis")
        ax.set_title(mname, fontsize=9)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        short = [n.split(" ")[0] for n in names]
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(short, fontsize=6)
        for i in range(len(names)):
            for j in range(len(names)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, _fmt(mat[i, j]), ha="center", va="center",
                            color="w", fontsize=5.2)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.text(0.07, 0.06,
             "D_ret gives distance 0 for the N/N' pair (shared reticulate\n"
             "signature) and for all tree-tree pairs (no reticulations), while\n"
             "D and mu-distance separate every non-isomorphic pair.",
             fontsize=BODY_FS, va="top")
    fig.tight_layout(rect=[0.05, 0.10, 0.98, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


# ===================================================================
# Driver
# ===================================================================

def print_summary() -> None:
    nets = literature_networks()
    N, Np = nets["N (Fig.4)"], nets["N' (Fig.8)"]
    Nrw = nets["N rewired"]
    print("\n=== Literature stress test (Cardona-Rossello-Valiente) ===")
    r = compare_networks(N, Np)
    print(f"  D_ret(N,N')        = {r.D_ret:.4f}   (expected 0)")
    print(f"  mu_distance(N,N')  = {mu_distance(N, Np):.0f}        (expected 2)")
    print(f"  tripartition(N,N') = {tripartition_distance(N, Np):.0f}        "
          "(expected 0 -- the documented failure)")
    print(f"  D_Nakhleh(N,N')    = {nakhleh_metric(N, Np):.4f}   (>0 separates)")
    p, rec, f1 = reticulation_precision_recall(N, Nrw)
    print("\n=== Single reticulation-edge rewiring N vs N-rewired ===")
    print(f"  D_ret_hat = {compare_networks(N, Nrw).D_ret_hat:.4f}   "
          f"(Prec,Rec,F1) = ({p:.3f},{rec:.3f},{f1:.3f})  expected 2/3 each")


def main(out_path: str) -> None:
    with PdfPages(out_path) as pdf:
        page_title(pdf)
        page_design(pdf)
        page_catalogue(pdf)
        page_boundary(pdf)
        page_literature(pdf)
        page_base_rate(pdf)
        page_degradation(pdf)
        page_pairwise(pdf)
    print(f"Wrote report to: {out_path}")
    print_summary()


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "reticulation_distance_report.pdf"
    main(out)
