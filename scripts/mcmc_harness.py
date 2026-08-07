#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##############################################################################

"""
Correctness + per-iteration-timing harness for the three PhyNetPy MCMC
samplers (gene trees, sequences, biallelic SNP markers) on a *known-truth*
6-taxon, 1-reticulation species network.

Why this harness exists
-----------------------
The three Bayesian samplers -- the multispecies network coalescent on
gene-tree topologies, co-estimation from DNA alignments, and the SNAPP-style
biallelic-marker likelihood -- are all supposed to converge on the *same* true
network when handed data simulated from it.  This module builds one canonical
ground-truth network, simulates each data type from it under the model each
sampler assumes, runs the sampler for a fixed budget, and reports two things
the overhaul cares about:

* **Accuracy**: does the MAP / best network recover the true topology,
  reticulation, and inheritance probability?  Measured with the network
  distances in :mod:`phynetpy.GraphUtils` (mu-distance, tripartition
  distance) plus explicit clade / gamma checks -- not just a single clade
  string match.
* **Speed**: milliseconds per iteration, so the performance work can be
  monitored against a stable reference.

The ground-truth network is constructed from explicit node *heights* so it is
guaranteed ultrametric (every root-to-tip path has the same length), which is
what the coalescent simulators and the timed MSNC likelihood assume.

Usage
-----
    py scripts/mcmc_harness.py --which gt seq          # quick baseline
    py scripts/mcmc_harness.py --which gt --iters 50000 --burnin 10000
    py scripts/mcmc_harness.py --which all --loci 20 --sites 500
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Optional

from phynetpy.Network import Network, Node, Edge
from phynetpy.criteria import Bayesian, Likelihood
from phynetpy.infer import JC69, MCMCSeqPriors, MCMC_GTPriors, infer, simulate
from phynetpy.models import MSC


# ======================================================================
# Ground-truth 6-taxon, 1-reticulation species network
# ======================================================================
#
# Heights (expected substitutions per site, present = 0), verified ultrametric:
#
#   leaves A B C D E F : 0.00
#   AB (mrca A,B)      : 0.04
#   CD (mrca C,D)      : 0.04
#   EF (mrca E,F)      : 0.04
#   H  (reticulation)  : 0.06   parent of the (C,D) clade
#   P1                 : 0.10   parents AB and H  (gamma major, 0.65)
#   P2                 : 0.12   parents EF and H  (gamma minor, 0.35)
#   R  (root)          : 0.15
#
# Every root->leaf path sums to 0.15, and both paths to H (via P1 and via P2)
# put it at height 0.06, so the network is ultrametric including the hybrid.

_TRUE_HEIGHTS: dict[str, float] = {
    "A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0, "F": 0.0,
    "AB": 0.04, "CD": 0.04, "EF": 0.04,
    "H": 0.06, "P1": 0.10, "P2": 0.12, "R": 0.15,
}

# (parent, child, gamma-or-None).  gamma is only set on the two hybrid edges.
_TRUE_EDGES: list[tuple[str, str, Optional[float]]] = [
    ("R", "P1", None),
    ("R", "P2", None),
    ("P1", "AB", None),
    ("P2", "EF", None),
    ("AB", "A", None),
    ("AB", "B", None),
    ("CD", "C", None),
    ("CD", "D", None),
    ("EF", "E", None),
    ("EF", "F", None),
    ("P1", "H", 0.65),   # major hybrid edge
    ("P2", "H", 0.35),   # minor hybrid edge
    ("H", "CD", None),
]

TAXA = ("A", "B", "C", "D", "E", "F")
MAPPING: dict[str, list[str]] = {sp: [sp] for sp in TAXA}
TRUE_GAMMA_MAJOR = 0.65

# Clades that the true *species tree backbone* induces and that a correct MAP
# network must contain (reticulation-aware).  (C,D), (E,F), (A,B) are the
# unambiguous cherries; the hybrid origin of (C,D) is checked separately.
TRUE_CLADES = [{"A", "B"}, {"C", "D"}, {"E", "F"}]


def build_true_network() -> Network:
    """Construct the canonical 6-taxon, 1-reticulation ground-truth network.

    Branch lengths are derived from :data:`_TRUE_HEIGHTS` so the result is
    ultrametric by construction.  The single reticulation ``H`` is the parent
    of the ``(C, D)`` clade and inherits from ``P1`` (gamma 0.65) and ``P2``
    (gamma 0.35).

    Returns:
        The true species :class:`~phynetpy.Network.Network`.
    """
    nodes: dict[str, Node] = {
        name: Node(name, is_reticulation=(name == "H"))
        for name in _TRUE_HEIGHTS
    }
    net = Network()
    net.add_nodes(*nodes.values())

    edges = []
    for parent, child, gamma in _TRUE_EDGES:
        length = _TRUE_HEIGHTS[parent] - _TRUE_HEIGHTS[child]
        if length <= 0:
            raise ValueError(
                f"non-positive branch {parent}->{child} (len={length}); "
                "check _TRUE_HEIGHTS."
            )
        if gamma is None:
            edges.append(Edge(nodes[parent], nodes[child], length=length))
        else:
            edges.append(
                Edge(nodes[parent], nodes[child], length=length, gamma=gamma)
            )
    net.add_edges(edges)
    _assert_ultrametric(net)
    return net


def _assert_ultrametric(net: Network, tol: float = 1e-9) -> None:
    """Raise if root->leaf path lengths disagree (min over hybrid parents)."""
    # Compute height of each node as max path length down to a leaf.
    def height(node: Node) -> float:
        kids = net.get_children(node)
        if not kids:
            return 0.0
        best = None
        for c in kids:
            e = net.get_edge(node, c)
            e = e[0] if isinstance(e, list) else e
            h = height(c) + float(e.get_length())
            best = h if best is None else max(best, h)
        return best

    leaf_depths = []

    def descend(node: Node, acc: float) -> None:
        kids = net.get_children(node)
        if not kids:
            leaf_depths.append(acc)
            return
        for c in kids:
            e = net.get_edge(node, c)
            e = e[0] if isinstance(e, list) else e
            descend(c, acc + float(e.get_length()))

    descend(net.root(), 0.0)
    lo, hi = min(leaf_depths), max(leaf_depths)
    if hi - lo > tol:
        raise AssertionError(
            f"true network not ultrametric: root->leaf depths in [{lo}, {hi}]"
        )


# ======================================================================
# Accuracy metrics vs ground truth
# ======================================================================

def _descendant_leaves(net: Network, node: Node) -> frozenset:
    kids = net.get_children(node)
    if not kids:
        return frozenset({node.label})
    acc: set = set()
    for c in kids:
        acc |= _descendant_leaves(net, c)
    return frozenset(acc)


def _all_clades(net: Network) -> set:
    leaves = {n.label for n in net.get_leaves()}
    clades = set()
    for v in net.V():
        ds = _descendant_leaves(net, v)
        if 1 < len(ds) < len(leaves):
            clades.add(ds)
    return clades


def _has_clade(net: Network, clade: set) -> bool:
    target = frozenset(clade)
    return any(_descendant_leaves(net, v) == target for v in net.V())


def _num_reticulations(net: Network) -> int:
    return sum(1 for v in net.V() if v.is_reticulation())


def _major_gamma(net: Network) -> Optional[float]:
    gammas = []
    for v in net.V():
        if v.is_reticulation():
            for e in net.in_edges(v):
                g = e.get_gamma()
                if g is not None:
                    gammas.append(float(g))
    return max(gammas) if gammas else None


@dataclass
class AccuracyReport:
    """Accuracy of an inferred network vs the ground truth."""
    clades_recovered: dict[str, bool]
    all_clades_recovered: bool
    num_reticulations: int
    true_num_reticulations: int
    gamma_major: Optional[float]
    gamma_error: Optional[float]
    mu_distance: Optional[int]
    tripartition_distance: Optional[float]

    def __str__(self) -> str:
        lines = []
        for name, ok in self.clades_recovered.items():
            lines.append(f"    clade {name}: {'RECOVERED' if ok else 'missed'}")
        lines.append(
            f"    all backbone clades : "
            f"{'RECOVERED' if self.all_clades_recovered else 'missed'}"
        )
        lines.append(
            f"    reticulations       : {self.num_reticulations} "
            f"(true {self.true_num_reticulations})"
        )
        if self.gamma_major is not None:
            lines.append(
                f"    gamma (major)       : {self.gamma_major:.3f} "
                f"(true {TRUE_GAMMA_MAJOR:.3f}, |err|={self.gamma_error:.3f})"
            )
        else:
            lines.append("    gamma (major)       : (no reticulation inferred)")
        lines.append(f"    mu-distance         : {self.mu_distance}")
        lines.append(f"    tripartition dist.  : {self.tripartition_distance}")
        return "\n".join(lines)


def score_accuracy(inferred: Network, true_net: Network) -> AccuracyReport:
    """Compare an inferred network to the ground truth on several metrics."""
    from phynetpy.GraphUtils import mu_distance, tripartition_distance

    clades = {
        "".join(sorted(c)): _has_clade(inferred, c) for c in TRUE_CLADES
    }
    g_major = _major_gamma(inferred)
    g_err = None if g_major is None else abs(g_major - TRUE_GAMMA_MAJOR)

    mu = tri = None
    try:
        mu = mu_distance(inferred, true_net)
    except Exception:
        pass
    try:
        tri = tripartition_distance(inferred, true_net, normalize=True)
    except Exception:
        pass

    return AccuracyReport(
        clades_recovered=clades,
        all_clades_recovered=_all_clades(true_net).issubset(_all_clades(inferred)),
        num_reticulations=_num_reticulations(inferred),
        true_num_reticulations=_num_reticulations(true_net),
        gamma_major=g_major,
        gamma_error=g_err,
        mu_distance=mu,
        tripartition_distance=tri,
    )


# ======================================================================
# Result container
# ======================================================================

@dataclass
class RunResult:
    """Timing + accuracy of one sampler run."""
    label: str
    num_iter: int
    wall_time_sec: float
    best_score: float
    accuracy: AccuracyReport
    acceptance_rate: Optional[float] = None
    extra: dict = field(default_factory=dict)

    @property
    def ms_per_iter(self) -> float:
        return 1000.0 * self.wall_time_sec / max(1, self.num_iter)

    def __str__(self) -> str:
        head = (
            f"[{self.label}]  {self.num_iter} it in {self.wall_time_sec:.2f}s  "
            f"= {self.ms_per_iter:.3f} ms/it  best_score={self.best_score:.3f}"
        )
        if self.acceptance_rate is not None:
            head += f"  acc={self.acceptance_rate:.3f}"
        body = head + "\n" + str(self.accuracy)
        ms_table = self.extra.get("model_selection")
        if ms_table:
            body += "\n" + ms_table
        return body


# ======================================================================
# Runners for each data type
# ======================================================================

def run_gt(true_net: Network, *, loci: int, sites: int, iters: int,
           burnin: int, thin: int, seed: int,
           max_reticulations: int = 2) -> RunResult:
    """Simulate gene trees on ``true_net`` and sample the posterior."""
    gts = simulate(MSC(theta=0.02), true_net, n=loci, data="gene_trees",
                   mapping=MAPPING, seed=seed)
    t0 = time.perf_counter()
    res = infer(
        gts,
        model=MSC(theta=0.02),
        criterion=Bayesian(objective=Likelihood(), prior=MCMC_GTPriors(),
                           chain_length=iters, burnin=burnin,
                           sample_freq=thin, seed=seed),
        max_reticulations=max_reticulations,
    )
    dt = time.perf_counter() - t0
    return RunResult(
        label="GT ", num_iter=iters, wall_time_sec=dt,
        best_score=res.score,
        accuracy=score_accuracy(res.best, true_net),
        acceptance_rate=res.acceptance_rate,
    )


def run_seq(true_net: Network, *, loci: int, sites: int, iters: int,
            burnin: int, thin: int, seed: int,
            max_reticulations: int = 2, max_level: "int | None" = None,
            warm_start: bool = True, gt_iters: int = 6000) -> RunResult:
    """Simulate DNA alignments on ``true_net`` and co-estimate from them.

    Only the data axis changes relative to :func:`run_gt`: an ``Alignment``
    instead of ``GeneTrees``, with the same criterion.

    ``warm_start`` (default) bootstraps the starting network with a fast
    gene-tree search so the coupled chain begins from a reticulation-bearing
    network it can refine, instead of a plain tree it would never leave (the
    joint-mode barrier).

    ``max_level`` (e.g. ``1``) restricts the sampler to networks of at most that
    level; reticulation-adding / relocating proposals that would exceed it
    self-reject before the expensive coupled scoring, which both bounds the
    state space and skips the displayed-tree combinatorial blow-up.
    """
    alignment = simulate(MSC(theta=0.02), true_net, n=loci, data="alignment",
                         mapping=MAPPING, seq_length=sites,
                         substitution_model=JC69(), seed=seed)
    t0 = time.perf_counter()
    res = infer(
        alignment,
        model=MSC(theta=0.02),
        criterion=Bayesian(
            objective=Likelihood(),
            prior=MCMCSeqPriors(max_reticulations=max_reticulations,
                                max_level=max_level),
            chain_length=iters, burnin=burnin, sample_freq=thin, seed=seed,
        ),
        warm_start=warm_start,
        warm_start_kwargs={"gt_iters": gt_iters},
    )
    dt = time.perf_counter() - t0
    return RunResult(
        label="SEQ", num_iter=iters, wall_time_sec=dt,
        best_score=res.score,
        accuracy=score_accuracy(res.best, true_net),
        extra={"map_theta": getattr(res, "map_theta", None),
               "model_selection": _format_model_selection(res)},
    )


def _format_model_selection(res) -> str:
    """Render the AIC/BIC-by-reticulation table for a sequence run.

    Reports, per reticulation count sampled, the best log likelihood, parameter
    count and information criteria with deltas to the best model -- so an extra
    reticulation is only "worth it" when its dAIC/dBIC stay near zero.
    """
    try:
        rows = res.model_selection_by_reticulation()
    except Exception:
        return ""
    if not rows:
        return ""
    out = ["    model selection (AIC/BIC by reticulation count):",
           "      r   best_logL     k      AIC     dAIC      BIC     dBIC"]
    for row in rows:
        out.append(
            f"      {int(row['num_reticulations']):<2}"
            f"{row['log_likelihood']:>11.2f}"
            f"{int(row['k']):>6}"
            f"{row['AIC']:>9.1f}"
            f"{row.get('dAIC', float('nan')):>9.1f}"
            f"{row.get('BIC', float('nan')):>9.1f}"
            f"{row.get('dBIC', float('nan')):>9.1f}"
        )
    best = min(rows, key=lambda r: r["AIC"])
    out.append(
        f"    -> AIC prefers {int(best['num_reticulations'])} reticulation(s)"
    )
    return "\n".join(out)


def run_snp(true_net: Network, *, sites: int, iters: int, burnin: int,
            thin: int, seed: int, u: float = 1.0, v: float = 1.0,
            coal: float = 0.005, max_reticulations: int = 2) -> RunResult:
    """Simulate biallelic SNP data on ``true_net`` and sample the posterior.

    The mutation rates ``u`` / ``v`` and the coalescent rate live on the model
    axis, so the same ``MSC`` object configures the simulator and the sampler --
    no NEXUS round-trip in between.
    """
    samples = {leaf.label: 1 for leaf in true_net.get_leaves()}
    model = MSC(u=u, v=v, coal=coal)
    markers = simulate(model, true_net, n=sites, data="markers",
                       mapping=MAPPING, samples=samples, seed=seed)

    t0 = time.perf_counter()
    res = infer(
        markers,
        model=model,
        criterion=Bayesian(objective=Likelihood(), chain_length=iters,
                           burnin=burnin, sample_freq=thin, seed=seed),
        max_reticulations=max_reticulations,
    )
    dt = time.perf_counter() - t0
    return RunResult(
        label="SNP", num_iter=iters, wall_time_sec=dt, best_score=res.score,
        accuracy=score_accuracy(res.best, true_net),
    )


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", nargs="+", default=["gt", "seq"],
                    choices=["gt", "seq", "snp", "all"],
                    help="which samplers to run")
    ap.add_argument("--loci", type=int, default=10)
    ap.add_argument("--sites", type=int, default=400)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--burnin", type=int, default=1000)
    ap.add_argument("--thin", type=int, default=20)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max-retic", type=int, default=2)
    ap.add_argument("--max-level", type=int, default=None,
                    help="cap network level (e.g. 1 for level-1 / galled "
                         "networks); rejects above-level proposals before "
                         "scoring. Default: no cap.")
    ap.add_argument("--no-warm-start", action="store_true",
                    help="disable the gene-tree warm start for the seq chain")
    ap.add_argument("--gt-iters", type=int, default=6000,
                    help="gene-tree bootstrap iterations for the seq warm start")
    args = ap.parse_args()

    which = set(args.which)
    if "all" in which:
        which = {"gt", "seq", "snp"}

    true_net = build_true_network()
    print(f"True network : {true_net.newick()}")
    print(f"Ground truth : 6 taxa, 1 reticulation, gamma_major={TRUE_GAMMA_MAJOR}")
    print(f"Budget       : {args.iters} iters (burn-in {args.burnin}, "
          f"thin {args.thin}), {args.loci} loci x {args.sites} sites\n")

    results: list[RunResult] = []
    if "gt" in which:
        results.append(run_gt(true_net, loci=args.loci, sites=args.sites,
                              iters=args.iters, burnin=args.burnin,
                              thin=args.thin, seed=args.seed,
                              max_reticulations=args.max_retic))
        print(results[-1]); print()
    if "seq" in which:
        results.append(run_seq(true_net, loci=args.loci, sites=args.sites,
                               iters=args.iters, burnin=args.burnin,
                               thin=args.thin, seed=args.seed,
                               max_reticulations=args.max_retic,
                               max_level=args.max_level,
                               warm_start=not args.no_warm_start,
                               gt_iters=args.gt_iters))
        print(results[-1]); print()
    if "snp" in which:
        results.append(run_snp(true_net, sites=args.sites, iters=args.iters,
                               burnin=args.burnin, thin=args.thin,
                               seed=args.seed,
                               max_reticulations=args.max_retic))
        print(results[-1]); print()

    print("=" * 60)
    print("SUMMARY (ms/iter | topology | mu-dist)")
    for r in results:
        topo = "OK" if r.accuracy.all_clades_recovered else "--"
        print(f"  {r.label}: {r.ms_per_iter:8.3f} ms/it   topo={topo}   "
              f"mu={r.accuracy.mu_distance}")


if __name__ == "__main__":
    main()
