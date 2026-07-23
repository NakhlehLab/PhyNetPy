"""Weekend stress-test: one long MCMC_SEQ chain on a 10-taxon, 1-reticulation
species network co-estimated from 50 loci x 1000-site DNA alignments.

The chain runs for ``--iters`` proposals (default 3,000,000) with a cooperative
``control`` check-in every ``--milestone`` iterations (default 500,000).  At each
milestone we build a throwaway result view over the samples gathered so far and
report per-iter timing, acceptance, the reticulation-count posterior,
convergence diagnostics (ESS / HPD), the gamma estimate, an AIC/BIC
model-selection table, and accuracy vs the known truth (mu-distance,
tripartition distance).  Everything is echoed to ``runs/weekend_10t/stats.txt``
and the trace / sampled networks are flushed to disk at every milestone so a
crash loses at most one milestone's worth of work.

Calibration: pass a small ``--iters`` (e.g. 3000) with ``--milestone`` >= iters
to just measure warm-start cost and steady-state ms/it before committing to the
full run.

Run in the background; tail ``runs/weekend_10t/stats.txt`` for milestone reports.
"""
from __future__ import annotations

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phynetpy.Network import Network, Node, Edge
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy._mcmc_seq import MCMCSeqResult
from phynetpy._chain_analysis import summarize


OUT_DIR = os.path.join("runs", "weekend_10t")


# ======================================================================
# Ground-truth 10-taxon, 1-reticulation species network (ultrametric)
# ======================================================================
#
# Heights (expected substitutions/site; leaves = 0), verified ultrametric:
#
#   L0..L9                : 0.00
#   AB CD EF GH IJ        : 0.03   (five cherries)
#   H  (reticulation)     : 0.05   parent of the (L8,L9) cherry
#   G1 = (AB,CD)          : 0.06
#   G2 = (EF,GH)          : 0.06
#   P1                    : 0.09   parents G1 and H (gamma major, 0.70)
#   P2                    : 0.11   parents G2 and H (gamma minor, 0.30)
#   R  (root)             : 0.15
#
# Every root->leaf path sums to 0.15, including both routes into the hybrid H.

_TRUE_HEIGHTS: dict[str, float] = {
    "L0": 0.0, "L1": 0.0, "L2": 0.0, "L3": 0.0, "L4": 0.0,
    "L5": 0.0, "L6": 0.0, "L7": 0.0, "L8": 0.0, "L9": 0.0,
    "AB": 0.03, "CD": 0.03, "EF": 0.03, "GH": 0.03, "IJ": 0.03,
    "H": 0.05, "G1": 0.06, "G2": 0.06, "P1": 0.09, "P2": 0.11, "R": 0.15,
}

# (parent, child, gamma-or-None); gamma set only on the two hybrid edges.
_TRUE_EDGES: list[tuple[str, str, "float | None"]] = [
    ("R", "P1", None),
    ("R", "P2", None),
    ("P1", "G1", None),
    ("P2", "G2", None),
    ("G1", "AB", None),
    ("G1", "CD", None),
    ("G2", "EF", None),
    ("G2", "GH", None),
    ("P1", "H", 0.70),   # major hybrid edge
    ("P2", "H", 0.30),   # minor hybrid edge
    ("H", "IJ", None),
    ("AB", "L0", None), ("AB", "L1", None),
    ("CD", "L2", None), ("CD", "L3", None),
    ("EF", "L4", None), ("EF", "L5", None),
    ("GH", "L6", None), ("GH", "L7", None),
    ("IJ", "L8", None), ("IJ", "L9", None),
]

TAXA = tuple(f"L{i}" for i in range(10))
MAPPING: dict[str, list[str]] = {sp: [sp] for sp in TAXA}
TRUE_GAMMA_MAJOR = 0.70
TRUE_NUM_RETIC = 1


def build_true_network() -> Network:
    """Construct the canonical 10-taxon, 1-reticulation ground-truth network.

    Branch lengths are derived from :data:`_TRUE_HEIGHTS` so the result is
    ultrametric by construction.  The single reticulation ``H`` is the parent of
    the ``(L8, L9)`` cherry and inherits from ``P1`` (gamma 0.70) and ``P2``
    (gamma 0.30).
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
                f"non-positive branch {parent}->{child} (len={length})."
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
    leaf_depths: list[float] = []

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
# Accuracy vs ground truth (topology-general)
# ======================================================================

def _num_reticulations(net: Network) -> int:
    return sum(1 for v in net.V() if v.is_reticulation())


def _major_gamma(net: Network) -> "float | None":
    gammas = []
    for v in net.V():
        if v.is_reticulation():
            for e in net.in_edges(v):
                g = e.get_gamma()
                if g is not None:
                    gammas.append(float(g))
    return max(gammas) if gammas else None


def _score_accuracy(inferred: Network, true_net: Network) -> dict:
    from phynetpy.GraphUtils import mu_distance, tripartition_distance

    mu = tri = None
    try:
        mu = mu_distance(inferred, true_net)
    except Exception:
        pass
    try:
        tri = tripartition_distance(inferred, true_net, normalize=True)
    except Exception:
        pass
    g = _major_gamma(inferred)
    return {
        "num_reticulations": _num_reticulations(inferred),
        "gamma_major": g,
        "gamma_error": None if g is None else abs(g - TRUE_GAMMA_MAJOR),
        "mu_distance": mu,
        "tripartition_distance": tri,
    }


# ======================================================================
# Reporting
# ======================================================================

def _emit(msg: str, fh) -> None:
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def _milestone_report(it, prog, sampler, t0, fh, tag=""):
    samples = list(prog["samples"])
    elapsed = time.perf_counter() - t0
    ms_it = 1000.0 * elapsed / max(1, it)
    view = MCMCSeqResult(
        map_network=None,
        map_log_posterior=prog["map_log_posterior"],
        map_theta=sampler.theta,
        samples=samples,
        acceptance_rate=prog["acceptance_rate"],
        num_iterations=it,
        num_leaves=len(sampler.mapping),
        total_sites=sampler._total_sites(),
    )
    _emit("\n" + "=" * 72, fh)
    _emit(f"[CHECK-IN {tag}] iter {it:,}  elapsed {elapsed/3600:.2f} h  "
          f"{ms_it:.2f} ms/it  acc={prog['acceptance_rate']:.3f}", fh)
    _emit(f"  samples so far: {len(samples)}   "
          f"MAP logP={prog['map_log_posterior']:.2f}", fh)

    retic_post = view.reticulation_posterior()
    _emit(f"  reticulation-count posterior: "
          f"{ {k: round(v, 3) for k, v in retic_post.items()} }", fh)

    try:
        summ = view.summary()
        _emit("  convergence (min ESS = {:.0f}, converged={}):".format(
            summ.min_ess, summ.converged), fh)
        for name in ("posterior", "likelihood", "theta",
                     "reticulationCount", "gammaMajor"):
            if name in summ.parameters:
                p = summ.parameters[name]
                ess = "nan" if p.ess != p.ess else f"{p.ess:.0f}"
                _emit(f"    {name:<18} mean={p.mean:<12.5g} "
                      f"95%HPD=[{p.lower_hpd:.4g}, {p.upper_hpd:.4g}] "
                      f"ESS={ess}", fh)
    except Exception as e:
        _emit(f"  (summary unavailable: {e})", fh)

    gammas = [s.gamma_major for s in samples if s.gamma_major is not None]
    if gammas:
        gp = summarize("gammaMajor", gammas, step_size=1)
        _emit(f"  gamma_major: posterior-mean={gp.mean:.3f} "
              f"95%HPD=[{gp.lower_hpd:.3f}, {gp.upper_hpd:.3f}] "
              f"(true {TRUE_GAMMA_MAJOR}); ESS="
              f"{'nan' if gp.ess != gp.ess else f'{gp.ess:.0f}'}", fh)

    rows = view.model_selection_by_reticulation()
    if rows:
        _emit("  AIC/BIC by reticulation count:", fh)
        _emit("    r  best_logL       k        AIC     dAIC        BIC     dBIC",
              fh)
        for row in rows:
            _emit(f"    {int(row['num_reticulations']):<2}"
                  f"{row['log_likelihood']:>11.1f}{int(row['k']):>8}"
                  f"{row['AIC']:>11.1f}{row.get('dAIC', float('nan')):>9.1f}"
                  f"{row.get('BIC', float('nan')):>11.1f}"
                  f"{row.get('dBIC', float('nan')):>9.1f}", fh)

    try:
        view.write_log(os.path.join(OUT_DIR, "trace.log"))
        view.write_networks(os.path.join(OUT_DIR, "networks.trees"))
    except Exception as e:
        _emit(f"  (checkpoint write failed: {e})", fh)


def _checkpoint(it, prog, sampler, t0, fh):
    """Lightweight, frequent checkpoint + heartbeat.

    Flushes the trace / sampled networks gathered so far and appends a single
    progress line so an unexpected interruption (e.g. a Windows-Update reboot)
    loses at most one checkpoint interval, and so progress is always visible
    between the coarse milestone reports.
    """
    samples = list(prog["samples"])
    elapsed = time.perf_counter() - t0
    ms_it = 1000.0 * elapsed / max(1, it)
    gammas = [s.gamma_major for s in samples if s.gamma_major is not None]
    g = f"{gammas[-1]:.3f}" if gammas else "n/a"
    rets = [s.num_reticulations for s in samples]
    r = rets[-1] if rets else "n/a"
    _emit(f"[hb] iter {it:,}  {elapsed/3600:.2f} h  {ms_it:.1f} ms/it  "
          f"acc={prog['acceptance_rate']:.3f}  n_samp={len(samples)}  "
          f"cur_r={r}  cur_gamma={g}  MAPlogP={prog['map_log_posterior']:.1f}",
          fh)
    try:
        view = MCMCSeqResult(
            map_network=None, map_log_posterior=prog["map_log_posterior"],
            map_theta=sampler.theta, samples=samples,
            acceptance_rate=prog["acceptance_rate"], num_iterations=it,
            num_leaves=len(sampler.mapping), total_sites=sampler._total_sites())
        view.write_log(os.path.join(OUT_DIR, "trace.log"))
        view.write_networks(os.path.join(OUT_DIR, "networks.trees"))
    except Exception as e:
        _emit(f"  (checkpoint write failed: {e})", fh)


def main() -> None:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--loci", type=int, default=50)
    ap.add_argument("--sites", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=3_000_000)
    ap.add_argument("--milestone", type=int, default=500_000)
    ap.add_argument("--checkpoint", type=int, default=25_000,
                    help="flush trace/networks + heartbeat every N iters so an "
                         "interruption (e.g. a Windows-Update reboot) loses at "
                         "most one interval and progress stays visible")
    ap.add_argument("--burnin", type=int, default=300_000)
    ap.add_argument("--thin", type=int, default=500)
    ap.add_argument("--gt-iters", type=int, default=10_000)
    ap.add_argument("--max-retic", type=int, default=2)
    ap.add_argument("--max-level", type=int, default=1,
                    help="cap network level (default 1: galled / level-1). "
                         "Rejects above-level proposals before scoring.")
    ap.add_argument("--seed", type=int, default=20260710,
                    help="per-chain MCMC/warm-start seed (vary across chains)")
    ap.add_argument("--data-seed", type=int, default=20260710,
                    help="seed for the simulated data; KEEP FIXED across chains "
                         "so every chain targets the same posterior (valid "
                         "cross-chain R-hat)")
    ap.add_argument("--outdir", type=str, default=OUT_DIR,
                    help="output directory for this chain's stats/trace/nets")
    a = ap.parse_args()

    OUT_DIR = a.outdir
    os.makedirs(OUT_DIR, exist_ok=True)
    fh = open(os.path.join(OUT_DIR, "stats.txt"), "w")

    true_net = build_true_network()
    _emit(f"True network : {true_net.newick()}", fh)
    _emit(f"Ground truth : 10 taxa, 1 reticulation, "
          f"gamma_major={TRUE_GAMMA_MAJOR}", fh)
    _emit(f"Budget       : {a.iters:,} iters (burn-in {a.burnin:,}, "
          f"thin {a.thin}); milestone every {a.milestone:,}", fh)
    _emit(f"Data         : {a.loci} loci x {a.sites} sites; max_retic="
          f"{a.max_retic}, max_level={a.max_level}; warm start "
          f"({a.gt_iters} GT iters)", fh)
    _emit(f"Seeds        : data_seed={a.data_seed} (shared), "
          f"chain_seed={a.seed} (per-chain)", fh)

    t_sim = time.perf_counter()
    data = simulate_multilocus(true_net, MAPPING, n_loci=a.loci,
                               seq_length=a.sites, theta=0.02, model=JC69(),
                               seed=a.data_seed)
    _emit(f"  (simulated data in {time.perf_counter() - t_sim:.1f} s)", fh)

    sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                       priors=MCMCSeqPriors(max_reticulations=a.max_retic,
                                            max_level=a.max_level))

    t0 = time.perf_counter()
    reported = {"last": 0, "hb": 0}
    # The control callback fires every ``check_every`` iters; drive it at the
    # (finer) checkpoint cadence and decide inside whether this tick is also a
    # full milestone report.
    tick = min(a.checkpoint, a.milestone)

    def control(prog):
        it = prog["iteration"]
        if it <= 0:
            return "continue"
        if it % a.milestone == 0 and it != reported["last"]:
            reported["last"] = it
            reported["hb"] = it
            _milestone_report(it, prog, sampler, t0, fh, tag=f"{it // 1000}k")
        elif it % tick == 0 and it != reported["hb"]:
            reported["hb"] = it
            _checkpoint(it, prog, sampler, t0, fh)
        return "continue"

    res = sampler.search(num_iter=a.iters, burn_in=a.burnin,
                         sample_freq=a.thin, seed=a.seed, warm_start=True,
                         warm_start_kwargs={"gt_iters": a.gt_iters},
                         control=control, check_every=tick)
    total = time.perf_counter() - t0

    res.write_log(os.path.join(OUT_DIR, "trace.log"))
    res.write_networks(os.path.join(OUT_DIR, "networks.trees"))
    acc = _score_accuracy(res.map_network, true_net)
    _emit("\n" + "#" * 72, fh)
    _emit(f"[FINAL] {a.iters:,} iters in {total/3600:.2f} h "
          f"({1000*total/a.iters:.2f} ms/it)  acc={res.acceptance_rate:.3f}", fh)
    _emit(f"  reticulation posterior: "
          f"{ {k: round(v, 3) for k, v in res.reticulation_posterior().items()} }",
          fh)
    ic = res.information_criteria()
    if ic:
        _emit(f"  MAP IC: r={ic['num_reticulations']} AIC={ic['AIC']:.1f} "
              f"BIC={ic['BIC']:.1f}", fh)
    gammas = [s.gamma_major for s in res.samples if s.gamma_major is not None]
    if gammas:
        gp = summarize("gammaMajor", gammas, step_size=1)
        _emit(f"  gamma_major posterior-mean={gp.mean:.3f} "
              f"95%HPD=[{gp.lower_hpd:.3f}, {gp.upper_hpd:.3f}] "
              f"(true {TRUE_GAMMA_MAJOR})", fh)
    _emit(f"  MAP reticulations={acc['num_reticulations']} (true "
          f"{TRUE_NUM_RETIC}); mu-distance={acc['mu_distance']}; "
          f"tripartition={acc['tripartition_distance']}", fh)
    _emit(f"  MAP logP={res.map_log_posterior:.2f}", fh)
    _emit(f"  MAP net: {res.map_network.newick()}", fh)
    _emit(f"  map topology (most-sampled): {res.map_topology()}", fh)
    fh.close()


if __name__ == "__main__":
    main()
