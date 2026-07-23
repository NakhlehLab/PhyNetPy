"""One continuous 1,000,000-iteration MCMC_SEQ run with a stats check-in every
100k iterations (and disk checkpoints so nothing is lost if it dies).

The check-in is done through the sampler's cooperative ``control`` hook: at each
100k milestone we build a throwaway result view over the samples gathered so
far and print convergence diagnostics (ESS, HPD), the reticulation-count
posterior, the inheritance-probability estimate, and the AIC/BIC model-selection
table -- then return "continue", so the same process runs straight through to
1M.  Everything is echoed to ``runs/million/stats.txt`` and the trace / networks
are flushed to disk at every milestone.

Run in the background; read ``runs/million/stats.txt`` (or the terminal file)
for the milestone reports.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np

from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy._mcmc_seq import MCMCSeqResult
from phynetpy._chain_analysis import summarize
from mcmc_harness import build_true_network, MAPPING, TRUE_GAMMA_MAJOR, score_accuracy


OUT_DIR = os.path.join("runs", "million")


def _emit(msg: str, fh):
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def _milestone_report(it, prog, sampler, true_net, t0, fh, tag=""):
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
    _emit("\n" + "=" * 70, fh)
    _emit(f"[CHECK-IN {tag}] iteration {it:,}  elapsed {elapsed/60:.1f} min  "
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
        _emit("    r  best_logL       k        AIC     dAIC        BIC     dBIC", fh)
        for row in rows:
            _emit(f"    {int(row['num_reticulations']):<2}"
                  f"{row['log_likelihood']:>11.1f}{int(row['k']):>8}"
                  f"{row['AIC']:>11.1f}{row.get('dAIC', float('nan')):>9.1f}"
                  f"{row.get('BIC', float('nan')):>11.1f}"
                  f"{row.get('dBIC', float('nan')):>9.1f}", fh)

    # Disk checkpoint of the trace + sampled networks (crash safety).
    try:
        view.write_log(os.path.join(OUT_DIR, "trace.log"))
        view.write_networks(os.path.join(OUT_DIR, "networks.trees"))
    except Exception as e:
        _emit(f"  (checkpoint write failed: {e})", fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loci", type=int, default=15)
    ap.add_argument("--sites", type=int, default=600)
    ap.add_argument("--iters", type=int, default=1_000_000)
    ap.add_argument("--milestone", type=int, default=100_000)
    ap.add_argument("--burnin", type=int, default=20_000)
    ap.add_argument("--thin", type=int, default=200)
    ap.add_argument("--gt-iters", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    fh = open(os.path.join(OUT_DIR, "stats.txt"), "w")

    true_net = build_true_network()
    data = simulate_multilocus(true_net, MAPPING, n_loci=a.loci,
                               seq_length=a.sites, theta=0.02, model=JC69(),
                               seed=a.seed)
    sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                       priors=MCMCSeqPriors(max_reticulations=2))

    _emit(f"True network : {true_net.newick()}", fh)
    _emit(f"Budget       : {a.iters:,} iters (burn-in {a.burnin:,}, "
          f"thin {a.thin}); milestone every {a.milestone:,}", fh)
    _emit(f"Data         : {a.loci} loci x {a.sites} sites; warm start "
          f"({a.gt_iters} GT iters)", fh)

    t0 = time.perf_counter()
    reported = {"last": 0}

    def control(prog):
        it = prog["iteration"]
        if it > 0 and it % a.milestone == 0 and it != reported["last"]:
            reported["last"] = it
            _milestone_report(it, prog, sampler, true_net, t0, fh,
                              tag=f"{it // 1000}k")
        return "continue"

    res = sampler.search(num_iter=a.iters, burn_in=a.burnin,
                         sample_freq=a.thin, seed=a.seed, warm_start=True,
                         warm_start_kwargs={"gt_iters": a.gt_iters},
                         control=control, check_every=a.milestone)
    total = time.perf_counter() - t0

    # ---- Final report on the completed 1M chain ----
    res.write_log(os.path.join(OUT_DIR, "trace.log"))
    res.write_networks(os.path.join(OUT_DIR, "networks.trees"))
    acc = score_accuracy(res.map_network, true_net)
    _emit("\n" + "#" * 70, fh)
    _emit(f"[FINAL] {a.iters:,} iters in {total/60:.1f} min "
          f"({1000*total/a.iters:.2f} ms/it)  acc={res.acceptance_rate:.3f}", fh)
    _emit(f"  reticulation posterior: "
          f"{ {k: round(v, 3) for k, v in res.reticulation_posterior().items()} }", fh)
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
    _emit(f"  MAP reticulations={acc.num_reticulations} (true 1); "
          f"clades AB/CD/EF recovered: "
          f"{[acc.clades_recovered.get(c) for c in ('AB','CD','EF')]}", fh)
    _emit(f"  mu-distance={acc.mu_distance}  MAP logP={res.map_log_posterior:.2f}", fh)
    _emit(f"  MAP net: {res.map_network.newick()}", fh)
    _emit(f"  map topology (most-sampled): {res.map_topology()}", fh)
    fh.close()


if __name__ == "__main__":
    main()
