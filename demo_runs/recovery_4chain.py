"""Real recovery test: 4 parallel MCMC_SEQ chains on known-truth network data.

Simulates multilocus sequences on a 1-reticulation true network, runs 4
independent chains, then asks whether we recover the true topology, the
reticulation, gamma, and theta -- and whether the chains agree (R-hat).
"""
import sys

import numpy as np

from phynetpy.Network import Network
from phynetpy.infer import (
    MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus, run_parallel_chains,
)

TRUE_NETWORK = (
    "((((A:0.04,B:0.04)AB:0.03)#H1:0.02[&gamma=0.65],C:0.09)ABC:0.04,"
    "(#H1:0.04[&gamma=0.35],D:0.11)DR:0.02)R;"
)
TRUE_CLADES = [{"A", "B"}, {"A", "B", "C"}]
TRUE_THETA = 0.02


def descendant_leaves(net, node) -> frozenset:
    kids = net.get_children(node)
    if not kids:
        return frozenset({node.label})
    acc = set()
    for c in kids:
        acc |= descendant_leaves(net, c)
    return frozenset(acc)


def all_clades(net) -> set:
    leaves = {n.label for n in net.get_leaves()}
    out = set()
    for v in net.V():
        ds = descendant_leaves(net, v)
        if 1 < len(ds) < len(leaves):
            out.add(ds)
    return out


def topology_recovered(map_net, true_net) -> bool:
    return all_clades(true_net).issubset(all_clades(map_net))


def n_retic(net) -> int:
    return sum(1 for v in net.V() if v.is_reticulation())


def major_gamma(net):
    gammas = []
    for v in net.V():
        if v.is_reticulation():
            for e in net.in_edges(v):
                g = e.get_gamma()
                if g is not None:
                    gammas.append(float(g))
    return max(gammas) if gammas else None


def main():
    max_retic = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    num_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 120_000
    temps = ([float(x) for x in sys.argv[3].split(",")]
             if len(sys.argv) > 3 else None)

    true_net = Network.from_newick(TRUE_NETWORK)
    mapping = {sp: [sp] for sp in ("A", "B", "C", "D")}

    LOCI, SITES = 25, 800
    print(f"True network: {TRUE_NETWORK}")
    print(f"max_reticulations = {max_retic}")
    print(f"Simulating {LOCI} loci x {SITES} bp, theta={TRUE_THETA} ...", flush=True)
    data = simulate_multilocus(
        true_net, mapping, n_loci=LOCI, seq_length=SITES,
        theta=TRUE_THETA, model=JC69(), seed=2024,
    )
    sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                       priors=MCMCSeqPriors(max_reticulations=max_retic))

    NUM_ITER, BURN, THIN = num_iter, num_iter // 4, 25
    print(f"Running 4 chains x {NUM_ITER} iters (burn {BURN}, thin {THIN}) "
          f"temps={temps} ...", flush=True)
    result = run_parallel_chains(
        sampler, n_chains=4, num_iter=NUM_ITER, burn_in=BURN,
        sample_freq=THIN, seed=2024, check_every=5_000,
        monitor=None, progress=False,
        temperatures=temps, swap_interval=10,
    )

    print("\n================ RECOVERY ================")
    print(f"wall_time={result.wall_time_sec:.1f}s  "
          f"chains={len(result.chains)}/{result.n_chains}  "
          f"errors={result.errors}")
    print("final R-hat: "
          + ", ".join(f"{k}={v:.3f}" for k, v in result.rhat.items()))

    print("\nPer-chain MAP:")
    for i, r in enumerate(result.chains):
        mn = r.map_network
        topo = "RECOVERED" if topology_recovered(mn, true_net) else "missed"
        g = major_gamma(mn)
        gstr = f"{g:.3f}" if g is not None else "none"
        print(f"  chain {i}: MAP logP={r.map_log_posterior:.2f} "
              f"reti={n_retic(mn)} gamma={gstr} topo={topo} "
              f"theta={r.map_theta:.4f} ess_min={r.summary().min_ess:.1f}")

    best = result.best()
    print(f"\nBest MAP across chains: logP={best.map_log_posterior:.2f}")
    print(f"  network: {best.map_network.newick()}")
    print(f"  topology recovered: {topology_recovered(best.map_network, true_net)}")
    print(f"  reticulations: {n_retic(best.map_network)} (true 1)")
    bg = major_gamma(best.map_network)
    print(f"  gamma (major): {bg if bg is None else round(bg,3)} (true 0.65)")

    # --- POSTERIOR (mass-based) summaries -- the trustworthy estimates ---
    pooled = result.pooled_samples()
    if pooled:
        tot = len(pooled)
        print(f"\nPooled posterior reticulation count (n={tot}):")
        for k, p in result.reticulation_posterior().items():
            print(f"  {k} retic: {p:.3f}")

        print("\nPosterior MAP TOPOLOGY (mass-based, ignores branch lengths):")
        topo = result.topology_posterior(top_n=3)
        for rank, (nwk, prob) in enumerate(topo):
            try:
                tnet = Network.from_newick(nwk)
                clades_ok = all(frozenset(c) in all_clades(tnet)
                                for c in TRUE_CLADES)
                nr = n_retic(tnet)
            except Exception:
                clades_ok, nr = False, "?"
            print(f"  #{rank+1} p={prob:.3f} retic={nr} "
                  f"trueClades={'YES' if clades_ok else 'no'}")
        mt = result.map_topology()
        if mt:
            mtn = Network.from_newick(mt)
            print(f"  MAP-topology clades recovered: "
                  f"{topology_recovered(mtn, true_net)}  "
                  f"reticulations={n_retic(mtn)} (true 1)")
        thetas = np.array([s.theta for s in pooled])
        print(f"theta posterior: mean={thetas.mean():.4f} "
              f"2.5%={np.percentile(thetas,2.5):.4f} "
              f"97.5%={np.percentile(thetas,97.5):.4f} (true {TRUE_THETA})")
        # clade recovery frequency across pooled MAP samples
        from phynetpy.Network import Network as _N
        hits = {frozenset(c): 0 for c in TRUE_CLADES}
        for s in pooled:
            try:
                net = _N.from_newick(s.network_newick)
            except Exception:
                continue
            cl = all_clades(net)
            for c in TRUE_CLADES:
                if frozenset(c) in cl:
                    hits[frozenset(c)] += 1
        print("clade posterior support:")
        for c in TRUE_CLADES:
            print(f"  {sorted(c)}: {hits[frozenset(c)]/tot:.3f}")


if __name__ == "__main__":
    main()
