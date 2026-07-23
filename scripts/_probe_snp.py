import sys, os, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from mcmc_harness import build_true_network, TAXA, score_accuracy


def main():
    from phynetpy.SNPSimulator import simulate
    from phynetpy.BiMarkers import MCMC_BIMARKERS, _snp_starting_tree, _snp_log_likelihood
    from phynetpy.MSA import MSA

    true_net = build_true_network()
    print("true:", true_net.newick())

    samples = {leaf.label: 1 for leaf in true_net.get_leaves()}
    u = v = 1.0
    coal = 0.005
    sim = simulate(n=len(TAXA), s=60, net=true_net, samples=samples,
                   u=u, v=v, coal=coal, seed=7)
    os.makedirs("runs", exist_ok=True)
    path = os.path.join("runs", "probe_6t_1r.nex")
    sim.write_nexus(path)
    print("wrote", path)

    # Sanity: score the true network and a starting caterpillar.
    aln = MSA(path)
    taxa = [rec.get_name() for rec in aln.get_records()]
    print("taxa:", taxa)
    start = _snp_starting_tree(taxa)
    print("start:", start.newick())
    ll_true = _snp_log_likelihood(true_net, aln, u, v, coal, samples, verbose=True)
    ll_start = _snp_log_likelihood(start, aln, u, v, coal, samples, verbose=False)
    print(f"ll_true={ll_true:.4f}  ll_start={ll_start:.4f}")

    t0 = time.perf_counter()
    result = MCMC_BIMARKERS(path, u=u, v=v, coal=coal, num_iter=400,
                            burn_in=100, sample_freq=20, seed=7,
                            samples=samples, max_reticulations=2)
    dt = time.perf_counter() - t0
    best_net = max(result, key=result.get)
    best_score = result[best_net]
    print(f"\nMCMC done: {dt:.2f}s  best_score={best_score:.4f}")
    print("best:", best_net.newick())
    print(score_accuracy(best_net, true_net))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
