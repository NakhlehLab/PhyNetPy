"""
Compare Hill Climbing vs Simulated Annealing on Scenario D,
focusing on high-ILS cases where HC gets stuck.

Tests multiple temperature schedules to find what works best.
"""

import copy
import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from phynetpy.IO import read_newick
from phynetpy.infer import MPAllopComponent
from phynetpy._infer_mp_allop import allele_map_set, partition_gene_trees
from phynetpy.ModelFactory import ModelFactory
from phynetpy.ModelMove import SwitchParentage
from phynetpy.MetropolisHastings import (
    SimulatedAnnealing, Infer_MP_Allop_Kernel,
)
from phynetpy.State import State
from phynetpy.GraphUtils import mu_distance, hardwired_cluster_distance

DEFJ_ROOT = os.path.join(os.path.dirname(__file__), "..", "DEFJ", "10Genes", "withOG")

GENE_MAP = {
    "a": ["01aA"], "b": ["01bA"], "o": ["01oA"],
    "x": ["01xA", "01xB"], "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
}

TRUE_NETWORK_NWK = "(o,((a,((z,y),x)#H1),(b,#H1)));"

TEMP_SCHEDULES = [
    # (t_start, t_end, label)
    (  1.0, 0.001, "cold:    T=1..0.001"),
    (  5.0, 0.01,  "mild:    T=5..0.01"),
    ( 10.0, 0.01,  "warm:    T=10..0.01"),
    ( 50.0, 0.01,  "hot:     T=50..0.01"),
    (100.0, 0.1,   "blazing: T=100..0.1"),
]

TEST_CASES = [
    # (g, n, t, r, n_iters, description)
    ( 1, 1,   4, 1, 200, "easy: 1g, low ILS"),
    (10, 1,   4, 1, 200, "med:  10g, low ILS"),
    ( 1, 1, 100, 1, 200, "hard: 1g, high ILS"),
    (10, 1, 100, 1, 300, "hard: 10g, high ILS"),
    (10, 1, 100, 3, 300, "hard: 10g, high ILS r3"),
]


def load_gene_trees(g: int, n: int, t: int, r: int) -> list:
    filename = f"D2GTg{g}n{n}t{t}r{r}-g_trees.newick"
    path = os.path.join(DEFJ_ROOT, "D", f"g{g}", f"n{n}", f"t{t}", f"r{r}", filename)
    trees = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("("):
                trees.append(read_newick(line))
    return trees


def build_model(g: int, n: int, t: int, r: int, seed: int = 42):
    """Build a fresh model for a given test case."""
    rng = np.random.default_rng(seed)
    gts = load_gene_trees(g, n, t, r)
    for gt in gts:
        af = allele_map_set(gt, GENE_MAP)
        gt.put_item("allele maps", af)
        gt.put_item("leaf descendants", gt.leaf_descendants_all())
    start_net = partition_gene_trees(GENE_MAP, rng=rng)
    comp = MPAllopComponent(start_net, GENE_MAP, gts, rng)
    return ModelFactory(comp).build()


def run_hill_climb(model, n_iters: int, true_net) -> dict:
    """Run strict hill climbing (baseline)."""
    state = State(copy.deepcopy(model))
    init_score = state.likelihood()

    t0 = time.perf_counter()
    accepted = 0
    for i in range(n_iters):
        move = SwitchParentage(i)
        is_valid = state.generate_next(move)
        if is_valid:
            cur = state.likelihood()
            proposed = state.proposed().likelihood()
            if cur - proposed < 0:
                state.commit(move)
                accepted += 1
            else:
                state.revert(move)
    elapsed = time.perf_counter() - t0

    net = state.current_model.network
    mu = mu_distance(net, true_net)
    hw = hardwired_cluster_distance(net, true_net)

    return {
        "method": "HC",
        "init_pars": -init_score,
        "final_pars": -state.likelihood(),
        "accepted": accepted,
        "uphill": 0,
        "mu_d": mu,
        "hw_d": hw,
        "elapsed": elapsed,
    }


def run_sa(model, n_iters: int, true_net,
           t_start: float, t_end: float, label: str,
           seed: int = 42) -> dict:
    """Run simulated annealing with a given temperature schedule."""
    kernel = Infer_MP_Allop_Kernel()
    model_copy = copy.deepcopy(model)
    init_score = State(copy.deepcopy(model_copy)).likelihood()

    t0 = time.perf_counter()
    sa = SimulatedAnnealing(
        pkernel=kernel,
        model=model_copy,
        num_iter=n_iters,
        t_start=t_start,
        t_end=t_end,
        n_restarts=1,
        seed=seed,
    )
    final_state = sa.run()
    elapsed = time.perf_counter() - t0

    net = final_state.current_model.network
    mu = mu_distance(net, true_net)
    hw = hardwired_cluster_distance(net, true_net)

    stats = sa.run_stats[0]
    return {
        "method": label,
        "init_pars": -init_score,
        "final_pars": -sa.best_score,
        "accepted": stats["accepted"],
        "uphill": stats["uphill"],
        "mu_d": mu,
        "hw_d": hw,
        "elapsed": elapsed,
    }


def main():
    true_net = read_newick(TRUE_NETWORK_NWK)
    print("=" * 95)
    print("  SCENARIO D: Hill Climbing vs Simulated Annealing")
    print(f"  True network: {TRUE_NETWORK_NWK.strip()}")
    print("=" * 95)

    for g, n, t, r, n_iters, desc in TEST_CASES:
        label = f"D-g{g}-t{t}-r{r}"
        print(f"\n{'-' * 95}")
        print(f"  {label} ({desc})  |  {n_iters} iters")
        print(f"{'-' * 95}")

        model = build_model(g, n, t, r)

        hdr = (f"  {'Method':<25} {'Pars0':>5} {'Pars':>5} "
               f"{'Acc':>5} {'Up':>4} {'mu_d':>5} {'hw_d':>5} {'Time':>7}")
        print(hdr)

        # Hill climbing baseline
        hc = run_hill_climb(model, n_iters, true_net)
        print(f"  {'HC (baseline)':<25} {hc['init_pars']:>5.0f} {hc['final_pars']:>5.0f} "
              f"{hc['accepted']:>5} {hc['uphill']:>4} "
              f"{hc['mu_d']:>5} {hc['hw_d']:>5} {hc['elapsed']:>6.2f}s")

        # SA with different temperatures
        for t_start, t_end, temp_label in TEMP_SCHEDULES:
            res = run_sa(model, n_iters, true_net, t_start, t_end, temp_label)
            print(f"  {temp_label:<25} {res['init_pars']:>5.0f} {res['final_pars']:>5.0f} "
                  f"{res['accepted']:>5} {res['uphill']:>4} "
                  f"{res['mu_d']:>5} {res['hw_d']:>5} {res['elapsed']:>6.2f}s")

    print(f"\n{'=' * 95}")
    print("  Pars = final parsimony score (lower is better)")
    print("  Acc = total accepted moves, Up = uphill (worse) moves accepted")
    print("  mu_d/hw_d = distance to true network (lower is better)")
    print(f"{'=' * 95}")


if __name__ == "__main__":
    main()
