"""
Comprehensive DEFJ benchmark: Hill Climbing vs Simulated Annealing
with ground truth species networks for all four scenarios.
"""

import copy
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from phynetpy.IO import read_newick
from phynetpy.Infer_MP_Allop import (
    Allop_MUL, MPAllopScorer, MPAllopComponent,
    allele_map_set, partition_gene_trees,
)
from phynetpy.ModelFactory import ModelFactory
from phynetpy.ModelMove import SwitchParentage
from phynetpy.MetropolisHastings import (
    SimulatedAnnealing, Infer_MP_Allop_Kernel,
)
from phynetpy.State import State
from phynetpy.GraphUtils import mu_distance, hardwired_cluster_distance

DEFJ_ROOT = os.path.join(os.path.dirname(__file__), "..", "DEFJ", "10Genes", "withOG")

# ---------- ground truth networks (from colleague) ----------

TRUE_NETWORKS = {
    "D": "(((b:0.009,((x:0.006,(y:0.003,z:0.003):0.003):0.003)#H1:0):0.003,(#H1:0,a:0.009):0.003):0.04366667,o:0.10233333);",
    "E": "(o:0.10283333,(((a:0.006,((y:0.003,z:0.003):0.003)#H1:0):0.003,(x:0.009)#H2:0):0.003,(#H2:0,(#H1:0,b:0.006):0.003):0.003):0.04316667);",
    "F": "(o:0.10383333,((((a:0.003,(z:0.003)#H1:0):0.003,(y:0.006)#H2:0):0.003,(x:0.009)#H3:0):0.003,((#H2:0,(#H1:0,b:0.003):0.003):0.003,#H3:0):0.003):0.04216667);",
    "J": "((((b:0.017,a:0.017):0.01,((c:0.011,(d:0.006,(v:0.006)#H1:0):0.005):0.012,(((x:0.013,(y:0.01,z:0.01):0.003):0.003,w:0.016):0.007)#H2:0):0.004):0.008,((((#H1:0,e:0.006):0.006,(t:0.003,u:0.003):0.009):0.013,#H2:0):0.005,f:0.032):0.003):0.002275,o:0.043225);",
}

# ---------- gene maps per scenario ----------

GENE_MAPS = {
    "D": {
        "a": ["01aA"], "b": ["01bA"], "o": ["01oA"],
        "x": ["01xA", "01xB"], "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
    },
    "E": {
        "a": ["01aA"], "b": ["01bA"], "o": ["01oA"],
        "x": ["01xA", "01xB"], "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
    },
    "F": {
        "a": ["01aA"], "b": ["01bA"], "o": ["01oA"],
        "x": ["01xA", "01xB"], "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
    },
    "J": {
        "o": ["01oA"],
        "a": ["01aA"], "b": ["01bA"], "c": ["01cA"],
        "d": ["01dA"], "e": ["01eA"], "f": ["01fA"],
        "t": ["01tA", "01tB"], "u": ["01uA", "01uB"], "v": ["01vA", "01vB"],
        "w": ["01wA", "01wB"], "x": ["01xA", "01xB"],
        "y": ["01yA", "01yB"], "z": ["01zA", "01zB"],
    },
}

# ---------- test cases ----------

TEST_CASES = [
    # (scenario, g, n, t, r, n_iters, description)
    ("D",  1, 1,   4, 1, 200, "1g low-ILS"),
    ("D", 10, 1,   4, 1, 200, "10g low-ILS"),
    ("D", 10, 1, 100, 1, 300, "10g high-ILS"),
    ("E",  1, 1,   4, 1, 200, "1g low-ILS"),
    ("E", 10, 1,   4, 1, 200, "10g low-ILS"),
    ("E", 10, 1, 100, 1, 300, "10g high-ILS"),
    ("F",  1, 1,   4, 1, 200, "1g low-ILS"),
    ("F", 10, 1,   4, 1, 200, "10g low-ILS"),
    ("F", 10, 1, 100, 1, 300, "10g high-ILS"),
    ("J",  1, 1,  20, 1, 300, "1g mod-ILS"),
    ("J", 10, 1,  20, 1, 400, "10g mod-ILS"),
]

P = lambda *a, **kw: print(*a, **kw, flush=True)


def load_gene_trees(scenario: str, g: int, n: int, t: int, r: int) -> list:
    prefix = f"{scenario}2GT"
    filename = f"{prefix}g{g}n{n}t{t}r{r}-g_trees.newick"
    path = os.path.join(DEFJ_ROOT, scenario, f"g{g}", f"n{n}", f"t{t}", f"r{r}", filename)
    trees = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line.startswith("("):
                trees.append(read_newick(line))
    return trees


def build_model(scenario: str, g: int, n: int, t: int, r: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    gene_map = GENE_MAPS[scenario]
    gts = load_gene_trees(scenario, g, n, t, r)
    for gt in gts:
        af = allele_map_set(gt, gene_map)
        gt.put_item("allele maps", af)
        gt.put_item("leaf descendants", gt.leaf_descendants_all())
    start_net = partition_gene_trees(gene_map, rng=rng)
    comp = MPAllopComponent(start_net, gene_map, gts, rng)
    return ModelFactory(comp).build()


def run_hc(model, n_iters: int, true_net) -> dict:
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
    return {
        "init_pars": -init_score,
        "final_pars": -state.likelihood(),
        "accepted": accepted,
        "uphill": 0,
        "mu_d": mu_distance(net, true_net),
        "hw_d": hardwired_cluster_distance(net, true_net),
        "elapsed": elapsed,
    }


def run_sa(model, n_iters: int, true_net,
           t_start: float = 5.0, t_end: float = 0.01,
           n_restarts: int = 1, seed: int = 42) -> dict:
    kernel = Infer_MP_Allop_Kernel()
    model_copy = copy.deepcopy(model)
    init_score = State(copy.deepcopy(model_copy)).likelihood()

    t0 = time.perf_counter()
    sa = SimulatedAnnealing(
        pkernel=kernel, model=model_copy,
        num_iter=n_iters, t_start=t_start, t_end=t_end,
        n_restarts=n_restarts, seed=seed,
    )
    final_state = sa.run()
    elapsed = time.perf_counter() - t0

    net = final_state.current_model.network
    total_acc = sum(s["accepted"] for s in sa.run_stats)
    total_up = sum(s["uphill"] for s in sa.run_stats)
    return {
        "init_pars": -init_score,
        "final_pars": -sa.best_score,
        "accepted": total_acc,
        "uphill": total_up,
        "mu_d": mu_distance(net, true_net),
        "hw_d": hardwired_cluster_distance(net, true_net),
        "elapsed": elapsed,
    }


def fmt_row(method: str, r: dict) -> str:
    return (f"  {method:<18} {r['init_pars']:>5.0f} {r['final_pars']:>5.0f} "
            f"{r['accepted']:>5}/{r['uphill']:<4} "
            f"{r['mu_d']:>5} {r['hw_d']:>5} {r['elapsed']:>6.1f}s")


def main():
    true_nets = {s: read_newick(nwk) for s, nwk in TRUE_NETWORKS.items()}

    P("=" * 85)
    P("  DEFJ BENCHMARK: HC vs SA (T=5..0.01) vs SA-3x restarts")
    P("=" * 85)

    for s, nwk in TRUE_NETWORKS.items():
        tn = true_nets[s]
        n_r = sum(1 for nd in tn.V() if nd.is_reticulation())
        n_l = len(list(tn.get_leaves()))
        P(f"  {s}: {n_l} taxa, {n_r} retics")
    P()

    hdr = (f"  {'Method':<18} {'Par0':>5} {'Pars':>5} "
           f"{'Acc/Up':<10} {'mu_d':>5} {'hw_d':>5} {'Time':>7}")

    for scenario, g, n, t, r, n_iters, desc in TEST_CASES:
        label = f"{scenario}-g{g}-t{t}-r{r}"
        true_net = true_nets[scenario]

        P(f"\n{'-' * 85}")
        P(f"  {label} ({desc})  |  {n_iters} iters")
        P(f"{'-' * 85}")
        P(hdr)

        model = build_model(scenario, g, n, t, r)

        P("    Running HC...", end="")
        hc = run_hc(model, n_iters, true_net)
        P(f" done")
        P(fmt_row("HC", hc))

        P("    Running SA T=5...", end="")
        sa1 = run_sa(model, n_iters, true_net, t_start=5.0, t_end=0.01,
                      n_restarts=1)
        P(f" done")
        P(fmt_row("SA T=5", sa1))

        P("    Running SA T=5 x3...", end="")
        sa3 = run_sa(model, n_iters, true_net, t_start=5.0, t_end=0.01,
                      n_restarts=3)
        P(f" done")
        P(fmt_row("SA T=5 x3", sa3))

    P(f"\n{'=' * 85}")
    P("  Par0 = initial parsimony | Pars = final parsimony (lower=better)")
    P("  Acc/Up = accepted moves / uphill moves")
    P("  mu_d/hw_d = distance to true network (lower=better)")
    P(f"{'=' * 85}")


if __name__ == "__main__":
    main()
