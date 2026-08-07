"""
Proof-of-concept: MPL network search with hill climbing.

Reads gene trees from subgeneset_3_ret1.txt, builds a random starting
species tree for the 10-taxon set from 5_nets.txt, runs 100 iterations
of hill climbing, and prints the log pseudo-likelihood at each accepted
move so we can verify scoring and convergence.
"""

import sys, os, copy, math, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from phynetpy.Network import Network
from phynetpy.IO import read_newick_file
from phynetpy._mpl import MPL
from phynetpy.infer import MPLScorer, MPLKernel
from phynetpy.ModelGraph import Model
from phynetpy.ModelMove import (
    AddReticulation, RemoveReticulation, FlipReticulation,
    ChangeNodeHeight, ChangeInheritanceProb,
    ChangeReticSource, ChangeReticDest,
)
from phynetpy.State import State


# ── 1. Load gene trees ────────────────────────────────────────────────
gt_file = os.path.join(os.path.dirname(__file__), "..", "tests", "testfiles", "subgeneset_3_ret1.txt")

taxa = ["t14", "t15", "t49", "t68", "t69", "t72", "t75", "t91", "t114", "t133"]
mapping = {t: [t] for t in taxa}

print("Loading gene trees ...")
t0 = time.time()
gene_trees = read_newick_file(gt_file, return_type="genetrees",
                              species_gene_mapping=mapping)
print(f"  {len(gene_trees.trees)} gene trees loaded in {time.time()-t0:.1f}s")


# ── 2. Build a random starting species tree ───────────────────────────
#    Simple balanced-ish binary tree with branch length 1.0
start_newick = (
    "((((t14:1,t15:1):1,(t49:1,t68:1):1):1,"
    "((t69:1,t72:1):1,(t75:1,t91:1):1):1):1,"
    "(t114:1,t133:1):1);"
)
species_net = Network.from_newick(start_newick)

print(f"Starting tree has {len(species_net.get_leaves())} leaves, "
      f"{len(list(species_net.V()))} nodes, "
      f"{len(species_net.E())} edges")
print(f"  Leaf labels: {sorted(n.label for n in species_net.get_leaves())}")


# ── 3. Score the starting tree ────────────────────────────────────────
print("\nInitializing MPL scorer (precomputing rho) ...")
t0 = time.time()
mpl = MPL(species_net, gene_trees, mapping)
init_score = mpl.score()
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Starting log pseudo-likelihood: {init_score:.4f}")


# ── 4. Manual search loop for monitoring ──────────────────────────────
#    Instead of calling mpl.search(), we replicate the HC loop here
#    so we can print progress at every iteration.

print("\n" + "="*60)
print("Beginning Hill Climbing search  (100 iterations)")
print("="*60 + "\n")

scorer = MPLScorer(mpl._rho, mpl._triplets)
model = Model()
model.network = copy.deepcopy(mpl.net)
model.set_likelihood_calculator(scorer)

kernel = MPLKernel(move_types=[
    AddReticulation,
    RemoveReticulation,
    FlipReticulation,
    ChangeNodeHeight,
    ChangeInheritanceProb,
    ChangeReticSource,
    ChangeReticDest,
])
state = State(model)

cur_score = state.likelihood()
best_score = cur_score
accepted_ct = 0
rejected_ct = 0
invalid_ct = 0

print(f"{'Iter':>5}  {'Action':>10}  {'Cur Score':>14}  {'Proposed':>14}  {'Best':>14}  {'Move':<25}")
print("-" * 95)

t_start = time.time()

error_ct = 0

for i in range(100):
    move = kernel.generate()
    move_name = type(move).__name__

    try:
        is_valid = state.generate_next(move)
    except Exception as exc:
        error_ct += 1
        print(f"{i:5d}  {'ERROR':>10}  {cur_score:14.4f}  {'---':>14}  {best_score:14.4f}  {move_name:<25}  {type(exc).__name__}")
        continue

    if not is_valid:
        invalid_ct += 1
        print(f"{i:5d}  {'INVALID':>10}  {cur_score:14.4f}  {'---':>14}  {best_score:14.4f}  {move_name:<25}")
        continue

    cur = state.likelihood()

    try:
        proposed = state.proposed().likelihood()
    except Exception as exc:
        state.revert(move)
        error_ct += 1
        print(f"{i:5d}  {'SCORE_ERR':>10}  {cur:14.4f}  {'---':>14}  {best_score:14.4f}  {move_name:<25}  {type(exc).__name__}")
        continue

    if math.isnan(proposed) or math.isinf(proposed):
        state.revert(move)
        rejected_ct += 1
        tag = "NaN/Inf"
        print(f"{i:5d}  {tag:>10}  {cur:14.4f}  {proposed:14.4f}  {best_score:14.4f}  {move_name:<25}")
        continue

    delta = cur - proposed

    if delta < 0:
        state.commit(move)
        accepted_ct += 1
        cur_score = state.likelihood()
        if cur_score > best_score:
            best_score = cur_score
        tag = "ACCEPT"
    else:
        state.revert(move)
        rejected_ct += 1
        tag = "reject"

    print(f"{i:5d}  {tag:>10}  {cur:14.4f}  {proposed:14.4f}  {best_score:14.4f}  {move_name:<25}")

elapsed = time.time() - t_start

print("\n" + "="*60)
print("Search complete")
print("="*60)
print(f"  Elapsed:  {elapsed:.1f}s")
print(f"  Accepted: {accepted_ct}")
print(f"  Rejected: {rejected_ct}")
print(f"  Invalid:  {invalid_ct}")
print(f"  Errors:   {error_ct}")
print(f"  Start score:  {init_score:.4f}")
print(f"  Final score:  {cur_score:.4f}")
print(f"  Best score:   {best_score:.4f}")

final_net = state.current_model.network
n_retics = sum(1 for n in final_net.V() if n.is_reticulation())
print(f"  Final network: {len(final_net.get_leaves())} leaves, "
      f"{n_retics} reticulations")
