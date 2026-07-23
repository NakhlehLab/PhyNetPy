"""What does the chain SEE at its starting state: tree vs true network?

Builds the exact starting state MCMC_SEQ would use (UPGMA per-locus gene trees,
theta start), then scores the full log posterior with:
  * the UPGMA species tree (0 retic, what the chain starts from), and
  * the true 1-reticulation network,
using the SAME (UPGMA) gene trees.  Then it also scores both using the TRUE
gene trees, to separate "gene trees not converged yet" from "reticulation not
supported".  This isolates whether the add-reticulation barrier is a mixing
problem (gene trees need to move first) or a target problem.
"""
import os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.mcmc_harness import build_true_network, MAPPING
from phynetpy.infer import MCMC_SEQ, MCMCSeqPriors, JC69, simulate_multilocus
from phynetpy._mcmc_seq import SeqState, _build_species_tree

def score_with(sampler, species_net, gene_trees):
    """Full log posterior of (species_net, gene_trees) under sampler settings."""
    st = SeqState(
        copy.deepcopy(species_net),
        [copy.deepcopy(gt) for gt in gene_trees],
        sampler.species_of, sampler.loci, sampler.priors,
        sampler.model, sampler.theta,
    )
    return st.log_posterior(), st.log_likelihood(), st.log_prior()

true_net = build_true_network()

for n_loci, sites in [(10, 400), (30, 400), (60, 500)]:
    data = simulate_multilocus(true_net, MAPPING, n_loci=n_loci, seq_length=sites,
                               theta=0.02, model=JC69(), seed=7)
    sampler = MCMC_SEQ(**data.to_mcmc_seq_kwargs(),
                       priors=MCMCSeqPriors(max_reticulations=2))
    upgma_gts = sampler.gene_trees
    true_gts = data.gene_trees
    tree_net = sampler.species_net  # UPGMA species tree

    print(f"\n=== loci={n_loci} sites={sites} ===")
    # With UPGMA (starting) gene trees:
    p_tree_u, l_tree_u, pr_tree_u = score_with(sampler, tree_net, upgma_gts)
    p_true_u, l_true_u, pr_true_u = score_with(sampler, true_net, upgma_gts)
    print(f"  [UPGMA gts]  tree logP={p_tree_u:10.2f} (L={l_tree_u:.2f} pr={pr_tree_u:.2f})")
    print(f"  [UPGMA gts]  true logP={p_true_u:10.2f} (L={l_true_u:.2f} pr={pr_true_u:.2f})")
    print(f"  [UPGMA gts]  true-tree = {p_true_u - p_tree_u:+.2f}  (positive => retic preferred)")
    # With TRUE gene trees:
    p_tree_t, l_tree_t, _ = score_with(sampler, tree_net, true_gts)
    p_true_t, l_true_t, _ = score_with(sampler, true_net, true_gts)
    print(f"  [TRUE  gts]  tree logP={p_tree_t:10.2f}")
    print(f"  [TRUE  gts]  true logP={p_true_t:10.2f}")
    print(f"  [TRUE  gts]  true-tree = {p_true_t - p_tree_t:+.2f}")
