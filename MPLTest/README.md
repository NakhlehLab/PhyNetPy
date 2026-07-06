# Dataset for "CAMUS: Scalable Phylogenetic Network Estimation"

This dataset contains all the data used for "CAMUS: Scalable Phylogenetic
Network Estimation" simulated performance study.

The simulated dataset is split up across six different model conditions (each
with a different number of species). These exist in six different directories, 
each labeled "nX", where X is the number of taxa (excluding the outgroup).

50 replicates were generated for each model condition - each exist in their own
subdirectory, labeled "00" - "49".

Inside each replicate folders, there are the following files:

- `true_net.nwk`, the true network in extended newick format.
- `g_true.nwk`, the 1000 true gene trees in newick format.
  
Additionally, for all conditions other than "n15," we estimated gene trees with
FastTree2. These estimated gene trees are labeled `g_500.nwk`. In the cases
where we estimated gene trees with IQTree3 with UltraFast Bootstrapping (e.g., 
the first 20 replicates for "n15" and "n25"), these 1000 gene trees are
included as `iqtree_500.nwk`.

Each replicate also contains a directory: `seqs-500`. This directory 
contains 1000 PHYLIP files, each containing the simulated sequences for one of
the gene trees.

Finally, we include an archive `scripts.zip` that contains miscellaneous
scripts used for this study. See the Supplementary Materials for the paper for
more details.
