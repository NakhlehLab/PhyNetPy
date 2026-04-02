"""
Demo: Score a species network against gene trees using MPL.

Inputs:
  - gt_file: path to gene trees (one Newick string per line)
  - st_file: path to species network (extended Newick, PhyloNet format)
  - taxa:    list of species labels present in the network
  - mapping: species -> allele label mapping
"""

from phynetpy.MPL_reference import MPL
from phynetpy.IO import read_newick_file

gt_file = "tests/testfiles/subgeneset_3_ret1.txt"
st_file = "tests/testfiles/5_nets.txt"

taxa = ["t14", "t15", "t49", "t68", "t69", "t72", "t75", "t91", "t114", "t133"]

# Identity mapping (one allele per species). For multi-allele:
#   mapping = {"SpeciesA": ["A1", "A2"], "SpeciesB": ["B1"], ...}
mapping = {t: [t] for t in taxa}

# Load gene trees
gts = read_newick_file(gt_file, return_type="genetrees", species_gene_mapping=mapping)

# Load species network (convert from PhyloNet ::gamma to PhyNetPy [&gamma=...])
species_net = read_newick_file(st_file)[0]

# Score
mpl = MPL(species_net, gts, mapping)
score = mpl.score()
print(f"Log pseudo-likelihood: {score:.4f}")
