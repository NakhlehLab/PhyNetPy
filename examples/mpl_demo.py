"""
Demo: Score species networks against gene trees using MPL.

Inputs:
  - gt_file: path to gene trees (one Newick string per line)
  - st_file: path to species networks (extended Newick; may contain multiple)
  - taxa:    list of species labels present in the networks
  - mapping: species -> allele label mapping
"""

from phynetpy.MPL import MPL
from phynetpy.Network import Network
from phynetpy.IO import read_newick_file, convert_newick

gt_file = "tests/testfiles/subgeneset_3_ret1.txt"
st_file = "tests/testfiles/5_nets.txt"

taxa = ["t14", "t15", "t49", "t68", "t69", "t72", "t75", "t91", "t114", "t133"]

# Identity mapping (one allele per species). For multi-allele:
#   mapping = {"SpeciesA": ["A1", "A2"], "SpeciesB": ["B1"], ...}
mapping = {t: [t] for t in taxa}

# Load gene trees
gts = read_newick_file(gt_file, return_type="genetrees", species_gene_mapping=mapping)

# Load candidate species networks. 5_nets.txt uses PhyloNet rich Newick
# (#H0:bl::gamma). read_newick_file() does not convert conventions, so we
# convert to PhyNetPy newick first (same as tests/test_mpl_5nets.py).
candidate_nets: list[Network] = []
with open(st_file, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "(" not in line or ";" not in line:
            continue
        try:
            net = Network.from_newick(convert_newick(line, standard="PhyNetPy"))
        except Exception:
            continue
        if net.get_leaves():
            candidate_nets.append(net)

# ──────────────────────────────────────────────────────────────────
# One-shot scoring: use MPL(...).score() when you have a single
# network and don't need the intermediate triplet data.
# ──────────────────────────────────────────────────────────────────
print("=== One-shot scoring (first network only) ===")
mpl = MPL(candidate_nets[0], gts, mapping)
score = mpl.score()
print(f"  Log pseudo-likelihood: {score:.4f}\n")


# ──────────────────────────────────────────────────────────────────
# Two-step scoring: useful when you want to score *multiple*
# candidate networks against the same gene-tree data. The gene-tree
# triplet computation (Step 1) is the expensive part, and it only
# needs to happen once.
# ──────────────────────────────────────────────────────────────────
print("=== Two-step scoring (all candidate networks) ===")

# Step 1: Precompute gene-tree triplet frequencies (done once).
gt_triplets = MPL.compute_gene_tree_triplets(gts, mapping, species_labels=taxa)
print(f"  Computed rho for {len(gt_triplets.triplets)} species triplets")

# Step 2: Score each candidate network against those frequencies.
for i, net in enumerate(candidate_nets, start=1):
    result = MPL.score_species_network_triplets(net, gt_triplets)
    print(f"  Network #{i}  log-PL = {result.log_pseudo_likelihood:.4f}")

# The SpeciesNetworkTripletResult also carries per-triplet probabilities
# if you need to inspect them:
#   result.probs_by_triplet[("t14", "t15", "t49")]  -> (P(xy|z), P(xz|y), P(yz|x))
