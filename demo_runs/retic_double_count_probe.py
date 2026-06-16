"""Confirm the reticulation double-counting mechanism on a minimal network.

THEOREM (per-gene-tree bound).  For a gene tree with n lineages, the network
MSNC density is an expectation over reticulation routings of a coalescent-
history density, each <= (2/theta)^(n-1) (one (2/theta) per coalescence; every
waiting-time exponential <= 1; routing weights sum to 1).  Hence

    log P(g | Psi)  <=  (n - 1) * log(2/theta).

(Yu, Degnan & Nakhleh 2012, PNAS; Rannala & Yang 2003 for the per-branch
factor.)  Any value above this bound is over-credited density.

Here a single reticulation H sits ABOVE the A,B coalescence; in the gene tree
A and B coalesce in the branch below H.  The correct density must obey the
bound for every theta.  If the implementation re-credits the descendant mass on
both parent edges of H, the A,B coalescence's (2/theta) factor is counted twice
and the density exceeds the bound -- diverging as theta -> 0.
"""
import math

from phynetpy.Network import Network
from phynetpy._seq_likelihood import gene_tree_msnc_log_density

# One reticulation H, whose descendant cherry (A,B) coalesces BELOW it.
# Both parents of H (P1 at 0.9, P2 at 0.9) lie above H (0.6).
NET = (
    "((C:0.9,((A:0.3,B:0.3)ab:0.3)#H1:0.3[&gamma=0.6])P1:0.2,"
    "(D:0.9,#H1:0.3[&gamma=0.4])P2:0.2)R;"
)
# Tree counterpart: H replaced by a plain branch (no second parent).
TREE = "((C:0.9,((A:0.3,B:0.3)ab:0.3)h:0.3)P1:0.2,D:1.1)R;"

# Gene tree with coalescences pinned at the species boundaries (ab=0.3,
# P1=0.9, R=1.1) so waiting-time penalties ~ 0 and the density approaches its
# bound. A,B coalesce at 0.3 -- inside H's descendant branch [0.3,0.6] -- so
# the A,B (2/theta) factor is the one at risk of being double-counted.
GENE = "(((A:0.3,B:0.3)g1:0.6,C:0.9)g2:0.2,D:1.1)gr;"

species_of = {x: x for x in ("A", "B", "C", "D")}


def main():
    net = Network.from_newick(NET)
    tree = Network.from_newick(TREE)
    gene = Network.from_newick(GENE)
    n = 4
    print(f"gene tree: {n} leaves -> {n - 1} coalescences\n")
    print(f"{'theta':>9} {'bound':>9} {'NET dens':>10} {'over?':>6} "
          f"{'TREE dens':>10}")
    for theta in [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
        bound = (n - 1) * math.log(2.0 / theta)
        dn = gene_tree_msnc_log_density(gene, net, species_of, theta=theta)
        dt = gene_tree_msnc_log_density(gene, tree, species_of, theta=theta)
        over = "YES" if dn > bound + 1e-6 else "no"
        print(f"{theta:>9.4f} {bound:>9.2f} {dn:>10.2f} {over:>6} {dt:>10.2f}")
    print("\nIf 'NET dens' exceeds 'bound' and the excess grows as theta->0, "
          "the reticulation descendant mass is double-counted across H's two "
          "parent edges. The TREE column (no reticulation) should always be "
          "<= bound.")


if __name__ == "__main__":
    main()
