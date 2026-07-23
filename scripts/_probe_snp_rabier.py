"""Golden cross-check against Charles Rabier's published level-1 table.

Network (from tests/testfiles/paper_net_largeseq.nex):
    ((C:.1, (B:.05)#H0[&gamma=.7]:.05)I1:.1, (A:.1, #H0:.05)I2:.1)I3;
with samples A:2, B:3, C:2 and u=v=1, coal=0.005.

For every (red_A, red_B, red_C) site pattern the biallelic likelihood must
match Rabier's expected value.  This is an external, published oracle -- the
strongest possible correctness check for the SNP likelihood engine.
"""
import sys, os, math, traceback
sys.path.insert(0, os.path.dirname(__file__))

TBL = {
    (0, 0, 0): 0.31581337186422315,
    (0, 0, 1): 1.853660371657668e-3,
    (0, 0, 2): 0.05677236283895234,
    (0, 1, 0): 1.6755618678903335e-3,
    (0, 1, 1): 1.0705941050619642e-5,
    (0, 1, 2): 4.789800667080107e-4,
    (0, 2, 0): 9.884301576368968e-4,
    (0, 2, 1): 7.570444581342921e-6,
    (0, 2, 2): 5.3322920321306e-4,
    (0, 3, 0): 0.04027355605172049,
    (0, 3, 1): 5.910438937463977e-4,
    (0, 3, 2): 0.07852626659131974,
    (1, 0, 0): 1.9618887485350735e-3,
    (1, 0, 1): 1.21627077880799976e-5,
    (1, 0, 2): 4.828155168689878e-4,
    (1, 1, 0): 1.09890103033996982e-5,
    (1, 1, 1): 9.099282927274783e-8,
    (1, 1, 2): 7.300548379825278e-6,
    (1, 2, 0): 7.30054837982544e-6,
    (1, 2, 1): 9.099282927274914e-9,
    (1, 2, 2): 1.098901030399711e-5,
    (1, 3, 0): 4.821551686898895e-4,
    (1, 3, 1): 1.2162707788079851e-5,
    (1, 3, 2): 1.9618887485350622e-3,
    (2, 0, 0): 0.0785262665913196,
    (2, 0, 1): 5.910438937463979e-4,
    (2, 0, 2): 0.040273556051720324,
    (2, 1, 0): 5.332292032130451e-4,
    (2, 1, 1): 7.5704445813427665e-6,
    (2, 1, 2): 9.884301576368857e-4,
    (2, 2, 0): 4.789800667080225e-4,
    (2, 2, 1): 1.0719114102479618e-5,
    (2, 2, 2): 1.6755618678903448e-3,
    (2, 3, 0): 0.0567723862838952165,
    (2, 3, 1): 1.8536603716576576548e-3,
    (2, 3, 2): 0.31581337186422315,
}

NEWICK = "((C:.1, (B:.05)#H0[&gamma=.7]:.05)I1:.1, (A:.1, #H0:.05)I2:.1)I3;"
SAMPLES = {"A": 2, "B": 3, "C": 2}


def write_one_site(path, reds):
    with open(path, "w", encoding="utf-8") as f:
        f.write("#NEXUS\n\nBEGIN TAXA;\nDIMENSIONS NTAX=3;\n")
        f.write("TAXLABELS A B C;\nEND;\n\n")
        f.write("BEGIN DATA;\n  Dimensions nchar=1;\n")
        f.write("  Format datatype=snp missing=? gap=- matchchar=.;\n  Matrix\n")
        f.write(f"    A {reds[0]}\n    B {reds[1]}\n    C {reds[2]}\n")
        f.write("  ;\nEND;\n\nBEGIN TREES;\n")
        f.write(f"Tree net = {NEWICK}\nEND;\n")


def main():
    from phynetpy.IO import read_nexus
    from phynetpy.MSA import MSA
    from phynetpy.BiMarkers import _snp_log_likelihood

    os.makedirs("runs", exist_ok=True)
    path = os.path.join("runs", "rabier_site.nex")

    worst = 0.0
    total = 0.0
    n_ok = 0
    for grouping, expected in sorted(TBL.items()):
        write_one_site(path, grouping)
        net = read_nexus(path)[0]
        aln = MSA(path)
        got = math.exp(_snp_log_likelihood(net, aln, 1.0, 1.0, 0.005, SAMPLES))
        total += got
        rel = abs(got - expected) / expected
        worst = max(worst, rel)
        ok = rel < 1e-6
        n_ok += ok
        flag = "ok " if ok else "XX "
        print(f"  {flag} {grouping}: got={got:.12e}  exp={expected:.12e}  rel={rel:.2e}")

    print(f"\n{n_ok}/{len(TBL)} patterns match (rel<1e-6). worst rel={worst:.3e}")
    print(f"sum over all patterns = {total:.10f}  (should be 1.0)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
