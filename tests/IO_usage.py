#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IO Module Usage Demo - PhyNetPy v1.1
=====================================

This script exercises every public function in PhyNetPy.IO against the sample
files in tests/testfiles/.  Run it from the repository root:

    python -m tests.IO_usage

or directly:

    python tests/IO_usage.py
"""

from __future__ import annotations

import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so we can import PhyNetPy as a package
# even when running this script directly.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TESTFILES = os.path.join(SCRIPT_DIR, "testfiles")

# ---------------------------------------------------------------------------
# PhyNetPy imports (using the public package API)
# ---------------------------------------------------------------------------
from src.IO import (
    read_fasta,
    read_fasta_records,
    write_fasta,
    read_vcf,
    read_vcf_metadata,
    write_vcf,
    read_newick,
    read_newick_file,
    write_newick,
    write_newick_file,
    read_nexus,
    write_nexus,
    convert_newick,
    detect_newick_standard,
)
from src.Network import Network, Node, Edge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = 0
FAIL = 0

def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def check(label: str, condition: bool, detail: str = "") -> None:
    """Report pass/fail for a single check."""
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"  --  {detail}"
    print(msg)


# =========================================================================
# 1. FASTA
# =========================================================================
section("1. FASTA - Reading & Writing")

# 1a. read_fasta_records (low-level)
fasta_path = os.path.join(TESTFILES, "sample.fasta")
records = read_fasta_records(fasta_path)
check("read_fasta_records returns list", isinstance(records, list))
check("read_fasta_records found 5 records", len(records) == 5,
      f"got {len(records)}")
check("First record name is Homo_sapiens",
      records[0].get_name() == "Homo_sapiens",
      f"got '{records[0].get_name()}'")
check("Sequence is a list of chars", isinstance(records[0].get_seq(), list))

# 1b. read_fasta (returns MSA)
msa = read_fasta(fasta_path)
check("read_fasta returns MSA", type(msa).__name__ == "MSA")
check("MSA contains 5 records", len(msa.get_records()) == 5)

# 1c. read_fasta with grouping
grouping = {
    "Great_Apes": ["Homo_sapiens", "Pan_troglodytes", "Gorilla_gorilla", "Pongo_abelii"],
    "Old_World_Monkeys": ["Macaca_mulatta"],
}
msa_grouped = read_fasta(fasta_path, grouping=grouping)
check("Grouped MSA has 2 groups", msa_grouped.groups == 2,
      f"got {msa_grouped.groups}")

# 1d. write_fasta round-trip
tmp_fasta = os.path.join(tempfile.gettempdir(), "phynetpy_test_roundtrip.fasta")
write_fasta(msa, tmp_fasta)
check("write_fasta creates file", os.path.exists(tmp_fasta))

# Re-read and compare
msa_rt = read_fasta(tmp_fasta)
check("Round-trip preserves record count",
      len(msa_rt.get_records()) == len(msa.get_records()))
for orig, rt in zip(msa.get_records(), msa_rt.get_records()):
    if orig.get_name() != rt.get_name():
        check(f"Round-trip name mismatch: {orig.get_name()} vs {rt.get_name()}", False)
        break
else:
    check("Round-trip preserves all sequence names", True)

os.remove(tmp_fasta)

# 1e. Small FASTA
small_fasta = os.path.join(TESTFILES, "sample_dna.fasta")
small_msa = read_fasta(small_fasta)
check("Small FASTA: 4 records", len(small_msa.get_records()) == 4)
print(f"       Sequences: {[r.get_name() for r in small_msa.get_records()]}")


# =========================================================================
# 2. VCF
# =========================================================================
section("2. VCF - Reading, Metadata & Writing")

vcf_path = os.path.join(TESTFILES, "sample.vcf")

# 2a. read_vcf_metadata
meta = read_vcf_metadata(vcf_path)
check("VCF metadata: fileformat detected",
      meta["fileformat"] == "VCFv4.1",
      f"got '{meta['fileformat']}'")
check("VCF metadata: 6 samples",
      len(meta["sample_names"]) == 6,
      f"got {len(meta['sample_names'])}: {meta['sample_names']}")

# 2b. read_vcf (no grouping)
vcf_msa = read_vcf(vcf_path)
check("read_vcf returns MSA", type(vcf_msa).__name__ == "MSA")
check("VCF MSA has 6 samples", len(vcf_msa.get_records()) == 6)

first_rec = vcf_msa.get_records()[0]
seq = first_rec.get_seq()
check("First sample has 10 variant sites", len(seq) == 10,
      f"got {len(seq)}")
print(f"       {first_rec.get_name()} genotype vector: {seq}")

# 2c. read_vcf with grouping (species mapping)
species_grouping = {
    "SpeciesA": ["Ind1_SpeciesA", "Ind2_SpeciesA"],
    "SpeciesB": ["Ind1_SpeciesB", "Ind2_SpeciesB"],
    "SpeciesC": ["Ind1_SpeciesC", "Ind2_SpeciesC"],
}
vcf_msa_grouped = read_vcf(vcf_path, grouping=species_grouping)
check("Grouped VCF MSA has 3 groups",
      vcf_msa_grouped.groups == 3,
      f"got {vcf_msa_grouped.groups}")

# 2d. write_vcf round-trip
tmp_vcf = os.path.join(tempfile.gettempdir(), "phynetpy_test_roundtrip.vcf")
write_vcf(vcf_msa, tmp_vcf)
check("write_vcf creates file", os.path.exists(tmp_vcf))

vcf_rt = read_vcf(tmp_vcf)
check("VCF round-trip preserves sample count",
      len(vcf_rt.get_records()) == len(vcf_msa.get_records()))
os.remove(tmp_vcf)


# =========================================================================
# 3. NEWICK STRINGS
# =========================================================================
section("3. Newick - Parsing & Writing Strings")

# 3a. Simple tree
newick_str = "((A:0.1,B:0.2):0.3,C:0.4);"
net = read_newick(newick_str)
check("read_newick returns Network", type(net).__name__ == "Network")
leaves = net.get_leaves()
leaf_names = sorted([str(l) for l in leaves])
check("Parsed tree has 3 leaves",
      len(leaves) == 3, f"got {len(leaves)}: {leaf_names}")

# 3b. Write back to newick
nwk_out = write_newick(net)
check("write_newick produces a string", isinstance(nwk_out, str) and len(nwk_out) > 0,
      f"got '{nwk_out}'")

# 3c. Network with reticulation
retic_str = "((A:0.1,(B:0.05)#H1[&gamma=0.7]:0.05):0.2,(#H1[&gamma=0.3]:0.1,(C:0.1,D:0.1):0.05):0.15);"
net_retic = read_newick(retic_str)
retic_leaves = net_retic.get_leaves()
retic_leaf_names = sorted([str(l) for l in retic_leaves])
check("Reticulate network has 4 leaves",
      len(retic_leaves) == 4, f"got {retic_leaf_names}")

# 3d. Read newick file (multiple trees)
newick_file = os.path.join(TESTFILES, "sample_newick.tre")
networks = read_newick_file(newick_file)
check("read_newick_file found 3 trees", len(networks) == 3,
      f"got {len(networks)}")
for i, n in enumerate(networks):
    lf = sorted([str(l) for l in n.get_leaves()])
    print(f"       Tree {i+1} leaves: {lf}")

# 3e. write_newick_file round-trip
tmp_nwk = os.path.join(tempfile.gettempdir(), "phynetpy_test_trees.tre")
write_newick_file(networks, tmp_nwk)
check("write_newick_file creates file", os.path.exists(tmp_nwk))
networks_rt = read_newick_file(tmp_nwk)
check("Round-trip preserves tree count",
      len(networks_rt) == len(networks))
os.remove(tmp_nwk)


# =========================================================================
# 4. NEXUS
# =========================================================================
section("4. Nexus - Reading & Writing")

# 4a. Read nexus with trees
nexus_path = os.path.join(TESTFILES, "sample_trees.nex")
nex_nets = read_nexus(nexus_path)
check("read_nexus returns list of Networks",
      isinstance(nex_nets, list) and all(type(n).__name__ == "Network" for n in nex_nets))
check("Nexus file has 2 trees", len(nex_nets) == 2,
      f"got {len(nex_nets)}")
for i, n in enumerate(nex_nets):
    lf = sorted([str(l) for l in n.get_leaves()])
    print(f"       Nexus tree {i+1} leaves: {lf}")

# 4b. Nexus with reticulation network
retic_nex = os.path.join(TESTFILES, "sample_network.nex")
retic_nets = read_nexus(retic_nex)
check("Reticulate nexus has 1 network", len(retic_nets) == 1)
retic_net = retic_nets[0]
retic_lf = sorted([str(l) for l in retic_net.get_leaves()])
check("Reticulate network has correct leaves",
      set(retic_lf) == {"A", "B", "C", "D"},
      f"got {retic_lf}")

# 4c. write_nexus round-trip
tmp_nex = os.path.join(tempfile.gettempdir(), "phynetpy_test_roundtrip.nex")
write_nexus(nex_nets, tmp_nex)
check("write_nexus creates file", os.path.exists(tmp_nex))

nex_rt = read_nexus(tmp_nex)
check("Nexus round-trip preserves tree count",
      len(nex_rt) == len(nex_nets))
os.remove(tmp_nex)

# 4d. write_nexus with custom taxa and PhyloNet commands
tmp_nex2 = os.path.join(tempfile.gettempdir(), "phynetpy_test_phylonet.nex")
write_nexus(
    nex_nets, tmp_nex2,
    taxa={"Homo_sapiens", "Pan_troglodytes", "Gorilla_gorilla",
          "Pongo_abelii", "Macaca_mulatta"},
    tree_prefix="primate",
    phylonet_cmds=[
        "InferNetwork_ML (primate1) 1 -bl;",
    ]
)
check("write_nexus with PhyloNet block creates file", os.path.exists(tmp_nex2))
with open(tmp_nex2, "r") as f:
    content = f.read()
check("PhyloNet block present in output",
      "BEGIN PHYLONET;" in content)
print(f"       Nexus file preview (first 400 chars):")
for line in content[:400].splitlines():
    print(f"         {line}")
os.remove(tmp_nex2)


# =========================================================================
# 5. NEWICK STANDARD CONVERSION
# =========================================================================
section("5. Newick Standard Conversion")

phynetpy_str = "((C:.1,(B:.05)#H0[&gamma=.7]:.05)I1:.1,(A:.1,#H0:.05)I2:.1)I3;"
phylonet_str = "((C:.1,(B:.05)#H0:.05::.7)I1:.1,(A:.1,#H0:.05)I2:.1)I3;"
beast_str    = "[&R] ((C:.1,(B:.05)#H0[&gamma=.7]:.05)I1:.1,(A:.1,#H0:.05)I2:.1)I3;"

# 5a. detect_newick_standard
check("Detect PhyNetPy standard",
      detect_newick_standard(phynetpy_str) == "PhyNetPy",
      f"got '{detect_newick_standard(phynetpy_str)}'")
check("Detect Phylonet standard",
      detect_newick_standard(phylonet_str) == "Phylonet",
      f"got '{detect_newick_standard(phylonet_str)}'")
check("Detect Beast standard",
      detect_newick_standard(beast_str) == "Beast",
      f"got '{detect_newick_standard(beast_str)}'")

# 5b. convert between standards
converted_to_phylonet = convert_newick(phynetpy_str, "Phylonet")
check("PhyNetPy -> Phylonet contains '::'",
      "::" in converted_to_phylonet,
      f"got '{converted_to_phylonet}'")

converted_to_beast = convert_newick(phynetpy_str, "Beast")
check("PhyNetPy -> Beast starts with [&R]",
      converted_to_beast.startswith("[&R]"),
      f"got '{converted_to_beast[:30]}...'")

converted_back = convert_newick(converted_to_phylonet, "PhyNetPy")
check("Phylonet -> PhyNetPy round-trip contains [&gamma=",
      "[&gamma=" in converted_back,
      f"got '{converted_back}'")

# 5c. convert plain newick (no reticulation - should be unchanged)
plain = "((A:0.1,B:0.2):0.3,C:0.4);"
converted_plain = convert_newick(plain, "Phylonet")
check("Plain newick unchanged by Phylonet conversion",
      converted_plain.strip() == plain.strip(),
      f"got '{converted_plain}'")


# =========================================================================
# 6. PROGRAMMATIC NETWORK -> NEWICK/NEXUS
# =========================================================================
section("6. Programmatic Network Creation -> Export")

# Build a small network programmatically
root = Node("Root")
i1 = Node("I1")
leaf_a = Node("A")
leaf_b = Node("B")
leaf_c = Node("C")

net_prog = Network()
net_prog.add_nodes(root)
net_prog.add_nodes(i1)
net_prog.add_nodes(leaf_a)
net_prog.add_nodes(leaf_b)
net_prog.add_nodes(leaf_c)

e1 = Edge(root, i1); e1.set_length(0.3)
e2 = Edge(i1, leaf_a); e2.set_length(0.1)
e3 = Edge(i1, leaf_b); e3.set_length(0.2)
e4 = Edge(root, leaf_c); e4.set_length(0.4)

net_prog.add_edges(e1)
net_prog.add_edges(e2)
net_prog.add_edges(e3)
net_prog.add_edges(e4)

nwk_prog = write_newick(net_prog)
check("Programmatic network produces newick",
      isinstance(nwk_prog, str) and nwk_prog.endswith(";"),
      f"got '{nwk_prog}'")
print(f"       Newick: {nwk_prog}")

# Export to nexus
tmp_prog_nex = os.path.join(tempfile.gettempdir(), "phynetpy_prog.nex")
write_nexus([net_prog], tmp_prog_nex, tree_prefix="manual")
check("Programmatic network exported to nexus", os.path.exists(tmp_prog_nex))

# Re-read
prog_rt = read_nexus(tmp_prog_nex)
check("Re-read programmatic network", len(prog_rt) == 1)
prog_leaves = sorted([str(l) for l in prog_rt[0].get_leaves()])
check("Re-read has correct leaves",
      set(prog_leaves) == {"A", "B", "C"},
      f"got {prog_leaves}")
os.remove(tmp_prog_nex)


# =========================================================================
# SUMMARY
# =========================================================================
section("SUMMARY")
total = PASS + FAIL
print(f"\n  {PASS}/{total} checks passed, {FAIL} failed.\n")
if FAIL > 0:
    print("  *** Some checks FAILED - review output above. ***\n")
    sys.exit(1)
else:
    print("  All checks passed! The IO module is working correctly.\n")
    sys.exit(0)

