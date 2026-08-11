# PhyNetPy IO Module Guide

> **Version:** 1.1.0  
> **Module:** `PhyNetPy.IO`  
> **Last updated:** February 2026

The `IO` module is the central hub for reading and writing all phylogenetic file formats in PhyNetPy. It replaces the older `NetworkParser` class with a clean, function-based API.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [FASTA Files](#fasta-files)
3. [VCF Files](#vcf-files)
4. [Newick Strings](#newick-strings)
5. [Newick Files](#newick-files)
6. [Nexus Files](#nexus-files)
7. [Newick Standard Conversion](#newick-standard-conversion)
8. [Function Reference](#function-reference)

---

## Quick Start

```python
from PhyNetPy.IO import read_fasta, read_vcf, read_newick, read_nexus, write_nexus
```

Or, since everything is re-exported from the top-level package:

```python
from PhyNetPy import read_fasta, read_vcf, read_newick, read_nexus
```

---

## FASTA Files

### Reading a FASTA into an MSA

```python
from PhyNetPy.IO import read_fasta

# Basic read — returns an MSA object
msa = read_fasta("sequences.fasta")

# Inspect what was loaded
for rec in msa.get_records():
    print(f"{rec.get_name()}: {len(rec.get_seq())} sites")
```

### Reading raw DataSequence records

If you need individual `DataSequence` objects (e.g. to attach to `Node`s):

```python
from PhyNetPy.IO import read_fasta_records

records = read_fasta_records("sequences.fasta")
for ds in records:
    print(ds.get_name(), "->", "".join(ds.get_seq()[:20]), "...")
```

### Grouping sequences by species

When multiple individuals belong to the same species:

```python
grouping = {
    "Great_Apes": ["Homo_sapiens", "Pan_troglodytes", "Gorilla_gorilla"],
    "Old_World_Monkeys": ["Macaca_mulatta"],
}
msa = read_fasta("sequences.fasta", grouping=grouping)
print(f"Number of groups: {msa.groups}")
```

### Writing FASTA

```python
from PhyNetPy.IO import write_fasta

write_fasta(msa, "output.fasta")
# Custom line width (default is 80)
write_fasta(msa, "output_narrow.fasta", line_width=60)
```

---

## VCF Files

The `IO` module reads VCF (Variant Call Format) files and converts genotype calls into allele-count vectors suitable for the BiMarkers/SNP pipeline.

### Reading a VCF

```python
from PhyNetPy.IO import read_vcf

msa = read_vcf("variants.vcf")

for rec in msa.get_records():
    print(f"{rec.get_name()}: {rec.get_seq()}")
# Each sequence is a list of allele counts: ['0', '1', '2', '?', ...]
```

### Inspecting VCF metadata

```python
from PhyNetPy.IO import read_vcf_metadata

meta = read_vcf_metadata("variants.vcf")
print("VCF version:", meta["fileformat"])
print("Samples:", meta["sample_names"])
print("INFO fields:", meta["info_fields"])
```

### VCF with species grouping

Map individual samples to species for the BiMarkers pipeline:

```python
species_map = {
    "SpeciesA": ["Ind1_SpeciesA", "Ind2_SpeciesA"],
    "SpeciesB": ["Ind1_SpeciesB", "Ind2_SpeciesB"],
    "SpeciesC": ["Ind1_SpeciesC", "Ind2_SpeciesC"],
}
msa = read_vcf("variants.vcf", grouping=species_map)
print(f"Species groups: {msa.groups}")
```

### Writing VCF

```python
from PhyNetPy.IO import write_vcf

write_vcf(msa, "output.vcf")
```

---

## Newick Strings

### Parsing a newick string

```python
from PhyNetPy.IO import read_newick

# Simple tree
net = read_newick("((A:0.1,B:0.2):0.3,C:0.4);")
print("Leaves:", [str(l) for l in net.get_leaves()])
```

### Extended newick (networks with reticulations)

PhyNetPy supports the extended newick format where reticulation nodes
are prefixed with `#` and inheritance probabilities are encoded in
`[&gamma=X]` comments:

```python
retic_str = "((A:0.1,(B:0.05)#H1[&gamma=0.7]:0.05):0.2,(#H1[&gamma=0.3]:0.1,(C:0.1,D:0.1):0.05):0.15);"
net = read_newick(retic_str)
print("Leaves:", sorted([str(l) for l in net.get_leaves()]))
# -> ['A', 'B', 'C', 'D']
```

### Converting a Network back to newick

```python
from PhyNetPy.IO import write_newick

newick_str = write_newick(net)
print(newick_str)
```

---

## Newick Files

### Reading multiple trees from a file

Each line in the file should contain one newick string:

```
((A:0.1,B:0.2):0.3,C:0.4);
(((A:0.05,B:0.05):0.1,C:0.15):0.2,D:0.35);
```

```python
from PhyNetPy.IO import read_newick_file

networks = read_newick_file("trees.tre")
for i, net in enumerate(networks):
    print(f"Tree {i+1}: {sorted([str(l) for l in net.get_leaves()])}")
```

### Writing multiple trees to a file

```python
from PhyNetPy.IO import write_newick_file

write_newick_file(networks, "output_trees.tre")
```

---

## Nexus Files

### Reading trees from a nexus file

```python
from PhyNetPy.IO import read_nexus

# Returns a list of Network objects (one per tree in the TREES block)
networks = read_nexus("primates.nex")
for net in networks:
    leaves = sorted([str(l) for l in net.get_leaves()])
    print(f"Tree with leaves: {leaves}")
```

### Reading sequence data (DATA block) from a nexus file

```python
from PhyNetPy.IO import read_nexus_msa

msa = read_nexus_msa("alignment.nex")
for rec in msa.get_records():
    print(f"{rec.get_name()}: {len(rec.get_seq())} sites")
```

### Writing networks to a nexus file

```python
from PhyNetPy.IO import write_nexus

# Basic write
write_nexus(networks, "output.nex")

# With explicit taxa set and PhyloNet commands
write_nexus(
    networks,
    "phylonet_input.nex",
    taxa={"A", "B", "C", "D"},
    tree_prefix="net",
    phylonet_cmds=[
        "InferNetwork_ML (net1) 1 -bl;",
    ]
)
```

The generated nexus file follows this structure:

```
#NEXUS

BEGIN TAXA;
DIMENSIONS NTAX=4;
TAXALABELS
A
B
C
D
;
END;
BEGIN TREES;
Tree net1 = ((A:0.1,B:0.2):0.3,...);
END;
BEGIN PHYLONET;
InferNetwork_ML (net1) 1 -bl;
END;
```

---

## Newick Standard Conversion

Different phylogenetic tools use slightly different extended newick conventions.
PhyNetPy can auto-detect and convert between three standards:

| Standard    | Gamma encoding                       | Example                                       |
|-------------|--------------------------------------|-----------------------------------------------|
| **PhyNetPy** | `[&gamma=0.7]` bracket comment       | `(B:0.05)#H0[&gamma=0.7]:0.05`              |
| **Phylonet** | `::0.7` double-colon suffix          | `(B:0.05)#H0:0.05::0.7`                     |
| **Beast**    | `[&R]` prefix + PhyNetPy annotations | `[&R] ((...)#H0[&gamma=0.7]:0.05,...);`       |

### Auto-detect the standard

```python
from PhyNetPy.IO import detect_newick_standard

print(detect_newick_standard("(B:.05)#H0[&gamma=.7]:.05"))   # "PhyNetPy"
print(detect_newick_standard("(B:.05)#H0:.05::.7"))           # "Phylonet"
print(detect_newick_standard("[&R] ((A,B),C);"))              # "Beast"
```

### Convert between standards

```python
from PhyNetPy.IO import convert_newick

phynetpy_str = "((C:.1,(B:.05)#H0[&gamma=.7]:.05)I1:.1,(A:.1,#H0:.05)I2:.1)I3;"

# Convert to PhyloNet format
phylonet = convert_newick(phynetpy_str, "Phylonet")
print(phylonet)
# ((C:.1,(B:.05)#H0:.05::.7)I1:.1,(A:.1,#H0:.05)I2:.1)I3;

# Convert to BEAST format
beast = convert_newick(phynetpy_str, "Beast")
print(beast)
# [&R] ((C:.1,(B:.05)#H0[&gamma=.7]:.05)I1:.1,(A:.1,#H0:.05)I2:.1)I3;

# Convert back from PhyloNet
back = convert_newick(phylonet, "PhyNetPy")
print(back)
```

---

## Function Reference

### FASTA

| Function | Signature | Returns |
|----------|-----------|---------|
| `read_fasta` | `(filepath, grouping=None, grouping_auto_detect=False)` | `MSA` |
| `read_fasta_records` | `(filepath)` | `list[DataSequence]` |
| `write_fasta` | `(msa, filepath, line_width=80)` | `None` |
| `write_fasta_from_network` | `(network, filepath, line_width=80)` | `None` |

### VCF

| Function | Signature | Returns |
|----------|-----------|---------|
| `read_vcf` | `(filepath, grouping=None, missing_value="?")` | `MSA` |
| `read_vcf_metadata` | `(filepath)` | `dict` |
| `write_vcf` | `(msa, filepath, chrom="chr1", start_pos=1, ref_allele="A", alt_allele="T")` | `None` |

### Newick

| Function | Signature | Returns |
|----------|-----------|---------|
| `read_newick` | `(newick_str)` | `Network` |
| `read_newick_file` | `(filepath)` | `list[Network]` |
| `write_newick` | `(network)` | `str` |
| `write_newick_file` | `(networks, filepath)` | `None` |

### Nexus

| Function | Signature | Returns |
|----------|-----------|---------|
| `read_nexus` | `(filepath, validate_input=False, print_validation_summary=False)` | `list[Network]` |
| `read_nexus_msa` | `(filepath)` | `MSA` |
| `write_nexus` | `(networks, filepath, taxa=None, tree_prefix="net", overwrite=True, phylonet_cmds=None)` | `None` |

### Conversion Utilities

| Function | Signature | Returns |
|----------|-----------|---------|
| `detect_newick_standard` | `(newick_str)` | `str` (`"PhyNetPy"`, `"Phylonet"`, or `"Beast"`) |
| `convert_newick` | `(newick_str, standard="PhyNetPy")` | `str` |

