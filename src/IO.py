#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --                                                              
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##
##  See "LICENSE.txt" for terms and conditions of usage.
##
##  If you use this work or any portion thereof in published work,
##  please cite it as:
##
##     Mark Kessler, Luay Nakhleh. 2025.
##
##############################################################################

""" 
Author : Mark Kessler
Last Stable Edit : 2/6/26
First Included in Version : 1.1.0
Approved for Release: No.

IO module for PhyNetPy. Handles reading and writing of various phylogenetic
file formats, starting with FASTA. Serves as the central I/O hub for all
supported file formats.
"""

from __future__ import annotations
import os
import re
import textwrap
import traceback
import warnings
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from Bio import Phylo, SeqIO
from nexus import NexusReader

from .MSA import DataSequence, MSA
from .Network import Edge, Network, Node


#####################
#### Error Class ####
#####################

class IOError(Exception):
    """
    Exception raised when file I/O operations fail within PhyNetPy.
    """
    def __init__(self, message: str = "An I/O error occurred") -> None:
        """
        Initialize an IOError with a descriptive message.

        Args:
            message (str): Custom error message describing the I/O failure.
        """
        self.message = message
        super().__init__(self.message)


###############################
#### FASTA Reading Functions ##
###############################

def read_fasta_records(filepath: str) -> List[DataSequence]:
    """
    Read a FASTA file and return a list of DataSequence objects.
    
    This is the lower-level reader that returns raw DataSequence objects 
    without wrapping them in an MSA. Useful for attaching sequences directly 
    to Node objects in an existing Network (via Node.set_seq()).

    A FASTA file looks like:
        >sequence_name_1
        ATCGATCGATCG...
        >sequence_name_2
        GCTAGCTAGCTA...

    Each record becomes a DataSequence where:
        - name = the FASTA header (sequence ID)
        - seq  = list of characters from the sequence string

    Args:
        filepath (str): Path to a FASTA file (.fasta, .fas, .fa, .fna, .ffn, 
                         .faa).

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If BioPython cannot parse the file or it contains no 
                 valid sequences.

    Returns:
        list[DataSequence]: A list of DataSequence objects, one per FASTA 
                            record.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"FASTA file not found: {filepath}")
    
    records: List[DataSequence] = []
    
    try:
        parsed = list(SeqIO.parse(filepath, "fasta"))
    except Exception as e:
        raise IOError(f"Failed to parse FASTA file '{filepath}': {str(e)}")
    
    if len(parsed) == 0:
        raise IOError(
            f"No valid FASTA sequences found in '{filepath}'. "
            "Ensure the file starts with '>' header lines followed by "
            "sequence data."
        )
    
    for idx, bio_rec in enumerate(parsed):
        seq_chars: list = list(str(bio_rec.seq))
        name: str = bio_rec.id if bio_rec.id else f"seq_{idx}"
        data_seq = DataSequence(seq_chars, name, gid=idx)
        records.append(data_seq)
    
    return records


def read_fasta(
    filepath: str,
    grouping: Optional[Dict[str, list]] = None,
    grouping_auto_detect: bool = False
) -> MSA:
    """
    Read a FASTA file and return an MSA object containing all sequences.

    This function parses a FASTA file, converts each record into a 
    DataSequence, and wraps them in an MSA for downstream phylogenetic 
    analyses such as distance calculations, alignment inspection, or 
    model-based inference.

    Args:
        filepath (str): Path to a FASTA file (.fasta, .fas, .fa, .fna, 
                         .ffn, .faa).
        grouping (dict[str, list], optional): A mapping from group names to 
                                              lists of sequence names that 
                                              belong to that group. If 
                                              provided, sequences will be 
                                              assigned group IDs accordingly. 
                                              Defaults to None.
        grouping_auto_detect (bool, optional): If True, attempt to 
                                               automatically group sequences 
                                               by name similarity. 
                                               Defaults to False.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be parsed or contains no valid sequences.

    Returns:
        MSA: A Multiple Sequence Alignment object containing all parsed 
             sequences.
    """
    records = read_fasta_records(filepath)
    
    # Build the MSA from the data list
    msa = MSA(data=records)
    
    # Apply grouping if provided
    if grouping is not None:
        # Rebuild group assignments based on the provided grouping map
        gid = 0
        msa.grouping = grouping
        msa.hash = {}
        msa.name2gid = {}
        
        for group_name, members in grouping.items():
            msa.name2gid[group_name] = gid
            msa.hash[gid] = []
            
            for rec in records:
                if rec.get_name() in members:
                    rec.gid = gid
                    msa.hash[gid].append(rec)
            
            gid += 1
        
        msa.groups = len(grouping)
    elif grouping_auto_detect:
        # Use the MSA's retroactive grouping to auto-detect groups
        msa.retroactive_group()
        msa.groups = len(msa.hash)
    
    return msa


###############################
#### FASTA Writing Functions ##
###############################

FASTA_LINE_WIDTH: int = 80
"""Standard FASTA line width for sequence wrapping."""


def write_fasta(msa: MSA, filepath: str, line_width: int = FASTA_LINE_WIDTH) -> None:
    """
    Write an MSA object to a FASTA file.

    Each DataSequence in the MSA is written as a FASTA record:
        >sequence_name
        ATCGATCG...  (wrapped at line_width characters)

    Args:
        msa (MSA): The Multiple Sequence Alignment to write.
        filepath (str): The output file path. Will be created or overwritten.
        line_width (int, optional): Number of characters per sequence line. 
                                     Standard FASTA convention is 80. 
                                     Defaults to 80.

    Raises:
        IOError: If the MSA has no records to write, or if the file 
                 cannot be written.
        ValueError: If line_width is less than 1.

    Returns:
        None
    """
    if line_width < 1:
        raise ValueError("line_width must be >= 1")
    
    records = msa.get_records()
    if len(records) == 0:
        raise IOError("Cannot write an empty MSA to FASTA: no records present.")
    
    try:
        with open(filepath, 'w') as f:
            for rec in records:
                # Write the header line
                f.write(f">{rec.get_name()}\n")
                
                # Join the sequence characters into a string
                seq_str = "".join(str(c) for c in rec.get_seq())
                
                # Wrap the sequence at line_width characters
                wrapped = textwrap.fill(seq_str, width=line_width, 
                                        break_on_hyphens=False,
                                        break_long_words=True)
                f.write(wrapped + "\n")
    except OSError as e:
        raise IOError(f"Failed to write FASTA file '{filepath}': {str(e)}")


def write_fasta_from_network(
    network: Network, 
    filepath: str, 
    line_width: int = FASTA_LINE_WIDTH
) -> None:
    """
    Extract sequences from the leaf nodes of a Network and write them to a 
    FASTA file.

    Only leaf nodes that have an associated DataSequence (set via 
    Node.set_seq()) will be written. The node label becomes the FASTA 
    header, and the attached sequence becomes the FASTA sequence.

    This is useful when a Network has been annotated with molecular data 
    and the user wants to export just the sequence data.

    Args:
        network (Network): A phylogenetic network whose leaf nodes may 
                           carry DataSequence objects.
        filepath (str): The output FASTA file path.
        line_width (int, optional): Characters per line for sequence 
                                     wrapping. Defaults to 80.

    Raises:
        IOError: If no leaf nodes in the network have sequence data 
                 attached, or if the file cannot be written.
        ValueError: If line_width is less than 1.

    Returns:
        None
    """
    if line_width < 1:
        raise ValueError("line_width must be >= 1")
    
    leaves = network.get_leaves()
    
    # Collect leaf nodes that have sequence data
    seq_records: List[DataSequence] = []
    for leaf in leaves:
        seq = leaf.get_seq()
        if seq is not None:
            # Use the node label as the sequence name if the DataSequence 
            # doesn't already have one
            if seq.get_name() == "" or seq.get_name() is None:
                seq.name = str(leaf)
            seq_records.append(seq)
    
    if len(seq_records) == 0:
        raise IOError(
            "No leaf nodes in the network have sequence data attached. "
            "Use Node.set_seq(DataSequence) to attach sequences before "
            "writing."
        )
    
    # Build a temporary MSA to reuse the write_fasta function
    temp_msa = MSA(data=seq_records)
    write_fasta(temp_msa, filepath, line_width=line_width)


############################
#### VCF Reading Functions #
############################

def _parse_vcf_genotype(gt_str: str) -> Optional[int]:
    """
    Parse a VCF genotype string into an allele count (number of ALT alleles).

    Supported formats:
        - "0/0" or "0|0" -> 0  (homozygous reference)
        - "0/1" or "0|1" -> 1  (heterozygous)
        - "1/1" or "1|1" -> 2  (homozygous alternate)
        - "./." or ".|." -> None  (missing data)

    For polyploid genotypes (e.g. "0/0/1"), the count is the sum of all
    allele values.

    Args:
        gt_str (str): Genotype field from a VCF record (e.g. "0/1").

    Returns:
        int or None: The ALT allele count, or None if the genotype is 
                     missing.
    """
    # Strip any additional format fields (GT may be followed by :DP:GQ etc.)
    gt_field = gt_str.split(":")[0]
    
    # Determine separator (phased "|" or unphased "/")
    if "|" in gt_field:
        alleles = gt_field.split("|")
    elif "/" in gt_field:
        alleles = gt_field.split("/")
    else:
        # Single allele (haploid)
        alleles = [gt_field]
    
    # Check for missing data
    if any(a == "." for a in alleles):
        return None
    
    try:
        return sum(int(a) for a in alleles)
    except ValueError:
        return None


def read_vcf(
    filepath: str,
    grouping: Optional[Dict[str, list]] = None,
    missing_value: str = "?"
) -> MSA:
    """
    Read a VCF (Variant Call Format) file and return an MSA object.

    Each sample in the VCF becomes a DataSequence whose sequence is the 
    vector of ALT allele counts across all variant sites. This maps 
    directly to the SNP/BiMarkers pipeline used in PhyNetPy.

    A typical VCF file looks like::

        ##fileformat=VCFv4.1
        ##INFO=<...>
        #CHROM  POS  ID  REF  ALT  QUAL  FILTER  INFO  FORMAT  Samp1  Samp2
        chr1    100  .   A    T    30    PASS    .     GT      0/0    0/1
        chr1    200  .   G    C    50    PASS    .     GT      1/1    0/1

    Genotype encoding:
        - 0/0 -> 0  (homozygous reference, 0 copies of ALT allele)
        - 0/1 -> 1  (heterozygous, 1 copy of ALT allele)
        - 1/1 -> 2  (homozygous alternate, 2 copies of ALT allele)
        - ./. -> missing_value  (missing genotype)

    Args:
        filepath (str): Path to a VCF file (.vcf).
        grouping (dict[str, list], optional): A mapping from group/species 
                                              names to lists of sample names 
                                              that belong to that group. 
                                              Used for the BiMarkers 
                                              pipeline where multiple 
                                              individuals map to a single 
                                              species. Defaults to None.
        missing_value (str, optional): The character to use for missing 
                                       genotype data (./.). Defaults to "?".

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be parsed or contains no variant data.

    Returns:
        MSA: A Multiple Sequence Alignment where each DataSequence 
             represents one sample's genotype vector across all sites.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"VCF file not found: {filepath}")
    
    sample_names: List[str] = []
    # Each sample accumulates a list of allele counts across sites
    sample_genotypes: Dict[str, List] = {}
    metadata_lines: List[str] = []
    num_variants: int = 0
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                
                # Meta-information lines
                if line.startswith("##"):
                    metadata_lines.append(line)
                    continue
                
                # Header line with sample names
                if line.startswith("#CHROM") or line.startswith("#chrom"):
                    fields = line.split("\t")
                    # Columns 0-8 are fixed: CHROM POS ID REF ALT QUAL 
                    # FILTER INFO FORMAT
                    # Columns 9+ are sample names
                    if len(fields) < 10:
                        raise IOError(
                            "VCF header line has fewer than 10 columns. "
                            "Expected at least: #CHROM POS ID REF ALT QUAL "
                            "FILTER INFO FORMAT <sample1>"
                        )
                    sample_names = fields[9:]
                    for sname in sample_names:
                        sample_genotypes[sname] = []
                    continue
                
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Data lines
                fields = line.split("\t")
                if len(fields) < 10:
                    continue  # Malformed line, skip
                
                # Find the GT field position in FORMAT
                fmt = fields[8]
                fmt_fields = fmt.split(":")
                try:
                    gt_index = fmt_fields.index("GT")
                except ValueError:
                    # No GT field in this record, skip
                    warnings.warn(
                        f"VCF line at position {fields[1]} has no GT "
                        f"field in FORMAT column; skipping."
                    )
                    continue
                
                num_variants += 1
                
                # Parse each sample's genotype
                for i, sname in enumerate(sample_names):
                    if 9 + i < len(fields):
                        sample_field = fields[9 + i]
                        sample_fmt = sample_field.split(":")
                        if gt_index < len(sample_fmt):
                            gt = _parse_vcf_genotype(sample_fmt[gt_index])
                        else:
                            gt = None
                    else:
                        gt = None
                    
                    if gt is not None:
                        sample_genotypes[sname].append(str(gt))
                    else:
                        sample_genotypes[sname].append(missing_value)
    
    except OSError as e:
        raise IOError(f"Failed to read VCF file '{filepath}': {str(e)}")
    
    if not sample_names:
        raise IOError(
            f"No sample columns found in VCF file '{filepath}'. "
            "Ensure the file has a #CHROM header line with sample names."
        )
    
    if num_variants == 0:
        raise IOError(
            f"No variant records found in VCF file '{filepath}'."
        )
    
    # Build DataSequence objects
    records: List[DataSequence] = []
    
    if grouping is not None:
        # Assign group IDs based on the grouping map
        gid_map: Dict[str, int] = {}
        gid = 0
        for group_name, members in grouping.items():
            for member in members:
                gid_map[member] = gid
            gid += 1
        
        for sname in sample_names:
            seq = sample_genotypes[sname]
            assigned_gid = gid_map.get(sname, -1)
            if assigned_gid == -1:
                warnings.warn(
                    f"Sample '{sname}' not found in grouping map; "
                    f"assigned to its own group."
                )
                assigned_gid = gid
                gid += 1
            records.append(DataSequence(seq, sname, gid=assigned_gid))
    else:
        for idx, sname in enumerate(sample_names):
            seq = sample_genotypes[sname]
            records.append(DataSequence(seq, sname, gid=idx))
    
    msa = MSA(data=records)
    
    # Apply grouping to MSA if provided
    if grouping is not None:
        msa.grouping = grouping
        msa.groups = len(grouping)
    
    return msa


def read_vcf_metadata(filepath: str) -> Dict[str, Any]:
    """
    Read only the metadata and header from a VCF file without loading 
    all variant data. Useful for inspecting what samples and fields 
    are available before a full parse.

    Args:
        filepath (str): Path to a VCF file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read.

    Returns:
        dict[str, Any]: A dictionary containing:
            - "fileformat": The VCF version string
            - "metadata_lines": List of all ## header lines
            - "sample_names": List of sample column names
            - "info_fields": List of INFO field IDs
            - "format_fields": List of FORMAT field IDs
            - "filter_fields": List of FILTER field IDs
            - "contig_fields": List of contig IDs
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"VCF file not found: {filepath}")
    
    result: Dict[str, Any] = {
        "fileformat": "",
        "metadata_lines": [],
        "sample_names": [],
        "info_fields": [],
        "format_fields": [],
        "filter_fields": [],
        "contig_fields": [],
    }
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                
                if line.startswith("##fileformat"):
                    result["fileformat"] = line.split("=", 1)[1]
                    result["metadata_lines"].append(line)
                elif line.startswith("##INFO"):
                    result["metadata_lines"].append(line)
                    match = re.search(r'ID=([^,>]+)', line)
                    if match:
                        result["info_fields"].append(match.group(1))
                elif line.startswith("##FORMAT"):
                    result["metadata_lines"].append(line)
                    match = re.search(r'ID=([^,>]+)', line)
                    if match:
                        result["format_fields"].append(match.group(1))
                elif line.startswith("##FILTER"):
                    result["metadata_lines"].append(line)
                    match = re.search(r'ID=([^,>]+)', line)
                    if match:
                        result["filter_fields"].append(match.group(1))
                elif line.startswith("##contig"):
                    result["metadata_lines"].append(line)
                    match = re.search(r'ID=([^,>]+)', line)
                    if match:
                        result["contig_fields"].append(match.group(1))
                elif line.startswith("##"):
                    result["metadata_lines"].append(line)
                elif line.startswith("#CHROM") or line.startswith("#chrom"):
                    fields = line.split("\t")
                    if len(fields) >= 10:
                        result["sample_names"] = fields[9:]
                    break  # Done with header
    except OSError as e:
        raise IOError(f"Failed to read VCF file '{filepath}': {str(e)}")
    
    return result


############################
#### VCF Writing Functions #
############################

def write_vcf(
    msa: MSA,
    filepath: str,
    chrom: str = "chr1",
    start_pos: int = 1,
    ref_allele: str = "A",
    alt_allele: str = "T"
) -> None:
    """
    Write an MSA of SNP/allele-count data to a simplified VCF file.

    This produces a minimal VCF where each site in the MSA becomes a 
    variant record, and each DataSequence becomes a sample column. The 
    allele count values (0, 1, 2, ...) are converted back to VCF genotype 
    notation (e.g., 0/0, 0/1, 1/1).

    Note: Because the MSA does not store chromosome position, reference 
    alleles, or other VCF-specific metadata, this output is a simplified 
    reconstruction. It is suitable for round-tripping SNP data or creating 
    test files, but will not preserve full VCF metadata from an original 
    file.

    Args:
        msa (MSA): The MSA containing allele count data (values like 
                    0, 1, 2 per site).
        filepath (str): The output VCF file path.
        chrom (str, optional): Chromosome name for all records. 
                                Defaults to "chr1".
        start_pos (int, optional): Starting position for the first 
                                    variant. Each subsequent variant 
                                    increments by 1. Defaults to 1.
        ref_allele (str, optional): Reference allele character. 
                                     Defaults to "A".
        alt_allele (str, optional): Alternate allele character. 
                                     Defaults to "T".

    Raises:
        IOError: If the MSA has no records, or the file cannot be written.

    Returns:
        None
    """
    records = msa.get_records()
    if len(records) == 0:
        raise IOError("Cannot write an empty MSA to VCF: no records present.")
    
    sample_names = [rec.get_name() for rec in records]
    num_sites = len(records[0].get_seq())
    
    try:
        with open(filepath, 'w') as f:
            # Write VCF header
            f.write("##fileformat=VCFv4.1\n")
            f.write('##FORMAT=<ID=GT,Number=1,Type=String,'
                     'Description="Genotype">\n')
            f.write("##source=PhyNetPy\n")
            
            # Write column header
            header_cols = [
                "#CHROM", "POS", "ID", "REF", "ALT", 
                "QUAL", "FILTER", "INFO", "FORMAT"
            ] + sample_names
            f.write("\t".join(header_cols) + "\n")
            
            # Write variant records
            for site_idx in range(num_sites):
                pos = start_pos + site_idx
                
                # Build genotype strings for each sample
                genotypes: List[str] = []
                for rec in records:
                    seq = rec.get_seq()
                    if site_idx < len(seq):
                        val = seq[site_idx]
                        gt = _allele_count_to_gt(val)
                    else:
                        gt = "./."
                    genotypes.append(gt)
                
                fields = [
                    chrom,
                    str(pos),
                    ".",
                    ref_allele,
                    alt_allele,
                    ".",
                    "PASS",
                    ".",
                    "GT"
                ] + genotypes
                
                f.write("\t".join(fields) + "\n")
    
    except OSError as e:
        raise IOError(f"Failed to write VCF file '{filepath}': {str(e)}")


def _allele_count_to_gt(value: Any) -> str:
    """
    Convert an allele count value back to VCF genotype notation.

    Assumes diploid (2 alleles). For higher ploidy, the genotype is 
    constructed by distributing ALT alleles across allele slots.

    Args:
        value (Any): The allele count (0, 1, 2, ...) or a missing 
                      data marker ("?", ".", etc.).

    Returns:
        str: A VCF genotype string (e.g. "0/0", "0/1", "1/1", "./.").
    """
    # Handle missing data
    if value in ("?", ".", "-", None, "None"):
        return "./."
    
    try:
        count = int(value)
    except (ValueError, TypeError):
        return "./."
    
    # Diploid genotype reconstruction
    if count == 0:
        return "0/0"
    elif count == 1:
        return "0/1"
    elif count == 2:
        return "1/1"
    else:
        # Higher ploidy: distribute ALT alleles
        # e.g., count=3 with ploidy inferred as count -> "0/1/1/1"
        # For simplicity, assume count is num ALT out of ploidy = count
        # This is a best-effort reconstruction
        num_alt = min(count, count)
        num_ref = 0
        alleles = ["0"] * num_ref + ["1"] * num_alt
        return "/".join(alleles)


##################################
#### Newick Reading Functions ####
##################################

def _merge_attributes(inheritances: Dict[str, Dict],
                      parsed_node: Node,
                      attr1: dict,
                      attr2: dict) -> dict:
    """
    Given two attribute dictionaries, combine them into one, taking the 
    union of the two.  Used when a reticulation node is encountered from 
    a second parent.

    Args:
        inheritances (dict[str, dict]): A mapping from node names to their 
                                        gamma entries.
        parsed_node (Node): The node for which to combine attributes parsed 
                            for different parents. Should be a reticulation 
                            node.
        attr1 (dict): The attribute dict derived from the first parent.
        attr2 (dict): The attribute dict derived from the second parent.

    Returns:
        dict: A combined attribute dictionary.
    """
    final_attr: Dict[Any, Any] = {}
    if "eventType" in attr1:
        final_attr["eventType"] = attr1["eventType"]
    if "index" in attr1:
        final_attr["index"] = attr1["index"]

    if parsed_node.label in inheritances:
        final_attr["gamma"] = inheritances[parsed_node.label]
    return final_attr


def _parse_reticulation_attributes(attr_str: str) -> Tuple[str, int]:
    """
    Takes the formatting string from extended newick grammar and parses 
    it into the event type and index.

    Examples:
        - "H1"    -> ("Hybridization", 1)
        - "LGT21" -> ("Lateral Gene Transfer", 21)
        - "R3"    -> ("Recombination", 3)

    Args:
        attr_str (str): A node name suffix carrying reticulation event 
                        information (after the '#').

    Raises:
        IOError: If the label format is invalid.

    Returns:
        tuple[str, int]: (event_type, index)
    """
    if len(attr_str) < 2:
        raise IOError("Reticulation event label formatting incorrect")

    index_lookup = 0

    if attr_str[0] == "R":
        event = "Recombination"
        index_lookup = 1
    elif attr_str[0] == "H":
        event = "Hybridization"
        index_lookup = 1
    elif attr_str[0] == "L":
        if len(attr_str) >= 3 and attr_str[1] == "G" and attr_str[2] == "T":
            event = "Lateral Gene Transfer"
            index_lookup = 3
        else:
            raise IOError("Invalid reticulation label format (event error)")
    else:
        raise IOError("Invalid reticulation label format (event error)")

    try:
        num = int(attr_str[index_lookup:])
        return event, num
    except ValueError:
        raise IOError("Invalid reticulation label format (number error)")


class _NewickTreeBuilder:
    """
    Internal helper that converts a BioPython Tree object into a PhyNetPy 
    Network. This replicates the logic from NetworkParser.parse_tree_block 
    and its associated methods but in a standalone, reusable form.
    """

    def __init__(self) -> None:
        self._internal_count: int = 0
        self._inheritance: Dict[str, Dict] = {}

    def build(self, tree: Any) -> Network:
        """
        Given a BioPython Tree object, walk through it and build a 
        PhyNetPy Network.

        Args:
            tree (Any): A BioPython Phylo.BaseTree.Tree object.

        Returns:
            Network: A PhyNetPy Network with the same topology, names, 
                     and branch lengths as the input tree.
        """
        # Build a parent dictionary from the BioPython tree
        parents: Dict[Any, Any] = {}
        for clade in tree.find_clades(order="level"):
            for child in clade:
                parents[child] = clade

        net = Network()

        for node, par in parents.items():
            parent_node = self._parse_parent(par, net)
            self._parse_child(node, net, parent_node)

        return net

    def _parse_comment(self, node: Any, parent: Node) -> dict:
        """
        Examine the comment block of a BioPython clade to extract 
        reticulation attributes (gamma/inheritance probabilities).

        Args:
            node (Any): A BioPython clade node.
            parent (Node): The PhyNetPy parent node.

        Returns:
            dict: An attribute dictionary for the node.
        """
        attr: dict = {}
        if node.name is None or node.name[0] != "#":
            if node.comment is not None:
                attr["comment"] = node.comment
            return attr

        # Reticulation node
        event, num = _parse_reticulation_attributes(node.name.split("#")[1])
        attr["eventType"] = event
        attr["index"] = num

        if node.comment is not None:
            if node.comment.split("=")[0] == "&gamma":
                gamma = float(node.comment.split("=")[1])
                attr["gamma"] = {parent.label: [gamma, node.branch_length]}

                if node.name in self._inheritance:
                    for par, info in self._inheritance[node.name].items():
                        if info[0] == -1:
                            old_info = [1 - gamma, info[1]]
                            self._inheritance[node.name] = {
                                par: old_info,
                                parent.label: [gamma, node.branch_length]
                            }
                        else:
                            if info[0] + gamma != 1:
                                raise IOError(
                                    "Gamma values provided in newick string "
                                    "do not add to 1"
                                )
                            self._inheritance[node.name] = {
                                par: info,
                                parent.label: [gamma, node.branch_length]
                            }
                        break
                else:
                    self._inheritance[node.name] = {
                        parent.label: [gamma, node.branch_length]
                    }
            else:
                attr["comment"] = node.comment
        else:
            # No comment on a reticulation node
            if node.name in self._inheritance:
                for par, info in self._inheritance[node.name].items():
                    if info[0] == -1:
                        gammas = {
                            par: [0.5, info[1]],
                            parent.label: [0.5, node.branch_length]
                        }
                        self._inheritance[node.name] = gammas
                    else:
                        self._inheritance[node.name] = {
                            par: info,
                            parent.label: [1 - info[0], node.branch_length]
                        }
                    break
            else:
                self._inheritance[node.name] = {
                    parent.label: [-1, node.branch_length]
                }

        return attr

    def _parse_parent(self, node: Any, network: Network,
                      parent: Optional[Node] = None) -> Node:
        """
        Process a parent node from the BioPython tree.

        Args:
            node (Any): A BioPython clade.
            network (Network): The partially built PhyNetPy Network.
            parent (Node, optional): The parent of this node. Defaults 
                                     to None (root).

        Returns:
            Node: The PhyNetPy Node for this clade.
        """
        parsed_node: Optional[Node] = network.has_node_named(node.name)

        if parsed_node is not None:
            return parsed_node

        if node.name is None:
            node.name = "Internal" + str(self._internal_count)
            self._internal_count += 1

        parsed_node = Node(name=node.name)

        if parent is None:
            parsed_node.set_time(0)
        else:
            par_time = parent.get_time()
            if node.branch_length is not None:
                parsed_node.set_time(par_time + node.branch_length)
            else:
                warnings.warn(
                    "No branch length provided for node; setting to 1."
                )
                parsed_node.set_time(par_time + 1)

        if node.name[0] == "#":
            parsed_node.set_is_reticulation(True)

        if parent is not None:
            parsed_node.set_attributes(self._parse_comment(node, parent))

        network.add_nodes(parsed_node)
        return parsed_node

    def _parse_child(self, node: Any, network: Network,
                     parent: Node) -> Node:
        """
        Process a child node from the BioPython tree.

        Args:
            node (Any): A BioPython clade.
            network (Network): The partially built PhyNetPy Network.
            parent (Node): The already-processed parent node.

        Returns:
            Node: The PhyNetPy Node for this clade.
        """
        parsed_node: Optional[Node] = network.has_node_named(node.name)

        if parsed_node is not None:
            more_attr = self._parse_comment(node, parent)
            parsed_node.set_attributes(
                _merge_attributes(
                    self._inheritance, parsed_node,
                    more_attr, parsed_node.get_attributes()
                )
            )
        else:
            if node.name is None:
                node.name = "Internal" + str(self._internal_count)
                self._internal_count += 1

            parsed_node = Node(name=node.name)

            if node.name[0] == "#":
                parsed_node.set_is_reticulation(True)

            if node.branch_length is not None:
                parsed_node.set_time(parent.get_time() + node.branch_length)
            else:
                warnings.warn(
                    "No branch length provided for node; setting to 1."
                )
                parsed_node.set_time(parent.get_time() + 1)

            parsed_node.set_attributes(self._parse_comment(node, parent))
            network.add_nodes(parsed_node)

        # Create edge from parent to child
        new_edge = Edge(parent, parsed_node)

        if node.branch_length is not None:
            new_edge.set_length(node.branch_length)
        else:
            new_edge.set_length(1)

        # Set gamma if applicable
        inheritance_prob = parsed_node.attribute_value("gamma")
        if inheritance_prob is not None:
            gamma_value = inheritance_prob[parent.label][0]
            new_edge.set_gamma(gamma_value)

        network.add_edges(new_edge)
        return parsed_node


def read_newick(newick_str: str) -> Network:
    """
    Parse a single newick/extended-newick string into a PhyNetPy Network.

    Supports standard newick features (branch lengths, internal node names) 
    as well as the extended newick format for phylogenetic networks 
    (reticulation nodes prefixed with '#', gamma inheritance comments).

    Examples of accepted strings::

        ((A:0.1,B:0.2):0.3,C:0.4);
        ((A:0.1,(B:0.2)#H1:0.3):0.4,(#H1:0.5,C:0.6):0.7);

    Args:
        newick_str (str): A newick or extended-newick string. Trailing 
                          semicolons are handled automatically.

    Raises:
        IOError: If the string cannot be parsed.

    Returns:
        Network: A PhyNetPy Network object with the same topology, names, 
                 and branch lengths as described in the newick string.
    """
    newick_str = newick_str.strip()
    if not newick_str:
        raise IOError("Cannot parse an empty newick string.")

    try:
        handle = StringIO(newick_str)
        tree = Phylo.read(handle, "newick")
    except Exception as e:
        raise IOError(f"Failed to parse newick string: {str(e)}")

    builder = _NewickTreeBuilder()
    return builder.build(tree)


def read_newick_file(filepath: str) -> List[Network]:
    """
    Read a file containing one or more newick strings (one per line) 
    and parse each into a PhyNetPy Network.

    Blank lines and lines starting with '#' are skipped.

    Args:
        filepath (str): Path to a file containing newick strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If no valid newick strings are found, or parsing fails.

    Returns:
        list[Network]: A list of parsed Network objects, one per newick 
                       string.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Newick file not found: {filepath}")

    networks: List[Network] = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                try:
                    net = read_newick(line)
                    networks.append(net)
                except Exception as e:
                    warnings.warn(
                        f"Line {line_num}: Failed to parse newick string "
                        f"'{line[:50]}...': {str(e)}"
                    )
    except OSError as e:
        raise IOError(f"Failed to read newick file '{filepath}': {str(e)}")

    if not networks:
        raise IOError(
            f"No valid newick strings found in '{filepath}'."
        )

    return networks


##################################
#### Newick Writing Functions ####
##################################

def write_newick(network: Network) -> str:
    """
    Convert a PhyNetPy Network into a newick string.

    Delegates to the Network's built-in ``newick()`` method, which 
    produces extended-newick notation for networks with reticulation 
    nodes.

    Args:
        network (Network): A PhyNetPy Network object.

    Returns:
        str: The newick representation of the network, ending with ';'.
    """
    return network.newick()


def write_newick_file(
    networks: List[Network],
    filepath: str
) -> None:
    """
    Write one or more Networks to a file as newick strings, one per line.

    Args:
        networks (list[Network]): Networks to write.
        filepath (str): Output file path. Will be created or overwritten.

    Raises:
        IOError: If the list is empty or the file cannot be written.

    Returns:
        None
    """
    if not networks:
        raise IOError("No networks provided to write.")

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            for net in networks:
                f.write(write_newick(net) + "\n")
    except OSError as e:
        raise IOError(f"Failed to write newick file '{filepath}': {str(e)}")


##################################
#### Nexus Reading Functions #####
##################################

def read_nexus(
    filepath: str,
    validate_input: bool = False,
    print_validation_summary: bool = False
) -> List[Network]:
    """
    Read a nexus file and parse all trees/networks in the TREES block 
    into PhyNetPy Network objects.

    This replicates the core functionality of ``NetworkParser`` as a 
    standalone function, making it easy to call without instantiating a 
    class.

    A typical nexus file looks like::

        #NEXUS
        BEGIN TAXA;
            DIMENSIONS NTAX=3;
            TAXALABELS A B C;
        END;
        BEGIN TREES;
            Tree t1 = ((A:0.1,B:0.2):0.3,C:0.4);
            Tree t2 = ((B:0.1,C:0.2):0.3,A:0.4);
        END;

    Args:
        filepath (str): Path to a nexus file (.nex, .nexus).
        validate_input (bool, optional): If True, run NexusValidator on 
                                          the file before parsing.  
                                          Defaults to False.
        print_validation_summary (bool, optional): If True and 
                                                    validate_input is 
                                                    True, print the 
                                                    validation summary.  
                                                    Defaults to False.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be parsed or contains no trees.

    Returns:
        list[Network]: A list of Network objects, one per tree/network 
                       in the file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Nexus file not found: {filepath}")

    # Optional validation
    validation_summary = None
    if validate_input:
        try:
            from .Validation import NexusValidator
            validator = NexusValidator()
            validation_summary = validator.validate(filepath)
            if print_validation_summary:
                print(validation_summary)
        except Exception as e:
            warnings.warn(f"Validation failed: {str(e)}")

    # Parse using NexusReader
    try:
        reader = NexusReader.from_file(filepath)
    except Exception as e:
        traceback.print_exc()
        raise IOError(
            f"NexusReader library could not find or parse '{filepath}': "
            f"{str(e)}"
        )

    if reader.trees is None:
        raise IOError(f"No trees listed in nexus file '{filepath}'.")

    networks: List[Network] = []
    name_map: Dict[str, Network] = {}

    for t in reader.trees:
        tree_str = str(t)
        parts = tree_str.split("=", 1)
        name = parts[0].split()[1] if len(parts[0].split()) > 1 else f"tree_{len(networks)}"
        newick_part = parts[1] if len(parts) > 1 else tree_str

        handle = StringIO(newick_part)
        try:
            bio_tree = Phylo.read(handle, "newick")
        except Exception as e:
            warnings.warn(
                f"Failed to parse tree '{name}' in nexus file: {str(e)}"
            )
            continue

        builder = _NewickTreeBuilder()
        network = builder.build(bio_tree)
        networks.append(network)
        name_map[name] = network

    if not networks:
        raise IOError(
            f"No valid trees could be parsed from nexus file '{filepath}'."
        )

    return networks


def read_nexus_msa(filepath: str) -> MSA:
    """
    Read the sequence data (DATA/CHARACTERS block) from a nexus file 
    and return it as an MSA object.

    This is a convenience wrapper around the MSA constructor's built-in 
    nexus parsing. Use this when you want the alignment data rather than 
    the tree topology.

    Args:
        filepath (str): Path to a nexus file containing a DATA or 
                        CHARACTERS block.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If no sequence data is found.

    Returns:
        MSA: The parsed Multiple Sequence Alignment.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Nexus file not found: {filepath}")

    try:
        return MSA(filename=filepath)
    except Exception as e:
        raise IOError(
            f"Failed to read sequence data from nexus file '{filepath}': "
            f"{str(e)}"
        )


##################################
#### Nexus Writing Functions #####
##################################

def _get_newick_labels(newick_str: str) -> Set[str]:
    """
    Given a newick string, extract all unique taxa labels present in 
    the string.

    Args:
        newick_str (str): A newick format string.

    Returns:
        set[str]: A set of unique taxa labels.
    """
    label_set: Set[str] = set()
    pos: int = 0
    cur_label = ""

    while pos < len(newick_str):
        if newick_str[pos] in {")", "(", ","}:
            if len(cur_label) > 0:
                cleaned = cur_label.split(":", maxsplit=1)[0].split("[")[0].strip()
                if cleaned:
                    label_set.add(cleaned)
            cur_label = ""
        else:
            cur_label += newick_str[pos]
        pos += 1

    # Handle any remaining label at the end
    if cur_label:
        cleaned = cur_label.split(":", maxsplit=1)[0].split("[")[0].strip().rstrip(";")
        if cleaned:
            label_set.add(cleaned)

    return label_set


def write_nexus(
    networks: List[Network],
    filepath: str,
    taxa: Optional[Set[str]] = None,
    tree_prefix: str = "net",
    overwrite: bool = True,
    phylonet_cmds: Optional[List[str]] = None
) -> None:
    """
    Write one or more Networks to a nexus file with TAXA and TREES blocks.

    This replicates the functionality of the ``NexusTemplate`` class 
    as a standalone function.

    The generated file follows the standard nexus format::

        #NEXUS

        BEGIN TAXA;
        DIMENSIONS NTAX=3;
        TAXALABELS
        A
        B
        C
        ;
        END;
        BEGIN TREES;
        Tree net1 = ((A:0.1,B:0.2):0.3,C:0.4);
        Tree net2 = ...;
        END;

    Args:
        networks (list[Network]): The networks to write.
        filepath (str): Output file path. 
        taxa (set[str], optional): An explicit set of taxa labels. If 
                                    None, taxa are inferred from the 
                                    newick strings. Defaults to None.
        tree_prefix (str, optional): Label prefix for each tree line. 
                                      Defaults to "net".
        overwrite (bool, optional): If False, raises IOError if the file 
                                     already exists. Defaults to True.
        phylonet_cmds (list[str], optional): A list of PhyloNet commands 
                                             to include in a PHYLONET 
                                             block. Defaults to None.

    Raises:
        IOError: If the list is empty, or the file cannot be written, 
                 or the file already exists and overwrite is False.

    Returns:
        None
    """
    if not networks:
        raise IOError("No networks provided to write.")

    if not overwrite and os.path.exists(filepath):
        raise IOError(f"File already exists: {filepath}")

    # Generate newick strings and collect taxa
    newick_strings: List[str] = []
    all_taxa: Set[str] = set()

    for net in networks:
        nwk = write_newick(net)
        newick_strings.append(nwk)
        all_taxa |= _get_newick_labels(nwk)

    # Use explicitly provided taxa if given
    if taxa is not None:
        all_taxa = taxa

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("#NEXUS\n\n")

            # TAXA block
            f.write("BEGIN TAXA;\n")
            f.write(f"DIMENSIONS NTAX={len(all_taxa)};\n")
            f.write("TAXALABELS\n")
            for taxon in sorted(all_taxa):
                f.write(f"{taxon}\n")
            f.write(";\n")
            f.write("END;\n")

            # TREES block
            f.write("BEGIN TREES;\n")
            for idx, nwk in enumerate(newick_strings, start=1):
                f.write(f"Tree {tree_prefix}{idx} = {nwk}\n")
            f.write("END;\n")

            # Optional PHYLONET block
            if phylonet_cmds:
                f.write("BEGIN PHYLONET;\n")
                for cmd in phylonet_cmds:
                    f.write(cmd + "\n")
                f.write("END;\n")

    except OSError as e:
        raise IOError(f"Failed to write nexus file '{filepath}': {str(e)}")


##########################################
#### Newick Standard Conversion Utilities
##########################################

# Regex pattern for a number in newick (integer, float, or scientific notation)
_NUM_RE = r'\d*\.?\d+(?:[eE][+-]?\d+)?'

# Valid target standards for convert_newick
_VALID_STANDARDS = {"PhyNetPy", "Phylonet", "Beast"}


def detect_newick_standard(newick_str: str) -> str:
    """
    Auto-detect which newick convention a string uses based on its 
    formatting.

    The detection heuristic is:
        1. If the string contains ``#Name:len::gamma`` double-colon 
           notation on a reticulation node → **Phylonet**
        2. If the string starts with ``[&R]`` or ``[&U]`` → **Beast**
        3. If the string contains ``[&...gamma=...]`` → **PhyNetPy**
        4. Otherwise (plain newick) → **PhyNetPy** (default)

    Args:
        newick_str (str): A newick or extended-newick string.

    Returns:
        str: One of ``"Phylonet"``, ``"Beast"``, or ``"PhyNetPy"``.
    """
    s = newick_str.strip()

    # PhyloNet: double-colon gamma on reticulation nodes
    if re.search(r'#[A-Za-z_]\w*(?::' + _NUM_RE + r')?::' + _NUM_RE, s):
        return "Phylonet"

    # BEAST: [&R] or [&U] rooted/unrooted prefix
    if re.match(r'^\[&[RU]\]', s):
        return "Beast"

    # PhyNetPy: [&gamma=X] comment annotation
    if re.search(r'\[&[^\]]*gamma=', s):
        return "PhyNetPy"

    return "PhyNetPy"


def convert_newick(newick_str: str, standard: str = "PhyNetPy") -> str:
    """
    Convert a newick/extended-newick string between different software 
    conventions.

    The three supported standards differ primarily in how they encode 
    inheritance probabilities (gamma) on reticulation edges:

    **PhyNetPy** uses BioPython-style bracket comments::

        ((C:.1,(B:.05)#H0[&gamma=.7]:.05)I1:.1,(A:.1,#H0:.05)I2:.1)I3;

    **Phylonet** uses Rich Newick double-colon notation::

        ((C:.1,(B:.05)#H0:.05::.7)I1:.1,(A:.1,#H0:.05)I2:.1)I3;

    **Beast** uses the same annotation as PhyNetPy but prefixes the 
    string with ``[&R]`` for rooted trees (or ``[&U]`` for unrooted)::

        [&R] ((C:.1,(B:.05)#H0[&gamma=.7]:.05)I1:.1,(A:.1,#H0:.05)I2:.1)I3;

    The function auto-detects the input convention and converts to the 
    target. Non-gamma metadata (e.g. ``[&posterior=0.95]``) on 
    non-reticulation nodes is preserved in all conversions.

    Args:
        newick_str (str): A newick or extended-newick string in any of 
                          the three conventions.
        standard (str, optional): Target convention. One of 
                                   ``"PhyNetPy"`` (default), 
                                   ``"Phylonet"``, or ``"Beast"``.

    Raises:
        ValueError: If ``standard`` is not one of the three valid 
                    options.
        IOError: If the input string is empty.

    Returns:
        str: The newick string reformatted for the target software.
    """
    if standard not in _VALID_STANDARDS:
        raise ValueError(
            f"Invalid standard '{standard}'. Must be one of: "
            f"{', '.join(sorted(_VALID_STANDARDS))}"
        )

    s = newick_str.strip()
    if not s:
        raise IOError("Cannot convert an empty newick string.")

    # ---- Step 1: Strip BEAST [&R]/[&U] prefix if present ----
    s = re.sub(r'^\[&[RU]\]\s*', '', s)

    # ---- Step 2: Normalize PhyloNet :: notation to [&gamma=X] ----
    # Match: #ReticName followed by optional :branch_length, then ::gamma
    # PhyloNet format:  #H0:0.05::0.7
    # PhyNetPy format:  #H0[&gamma=0.7]:0.05
    phylonet_pattern = (
        r'(#[A-Za-z_]\w*)'                   # Group 1: retic name
        r'(:' + _NUM_RE + r')?'               # Group 2: optional :branch
        r'::(' + _NUM_RE + r')'               # Group 3: gamma value
    )

    def _phylonet_to_intermediate(m: re.Match) -> str:
        name = m.group(1)
        branch = m.group(2) or ''
        gamma = m.group(3)
        # PhyNetPy order: name [&gamma=X] :branch
        return name + '[&gamma=' + gamma + ']' + branch

    s = re.sub(phylonet_pattern, _phylonet_to_intermediate, s)

    # Now the string is fully in PhyNetPy [&gamma=X] form.

    # ---- Step 3: Convert to target standard ----
    if standard == "PhyNetPy":
        return s

    elif standard == "Phylonet":
        # Convert [&...gamma=X...] on reticulation nodes to ::X
        # Also strips gamma from compound metadata blocks, leaving
        # other metadata intact.
        retic_metadata_pattern = (
            r'(#[A-Za-z_]\w*)'                # Group 1: retic name
            r'\[&([^\]]+)\]'                   # Group 2: metadata body
            r'(:' + _NUM_RE + r')?'            # Group 3: optional :branch
        )

        def _to_phylonet(m: re.Match) -> str:
            name = m.group(1)
            metadata = m.group(2)
            branch = m.group(3) or ''

            # Extract gamma value from metadata
            gamma_match = re.search(
                r'gamma=(' + _NUM_RE + r')', metadata
            )
            if not gamma_match:
                # No gamma found; leave this node untouched
                return m.group(0)

            gamma_val = gamma_match.group(1)

            # Remove gamma entry from metadata
            # Handle both "gamma=X,other" and "other,gamma=X" orderings
            remaining = re.sub(
                r',?\s*gamma=' + _NUM_RE, '', metadata
            )
            remaining = remaining.strip(',').strip()

            result = name
            if remaining:
                result += '[&' + remaining + ']'
            # PhyloNet order: name :branch ::gamma
            result += branch + '::' + gamma_val
            return result

        s = re.sub(retic_metadata_pattern, _to_phylonet, s)
        return s

    elif standard == "Beast":
        # Add [&R] prefix for rooted tree annotation
        return "[&R] " + s

    # Should never reach here due to validation above
    return s

