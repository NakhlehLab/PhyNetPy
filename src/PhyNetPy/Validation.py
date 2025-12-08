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
Last Stable Edit : 3/11/25
First Included in Version : 1.0.0
Approved for Release : Yes. Fully Documented and Tested.

Validation module for phylogenetic file formats. Provides comprehensive
validation and summary reporting for common phylogenetic data formats
including Newick, Nexus, FASTA, PHYLIP, Clustal, XML, and GenBank.
"""

from __future__ import annotations
import os
import re
import warnings
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from io import StringIO
from pathlib import Path

# BioPython imports
try:
    from Bio import Phylo, SeqIO, AlignIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.Align import MultipleSeqAlignment
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    warnings.warn("BioPython not available. Some validation features will be limited.")

# Nexus library import
try:
    from nexus import NexusReader
    HAS_NEXUS = True
except ImportError:
    HAS_NEXUS = False
    warnings.warn("python-nexus not available. Nexus validation will be limited.")

# XML parsing
try:
    import xml.etree.ElementTree as ET
    HAS_XML = True
except ImportError:
    HAS_XML = False
    warnings.warn("XML parsing not available.")


#####################
#### Error Classes ####
#####################

class ValidationError(Exception):
    """
    Base exception for validation errors.
    """
    def __init__(self, message: str = "Validation error occurred") -> None:
        self.message = message
        super().__init__(self.message)


class FileFormatError(ValidationError):
    """
    Exception raised when file format is invalid or corrupted.
    """
    pass


class DataIntegrityError(ValidationError):
    """
    Exception raised when data integrity checks fail.
    """
    pass


######################
#### Summary Classes ####
######################

class ValidationSummary:
    """
    Container for validation results and summary information.
    """
    
    def __init__(self, file_path: str, file_format: str):
        self.file_path = file_path
        self.file_format = file_format
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.summary_stats: Dict[str, Any] = {}
        
    def add_error(self, error: str) -> None:
        """Add an error message and mark validation as failed."""
        self.errors.append(error)
        self.is_valid = False
        
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
        
    def add_stat(self, key: str, value: Any) -> None:
        """Add a summary statistic."""
        self.summary_stats[key] = value
        
    def __str__(self) -> str:
        """Return formatted summary report."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"VALIDATION SUMMARY: {os.path.basename(self.file_path)}")
        lines.append("=" * 60)
        lines.append(f"Format: {self.file_format}")
        lines.append(f"Status: {'VALID' if self.is_valid else 'INVALID'}")
        lines.append("")
        
        if self.summary_stats:
            lines.append("SUMMARY STATISTICS:")
            lines.append("-" * 20)
            for key, value in self.summary_stats.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
            
        if self.warnings:
            lines.append("WARNINGS:")
            lines.append("-" * 10)
            for warning in self.warnings:
                lines.append(f"  • {warning}")
            lines.append("")
            
        if self.errors:
            lines.append("ERRORS:")
            lines.append("-" * 8)
            for error in self.errors:
                lines.append(f"  ✗ {error}")
            lines.append("")
            
        lines.append("=" * 60)
        return "\n".join(lines)


#########################
#### Base Validator ####
#########################

class BaseValidator(ABC):
    """
    Abstract base class for file format validators.
    """
    
    def __init__(self):
        self.supported_extensions: Set[str] = set()
        
    @abstractmethod
    def validate(self, file_path: str) -> ValidationSummary:
        """
        Validate a file and return summary.
        
        Args:
            file_path (str): Path to the file to validate
            
        Returns:
            ValidationSummary: Validation results and summary
        """
        pass
        
    def _check_file_exists(self, file_path: str) -> None:
        """Check if file exists and is readable."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not os.path.isfile(file_path):
            raise ValueError(f"Path is not a file: {file_path}")
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"File is not readable: {file_path}")
            
    def _get_file_stats(self, file_path: str) -> Dict[str, Any]:
        """Get basic file statistics."""
        stat = os.stat(file_path)
        return {
            "File Size (bytes)": stat.st_size,
            "File Size (KB)": round(stat.st_size / 1024, 2)
        }


##########################
#### Newick Validator ####
##########################

class NewickValidator(BaseValidator):
    """
    Validator for Newick format files (.nwk, .newick, .tre, .tree).
    """
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.nwk', '.newick', '.tre', '.tree'}
        
    def validate(self, file_path: str) -> ValidationSummary:
        """Validate Newick format file."""
        summary = ValidationSummary(file_path, "Newick")
        
        try:
            self._check_file_exists(file_path)
            summary.summary_stats.update(self._get_file_stats(file_path))
            
            if not HAS_BIOPYTHON:
                summary.add_error("BioPython required for Newick validation")
                return summary
                
            trees = self._parse_newick_trees(file_path, summary)
            if trees:
                self._analyze_trees(trees, summary)
                
        except Exception as e:
            summary.add_error(f"Validation failed: {str(e)}")
            
        return summary
        
    def _parse_newick_trees(self, file_path: str, summary: ValidationSummary) -> List[Any]:
        """Parse Newick trees from file."""
        trees = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()
                
            # Split on semicolons to handle multiple trees
            tree_strings = [t.strip() + ';' for t in content.split(';') if t.strip()]
            
            for i, tree_str in enumerate(tree_strings):
                try:
                    tree = Phylo.read(StringIO(tree_str), "newick")
                    trees.append(tree)
                except Exception as e:
                    summary.add_error(f"Failed to parse tree {i+1}: {str(e)}")
                    
        except Exception as e:
            summary.add_error(f"Failed to read file: {str(e)}")
            
        return trees
        
    def _analyze_trees(self, trees: List[Any], summary: ValidationSummary) -> None:
        """Analyze parsed trees and generate statistics."""
        summary.add_stat("Number of Trees", len(trees))
        
        if not trees:
            return
            
        # Analyze first tree in detail
        tree = trees[0]
        taxa = set()
        internal_nodes = 0
        total_branch_length = 0.0
        has_branch_lengths = True
        
        for clade in tree.find_clades():
            if clade.is_terminal():
                if clade.name:
                    taxa.add(clade.name)
            else:
                internal_nodes += 1
                
            if clade.branch_length is not None:
                total_branch_length += clade.branch_length
            else:
                has_branch_lengths = False
                
        summary.add_stat("Number of Taxa", len(taxa))
        summary.add_stat("Taxa Names", sorted(list(taxa)))
        summary.add_stat("Internal Nodes", internal_nodes)
        summary.add_stat("Has Branch Lengths", has_branch_lengths)
        
        if has_branch_lengths:
            summary.add_stat("Total Tree Length", round(total_branch_length, 6))
            
        # Check for common issues
        if len(taxa) < 3:
            summary.add_warning("Tree has fewer than 3 taxa")
            
        if not has_branch_lengths:
            summary.add_warning("Tree lacks branch lengths")
            
        # Check consistency across multiple trees
        if len(trees) > 1:
            self._check_tree_consistency(trees, summary)
            
    def _check_tree_consistency(self, trees: List[Any], summary: ValidationSummary) -> None:
        """Check consistency across multiple trees."""
        taxa_sets = []
        
        for tree in trees:
            taxa = {clade.name for clade in tree.find_clades() if clade.is_terminal() and clade.name}
            taxa_sets.append(taxa)
            
        # Check if all trees have same taxa (this is expected to vary due to gene loss)
        unique_taxa_sets = set(frozenset(taxa) for taxa in taxa_sets)
        if len(unique_taxa_sets) > 1:
            # This is normal - trees can have different taxa due to gene loss
            all_tree_taxa = set().union(*taxa_sets)
            summary.add_stat("Trees Have Variable Taxa", True)
            summary.add_stat("Total Unique Taxa Across Trees", len(all_tree_taxa))
            summary.add_stat("Taxa Set Variations", len(unique_taxa_sets))
        else:
            summary.add_stat("Consistent Taxa Across Trees", True)


#########################
#### Nexus Validator ####
#########################

class NexusValidator(BaseValidator):
    """
    Validator for Nexus format files (.nex, .nexus).
    """
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.nex', '.nexus'}
        
    def validate(self, file_path: str) -> ValidationSummary:
        """Validate Nexus format file."""
        summary = ValidationSummary(file_path, "Nexus")
        
        try:
            self._check_file_exists(file_path)
            summary.summary_stats.update(self._get_file_stats(file_path))
            
            if not HAS_NEXUS:
                summary.add_error("python-nexus required for Nexus validation")
                return summary
                
            self._parse_nexus_file(file_path, summary)
                
        except Exception as e:
            summary.add_error(f"Validation failed: {str(e)}")
            
        return summary
        
    def _parse_nexus_file(self, file_path: str, summary: ValidationSummary) -> None:
        """Parse and analyze Nexus file."""
        try:
            reader = NexusReader.from_file(file_path)
            
            # Check for different data blocks
            has_taxa = reader.taxa is not None
            has_trees = reader.trees is not None
            has_data = reader.data is not None
            
            summary.add_stat("Has Taxa Block", has_taxa)
            summary.add_stat("Has Trees Block", has_trees)
            summary.add_stat("Has Data Block", has_data)
            
            if has_taxa:
                self._analyze_taxa_block(reader, summary)
                
            if has_trees:
                self._analyze_trees_block(reader, summary)
                
            if has_data:
                self._analyze_data_block(reader, summary)
                
        except Exception as e:
            summary.add_error(f"Failed to parse Nexus file: {str(e)}")
            
    def _analyze_taxa_block(self, reader: Any, summary: ValidationSummary) -> None:
        """Analyze taxa block."""
        if reader.taxa:
            taxa_list = list(reader.taxa)
            summary.add_stat("Number of Taxa (from taxa block)", len(taxa_list))
            summary.add_stat("Taxa Names", sorted(taxa_list))
            
    def _analyze_trees_block(self, reader: Any, summary: ValidationSummary) -> None:
        """Analyze trees block."""
        if reader.trees:
            trees = list(reader.trees)
            summary.add_stat("Number of Trees/Networks", len(trees))
            
            # Analyze tree names and structures
            tree_names = []
            network_indicators = []
            
            for tree_def in trees:
                tree_str = str(tree_def)
                name = tree_str.split("=")[0].split()[-1] if "=" in tree_str else "unnamed"
                tree_names.append(name)
                
                # Check for network indicators (reticulation nodes)
                tree_content = "=".join(tree_str.split("=")[1:]) if "=" in tree_str else tree_str
                has_reticulation = "#" in tree_content
                network_indicators.append(has_reticulation)
                
            summary.add_stat("Tree/Network Names", tree_names)
            summary.add_stat("Networks Detected", sum(network_indicators))
            summary.add_stat("Pure Trees", len(network_indicators) - sum(network_indicators))
            
            # Try to extract taxa from trees and check against taxa block
            self._extract_and_validate_tree_taxa(reader, summary)
            
    def _analyze_data_block(self, reader: Any, summary: ValidationSummary) -> None:
        """Analyze data block (sequences)."""
        if reader.data:
            try:
                data_dict = reader.data
                summary.add_stat("Number of Sequences", len(data_dict))
                
                if data_dict:
                    # Get sequence lengths
                    seq_lengths = [len(seq) for seq in data_dict.values()]
                    summary.add_stat("Sequence Length", seq_lengths[0] if seq_lengths else 0)
                    
                    # Check if all sequences have same length
                    if len(set(seq_lengths)) > 1:
                        summary.add_warning("Sequences have different lengths")
                        summary.add_stat("Sequence Length Range", f"{min(seq_lengths)}-{max(seq_lengths)}")
                    
                    # Analyze character composition
                    all_chars = set()
                    for seq in data_dict.values():
                        all_chars.update(seq.upper())
                    
                    summary.add_stat("Character Set", sorted(list(all_chars)))
                    
                    # Determine likely data type
                    dna_chars = set('ATCG')
                    protein_chars = set('ACDEFGHIKLMNPQRSTVWY')
                    
                    if all_chars.issubset(dna_chars | {'N', '-', '?'}):
                        summary.add_stat("Likely Data Type", "DNA")
                    elif all_chars.issubset(protein_chars | {'-', '?', 'X'}):
                        summary.add_stat("Likely Data Type", "Protein")
                    else:
                        summary.add_stat("Likely Data Type", "Unknown/Mixed")
                        
            except Exception as e:
                summary.add_warning(f"Could not analyze data block: {str(e)}")
                
    def _extract_and_validate_tree_taxa(self, reader: Any, summary: ValidationSummary) -> None:
        """Extract taxa names from tree definitions and validate against taxa block."""
        if not HAS_BIOPYTHON:
            return
            
        try:
            all_tree_taxa = set()
            tree_taxa_by_tree = []
            
            for i, tree_def in enumerate(reader.trees):
                tree_str = str(tree_def)
                if "=" in tree_str:
                    newick_part = "=".join(tree_str.split("=")[1:])
                    tree_taxa = set()
                    
                    try:
                        tree = Phylo.read(StringIO(newick_part), "newick")
                        for clade in tree.find_clades():
                            if clade.is_terminal() and clade.name:
                                # Clean up reticulation node names
                                clean_name = clade.name.split('#')[0] if '#' in clade.name else clade.name
                                tree_taxa.add(clean_name)
                                all_tree_taxa.add(clean_name)
                    except:
                        # If BioPython fails, try simple regex extraction
                        taxa_matches = re.findall(r'([A-Za-z_][A-Za-z0-9_]*)', newick_part)
                        tree_taxa.update(taxa_matches)
                        all_tree_taxa.update(taxa_matches)
                    
                    tree_taxa_by_tree.append(tree_taxa)
                        
            if all_tree_taxa:
                summary.add_stat("Taxa from Trees", sorted(list(all_tree_taxa)))
                
                # Check against taxa block if it exists
                if reader.taxa:
                    defined_taxa = set(reader.taxa)
                    
                    # Check for undefined taxa in trees (ERROR condition)
                    undefined_taxa = all_tree_taxa - defined_taxa
                    if undefined_taxa:
                        summary.add_error(f"Trees contain taxa not defined in taxa block: {sorted(list(undefined_taxa))}")
                    
                    # Check for defined taxa missing from all trees (WARNING - could be gene loss)
                    missing_taxa = defined_taxa - all_tree_taxa
                    if missing_taxa:
                        summary.add_warning(f"Taxa defined but not present in any tree (possible gene loss): {sorted(list(missing_taxa))}")
                    
                    # Report coverage statistics
                    summary.add_stat("Taxa Coverage", f"{len(all_tree_taxa)}/{len(defined_taxa)} defined taxa present in trees")
                    
                    # Analyze per-tree taxa coverage
                    if len(tree_taxa_by_tree) > 1:
                        coverage_stats = []
                        for i, tree_taxa in enumerate(tree_taxa_by_tree):
                            coverage = len(tree_taxa & defined_taxa) / len(defined_taxa) * 100
                            coverage_stats.append(f"Tree {i+1}: {coverage:.1f}%")
                        summary.add_stat("Per-Tree Taxa Coverage", coverage_stats[:5])  # Show first 5
                        
                        if len(coverage_stats) > 5:
                            summary.add_stat("... and {} more trees".format(len(coverage_stats) - 5), "")
                
        except Exception as e:
            summary.add_warning(f"Could not extract taxa from trees: {str(e)}")


############################
#### Sequence Validators ####
############################

class FastaValidator(BaseValidator):
    """
    Validator for FASTA format files (.fasta, .fas, .fa).
    """
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.fasta', '.fas', '.fa', '.fna', '.ffn', '.faa'}
        
    def validate(self, file_path: str) -> ValidationSummary:
        """Validate FASTA format file."""
        summary = ValidationSummary(file_path, "FASTA")
        
        try:
            self._check_file_exists(file_path)
            summary.summary_stats.update(self._get_file_stats(file_path))
            
            if not HAS_BIOPYTHON:
                summary.add_error("BioPython required for FASTA validation")
                return summary
                
            self._parse_fasta_file(file_path, summary)
                
        except Exception as e:
            summary.add_error(f"Validation failed: {str(e)}")
            
        return summary
        
    def _parse_fasta_file(self, file_path: str, summary: ValidationSummary) -> None:
        """Parse and analyze FASTA file."""
        try:
            sequences = list(SeqIO.parse(file_path, "fasta"))
            
            if not sequences:
                summary.add_error("No valid sequences found in FASTA file")
                return
                
            summary.add_stat("Number of Sequences", len(sequences))
            
            # Analyze sequences
            seq_lengths = [len(seq.seq) for seq in sequences]
            seq_ids = [seq.id for seq in sequences]
            
            summary.add_stat("Sequence IDs", seq_ids[:10] if len(seq_ids) > 10 else seq_ids)
            if len(seq_ids) > 10:
                summary.add_stat("... and {} more".format(len(seq_ids) - 10), "")
                
            summary.add_stat("Sequence Length Range", f"{min(seq_lengths)}-{max(seq_lengths)}")
            summary.add_stat("Average Sequence Length", round(sum(seq_lengths) / len(seq_lengths), 2))
            
            # Check for alignment (equal length sequences)
            if len(set(seq_lengths)) == 1:
                summary.add_stat("Alignment Status", "Aligned (equal length sequences)")
            else:
                summary.add_stat("Alignment Status", "Unaligned (variable length sequences)")
                
            # Analyze character composition
            all_chars = set()
            for seq in sequences:
                all_chars.update(str(seq.seq).upper())
                
            summary.add_stat("Character Set", sorted(list(all_chars)))
            
            # Determine sequence type
            self._determine_sequence_type(all_chars, summary)
            
            # Check for duplicate IDs
            if len(set(seq_ids)) != len(seq_ids):
                summary.add_warning("Duplicate sequence IDs found")
                
        except Exception as e:
            summary.add_error(f"Failed to parse FASTA file: {str(e)}")
            
    def _determine_sequence_type(self, chars: Set[str], summary: ValidationSummary) -> None:
        """Determine the type of sequences based on character composition."""
        dna_chars = set('ATCG')
        rna_chars = set('AUCG')
        protein_chars = set('ACDEFGHIKLMNPQRSTVWY')
        
        if chars.issubset(dna_chars | {'N', '-', '?', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V'}):
            summary.add_stat("Sequence Type", "DNA")
        elif chars.issubset(rna_chars | {'N', '-', '?', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V'}):
            summary.add_stat("Sequence Type", "RNA")
        elif chars.issubset(protein_chars | {'-', '?', 'X', 'B', 'Z', 'J', 'U', 'O'}):
            summary.add_stat("Sequence Type", "Protein")
        else:
            summary.add_stat("Sequence Type", "Unknown/Mixed")


class PhylipValidator(BaseValidator):
    """
    Validator for PHYLIP format files (.phy, .phylip).
    """
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.phy', '.phylip'}
        
    def validate(self, file_path: str) -> ValidationSummary:
        """Validate PHYLIP format file."""
        summary = ValidationSummary(file_path, "PHYLIP")
        
        try:
            self._check_file_exists(file_path)
            summary.summary_stats.update(self._get_file_stats(file_path))
            
            if not HAS_BIOPYTHON:
                summary.add_error("BioPython required for PHYLIP validation")
                return summary
                
            self._parse_phylip_file(file_path, summary)
                
        except Exception as e:
            summary.add_error(f"Validation failed: {str(e)}")
            
        return summary
        
    def _parse_phylip_file(self, file_path: str, summary: ValidationSummary) -> None:
        """Parse and analyze PHYLIP file."""
        try:
            # Try both sequential and interleaved formats
            alignment = None
            format_type = None
            
            for fmt in ['phylip-sequential', 'phylip']:
                try:
                    alignment = AlignIO.read(file_path, fmt)
                    format_type = fmt
                    break
                except:
                    continue
                    
            if alignment is None:
                summary.add_error("Could not parse as PHYLIP format")
                return
                
            summary.add_stat("PHYLIP Format", format_type)
            summary.add_stat("Number of Sequences", len(alignment))
            summary.add_stat("Alignment Length", alignment.get_alignment_length())
            
            # Get sequence IDs
            seq_ids = [record.id for record in alignment]
            summary.add_stat("Sequence IDs", seq_ids)
            
            # Analyze character composition
            all_chars = set()
            for record in alignment:
                all_chars.update(str(record.seq).upper())
                
            summary.add_stat("Character Set", sorted(list(all_chars)))
            
            # Determine sequence type
            FastaValidator()._determine_sequence_type(all_chars, summary)
            
        except Exception as e:
            summary.add_error(f"Failed to parse PHYLIP file: {str(e)}")


class ClustalValidator(BaseValidator):
    """
    Validator for Clustal format files (.aln, .clustal).
    """
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.aln', '.clustal'}
        
    def validate(self, file_path: str) -> ValidationSummary:
        """Validate Clustal format file."""
        summary = ValidationSummary(file_path, "Clustal")
        
        try:
            self._check_file_exists(file_path)
            summary.summary_stats.update(self._get_file_stats(file_path))
            
            if not HAS_BIOPYTHON:
                summary.add_error("BioPython required for Clustal validation")
                return summary
                
            self._parse_clustal_file(file_path, summary)
                
        except Exception as e:
            summary.add_error(f"Validation failed: {str(e)}")
            
        return summary
        
    def _parse_clustal_file(self, file_path: str, summary: ValidationSummary) -> None:
        """Parse and analyze Clustal file."""
        try:
            alignment = AlignIO.read(file_path, "clustal")
            
            summary.add_stat("Number of Sequences", len(alignment))
            summary.add_stat("Alignment Length", alignment.get_alignment_length())
            
            # Get sequence IDs
            seq_ids = [record.id for record in alignment]
            summary.add_stat("Sequence IDs", seq_ids)
            
            # Analyze character composition
            all_chars = set()
            for record in alignment:
                all_chars.update(str(record.seq).upper())
                
            summary.add_stat("Character Set", sorted(list(all_chars)))
            
            # Determine sequence type
            FastaValidator()._determine_sequence_type(all_chars, summary)
            
        except Exception as e:
            summary.add_error(f"Failed to parse Clustal file: {str(e)}")


##########################
#### XML/GenBank Validators ####
##########################

class XMLValidator(BaseValidator):
    """
    Validator for XML format files (.xml).
    """
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.xml'}
        
    def validate(self, file_path: str) -> ValidationSummary:
        """Validate XML format file."""
        summary = ValidationSummary(file_path, "XML")
        
        try:
            self._check_file_exists(file_path)
            summary.summary_stats.update(self._get_file_stats(file_path))
            
            if not HAS_XML:
                summary.add_error("XML parsing not available")
                return summary
                
            self._parse_xml_file(file_path, summary)
                
        except Exception as e:
            summary.add_error(f"Validation failed: {str(e)}")
            
        return summary
        
    def _parse_xml_file(self, file_path: str, summary: ValidationSummary) -> None:
        """Parse and analyze XML file."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            summary.add_stat("Root Element", root.tag)
            summary.add_stat("XML Namespace", root.attrib.get('xmlns', 'None'))
            
            # Count different element types
            element_counts = Counter()
            for elem in root.iter():
                element_counts[elem.tag] += 1
                
            summary.add_stat("Element Types", dict(element_counts.most_common(10)))
            
            # Check for phylogenetic-specific elements
            phylo_elements = ['tree', 'node', 'edge', 'taxon', 'sequence', 'alignment']
            found_phylo = [elem for elem in phylo_elements if any(elem in tag.lower() for tag in element_counts)]
            
            if found_phylo:
                summary.add_stat("Phylogenetic Elements Found", found_phylo)
            else:
                summary.add_warning("No obvious phylogenetic elements detected")
                
        except ET.ParseError as e:
            summary.add_error(f"XML parsing error: {str(e)}")
        except Exception as e:
            summary.add_error(f"Failed to parse XML file: {str(e)}")


class GenBankValidator(BaseValidator):
    """
    Validator for GenBank format files (.gb, .gbk, .genbank).
    """
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = {'.gb', '.gbk', '.genbank'}
        
    def validate(self, file_path: str) -> ValidationSummary:
        """Validate GenBank format file."""
        summary = ValidationSummary(file_path, "GenBank")
        
        try:
            self._check_file_exists(file_path)
            summary.summary_stats.update(self._get_file_stats(file_path))
            
            if not HAS_BIOPYTHON:
                summary.add_error("BioPython required for GenBank validation")
                return summary
                
            self._parse_genbank_file(file_path, summary)
                
        except Exception as e:
            summary.add_error(f"Validation failed: {str(e)}")
            
        return summary
        
    def _parse_genbank_file(self, file_path: str, summary: ValidationSummary) -> None:
        """Parse and analyze GenBank file."""
        try:
            records = list(SeqIO.parse(file_path, "genbank"))
            
            if not records:
                summary.add_error("No valid GenBank records found")
                return
                
            summary.add_stat("Number of Records", len(records))
            
            # Analyze first record in detail
            record = records[0]
            summary.add_stat("Record ID", record.id)
            summary.add_stat("Record Description", record.description)
            summary.add_stat("Sequence Length", len(record.seq))
            summary.add_stat("Number of Features", len(record.features))
            
            # Analyze features
            feature_types = Counter(feat.type for feat in record.features)
            summary.add_stat("Feature Types", dict(feature_types.most_common()))
            
            # Check for annotations
            if record.annotations:
                summary.add_stat("Annotations", list(record.annotations.keys())[:10])
                
            # Analyze sequence composition
            if record.seq:
                chars = set(str(record.seq).upper())
                summary.add_stat("Character Set", sorted(list(chars)))
                FastaValidator()._determine_sequence_type(chars, summary)
                
        except Exception as e:
            summary.add_error(f"Failed to parse GenBank file: {str(e)}")


#############################
#### Main Validator Class ####
#############################

class PhylogeneticValidator:
    """
    Main validator class that handles multiple file formats.
    """
    
    def __init__(self):
        self.validators = {
            'newick': NewickValidator(),
            'nexus': NexusValidator(),
            'fasta': FastaValidator(),
            'phylip': PhylipValidator(),
            'clustal': ClustalValidator(),
            'xml': XMLValidator(),
            'genbank': GenBankValidator()
        }
        
        # Build extension to validator mapping
        self.extension_map = {}
        for name, validator in self.validators.items():
            for ext in validator.supported_extensions:
                self.extension_map[ext] = name
                
    def validate_file(self, file_path: str, format_hint: Optional[str] = None) -> ValidationSummary:
        """
        Validate a phylogenetic file.
        
        Args:
            file_path (str): Path to the file to validate
            format_hint (str, optional): Hint about the file format
            
        Returns:
            ValidationSummary: Validation results and summary
        """
        # Determine format
        if format_hint:
            validator_name = format_hint.lower()
        else:
            ext = Path(file_path).suffix.lower()
            validator_name = self.extension_map.get(ext)
            
        if not validator_name or validator_name not in self.validators:
            summary = ValidationSummary(file_path, "Unknown")
            summary.add_error(f"Unsupported file format. Extension: {Path(file_path).suffix}")
            return summary
            
        validator = self.validators[validator_name]
        return validator.validate(file_path)
        
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get dictionary of supported formats and their extensions."""
        return {
            name: sorted(list(validator.supported_extensions))
            for name, validator in self.validators.items()
        }
        
    def validate_directory(self, directory_path: str, recursive: bool = False) -> List[ValidationSummary]:
        """
        Validate all supported files in a directory.
        
        Args:
            directory_path (str): Path to directory
            recursive (bool): Whether to search recursively
            
        Returns:
            List[ValidationSummary]: List of validation results
        """
        results = []
        
        if recursive:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if Path(file_path).suffix.lower() in self.extension_map:
                        results.append(self.validate_file(file_path))
        else:
            for file in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file)
                if os.path.isfile(file_path) and Path(file_path).suffix.lower() in self.extension_map:
                    results.append(self.validate_file(file_path))
                    
        return results


#########################
#### Utility Functions ####
#########################

def validate_file(file_path: str, format_hint: Optional[str] = None, print_summary: bool = True) -> ValidationSummary:
    """
    Convenience function to validate a single file.
    
    Args:
        file_path (str): Path to the file to validate
        format_hint (str, optional): Hint about the file format
        print_summary (bool): Whether to print the summary
        
    Returns:
        ValidationSummary: Validation results
    """
    validator = PhylogeneticValidator()
    summary = validator.validate_file(file_path, format_hint)
    
    if print_summary:
        print(summary)
        
    return summary


def validate_directory(directory_path: str, recursive: bool = False, print_summaries: bool = True) -> List[ValidationSummary]:
    """
    Convenience function to validate all files in a directory.
    
    Args:
        directory_path (str): Path to directory
        recursive (bool): Whether to search recursively
        print_summaries (bool): Whether to print summaries
        
    Returns:
        List[ValidationSummary]: List of validation results
    """
    validator = PhylogeneticValidator()
    summaries = validator.validate_directory(directory_path, recursive)
    
    if print_summaries:
        for summary in summaries:
            print(summary)
            print()
            
    return summaries


def get_supported_formats() -> Dict[str, List[str]]:
    """
    Get dictionary of supported formats and their extensions.
    
    Returns:
        Dict[str, List[str]]: Format names mapped to extension lists
    """
    validator = PhylogeneticValidator()
    return validator.get_supported_formats()


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        validate_file(file_path)
    else:
        print("Usage: python Validation.py <file_path>")
        print("\nSupported formats:")
        for fmt, exts in get_supported_formats().items():
            print(f"  {fmt}: {', '.join(exts)}")
