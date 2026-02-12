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
Last Stable Edit : 2/10/26
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
                # Skip special objects that have their own display
                if key in ("Gene Tree Aggregate", "Gene Tree Reports"):
                    continue
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        # Display per-tree gene tree reports if present
        gene_tree_reports = self.summary_stats.get("Gene Tree Reports")
        if gene_tree_reports:
            lines.append("PER-TREE GENE TREE DIAGNOSTICS:")
            lines.append("-" * 35)
            for report in gene_tree_reports:
                lines.append(str(report))
                lines.append("")
        
        # Display aggregate gene tree summary if present
        aggregate = self.summary_stats.get("Gene Tree Aggregate")
        if aggregate:
            lines.append(str(aggregate))
            lines.append("")
            
        if self.warnings:
            lines.append("WARNINGS:")
            lines.append("-" * 10)
            for warning in self.warnings:
                lines.append(f"  * {warning}")
            lines.append("")
            
        if self.errors:
            lines.append("ERRORS:")
            lines.append("-" * 8)
            for error in self.errors:
                lines.append(f"  [X] {error}")
            lines.append("")
            
        lines.append("=" * 60)
        return "\n".join(lines)


################################
#### Gene Tree Report Class ####
################################

class GeneTreeReport:
    """
    Container for per-gene-tree diagnostic results.
    
    Each gene tree parsed from a nexus file gets its own GeneTreeReport 
    that captures rooted/unrooted status, missing/duplicate taxa, whether 
    the tree is binary or multifurcating, branch length statistics, and 
    basic tree size metrics.
    
    These reports are embedded within a ValidationSummary so callers can 
    inspect them programmatically or print the human-readable summary.
    """
    
    def __init__(self, tree_index: int, tree_name: str) -> None:
        """
        Initialize a GeneTreeReport for a single gene tree.
        
        Args:
            tree_index (int): Zero-based index of this tree in the file.
            tree_name (str): The label/name of this tree from the nexus file.
        """
        self.tree_index: int = tree_index
        self.tree_name: str = tree_name
        
        # Topology flags
        self.is_rooted: Optional[bool] = None
        self.is_binary: Optional[bool] = None
        
        # Tree size
        self.num_leaves: int = 0
        self.num_internal_nodes: int = 0
        
        # Taxa tracking
        self.taxa: List[str] = []
        self.missing_taxa: List[str] = []
        self.duplicate_taxa: List[str] = []
        
        # Branch length analysis
        self.has_branch_lengths: bool = True
        self.branch_length_min: Optional[float] = None
        self.branch_length_max: Optional[float] = None
        self.branch_length_mean: Optional[float] = None
        self.negative_branch_lengths: int = 0
        self.zero_branch_lengths: int = 0
        
        # Polytomy tracking (internal nodes with >2 children)
        self.polytomy_nodes: int = 0
        
        # Network detection
        self.has_reticulation: bool = False
        
        # Issues
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def __str__(self) -> str:
        """Return a formatted single-tree report string."""
        lines = []
        lines.append(f"  Tree {self.tree_index + 1}: '{self.tree_name}'")
        lines.append(f"    Rooted: {'Yes' if self.is_rooted else 'No' if self.is_rooted is False else 'Unknown'}")
        lines.append(f"    Binary: {'Yes' if self.is_binary else 'No' if self.is_binary is False else 'Unknown'}")
        lines.append(f"    Leaves: {self.num_leaves}  |  Internal Nodes: {self.num_internal_nodes}")
        lines.append(f"    Taxa: {sorted(self.taxa)}")
        
        if self.has_reticulation:
            lines.append(f"    Network: Yes (contains reticulation nodes)")
        
        if self.duplicate_taxa:
            lines.append(f"    [!] DUPLICATE TAXA: {sorted(self.duplicate_taxa)}")
        
        if self.missing_taxa:
            lines.append(f"    Missing Taxa (vs. reference): {sorted(self.missing_taxa)}")
        
        if self.has_branch_lengths and self.branch_length_min is not None:
            lines.append(f"    Branch Lengths: min={self.branch_length_min:.6f}, "
                         f"max={self.branch_length_max:.6f}, "
                         f"mean={self.branch_length_mean:.6f}")
            if self.negative_branch_lengths > 0:
                lines.append(f"    [!] {self.negative_branch_lengths} NEGATIVE branch length(s)")
            if self.zero_branch_lengths > 0:
                lines.append(f"    [~] {self.zero_branch_lengths} zero-length branch(es)")
        elif not self.has_branch_lengths:
            lines.append(f"    Branch Lengths: Not present")
        
        if self.polytomy_nodes > 0:
            lines.append(f"    Polytomies: {self.polytomy_nodes} node(s) with >2 children")
        
        for err in self.errors:
            lines.append(f"    [X] ERROR: {err}")
        for warn_msg in self.warnings:
            lines.append(f"    [~] WARNING: {warn_msg}")
        
        return "\n".join(lines)


class GeneTreeAggregateSummary:
    """
    Aggregate summary across all gene trees in a nexus file.
    
    Provides high-level statistics about the entire collection of gene 
    trees so a biologist can quickly understand the overall quality and 
    characteristics of their gene tree dataset.
    """
    
    def __init__(self) -> None:
        """Initialize an empty aggregate summary."""
        self.total_trees: int = 0
        self.num_rooted: int = 0
        self.num_unrooted: int = 0
        self.num_binary: int = 0
        self.num_multifurcating: int = 0
        self.num_with_duplicates: int = 0
        self.num_networks: int = 0
        self.num_pure_trees: int = 0
        
        # Taxa coverage: taxon name -> number of trees it appears in
        self.taxa_frequency: Dict[str, int] = {}
        self.all_taxa: Set[str] = set()
        self.taxa_in_all_trees: Set[str] = set()
        self.taxa_missing_from_some: Dict[str, List[int]] = {}
        
        # Per-tree reports
        self.tree_reports: List[GeneTreeReport] = []
    
    def add_report(self, report: GeneTreeReport) -> None:
        """
        Incorporate a single GeneTreeReport into the aggregate.
        
        Args:
            report (GeneTreeReport): A per-tree diagnostic report.
        """
        self.tree_reports.append(report)
        self.total_trees += 1
        
        if report.is_rooted is True:
            self.num_rooted += 1
        elif report.is_rooted is False:
            self.num_unrooted += 1
        
        if report.is_binary is True:
            self.num_binary += 1
        elif report.is_binary is False:
            self.num_multifurcating += 1
        
        if report.duplicate_taxa:
            self.num_with_duplicates += 1
        
        if report.has_reticulation:
            self.num_networks += 1
        else:
            self.num_pure_trees += 1
        
        # Track taxa frequency
        for taxon in report.taxa:
            self.all_taxa.add(taxon)
            self.taxa_frequency[taxon] = self.taxa_frequency.get(taxon, 0) + 1
    
    def finalize(self) -> None:
        """
        Compute final aggregate statistics after all reports have been added.
        Call this after all tree reports have been incorporated.
        """
        if self.total_trees == 0:
            return
        
        # Determine which taxa appear in ALL trees vs only some
        self.taxa_in_all_trees = {
            taxon for taxon, count in self.taxa_frequency.items()
            if count == self.total_trees
        }
        
        # Track which trees each taxon is missing from
        for taxon in self.all_taxa:
            missing_from = []
            for report in self.tree_reports:
                if taxon not in report.taxa:
                    missing_from.append(report.tree_index + 1)
            if missing_from:
                self.taxa_missing_from_some[taxon] = missing_from
    
    def __str__(self) -> str:
        """Return a formatted aggregate summary string."""
        lines = []
        lines.append("-" * 55)
        lines.append("GENE TREE AGGREGATE SUMMARY")
        lines.append("-" * 55)
        lines.append(f"  Total Gene Trees: {self.total_trees}")
        lines.append(f"  Rooted: {self.num_rooted}  |  Unrooted: {self.num_unrooted}")
        lines.append(f"  Binary: {self.num_binary}  |  Multifurcating: {self.num_multifurcating}")
        lines.append(f"  Pure Trees: {self.num_pure_trees}  |  Networks: {self.num_networks}")
        lines.append(f"  Trees with Duplicate Taxa: {self.num_with_duplicates}")
        lines.append("")
        
        lines.append(f"  Total Unique Taxa: {len(self.all_taxa)}")
        lines.append(f"  Taxa in ALL Trees: {len(self.taxa_in_all_trees)}")
        if self.taxa_in_all_trees:
            lines.append(f"    {sorted(self.taxa_in_all_trees)}")
        
        taxa_missing = self.all_taxa - self.taxa_in_all_trees
        if taxa_missing:
            lines.append(f"  Taxa Missing from Some Trees: {len(taxa_missing)}")
            for taxon in sorted(taxa_missing):
                freq = self.taxa_frequency.get(taxon, 0)
                pct = (freq / self.total_trees) * 100
                missing_trees = self.taxa_missing_from_some.get(taxon, [])
                if len(missing_trees) <= 5:
                    lines.append(f"    '{taxon}': present in {freq}/{self.total_trees} "
                                 f"({pct:.1f}%), missing from tree(s) {missing_trees}")
                else:
                    lines.append(f"    '{taxon}': present in {freq}/{self.total_trees} "
                                 f"({pct:.1f}%), missing from {len(missing_trees)} trees")
        
        lines.append("-" * 55)
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
        """
        Analyze trees block with detailed per-gene-tree diagnostics.
        
        For each gene tree in the nexus file, produces a GeneTreeReport 
        capturing rooted/unrooted status, missing/duplicate taxa, binary 
        vs multifurcating topology, and branch length statistics. Results 
        are aggregated into a GeneTreeAggregateSummary.
        
        Args:
            reader (Any): NexusReader object with parsed trees.
            summary (ValidationSummary): The validation summary to populate.
        """
        if not reader.trees:
            return
        
        trees_list = list(reader.trees)
        summary.add_stat("Number of Trees/Networks", len(trees_list))
        
        # Determine the reference taxa set (from taxa block or union 
        # across all trees)
        defined_taxa: Optional[Set[str]] = None
        if reader.taxa:
            defined_taxa = set(reader.taxa)
        
        # Build per-tree reports
        aggregate = GeneTreeAggregateSummary()
        
        for idx, tree_def in enumerate(trees_list):
            tree_str = str(tree_def)
            
            # Extract tree name
            name = tree_str.split("=")[0].split()[-1] if "=" in tree_str else "unnamed"
            
            report = GeneTreeReport(idx, name)
            
            # Check for network indicators (reticulation nodes)
            newick_part = "=".join(tree_str.split("=")[1:]) if "=" in tree_str else tree_str
            report.has_reticulation = "#" in newick_part
            
            # Attempt to parse with BioPython for detailed analysis
            if HAS_BIOPYTHON:
                self._analyze_single_tree(newick_part, report)
            else:
                # Fallback: regex-based taxa extraction only
                taxa_matches = re.findall(
                    r'([A-Za-z_][A-Za-z0-9_]*)', newick_part
                )
                report.taxa = list(set(taxa_matches))
                report.num_leaves = len(report.taxa)
                report.warnings.append(
                    "BioPython not available; detailed analysis skipped"
                )
            
            # Check for missing taxa against the reference set
            if defined_taxa is not None:
                taxa_set = set(report.taxa)
                report.missing_taxa = sorted(
                    defined_taxa - taxa_set
                )
            
            aggregate.add_report(report)
        
        # If no explicit taxa block, compute reference from the union
        if defined_taxa is None and aggregate.all_taxa:
            for report in aggregate.tree_reports:
                report.missing_taxa = sorted(
                    aggregate.all_taxa - set(report.taxa)
                )
        
        # Finalize aggregate stats
        aggregate.finalize()
        
        # Store the aggregate and reports in the summary
        summary.add_stat("Gene Tree Aggregate", aggregate)
        
        # Also store the old-style top-level stats for backward compatibility
        tree_names = [r.tree_name for r in aggregate.tree_reports]
        summary.add_stat("Tree/Network Names", tree_names)
        summary.add_stat("Networks Detected", aggregate.num_networks)
        summary.add_stat("Pure Trees", aggregate.num_pure_trees)
        
        # Store per-tree reports list for programmatic access
        summary.add_stat("Gene Tree Reports", aggregate.tree_reports)
        
        # Validate taxa against the taxa block (if present)
        self._validate_tree_taxa_against_block(
            reader, aggregate, summary
        )
    
    def _analyze_single_tree(
        self, newick_str: str, report: GeneTreeReport
    ) -> None:
        """
        Analyze a single parsed tree and populate its GeneTreeReport.
        
        Inspects the tree for:
          - Rooted vs unrooted (root has 2 children = rooted, 3+ = unrooted)
          - Binary vs multifurcating (any internal node with >2 children)
          - Duplicate taxa (same leaf name appearing multiple times)
          - Branch length statistics (min, max, mean, negatives, zeros)
          - Tree size (leaf count, internal node count)
        
        Args:
            newick_str (str): The Newick string for this tree.
            report (GeneTreeReport): The report to populate.
        """
        try:
            tree = Phylo.read(StringIO(newick_str), "newick")
        except Exception as e:
            report.errors.append(f"Failed to parse: {str(e)}")
            return
        
        # Collect taxa names (handling reticulation node name cleanup)
        taxa_names: List[str] = []
        internal_count = 0
        branch_lengths: List[float] = []
        polytomy_count = 0
        
        for clade in tree.find_clades():
            if clade.is_terminal():
                if clade.name:
                    # Clean up reticulation node names
                    clean_name = (
                        clade.name.split('#')[0] 
                        if '#' in clade.name 
                        else clade.name
                    )
                    if clean_name:
                        taxa_names.append(clean_name)
            else:
                internal_count += 1
                # Check for polytomies (>2 children)
                num_children = len(clade.clades) if clade.clades else 0
                if num_children > 2:
                    polytomy_count += 1
            
            # Branch lengths
            if clade.branch_length is not None:
                branch_lengths.append(clade.branch_length)
            else:
                report.has_branch_lengths = False
        
        report.taxa = list(set(taxa_names))
        report.num_leaves = len(report.taxa)
        report.num_internal_nodes = internal_count
        report.polytomy_nodes = polytomy_count
        
        # Rooted detection: a rooted tree has exactly 2 children at the 
        # root; an unrooted tree typically has 3 (trifurcation at root)
        root = tree.root
        root_children = len(root.clades) if root.clades else 0
        if root_children == 2:
            report.is_rooted = True
        elif root_children >= 3:
            report.is_rooted = False
        elif root_children <= 1:
            # Degenerate case: single-child root or leaf-only tree
            report.is_rooted = True
            if root_children == 0:
                report.warnings.append("Tree has no children (single node)")
        
        # Binary detection: binary if all internal nodes have exactly 
        # 2 children
        report.is_binary = (polytomy_count == 0)
        
        # Duplicate taxa detection
        name_counts = Counter(taxa_names)
        duplicates = [name for name, count in name_counts.items() if count > 1]
        if duplicates:
            report.duplicate_taxa = sorted(duplicates)
            report.errors.append(
                f"Duplicate taxa found: {sorted(duplicates)}"
            )
        
        # Branch length statistics
        if branch_lengths and report.has_branch_lengths:
            report.branch_length_min = min(branch_lengths)
            report.branch_length_max = max(branch_lengths)
            report.branch_length_mean = (
                sum(branch_lengths) / len(branch_lengths)
            )
            report.negative_branch_lengths = sum(
                1 for bl in branch_lengths if bl < 0
            )
            report.zero_branch_lengths = sum(
                1 for bl in branch_lengths if bl == 0.0
            )
            
            if report.negative_branch_lengths > 0:
                report.errors.append(
                    f"{report.negative_branch_lengths} negative branch "
                    f"length(s) detected"
                )
            if report.zero_branch_lengths > 0:
                report.warnings.append(
                    f"{report.zero_branch_lengths} zero-length branch(es)"
                )
        elif not report.has_branch_lengths:
            report.warnings.append("Tree lacks branch lengths")
        
        # Size warnings
        if report.num_leaves < 3:
            report.warnings.append(
                f"Tree has only {report.num_leaves} taxa (fewer than 3)"
            )
    
    def _validate_tree_taxa_against_block(
        self,
        reader: Any,
        aggregate: GeneTreeAggregateSummary,
        summary: ValidationSummary
    ) -> None:
        """
        Compare taxa found across all trees against the taxa block 
        (if present) and report discrepancies.
        
        Args:
            reader (Any): NexusReader with the parsed nexus file.
            aggregate (GeneTreeAggregateSummary): The aggregate summary.
            summary (ValidationSummary): The validation summary.
        """
        all_tree_taxa = aggregate.all_taxa
        
        if all_tree_taxa:
            summary.add_stat("Taxa from Trees", sorted(list(all_tree_taxa)))
        
        if reader.taxa:
            defined_taxa = set(reader.taxa)
            
            # Check for taxa in trees that are NOT in the taxa block
            undefined_taxa = all_tree_taxa - defined_taxa
            if undefined_taxa:
                summary.add_error(
                    f"Trees contain taxa not defined in taxa block: "
                    f"{sorted(list(undefined_taxa))}"
                )
            
            # Check for taxa defined but missing from ALL trees
            missing_all = defined_taxa - all_tree_taxa
            if missing_all:
                summary.add_warning(
                    f"Taxa defined but not present in any tree "
                    f"(possible gene loss): {sorted(list(missing_all))}"
                )
            
            # Report coverage
            summary.add_stat(
                "Taxa Coverage",
                f"{len(all_tree_taxa & defined_taxa)}/"
                f"{len(defined_taxa)} defined taxa present in trees"
            )
            
            # Per-tree taxa coverage percentages
            if aggregate.total_trees > 1:
                coverage_stats = []
                for report in aggregate.tree_reports:
                    tree_taxa = set(report.taxa)
                    coverage = (
                        len(tree_taxa & defined_taxa) 
                        / len(defined_taxa) * 100
                    )
                    coverage_stats.append(
                        f"Tree {report.tree_index + 1} "
                        f"('{report.tree_name}'): {coverage:.1f}%"
                    )
                summary.add_stat(
                    "Per-Tree Taxa Coverage",
                    coverage_stats[:10]
                )
                if len(coverage_stats) > 10:
                    summary.add_stat(
                        "... and {} more trees".format(
                            len(coverage_stats) - 10
                        ), ""
                    )
            
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
