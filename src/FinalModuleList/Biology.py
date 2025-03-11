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
Module that contains classes and functions for dealing with biological concepts
that are relevant to networks and computational biology.

Release Version: 2.0.0

Author: Mark Kessler
"""

from __future__ import annotations

class IUPAC:
    """
    The following mappings are based on IUPAC codes from
    https://www.bioinformatics.org/sms/iupac.html
    """
    
    CODON_TO_PROTEIN = {
        'UUU': 'F', 'UUC': 'F',  # Phenylalanine
        'UUA': 'L', 'UUG': 'L',  # Leucine
        'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',  # Leucine
        'AUU': 'I', 'AUC': 'I', 'AUA': 'I',  # Isoleucine
        'AUG': 'M',  # Methionine (Start codon)
        'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',  # Valine
        'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',  # Serine
        'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',  # Proline
        'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',  # Threonine
        'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',  # Alanine
        'UAU': 'Y', 'UAC': 'Y',  # Tyrosine
        'UAA': '*', 'UAG': '*', 'UGA': '*',  # Stop codons
        'CAU': 'H', 'CAC': 'H',  # Histidine
        'CAA': 'Q', 'CAG': 'Q',  # Glutamine
        'AAU': 'N', 'AAC': 'N',  # Asparagine
        'AAA': 'K', 'AAG': 'K',  # Lysine
        'GAU': 'D', 'GAC': 'D',  # Aspartic acid
        'GAA': 'E', 'GAG': 'E',  # Glutamic acid
        'UGU': 'C', 'UGC': 'C',  # Cysteine
        'UGG': 'W',  # Tryptophan
        'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',  # Arginine
        'AGU': 'S', 'AGC': 'S',  # Serine
        'AGA': 'R', 'AGG': 'R',  # Arginine
        'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'   # Glycine
    }

    AMINO_TO_DNA = {
        'F': {'TTT', 'TTC'},        # Phenylalanine
        'L': {'TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'},  # Leucine
        'I': {'ATT', 'ATC', 'ATA'}, # Isoleucine
        'M': {'ATG'},               # Methionine (Start codon)
        'V': {'GTT', 'GTC', 'GTA', 'GTG'},  # Valine
        'S': {'TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'},  # Serine
        'P': {'CCT', 'CCC', 'CCA', 'CCG'},  # Proline
        'T': {'ACT', 'ACC', 'ACA', 'ACG'},  # Threonine
        'A': {'GCT', 'GCC', 'GCA', 'GCG'},  # Alanine
        'Y': {'TAT', 'TAC'},        # Tyrosine
        '*': {'TAA', 'TAG', 'TGA'}, # Stop codons
        'H': {'CAT', 'CAC'},        # Histidine
        'Q': {'CAA', 'CAG'},        # Glutamine
        'N': {'AAT', 'AAC'},        # Asparagine
        'K': {'AAA', 'AAG'},        # Lysine
        'D': {'GAT', 'GAC'},        # Aspartic acid
        'E': {'GAA', 'GAG'},        # Glutamic acid
        'C': {'TGT', 'TGC'},        # Cysteine
        'W': {'TGG'},               # Tryptophan
        'R': {'CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'},  # Arginine
        'G': {'GGT', 'GGC', 'GGA', 'GGG'}   # Glycine
    }

    AMINO_TO_RNA = {
        'F': {'UUU', 'UUC'},        # Phenylalanine
        'L': {'UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'},  # Leucine
        'I': {'AUU', 'AUC', 'AUA'}, # Isoleucine
        'M': {'AUG'},               # Methionine (Start codon)
        'V': {'GUU', 'GUC', 'GUA', 'GUG'},  # Valine
        'S': {'UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'},  # Serine
        'P': {'CCU', 'CCC', 'CCA', 'CCG'},  # Proline
        'T': {'ACU', 'ACC', 'ACA', 'ACG'},  # Threonine
        'A': {'GCU', 'GCC', 'GCA', 'GCG'},  # Alanine
        'Y': {'UAU', 'UAC'},        # Tyrosine
        '*': {'UAA', 'UAG', 'UGA'}, # Stop codons
        'H': {'CAU', 'CAC'},        # Histidine
        'Q': {'CAA', 'CAG'},        # Glutamine
        'N': {'AAU', 'AAC'},        # Asparagine
        'K': {'AAA', 'AAG'},        # Lysine
        'D': {'GAU', 'GAC'},        # Aspartic acid
        'E': {'GAA', 'GAG'},        # Glutamic acid
        'C': {'UGU', 'UGC'},        # Cysteine
        'W': {'UGG'},               # Tryptophan
        'R': {'CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'},  # Arginine
        'G': {'GGU', 'GGC', 'GGA', 'GGG'}   # Glycine
    }

    DNA_MAP : dict[str, int] = {"A" : 1, "C" : 2, "M" : 3, "G" : 4,
                                "R" : 5, "S" : 6, "V" : 7, "T" : 8, 
                                "W" : 9, "Y" : 10, "H" : 11, "K" : 12,
                                "D" : 13, "B" : 14, "N" : 15, "?" : 15,
                                "-" : 0, "X" : 15}

    RNA_MAP : dict[str, int] = {"A" : 1, "C" : 2, "M" : 3, "G" : 4, "R" : 5, 
                                "S" : 6, "V" : 7, "T" : 8, "U" : 8, "W" : 9, 
                                "Y" : 10, "H" : 11, "K" : 12, "D" : 13, 
                                "B" : 14, "N" : 15, "?" : 15, "-" : 0, "X" : 15}

    PROTEIN_MAP : dict[str, int] = {
        'F': 1,  # Phenylalanine
        'L': 2,  # Leucine
        'I': 3,  # Isoleucine
        'M': 4,  # Methionine (Start codon)
        'V': 5,  # Valine
        'S': 6,  # Serine
        'P': 7,  # Proline
        'T': 8,  # Threonine
        'A': 9,  # Alanine
        'Y': 10, # Tyrosine
        '*': 11, # Stop codons
        'H': 12, # Histidine
        'Q': 13, # Glutamine
        'N': 14, # Asparagine
        'K': 15, # Lysine
        'D': 16, # Aspartic acid
        'E': 17, # Glutamic acid
        'C': 18, # Cysteine
        'W': 19, # Tryptophan
        'R': 20, # Arginine
        'G': 21, # Glycine
        '.': 22, # Match Character
        '-': 23  # Gap Character
    }
    
    CODON_MAP : dict[str, int] = {"-" : 0, "A" : 1, "C" : 2, "M" : 3, "G" : 4,
                                  "R" : 5, "S" : 6, "V" : 7, "T" : 8, "U" : 8, 
                                  "W" : 9, "Y" : 9, "H" : 10, "K" : 11,
                                  "D" : 12, "B" : 13, "N" : 14, "X" : 14, 
                                  "." : 14}

    ALPHABETS : list[dict[str, int]] = [DNA_MAP,
                                        RNA_MAP,
                                        PROTEIN_MAP, 
                                        CODON_MAP]

    ALPHABET_NAMES : list[str] = ["DNA", "RNA", "PROTEIN", "CODON"]
    
class DNA(list):
    def __init__(self, *args):
        super().__init__()
        for item in args:
            self.append(item)

    def _validate_string(self, item):
        if not isinstance(item, str) or \
            item.capitalize() not in IUPAC.DNA_MAP.keys():
            raise TypeError(f"Only DNA string characters are allowed \
                              (lower or upper is fine).")

    def append(self, item : str):
        self._validate_string(item)
        super().append(item)

    def extend(self, iterable):
        for item in iterable:
            self._validate_string(item)
        super().extend(iterable)

    def insert(self, index, item):
        self._validate_string(item)
        super().insert(index, item)

    def __setitem__(self, index, item):
        self._validate_string(item)
        super().__setitem__(index, item)

    def __add__(self, other):
        if not all(isinstance(item, str) for item in other):
            raise TypeError("All elements in the added list must be strings.")
        return DNA(*super().__add__(other))

    def __iadd__(self, other):
        self.extend(other)
        return self
    
    def to_RNA(self) -> RNA:
        equivalent = []
        base : str
        for base in self:
            if base.capitalize() == "T":
                equivalent.append("U")
            else:
                equivalent.append(base)
    
        return RNA(equivalent)

class RNA(list):
    def __init__(self, *args):
        super().__init__()
        for item in args:
            self.append(item)

    def _validate_string(self, item):
        if not isinstance(item, str) or \
            item.capitalize() not in IUPAC.RNA_MAP.keys():
                raise TypeError(f"Only RNA string characters are allowed \
                                  (lower or upper is fine).")

    def append(self, item : str):
        self._validate_string(item)
        super().append(item)

    def extend(self, iterable):
        for item in iterable:
            self._validate_string(item)
        super().extend(iterable)

    def insert(self, index, item):
        self._validate_string(item)
        super().insert(index, item)

    def __setitem__(self, index, item):
        self._validate_string(item)
        super().__setitem__(index, item)

    def __add__(self, other):
        if not all(isinstance(item, str) for item in other):
            raise TypeError("All elements in the added list must be strings.")
        return DNA(*super().__add__(other))

    def __iadd__(self, other):
        self.extend(other)
        return self
    
    def to_DNA(self) -> DNA:
        equivalent = []
        base : str
        for base in self:
            if base.capitalize() == "U":
                equivalent.append("T")
            else:
                equivalent.append(base)
    
        return DNA(equivalent)

class PROTEIN(list):
    def __init__(self, *args):
        super().__init__()
        for item in args:
            self.append(item)

    def _validate_string(self, item):
        if not isinstance(item, str) or \
            item.capitalize() not in IUPAC.RNA_MAP.keys():
                raise TypeError(f"Only RNA string characters are allowed \
                                  (lower or upper is fine).")

    def append(self, item : str):
        self._validate_string(item)
        super().append(item)

    def extend(self, iterable):
        for item in iterable:
            self._validate_string(item)
        super().extend(iterable)

    def insert(self, index, item):
        self._validate_string(item)
        super().insert(index, item)

    def __setitem__(self, index, item):
        self._validate_string(item)
        super().__setitem__(index, item)

    def __add__(self, other):
        if not all(isinstance(item, str) for item in other):
            raise TypeError("All elements in the added list must be strings.")
        return DNA(*super().__add__(other))

    def __iadd__(self, other):
        self.extend(other)
        return self
    
    def to_RNA(self) -> RNA:
        equivalent = []
        base : str
        for codon in self:
            rna = IUPAC.AMINO_TO_RNA[codon]
            for base in rna:
                equivalent.append(base)
    
        return RNA(equivalent)
    
    def to_DNA(self) -> DNA:
        equivalent = []
        base : str
        for codon in self:
            dna = IUPAC.AMINO_TO_DNA[codon]
            for base in dna:
                equivalent.append(base)
    
        return DNA(equivalent)

