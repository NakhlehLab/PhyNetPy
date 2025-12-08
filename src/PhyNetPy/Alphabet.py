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
Last Edit : 11/6/25
First Included in Version : 1.0.0
Docs   - [x]
Tests  - [x] Passed 5/5 tests with 100% coverage on 11/7/25
Design - [x]
"""

from dataclasses import dataclass


########################
### MODULE CONSTANTS ###
########################


@dataclass(frozen=True)
class AlphabetMapping:
    name : str
    mapping : dict[str, int]

DNA : AlphabetMapping = AlphabetMapping("DNA", 
                                        { "-" : 0, "A" : 1, "C" : 2, "M" : 3, "G" : 4, "R" : 5, "S" : 6, 
                                          "V" : 7, "T" : 8, "W" : 9, "Y" : 10, "H" : 11, "K" : 12,
                                          "D" : 13, "B" : 14, "X" : 15})
    
RNA : AlphabetMapping = AlphabetMapping("RNA", 
                                        {"-" : 0,"A" : 1, "C" : 2, "M" : 3, "G" : 4, "R" : 5, "S" : 6, 
                                         "V" : 7, "U" : 8, "W" : 9, "Y" : 10, "H" : 11,
                                         "K" : 12, "D" : 13, "B" : 14,  "X" : 15})

PROTEIN : AlphabetMapping = AlphabetMapping("PROTEIN", {"-" : 0, "A" : 1, "B" : 2, "C" : 3, "D" : 4, 
                            "E" : 5, "F" : 6, "G" : 7, "H" : 8, "I" : 9, 
                            "J" : 10, "K" : 11, "L" : 12, "M" : 13, "N" : 14, 
                            "P" : 15, "Q" : 16, "R" : 17, "S" : 18, "T" : 19, 
                            "V" : 20, "W" : 21, "X" : 22, "Y" : 23, "Z" : 24, 
                            "." : 25})

CODON : AlphabetMapping = AlphabetMapping("CODON", {"-" : 0, "A" : 1, "C" : 2, "M" : 3, "G" : 4, "R" : 5, 
                          "S" : 6, "V" : 7, "T" : 8, "W" : 9, "Y" : 10, 
                          "H" : 11, "K" : 12, "D" : 13, "B" : 14, "." : 15})

_ALPHABETS : list[AlphabetMapping] = [DNA, RNA, PROTEIN, CODON]

_ALPHABET_NAMES : list[str] = ["DNA", "RNA", "PROTEIN", "CODON"]

# Hard-coded reverse mappings for standard alphabets
_DNA_REVERSE : dict[int, str] = AlphabetMapping("DNA_REVERSE", 
                                               {0: "-", 1: "A", 2: "C", 3: "M", 4: "G", 5: "R", 6: "S", 
                                                7: "V", 8: "T", 9: "W", 10: "Y", 11: "H", 12: "K", 
                                                13: "D", 14: "B", 15: "X"})

_RNA_REVERSE: dict[int, str] = AlphabetMapping("RNA_REVERSE", 
                                              {0: "-", 1: "A", 2: "C", 3: "M", 4: "G", 5: "R", 6: "S", 
                                               7: "V", 8: "U", 9: "W", 10: "Y", 11: "H", 12: "K", 
                                              13: "D", 14: "B", 15: "X"})

_PROTEIN_REVERSE: dict[int, str] = AlphabetMapping("PROTEIN_REVERSE", 
                                                  {0: "-", 1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 
                                                   7: "G", 8: "H", 9: "I", 10: "J", 11: "K", 12: "L", 
                                                  13: "M", 14: "N", 15: "P", 16: "Q", 17: "R", 18: "S", 
                                                  19: "T", 20: "V", 21: "W", 22: "X", 23: "Y", 24: "Z", 
                                                  25: "."})

_CODON_REVERSE: dict[int, str] = AlphabetMapping("CODON_REVERSE", 
                                                {0: "-", 1: "A", 2: "C", 3: "M", 4: "G", 5: "R", 6: "S", 
                                                 7: "V", 8: "T", 9: "W", 10: "Y", 11: "H", 12: "K", 
                                                13: "D", 14: "B", 15: "."})

_REVERSE_MAPPINGS : [str, AlphabetMapping] = {DNA.name: _DNA_REVERSE, RNA.name: _RNA_REVERSE, PROTEIN.name: _PROTEIN_REVERSE, CODON.name: _CODON_REVERSE}


#########################
#### EXCEPTION CLASS ####
#########################

class AlphabetError(Exception):
    """
    Error class for all errors relating to alphabet mappings.
    """
    def __init__(self, message : str = "Error during Alphabet class mapping\
                                        operation") -> None:
        """
        Initialize an AlphabetError with a message.
        
        Args:
            message (str): error message
        Returns:
            N/A
        """
        self.message = message
        super().__init__(self.message)

##########################
#### HELPER FUNCTIONS ####
##########################

def _build_reverse_mapping(mapping: dict[str, int]) -> dict[int, str]:
    """
    Build a reverse mapping dictionary from state to character.
    If multiple characters map to the same state, keeps the first one encountered.
    
    Args:
        mapping (dict[str, int]): Dictionary mapping characters to states
    Returns:
        dict[int, str]: Dictionary mapping states to characters
    """
    reverse_mapping : dict[int, str] = {}
    for char, state in mapping.items():
        if state not in reverse_mapping:
            reverse_mapping[state] = char
    return reverse_mapping

def snp_alphabet(ploidy : int) -> AlphabetMapping:
    """
    For SNP alphabet initialization. For data sets in which the maximum ploidy 
    is Xn, use X as @ploidy.
    
    For phased SNP data, use 1. For unphased SNP data, use 2.

    Args:
        ploidy (int): The ploidyness value of a species 
                      (ie, humans = 2, some plants > 2, etc)

    Returns:
        dict[str, int]: Returns an SNP alphabet map that maps str(int)->int 
              for 0 <= int <= ploidy, plus the various extra character mappings.
              
    """
    alphabet : dict[str, int] = {}
    for num in range(ploidy + 1):
        alphabet[str(num)] = num
        
    alphabet["-"] = ploidy + 1
    
    return AlphabetMapping("SNP", alphabet)

########################
#### ALPHABET CLASS ####
########################

class Alphabet:
    """
    Class that deals with the mapping from characters to state values that 
    have partial likelihood values associated with them.
    This state mapping is primarily based on Base10 -> Binary 
    conversions such that the decimal numbers become a generalized
    version of the one-hot encoding scheme.
    
    DNA MAPPING INFORMATION
     Symbol(s)	Name	   Partial Likelihood
         A	  Adenine	   [1,0,0,0] -> 1
         C	  Cytosine	   [0,1,0,0] -> 2
         G	  Guanine	   [0,0,1,0] -> 4
         T U	  Thymine  [0,0,0,1] -> 8
     Symbol(s)	Name	   Partial Likelihood
         X	    Any 	   A C G T ([1,1,1,1] -> 15)
         V	    Not T	   A C G ([1,1,1,0] -> 7)
         H	    Not G	   A C T ([1,1,0,1] -> 11)
         D	    Not C	   A G T ([1,0,1,1] -> 13)
         B	    Not A	   C G T ([0,1,1,1] -> 14)
         M	    Amino	   A C ([1,1,0,0] -> 3)
         R	    Purine	   A G ([1,0,1,0] -> 5)
         W	    Weak	   A T ([1,0,0,1] -> 9)
         S	    Strong	   C G ([0,1,1,0] -> 6)
         Y	    Pyrimidine C T ([0,1,0,1] -> 10)
         K	    Keto	   G T ([0,0,1,1] -> 12)
    """


    def __init__(self, mapping : AlphabetMapping) -> None:
        """
        Initialize this Alphabet object with a mapping of choice. May be from 
        any of the predefined mappings {DNA, RNA, PROTEIN, CODON}, or it 
        can be a special user defined alphabet. 
        
        For SNP alphabets, use the helper function 'snp_alphabet' with your 
        desired ploidy upperbound and generate a custom alphabet that way.
        
        
        Args:
            mapping (AlphabetMapping): Any of the constant type alphabets 
                                       (from the set {DNA, RNA, PROTEIN, 
                                       CODON}), or a user defined alphabet.
        Returns:
            N/A
        """
        
        self.alphabet : AlphabetMapping = mapping
        
        # Use pre-computed reverse mapping for standard alphabets,
        # only compute for user-defined alphabets
        if self.alphabet.name in _REVERSE_MAPPINGS.keys():
            self._reverse_mapping : AlphabetMapping = _REVERSE_MAPPINGS[self.alphabet.name]
        else:
            # Build reverse mapping dictionary for user-defined alphabets
            # Note: If multiple characters map to the same state, we keep the first
            # character encountered (maintains original behavior)
            self._reverse_mapping : AlphabetMapping = AlphabetMapping("USER_REVERSE", _build_reverse_mapping(mapping.mapping))


    def map(self, char : str) -> int:
        """
        Return mapping for a character encountered in a nexus file

        Raises:
            AlphabetError: if the char encountered is undefined for the data 
                           mapping.
                           
        Args:
            char (str): nexus file matrix data point
        Returns:
            int: the integer corresponding to char in the alphabet mapping
        """
        try:
            return self.alphabet.mapping[char.upper()]
        except KeyError:
            raise AlphabetError("Attempted to map <" + char + ">. That \
                                 character is invalid for this alphabet")
        

    def get_type(self) -> str:
        """
        Returns a string that is equal to the alphabet constant name.
        
        ie. if one is using the DNA alphabet, 
        this function will return "DNA"

        Args:
            N/A
        Returns:
            str: the type of alphabet being used
        """
        return self.alphabet.name

    def reverse_map(self, state : int) -> str:
        """
        Get the character that maps to "state" in the given alphabet

        Raises:
            AlphabetError: if the provided state is not a valid one in the 
                           alphabet

        Args:
            state (int): a value in the alphabet map
        Returns:
            str: the key that maps to "state"
        """
        try:
            return self._reverse_mapping.mapping[state]
        except KeyError:
            raise AlphabetError("Given state does not exist in alphabet")
    