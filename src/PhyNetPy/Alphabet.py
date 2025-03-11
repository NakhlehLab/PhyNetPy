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
Last Edit : 3/11/25
First Included in Version : 1.0.0
Docs   - [x]
Tests  - [x]
Design - [x]
"""

########################
### MODULE CONSTANTS ###
########################

DNA : dict[str, int] = {"A" : 1, "C" : 2, "M" : 3, "G" : 4, "R" : 5, "S" : 6, 
                        "V" : 7, "T" : 8, "W" : 9, "Y" : 10, "H" : 11, "K" : 12,
                        "D" : 13, "B" : 14, "N" : 15, "?" : 15, "-" : 0, 
                        "X" : 15}
    
RNA : dict[str, int] = {"A" : 1, "C" : 2, "M" : 3, "G" : 4, "R" : 5, "S" : 6, 
                        "V" : 7, "T" : 8, "U" : 8, "W" : 9, "Y" : 10, "H" : 11,
                        "K" : 12, "D" : 13, "B" : 14, "N" : 15, "?" : 15, 
                        "-" : 0, "X" : 15}

PROTEIN : dict[str, int] = {"-" : 0, "A" : 1, "B" : 2, "C" : 3, "D" : 4, 
                            "E" : 5, "F" : 6, "G" : 7, "H" : 8, "I" : 9, 
                            "J" : 10, "K" : 11, "L" : 12, "M" : 13, "N" : 14, 
                            "P" : 15, "Q" : 16, "R" : 17, "S" : 18, "T" : 19, 
                            "V" : 20, "W" : 21, "X" : 22, "Y" : 23, "Z" : 24, 
                            "." : 25}

CODON : dict[str, int] = {"-" : 0, "A" : 1, "C" : 2, "M" : 3, "G" : 4, "R" : 5, 
                          "S" : 6, "V" : 7, "T" : 8, "U" : 8, "W" : 9, "Y" : 9, 
                          "H" : 10, "K" : 11, "D" : 12, "B" : 13, "N" : 14,
                          "X" : 14, "." : 14}

ALPHABETS : list[dict[str, int]] = [DNA, RNA, PROTEIN, CODON]

ALPHABET_NAMES : list[str] = ["DNA", "RNA", "PROTEIN", "CODON"]

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

def snp_alphabet(ploidy : int) -> dict[str, int]:
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
    for num in range(ploidy+1):
        alphabet[str(num)] = num
    
    alphabet["?"] = ploidy + 1
    alphabet["N"] = ploidy + 1
    alphabet["-"] = ploidy + 1
    
    return alphabet

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
         N ? X	Any 	   A C G T ([1,1,1,1] -> 15)
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


    def __init__(self, alphabet : dict[str, int]) -> None:
        """
        Initialize this Alphabet object with a mapping of choice. May be from 
        any of the predefined mappings {DNA, RNA, PROTEIN, CODON}, or it 
        can be a special user defined alphabet. 
        
        For SNP alphabets, use the helper function 'snp_alphabet' with your 
        desired ploidy upperbound and generate a custom alphabet that way.
        
        
        Args:
            alphabet (dict[str, int]): Any of the constant type alphabets 
                                       (from the set {DNA, RNA, PROTEIN, 
                                       CODON}), or a user defined alphabet.
        Returns:
            N/A
        """
        
        self.alphabet = alphabet
            

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
            return self.alphabet[char]
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
        if self.alphabet in ALPHABETS:
            return ALPHABET_NAMES[ALPHABETS.index(self.alphabet)]
        else:
            return "USER"

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
    
        for key in self.alphabet.keys():
            if self.alphabet[key] == state:
                return key
        raise AlphabetError("Given state does not exist in alphabet")
    