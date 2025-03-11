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
Module that contains classes and structures for working with biological data
or networks.

Release Version: 2.0.0

Author: Mark Kessler
"""

from __future__ import annotations
from collections import Counter
from itertools import combinations
from typing import Iterable, Union, Any
import warnings
from Biology import *
from Bio.Nexus.Nexus import NexusError
from Bio import AlignIO
from nexus import NexusReader
from Bio.AlignIO import MultipleSeqAlignment
from Bio.Align import SeqRecord
import numpy.typing as npt
import numpy as np
import math


###########################
#### EXCEPTION CLASSES ####
###########################

class AlphabetError(Exception):
    """
    Error class for all errors relating to alphabet mappings.
    """
    def __init__(self, message : str = "Error during Alphabet class mapping\
                                        operation"):
        self.message = message
        super().__init__(self.message)

class MatrixError(Exception):
    """
    This exception is raised when there is an error either in parsing data
    into the matrix object, or if there is an error during any sort of 
    operation
    """

    def __init__(self, message : str = "Matrix Error"):
        self.message = message
        super().__init__(self.message)

class MSAError(Exception):
    """
    This exception is raised when there is an error in initializing or working
    with an MSA object.
    """

    def __init__(self, message : str = "MSA Error"):
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

_stripJunk = str.maketrans("","","- ")

def _ratio(a, b):
    """
    stuff...

    Args:
        a (_type_): _description_
        b (_type_): _description_

    Returns:
        _type_: _description_
    """
    a = a.lower().translate(_stripJunk)
    b = b.lower().translate(_stripJunk)
    total  = len(a)+len(b)
    counts = (Counter(a) - Counter(b)) +(Counter(b) - Counter(a))
    return 100 - 100 * sum(counts.values()) / total

def _group_some_strings(data : list):
    threshold = 40
    minGroupSize = 1

    paired = {c:{c} for c in data}
    for a, b in combinations(data, 2):
        if _ratio(a, b) < threshold: continue
        paired[a].add(b)
        paired[b].add(a)

    groups = list()
    ungrouped = set(data)
    while ungrouped:
        bestGroup = {}
        for taxa in ungrouped:
            g = paired[taxa] & ungrouped
            for c in g.copy():
                g &= paired[c] 
            if len(g) > len(bestGroup):
                bestGroup = g
        
        # to terminate grouping early change minGroupSize to 3
        if len(bestGroup) < minGroupSize : break  
        ungrouped -= bestGroup # type: ignore
        groups.append(bestGroup)
    
    return groups

def _list_to_string(my_list : list[str]) -> str:
    """
    Turns a list of characters into a string
    

    Args:
        my_list (list[str]): A list of characters/strings

    Returns:
        str: one string that is the concatentation of each element of the list
    """
    sb = ""
    for char in my_list:
        sb += char
    return sb

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
        """
        
        self.alphabet = alphabet
            
    def map(self, char : str) -> int:
        """
        Return mapping for a character encountered in a nexus file

        Args:
            char (str): nexus file matrix data point

        Raises:
            AlphabetError: if the char encountered is undefined for the data 
                           mapping.

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

        Returns:
            str: the type of alphabet being used
        """
        if self.alphabet in IUPAC.ALPHABETS:
            return IUPAC.ALPHABET_NAMES[IUPAC.ALPHABETS.index(self.alphabet)]
        else:
            return "USER"

    def reverse_map(self, state : int) -> str:
        """
        Get the character that maps to "state" in the given alphabet

        Args:
            state (int): a value in the alphabet map

        Raises:
            AlphabetError: if the provided state is not a valid one in the 
                           alphabet

        Returns:
            str: the key that maps to "state"
        """
    
        for key in self.alphabet.keys():
            if self.alphabet[key] == state:
                return key
        raise AlphabetError("Given state does not exist in alphabet")

class DataSequence:
    """
    An individual sequence record. A sequence record is defined by 
    1) the data sequence 
    2) a name/string identifier
    3) potentially a group ID number [0, inf).
    """
    
    def __init__(self, 
                 sequence : list, 
                 name : str, 
                 gid : int = -1) -> None:
        """
        Initialize a Sequence Record

        Args:
            sequence (list): a sequence of data
            name (str): some name or label
            gid (int, optional): a group id number. Defaults to None.
        """
        #data sequence
        self.seq : list = sequence
        
        #sequence name
        self.name : str = name
        
        #group id 
        self.gid : int = gid
        
        #field to store the ploidyness of the data, if applicable.
        self.ploidyness : int = -1

    def get_name(self) -> str:
        """
        Get the name of the sequence.

        Returns:
            str: sequence label
        """
        return self.name

    def get_seq(self) -> list[object]:
        """
        This getter returns the sequence, as parsed from any sort of file.
        The likely type is a list[str] or a list of characters/strings.

        Returns:
            list[object]: A list of data (of some type, commonly a string)
        """
        return self.seq

    def get_numerical_seq(self) -> list[int]:
        """
        This getter returns the sequence, but in the event that the sequence is
        not already a list[int], translates each character into an integer.
        
        In the event that the sequence contains a character that is not mappable 
        in hexadecimal, then it will be skipped.

        Returns:
            list[int]: an integer data sequence, in hexadecimal.
        """
        num_seq : list[int] = [int(char, 16) for char in self.seq 
                               if char.isdigit() 
                               or char in set(["A", "B", "C", "D", "E", "F"])]
        
        if len(num_seq) != len(self):
            warnings.warn("Some characters were not able to be mapped to a \
                           hexadecimal number. Please double check your \
                           sequence to be sure all characters come from the set\
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E, F] \
                           -- case sensitive.")
        
        return num_seq

    def get_gid(self) -> int:
        """
        Get the group id for this sequence

        Returns:
            int: group id.
        """
        return self.gid
    
    def set_ploidy(self, ploidyness : int) -> None:
        """
        Set the ploidy of a data sequence. Only applicable for bimarker data, 
        but there is no consequence for setting this value for other data.

        Args:
            ploidyness (int): the level of ploidy for a data sequence
        """
        self.ploidyness = ploidyness
    
    def ploidy(self) -> int:
        """
        Get the ploidy for this sequence. 
        Only relevant for Bimarker data.

        Returns:
            int: ploidy value.
        """
        if self.ploidyness == -1:
            warnings.warn("Retrieving ploidyness of a SNP data sequence that \
                           has not had that attribute set. Returning -1.")
        return self.ploidyness

    def __len__(self) -> int:
        """
        Define the length of a DataSequence to be the length of the sequence

        Returns:
            int: _description_
        """
        return len(self.seq)

class MSA(Iterable[DataSequence]):
    
    """
    Class that provides all packaging and functionality services to do with 
    Multiple Sequence Alignments. This class stores all data and metadata 
    about a sequence alignment, and can handle file I/O from nexus files 
    that contain a matrix data block.
    
    If there is a grouping that applies to a set of sequences, 
    it can be defined here.
    """

    def __init__(self,
                 filename : str,
                 grouping : Union[dict[DataSequence, int], None] = None,
                 grouping_auto_detect : bool = False) -> None:
        """
        Initialize a Multiple Sequence Alignment (MSA).

        Args:
            filename (str): file name to a commonly accepted extension.
            grouping (dict, optional): Map from sequ. Defaults to None.
            grouping_auto_detect (bool, optional): _description_. Defaults to False.
        """
        self.filename : str = filename
        self.grouping : Union[dict[DataSequence, int], None ]= grouping
        
            
        self.hash : dict[int, list[DataSequence]] = {}  # map GIDs to a list of DataSequences
        self.name2gid : dict = {}
        
        # if grouping_auto_detect:
        #     self.grouping = self.group_auto_detect() TODO: FIX THIS

        self.records : list[DataSequence] = self.parse()
    
        # Either the number of records (1 taxa / group) or the number of groups
        if self.grouping is None:
            self.groups : int  = len(self.records)
        else:
            self.groups : int  = len(list(self.grouping.keys()))

    def get_records(self) -> list[DataSequence]:
        """
        Retrieve all sequences that are in this alignment.

        Returns:
            list[DataSequence]: list of all sequence records.
        """
        return self.records

    def parse(self) -> list[DataSequence]:
        """
        Take a filename and grab the sequences and put them into 
        DataSequence objects. If a grouping is defined (in the case of SNPs), 
        group IDs will be assigned to each DataSequence for ease of counting 
        red alleles.

        Returns: A list of DataSequence objs
        """

        recs : list[DataSequence] = []
        gid = 0
        
        #Setup
        if self.grouping is not None:
            for group_type in self.grouping.keys():
                self.name2gid[group_type] = gid
                
                # Map each Group ID to a list of DataSequences, empty at first
                self.hash[gid] = [] 
                gid += 1
        
        #Now parse
        try:
            # If the nexus file is in a Biopython supported data type
            msa : MultipleSeqAlignment = AlignIO.read(self.filename, "nexus")
            index = 0

            rec : SeqRecord
            for rec in list(msa):
                if self.grouping is None:  # Do nothing special, no GID applied
                    new_record = DataSequence(rec.seq, rec.name, gid = index)
                    recs.append(new_record)
                    self.hash[index] = [new_record]
                    index += 1
                else:
                    # assign new DataSequence its own correct GID based on grouping 
                    # specified
                    new_record = DataSequence(rec.seq, 
                                           rec.name, 
                                           gid = self.get_category(rec.name))
                    recs.append(new_record)
                    self.hash[new_record.get_gid()].append(new_record)
                    index += 1
        except NexusError:
            # do same as above, just using the NexusReader as a work-around.
            reader : NexusReader = NexusReader.from_file(self.filename)
        
            recs = []
            index = 0
            for taxa_data_pair in reader.data: # type: ignore

                #Where taxa is index 0, sequence is index 1
                if self.grouping is None:
                    new_record = DataSequence(taxa_data_pair[1],
                                           taxa_data_pair[0],
                                           gid = index)
                    recs.append(new_record)
                    self.hash[index] = [new_record]
                    index += 1
                else:
                    new_record = DataSequence(taxa_data_pair[1], 
                                           taxa_data_pair[0],
                                           gid = self.get_category(taxa_data_pair[0]))
                    recs.append(new_record)
                    self.hash[new_record.get_gid()].append(new_record)
                    index += 1         
        finally:  
            return recs

    def num_groups(self) -> int:
        """
        Returns: the number of groups in the MSA.
        """
        return self.groups

    def group_given_id(self, gid) -> list[DataSequence]:
        """
        Returns: the set (as a list) of DataSequences that have a given gid
        """
        return self.hash[gid]
    
    def get_category(self, name : str) -> int:
        """
        

        Args:
            name (str): _description_

        Raises:
            KeyError: _description_

        Returns:
            int: _description_
        """
        for groupname, members in self.grouping.items():
            if name in members:
                return self.name2gid[groupname]
        raise KeyError("name : " + name + " is not found in the grouping map \
                        provided, or there is not a grouping provided and an \
                        attempt was made to query one")
    
    def group_auto_detect(self) -> dict[int, str]:
        """
        If no grouping of sequences is provided, but a grouping is still 
        desired, group the sequences by name/label string "likeness".
        
        Note: not guaranteed to group things properly if the labels used for 
        sequences does not follow some sort of general pattern.
        
        IE:
        
        human1
        human2
        human3
        gorilla1
        gorilla2
        chimp1
        
        is group-able.
        
        xh1
        jp0
        an2
        am3
        
        is less group-able.

        Returns:
            dict[int, str]: a grouping map from gid's to sequence names
        """
        reader = NexusReader.from_file(self.filename)
        data = list()
        for taxa, chars in reader.data: # type: ignore
            data.append(taxa)
        
        groups = _group_some_strings(data)
        
        grouping : dict = {}
        groupno = 0
        for group in groups:
            grouping[groupno] = group
            groupno += 1
        
        return grouping
    
    def seq_by_name(self, name : str) -> DataSequence:
        """
        Retrieves the sequence that belongs to this MSA that has a given name

        Args:
            name (str): The taxa/label name of the sequence.
                        Must match exactly (same case, spacing, etc)

        Returns:
            DataSequence: the sequence with the label 'name'
        """
        for record in self.records:
            if record.label == name: # type: ignore
                return record
        raise MSAError()
    
    def total_samples(self) -> int:
        """
        For each record, accumulate the ploidyness to gather the total number 
        of samples of alleles
        
        Returns:
            int: the total number of samples 
        """
        return sum([rec.ploidy() for rec in self.get_records()])
    
    def samples_given_group(self, gid : int) -> int:
        """
        Return the number of samples within a given group.

        Args:
            gid (int): group id

        Returns:
            int: total samples within the group defined by 'gid'
        """
        return sum([rec.ploidy() for rec in self.group_given_id(gid)])
                
    def set_sequence_ploidy(self, 
                            seq_ploidy : Union[list[int], None] = None) -> None:
        """
        Sets the ploidy of each group of sequences in the MSA. 
        If sequence_ploidy is provided, it should be a list of numbers >= 1 
        where each index corresponds to the group ID.
        
        For example: [1,2,1] indicates that group 0 has ploidy 1, group 1 has 
        ploidy 2, and group 2 has ploidy 1.
        
        If sequence_ploidy is not given, then the ploidy will be set to the 
        maximum SNP data point found in the sequence. For a SNP sequence of 
        010120022202, the ploidy is 2. 
        
        NOTE: It is assumed that if sequence_ploidy is not given, that ploidy 
        values for each record within a group are identical!

        Args:
            sequence_ploidy (list[int], optional): implicit mapping from group 
                                                   ids (index) to the ploidy of 
                                                   that sequence, or set of 
                                                   sequences. Defaults to None.
        """
        
        if seq_ploidy is None:
            for record in self.records:
                record.set_ploidy(max(record.get_numerical_seq()))
        else:
            # Set each record in each group to be the ploidy 
            # at index "gid" in sequence_ploidy
            for gid in range(len(seq_ploidy)):
                for record in self.group_given_id(gid):
                    record.set_ploidy(seq_ploidy[gid])
                
    def dim(self) -> tuple[int, int]:
        """
        Return the dimensions of the MSA.
        
        The number of rows (first index) is equal to the number of DataSequence 
        objects, and the number of columns (second index), is equal to the 
        length of each DataSequence (they should all be the same).

        Returns:
            tuple[int]: row, col tuple that describes the dimensions of the MSA.
        """
        if len(self.records) > 0:
            return (len(self.records), len(self.records[0]))
        else:
            return (0,0)
    
    def distance(self, 
                 seq1 : DataSequence, 
                 seq2 : DataSequence, 
                 case_insensitive : bool = True) -> float:
        """
        Returns the hamming distance between two data sequences. For strings, 
        we are generally working with dna or rna characters, so case should 
        generally be ignored by default ("A" = "a").
        
        IE:
        ["A", "C", "C", "T"] and ["A", "G", "G, "T"] -> 2.
        ["A", "C", "C", "T"] and ["a", "c", "c", "t"] -> 0 (case insensitive!)
        ["A", "C", "C", "T"] and ["A", "G"] -> 3

        Args:
            seq1 (DataSequence): A DataSequence object
            seq2 (DataSequence): A DataSequence object
            case_insensitive (bool): By default will count strings as equal
                                     regardless of upper or lower case 
                                     differences.

        Returns:
            float: The number of unequal data points for two DataSequence 
                   objects.
        """
        results = []
        data1 = seq1.get_seq()
        data2 = seq2.get_seq()
        
        for index in range(max(len(seq1), len(seq2))):
            obj1 = None
            obj2 = None
            if index < len(seq1):
                obj1 = data1[index]
            if index < len(seq2):
                obj2 = data2[index]
            
            if obj1 is not None and obj2 is not None:
                if case_insensitive:   
                    if type(obj1) is str:
                        obj1 = obj1.upper().strip()
                    if type(obj2) is str:
                        obj2 = obj2.upper().strip()
                
                if obj1 == obj2:
                    results.append(0)
            else:
                results.append(1)
        
        return sum(results)
            
    def distance_matrix(self) -> dict[tuple[DataSequence], float]:
        """
        Using the distance helper, calculates pairwise distances for each pair
        of (different) DataSequences in this MSA.

        Returns:
            dict[tuple[DataSequence], float]: Map from DataSequence pairs to the
                                           distance between them.
        """
        D = dict()
        
        for seqr in self.records:
            for seqr2 in self.records:
                if seqr != seqr2:
                    D[(seqr, seqr2)] = self.distance(seqr, seqr2)
                    
        return D
    
class Matrix:
    """
    Class that stores and reduces MSA data to only the relevant/unique sites
    that exist. The only reduction mechanism so far is applicable only to DNA.
    All other data types will simply be stored in a 2d numpy matrix.
    
    Accepts any data that is defined by the Alphabet class in Alphabet.py.
    """
    
    def __init__(self, 
                 alignment : MSA, 
                 alphabet : Alphabet = Alphabet(IUPAC.DNA_MAP)) -> None:
        """
        Takes one single MSA object, along with an Alphabet object,
        represented as either DNA, RNA, PROTEIN, CODON, or USER. 
        The default is DNA.              
        """

        # ith element of the array = column i's distinct site pattern index in
        # the compressed matrix
        self.unique_sites : int = 0
        self.data : npt.NDArray[np.int_] = np.ndarray((0,), dtype = np.uint8)
        self.locations : list = list()

        # ith element of the array = count of the number of times
        # column i appears in the original uncompressed matrix
        self.count : list = list()
        self.alphabet : Alphabet = alphabet
        self.type : str = alphabet.get_type()
        self.taxa_to_rows : dict[str, int] = dict()
        self.rows_to_taxa : dict[int, str]= dict()

        # the next binary state to map a new character to
        self.next_state : int = 0

        ##Parse the input file into a list of sequence records
        self.seqs : list[DataSequence] = alignment.get_records()
        self.aln : MSA = alignment

        ##turn sequence record objects into the matrix data
        self.populate_data()
    
    def populate_data(self) -> None:
        """
        Stores and simplifies the MSA data.
        """
        
        # init the map from chars to binary
        # set the number of mappable states based on the alphabet type
        if self.type == "DNA" or self.type == "RNA":
            self.bits = math.pow(2, 8)  # 2^4?
            self.data = np.array([], dtype = np.int8)
        elif self.type == "SNP":
            self.bits = math.pow(2, 8)  # 2^2?
            self.data = np.array([], dtype = np.int8)
        elif self.type == "PROTEIN":
            # Prespecified substitution rates between aminos
            self.bits = math.pow(2, 32)
            self.data = np.array([], dtype = np.int32)
        else:
            self.bits = math.pow(2, 64)
            self.data = np.array([], dtype = np.int64)

        self.state_map : dict[str, int]= {}

        # translate the data into the matrix
        index = 0
        
        for r in self.seqs:

            self.taxa_to_rows[r.label] = index
            self.rows_to_taxa[index] = r.label
            # print("mapping " + str(r.label) + \
            #       " to row number " + str(index))
        
            for char in r.get_seq():
                # use the alphabet to map characters to their bit states 
                # and add to the data as a column
                char_as_array = np.array([self.alphabet.map(char)])
                self.data = np.append(self.data, char_as_array, axis = 0)
        
            index += 1

        # the dimensions of the uncompressed matrix
        # = num taxa if each group is only made of one taxa
        self.num_taxa = self.aln.num_groups()  
        
        self.seq_len = len(self.seqs[0].get_seq())

        # compress the matrix and fill out the locations and count fields
        # only compresses for DNA as of 4/3/24
        if self.type == "DNA":
            self.simplify()
        else:
            self.unique_sites = self.seq_len

    def simplify(self) -> None:
        """
        Reduces the matrix of data by removing non-unique site patterns, 
        and records the location and count of the unique site patterns.
        """

        new_data : np.ndarray = np.empty((self.num_taxa, 0), dtype = np.int8)

        column_data = dict()
        unique_sites = 0

        for i in range(self.seq_len):

            col = self.get_column(i, self.data, 0)
            col_str = _list_to_string(col)

            if col_str in column_data:
                self.locations.append(column_data[col_str])
            else:
                column_data[col_str] = i
                self.locations.append(i)
                unique_sites += 1
                new_data = np.append(new_data, 
                                     col.reshape((col.size, 1)),
                                     axis = 1)

        self.unique_sites = unique_sites
        self.populate_counts(new_data)
        self.data = new_data

    def get_ij(self, i : int, j : int) -> int:
        """
        Returns the data point at row i, and column j.

        Args:
            i (int): row index
            j (int): column index

        Returns:
            int: the data point.
        """
        return self.data[i][j]

    def get_ij_char(self, i : int, j : int) -> str:
        """
        get the character at row i, column j in the character matrix that is
        associated with the data.

        Args:
            i (int): row index
            j (int): column index

        Returns:
            str: the character at [i][j]
        """
        return self.char_matrix()[i][j]

    def row_given_name(self, label : str) -> int:
        """
        Retrieves the row index of the taxa that has name 'label'

        Args:
            label (str): name of a taxon.

        Returns:
            int: a row index
        """
        return self.taxa_to_rows[label]

    def get_seq(self, label : str) -> np.ndarray:
        """
        Gets the array of characters for a given taxon.

        Args:
            label (str): the name of a taxon.

        Returns:
            np.ndarray: an array of characters, with data type 'U1'.
        """
        return self.char_matrix()[self.row_given_name(label)]

    def get_number_seq(self, label : str) -> np.ndarray:
        """
        Gets the numerical data for a given taxon with the name 'label'.

        Args:
            label (str): name of a taxon

        Returns:
            np.ndarray: a 1 dimensional array of integers, of some specific type
        """
        return self.data[self.row_given_name(label)]

    def get_column(self, i : int, data : np.ndarray, sites : int) -> np.ndarray:
        """
        Returns ith column of a data matrix, with 'sites' elements

        Args:
            i (int): column index
            data (np.ndarray): a matrix 
            sites (int): dimension of the column

        Returns:
            np.ndarray: the data at column 'i' with length 'sites'
        """

        if sites == 0:
            data = data.reshape(self.num_taxa, self.seq_len)
        else:
            data = data.reshape(self.num_taxa, sites)

        return data[:, i]

    def get_column_at(self, i : int) -> np.ndarray:
        """
        Returns ith column of the data matrix

        Args:
            i (int): column index

        Returns:
            np.ndarray: the data at column i
        """
        return self.data[:, i]

    def site_count(self) -> int:
        """
        Returns the number of unique sites in the MSA/Data

        Returns:
            int: number of unique sites
        """
        return self.unique_sites

    def populate_counts(self, new_data : np.ndarray) -> None:
        """
        Generates a count list that maps the ith distinct column to the number
        of times it appears in the original alignment matrix.

        Args:
            new_data (np.ndarray): The simplified data matrix, that only has
                                   distinct column values.
        """
        
        for i in range(self.unique_sites):
            col = self.get_column(i, new_data, self.unique_sites)
            first = True
            for k in range(self.seq_len):
                col2 = self.get_column(k, self.data, 0)

                if list(col) == list(col2):
                    if first:
                        self.count.append(1)
                        first = False
                    else:
                        self.count[i] += 1

    def char_matrix(self) -> np.ndarray:
        """
        Get the character matrix from the matrix of alphabet states.

        Returns:
            np.ndarray: the character matrix, that will have equivalent 
                        dimensionality to the state matrix.
        """
        matrix = np.zeros(self.data.shape, dtype = 'U1')
        rows, cols = matrix.shape

        for i in range(rows):
            for j in range(cols):
                matrix[i][j] = self.alphabet.reverse_map(self.data[i][j])

        return matrix

    def get_num_taxa(self) -> int:
        """
        Get the number of taxa represented in this matrix.

        Returns:
            int: the number of taxa
        """
        return self.num_taxa

    def name_given_row(self, index : int) -> str:
        """
        Get the name of the taxa associated with the row of data at 'index'

        Args:
            index (int): a row index

        Returns:
            str: the taxon name
        """
        return self.rows_to_taxa[index]

    def get_type(self) -> str:
        """
        Get the type of data of this matrix

        Returns:
            str: the data type
        """
        return self.type
