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
Approved for Release: No.
"""


from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Union
import warnings
from collections import Counter
from itertools import combinations
from Bio.Nexus.Nexus import NexusError
from Bio import AlignIO
from nexus import NexusReader
from Bio.AlignIO import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord




##########################
#### HELPER FUNCTIONS ####
##########################

_stripJunk = str.maketrans("","","- ")
def ratio(a : str, b : str) -> float:
    """
    Compute the similarity ratio between two strings.

    Args:
        a (str): first string
        b (str): second string
    Returns:
        float: similarity ratio between a and b
    """
    a = a.lower().translate(_stripJunk)
    b = b.lower().translate(_stripJunk)
    total  = len(a) + len(b)
    counts = (Counter(a)-Counter(b))+(Counter(b)-Counter(a))
    return 100 - 100 * sum(counts.values()) / total

def group_some_strings(data : list[str]) -> list[list[str]]:
    """
    Group a list of strings by similarity.

    Args:
        data (list[str]): a list of strings

    Returns:
        list[list[str]]: a list of groups of strings
    """
    threshold = 40
    minGroupSize = 1

    paired = {c:{c} for c in data}
    for a, b in combinations(data, 2):
        if ratio(a, b) < threshold: continue
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
        ungrouped -= bestGroup
        groups.append(bestGroup)
    
    return groups
 
############################## 
#### SEQUENCE RECORD, MSA ####
##############################


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
            sequence (list): a sequence of some type of biological data
            name (str): some name or label
            gid (int, optional): a group id number. Defaults to -1. This signals
                                 that the sequence does not belong to any group.
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

        Args:
            N/A
        Returns:
            str: sequence label
        """
        return self.name

    def get_seq(self) -> list[object]:
        """
        This getter returns the sequence, as parsed from any sort of file.
        The likely type is a list[str] or a list of characters/strings.

        Args:
            N/A
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

        Args:
            N/A
        Returns:
            list[int]: an integer data sequence, in hexadecimal.
        """
        # Handle case where seq is already an integer (single value)
        if isinstance(self.seq, (int, float)):
            return [int(self.seq)]
        
        # Handle case where seq is already a list of integers
        if isinstance(self.seq, list) and all(isinstance(x, (int, float)) for x in self.seq):
            return [int(x) for x in self.seq]
        
        # Original behavior: convert string/character sequence to integers
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

        Args:
            N/A
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
        Returns:
            N/A
        """
        self.ploidyness = ploidyness
    
    def ploidy(self) -> int:
        """
        Get the ploidy for this sequence. 
        Only relevant for Bimarker data.

        Args:
            N/A
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

        Args:
            N/A
        Returns:
            int: _description_
        """
        return len(self.seq)
    
    def distance(self, seq2: DataSequence) -> float:
        """
        Calculate the distance between two DataSequence objects. 
        The distance is calculated by the number of differences between the
        two sequences. If the sequences are of different lengths, the distance
        is the difference in length plus the number of differences in the 
        shorter sequence compared to the longer sequence for the length of the
        shorter sequence.

        Args:
            seq2 (DataSequence): The second sequence to compare.
        Returns:
            float: The distance between the two sequences.
        """
        len1, len2 = len(self), len(seq2)
        min_len = min(len1, len2)
        distance = abs(len1 - len2)
        
        for i in range(min_len):
            if self.get_seq()[i] != seq2.get_seq()[i]:
                distance += 1
        
        return float(distance)

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
                 filename : Union[str, None] = None,
                 data : Union[list[DataSequence], None] = None,
                 grouping : Union[dict[DataSequence, int], None] = None,
                 grouping_auto_detect : bool = False
                 ) -> None:
        """
        Initialize an MSA object. Either a filename or a list of DataSequence
        objects can be provided. If a filename is provided, the MSA will be
        parsed from the file. If a list of DataSequence objects is provided,
        the MSA will be constructed from those objects. If grouping is provided,
        the sequences will be grouped accordingly.

        Args:
            filename (Union[str, None], optional): A filename to a nexus file
                                                   that contains a matrix data
                                                   block. Defaults to None.
            data (Union[list[DataSequence], None], optional): A list of
                                                              DataSequence
                                                              objects. Defaults
                                                              to None.
            grouping (Union[dict[DataSequence, int], None], optional): A
                                                                      grouping
                                                                      map from
                                                                      DataSequence
                                                                      objects to
                                                                      group IDs.
                                                                      Defaults to
                                                                      None.
            grouping_auto_detect (bool, optional): If set to True, the MSA will
                                                   attempt to group sequences
                                                   based on sequence name
                                                   similarity. Defaults
                                                   to False.
        Returns:
            N/A
        """
        if filename is not None:
            self.filename : str = filename
            self.grouping : Union[dict[DataSequence, int], None ]= grouping
            
                
            self.hash : dict[int, list[DataSequence]] = {}  # map GIDs to a list of DataSequences
            self.name2gid : dict = {}
            
            if grouping_auto_detect:
                self.grouping = self.group_auto_detect()

            self.records : list[DataSequence] = self.parse()
            
            # Auto-detect ploidy from max value in each sequence
            self.set_sequence_ploidy()
        
            # Either the number of records (1 taxa / group) or the number of groups
            if self.grouping is None:
                self.groups : int  = len(self.records)
            else:
                self.groups : int  = len(list(self.grouping.keys()))
        else:
            self.filename : str = "No Filename Given"
            self.grouping : Union[dict[DataSequence, int], None ] = {} 
            
            # map GIDs to a list of DataSequences
            self.hash : dict[int, list[DataSequence]] = {} 
            self.name2gid : dict = {}
            if data is not None:
                self.records : list[DataSequence] = data
               
                for data_seq in self.records: 
                    self.add_data(data_seq)
                    
                # Auto-detect ploidy from max value in each sequence
                self.set_sequence_ploidy()
            else:
                self.records : list[DataSequence] = []
        
            # Either the number of records (1 taxa / group) 
            # or the number of groups
            if self.grouping is None:
                self.groups : int  = len(self.records)
            else:
                self.groups : int  = len(list(self.grouping.keys()))
            
    
    def add_data(self, data_seq : DataSequence) -> None:
        """
        Add a DataSequence object to the MSA. If the DataSequence object has a
        group ID, it will be added to the appropriate group. If the DataSequence
        object does not have a group ID, it will be added to the MSA as a new
        group.

        Args:
            data_seq (DataSequence): A DataSequence object to add to the MSA
        """
        
        self.name2gid[data_seq.get_name()] = data_seq.get_gid()
        
        if data_seq.get_gid() in self.hash.keys():
            self.hash[data_seq.get_gid()].append(data_seq)
        else:
            self.hash[data_seq.get_gid()] = [data_seq]

    def retroactive_group(self) -> None:
        """
        Take all DataSequence objects placed inside the MSA and if the groupid 
        is -1 (indicating the DataSequence does not belong to any grouping), 
        attempt to use autogrouping to pair with other like data. After calling 
        this function, no DataSequence objects will have gid of -1.
        
        If autogrouping is not possible, the DataSequence will be placed in a
        group of its own.
        
        Args:
            N/A
        Returns:
            N/A
        """
        ungrouped = [seq for seq in self.records if seq.get_gid() == -1]
        if not ungrouped:
            return

        # Extract names of ungrouped sequences
        ungrouped_names = [seq.get_name() for seq in ungrouped]

        # Group ungrouped sequence names
        detected_groups = group_some_strings(ungrouped_names)

        # Assign new group IDs to ungrouped sequences
        new_gid = max(self.hash.keys(), default=-1) + 1
        for group in detected_groups:
            for seq_name in group:
                for seq in ungrouped:
                    if seq.get_name() == seq_name:
                        seq.gid = new_gid
                        self.add_data(seq)
            new_gid += 1

        # Handle any remaining ungrouped sequences
        for seq in ungrouped:
            if seq.get_gid() == -1:
                seq.gid = new_gid
                self.add_data(seq)
                new_gid += 1

    def get_records(self) -> list[DataSequence]:
        """
        Retrieve all sequences that are in this alignment.

        Args:
            N/A
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

        Args:
            N/A
        Returns: 
            list[DataSequence]: A list of DataSequence objs
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
          
        return recs

    def num_groups(self) -> int:
        """
        Get the number of groups in the MSA.
        
        Args:
            N/A
        Returns: 
            int: the number of groups in the MSA.
        """
        return self.groups

    def group_given_id(self, gid) -> list[DataSequence]:
        """
        Get the set of DataSequences that belong to a given group ID.
        
        Args:
            gid (int): group id
        Returns: 
            list[DataSequence] : the set (as a list) of DataSequences that 
                                 have a given gid
        """
        return self.hash[gid]
    
    def get_category(self, name : str) -> int:
        """
        Get the group ID for a given sequence name. If the sequence name is not
        found in the grouping map, raise a KeyError.
        
        Raises:
            KeyError: If the sequence name is not found in the grouping map.
        Args:
            name (str): The name of the sequence to get the group ID for.
        Returns:
            int: The group ID for the sequence.
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

        Args:
            N/A
        Returns:
            dict[int, str]: a grouping map from gid's to sequence names
        """
        reader = NexusReader.from_file(self.filename)
        data = list[Any]()
        for taxa, chars in reader.data:
            data.append(taxa)
        
        groups = group_some_strings(data)
        
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
            if record.name == name:
                return record
        raise ValueError("Sequence with name " + name + " not found in MSA")
    
    def total_samples(self) -> int:
        """
        For each record, accumulate the ploidyness to gather the total number 
        of samples of alleles. If ploidy is not set (-1), treats as 1 sample.
        
        Args:
            N/A
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
                
    def set_sequence_ploidy(self, sequence_ploidy : list[int] = None) -> None:
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
        Returns:    
            N/A
        """
        
        if sequence_ploidy is None:
            for record in self.records:
                record.set_ploidy(max(record.get_numerical_seq()))
        else:
            # Set each record in each group to be the ploidy 
            # at index "gid" in sequence_ploidy
            for gid in range(len(sequence_ploidy)):
                for record in self.group_given_id(gid):
                    record.set_ploidy(sequence_ploidy[gid])
                
    def dim(self) -> tuple[int]:
        """
        Return the dimensions of the MSA.
        
        The number of rows (first index) is equal to the number of DataSequence 
        objects, and the number of columns (second index), is equal to the 
        length of each DataSequence (they should all be the same).

        Args:
            N/A
        Returns:
            tuple[int]: row, col tuple that describes the dimensions of the MSA.
        """
        if len(self.records) > 0:
            return (len(self.records), len(self.records[0]))
        else:
            return (0,0)
    
    

    def distance_matrix(self) -> dict[tuple[DataSequence, DataSequence], float]:
        """
        Using the distance helper, calculates pairwise distances for each pair
        of (different) DataSequences in this MSA.

        Args:
            N/A
        Returns:
            dict[tuple[DataSequence, DataSequence], float]: Map from DataSequence pairs to the
                                           distance between them.
        """
        D = dict[tuple[DataSequence, DataSequence], float]()
        
        for i, seqr in enumerate(self.records):
            for j, seqr2 in enumerate(self.records):
                if i < j:  # Avoid duplicate calculations
                    D[(seqr, seqr2)] = seqr.distance(seqr2)
                    
        return D

    def __iter__(self) -> Iterable[DataSequence]:
        """
        Iterate over the DataSequence objects in this MSA.
        
        Args:
            N/A
        Returns:
            Iterable[DataSequence]: an iterable of DataSequence objects.
        """
        return iter(self.records)