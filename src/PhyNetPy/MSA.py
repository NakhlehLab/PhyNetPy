""" 
Author : Mark Kessler
Last Stable Edit : 4/8/24
First Included in Version : 1.0.0
Approved for Release: No.
"""


import warnings
from Bio.Nexus.Nexus import NexusError
from Bio import AlignIO
from nexus import NexusReader
from collections import Counter
from itertools import combinations


##########################
#### HELPER FUNCTIONS ####
##########################

stripJunk = str.maketrans("","","- ")
def ratio(a, b):
    """
    stuff...

    Args:
        a (_type_): _description_
        b (_type_): _description_

    Returns:
        _type_: _description_
    """
    a = a.lower().translate(stripJunk)
    b = b.lower().translate(stripJunk)
    total  = len(a)+len(b)
    counts = (Counter(a)-Counter(b))+(Counter(b)-Counter(a))
    return 100 - 100 * sum(counts.values()) / total

def group_some_strings(data : list):
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


class SeqRecord:
    """
    An individual sequence record. A sequence record is defined by 
    1) the data sequence 
    2) a name/string identifier
    3) potentially a group ID number [0, inf).
    """
    
    def __init__(self, sequence : list[object], 
                 name : str, 
                 gid : int = None) -> None:
        """
        Initialize a Sequence Record

        Args:
            sequence (list[object]): a sequence of data
            name (str): some name or label
            gid (int, optional): a group id number. Defaults to None.
        """
        #data sequence
        self.seq : list[object] = sequence
        
        #sequence name
        self.name : str = name
        
        #group id 
        self.gid : int = gid
        
        #field to store the ploidyness of the data, if applicable.
        self.ploidyness : int = None

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
        
        if len(num_seq) != len(self.seq):
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
        return self.ploidyness

class MSA:
    
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
                 grouping : dict = None, 
                 grouping_auto_detect : bool = False) -> None:
        """
        Initialize a Multiple Sequence Alignment (MSA).

        Args:
            filename (str): file name to a commonly accepted extension.
            grouping (dict, optional): Map from sequ. Defaults to None.
            grouping_auto_detect (bool, optional): _description_. Defaults to False.
        """
        self.filename : str = filename
        self.grouping : dict = grouping
        self.hash : dict = {}  # map GIDs to a list of SeqRecords
        self.name2gid = {}
        
        if grouping_auto_detect:
            self.grouping = self.group_auto_detect()

        self.records : list[SeqRecord] = self.parse()
    
        # Either the number of records (1 taxa / group) or the number of groups
        if self.grouping is None:  
            self.groups : int  = len(self.records)
        else:
            self.groups : int  = len(list(self.grouping.keys()))

    def get_records(self) -> list[SeqRecord]:
        """
        Retrieve all sequences that are in this alignment.

        Returns:
            list[SeqRecord]: list of all sequence records.
        """
        return self.records

    def parse(self) -> list[SeqRecord]:
        """
        Take a filename and grab the sequences and put them into 
        SeqRecord objects. If a grouping is defined (in the case of SNPs), 
        group IDs will be assigned to each SeqRecord for ease of counting 
        red alleles.

        Returns: A list of SeqRecord objs
        """

        recs = []
        gid = 0

        if self.grouping is not None:
            for group_type in self.grouping.keys():
                self.name2gid[group_type] = gid
                
                # Map each Group ID to a list of SeqRecords, empty at first
                self.hash[gid] = [] 
                gid += 1
        try:
            # If the nexus file is in a Biopython supported data type
            msa = AlignIO.read(self.filename, "nexus")
            index = 0

            for rec in list(msa):
                if self.grouping is None:  # Do nothing special, no GID applied
                    new_record = SeqRecord(rec.seq, rec.name, gid = index)
                    recs.append(new_record)
                    self.hash[index] = [new_record]
                    index += 1
                else:
                    # assign new SeqRecord its own correct GID based on grouping 
                    # specified
                    new_record = SeqRecord(rec.seq, 
                                           rec.name, 
                                           gid = self.get_category(rec.name))
                    recs.append(new_record)
                    self.hash[new_record.get_gid()].append(new_record)
                    index += 1
        except NexusError:
            # do same as above, just using the NexusReader as a work-around.
            reader = NexusReader.from_file(self.filename)
        
            recs = []
            index = 0
            for taxa_data_pair in reader.data:

                #Where taxa is index 0, sequence is index 1
                if self.grouping is None:
                    new_record = SeqRecord(taxa_data_pair[1],
                                           taxa_data_pair[0],
                                           gid = index)
                    recs.append(new_record)
                    self.hash[index] = [new_record]
                    index += 1
                else:
                    new_record = SeqRecord(taxa_data_pair[1], 
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

    def group_given_id(self, gid) -> list[SeqRecord]:
        """
        Returns: the set (as a list) of SeqRecords that have a given gid
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
        for taxa, chars in reader.data:
            data.append(taxa)
        
        groups = group_some_strings(data)
        
        grouping : dict = {}
        groupno = 0
        for group in groups:
            grouping[groupno] = group
            groupno += 1
        
        return grouping
    
    def seq_by_name(self, name : str) -> SeqRecord:
        """
        Retrieves the sequence that belongs to this MSA that has a given name

        Args:
            name (str): The taxa/label name of the sequence.
                        Must match exactly (same case, spacing, etc)

        Returns:
            SeqRecord: the sequence with the label 'name'
        """
        for record in self.records:
            if record.get_name() == name:
                return record
    
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
                
                    
            
