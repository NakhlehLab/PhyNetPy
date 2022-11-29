from Bio.Nexus.Nexus import NexusError
from Bio import AlignIO
from nexus import NexusReader
from collections import Counter
from itertools import combinations

stripJunk = str.maketrans("","","- ")
def getRatio(a,b):
    a = a.lower().translate(stripJunk)
    b = b.lower().translate(stripJunk)
    total  = len(a)+len(b)
    counts = (Counter(a)-Counter(b))+(Counter(b)-Counter(a))
    return 100 - 100 * sum(counts.values()) / total



def group_some_strings(data:list):
    treshold = 40
    minGroupSize = 1


    paired = {c:{c} for c in data}
    for a,b in combinations(data,2):
        if getRatio(a,b) < treshold: continue
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
        if len(bestGroup) < minGroupSize : break  # to terminate grouping early change minGroupSize to 3
        ungrouped -= bestGroup
        groups.append(bestGroup)
    
    return groups
  

class SeqRecord:
    def __init__(self, sequence, name, gid=None):
        self.seq = sequence
        self.name = name
        self.gid = gid

    def get_name(self):
        return self.name

    def get_seq(self):
        return self.seq

    def get_numerical_seq(self):
        # TODO: TEMPORARY METHOD
        return [int(char) for char in self.seq]

    def get_gid(self):
        return self.gid


class MSA:
    """
    Wrapper class for a biopython MSA or a self-created one.
    Provides taxa name and sequence get services.
    """

    def __init__(self, file:str, grouping:dict=None, grouping_auto_detect : bool = False):
        self.filename : str = file
        self.grouping  = grouping
        self.hash : dict = {}  # map GIDs to a list of SeqRecords
        self.name2gid = {}
        
        if grouping_auto_detect:
            self.grouping = self.group_auto_detect()

        self.records : list = self.parse()

        if self.grouping is None:  # Either the number of records (1 taxa per group) or the number of groups
            self.groups : int  = len(self.records)
        else:
            self.groups : int  = len(list(self.grouping.keys()))

    def get_records(self) -> list:
        return self.records

    def parse(self) -> list:
        """
        Take a filename and grab the sequences and put them into SeqRecord objects.
        If a grouping is defined (in the case of SNPS), group IDs will be assigned to each SeqRecord
        for ease of counting red alleles.

        Returns: A list of SeqRecord objs
        """

        recs = []
        gid = 0

        if self.grouping is not None:
            for group_type in self.grouping.keys():
                self.name2gid[group_type] = gid
                self.hash[gid] = [] # Map each Group ID to a list of SeqRecords, empty at first
                gid += 1

        try:
            # If the nexus file is in a Biopython supported data type
            msa = AlignIO.read(self.filename, "nexus")
            index = 0

            for rec in list(msa):
                if self.grouping is None:  # Do nothing special, no GID applied
                    new_record = SeqRecord(rec.seq, rec.name, gid=index)
                    recs.append(new_record)
                    self.hash[index] = [new_record]
                    index += 1
                else:
                    # assign new SeqRecord its own correct GID based on grouping specified
                    new_record = SeqRecord(rec.seq, rec.name, gid=self.get_category(rec.name))
                    recs.append(new_record)
                    self.hash[new_record.get_gid()].append(new_record)
                    index += 1

        except NexusError:

            # do same as above, just using the NexusReader as a work-around.
            reader = NexusReader.from_file(self.filename)
            recs = []
            index = 0
            for taxa, chars in reader.data:
                if self.grouping is None:
                    new_record = SeqRecord(chars, taxa, gid=index)
                    recs.append(new_record)
                    self.hash[index] = [new_record]
                    index += 1
                else:
                    new_record = SeqRecord(chars, taxa, gid = self.get_category(taxa))
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

    def group_given_id(self, gid) -> list:
        """
        Returns: the set (as a list) of SeqRecords that have a given gid
        """
        return self.hash[gid]
    
    def get_category(self, name:str) -> int:
        for groupname, members in self.grouping.items():
            if name in members:
                return self.name2gid[groupname]
        raise KeyError("name : " + name + " is not found in the grouping map provided, or there is not a grouping provided and an attempt was made to querry one")
    
    def group_auto_detect(self) -> dict:
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
    
    def seq_by_name(self, name) -> list:
        for record in self.records:
            if record.get_name() == name:
                return [record]
                


