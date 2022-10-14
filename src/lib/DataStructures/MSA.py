from Bio.Nexus.Nexus import NexusError
from Bio import AlignIO
from nexus import NexusReader


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
        #TODO: TEMPORARY METHOD
        return [int(char) for char in self.seq]

    def get_gid(self):
        return self.gid


class MSA:
    """
    Wrapper class for a biopython MSA or a self-created one.
    Provides taxa name and sequence get services.
    """

    def __init__(self, file, grouping=None):
        self.filename = file
        self.grouping = grouping
        self.hash = {}

        self.records = self.parse()

        if self.grouping is None:  # Either the number of records (1 taxa per group) or the number of groups
            self.groups = len(self.records)
        else:
            self.groups = len(self.grouping)

    def get_records(self):
        return self.records

    def parse(self):
        recs = []
        ids = []
        gid = 0
        if self.grouping is not None:
            for groupsize in self.grouping:
                for dummy in range(groupsize):
                    ids.append(gid)
                self.hash[gid] = []
                gid += 1

        try:
            # If the nexus file is in a Biopython supported data type
            msa = AlignIO.read(self.filename, "nexus")
            index = 0

            for rec in list(msa):
                if self.grouping is None:
                    recs.append(SeqRecord(rec.seq, rec.name))
                else:
                    new_record = SeqRecord(rec.seq, rec.name, gid=ids[index])
                    recs.append(new_record)
                    self.hash[ids[index]].append(new_record)
                    index += 1

        except NexusError:
            reader = NexusReader.from_file(self.filename)
            recs = []
            index = 0
            for taxa, chars in reader.data:
                if self.grouping is None:
                    recs.append(SeqRecord(chars, taxa))
                else:
                    new_record = SeqRecord(chars, taxa, gid=ids[index])
                    recs.append(new_record)
                    self.hash[ids[index]].append(new_record)
                    index += 1

        finally:
            return recs

    def num_groups(self):
        return self.groups

    def group_given_id(self, gid):
        return self.hash[gid]


