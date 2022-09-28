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
        self.records = self.parse()

        if self.grouping is None:  # Either the number of records (1 taxa per group) or the number of groups
            self.groups = len(self.records)
        else:
            self.groups = len(self.grouping)

    def get_records(self):
        return self.records

    def parse(self):
        recs = []

        try:
            # If the nexus file is in a Biopython supported data type
            msa = AlignIO.read(self.filename, "nexus")

            for rec in list(msa):
                recs.append(SeqRecord(rec.seq, rec.name))

        except NexusError:
            reader = NexusReader.from_file(self.filename)
            recs = []
            for taxa, chars in reader.data:
                recs.append(SeqRecord(chars, taxa))

        finally:
            return recs

    def num_groups(self):
        return self.groups
