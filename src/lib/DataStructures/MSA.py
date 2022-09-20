from Bio.Nexus.Nexus import NexusError
from Bio import AlignIO
from nexus import NexusReader


class SeqRecord:
    def __init__(self, sequence, name):
        self.seq = sequence
        self.name = name

    def get_name(self):
        return self.name

    def get_seq(self):
        return self.seq


class MSA:
    """
    Wrapper class for a biopython MSA or a self-created one.
    Provides taxa name and sequence get services.
    """

    def __init__(self, file):
        self.filename = file
        self.records = self.parse()

    def get_records(self):
        return self.records

    def parse(self):
        recs = []
        try:
            # If the nexus file is in a Biopython supported data type (DNA)
            msa = AlignIO.read(self.filename, "nexus")
            for rec in list(msa):
                recs.append(SeqRecord(rec.seq, rec.name))

        except NexusError:
            # for SNPs
            reader = NexusReader.from_file(self.filename)
            recs = []
            for taxa, chars in reader.data:
                recs.append(SeqRecord(chars, taxa))

        finally:
            return recs
