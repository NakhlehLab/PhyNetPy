from Bio import SeqIO

class seqFile:

        

        def __init__(self, file, ext):
                self.filename = file
                self.extension = ext
        
        def printNexus(self):
                if self.extension == ".nex" or self.extension == ".nxs":
                        print("PRINTING NEXUS FILE CONTENTS")
                        print("===========================")
                        
                        for seq_record in SeqIO.parse(self.filename, "nexus"):
                                print(seq_record.seq)

                        return
                return



f = seqFile("C:/Users/markk/OneDrive/Documents/PhyloPy/PhyloPy/src/io/testfile.nex", ".nex")
f.printNexus()
