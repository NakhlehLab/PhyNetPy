"""
This MSA data type is built from any of the usual file types (nexus, fasta, etc)
and packages it into an easy to use and space efficient data type

"""

from Bio import SeqIO
import sys

def list2Str(myList):
        """
        Turns a list of characters into a string
        """
        sb = ""
        for char in myList:
                sb += char
        return sb

class MSA:

        def __init__(self, filename, ext):
                self.data = []
                self.locations = []
                self.count = []
                
                
                seqRecords = []
                
                if ext == ".nex" or ext == ".nxs":
                        seqRecords = SeqIO.parse(filename, "nexus")
                elif ext == ".fasta":
                        seqRecords = SeqIO.parse(filename, "fasta")
                else:
                        print("ERROR: UNSUPPORTED FILE TYPE")
                        return
                
                index = 0
                entries = []
                for r in seqRecords:
                        lenCount = 0
                        for char in r.seq:
                                entries.append(char)
                                lenCount+=1
                        

                        self.data.append(entries)
                        index += 1
                        entries=[]

                self.numTaxa = index
                self.seqLen = lenCount
                self.simplify()
                                
                                        


        def simplify(self):
                """
                Reduces the matrix of taxa and removes non-unique site patterns, 
                and records the location and count of the unique site patterns
                """

                newData = []
                for dummy in range(self.numTaxa):
                        newData.append([])

                columnData = {}
                uniqueSites = 0

                for i in range(self.seqLen):
                        col = self.getColumn(i, self.data)
                        colStr = list2Str(col)
                        
                        if colStr in columnData:       
                                self.locations.append(columnData[colStr])
                        else:
                                columnData[colStr] = i
                                self.locations.append(i)
                                uniqueSites+=1
                                for j in range(self.numTaxa):
                                        newData[j].append(col[j])
                                
                
                self.uniqueSites = uniqueSites
                self.populateCounts(newData)
                self.data = newData
                

        
        def verification(self):
                print(self.data)
                print(self.locations)
                print(self.count)
                                       


        def getColumn(self, i, data):
                """
                Returns ith column of data matrix
                """
                col = []
                for j in range(self.numTaxa):
                        col.append(data[j][i])
                return col
        
        
        def populateCounts(self , newData):
                """
                Generates a count list that maps the ith distinct column to the number
                of times it appears in the original alignment matrix
                """
                for i in range(self.uniqueSites):
                        col = self.getColumn(i, newData)
                        first = True
                        for k in range(self.seqLen):
                                col2 = self.getColumn(k, self.data)
                                if col == col2:
                                        if first:
                                                self.count.append(1)
                                                first = False
                                        else:
                                                self.count[i]+=1
                                        









aln = MSA("src/io/testfile.nex", ".nex")
aln.verification()


        





        
                

        
        

