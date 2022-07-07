"""
This Matrix data type is built from any of the usual file types (nexus, fasta, etc)
and packages it into an easy to use and space efficient data type.

The space the matrix takes up is determined by the size of its alphabet, and is 
abstracted away. This class can represent a 4bit, 8bit, 32bit, or 64bit matrix.

"""

from Bio import SeqIO
import sys
import numpy as np

def list2Str(myList):
        """
        Turns a list of characters into a string
        """
        sb = ""
        for char in myList:
                sb += str(char)
        return sb


class MatrixException(Exception):
        """
        This exception is raised when the user tries to instantiate a Matrix
        instance with a file type that is not parsable into a bit matrix
        """

        def __init__(self, filename, ext, message="File extension not supported for this operation"):
                self.filename = filename
                self.ext = ext
                self.message = message
                super().__init__(self.message)

class Matrix:

        def __init__(self, filename, ext, alphabet):

                #actual matrix. max is 64 states, so 1 byte covers that
                self.data = np.array([], dtype=np.int8)

                #ith element of the array = column i's distinct site pattern index in
                #the compressed matrix
                self.locations = []

                #ith element of the array = count of the number of times
                #column i appears in the original uncompressed matrix
                self.count = []


                #the next binary state to map a new character to
                self.nextState = 0
                
                #set the number of mappable states based on the alphabet type
                if alphabet == "DNA" or alphabet == "RNA":
                        self.bits = 8
                elif alphabet == "SNP":
                        self.bits = 8
                elif alphabet == "Proteins":
                        self.bits = 32
                else:
                        self.bits = 64 #for codons
                

                ##Parse the input file into a list of sequence records
                seqRecords = []
                
                if ext == ".nex" or ext == ".nxs":
                        seqRecords = SeqIO.parse(filename, "nexus")
                elif ext == ".fasta":
                        seqRecords = SeqIO.parse(filename, "fasta")
                else:
                        raise MatrixException(self.filename, self.ext)
                
                
                #init the map from chars to binary
                self.stateMap = {}


                #translate the data into the matrix
                index = 0
                for r in seqRecords:
                        lenCount = 0
                        for char in r.seq:
                                self.data = np.append(self.data, np.array([self.map(char)]), axis=0)
                                lenCount+=1
                        
                        index += 1

                
                #the dimensions of the uncompressed matrix
                self.numTaxa = index
                self.seqLen = lenCount

                #compress the matrix and fill out the locations and count fields
                self.simplify()
                                

        def map(self, state):
                """
                Return f(state), where f: {alphabet} -> binary.
                
                If state is not yet defined in the map, then select its binary 
                representation and then return it. Otherwise, simply return f(state).
                """

                if state not in self.stateMap:
                        if len(self.stateMap.keys()) > self.bits:
                                print("ERROR: FILE CONTAINS MORE STATES THAN IS ALLOWED FOR THIS ALPHABET")
                                return None
                        else:
                                #map state to the next state 
                                self.stateMap[state] = self.nextState
                                self.nextState += 1
                                return self.stateMap[state]
                else:
                        return self.stateMap[state]



        def simplify(self):
                """
                Reduces the matrix of taxa and removes non-unique site patterns, 
                and records the location and count of the unique site patterns
                """

                newData = np.empty((self.numTaxa,0), dtype=np.int8)

                columnData = {}
                uniqueSites = 0

                for i in range(self.seqLen):
                        
                        col = self.getColumn(i, self.data, 0)
                        colStr = list2Str(col)
                        
                        if colStr in columnData:       
                                self.locations.append(columnData[colStr])
                        else:
                                columnData[colStr] = i
                                self.locations.append(i)
                                uniqueSites+=1
                                newData = np.append(newData, col.reshape((col.size, 1)), axis=1)
                                         
                
                self.uniqueSites = uniqueSites
                self.populateCounts(newData)
                self.data = newData
                

        
        def verification(self):
                print(self.data.reshape(self.numTaxa, self.uniqueSites))
                print(self.locations)
                print(self.count)
                print(self.stateMap)
                                       


        def getColumn(self, i, data, sites):
                """
                Returns ith column of data matrix
                """
        
                if sites == 0:
                        data = data.reshape(self.numTaxa, self.seqLen)
                else:
                        data = data.reshape(self.numTaxa, sites)
                
                return data[:, i]
        
        
        def populateCounts(self , newData):
                """
                Generates a count list that maps the ith distinct column to the number
                of times it appears in the original alignment matrix
                """
                for i in range(self.uniqueSites):
                        col = self.getColumn(i, newData, self.uniqueSites)
                        first = True
                        for k in range(self.seqLen):
                                col2 = self.getColumn(k, self.data, 0)

                                if np.array_equiv(col, col2):
                                        if first:
                                                self.count.append(1)
                                                first = False
                                        else:
                                                self.count[i]+=1
                                        









aln = Matrix("src/io/testfile.nex", ".nex", "DNA")
aln.verification()

print("========================================")

aln2 = Matrix("src/io/testfile2.nex", ".nex", "DNA")
aln2.verification()


