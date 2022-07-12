"""
This Matrix data type is built from any of the usual file types (nexus, fasta, etc)
and packages it into an easy to use and space efficient data type.

The space the matrix takes up is determined by the size of its alphabet, and is 
abstracted away. This class can represent a 4bit, 8bit, 32bit, or 64bit matrix.

"""

from Bio import SeqIO, AlignIO
import sys
import numpy as np
import math
from Alphabet import Alphabet

def list2Str(myList):
        """
        Turns a list of characters into a string
        """
        sb = ""
        for char in myList:
                sb += str(char)
        return sb



class MatrixException2(Exception):
        """
        This exception is raised when the file contains too many different labels
        given the alphabet supplied
        """

        def __init__(self, message="File tried to list too many states!"):
                self.message = message
                super().__init__(self.message)


class MatrixCastError(Exception):
        """
        This exception is raised when the file contains too many different labels
        given the alphabet supplied
        """

        def __init__(self, type1, type2):
                self.message = "Disallowed Matrix cast from " + type1 + "to " + type2
                super().__init__(self.message)

class Matrix:

        def __init__(self, alignment, alphabet = Alphabet("DNA")):
                """
                Takes one single MultipleSequenceAlignment object, along with an Alphabet object,
                represented as either DNA, RNA, PROTEIN, CODON, SNP, or BINARY (for now). The default
                is DNA                
                """
        

                #ith element of the array = column i's distinct site pattern index in
                #the compressed matrix
                self.locations = []

                #ith element of the array = count of the number of times
                #column i appears in the original uncompressed matrix
                self.count = []
                self.alphabet = alphabet
                self.type = alphabet.getType()


                #the next binary state to map a new character to
                self.nextState = 0


                ##Parse the input file into a list of sequence records
                self.seqRecords = list(alignment)

                ##turn sequence record objects into the matrix data
                self.populateData()
                                

        def populateData(self):
                #init the map from chars to binary
                #set the number of mappable states based on the alphabet type
                if self.type == "DNA" or self.type == "RNA":
                        self.bits = math.pow(2,8) #2^4?
                        self.data = np.array([], dtype=np.int8)
                elif self.type == "SNP":
                        self.bits = math.pow(2,8) #2^2?
                        self.data = np.array([], dtype=np.int8)
                elif self.type == "PROTEIN":
                        #Prespecified substitution rates between aminos
                        #
                        self.bits = math.pow(2,32)
                        self.data = np.array([], dtype=np.int32)
                else:
                        self.bits = math.pow(2,64)
                        self.data = np.array([], dtype=np.int64)


                self.stateMap = {}


                #translate the data into the matrix
                index = 0
                for r in self.seqRecords:
                        lenCount = 0
                        for char in r.seq:
                                #use the alphabet to map characters to their bit states and add to 
                                #the data as a column
                                self.data = np.append(self.data, np.array([self.alphabet.map(char)]), axis=0)
                                lenCount+=1
                        
                        index += 1

                
                #the dimensions of the uncompressed matrix
                self.numTaxa = index
                self.seqLen = lenCount

                #compress the matrix and fill out the locations and count fields
                self.simplify()



        def map(self, state):
                """

                UNUSED FOR NOW
                Return f(state), where f: {alphabet} -> int.
                
                If state is not yet defined in the map, then select its binary 
                representation and then return it. Otherwise, simply return f(state).
                """

                

                if state not in self.stateMap:
                        if len(self.stateMap.keys()) > self.bits:
                                raise MatrixException2
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
        


        def asDNA(self):
                if(self.type == "RNA" or self.type == "Proteins"):
                        raise MatrixCastError(self.type, "DNA")
                elif self.type == "codon":
                        #switch from codon matrix to DNA matrix
                        self.type = "DNA"
                        self.populateData()
                return
        

        def asProtein(self):
                if(self.type == "RNA" or self.type == "Proteins"):
                        raise MatrixCastError(self.type, "DNA")
                elif self.type == "codon":
                        #switch from codon matrix to DNA matrix
                        self.type = "DNA"
                        self.populateData()
                return
        
        
        def charMatrix(self):
                matrix = np.zeros(self.data.shape, dtype='U1')
                rows, cols = matrix.shape
                
                for i in range(rows):
                        for j in range(cols):
                                matrix[i][j] = self.alphabet.reverseMap(self.data[i][j])
                                
                return matrix





##Simply use AlignIO.read
msa = AlignIO.read("src/io/testfile.nex", "nexus")
msa2 = AlignIO.read("src/io/testfile2.nex", "nexus")



aln = Matrix(msa) #default is to use the DNA alphabet
aln.verification()
print(aln.charMatrix())

print("========================================")

aln2 = Matrix(msa2)
aln2.verification()
print(aln2.charMatrix())


"""

OUTPUT:



[[1 8 2 4 4 8 0 8 4 1 1 4 4 8 4]
 [1 8 2 0 4 1 0 8 2 1 4 2 1 2 4]
 [1 8 2 4 4 1 0 8 8 1 4 4 4 8 4]
 [1 8 2 4 4 1 0 4 4 8 1 4 4 8 1]
 [1 8 2 4 4 1 0 4 4 4 1 4 4 8 2]
 [1 8 2 4 4 1 0 4 4 2 1 4 4 8 8]
 [1 8 2 4 4 1 0 4 4 1 1 4 4 8 1]]
[0, 1, 0, 1, 0, 5, 6, 6, 8, 8, 10, 1, 0, 13, 13, 13, 16, 1, 0, 19, 20, 13, 13, 13, 13, 0, 26, 0, 0, 1, 30, 1, 32, 1, 8, 1, 30, 1, 8, 39, 1, 1, 1, 39, 1, 1, 1, 1, 1, 5, 0, 1, 8, 1, 8, 55]
[9, 19, 2, 2, 6, 1, 7, 1, 1, 1, 1, 2, 1, 2, 1]
{}
[['A' 'T' 'C' 'G' 'G' 'T' '-' 'T' 'G' 'A' 'A' 'G' 'G' 'T' 'G']
 ['A' 'T' 'C' '-' 'G' 'A' '-' 'T' 'C' 'A' 'G' 'C' 'A' 'C' 'G']
 ['A' 'T' 'C' 'G' 'G' 'A' '-' 'T' 'T' 'A' 'G' 'G' 'G' 'T' 'G']
 ['A' 'T' 'C' 'G' 'G' 'A' '-' 'G' 'G' 'T' 'A' 'G' 'G' 'T' 'A']
 ['A' 'T' 'C' 'G' 'G' 'A' '-' 'G' 'G' 'G' 'A' 'G' 'G' 'T' 'C']
 ['A' 'T' 'C' 'G' 'G' 'A' '-' 'G' 'G' 'C' 'A' 'G' 'G' 'T' 'T']
 ['A' 'T' 'C' 'G' 'G' 'A' '-' 'G' 'G' 'A' 'A' 'G' 'G' 'T' 'A']]
========================================
[[1 2 1]
 [1 2 2]
 [1 2 2]
 [1 2 2]]
[0, 1, 2, 1]
[1, 2, 1]
{}
[['A' 'C' 'A']
 ['A' 'C' 'C']
 ['A' 'C' 'C']
 ['A' 'C' 'C']]




"""

