""" 
Author : Mark Kessler
Last Stable Edit : 7/16/23
First Included in Version : 0.1.0

"""


import numpy as np
import math
from Alphabet import Alphabet
from MSA import MSA



def list2Str(my_list: list) -> str:
    """
    Turns a list of characters into a string
    """
    sb = ""
    for char in my_list:
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

    def __init__(self, alignment: MSA, alphabet=Alphabet("DNA")):
        """
        Takes one single MultipleSequenceAlignment object, along with an Alphabet object,
        represented as either DNA, RNA, PROTEIN, CODON, SNP, or BINARY (for now). The default
        is DNA                
        """

        # ith element of the array = column i's distinct site pattern index in
        # the compressed matrix
        self.uniqueSites : int = None
        self.data : np.ndarray = None
        self.locations : list = list()

        # ith element of the array = count of the number of times
        # column i appears in the original uncompressed matrix
        self.count : list = list()
        self.alphabet : Alphabet= alphabet
        self.type : str = alphabet.get_type()
        self.taxa2Rows = dict()
        self.rows2Taxa = dict()

        # the next binary state to map a new character to
        self.nextState : int = 0

        ##Parse the input file into a list of sequence records
        self.seqRecords : list = alignment.get_records()
        self.aln : MSA = alignment

        ##turn sequence record objects into the matrix data
        self.populateData()
    

    def populateData(self):
        # init the map from chars to binary
        # set the number of mappable states based on the alphabet type
        if self.type == "DNA" or self.type == "RNA":
            self.bits = math.pow(2, 8)  # 2^4?
            self.data = np.array([], dtype=np.int8)
        elif self.type == "SNP":
            self.bits = math.pow(2, 8)  # 2^2?
            self.data = np.array([], dtype=np.int8)
        elif self.type == "PROTEIN":
            # Prespecified substitution rates between aminos
            #
            self.bits = math.pow(2, 32)
            self.data = np.array([], dtype=np.int32)
        else:
            self.bits = math.pow(2, 64)
            self.data = np.array([], dtype=np.int64)

        self.stateMap = {}

        # translate the data into the matrix
        index = 0
        
        for r in self.seqRecords:
            self.taxa2Rows[r.get_name()] = index
            self.rows2Taxa[index] = r.get_name()
            print("mapping " + str(r.get_name()) + " to row number " + str(index))
            
            for char in r.get_seq():
                # use the alphabet to map characters to their bit states and add to
                # the data as a column
                self.data = np.append(self.data, np.array([self.alphabet.map(char)]), axis=0)
        
            index += 1

        # the dimensions of the uncompressed matrix
        self.numTaxa = self.aln.num_groups()  # = num taxa if each group is only made of one taxa
        self.seqLen = len(self.seqRecords[0].get_seq())

        # compress the matrix and fill out the locations and count fields
        # TODO: ASK ABOUT SIMPLIFICATION SCHEME
        if self.type == "DNA":
            self.simplify()
            #self.uniqueSites = self.seqLen
        else:
            self.uniqueSites = self.seqLen

    def map(self, state):
        """

        TODO: STILL UNUSED AS OF 12/16/22
        Return f(state), where f: {alphabet} -> int.
        
        If state is not yet defined in the map, then select its binary 
        representation and then return it. Otherwise, simply return f(state).
        """

        if state not in self.stateMap:
            if len(self.stateMap.keys()) > self.bits:
                raise MatrixException2
            else:
                # map state to the next state
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

        newData : np.ndarray = np.empty((self.numTaxa, 0), dtype=np.int8)

        columnData = dict()
        uniqueSites = 0

        for i in range(self.seqLen):

            col = self.getColumn(i, self.data, 0)
            colStr = list2Str(col)

            if colStr in columnData:
                self.locations.append(columnData[colStr])
            else:
                columnData[colStr] = i
                self.locations.append(i)
                uniqueSites += 1
                newData = np.append(newData, col.reshape((col.size, 1)), axis=1)

        self.uniqueSites = uniqueSites
        self.populateCounts(newData)
        self.data = newData

    def verification(self):
        print(self.data.reshape(self.get_num_taxa, self.uniqueSites))
        print(self.locations)
        print(self.count)
        print(self.stateMap)

    def getIJ(self, row, col):
        return self.data[row][col]

    def getIJ_char(self, row, col):
        return self.charMatrix()[row][col]

    def rowGivenName(self, label):
        return self.taxa2Rows[label]

    def getSeq(self, label):
        return self.charMatrix()[self.rowGivenName(label)]

    def get_number_seq(self, label):
        return self.data[self.rowGivenName(label)]

    def getColumn(self, i, data, sites):
        """
        Returns ith column of data matrix
        """

        if sites == 0:
            data = data.reshape(self.numTaxa, self.seqLen)
        else:
            data = data.reshape(self.numTaxa, sites)

        return data[:, i]

    def getColumnAt(self, i):
        """
        Returns ith column of data matrix
        """
        return self.data[:, i]

    def siteCount(self):
        return self.uniqueSites

    def populateCounts(self, newData):
        """
                Generates a count list that maps the ith distinct column to the number
                of times it appears in the original alignment matrix
                """
        for i in range(self.uniqueSites):
            col = self.getColumn(i, newData, self.uniqueSites)
            first = True
            for k in range(self.seqLen):
                col2 = self.getColumn(k, self.data, 0)

                if list(col) == list(col2):
                    if first:
                        self.count.append(1)
                        first = False
                    else:
                        self.count[i] += 1

    def asDNA(self):
        if (self.type == "RNA" or self.type == "Proteins"):
            raise MatrixCastError(self.type, "DNA")
        elif self.type == "codon":
            # switch from codon matrix to DNA matrix
            self.type = "DNA"
            self.populateData()
        return

    def asProtein(self):
        if (self.type == "RNA" or self.type == "Proteins"):
            raise MatrixCastError(self.type, "DNA")
        elif self.type == "codon":
            # switch from codon matrix to DNA matrix
            self.type = "DNA"
            self.populateData()
        return

    def charMatrix(self):
        matrix = np.zeros(self.data.shape, dtype='U1')
        rows, cols = matrix.shape

        for i in range(rows):
            for j in range(cols):
                matrix[i][j] = self.alphabet.reverse_map(self.data[i][j])

        return matrix

    def get_num_taxa(self):
        return self.numTaxa

    def name_given_row(self, index):
        return self.rows2Taxa[index]

    def get_type(self):
        return self.type
