#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
##  -- PhyNetPy --
##  Library for the Development and use of Phylogenetic Network Methods
##
##  Copyright 2025 Mark Kessler, Luay Nakhleh.
##  All rights reserved.
##
##  See "LICENSE.txt" for terms and conditions of usage.
##
##  If you use this work or any portion thereof in published work,
##  please cite it as:
##
##     Mark Kessler, Luay Nakhleh. 2025.
##
##############################################################################

""" 
Author : Mark Kessler
Last Edit : 3/11/25
First Included in Version : 1.0.0

Docs   - [x]
Tests  - [ ]
Design - [ ]
"""



import numpy as np
import math
from Alphabet import *
from MSA import MSA, DataSequence
import numpy.typing as npt


###################
#### CONSTANTS ####
###################



##########################
#### HELPER FUNCTIONS ####
##########################

def _list_to_string(my_list : list[str]) -> str:
    """
    Turns a list of characters into a string
    
    Args:
        my_list (list[str]): A list of characters/strings

    Returns:
        str: one string that is the concatentation of each element of the list
    """
    sb = ""
    for char in my_list:
        sb += char
    return sb

#########################
#### EXCEPTION CLASS ####
#########################

class MatrixError(Exception):
    """
    This exception is raised when there is an error either in parsing data
    into the matrix object, or if there is an error during any sort of 
    operation
    """

    def __init__(self, message : str = "Matrix Error") -> None:
        """
        Create new MatrixError with custom message.

        Args:
            message (str, optional): Custom error message. Defaults to 
                                     "Matrix Error".
        Returns:
            N/A                            
        """
        self.message = message
        super().__init__(self.message)

################
#### MATRIX ####
################

class Matrix:
    """
    Class that stores and reduces MSA data to only the relevant/unique sites
    that exist. The only reduction mechanism so far is applicable only to DNA.
    All other data types will simply be stored in a 2d numpy matrix.
    
    Accepts any data that is defined by the Alphabet class in Alphabet.py.
    """
    
    def __init__(self, 
                 alignment : MSA, 
                 alphabet : Alphabet = Alphabet(DNA)) -> None:
        """
        Takes one single MSA object, along with an Alphabet object,
        represented as either DNA, RNA, PROTEIN, CODON, or USER. 
        The default is DNA. 

        Args:
            alignment (MSA): Multiple Sequence Alignment (MSA) object.
            alphabet (Alphabet, optional): An alphabet for mapping characters
                                           to numerics. Defaults to
                                           Alphabet(DNA).
        Returns:
            N/A
        """
        # ith element of the array = column i's distinct site pattern index in
        # the compressed matrix
        self.unique_sites : int = 0
        self.data : npt.NDArray[np.int_] = np.ndarray((0,), dtype = np.uint8)
        self.locations : list = list()

        # ith element of the array = count of the number of times
        # column i appears in the original uncompressed matrix
        self.count : list = list()
        self.alphabet : Alphabet = alphabet
        self.type : str = alphabet.get_type()
        self.taxa_to_rows : dict[str, int] = dict()
        self.rows_to_taxa : dict[int, str]= dict()

        # the next binary state to map a new character to
        self.next_state : int = 0

        ##Parse the input file into a list of sequence records
        self.seqs : list[DataSequence] = alignment.get_records()
        self.aln : MSA = alignment

        ##turn sequence record objects into the matrix data
        self.populate_data()
    
    def populate_data(self) -> None:
        """
        Stores and simplifies the MSA data.
        
        Args:
            N/A
        Returns:
            N/A
        """
        
        # init the map from chars to binary
        # set the number of mappable states based on the alphabet type
        if self.type == "DNA" or self.type == "RNA":
            self.bits = math.pow(2, 8)  # 2^4?
            self.data = np.array([], dtype = np.int8)
        elif self.type == "SNP":
            self.bits = math.pow(2, 8)  # 2^2?
            self.data = np.array([], dtype = np.int8)
        elif self.type == "PROTEIN":
            # Prespecified substitution rates between aminos
            self.bits = math.pow(2, 32)
            self.data = np.array([], dtype = np.int32)
        else:
            self.bits = math.pow(2, 64)
            self.data = np.array([], dtype = np.int64)

        self.state_map : dict[str, int]= {}

        # translate the data into the matrix
        index = 0
        
        for r in self.seqs:

            self.taxa_to_rows[r.label] = index # type: ignore
            self.rows_to_taxa[index] = r.label # type: ignore
            # print("mapping " + str(r.label) + \
            #       " to row number " + str(index))
        
            for char in r.get_seq():
                # use the alphabet to map characters to their bit states 
                # and add to the data as a column
                char_array : list[int] = [self.alphabet.map(char)] # type: ignore
                char_as_array = np.array(char_array)
                self.data = np.append(self.data, char_as_array, axis = 0)
        
            index += 1

        # the dimensions of the uncompressed matrix
        # = num taxa if each group is only made of one taxa
        self.num_taxa = self.aln.num_groups()  
        
        self.seq_len = len(self.seqs[0].get_seq())

        # compress the matrix and fill out the locations and count fields
        # only compresses for DNA as of 4/3/24
        if self.type == "DNA":
            self.simplify()
        else:
            self.unique_sites = self.seq_len

    def simplify(self) -> None:
        """
        Reduces the matrix of data by removing non-unique site patterns, 
        and records the location and count of the unique site patterns.
        
        Args:
            N/A
        Returns:
            N/A
        """

        new_data : np.ndarray = np.empty((self.num_taxa, 0), dtype = np.int8)

        column_data = dict()
        unique_sites = 0

        for i in range(self.seq_len):

            col = self.get_column(i, self.data, 0)
            col_str = _list_to_string(col) # type: ignore

            if col_str in column_data:
                self.locations.append(column_data[col_str])
            else:
                column_data[col_str] = i
                self.locations.append(i)
                unique_sites += 1
                new_data = np.append(new_data, 
                                     col.reshape((col.size, 1)),
                                     axis = 1)

        self.unique_sites = unique_sites
        self.populate_counts(new_data)
        self.data = new_data

    def get_ij(self, i : int, j : int) -> int:
        """
        Returns the data point at row i, and column j.

        Args:
            i (int): row index
            j (int): column index

        Returns:
            int: the data point.
        """
        return self.data[i][j]

    def get_ij_char(self, i : int, j : int) -> str:
        """
        get the character at row i, column j in the character matrix that is
        associated with the data.

        Args:
            i (int): row index
            j (int): column index

        Returns:
            str: the character at [i][j]
        """
        return self.char_matrix()[i][j]

    def row_given_name(self, label : str) -> int:
        """
        Retrieves the row index of the taxa that has name 'label'

        Args:
            label (str): name of a taxon.

        Returns:
            int: a row index
        """
        return self.taxa_to_rows[label]

    def get_seq(self, label : str) -> np.ndarray:
        """
        Gets the array of characters for a given taxon.

        Args:
            label (str): the name of a taxon.

        Returns:
            np.ndarray: an array of characters, with data type 'U1'.
        """
        return self.char_matrix()[self.row_given_name(label)]

    def get_number_seq(self, label : str) -> np.ndarray:
        """
        Gets the numerical data for a given taxon with the name 'label'.

        Args:
            label (str): name of a taxon

        Returns:
            np.ndarray: a 1 dimensional array of integers, of some specific type
        """
        return self.data[self.row_given_name(label)]

    def get_column(self, i : int, data : np.ndarray, sites : int) -> np.ndarray:
        """
        Returns ith column of a data matrix, with 'sites' elements

        Args:
            i (int): column index
            data (np.ndarray): a matrix 
            sites (int): dimension of the column

        Returns:
            np.ndarray: the data at column 'i' with length 'sites'
        """

        if sites == 0:
            data = data.reshape(self.num_taxa, self.seq_len)
        else:
            data = data.reshape(self.num_taxa, sites)

        return data[:, i]

    def get_column_at(self, i : int) -> np.ndarray:
        """
        Returns ith column of the data matrix

        Args:
            i (int): column index

        Returns:
            np.ndarray: the data at column i
        """
        return self.data[:, i]

    def site_count(self) -> int:
        """
        Returns the number of unique sites in the MSA/Data

        Args:
            N/A
        Returns:
            int: number of unique sites
        """
        return self.unique_sites

    def populate_counts(self, new_data : np.ndarray) -> None:
        """
        Generates a count list that maps the ith distinct column to the number
        of times it appears in the original alignment matrix.

        Args:
            new_data (np.ndarray): The simplified data matrix, that only has
                                   distinct column values.
        Returns:
            N/A
        """
        
        for i in range(self.unique_sites):
            col = self.get_column(i, new_data, self.unique_sites)
            first = True
            for k in range(self.seq_len):
                col2 = self.get_column(k, self.data, 0)

                if list(col) == list(col2):
                    if first:
                        self.count.append(1)
                        first = False
                    else:
                        self.count[i] += 1

    def char_matrix(self) -> np.ndarray:
        """
        Get the character matrix from the matrix of alphabet states.

        Args:
            N/A
        Returns:
            np.ndarray: the character matrix, that will have equivalent 
                        dimensionality to the state matrix.
        """
        matrix = np.zeros(self.data.shape, dtype = 'U1')
        rows, cols = matrix.shape

        for i in range(rows):
            for j in range(cols):
                matrix[i][j] = self.alphabet.reverse_map(self.data[i][j])

        return matrix

    def get_num_taxa(self) -> int:
        """
        Get the number of taxa represented in this matrix.

        Args:
            N/A
        Returns:
            int: the number of taxa
        """
        return self.num_taxa

    def name_given_row(self, index : int) -> str:
        """
        Get the name of the taxa associated with the row of data at 'index'

        Args:
            index (int): a row index

        Returns:
            str: the taxon name
        """
        return self.rows_to_taxa[index]

    def get_type(self) -> str:
        """
        Get the type of data of this matrix

        Args:
            N/A
        Returns:
            str: the data type
        """
        return self.type
