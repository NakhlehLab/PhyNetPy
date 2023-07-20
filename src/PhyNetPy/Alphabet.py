""" 
Author : Mark Kessler
Last Stable Edit : 7/16/23
First Included in Version : 0.1.0

"""


class AlphabetError(Exception):
    def __init__(self, message="Something went wrong mapping chars to numbers"):
        self.message = message
        super().__init__(self.message)


### HELPER FUNCTIONS ###

def n_ploidy(ploidy : int)->dict:
    """
    Only for SNP alphabet initialization

    Args:
        ploidy (int): The ploidyness value of a species (ie, humans = 2, some plants > 2, etc)

    Returns:
        dict: Returns an SNP alphabet map that maps str(int)->int for 0 <= int <= ploidy, plus the various extra character mappings
    """
    alphabet = {}
    for num in range(ploidy+1):
        alphabet[str(num)] = num
    
    alphabet["?"] = ploidy + 1
    alphabet["N"] = ploidy + 1
    alphabet["-"] = ploidy + 1
    
    return alphabet




class Alphabet:
    """
    Class that deals with the mapping from characters to state values that have partial likelihood values associated with them
    This state mapping is primarily based on Base10 -> Binary conversions such that the decimal numbers become a generalized
    version of the one-hot encoding scheme.
    """


    ### DNA MAPPING INFORMATION ###
    # Symbol(s)	Name	   Partial Likelihood
    #     A	  Adenine	   [1,0,0,0] -> 1
    #     C	  Cytosine	   [0,1,0,0] -> 2
    #     G	  Guanine	   [0,0,1,0] -> 4
    #     T U	  Thymine  [0,0,0,1] -> 8
    # Symbol(s)	Name	   Partial Likelihood
    #     N ? X	Any 	   A C G T ([1,1,1,1] -> 15)
    #     V	    Not T	   A C G ([1,1,1,0] -> 7)
    #     H	    Not G	   A C T ([1,1,0,1] -> 11)
    #     D	    Not C	   A G T ([1,0,1,1] -> 13)
    #     B	    Not A	   C G T ([0,1,1,1] -> 14)
    #     M	    Amino	   A C ([1,1,0,0] -> 3)
    #     R	    Purine	   A G ([1,0,1,0] -> 5)
    #     W	    Weak	   A T ([1,0,0,1] -> 9)
    #     S	    Strong	   C G ([0,1,1,0] -> 6)
    #     Y	    Pyrimidine C T ([0,1,0,1] -> 10)
    #     K	    Keto	   G T ([0,0,1,1] -> 12)

    DNA = {"A": 1, "C": 2, "M": 3, "G": 4, "R": 5, "S": 6, "V": 7, "T": 8, "W": 9, "Y": 10,
           "H": 11, "K": 12, "D": 13, "B": 14, "N": 15, "?": 15, "-": 0, "X":15}
    

    #Contains the T==U equivalency
    
    RNA = {"A": 1, "C": 2, "M": 3, "G": 4, "R": 5, "S": 6, "V": 7, "T": 8, "U": 8, "W": 9, "Y": 10,
           "H": 11, "K": 12, "D": 13, "B": 14, "N": 15, "?": 15, "-": 0, "X":15}
    

    PROTEIN = {"-": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10,
               "K": 11, "L": 12, "M": 13, "N": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19, "V": 20, "W": 21,
               "X": 22, "Y": 23, "Z": 24, ".": 25}


    CODON = {"-": 0, "A": 1, "C": 2, "M": 3, "G": 4, "R": 5, "S": 6, "V": 7, "T": 8, "U": 8, "W": 9, "Y": 9,
             "H": 10, "K": 11, "D": 12, "B": 13, "N": 14, "X": 14, ".": 14}


    SNP = {"-": 3, "N": 3, "?": 3, "0": 0, "1": 1, "2": 2}

  

    def __init__(self, type : str, myAlphabet : dict = {}, snp_ploidy : int = None):
        """

        Args:
            type (str): A string from the set {"user", "DNA", "RNA", "CODON", "PROTEIN", "SNP"}
            myAlphabet (dict, optional): A user alphabet if none of the provided fits the data. Defaults to {}.
            snp_ploidy (int, optional): Only used for SNP alphabets. Describes the maximum ploidyness value of the data set. Defaults to None.

        Raises:
            AlphabetError: On any alphabet construction error
        """
        self.type = type

        if type == "user":
            self.alphabet = myAlphabet
        else:
            if type == "DNA" or type == "RNA" or type == "CODON":
                self.alphabet = self.DNA
            elif type == "PROTEIN":
                self.alphabet = self.PROTEIN
            elif type == "SNP":
                if snp_ploidy is None:
                    raise AlphabetError("Need ploidyness value as input when SNP is the alphabet")
                self.alphabet = n_ploidy(snp_ploidy)
            else:
                raise AlphabetError("Undefined type given. Please choose from the valid types or provide your own alphabet")

    def map(self, char:str) -> int:
        """
        Return mapping for a character encountered in a nexus file

        Args:
            char (str): nexus file matrix data point

        Raises:
            AlphabetError: if the char encountered is undefined for the data mapping

        Returns:
            int: the integer corresponding to char in the alphabet mapping
        """
        try:
            return self.alphabet[char]
        except KeyError:
            raise AlphabetError("Nexus contents contain character <" + char + "> that is invalid for alphabet type")
        finally:
            pass

    def get_type(self):
        return self.type

    def reverse_map(self, state:int)->str:
        """
        Get the character that maps to "state" in the given alphabet

        Args:
            state (int): a value in the alphabet map

        Raises:
            AlphabetError: if the provided state is not a valid one in the alphabet

        Returns:
            str: the key that maps to "state"
        """
        if state not in self.alphabet.values():
            raise AlphabetError("Given state does not exist in alphabet")
        
        for key in self.alphabet.keys():
            if self.alphabet[key] == state:
                return key
        
    