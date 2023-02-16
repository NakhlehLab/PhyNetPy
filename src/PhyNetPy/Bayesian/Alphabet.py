class AlphabetError(Exception):
    def __init__(self, message="Something went wrong mapping chars to numbers"):
        self.message = message
        super().__init__(self.message)


class Alphabet:
    """
    Class that deals with the mapping from characters to state values that have partial likelihood values associated with them
    This state mapping is primarily based on Base10 -> Binary conversions such that the decimal numbers become a generalized
    version of the one-hot encoding scheme.
    
    TODO: investigate alphabet correctness for phased and unphased SNPs
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

    BINARY = {"-": 0, "0": 1, "1": 2}
  

    def __init__(self, type, myAlphabet={}):

        self.type = type

        if type == "user":
            self.alphabet = myAlphabet
        else:
            if type == "DNA" or type == "RNA" or type == "CODON":
                self.alphabet = self.DNA
            elif type == "PROTEIN":
                self.alphabet = self.PROTEIN
            elif type == "SNP":
                self.alphabet = self.SNP
            elif type == "BINARY":
                self.alphabet = self.BINARY
            else:
                # Other matrix type?
                pass

    def map(self, char):
        try:
            return self.alphabet[char]
        except KeyError:
            raise AlphabetError("Nexus contents contain character <" + char + "> that is invalid for alphabet type")
        finally:
            pass

    def getType(self):
        return self.type

    def reverseMap(self, state):
        for key in self.alphabet.keys():
            if self.alphabet[key] == state:
                return key
