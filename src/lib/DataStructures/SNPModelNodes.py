import numpy as np

from ModelGraph import CalculationNode
from math import comb, pow

from src.lib.DataStructures.Alphabet import Alphabet
from src.lib.DataStructures.MSA import MSA
from src.lib.DataStructures.Matrix import Matrix


def SNP_compute_partials(matrix : Matrix, phased=False):
    if phased:
        r = [matrix.get_num_taxa() - sum(matrix.getColumnAt(i)) for i in range(matrix.uniqueSites)] #sum of the columns
        x = [r[i] / matrix.get_num_taxa() for i in range(matrix.uniqueSites)] #r_i / n
    else:
        r = [2 * matrix.get_num_taxa() - sum(matrix.getColumnAt(i)) for i in range(matrix.uniqueSites)]
        x = [r[i] / 2 * matrix.get_num_taxa() for i in range(matrix.uniqueSites)]

    partials = []

    for taxa in range(matrix.get_num_taxa()):
        likelihoods = np.zeros(matrix.uniqueSites)
        for site in range(matrix.uniqueSites):
            likelihoods[site] = comb(2 * matrix.get_num_taxa(), r[site]) * pow(x[site], r[site]) * pow((1 - x[site]), 2 * matrix.get_num_taxa() - r[site])

        partials.append(likelihoods)





class SNPLeafNode(CalculationNode):

    def update(self, *args, **kwargs):
        pass

    def get(self):
        pass

    def calc(self, *args, **kwargs):
        pass


class SNPInternalNode(CalculationNode):

    def update(self, *args, **kwargs):
        pass

    def get(self):
        pass

    def calc(self, *args, **kwargs):
        pass




msa = MSA("C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\SNPtests\\snptest1.nex")
mat = Matrix(msa, Alphabet("SNP"))
print(mat.charMatrix())

print(SNP_compute_partials(mat))