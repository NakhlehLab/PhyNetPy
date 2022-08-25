from Probability import Probability
from Matrix import Matrix
from Bio import AlignIO
from NetworkBuilder import NetworkBuilder
from GTR import *


def felsenstein(filenames, model=JC()):
    """
    Calculates the log likelihood of a tree using Felsenstein's Algorithm.

    Inputs:

    1) A list of file names, all of which should be nexus files that define trees in a tree block and define a DNA
    multiple sequence alignment. Each file may define multiple trees.

    2) DEFAULT: The substitution model for the MSA. Defaults to Jukes Cantor (JC).

    Outputs:

    1) A dictionary that maps trees to their likelihood values.
    """
    trees_2_likelihood = {}

    for file in filenames:
        nb = NetworkBuilder(file)
        trees = nb.get_all_networks()
        msa = AlignIO.read(file, "nexus")
        data = Matrix(msa)
        for tree in trees:
            name = nb.name_of_network(tree)
            tree_prob = Probability(tree, model=model, data=data)
            if name in trees_2_likelihood.keys():
                name = name + "__" + file
            trees_2_likelihood[name] = tree_prob.felsenstein_likelihood()

    return trees_2_likelihood


files = ["C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxaMultipleSites.nex",
         "C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxa1Site.nex"]

print(felsenstein(files))
