from Probability import Probability
from Matrix import Matrix
from Bio import AlignIO
from NetworkBuilder import NetworkBuilder
from GTR import *
from src.lib.DataStructures.Alphabet import Alphabet
from src.lib.DataStructures.BirthDeath import CBDP
from src.lib.DataStructures.MSA import MSA
from src.lib.DataStructures.MetropolisHastings import ProposalKernel, HillClimbing
from src.lib.DataStructures.ModelGraph import Model


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


def ML_TREE(filenames, treeout, outfile, submodel=JC(), num_iter=3000):
    """
    Finds the maximum likelihood *tree* given a nexus file containing taxa and DNA sequences.

    Using Hill Climbing with a random starting tree conditioned on the number of taxa

    Inputs:
        filenames: A list of nexus files to run ML_TREE on
        outfile: A filename to write a summary of the calculations
        treeout: A filename to write the final ML tree to (newick format)
        submodel: A substitution model for the model graph. Default is JC
        num_iter: Maximum number of iterations to use for hill climbing. Default is 3000 iterations

    Outputs:
        An array of likelihood values
    """
    likelihoods = []
    for file in filenames:
        msa = AlignIO.read(file, "nexus")

        data = Matrix(msa)  # default is to use the DNA alphabet

        hill = HillClimbing(ProposalKernel(), submodel, data, num_iter)

        final_state = hill.run()

        final_state.current_model.summary(treeout, outfile)
        likelihoods.append(final_state.current_model.likelihood())


def SNAPP_Likelihood(filename, grouping=None):
    """
    Computes the SNAPP Likelihood for a nexus file that contains Bi-allelic data for a sampled set of taxa

    filename-- the path to a nexus file
    grouping-- an array that tells the MSA how to group the taxa for sampling. If left None, then each taxon will be its
               own group. Format is the number of taxa in each group, ie for 6 total, [2, 3, 1] is a valid grouping
    """

    aln = MSA(filename, grouping)
    network = CBDP(1, .5,
                   aln.num_groups()).generateTree()
    snp_model = Model(network, Matrix(aln, Alphabet("SNP")), None)

    return snp_model.SNP_likelihood()


print(SNAPP_Likelihood("C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\SNPtests\\snptest1.nex", [3,2,1]))
