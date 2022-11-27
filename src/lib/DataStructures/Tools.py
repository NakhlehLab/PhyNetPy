from Matrix import Matrix
from Bio import AlignIO
from NetworkBuilder import NetworkBuilder
from GTR import *
from Alphabet import Alphabet
from BirthDeath import CBDP
from MSA import MSA
from MetropolisHastings import ProposalKernel, MetropolisHastings
from ModelGraph import Model

def MCMC_TREE_SEQ(filenames: list, treeout: str, outfile: str, submodel=JC(), num_iter: int = 3000):
    """
    Finds the maximum likelihood *tree* given a nexus file containing taxa and DNA sequences.

    Using Hill Climbing with a random starting tree conditioned on the number of taxa

    Inputs:
        filenames: A list of nexus files to run MCMC_SEQ on
        outfile: A filename to write a summary of the calculations
        treeout: A filename to write the final Maximum Likelihood tree to (newick format)
        submodel: A substitution model for the model graph. Default is JC
        num_iter: Maximum number of iterations to use for hill climbing. Default is 3000 iterations

    Outputs:
        An array of likelihood values
    """
    
    likelihoods = []
    for file in filenames:
        msa = AlignIO.read(file, "nexus")

        data = Matrix(msa)  # default is to use the DNA alphabet

        hill = MetropolisHastings(ProposalKernel(), submodel, data, num_iter)

        final_state = hill.run()

        final_state.current_model.summary(treeout, outfile) #TODO: Duplicate writes
        likelihoods.append(final_state.current_model.likelihood())


def MCMC_BiMarkers(filenames: list, treeout: str, outfile: str, submodel=JC(), num_iter: int = 3000):
    
    """
    Finds the maximum likelihood network given a nexus file containing taxa and DNA sequences.

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

        data = Matrix(msa, Alphabet("SNP"))  # default is to use the DNA alphabet

        hill = MetropolisHastings(ProposalKernel(), submodel, data, num_iter)

        final_state = hill.run()

        final_state.current_model.summary(treeout, outfile)
        likelihoods.append(final_state.current_model.likelihood())
    
    
    
    
def SNAPP_Likelihood(filename, grouping=None):
    """
    Computes the SNAPP Likelihood for a nexus file that contains Bi-allelic data for a sampled set of taxa

    filename-- the path to a nexus file
    grouping-- an array that tells the MSA how to group the taxa for sampling. If left None, then each taxon will be its
               own group. Format is the number of taxa in each group, ie for 6 total, [3, 2, 1] is a valid grouping
    """

    aln = MSA(filename, grouping)
    network = CBDP(1, .5, aln.num_groups()).generateTree()
    network.printGraph()
    u = 1
    v = 1
    coal = .2
    snp_params={"samples": len(aln.get_records()), "u": u, "v": v, "coal" : coal, "grouping":True}
    m = Matrix(aln, Alphabet("SNP"))
    snp_model = Model(network, m, snp_params=snp_params)

    return snp_model.SNP_likelihood()


def SNAPP_with_tree(filename, u, v, coal, show_partials = False, path=None):
    aln = MSA(filename)
    network = NetworkBuilder(filename).get_all_networks()[0]
    network.printGraph()
    snp_params={"samples": len(aln.get_records()), "u": u, "v": v, "coal" : coal, "grouping":False}
    snp_model = Model(network, Matrix(aln, Alphabet("SNP")), None, snp_params=snp_params, verbose = show_partials)
    #snp_model.network.visualize_graph(path)

    return snp_model.SNP_likelihood()


print(SNAPP_Likelihood("C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\SNPtests\\snp_samples.nex", [3, 3, 3]))

print(SNAPP_with_tree("C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\SNPtests\\snptest_ez.nex", 1, 1, .2,  show_partials = True, path="tree.html"))

print(SNAPP_with_tree("C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\SNPtests\\snp_network_test.nex", 1, 1, .2,  show_partials = True, path="tree.html"))

# print(ML_TREE(["C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\felsensteinTests\\4taxaMultipleSites.nex"], ))

# n = CBDP(.2, .02, 50).generateTree()
# print(n.sim_seqs(15))