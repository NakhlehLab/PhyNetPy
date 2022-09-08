from dendropy.simulate.treesim import birth_death_tree
from dendropy.interop import seqgen
import dendropy

#birth_death_tree(1, .5, num_extant_tips= 10, gsa_ntax=20).as_string(schema="nexus")


trees = dendropy.TreeList.get(
        path="C:\\Users\\markk\\OneDrive\\Documents\\PhyloPy\\PhyloPy\\src\\test\\MetroHastingsTests\\sample.nex",
        schema="nexus")
s = seqgen.SeqGen()

# generate one alignment per tree
# as substitution model is not specified, defaults to a JC model
# will result in a DataSet object with one DnaCharacterMatrix per input tree
d0 = s.generate(trees)
print(len(d0.char_matrices))
print(d0.char_matrices[0].as_string("nexus"))

# instruct Seq-Gen to scale branch lengths by factor of 0.1
# note that this does not modify the input trees
s.scale_branch_lens = 0.1

# more complex model
s.char_model = seqgen.SeqGen.GTR
s.state_freqs = [0.25, 0.25, 0.25, 0.25]
s.general_rates = [1, 1, 1, 1, 1, 1]
d1 = s.generate(trees)
print(len(d0.char_matrices))
print(d0.char_matrices[0].as_string("nexus"))