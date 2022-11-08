import numpy
import scipy
import ete3

from scipy.special import binom

# leaf_samples = {"A": {"a1": "01"},
# 				"B": {"b1": "11", "b2": "11"},
# 				"C": {"c1": "00", "c2": "10", "c3": "10"}}

leaf_samples = {"A": {"a1": "01"},
                                "B": {"b1": "11"},
                                "C": {"c1": "00"}}

ploidy = 1

n_leaves = len(leaf_samples)

n_sites = None
n_samples_per_site = None
r_samples_per_site = None

max_lineages_per_site = None

leaf_order = sorted(leaf_samples)

for leaf_i, leaf_name in enumerate(leaf_order):
        for sample_name in leaf_samples[leaf_name]:
                sample_sequence = leaf_samples[leaf_name][sample_name]

                if n_sites == None:
                        n_sites = len(sample_sequence)
                        n_samples_per_site = numpy.zeros((n_sites, n_leaves), dtype = numpy.uint32)
                        r_samples_per_site = numpy.zeros((n_sites, n_leaves), dtype = numpy.uint32)
                        max_lineages_per_site = numpy.zeros(n_sites, dtype = numpy.uint32)

                for site_i in range(n_sites):
                        max_lineages_per_site[site_i] += ploidy
                        n_samples_per_site[site_i, leaf_i] += ploidy
                        r_samples_per_site[site_i, leaf_i] += int(sample_sequence[site_i])

theta = 0.2

u = 1
v = 1

tree_newick = "((A:0.3079429072013367,C:0.30794290720133694):0.37234262467408746,B:0.6802855318754237);"

tree_root = ete3.Tree(tree_newick)

def calculate_partials(node, site_i, n_r_indices, q_matrix):
        # partial likelihoods for a leaf
        if node.is_leaf():
                leaf_i = leaf_order.index(node.name)
                m = n_samples_per_site[site_i, leaf_i]
                r = r_samples_per_site[site_i, leaf_i]
                m_r_i = n_r_indices[m, r]

                n_r_pair_count = numpy.sum(numpy.arange(m) + 2)
                fb = numpy.zeros(n_r_pair_count)
                ft = numpy.zeros(n_r_pair_count)
                fb[m_r_i] = 1
                print("----leaf fb----")
                print(fb)
                print("---------------")
        # partial likelihoods at a speciation
        else:
                m_y, ft_y = calculate_partials(node.children[0], site_i, n_r_indices, q_matrix)
                m_z, ft_z = calculate_partials(node.children[1], site_i, n_r_indices, q_matrix)

                m = m_y + m_z
                n_r_pair_count = numpy.sum(numpy.arange(m) + 2)
                fb = numpy.zeros(n_r_pair_count)

                for n_y in range(1, m_y + 1):
                        for n_z in range(1, m_z + 1):
                                n = n_y + n_z

                                for r_y in range(0, n_y + 1):
                                        for r_z in range(0, n_z + 1):
                                                r = r_y + r_z

                                                n_r_i = n_r_indices[n, r]

                                                n_y_r_y_i = n_r_indices[n_y, r_y]
                                                n_z_r_z_i = n_r_indices[n_z, r_z]

                                                fb[n_r_i] += ft_y[n_y_r_y_i] * ft_z[n_z_r_z_i] * ((binom(n_y, r_y) * binom(n_z, r_z)) / binom(n, r))
                
                print("----internal fb----")
                print(fb)
                print("-------------------")

        if node.is_root():
                return m, fb
        # partial likelihoods along a branch
        else:
                ft = numpy.zeros(n_r_pair_count)

                exp_qt = scipy.linalg.expm(q_matrix[0:n_r_pair_count,0:n_r_pair_count] * node.dist)

                for nt in range(1, m + 1):
                        for rt in range(0, nt + 1):
                                nt_rt_i = n_r_indices[nt, rt]

                                for nb in range(nt, m + 1):
                                        for rb in range(nb + 1):
                                                nb_rb_i = n_r_indices[nb, rb]

                                                ft[nt_rt_i] += fb[nb_rb_i] * exp_qt[nb_rb_i, nt_rt_i]
                
                print("----ft----")
                print(ft)
                print("----------")

                return m, ft

log_likelihood = 0

max_lineages = max(max_lineages_per_site)

n_r_pair_count = 0
n_r_indices = numpy.zeros((max_lineages + 1, max_lineages + 1), dtype = numpy.uint32)
for n in range(1, max_lineages + 1):
        for r in range(n + 1):
                n_r_indices[n, r] = n_r_pair_count
                n_r_pair_count += 1

q_matrix = numpy.zeros((n_r_pair_count, n_r_pair_count))

for n in range(1, max_lineages + 1):
        for r in range(n + 1):
                n_r_i = n_r_indices[n, r]

                # off-diagonals
                if r > 0:
                        n_r_minus_i = n_r_indices[n, r - 1]
                        q_matrix[n_r_i, n_r_minus_i] = (n - r + 1) * v
                if r < n:
                        n_r_plus_i = n_r_indices[n, r + 1]
                        q_matrix[n_r_i, n_r_plus_i] = (r + 1) * u
                if n > 1 and r < n:
                        n_minus_r_i = n_r_indices[n - 1, r]
                        q_matrix[n_r_i, n_minus_r_i] = ((n - 1 - r) * n) / theta
                if n > 1 and r > 0:
                        n_minus_r_minus_i = n_r_indices[n - 1, r - 1]
                        q_matrix[n_r_i, n_minus_r_minus_i] = ((r - 1) * n) / theta

                # diagonal
                q_matrix[n_r_i, n_r_i] = -((n * (n - 1)) / theta) - ((n - r) * v) - (r * u)

q_null_space = scipy.linalg.null_space(q_matrix).reshape(n_r_pair_count)
x = q_null_space / (q_null_space[0] + q_null_space[1]) # normalized so the first two values sum to one

for site_i in range(n_sites):
        m_root, root_fb = calculate_partials(tree_root, site_i, n_r_indices, q_matrix)
        print(root_fb)
        log_likelihood += numpy.log(numpy.sum(root_fb[2:] * x[2:]))

print(log_likelihood)

