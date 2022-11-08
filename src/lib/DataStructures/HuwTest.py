import numpy
import scipy
import ete3
import sympy
import mpmath

from scipy.special import binom

def solveCentralBlockTransposed(_n, y, offset, coal):

	x = numpy.zeros(_n + 1)
	K = (-(coal * (_n * (_n - 1.0))) / 2.0) - ((_n * v) + offset)

	if u == 0.0 and v == 0.0:
		for r in range(0, _n + 1):
			x[r] = y[r] / K
	elif u == 0.0:
		Mrr = K
		x[0] = y[0] / Mrr
		for r in range(1, _n + 1):
			Mrr = K + r * (v - u)
			x[r] = (y[r] - ((_n - r + 1.0) * v) * x[r - 1]) / Mrr
	elif v == 0.0:
		Mrr = K + _n * (v - u)
		x[_n] = y[_n] / Mrr
		for r in range(0, _n):
			r = _n - r - 1
			Mrr = (K + r * (v - u))
			x[r] = (y[r] - ((r + 1.0) * u) * x[r + 1]) / Mrr
	else:
		d = numpy.zeros(_n + 1)
		e = numpy.zeros(_n + 1)
		d[0] = K
		e[0] = y[0]
		for r in range(1, _n + 1):
			m = ((_n - r + 1.0) * v) / d[r - 1]
			d[r] = K + r * (v - u) - m * r * u
			e[r] = y[r] - m * e[r - 1]

		x[_n] = e[_n] / d[_n]
		for r in range(0, _n):
			r = _n - r - 1
			x[r] = (e[r] - (r + 1) * u * x[r + 1]) / d[r]

	return x

def findOrthogonalVector(u, v, m_root, n_r_pair_count, theta):
	coal = 2/theta

	x = numpy.zeros(n_r_pair_count + 1)
	xn = numpy.zeros(m_root + 1)
	yn = numpy.zeros(m_root + 1)

	xn[0] = u
	xn[1] = v
	x[1] = u
	x[2] = v
	xptr = 3

	for _n in range(2, n + 1):
		yn[0] = - ((coal * (_n - 1.0) * _n) / 2.0) * xn[0]
		for r in range(1, _n):
			yn[r] = - ((coal * (r - 1.0) * _n) / 2.0) * xn[r - 1] - ((coal * (_n - 1.0 - r) * _n) / 2.0) * xn[r]

		yn[_n] = - ((coal * (_n - 1.0) * _n) / 2.0) * xn[_n - 1]

		xn = solveCentralBlockTransposed(_n, yn, 0, coal)

		for i in range(0, len(xn)):
			x[xptr] = xn[i]
			xptr += 1

	return x

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

tree_newick = "(A:0.4223922574136966,(B:0.1395037624923693,C:0.1395037624923693):0.28288849492132734);"

tree_root = ete3.Tree(tree_newick)

def calculate_partials(node, site_i, m_root, n_r_indices, q_matrix):
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
	# partial likelihoods at a speciation
	else:
		m_y, ft_y = calculate_partials(node.children[0], site_i, m_root, n_r_indices, q_matrix)
		m_z, ft_z = calculate_partials(node.children[1], site_i, m_root, n_r_indices, q_matrix)

		m = m_y + m_z
		n_r_pair_count = numpy.sum(numpy.arange(m) + 2)
		fb = numpy.zeros(n_r_pair_count)
		ft = numpy.zeros(n_r_pair_count)

		for n_y in range(1, m_y + 1):
			for n_z in range(1, m_z + 1):
				n = n_y + n_z

				for r_y in range(0, n_y + 1):
					for r_z in range(0, n_z + 1):
						r = r_y + r_z

						n_r_i = n_r_indices[n, r]

						n_x = n - n_y
						r_x = r - r_y

						n_y_r_y_i = n_r_indices[n_y, r_y]
						n_z_r_z_i = n_r_indices[n_x, r_x]

						fb[n_r_i] += ft_y[n_y_r_y_i] * ft_z[n_z_r_z_i] * ((binom(n_y, r_y) * binom(n_x, r_x)) / binom(n, r))

        
	# partial likelihoods along a branch
	# exp_qt = scipy.linalg.expm(q_matrix * node.dist)
	exp_qt = mpmath.expm(q_matrix[0:n_r_pair_count,0:n_r_pair_count] * node.dist, method = "taylor")

	for nt in range(1, m + 1):
		for rt in range(0, nt + 1):
			nt_rt_i = n_r_indices[nt, rt]

			for nb in range(nt, m + 1):
				for rb in range(nb + 1):
					nb_rb_i = n_r_indices[nb, rb]

					ft[nt_rt_i] += fb[nb_rb_i] * exp_qt[nb_rb_i, nt_rt_i]

	return m, ft

for site_i in range(n_sites):
	m_root = max_lineages_per_site[site_i]

	n_r_pair_count = 0
	n_r_indices = numpy.zeros((m_root + 1, m_root + 1), dtype = numpy.uint32)
	for n in range(1, m_root + 1):
		for r in range(n + 1):
			n_r_indices[n, r] = n_r_pair_count
			n_r_pair_count += 1

	q_matrix = numpy.zeros((n_r_pair_count, n_r_pair_count))

	for n in range(1, m_root + 1):
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

	x = findOrthogonalVector(u, v, m_root, n_r_pair_count, theta)

	m_y, ft_y = calculate_partials(tree_root.children[0], site_i, m_root, n_r_indices, q_matrix)
	m_z, ft_z = calculate_partials(tree_root.children[1], site_i, m_root, n_r_indices, q_matrix)

	fb = numpy.zeros(n_r_pair_count)

	for n in range(2, m_root + 1):
		for r in range(0, n + 1):
			n_r_i = n_r_indices[n, r]

			for n_y in range(1, m_y + 1):
				for r_y in range(0, min(r + 1, n_y + 1)):
					n_x = n - n_y
					r_x = r - r_y
					n_y_r_y_i = n_r_indices[n_y, r_y]
					n_z_r_z_i = n_r_indices[n_x, r_x]
					fb[n_r_i] += ft_y[n_y_r_y_i] * ft_z[n_z_r_z_i] * ((binom(n_y, r_y) * binom(n_x, r_x)) / binom(n, r))

	print("site %d root Fb = %s" % (site_i, ", ".join(["%.06f" % x for x in fb])))