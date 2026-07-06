#!/usr/bin/env julia
if length(ARGS) != 5 
        print("usage: run_snaq.jl [gtrees] [start tree] [output] [num ret] [threads]\n")
        exit(1)
end
path_gtrees = ARGS[1]
path_start_tree  = ARGS[2]
output   = ARGS[3]
n_ret = parse(Int, ARGS[4])
threads = parse(Int, ARGS[5])

using Distributed
addprocs(threads)
@everywhere using PhyloNetworks, SNaQ, DataFrames

gtrees = readmultinewick(path_gtrees)
q, t = countquartetsintrees(gtrees)
cf = readtableCF(DataFrame(tablequartetCF(q,t), copycols=false))

start_tree = readnewick(path_start_tree)

result_net = snaq!(start_tree, cf, hmax=n_ret, filename=output, seed=0, runs=threads)
writenewick(result_net, output)