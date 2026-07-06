# Script modified from https://github.com/NathanKolbow/InPhyNetSimulations
args <- commandArgs(trailingOnly = TRUE)
ntaxa <- as.numeric(args[1])
nrep <- as.numeric(args[2])
mu <- as.numeric(args[3])
nu <- as.numeric(args[4])

library(SiPhyNetwork)
library(stringr)
set.seed(0)

data_dir <- paste0("n", ntaxa)
dir.create(data_dir)

j <- 1
while (j <= nrep) {
  iter_net <- sim.bdh.taxa.ssa(
    n = ntaxa,
    numbsim = 1,
    lambda = 1,
    mu = mu,
    nu = nu,
    hybprops = c(0.5, 0.25, 0.25),
    hyb.inher.fxn = make.beta.draw(10, 10)
  )[[1]]
  if (is.numeric(iter_net)) {
    next
  }
  if (length(iter_net$tip.label) != ntaxa) {
    next
  }
  if (getNetworkLevel(iter_net) > 1) {
    next
  }
  if (nrow(iter_net$reticulation) == 0) {
    next
  }

  rep_dir <- paste0(data_dir, "/", str_pad(j - 1, width=nchar(as.character(nrep - 1)), pad="0"))
  dir.create(rep_dir)
  write.net(iter_net, paste0(rep_dir, "/true_net.nwk"))

  j <- j + 1
  cat(paste0(
    "\rFound #",
    j - 1,
    " - nretic = ",
    nrow(iter_net$reticulation)
  ))
}