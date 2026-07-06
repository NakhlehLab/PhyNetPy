import sys
import random as r

r.seed(0)


def add_outgroup(t):
    return f"(OUT:{r.uniform(0.9, 1.0)},{t[:-2]}:{r.uniform(0.0, 0.1)});"


with open(sys.argv[1], "r") as f:
    nets = list(map(add_outgroup, [t for t in f]))
with open(sys.argv[1], "w") as f:
    for net in nets:
        f.write(net + "\n")