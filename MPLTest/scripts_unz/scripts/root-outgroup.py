import treeswift as ts
import warnings
import sys


def reroot_on_edge(tree, node):
    # suppress reroot is maybe buggy warning from treeswift
    with warnings.catch_warnings(action="ignore"):
        if not node.is_root():
            if (
                not hasattr(node, "edge_length")
                or node.edge_length is None
                or node.edge_length == 0
            ):
                node.edge_length = 1
            tree.reroot(node, length=node.edge_length / 2)


def root_outgroup(tree):
    if "OUT" in [c.label for c in tree.root.children]:
        return tree
    outgroup = [l for l in tree.traverse_leaves() if l.label == "OUT"][0]
    outgroup.edge_length = 1
    reroot_on_edge(tree, outgroup)
    tree.is_rooted = False
    tree.suppress_unifurcations()
    return tree


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"USAGE: {sys.argv[0]} IN_FILE")
        sys.exit(1)
    in_file = sys.argv[1]
    print(in_file, sep=' ')
    parts = in_file.rsplit('.', 1)
    out_file = f"{parts[0]}-rooted.{parts[1]}"
    with open(in_file, "r") as fi, open(out_file, "w") as fo:
        for line in fi:
            fo.write(f"{root_outgroup(ts.read_tree_newick(line)).newick()}\n")