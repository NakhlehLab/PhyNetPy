"""Script to create Nexus file and run PhyloNet MPL
Summer 2025, James Willson
"""

import argparse
import os


def create_nexus_file(path, gts, st, nret, threads, fixed):
    g_name, s_name = (
        os.path.basename(gts).rsplit(".", 1)[0],
        os.path.basename(st).rsplit(".", 1)[0],
    )
    identifier = f"{g_name}_{s_name}_{nret}{"_fixed" if fixed else ""}"
    output_fname = os.path.join(path, f"phylonet_output_{identifier}.txt")
    nex_file = os.path.join(path, f"{identifier}.nex")
    with open(gts, "r") as f_gts, open(st, "r") as f_st, open(nex_file, "w") as f_nex:
        nex_contents = f"""#NEXUS

BEGIN NETWORKS;

Network st = {f_st.read().strip()}

END;

BEGIN TREES;

"""
        for i, gt in enumerate(f_gts):
            nex_contents += f"Tree gt{i} = {gt.strip()}\n"
        nex_contents += f"""
END;

BEGIN PHYLONET;

InferNetwork_MPL (all) {nret} -s st {'-fs' if fixed  else ''}  -pl {threads} {output_fname};

END;
"""
        f_nex.write(nex_contents)
    return nex_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="script to create nexus file and run PhyloNet MPL",
    )
    parser.add_argument(
        "-g", "--gene-trees", type=str, required=True, help="gene tree file"
    )
    parser.add_argument(
        "-s", "--start-tree", type=str, required=True, help="start tree file"
    )
    parser.add_argument(
        "-r",
        "--reticulations",
        type=int,
        required=True,
        help="maximum number of reticulations",
    )
    parser.add_argument(
        "-n", "--threads", type=int, required=True, help="number of threads"
    )
    parser.add_argument(
        "-p", "--phylonet-path", type=str, required=True, help="path to PhyloNet.jar"
    )
    parser.add_argument("-l", "--log", type=str, help="path to save log file")
    parser.add_argument("-f", "--fixed", action="store_true", help="fix start tree")
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        help="dry run -- don't run phylonet",
    )
    args = parser.parse_args()
    path = os.path.dirname(args.gene_trees)
    nexus_file = create_nexus_file(
        path,
        args.gene_trees,
        args.start_tree,
        args.reticulations,
        args.threads,
        args.fixed,
    )
    if not args.dry_run:
        command = f"java -jar {args.phylonet_path} {nexus_file}"
        if args.log is not None:
            command += f" > {args.log}"
        os.system(command)