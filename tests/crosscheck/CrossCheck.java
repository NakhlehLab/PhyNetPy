import java.io.BufferedReader;
import java.io.FileReader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import edu.rice.cs.bioinfo.programs.phylonet.structs.network.Network;
import edu.rice.cs.bioinfo.programs.phylonet.structs.network.NetNode;
import edu.rice.cs.bioinfo.programs.phylonet.structs.network.util.Networks;
import edu.rice.cs.bioinfo.programs.phylonet.structs.tree.io.NewickReader;
import edu.rice.cs.bioinfo.programs.phylonet.structs.tree.model.Tree;
import edu.rice.cs.bioinfo.programs.phylonet.structs.tree.model.sti.STITree;

import edu.rice.cs.bioinfo.programs.phylonet.algos.MCMCseq.structs.NetNodeInfo;
import edu.rice.cs.bioinfo.programs.phylonet.algos.MCMCseq.structs.UltrametricTree;
import edu.rice.cs.bioinfo.programs.phylonet.algos.MCMCseq.distribution.GeneTreeBrSpeciesNetDistribution;
import edu.rice.cs.bioinfo.programs.phylonet.algos.MCMCseq.felsenstein.alignment.Alignment;
import edu.rice.cs.bioinfo.programs.phylonet.algos.MCMCseq.util.Utils;

/**
 * Stand-alone cross-check driver that exercises PhyloNet's OWN MCMC_SEQ
 * likelihood components on a fully-specified (gene tree, species network,
 * theta) state, so the numbers can be diffed against PhyNetPy's engine.
 *
 *   - MSNC branch-length density  log P(g | Psi)   via
 *       GeneTreeBrSpeciesNetDistribution.calculateGTDistribution
 *   - Felsenstein log-likelihood  log P(S | g)      via
 *       UltrametricTree(newick, alignment).logDensity()  (BEAGLE)
 *
 * Reads a plain-text spec file (see run_crosscheck.py for the format) and
 * prints one "RESULT <case> <FACTOR> <value>" line per computed quantity.
 */
public class CrossCheck {

    public static void main(String[] args) throws Exception {
        // Single-constant-theta regime that matches PhyNetPy's sampler.
        Utils._CONST_POP_SIZE = true;      // -> varyPopSizeAcrossBranches() == false
        Utils._ESTIMATE_POP_SIZE = true;
        Utils._START_GT_BURN_IN = false;
        Utils._START_NET_BURN_IN = false;
        Utils.SAMPLE_EMBEDDINGS = false;
        Utils._NUM_THREADS = 1;

        if (args.length < 1) {
            System.err.println("usage: CrossCheck <spec-file>");
            System.exit(2);
        }

        List<Case> cases = parse(args[0]);
        for (Case c : cases) {
            // ---- MSNC density ------------------------------------------
            try {
                double d = msncLogDensity(c);
                System.out.printf("RESULT %s MSNC %.10f%n", c.name, d);
            } catch (Throwable t) {
                System.out.printf("RESULT %s MSNC ERROR %s%n", c.name,
                        t.getClass().getSimpleName() + ":" + t.getMessage());
            }
            // ---- Felsenstein (only if sequences were supplied) ---------
            if (!c.seqs.isEmpty()) {
                try {
                    double f = felsensteinLogL(c);
                    System.out.printf("RESULT %s FELSEN %.10f%n", c.name, f);
                } catch (Throwable t) {
                    System.out.printf("RESULT %s FELSEN ERROR %s%n", c.name,
                            t.getClass().getSimpleName() + ":" + t.getMessage());
                }
            }
        }
    }

    /** log P(g | Psi) under the timed multispecies network coalescent. */
    static double msncLogDensity(Case c) throws Exception {
        Network<NetNodeInfo> net = Networks.readNetwork(c.net);
        // Replicate UltrametricNetwork.initNetHeights(): heights from branch
        // lengths, leaves at 0, single constant pop size at the root.
        for (NetNode<NetNodeInfo> node : Networks.postTraversal(net)) {
            if (node.getData() == null) {
                node.setData(new NetNodeInfo(0.0));
            }
            for (NetNode<NetNodeInfo> par : node.getParents()) {
                double dist = node.getParentDistance(par);
                if (par.getData() == null) {
                    par.setData(new NetNodeInfo(node.getData().getHeight() + dist));
                }
            }
        }
        net.getRoot().setRootPopSize(c.theta);

        UltrametricTree ut = buildUltrametricTree(c.gt);
        GeneTreeBrSpeciesNetDistribution dist =
                new GeneTreeBrSpeciesNetDistribution(net, c.s2a);
        return dist.calculateGTDistribution(ut, null);
    }

    /** Felsenstein log P(S | g) via BEAGLE on the fixed gene tree. */
    static double felsensteinLogL(Case c) throws Exception {
        // Configure the substitution model for this case (process-global).
        Utils._SUBSTITUTION_MODEL = c.model;
        Utils._BASE_FREQS = c.freqs;     // null -> uniform inside PhyloNet
        Utils._TRANS_RATES = c.rates;    // null -> all-ones GTR
        Alignment aln = new Alignment(c.seqs, "locus");
        UltrametricTree ut = new UltrametricTree(c.gt, aln);
        return ut.logDensity();
    }

    /** Gene tree with node heights from branch lengths, no alignment/BEAGLE. */
    static UltrametricTree buildUltrametricTree(String newick) throws Exception {
        NewickReader nr = new NewickReader(new StringReader(newick));
        STITree<Double> tree = new STITree<Double>();
        nr.readTree(tree);
        return new UltrametricTree((Tree) tree);
    }

    // ----- spec-file parsing -------------------------------------------
    static class Case {
        String name;
        String net;
        String gt;
        double theta = 0.02;
        String model = "JC";
        double[] freqs = null;
        double[] rates = null;
        Map<String, List<String>> s2a = null;
        Map<String, String> seqs = new LinkedHashMap<>();
    }

    static double[] parseDoubles(String s) {
        String[] toks = s.trim().split("\\s+");
        double[] out = new double[toks.length];
        for (int i = 0; i < toks.length; i++) out[i] = Double.parseDouble(toks[i]);
        return out;
    }

    static List<Case> parse(String path) throws Exception {
        List<Case> out = new ArrayList<>();
        Case cur = null;
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) continue;
                int sp = line.indexOf(' ');
                String tag = sp < 0 ? line : line.substring(0, sp);
                String rest = sp < 0 ? "" : line.substring(sp + 1).trim();
                switch (tag) {
                    case "CASE":
                        cur = new Case();
                        cur.name = rest;
                        out.add(cur);
                        break;
                    case "NET":
                        cur.net = rest;
                        break;
                    case "GT":
                        cur.gt = rest;
                        break;
                    case "THETA":
                        cur.theta = Double.parseDouble(rest);
                        break;
                    case "MODEL":
                        cur.model = rest;
                        break;
                    case "FREQS":
                        cur.freqs = parseDoubles(rest);
                        break;
                    case "RATES":
                        cur.rates = parseDoubles(rest);
                        break;
                    case "MAP":
                        // "A:a1,a2 B:b1 C:c1"
                        cur.s2a = new HashMap<>();
                        for (String grp : rest.split("\\s+")) {
                            int colon = grp.indexOf(':');
                            String spName = grp.substring(0, colon);
                            String[] alleles = grp.substring(colon + 1).split(",");
                            cur.s2a.put(spName, Arrays.asList(alleles));
                        }
                        break;
                    case "SEQ":
                        int s2 = rest.indexOf(' ');
                        cur.seqs.put(rest.substring(0, s2), rest.substring(s2 + 1).trim());
                        break;
                    case "END":
                        break;
                    default:
                        System.err.println("unknown tag: " + tag);
                }
            }
        }
        return out;
    }
}
