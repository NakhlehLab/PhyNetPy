import growtree
import cProfile

# check to see that all sim trees have 32 leaves, since in abc_tree some trees aren't hitting the leaf goal
# pr = cProfile.Profile()
# pr.enable()
for i in range(0,1):
    t = growtree.gen_tree(.1, .1, .1, 1, 1, 100, 1, 100, goal_leaves = 32, sampling_rate=0.01)
    #print(t)
    print("num leaves", growtree.tree_nleaf(t))
    print("height", growtree.tree_height(t))
    #print(growtree.getNewick(t))
    growtree.outputNewick(t, "NWtree")
    #growtree.print_seq()
print("done")
# pr.disable()
# pr.print_stats()