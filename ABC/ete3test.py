import growtree
import cProfile

# check to see that all sim trees have 32 leaves, since in abc_tree some trees aren't hitting the leaf goal
pr = cProfile.Profile()
pr.enable()
for i in range(0,1):
    t = growtree.gen_tree(4, 0.01, 0.5, 1, 1, 1, 1, 100, goal_leaves = 32)
    #print(t)
    print(growtree.tree_nleaf(t))
    #print(growtree.getNewick(t))
    growtree.outputNewick(t, "NWtree")
    #growtree.print_seq()
print("done")
pr.disable()
pr.print_stats()